"""
Compare baseline schedulers on the same workloads, and check reward weights
against them:

    python -m vidur.rl.baselines --qps 10 --num-blocks 4000 --seeds 0 1 2 3 4
    python -m vidur.rl.baselines --qps 10 --num-blocks 4000 --weight-check

FCFS is stock vLLM. The oracle SJF baselines read true output lengths, so they
are reference points, not deployable policies. The bucketed (_<L>lvl) variants
are rankings the RL agent itself could express with its priority levels, and
the aging (_age<rate>) variants trade some mean latency for tails.

The weight check runs each baseline once per seed, then rescores the episodes
offline (episode_cost_components) under a grid of reward configs: a
mean-latency term plus a weighted tail hinge with per-request thresholds from
FCFS. A good weight is one where FCFS is not the best baseline (the reward would
push the policy to copy FCFS) and plain SJF is not either (the tail term would
not matter).
"""

import argparse
import itertools
from typing import List, Optional, Sequence

import pandas as pd

from vidur.rl.env import VidurSchedulingEnv, make_simulation_config
from vidur.rl.reward import (
    RewardConfig,
    episode_cost,
    episode_cost_components,
    window_thresholds,
)

METRICS = (
    "mean_e2e",
    "p99_e2e",
    "mean_ttft",
    "p99_ttft",
    "mean_tbt",
    "p99_tbt",
    "output_tokens_per_s",
    "num_restarts",
)


def default_baselines(num_levels: int) -> List[str]:
    baselines = ["fcfs", "random"]
    for key in ("total_tokens", "decode_tokens", "iterations"):
        baselines += [f"sjf_{key}", f"sjf_{key}_{num_levels}lvl"]
    return baselines


# SJF with aging: rates in job-size units per second of waiting, picked so a
# request's priority improves by a typical job size within ~5-20 s of waiting
WEIGHT_CHECK_BASELINES = (
    "fcfs",
    "random",
    "sjf_total_tokens",
    "sjf_decode_tokens",
    "sjf_decode_tokens_16lvl",
    "sjf_total_tokens_age100",
    "sjf_total_tokens_age400",
    "sjf_decode_tokens_age20",
    "sjf_decode_tokens_age80",
)


def compare_baselines(
    env: VidurSchedulingEnv,
    baselines: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """One row per (baseline, seed): the episode summary and cost under the
    env's reward config."""
    rows = []
    for name in baselines:
        for seed in seeds:
            episode = env.reference_episode(name, seed)
            rows.append(
                {
                    "baseline": name,
                    "seed": seed,
                    "cost": episode_cost(
                        episode, env.reward_config, episode.request_thresholds
                    ),
                    **episode.summary,
                }
            )
    return pd.DataFrame(rows)


def weight_check_costs(
    env: VidurSchedulingEnv,
    baselines: Sequence[str],
    seeds: Sequence[int],
    mean_terms: Sequence[str] = ("e2e_normalized", "e2e"),
    tail_terms: Sequence[str] = ("e2e_hinge", "ttft_hinge"),
    windows: Sequence[Optional[int]] = (50, None),
    quantile: float = 0.95,
) -> pd.DataFrame:
    """
    One row per (mean term, tail term, window, baseline, seed) with the
    baseline's unweighted mean-term and tail-term costs. The tail term's
    threshold is the quantile of FCFS's metric over a window of `window`
    requests around each request's arrival (None = the whole episode, i.e. one
    threshold per workload). Each baseline runs once per seed; everything else
    is rescored offline.
    """
    episodes = {
        (name, seed): env.reference_episode(name, seed)
        for name in set(baselines) | {"fcfs"}
        for seed in seeds
    }
    rows = []
    for mean_term, tail_term, window in itertools.product(
        mean_terms, tail_terms, windows
    ):
        config = RewardConfig(
            weights={mean_term: 1.0, tail_term: 1.0},
            threshold_mode="reference_window",
            calibration_quantile=quantile,
            reference_window_requests=window or 10**9,
        )
        for seed in seeds:
            thresholds = window_thresholds(episodes[("fcfs", seed)], config)
            for name in baselines:
                components = episode_cost_components(
                    episodes[(name, seed)], config, thresholds
                )
                rows.append(
                    {
                        "mean_term": mean_term,
                        "tail_term": tail_term,
                        "window": window or "episode",
                        "baseline": name,
                        "seed": seed,
                        "mean_cost": components[mean_term],
                        "tail_cost": components[tail_term],
                    }
                )
    return pd.DataFrame(rows)


def weight_check(
    costs: pd.DataFrame,
    tail_weights: Sequence[float] = (0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100),
) -> pd.DataFrame:
    """
    One row per (reward config, baseline): improvement over FCFS under
    reward = -(mean_term + tail_weight * tail_term), i.e.
    1 - cost / cost_fcfs averaged over seeds (0 = FCFS, higher is better).
    """
    keys = ["mean_term", "tail_term", "window"]
    fcfs = costs[costs["baseline"] == "fcfs"].set_index(keys + ["seed"])
    rows = []
    for weight in tail_weights:
        table = costs.join(
            fcfs[["mean_cost", "tail_cost"]], on=keys + ["seed"], rsuffix="_fcfs"
        )
        cost = table["mean_cost"] + weight * table["tail_cost"]
        cost_fcfs = table["mean_cost_fcfs"] + weight * table["tail_cost_fcfs"]
        table = table.assign(tail_weight=weight, improvement=1 - cost / cost_fcfs)
        rows.append(table)
    table = pd.concat(rows)
    return (
        table.groupby(keys + ["tail_weight", "baseline"], sort=False)["improvement"]
        .agg(["mean", "std"])
        .reset_index()
    )


def summarize_costs(costs: pd.DataFrame) -> pd.DataFrame:
    """Each baseline's mean-term and tail-term cost relative to FCFS (1 = FCFS)."""
    keys = ["mean_term", "tail_term", "window", "seed"]
    fcfs = costs[costs["baseline"] == "fcfs"].set_index(keys)
    table = costs.join(fcfs[["mean_cost", "tail_cost"]], on=keys, rsuffix="_fcfs")
    table["mean_vs_fcfs"] = table["mean_cost"] / table["mean_cost_fcfs"]
    table["tail_vs_fcfs"] = table["tail_cost"] / table["tail_cost_fcfs"]
    return table.pivot_table(
        index=["mean_term", "tail_term", "window", "baseline"],
        values=["mean_vs_fcfs", "tail_vs_fcfs"],
        sort=False,
    )


def summarize_weight_check(table: pd.DataFrame) -> pd.DataFrame:
    """Per reward config: each baseline's improvement, and the best one."""
    keys = ["mean_term", "tail_term", "window", "tail_weight"]
    scores = table.pivot_table(
        index=keys, columns="baseline", values="mean", sort=False
    )
    scores["best"] = scores.idxmax(axis=1)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--num-requests", type=int, default=300)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--num-priority-levels", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--weight-check", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="CSV of all rows")
    args = parser.parse_args()

    env = VidurSchedulingEnv(
        make_simulation_config(
            qps=args.qps,
            num_requests=args.num_requests,
            num_blocks=args.num_blocks,
            num_priority_levels=args.num_priority_levels,
        )
    )

    display = pd.option_context(
        "display.width", 250, "display.max_columns", 30, "display.max_rows", 500
    )
    if args.weight_check:
        costs = weight_check_costs(env, WEIGHT_CHECK_BASELINES, args.seeds)
        table = weight_check(costs)
        if args.output:
            costs.to_csv(args.output, index=False)
        with display, pd.option_context("display.float_format", "{:.2f}".format):
            print(summarize_costs(costs))
            print()
            print(summarize_weight_check(table))
        return

    table = compare_baselines(
        env, default_baselines(args.num_priority_levels), args.seeds
    )
    if args.output:
        table.to_csv(args.output, index=False)
    columns = ["cost", *METRICS]
    means = table.groupby("baseline", sort=False)[columns].mean()
    with display, pd.option_context("display.float_format", "{:.3f}".format):
        print(means)


if __name__ == "__main__":
    main()
