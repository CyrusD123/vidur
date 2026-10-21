"""
Compare baseline schedulers on the same workloads, and check reward weights
against them:

    python -m vidur.rl.baselines --qps 10 --num-blocks 4000 --seeds 0 1 2 3 4
    python -m vidur.rl.baselines --qps 10 --num-blocks 4000 --weight-check
    python -m vidur.rl.baselines --check-assumptions \
        --workload '{name: sharegpt_qps20, trace: sharegpt, qps: 20}' \
        --workload '{name: azure_conv_replay, trace: azure_conv, arrivals: trace, qps: null}'

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

The assumption check tests what the reward design takes for granted, per
workload and per seed: FCFS has the lowest tail latency, oracle SJF the lowest
mean, and SJF with aging lands between the two on both.
"""

import argparse
import itertools
from typing import List, Optional, Sequence

import pandas as pd
import yaml

from vidur.rl.env import VidurSchedulingEnv, make_simulation_config
from vidur.rl.observation import ObservationConfig
from vidur.rl.reward import (
    RewardConfig,
    episode_cost,
    episode_cost_components,
    window_thresholds,
)
from vidur.rl.rollout import WorkloadConfig

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


# plain SJF -> its aging variants, for the assumption check
ASSUMPTION_BASELINES = {
    "sjf_total_tokens": ("sjf_total_tokens_age100", "sjf_total_tokens_age400"),
    "sjf_decode_tokens": ("sjf_decode_tokens_age20", "sjf_decode_tokens_age80"),
}
# lower is better for all of these
MEAN_METRICS = ("mean_e2e", "mean_e2e_normalized")
TAIL_METRICS = ("p99_e2e", "max_e2e", "p99_ttft", "p99_e2e_normalized")


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


def compare_workloads(
    workloads: Sequence[WorkloadConfig],
    baselines: Sequence[str],
    seeds: Sequence[int],
    num_priority_levels: int = 16,
) -> pd.DataFrame:
    """compare_baselines on each workload, with a `workload` column."""
    tables = []
    for workload in workloads:
        env = workload.make_env(
            RewardConfig(), ObservationConfig(), num_priority_levels, "simulator_output"
        )
        table = compare_baselines(env, baselines, seeds)
        tables.append(table.assign(workload=workload.name))
    return pd.concat(tables, ignore_index=True)


def check_assumptions(table: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """
    For a compare_workloads table with ASSUMPTION_BASELINES, one row per
    (workload, claim) with the fraction of seeds where it holds, comparing
    within a relative tolerance tol (so near-ties count as holding):
      fcfs_best_<tail>      FCFS has the lowest tail metric of all baselines
      sjf_best_<mean>       some plain SJF has the lowest mean of all baselines
      <aged>_between_<m>    the aging variant is between its plain SJF and FCFS
                            on metric m (no worse than the worse of the two,
                            no better than the better)
    plus spread_<mean>: the fraction of seeds where the baselines' means differ
    by more than tol at all. Where it is low, ordering barely matters and the
    other claims hold trivially.
    """
    rows = []
    for (workload, seed), group in table.groupby(["workload", "seed"], sort=False):
        by_name = group.set_index("baseline")
        claims = {}
        for metric in MEAN_METRICS:
            values = by_name[metric]
            claims[f"spread_{metric}"] = values.max() > values.min() * (1 + tol)
        for metric in TAIL_METRICS:
            claims[f"fcfs_best_{metric}"] = (
                by_name.loc["fcfs", metric] <= by_name[metric].min() * (1 + tol)
            )
        for metric in MEAN_METRICS:
            best_sjf = by_name.loc[list(ASSUMPTION_BASELINES), metric].min()
            claims[f"sjf_best_{metric}"] = best_sjf <= by_name[metric].min() * (1 + tol)
        for sjf, aged_names in ASSUMPTION_BASELINES.items():
            for aged in aged_names:
                for metric in MEAN_METRICS + TAIL_METRICS:
                    ends = by_name.loc[["fcfs", sjf], metric]
                    value = by_name.loc[aged, metric]
                    claims[f"{aged}_between_{metric}"] = (
                        ends.min() * (1 - tol) <= value <= ends.max() * (1 + tol)
                    )
        rows += [
            {"workload": workload, "seed": seed, "claim": claim, "holds": bool(v)}
            for claim, v in claims.items()
        ]
    return (
        pd.DataFrame(rows)
        .groupby(["workload", "claim"], sort=False)["holds"]
        .mean()
        .unstack("workload")
    )


def relative_to_fcfs(table: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Seed-mean of each metric, divided by FCFS's on the same seed."""
    metrics = list(metrics)
    fcfs = table.loc[table["baseline"] == "fcfs", ["workload", "seed", *metrics]]
    ratios = table.merge(fcfs, on=["workload", "seed"], suffixes=("", "_fcfs"))
    for metric in metrics:
        ratios[metric] /= ratios[f"{metric}_fcfs"]
    return ratios.groupby(["workload", "baseline"], sort=False)[metrics].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--num-requests", type=int, default=300)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--num-priority-levels", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--trace", type=str, default="azure_conv")
    parser.add_argument("--arrivals", choices=["poisson", "trace"], default="poisson")
    parser.add_argument(
        "--workload",
        action="append",
        default=[],
        help="YAML/JSON WorkloadConfig (repeatable); overrides the flags above",
    )
    parser.add_argument("--weight-check", action="store_true")
    parser.add_argument("--check-assumptions", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="CSV of all rows")
    args = parser.parse_args()

    display = pd.option_context(
        "display.width", 250, "display.max_columns", 30, "display.max_rows", 500
    )
    workloads = [WorkloadConfig(**yaml.safe_load(w)) for w in args.workload]
    if args.check_assumptions:
        workloads = workloads or [
            WorkloadConfig(
                name=f"{args.trace}_{args.arrivals}_qps{args.qps}",
                qps=args.qps,
                num_requests=args.num_requests,
                num_blocks=args.num_blocks,
                trace=args.trace,
                arrivals=args.arrivals,
            )
        ]
        baselines = ["fcfs"] + [
            name
            for sjf, aged in ASSUMPTION_BASELINES.items()
            for name in (sjf, *aged)
        ]
        table = compare_workloads(
            workloads, baselines, args.seeds, args.num_priority_levels
        )
        if args.output:
            table.to_csv(args.output, index=False)
        with display, pd.option_context("display.float_format", "{:.3f}".format):
            print("absolute (seed means)")
            print(
                table.groupby(["workload", "baseline"], sort=False)[
                    list(MEAN_METRICS + TAIL_METRICS)
                ].mean()
            )
            print("\nrelative to FCFS (< 1 = better)")
            print(relative_to_fcfs(table, MEAN_METRICS + TAIL_METRICS))
            print("\nfraction of seeds where each claim holds")
            print(check_assumptions(table))
        return

    if workloads:
        assert len(workloads) == 1, "one --workload outside --check-assumptions"
        env = workloads[0].make_env(
            RewardConfig(),
            ObservationConfig(),
            args.num_priority_levels,
            "simulator_output",
        )
    else:
        env = VidurSchedulingEnv(
            make_simulation_config(
                qps=args.qps,
                num_requests=args.num_requests,
                num_blocks=args.num_blocks,
                num_priority_levels=args.num_priority_levels,
                trace_file=args.trace,
                arrivals=args.arrivals,
            )
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
