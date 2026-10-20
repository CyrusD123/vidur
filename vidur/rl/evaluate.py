"""
Evaluate a trained policy against baselines on held-out seeds
(docs/rl_plan.md §6 step 8):

    python -m vidur.rl.evaluate --checkpoint rl_runs/ppo_default/best.pt
    python -m vidur.rl.evaluate --checkpoint ... --seeds 2000000 2000001 \\
        --workload '{name: qps15, qps: 15, num_blocks: 6000}'

For each workload and seed it runs the policy with each decoding in
--decodings (policy_rank, policy_argmax, policy_sample; see TorchAgent) and
the baselines, and reports per workload: improvement over FCFS
under the training reward, the fraction of the FCFS-to-SJF mean gap closed,
mean/p99 latencies, and the policy's inference time per decision.
"""

import argparse
from typing import Dict, List, Sequence

import pandas as pd
import yaml

from vidur.rl.reward import episode_cost
from vidur.rl.rollout import RolloutRunner, RolloutTask, WorkloadConfig
from vidur.rl.torch_agent import DECODINGS
from vidur.rl.train import TrainConfig, load_checkpoint

DEFAULT_BASELINES = (
    "fcfs",
    "sjf_total_tokens",
    "sjf_decode_tokens",
    "sjf_total_tokens_age400",
)
COLUMNS = (
    "improvement_over_fcfs",
    "mean_gap_closed",
    "mean_e2e",
    "p99_e2e",
    "mean_ttft",
    "p99_ttft",
    "p99_tbt",
    "decision_ms",
)


def evaluate(
    checkpoint_path: str,
    workloads: Sequence[WorkloadConfig] = (),
    seeds: Sequence[int] = (),
    baselines: Sequence[str] = DEFAULT_BASELINES,
    decodings: Sequence[str] = ("rank", "argmax"),
) -> pd.DataFrame:
    """One row per (workload, seed, policy)."""
    checkpoint = load_checkpoint(checkpoint_path)
    train_config = TrainConfig(**checkpoint["train_config"])
    workloads = list(workloads) or train_config.eval_workload_configs()
    seeds = list(seeds) or train_config.eval_seeds
    model = checkpoint["model_object"]

    runner = RolloutRunner(
        workloads,
        train_config.reward_config(),
        train_config.observation_config(),
        model.config,
        "simulator_output",
    )
    runner.model.load_state_dict(model.state_dict())

    assert set(decodings) <= set(DECODINGS), f"decodings must be in {DECODINGS}"
    rows: List[Dict] = []
    for workload in workloads:
        env = runner.env(workload.name)
        for seed in seeds:
            fcfs = env.reference_episode("fcfs", seed)
            for decoding in decodings:
                trajectory = runner.run(
                    RolloutTask(workload.name, seed, 0, seed, decoding)
                )
                rows.append(
                    {
                        "workload": workload.name,
                        "seed": seed,
                        "policy": f"policy_{decoding}",
                        "improvement_over_fcfs": trajectory.improvement_over_fcfs,
                        "mean_gap_closed": trajectory.mean_gap_closed,
                        "decision_ms": 1e3 * trajectory.decision_time_s,
                        **trajectory.summary,
                    }
                )
            for name in baselines:
                episode = env.reference_episode(name, seed)
                thresholds = env.reward_context(
                    env.reward_config, seed
                ).request_thresholds
                cost = episode_cost(episode, env.reward_config, thresholds)
                cost_fcfs = episode_cost(fcfs, env.reward_config, thresholds)
                rows.append(
                    {
                        "workload": workload.name,
                        "seed": seed,
                        "policy": name,
                        "improvement_over_fcfs": 1 - cost / cost_fcfs,
                        "mean_gap_closed": env.mean_gap_closed(episode, seed),
                        "decision_ms": float("nan"),
                        **episode.summary,
                    }
                )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby(["workload", "policy"], sort=False)[list(COLUMNS)].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument(
        "--workload", action="append", default=[], help="YAML/JSON WorkloadConfig"
    )
    parser.add_argument("--baselines", nargs="*", default=list(DEFAULT_BASELINES))
    parser.add_argument(
        "--decodings", nargs="*", default=["rank", "argmax"], choices=DECODINGS
    )
    parser.add_argument("--output", default=None, help="CSV of all rows")
    args = parser.parse_args()

    workloads = [WorkloadConfig(**yaml.safe_load(w)) for w in args.workload]
    table = evaluate(
        args.checkpoint, workloads, args.seeds, args.baselines, args.decodings
    )
    if args.output:
        table.to_csv(args.output, index=False)
    with pd.option_context(
        "display.width",
        200,
        "display.max_columns",
        20,
        "display.float_format",
        "{:.3f}".format,
    ):
        print(summarize(table))


if __name__ == "__main__":
    main()
