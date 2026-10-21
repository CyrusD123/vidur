"""
CDFs of TTFT, TBT, request e2e time and batch size (the plots in the README)
for baselines and a trained policy, pooled over seeds:

    python -m vidur.rl.plot_cdfs --checkpoint rl_runs/run2/best.pt \\
        --output-dir rl_runs/cdf_plots

--policies picks the lines (baseline names, and policy_rank for the
checkpoint), e.g. an imitation student against its teacher:

    python -m vidur.rl.plot_cdfs \\
        --checkpoint rl_runs/imitation_sjf_age400/latest.pt \\
        --output-dir rl_runs/cdf_plots_imitation_age400 \\
        --policies fcfs sjf_total_tokens_age400 \\
            sjf_prefill_tokens_age400_16lvl policy_rank \\
        --policy-label "Imitation student (rank)"

Raw samples are cached per (workload, policy) in <output-dir>/data, so
re-plotting skips the simulation. Writes
    all_policies/<workload>/<metric>.png
and, when both are among the policies,
    fcfs_vs_sjf/<workload>/<metric>.png
plus a 1x4 overview.png per workload in each.
"""

import argparse
import multiprocessing as mp
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from vidur.rl.agents import baseline_agent
from vidur.rl.reward import RewardConfig, episode_metric_samples
from vidur.rl.rollout import WorkloadConfig
from vidur.rl.torch_agent import TorchAgent
from vidur.rl.train import TrainConfig, load_checkpoint
from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
    RLVLLMReplicaScheduler,
)

WORKLOADS = [
    {"name": "qps10_blocks4000", "qps": 10, "num_blocks": 4000, "num_requests": 300},
    {"name": "qps20", "qps": 20, "num_requests": 300},
    {"name": "sharegpt_q24_b8500", "trace": "sharegpt", "qps": 24,
     "num_blocks": 8500, "num_requests": 1000},
    {"name": "azure_replay_q5_b4000", "trace": "azure_conv", "arrivals": "trace",
     "qps": 5, "num_blocks": 4000, "num_requests": 300},
    {"name": "azure_replay_b8500", "trace": "azure_conv", "arrivals": "trace",
     "qps": None, "num_blocks": 8500, "num_requests": 600},
]
POLICIES = ("fcfs", "sjf_total_tokens", "sjf_total_tokens_age400", "policy_rank")
LABELS = {
    "fcfs": "FCFS",
    "sjf_total_tokens": "SJF (total tokens)",
    "sjf_total_tokens_age400": "SJF + aging 400 tok/s",
    "sjf_prefill_tokens_age400_16lvl": "Prompt-only SJF + aging 400 (16 lvl)",
    "policy_rank": "RL policy (rank)",
}
# validated categorical slots 1-5, plus line style so identity isn't color alone
STYLES = {
    "fcfs": ("#2a78d6", "-"),
    "sjf_total_tokens": ("#eb6834", "--"),
    "sjf_total_tokens_age400": ("#1baf7a", "-."),
    "sjf_prefill_tokens_age400_16lvl": ("#8a5cd6", "--"),
    "policy_rank": ("#eda100", ":"),
}
METRICS = {
    "ttft": "TTFT (s)",
    "tbt": "Time between tokens (s)",
    "e2e": "Request E2E time (s)",
    "batch_size": "Batch size (requests)",
}


def _record_batch_sizes(sizes: List[int]) -> None:
    on_batch_end = RLVLLMReplicaScheduler.on_batch_end

    def wrapped(self, batch):
        sizes.append(batch.size)
        on_batch_end(self, batch)

    RLVLLMReplicaScheduler.on_batch_end = wrapped


def run_workload(args) -> str:
    workload, checkpoint_path, seeds, data_dir, policies = args
    checkpoint = load_checkpoint(checkpoint_path)
    train_config = TrainConfig(**checkpoint["train_config"])
    env = WorkloadConfig(**workload).make_env(
        RewardConfig(),
        train_config.observation_config(),
        train_config.num_priority_levels,
        "simulator_output",
    )
    batch_sizes: List[int] = []
    _record_batch_sizes(batch_sizes)

    for policy in policies:
        path = os.path.join(data_dir, f"{workload['name']}__{policy}.npz")
        if os.path.exists(path):
            continue
        samples: Dict[str, List[np.ndarray]] = {m: [] for m in METRICS}
        for seed in seeds:
            if policy == "policy_rank":
                agent = TorchAgent(checkpoint["model_object"], seed, "rank")
            else:
                agent = baseline_agent(policy, seed=seed)
            batch_sizes.clear()
            # plain reward config: no FCFS reference runs needed
            episode = env.run_episode(agent, seed, reward_config=RewardConfig())
            for name, values in episode_metric_samples(episode).items():
                samples[name].append(values)
            samples["batch_size"].append(np.array(batch_sizes))
            print(
                f"{workload['name']} {policy} seed {seed}: "
                f"mean e2e {episode.summary['mean_e2e']:.2f}s",
                flush=True,
            )
        np.savez(path, **{m: np.concatenate(v) for m, v in samples.items()})
    return workload["name"]


def load_samples(
    data_dir: str, workload: str, policies: Sequence[str]
) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        policy: dict(np.load(os.path.join(data_dir, f"{workload}__{policy}.npz")))
        for policy in policies
    }


def plot_cdf(ax, samples, policies: Sequence[str], metric: str) -> None:
    for policy in policies:
        values = np.sort(samples[policy][metric])
        color, linestyle = STYLES[policy]
        ax.plot(
            values,
            np.arange(1, len(values) + 1) / len(values),
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{LABELS[policy]} (p50 {np.median(values):.3g}, "
            f"p99 {np.quantile(values, 0.99):.3g})",
            drawstyle="steps-post" if metric == "batch_size" else "default",
        )
    if metric == "tbt":
        ax.set_xscale("log")
    ax.set_xlabel(METRICS[metric])
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(True, color="#e5e4df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def comparisons(policies: Sequence[str]) -> Dict[str, Sequence[str]]:
    result = {"all_policies": tuple(policies)}
    if "fcfs" in policies and "sjf_total_tokens" in policies:
        result["fcfs_vs_sjf"] = ("fcfs", "sjf_total_tokens")
    return result


def plot_all(
    output_dir: str, data_dir: str, seeds: Sequence[int], policies: Sequence[str]
) -> None:
    for comparison, compared in comparisons(policies).items():
        for workload in WORKLOADS:
            name = workload["name"]
            samples = load_samples(data_dir, name, compared)
            directory = os.path.join(output_dir, comparison, name)
            os.makedirs(directory, exist_ok=True)
            suffix = f"{name}, {len(seeds)} seeds pooled"
            for metric in METRICS:
                fig, ax = plt.subplots(figsize=(6.4, 4.2))
                plot_cdf(ax, samples, compared, metric)
                ax.set_title(f"{METRICS[metric]} — {suffix}", fontsize=10)
                fig.tight_layout()
                fig.savefig(os.path.join(directory, f"{metric}.png"), dpi=150)
                plt.close(fig)
            fig, axes = plt.subplots(1, 4, figsize=(22, 4.4))
            for ax, metric in zip(axes, METRICS):
                plot_cdf(ax, samples, compared, metric)
            fig.suptitle(suffix)
            fig.tight_layout()
            fig.savefig(os.path.join(directory, "overview.png"), dpi=120)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", default="rl_runs/run2/best.pt")
    parser.add_argument("--output-dir", default="rl_runs/cdf_plots")
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=list(range(1000000, 1000005))
    )
    parser.add_argument("--workers", type=int, default=len(WORKLOADS))
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--policy-label", default=LABELS["policy_rank"])
    args = parser.parse_args()
    unknown = sorted(set(args.policies) - set(STYLES))
    assert not unknown, f"add {unknown} to LABELS and STYLES"
    LABELS["policy_rank"] = args.policy_label

    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "workloads.yaml"), "w") as f:
        yaml.safe_dump(
            {"checkpoint": args.checkpoint, "seeds": args.seeds,
             "policies": args.policies, "workloads": WORKLOADS},
            f, sort_keys=False,
        )
    tasks = [
        (w, args.checkpoint, args.seeds, data_dir, args.policies) for w in WORKLOADS
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name in pool.imap_unordered(run_workload, tasks):
            print(f"done: {name}", flush=True)
    plot_all(args.output_dir, data_dir, args.seeds, args.policies)


if __name__ == "__main__":
    main()
