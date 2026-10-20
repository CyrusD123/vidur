"""
Train the queue-ranking scheduler with PPO (docs/rl_plan.md §6 step 7):

    python -m vidur.rl.train --config configs/rl/ppo_default.yaml
    python -m vidur.rl.train --config configs/rl/ppo_default.yaml \\
        --set iterations=5 --set ppo.epochs=2 --set output_dir=rl_runs/smoke

Each iteration samples workload seeds, runs group_size rollouts per seed,
updates the policy, and logs metrics (improvement over FCFS on the same seed,
fraction of the FCFS-to-SJF mean gap closed, latency tails, PPO stats) to
<output_dir>/metrics.jsonl (and wandb if enabled). Every eval_every iterations
the deterministic policy runs on held-out eval seeds; the best checkpoint by
eval improvement over FCFS is kept.
"""

import argparse
import dataclasses
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

from vidur.rl.model import ModelConfig, SchedulerModel, config_for
from vidur.rl.observation import ObservationConfig
from vidur.rl.ppo import PPOConfig, RunningStd, ppo_update
from vidur.rl.reward import RewardConfig
from vidur.rl.rollout import (
    RolloutCollector,
    RolloutTask,
    Trajectory,
    WorkloadConfig,
    default_num_workers,
)

# training seeds are drawn below this; eval seeds should be at or above it
TRAIN_SEED_LIMIT = 1_000_000


@dataclass
class TrainConfig:
    workloads: List[Dict[str, Any]]
    # defaults to the training workloads
    eval_workloads: Optional[List[Dict[str, Any]]] = None
    observation: Dict[str, Any] = field(default_factory=dict)
    reward: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    ppo: Dict[str, Any] = field(default_factory=dict)
    num_priority_levels: int = 16

    iterations: int = 200
    seeds_per_iteration: int = 8
    group_size: int = 4
    # None: all cores but two; 0: in-process
    num_workers: Optional[int] = None
    eval_every: int = 10
    eval_seeds: List[int] = field(
        default_factory=lambda: list(range(TRAIN_SEED_LIMIT, TRAIN_SEED_LIMIT + 5))
    )
    seed: int = 0
    output_dir: str = "rl_runs/default"
    wandb: bool = False
    wandb_project: str = "vidur-rl"

    def __post_init__(self):
        assert self.workloads, "at least one workload"
        assert all(
            s >= TRAIN_SEED_LIMIT for s in self.eval_seeds
        ), f"eval seeds must be >= {TRAIN_SEED_LIMIT} (training seeds are below)"

    def workload_configs(self) -> List[WorkloadConfig]:
        return [WorkloadConfig(**w) for w in self.workloads]

    def eval_workload_configs(self) -> List[WorkloadConfig]:
        return [WorkloadConfig(**w) for w in (self.eval_workloads or self.workloads)]

    def observation_config(self) -> ObservationConfig:
        return ObservationConfig(**self.observation)

    def reward_config(self) -> RewardConfig:
        reward = dict(self.reward)
        for key in ("calibration_seeds", "normalize_by_baselines"):
            if reward.get(key) is not None:
                reward[key] = tuple(reward[key])
        return RewardConfig(**reward)

    def model_config(self) -> ModelConfig:
        return config_for(
            self.observation_config(), self.num_priority_levels, **self.model
        )

    def ppo_config(self) -> PPOConfig:
        return PPOConfig(**self.ppo)


def _set_dotted(config: Dict[str, Any], key: str, value: str) -> None:
    *path, last = key.split(".")
    for part in path:
        config = config.setdefault(part, {})
    config[last] = yaml.safe_load(value)


def load_config(path: Optional[str], overrides: Sequence[str] = ()) -> TrainConfig:
    raw: Dict[str, Any] = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for override in overrides:
        key, _, value = override.partition("=")
        _set_dotted(raw, key, value)
    return TrainConfig(**raw)


# --- metrics -------------------------------------------------------------------


def trajectory_metrics(
    trajectories: Sequence[Trajectory], prefix: str
) -> Dict[str, float]:
    """Means over trajectories, overall and per workload."""
    by_workload = defaultdict(list)
    for t in trajectories:
        by_workload[t.task.workload].append(t)

    def summarize(group: Sequence[Trajectory]) -> Dict[str, float]:
        def mean(values):
            values = [v for v in values if np.isfinite(v)]
            return float(np.mean(values)) if values else float("nan")

        return {
            "return": mean([t.total_reward for t in group]),
            "improvement_over_fcfs": mean([t.improvement_over_fcfs for t in group]),
            "mean_gap_closed": mean([t.mean_gap_closed for t in group]),
            "mean_e2e": mean([t.summary["mean_e2e"] for t in group]),
            "p99_e2e": mean([t.summary["p99_e2e"] for t in group]),
            "mean_ttft": mean([t.summary["mean_ttft"] for t in group]),
            "p99_ttft": mean([t.summary["p99_ttft"] for t in group]),
            "p99_tbt": mean([t.summary["p99_tbt"] for t in group]),
            "decision_ms": 1e3 * mean([t.decision_time_s for t in group]),
        }

    metrics = {f"{prefix}/{k}": v for k, v in summarize(trajectories).items()}
    if len(by_workload) > 1:
        for workload, group in by_workload.items():
            for k, v in summarize(group).items():
                metrics[f"{prefix}/{workload}/{k}"] = v
    return metrics


# --- training loop ---------------------------------------------------------------


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        with open(os.path.join(config.output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)

        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.workloads = config.workload_configs()
        self.eval_workloads = config.eval_workload_configs()
        self.model_config = config.model_config()
        self.ppo_config = config.ppo_config()
        self.model = SchedulerModel(self.model_config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.ppo_config.learning_rate
        )
        self.return_std = RunningStd()
        self.iteration = 0
        self.best_eval = -float("inf")

        num_workers = (
            default_num_workers() if config.num_workers is None else config.num_workers
        )
        all_workloads = {w.name: w for w in self.workloads + self.eval_workloads}
        self.collector = RolloutCollector(
            list(all_workloads.values()),
            config.reward_config(),
            config.observation_config(),
            self.model_config,
            os.path.join(config.output_dir, "sim"),
            num_workers,
        )
        self.metrics_file = open(os.path.join(config.output_dir, "metrics.jsonl"), "a")
        self.wandb = None
        if config.wandb:
            import wandb

            self.wandb = wandb.init(
                project=config.wandb_project,
                dir=config.output_dir,
                config=dataclasses.asdict(config),
            )

    def training_tasks(self) -> List[RolloutTask]:
        weights = np.array([w.weight for w in self.workloads], dtype=float)
        tasks = []
        for i in range(self.config.seeds_per_iteration):
            workload = self.workloads[
                self.rng.choice(len(self.workloads), p=weights / weights.sum())
            ]
            seed = int(self.rng.integers(0, TRAIN_SEED_LIMIT))
            for k in range(self.config.group_size):
                agent_seed = int(self.rng.integers(0, 2**31))
                tasks.append(RolloutTask(workload.name, seed, k, agent_seed))
        return tasks

    def eval_tasks(self) -> List[RolloutTask]:
        return [
            RolloutTask(w.name, seed, 0, 0, deterministic=True)
            for w in self.eval_workloads
            for seed in self.config.eval_seeds
        ]

    def log(self, metrics: Dict[str, float]) -> None:
        metrics = {"iteration": self.iteration, **metrics}
        self.metrics_file.write(json.dumps(metrics) + "\n")
        self.metrics_file.flush()
        if self.wandb is not None:
            self.wandb.log(metrics, step=self.iteration)

    def save(self, name: str) -> str:
        path = os.path.join(self.config.output_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": dataclasses.asdict(self.model_config),
                "train_config": dataclasses.asdict(self.config),
                "optimizer": self.optimizer.state_dict(),
                "return_std": self.return_std.state_dict(),
                "iteration": self.iteration,
                "best_eval": self.best_eval,
            },
            path,
        )
        return path

    def evaluate(self) -> Dict[str, float]:
        trajectories = self.collector.collect(
            self.model.state_dict(), self.eval_tasks()
        )
        return trajectory_metrics(trajectories, "eval")

    def train(self) -> None:
        config = self.config
        while self.iteration < config.iterations:
            start = time.time()
            trajectories = self.collector.collect(
                self.model.state_dict(), self.training_tasks()
            )
            rollout_s = time.time() - start
            stats = ppo_update(
                self.model,
                self.optimizer,
                trajectories,
                self.ppo_config,
                self.return_std,
                self.rng,
            )
            self.iteration += 1
            metrics = trajectory_metrics(trajectories, "train")
            metrics.update({f"ppo/{k}": v for k, v in stats.items()})
            metrics["time/rollout_s"] = rollout_s
            metrics["time/update_s"] = time.time() - start - rollout_s

            if (
                self.iteration % config.eval_every == 0
                or self.iteration == config.iterations
            ):
                metrics.update(self.evaluate())
                score = metrics["eval/improvement_over_fcfs"]
                if score > self.best_eval:
                    self.best_eval = score
                    self.save("best.pt")
                self.save("latest.pt")

            self.log(metrics)
            print(
                f"iter {self.iteration:4d} "
                f"return {metrics['train/return']:8.3f} "
                f"vs_fcfs {metrics['train/improvement_over_fcfs']:+.3f} "
                f"sjf_gap {metrics['train/mean_gap_closed']:+.3f} "
                f"p99_e2e {metrics['train/p99_e2e']:6.2f} "
                f"kl {stats['approx_kl']:.4f} ent {stats['entropy']:.3f} "
                + (
                    f"| eval vs_fcfs {metrics['eval/improvement_over_fcfs']:+.3f} "
                    f"sjf_gap {metrics['eval/mean_gap_closed']:+.3f} "
                    if "eval/improvement_over_fcfs" in metrics
                    else ""
                )
                + f"({metrics['time/rollout_s']:.0f}s + {metrics['time/update_s']:.0f}s)",
                flush=True,
            )

    def close(self) -> None:
        self.collector.close()
        self.metrics_file.close()
        if self.wandb is not None:
            self.wandb.finish()


def load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["model_object"] = SchedulerModel(
        ModelConfig(**checkpoint["model_config"])
    )
    checkpoint["model_object"].load_state_dict(checkpoint["model"])
    checkpoint["model_object"].eval()
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--set", action="append", default=[], help="dotted override, e.g. ppo.epochs=2"
    )
    args = parser.parse_args()

    trainer = Trainer(load_config(args.config, args.set))
    print(
        f"{trainer.model.num_parameters()} parameters, output in {trainer.config.output_dir}",
        flush=True,
    )
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
