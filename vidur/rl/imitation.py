"""
Imitation learning from an oracle SJF teacher with DAgger (docs/rl_plan.md §8),
as an initialization for PPO fine-tuning:

    python -m vidur.rl.imitation --config configs/rl/imitation_sjf_age800.yaml
    python -m vidur.rl.train --config configs/rl/imitation_sjf_age800.yaml \\
        --set init_checkpoint=rl_runs/imitation_sjf_age800/best.pt \\
        --set output_dir=rl_runs/imitation_sjf_age800_ppo

The teacher (imitation.teacher, an SJF baseline name such as
sjf_total_tokens_age800) ranks by the true output length; the student sees the
usual observation, which never contains it (vidur/rl/teacher.py).

DAgger: the first rollouts are the teacher's own (initial_seeds). Each
iteration then trains the student on every labeled decision collected so far,
evaluates it, and collects more rollouts with the mixture policy (teacher with
probability beta, else the student with collect_decoding), beta decaying from
beta_start by beta_decay per iteration. The student's own states are labeled by
the teacher, so it learns to recover from its mistakes, which training on the
teacher's states alone (behavior cloning) does not teach.

Only decisions with an admission boundary are trained on: when everything
waiting was admitted, the ranking changed nothing.

Checkpoints have the same format as vidur.rl.train's, so vidur.rl.evaluate and
init_checkpoint both read them. Metrics go to <output_dir>/metrics.jsonl:
imitation losses, the student's pair accuracy on its training data, and per
eval decoding the usual eval metrics plus agreement with the teacher
(pair_agreement, admitted_overlap, head_agreement). teacher/... is the
teacher's own score on the eval seeds, the target to match.
"""

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from vidur.rl.model import collate, interval_targets, modeled_indices
from vidur.rl.ppo import make_optimizer, minibatches
from vidur.rl.rollout import RolloutTask, Trajectory
from vidur.rl.teacher import has_labels, imitation_losses, imitation_targets
from vidur.rl.torch_agent import DECODINGS
from vidur.rl.train import TRAIN_SEED_LIMIT, Trainer, load_config, trajectory_metrics


@dataclass
class ImitationConfig:
    # SJF baseline name (vidur.rl.agents.baseline_agent); an _<L>lvl suffix is
    # ignored, the labels are cut into the env's priority levels
    teacher: str = "sjf_total_tokens_age800"
    iterations: int = 10
    # pure-teacher rollouts before the first update
    initial_seeds: int = 64
    # rollouts per iteration with the mixture policy
    seeds_per_iteration: int = 32
    beta_start: float = 0.5
    beta_decay: float = 0.5
    # how the student acts in mixture rollouts; "rank" visits the states the
    # deployed (rank-decoded) policy visits
    collect_decoding: str = "rank"
    epochs_per_iteration: int = 2
    # DAgger iterations between evals (the top-level eval_every is PPO's)
    eval_every: int = 1
    learning_rate: float = 3e-4
    minibatch_size: int = 128
    max_grad_norm: float = 1.0
    # loss = level_coef * level NLL + rank_coef * pairwise rank loss
    level_coef: float = 1.0
    rank_coef: float = 1.0
    # keep only the most recent decisions (None: all)
    max_dataset_size: Optional[int] = None
    pad_to_multiple: int = 32
    bucket_jitter: float = 16.0

    def __post_init__(self):
        assert self.collect_decoding in DECODINGS
        assert 0.0 <= self.beta_start <= 1.0 and 0.0 <= self.beta_decay <= 1.0

    def beta(self, iteration: int) -> float:
        """Teacher probability for the rollouts collected after `iteration`
        updates (1 before the first)."""
        if iteration == 0:
            return 1.0
        return self.beta_start * self.beta_decay ** (iteration - 1)


class ImitationTrainer(Trainer):
    """DAgger on top of Trainer's rollout workers, eval, logging and checkpoints."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.imitation = ImitationConfig(**config.imitation)
        self.optimizer = make_optimizer(self.model, self.imitation.learning_rate)
        # labeled decisions with an admission boundary
        self.dataset: List = []

    def collection_tasks(self, num_seeds: int, beta: float) -> List[RolloutTask]:
        weights = np.array([w.weight for w in self.workloads], dtype=float)
        tasks = []
        for _ in range(num_seeds):
            workload = self.workloads[
                self.rng.choice(len(self.workloads), p=weights / weights.sum())
            ]
            tasks.append(
                RolloutTask(
                    workload.name,
                    int(self.rng.integers(0, TRAIN_SEED_LIMIT)),
                    0,
                    int(self.rng.integers(0, 2**31)),
                    self.imitation.collect_decoding,
                    teacher=self.imitation.teacher,
                    teacher_mix=beta,
                )
            )
        return tasks

    def collect(self, num_seeds: int, beta: float) -> List[Trajectory]:
        trajectories = self.collector.collect(
            self.model.state_dict(), self.collection_tasks(num_seeds, beta)
        )
        for trajectory in trajectories:
            self.dataset.extend(
                t
                for t in trajectory.transitions
                if has_labels(t) and t.boundary_index is not None
            )
        if self.imitation.max_dataset_size is not None:
            self.dataset = self.dataset[-self.imitation.max_dataset_size :]
        return trajectories

    def teacher_metrics(self) -> Dict[str, float]:
        """The teacher's own eval score (it acts with its bucketed ranking)."""
        tasks = [
            RolloutTask(
                w.name, seed, 0, seed, teacher=self.imitation.teacher, teacher_mix=1.0
            )
            for w in self.eval_workloads
            for seed in self.config.eval_seeds
        ]
        trajectories = self.collector.collect(self.model.state_dict(), tasks)
        # agreement with itself is 1, and the student never acted
        skip = ("pair_agreement", "admitted_overlap", "head_agreement", "decision_ms")
        return {
            k: v
            for k, v in trajectory_metrics(trajectories, "teacher").items()
            if k.rsplit("/", 1)[-1] not in skip
        }

    def fit(self) -> Dict[str, float]:
        """epochs_per_iteration passes over the dataset."""
        config, model = self.imitation, self.model
        modeled = [
            modeled_indices(t.obs, model.config.max_queue_len) for t in self.dataset
        ]
        lengths = np.array(
            [
                len(m) + min(t.obs.num_running, model.config.max_running_len)
                for t, m in zip(self.dataset, modeled)
            ],
            dtype=float,
        )
        model.train()
        stats = defaultdict(list)
        for _ in range(config.epochs_per_iteration):
            for indices in minibatches(
                lengths, config.minibatch_size, self.rng, True, config.bucket_jitter
            ):
                transitions = [self.dataset[j] for j in indices]
                chunk_modeled = [modeled[j] for j in indices]
                batch = collate(
                    [t.obs for t in transitions],
                    model.config,
                    chunk_modeled,
                    config.pad_to_multiple,
                ).to(self.device)
                out = model(batch, with_value=False)
                width = batch.waiting.shape[1]
                targets = {
                    k: v.to(self.device)
                    for k, v in imitation_targets(
                        transitions, chunk_modeled, width
                    ).items()
                }
                mask = (
                    interval_targets(transitions, chunk_modeled, width)["loss_mask"].to(
                        self.device
                    )
                    & batch.waiting_mask
                )
                losses = imitation_losses(out.dist, targets, mask)
                level_loss = losses["level_loss"].mean()
                rank_loss = losses["rank_loss"].mean()
                loss = config.level_coef * level_loss + config.rank_coef * rank_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.policy_parameters(), config.max_grad_norm
                )
                self.optimizer.step()

                entropy = (out.dist.entropy() * mask).sum() / mask.sum().clamp(min=1)
                stats["step"].append(
                    torch.stack(
                        [
                            level_loss.detach(),
                            rank_loss.detach(),
                            losses["pair_accuracy"].mean(),
                            entropy.detach(),
                            grad_norm,
                        ]
                    )
                )
        model.eval()
        values = torch.stack(stats["step"]).cpu().numpy()
        # the last epoch's means: closest to the current weights
        last = values[-len(values) // config.epochs_per_iteration :]
        names = ("level_loss", "rank_loss", "pair_accuracy", "entropy", "grad_norm")
        summary = {
            f"imitation/{n}": float(last[:, i].mean()) for i, n in enumerate(names)
        }
        summary["imitation/dataset_size"] = len(self.dataset)
        return summary

    def train(self) -> None:
        config, imitation = self.config, self.imitation
        teacher = self.teacher_metrics()
        self.log(teacher)
        print(
            f"teacher {imitation.teacher}: eval vs_fcfs "
            f"{teacher['teacher/improvement_over_fcfs']:+.3f} sjf_gap "
            f"{teacher['teacher/mean_gap_closed']:+.3f}",
            flush=True,
        )

        start = time.time()
        collected = self.collect(imitation.initial_seeds, imitation.beta(0))
        collect_s = time.time() - start
        while self.iteration < imitation.iterations:
            collect_metrics = trajectory_metrics(collected, "collect")
            collect_metrics["collect/beta"] = imitation.beta(self.iteration)

            start = time.time()
            metrics = self.fit()
            self.iteration += 1
            metrics.update(collect_metrics)
            metrics["time/collect_s"] = collect_s
            metrics["time/fit_s"] = time.time() - start

            if (
                self.iteration % imitation.eval_every == 0
                or self.iteration == imitation.iterations
            ):
                metrics.update(self.evaluate())
                score = metrics[f"eval/{config.select_decoding}/improvement_over_fcfs"]
                if score > self.best_eval:
                    self.best_eval = score
                    self.save("best.pt")
                self.save("latest.pt")
                if config.keep_eval_checkpoints:
                    self.save(f"checkpoints/iter_{self.iteration:04d}.pt")

            self.log(metrics)
            print(
                f"iter {self.iteration:3d} data {len(self.dataset):6d} "
                f"level {metrics['imitation/level_loss']:.3f} "
                f"rank {metrics['imitation/rank_loss']:.3f} "
                f"pair_acc {metrics['imitation/pair_accuracy']:.3f} "
                f"ent {metrics['imitation/entropy']:.3f} "
                + "".join(
                    f"| eval {d} vs_fcfs "
                    f"{metrics[f'eval/{d}/improvement_over_fcfs']:+.3f} "
                    f"sjf_gap {metrics[f'eval/{d}/mean_gap_closed']:+.3f} "
                    f"agree {metrics.get(f'eval/{d}/pair_agreement', float('nan')):.3f} "
                    for d in config.eval_decodings
                    if f"eval/{d}/improvement_over_fcfs" in metrics
                )
                + f"({collect_s:.0f}s + {metrics['time/fit_s']:.0f}s)",
                flush=True,
            )

            if self.iteration < imitation.iterations:
                start = time.time()
                collected = self.collect(
                    imitation.seeds_per_iteration, imitation.beta(self.iteration)
                )
                collect_s = time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="dotted override, e.g. imitation.iterations=5",
    )
    args = parser.parse_args()

    trainer = ImitationTrainer(load_config(args.config, args.set))
    print(
        f"{trainer.model.num_parameters()} parameters on {trainer.device}, "
        f"teacher {trainer.imitation.teacher}, output in {trainer.config.output_dir}",
        flush=True,
    )
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
