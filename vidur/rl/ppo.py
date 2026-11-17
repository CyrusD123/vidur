"""
PPO for the queue-ranking scheduler (docs/rl_plan.md §3, §4, §6 step 6).

- Continuous-time discounting: a transition lasting d seconds discounts what
  follows by exp(-d / tau_s).
- Advantages: "gae" (critic), or "group" (time-aligned leave-one-out over the
  rollouts of the same workload seed: the baseline for a decision at time t is
  the other rollouts' mean discounted return-to-go from t). "group" needs
  group_size > 1 and is the default, per the fork experiment.
- Policy loss: clipped PPO on the coarsened log-probability
  Σ_i log P(low_i <= a_i <= high_i) over the requests this decision ranked.
  Transitions where everything was admitted have log-prob 0 and drop out.
- Value loss (decomposed critic, V = Σ v_j + v_global): each per-request value
  v_j is fit to that request's own discounted return, and v_global to the rest
  of the total return (future arrivals, batch_time). Per-request returns give
  the encoder supervision and V; they are never used for the policy.
- Returns are divided by a running std, so reward scale changes (QPS, reward
  weights) do not need retuning.
- Optional imitation anchor (imitation_coef > 0, for fine-tuning an imitation
  policy): the teacher's imitation loss (vidur/rl/teacher.py) on the rollouts'
  labeled decisions is added to the policy loss.
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from vidur.rl.model import SchedulerModel, collate, interval_targets, modeled_indices
from vidur.rl.rollout import Trajectory
from vidur.rl.teacher import imitation_losses, imitation_targets

ADVANTAGE_MODES = ("group", "gae")
ADVANTAGE_NORMALIZATIONS = ("workload", "batch", "none")
STEP_STATS = (
    "policy_loss",
    "value_loss",
    "request_value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
    "value_grad_norm",
    "imitation_loss",
)


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 64
    # continuous-time discount: exp(-duration / tau_s) per transition
    tau_s: float = 25.0
    gae_lambda: float = 0.95
    advantage: str = "group"
    value_coef: float = 0.5
    request_value_coef: float = 0.5
    entropy_coef: float = 0.003
    max_grad_norm: float = 0.5
    # Build minibatches from states of similar size (number of tokens), so less
    # attention compute goes to padding; minibatch order is still shuffled.
    # The jitter (in tokens) varies minibatch composition between epochs.
    bucket_by_length: bool = True
    # learner device: "auto" picks cuda, then mps, then cpu. Rollout workers
    # always act on CPU.
    device: str = "auto"
    # round padded token counts up to this multiple (fewer distinct shapes)
    pad_to_multiple: int = 32
    bucket_jitter: float = 16.0
    # stop the epoch loop once the approximate KL passes this (None: never)
    target_kl: float = 0.05
    normalize_returns: bool = True
    # Scale advantages to unit std separately per workload ("workload"), over
    # the whole batch ("batch"), or not at all. Per workload keeps a workload
    # where reordering moves the cost a lot (memory pressure) from drowning out
    # the signal from one where it moves it little.
    advantage_normalization: str = "workload"
    # weight of the imitation loss (level + rank terms) on decisions labeled by
    # a teacher; the rollouts must carry labels (RolloutTask.teacher)
    imitation_coef: float = 0.0

    def __post_init__(self):
        assert (
            self.advantage in ADVANTAGE_MODES
        ), f"advantage must be in {ADVANTAGE_MODES}"
        assert (
            self.advantage_normalization in ADVANTAGE_NORMALIZATIONS
        ), f"advantage_normalization must be in {ADVANTAGE_NORMALIZATIONS}"


class RunningStd:
    """Running standard deviation (Welford, merged per batch)."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if not len(values):
            return
        n, mean, m2 = len(values), values.mean(), ((values - values.mean()) ** 2).sum()
        delta = mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.m2 += m2 + delta**2 * self.count * n / total
        self.count = total

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max(math.sqrt(self.m2 / (self.count - 1)), 1e-8)

    def state_dict(self) -> Dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.count, self.mean, self.m2 = state["count"], state["mean"], state["m2"]


# --- returns and advantages --------------------------------------------------


def discounts(trajectory: Trajectory, tau_s: float) -> np.ndarray:
    durations = np.array([t.duration for t in trajectory.transitions])
    return np.exp(-durations / tau_s)


def discounted_returns(rewards: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """G_k = r_k + gamma_k G_{k+1}; rewards may be [T] or [T, R]."""
    returns = np.zeros_like(rewards, dtype=float)
    running = np.zeros_like(rewards[0], dtype=float) if len(rewards) else 0.0
    for k in range(len(rewards) - 1, -1, -1):
        running = rewards[k] + gammas[k] * running
        returns[k] = running
    return returns


def gae(
    rewards: np.ndarray, values: np.ndarray, gammas: np.ndarray, lam: float
) -> np.ndarray:
    advantages = np.zeros(len(rewards))
    running = 0.0
    for k in range(len(rewards) - 1, -1, -1):
        next_value = values[k + 1] if k + 1 < len(values) else 0.0
        delta = rewards[k] + gammas[k] * next_value - values[k]
        running = delta + gammas[k] * lam * running
        advantages[k] = running
    return advantages


def return_at(
    times: np.ndarray,
    durations: np.ndarray,
    rewards: np.ndarray,
    returns: np.ndarray,
    t: float,
) -> float:
    """
    A rollout's discounted return-to-go from time t, read off its transitions:
    the return of the transition in progress at t minus the part of its reward
    already accrued (spread linearly over its duration). Before the first
    decision it is the first transition's return.
    """
    k = int(np.searchsorted(times, t, side="right")) - 1
    if k < 0:
        return float(returns[0])
    fraction = min(1.0, (t - times[k]) / durations[k]) if durations[k] > 0 else 1.0
    return float(returns[k] - rewards[k] * fraction)


def group_advantages(
    group: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> List[np.ndarray]:
    """
    Time-aligned leave-one-out baseline over rollouts of one workload seed.
    Each entry is (times, durations, rewards, returns) of one rollout.
    """
    advantages = []
    for i, (times, _, _, returns) in enumerate(group):
        others = [g for j, g in enumerate(group) if j != i]
        baseline = np.array(
            [np.mean([return_at(*other, t) for other in others]) for t in times]
        )
        advantages.append(returns - baseline)
    return advantages


def minibatches(
    lengths: np.ndarray,
    size: int,
    rng: np.random.Generator,
    bucket: bool,
    jitter: float,
) -> List[np.ndarray]:
    """
    Index arrays covering every sample once, in shuffled order. With bucket,
    each minibatch holds samples of similar length (sorted by length plus
    uniform jitter), so padding is small.
    """
    if bucket:
        order = np.argsort(
            lengths + rng.uniform(0, jitter, len(lengths)), kind="stable"
        )
    else:
        order = rng.permutation(len(lengths))
    chunks = [order[i : i + size] for i in range(0, len(order), size)]
    return [chunks[i] for i in rng.permutation(len(chunks))]


def normalize_advantages(samples, relevant: np.ndarray, workloads, mode: str) -> None:
    """In place: zero mean, unit std per workload or over the batch, with the
    statistics taken over the relevant (admission-affecting) samples."""
    if mode == "none":
        return
    keys = np.array(workloads) if mode == "workload" else np.zeros(len(samples))
    advantages = np.array([s.advantage for s in samples])
    for key in np.unique(keys):
        members = keys == key
        stats = advantages[members & relevant]
        if len(stats) < 2:
            continue
        mean, std = stats.mean(), stats.std() + 1e-8
        for i in np.flatnonzero(members):
            samples[i].advantage = (advantages[i] - mean) / std


# --- the update ---------------------------------------------------------------


@dataclass
class _Sample:
    trajectory: int
    index: int
    modeled: np.ndarray
    advantage: float
    value_target: float
    # per-token targets: waiting (modeled order) then running
    waiting_targets: np.ndarray
    running_targets: np.ndarray
    old_log_prob: float = 0.0


def _request_targets(
    trajectory: Trajectory,
    column: Dict[int, int],
    request_returns: np.ndarray,
    index: int,
    modeled: np.ndarray,
    max_running: int,
) -> Tuple[np.ndarray, np.ndarray]:
    obs = trajectory.transitions[index].obs
    row = request_returns[index]
    waiting = np.array(
        [row[column[obs.request_ids[i]]] for i in modeled], dtype=np.float32
    )
    running = np.array(
        [row[column[rid]] for rid in obs.running_request_ids[:max_running]],
        dtype=np.float32,
    )
    return waiting, running


def _device_of(model: SchedulerModel) -> torch.device:
    return next(model.parameters()).device


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class _PhaseTimer:
    """Wall time per phase, synchronizing the device so GPU work is counted
    in the phase that launched it."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._last = time.perf_counter()
        self.phases: Dict[str, float] = {}

    def mark(self, name: str) -> None:
        if self._device.type == "mps":
            torch.mps.synchronize()
        elif self._device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.phases[name] = now - self._last
        self._last = now


def make_optimizer(
    model: SchedulerModel, learning_rate: float
) -> torch.optim.Optimizer:
    """Adam with the fused kernel where supported (cuda, mps), else foreach;
    the per-parameter fallback launches ~10 small kernels per parameter."""
    device = _device_of(model)
    if device.type in ("cuda", "mps"):
        try:
            return torch.optim.Adam(model.parameters(), lr=learning_rate, fused=True)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=True)


def _padded(rows: Sequence[np.ndarray], width: int) -> torch.Tensor:
    out = np.zeros((len(rows), width), np.float32)
    for b, row in enumerate(rows):
        out[b, : len(row)] = row
    return torch.from_numpy(out)


def _forward(
    model: SchedulerModel,
    samples: Sequence[_Sample],
    trajectories,
    pad_to_multiple: int = 1,
) -> Tuple:
    transitions = [trajectories[s.trajectory].transitions[s.index] for s in samples]
    observations = [t.obs for t in transitions]
    modeled = [s.modeled for s in samples]
    device = _device_of(model)
    batch = collate(observations, model.config, modeled, pad_to_multiple).to(device)
    out = model(batch)
    targets = {
        k: v.to(device)
        for k, v in interval_targets(
            transitions, modeled, batch.waiting.shape[1]
        ).items()
    }
    mask = targets["loss_mask"] & batch.waiting_mask
    per_request = out.dist.interval_log_prob(targets["low"], targets["high"])
    log_prob = (per_request * mask).sum(1)
    return out, batch, mask, log_prob


def ppo_update(
    model: SchedulerModel,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[Trajectory],
    config: PPOConfig,
    return_std: RunningStd,
    rng: np.random.Generator,
) -> Dict[str, float]:
    timer = _PhaseTimer(_device_of(model))
    # rewards, scaled by the running std of returns
    gammas = [discounts(t, config.tau_s) for t in trajectories]
    raw_returns = [
        discounted_returns(np.array([x.reward for x in t.transitions]), g)
        for t, g in zip(trajectories, gammas)
    ]
    if config.normalize_returns:
        return_std.update(np.concatenate(raw_returns))
    scale = 1.0 / return_std.std if config.normalize_returns else 1.0

    # build samples (values for GAE from the current model, before any update)
    samples: List[_Sample] = []
    groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i, t in enumerate(trajectories):
        groups[(t.task.workload, t.task.workload_seed)].append(i)
    if config.advantage == "group":
        assert all(
            len(g) > 1 for g in groups.values()
        ), "group advantages need group_size > 1"

    per_trajectory = []
    for i, trajectory in enumerate(trajectories):
        rewards = np.array([x.reward for x in trajectory.transitions]) * scale
        returns = raw_returns[i] * scale
        request_returns = discounted_returns(
            trajectory.request_rewards * scale, gammas[i]
        )
        times = np.array([x.time for x in trajectory.transitions])
        durations = np.array([x.duration for x in trajectory.transitions])
        per_trajectory.append((times, durations, rewards, returns, request_returns))

    if config.advantage == "group":
        advantages = [None] * len(trajectories)
        for members in groups.values():
            group = [per_trajectory[i][:4] for i in members]
            for i, adv in zip(members, group_advantages(group)):
                advantages[i] = adv
    else:
        advantages = []
        with torch.no_grad():
            for i, trajectory in enumerate(trajectories):
                observations = [x.obs for x in trajectory.transitions]
                values = []
                for start in range(0, len(observations), 256):
                    chunk = observations[start : start + 256]
                    batch = collate(chunk, model.config).to(_device_of(model))
                    values.append(model(batch).value.cpu().numpy())
                values = np.concatenate(values) if values else np.zeros(0)
                _, _, rewards, _, _ = per_trajectory[i]
                advantages.append(gae(rewards, values, gammas[i], config.gae_lambda))

    for i, trajectory in enumerate(trajectories):
        _, _, _, returns, request_returns = per_trajectory[i]
        column = {rid: c for c, rid in enumerate(trajectory.request_ids)}
        for k, transition in enumerate(trajectory.transitions):
            modeled = modeled_indices(transition.obs, model.config.max_queue_len)
            waiting_targets, running_targets = _request_targets(
                trajectory,
                column,
                request_returns,
                k,
                modeled,
                model.config.max_running_len,
            )
            samples.append(
                _Sample(
                    i,
                    k,
                    modeled,
                    float(advantages[i][k]),
                    float(returns[k]),
                    waiting_targets,
                    running_targets,
                )
            )

    # normalize advantages, using the decisions that affected admission
    has_boundary = np.array(
        [
            trajectories[s.trajectory].transitions[s.index].boundary_index is not None
            for s in samples
        ]
    )
    normalize_advantages(
        samples,
        has_boundary,
        [trajectories[s.trajectory].task.workload for s in samples],
        config.advantage_normalization,
    )

    timer.mark("prepare_s")
    lengths = np.array(
        [len(s.modeled) + len(s.running_targets) for s in samples],
        dtype=float,
    )

    # old log-probs: the weights that collected the rollouts
    model.eval()
    with torch.no_grad():
        for indices in minibatches(lengths, 256, rng, bucket=True, jitter=0.0):
            chunk = [samples[j] for j in indices]
            _, _, _, log_prob = _forward(
                model, chunk, trajectories, config.pad_to_multiple
            )
            for s, lp in zip(chunk, log_prob.tolist()):
                s.old_log_prob = lp

    timer.mark("old_log_prob_s")
    model.train()
    stats = defaultdict(list)
    stopped_early = False
    for epoch in range(config.epochs):
        step_stats = []
        for indices in minibatches(
            lengths,
            config.minibatch_size,
            rng,
            config.bucket_by_length,
            config.bucket_jitter,
        ):
            chunk = [samples[j] for j in indices]
            out, batch, mask, log_prob = _forward(
                model, chunk, trajectories, config.pad_to_multiple
            )

            device = log_prob.device
            advantage = torch.tensor(
                [s.advantage for s in chunk], dtype=torch.float32, device=device
            )
            old = torch.tensor(
                [s.old_log_prob for s in chunk], dtype=torch.float32, device=device
            )
            relevant = torch.tensor(
                [
                    trajectories[s.trajectory].transitions[s.index].boundary_index
                    is not None
                    for s in chunk
                ],
                device=device,
            )
            log_ratio = (log_prob - old).clamp(-20, 20)
            ratio = log_ratio.exp()
            surrogate = torch.minimum(
                ratio * advantage,
                ratio.clamp(1 - config.clip, 1 + config.clip) * advantage,
            )
            num_relevant = relevant.sum().clamp(min=1)
            policy_loss = -(surrogate * relevant).sum() / num_relevant

            value_target = torch.tensor(
                [s.value_target for s in chunk], dtype=torch.float32, device=device
            )
            # V = Σ v_j + v_global. The per-request values learn from their own
            # returns (below); the total return only trains v_global, on the
            # part the present requests do not explain. Otherwise the error of
            # V would reach the per-request head once per token.
            request_sum = out.waiting_values.sum(1) + out.running_values.sum(1)
            value_loss = (
                (out.global_value + request_sum.detach() - value_target) ** 2
            ).mean()

            waiting_t = _padded(
                [s.waiting_targets for s in chunk], out.waiting_values.shape[1]
            ).to(device)
            running_t = _padded(
                [s.running_targets for s in chunk], out.running_values.shape[1]
            ).to(device)
            token_mask = torch.cat([batch.waiting_mask, batch.running_mask], 1).float()
            token_err = torch.cat(
                [out.waiting_values - waiting_t, out.running_values - running_t], 1
            )
            request_value_loss = (
                token_err**2 * token_mask
            ).sum() / token_mask.sum().clamp(min=1)

            entropy = (out.dist.entropy() * mask).sum() / mask.sum().clamp(min=1)

            imitation_loss = torch.zeros((), device=device)
            if config.imitation_coef > 0:
                transitions = [
                    trajectories[s.trajectory].transitions[s.index] for s in chunk
                ]
                targets = {
                    k: v.to(device)
                    for k, v in imitation_targets(
                        transitions,
                        [s.modeled for s in chunk],
                        batch.waiting.shape[1],
                    ).items()
                }
                losses = imitation_losses(out.dist, targets, mask)
                weight = (targets["labeled"] & relevant).float()
                imitation_loss = (
                    (losses["level_loss"] + losses["rank_loss"]) * weight
                ).sum() / weight.sum().clamp(min=1)

            loss = (
                policy_loss
                + config.value_coef * value_loss
                + config.request_value_coef * request_value_loss
                - config.entropy_coef * entropy
                + config.imitation_coef * imitation_loss
            )
            optimizer.zero_grad()
            loss.backward()
            # clipped per group, so the (larger) value gradients do not shrink
            # the policy's
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.policy_parameters(), config.max_grad_norm
            )
            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.value_parameters(), config.max_grad_norm
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = (((ratio - 1) - log_ratio) * relevant).sum() / num_relevant
                clipped = (
                    ((ratio - 1).abs() > config.clip) & relevant
                ).float().sum() / num_relevant
            # kept on the device; reading them every step would block on it
            step_stats.append(
                torch.stack(
                    [
                        policy_loss.detach(),
                        value_loss.detach(),
                        request_value_loss.detach(),
                        entropy.detach(),
                        approx_kl,
                        clipped,
                        grad_norm,
                        value_grad_norm,
                        imitation_loss.detach(),
                    ]
                )
            )
        epoch_stats = torch.stack(step_stats).cpu().numpy()
        for column, name in enumerate(STEP_STATS):
            stats[name].extend(epoch_stats[:, column].tolist())
        stats["epochs"] = [epoch + 1]
        if (
            config.target_kl is not None
            and epoch_stats[:, STEP_STATS.index("approx_kl")].mean() > config.target_kl
        ):
            stopped_early = True
            break

    timer.mark("epochs_s")
    model.eval()
    summary = {name: float(np.mean(values)) for name, values in stats.items()}
    summary.update(timer.phases)
    summary["stopped_early"] = float(stopped_early)
    summary["return_std"] = return_std.std
    summary["num_samples"] = len(samples)
    summary["num_policy_samples"] = int(has_boundary.sum())
    return summary
