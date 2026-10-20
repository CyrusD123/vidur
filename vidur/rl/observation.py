from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from vidur.entities import Request
from vidur.rl.features import (
    GLOBAL_FEATURES,
    RUNNING_FEATURES,
    WAITING_FEATURES,
    Feature,
    FeatureContext,
)

if TYPE_CHECKING:
    from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
        RLVLLMReplicaScheduler,
    )

DEFAULT_WAITING_FEATURES = (
    "num_prefill_tokens",
    "wait_time",
    "num_restarts",
    "num_blocks_needed",
    "blocks_needed_over_free",
    "arrival_rank",
    "is_new",
    "times_passed_over",
    "num_overtaken_by",
    "prev_rank_percentile",
    "has_prev_rank",
    "current_priority",
    "has_priority",
    "to_rank",
)
DEFAULT_RUNNING_FEATURES = tuple(RUNNING_FEATURES)
# everything registered: counts, free memory, and all sliding-window statistics
DEFAULT_GLOBAL_FEATURES = tuple(GLOBAL_FEATURES)


def _lookup(registry, names, kind) -> Tuple[Feature, ...]:
    unknown = [name for name in names if name not in registry]
    assert not unknown, f"Unknown {kind} features {unknown}. Known: {list(registry)}"
    return tuple(registry[name] for name in names)


@dataclass
class ObservationConfig:
    """Which registered features (vidur/rl/features.py) the policy observes."""

    waiting_features: Tuple[str, ...] = DEFAULT_WAITING_FEATURES
    running_features: Tuple[str, ...] = DEFAULT_RUNNING_FEATURES
    global_features: Tuple[str, ...] = DEFAULT_GLOBAL_FEATURES

    _waiting: Tuple[Feature, ...] = field(init=False, repr=False)
    _running: Tuple[Feature, ...] = field(init=False, repr=False)
    _global: Tuple[Feature, ...] = field(init=False, repr=False)

    def __post_init__(self):
        self.waiting_features = tuple(self.waiting_features)
        self.running_features = tuple(self.running_features)
        self.global_features = tuple(self.global_features)
        self._waiting = _lookup(WAITING_FEATURES, self.waiting_features, "waiting")
        self._running = _lookup(RUNNING_FEATURES, self.running_features, "running")
        self._global = _lookup(GLOBAL_FEATURES, self.global_features, "global")

    def __reduce__(self):
        # pickle by feature names (the resolved features hold functions)
        return (
            ObservationConfig,
            (self.waiting_features, self.running_features, self.global_features),
        )

    @property
    def num_waiting_features(self) -> int:
        return len(self.waiting_features)

    @property
    def num_running_features(self) -> int:
        return len(self.running_features)

    @property
    def num_global_features(self) -> int:
        return len(self.global_features)


@dataclass
class Observation:
    # [num_waiting, num_waiting_features], in current queue order (= action order)
    waiting_features: np.ndarray
    # [num_running, num_running_features]
    running_features: np.ndarray
    # [num_global_features]
    global_features: np.ndarray
    # [num_waiting] whether this decision sets the request's priority (all True
    # except in arrival_and_periodic mode between full re-rankings)
    rank_mask: np.ndarray
    num_priority_levels: int
    time: float
    request_ids: List[int]
    running_request_ids: List[int]
    # [num_waiting] raw arrival times; not a feature (absolute time), but with
    # request_ids it gives the tie-break order of the ranking
    arrival_times: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def ranking_keys(self, action: np.ndarray) -> List[Tuple[int, float, int]]:
        """Sort key of each waiting request under `action` (tie_break=arrival)."""
        return [
            (int(p), float(t), i)
            for p, t, i in zip(action, self.arrival_times, self.request_ids)
        ]

    @property
    def num_requests(self) -> int:
        return len(self.request_ids)

    @property
    def num_running(self) -> int:
        return len(self.running_request_ids)


def _stack(features: Tuple[Feature, ...], ctx: FeatureContext, n: int) -> np.ndarray:
    if not features or n == 0:
        return np.zeros((n, len(features)), dtype=np.float32)
    return np.stack([f.compute(ctx) for f in features], axis=1)


def build_observation(
    waiting_requests: List[Request],
    scheduler: "RLVLLMReplicaScheduler",
    config: ObservationConfig,
) -> Observation:
    running_requests = scheduler.running_requests
    ctx = FeatureContext(
        time=scheduler.current_time,
        scheduler=scheduler,
        waiting_requests=waiting_requests,
        waiting_histories=[scheduler.request_history(r.id) for r in waiting_requests],
        rank_mask=scheduler.rank_mask,
        running_requests=running_requests,
    )

    global_features = np.array(
        [f.compute(ctx) for f in config._global], dtype=np.float32
    ).reshape(len(config._global))

    return Observation(
        waiting_features=_stack(config._waiting, ctx, len(waiting_requests)),
        running_features=_stack(config._running, ctx, len(running_requests)),
        global_features=global_features,
        rank_mask=np.asarray(scheduler.rank_mask, dtype=bool),
        num_priority_levels=scheduler.num_priority_levels,
        time=scheduler.current_time,
        request_ids=[r.id for r in waiting_requests],
        running_request_ids=[r.id for r in running_requests],
        arrival_times=np.array([r.arrived_at for r in waiting_requests]),
    )
