"""
The Vidur RL stack (features, rewards, PPO) on Frontier's simulator:

    WorkloadConfig(name="qps20", qps=20, backend="frontier", frontier={...})

FrontierEnv is a VidurSchedulingEnv whose episodes run in Frontier (../Frontier
next to this repo, or $FRONTIER_ROOT), on its vLLM v1 scheduler in priority
mode, through the hook in frontier/rl/scheduler.py. Everything the trainer sees
is built the Vidur way:

- Requests come from Vidur's own request generators for the workload seed,
  written to a CSV and replayed by Frontier, so both simulators see the same
  requests (and request ids, 0..N-1 in arrival order).
- Observations are the vidur/rl/features.py features, computed from Frontier's
  scheduler state; FrontierFeatureScheduler keeps the request histories and
  sliding windows that RLVLLMReplicaScheduler keeps.
- As in vidur/rl/env.py, the policy ranks only when at least 2 requests wait,
  a ranking that admitted nothing is undone and dropped, and each kept ranking
  gets admission action bounds.
- Rewards are computed after the episode from request and token timings with
  vidur/rl/reward.py, which gives the numbers its online RewardTracker would.

What stays Frontier's: batch timing (its profiled predictors, CPU overhead),
admission and preemption. Two of those differ from Vidur's scheduler: vLLM
admits a request when its first chunk fits (Vidur: its whole prompt), and in
priority mode it preempts the running request with the largest (priority,
arrival), so a request's priority still matters after admission (Vidur
preempts the most recently admitted one).

Importing this module swaps sklearn's single-row forest prediction for
vidur/rl/fast_forest.py (bit-identical, ~10x faster episodes); set
VIDUR_FRONTIER_FAST_FOREST=0 to keep sklearn's.
"""

import copy
import hashlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from vidur.config import SimulationConfig
from vidur.entities.replica import Replica
from vidur.request_generator import RequestGeneratorRegistry
from vidur.rl import fast_forest
from vidur.rl.action import admission_action_bounds
from vidur.rl.agents import BaseAgent
from vidur.rl.env import (
    Episode,
    Transition,
    VidurSchedulingEnv,
    _act,
    _set_workload_seed,
    _summarize,
)
from vidur.rl.observation import ObservationConfig, build_observation
from vidur.rl.reward import (
    RewardConfig,
    episode_cost,
    per_request_reward_components,
    zero_components,
)
from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
    WINDOW_METRICS,
    WINDOW_SIZES,
    RequestHistory,
    SlidingWindow,
)
from vidur.scheduler.utils.memory_planner import MemoryPlanner

FRONTIER_ROOT = os.path.abspath(
    os.environ.get(
        "FRONTIER_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "Frontier"),
    )
)
if FRONTIER_ROOT not in sys.path:
    sys.path.insert(0, FRONTIER_ROOT)

from frontier.entities import Request as FrontierRequest  # noqa: E402
from frontier.rl.env import (  # noqa: E402
    WorkloadSpec,
    _engine_key,
    _RLSimulator,
    _shared_predictors,
    _working_dir,
)
from frontier.rl.env import make_simulation_config as frontier_config  # noqa: E402
from frontier.rl.scheduler import (  # noqa: E402
    PriorityPolicy,
    RLVLLMv1EngineReplicaScheduler,
    replica_schedulers,
)
from frontier.scheduler.replica_scheduler.replica_scheduler_registry import (  # noqa: E402
    ReplicaSchedulerRegistry,
)
from frontier.types import ReplicaSchedulerType  # noqa: E402
from frontier.utils.random import set_seeds  # noqa: E402

if os.environ.get("VIDUR_FRONTIER_FAST_FOREST", "1") != "0":
    fast_forest.install()

# WorkloadSpec fields (frontier/rl/env.py) a workload's `frontier` dict can set.
# Llama-2-7B on H200 is the dense model Frontier has full profiles for (mixed
# and chunked-prefill attention, CPU overhead).
FRONTIER_DEFAULTS: Dict[str, Any] = {
    "model": "llama2_7b_dense_example",
    "device": "h200",
    "network_device": "h200_dgx",
    "tensor_parallel": 1,
    "model_cpu_overhead": True,
    "prediction_rounding_bits": None,
    "enable_prefix_caching": False,
}


class FrontierFeatureScheduler(RLVLLMv1EngineReplicaScheduler):
    """
    RLVLLMv1EngineReplicaScheduler plus the per-request histories and
    sliding-window statistics of RLVLLMReplicaScheduler, which the Vidur
    features read. Histories are updated by _BridgePolicy, which sees the
    rankings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: Dict[int, RequestHistory] = {}
        self._windows = {
            (metric, window): SlidingWindow(size)
            for metric in WINDOW_METRICS
            for window, size in WINDOW_SIZES.items()
        }

    def window_stats(self, metric: str, window: str):
        return self._windows[(metric, window)].stats()

    def _record(self, metric: str, value: float) -> None:
        for window in WINDOW_SIZES:
            self._windows[(metric, window)].add(value)

    def _get_next_batch(self, is_micro_batch: bool = False):
        self._record(
            "queue_length", len(self._preempted_requests) + len(self._request_queue)
        )
        self._record("free_block_fraction", self.num_free_blocks / self.num_blocks)
        batch = super()._get_next_batch(is_micro_batch)
        if batch:
            self._record("batch_num_tokens", batch.total_num_tokens)
        return batch

    def on_batch_end(self, batch) -> None:
        self._record("batch_time", batch.completed_at - batch.scheduled_at)
        super().on_batch_end(batch)
        for request in batch.requests:
            if request.completed:
                self.history.pop(request.id, None)


@contextmanager
def _feature_scheduler_installed() -> Iterator[None]:
    key = ReplicaSchedulerType.VLLM_V1
    original = ReplicaSchedulerRegistry.get_class(key)
    ReplicaSchedulerRegistry._registry[key] = FrontierFeatureScheduler
    try:
        yield
    finally:
        ReplicaSchedulerRegistry._registry[key] = original


@dataclass
class _RequestView:
    """The Vidur Request attributes vidur/rl/features.py reads."""

    id: int
    arrived_at: float
    num_prefill_tokens: int
    num_processed_tokens: int
    num_processed_prefill_tokens: int
    num_processed_decode_tokens: int
    num_restarts: int
    is_prefill_complete: bool
    prefill_completed_at: float


def _view(request: FrontierRequest, waiting: bool, first_token_at: Dict[int, float]):
    prompt = request.num_prefill_tokens
    processed = request.num_processed_tokens
    if waiting:
        # Vidur restarts a preempted request with its output folded into the
        # prompt; a Frontier decode victim keeps those tokens as processed
        prompt = max(prompt, processed)
    return _RequestView(
        id=request.id,
        arrived_at=request.arrived_at,
        num_prefill_tokens=prompt,
        num_processed_tokens=processed,
        num_processed_prefill_tokens=min(processed, prompt),
        num_processed_decode_tokens=request.num_processed_decode_tokens,
        num_restarts=request.get_total_preemption_count(),
        is_prefill_complete=request.is_prefill_complete,
        prefill_completed_at=first_token_at.get(request.id, 0.0),
    )


class _SchedulerView:
    """What vidur/rl/observation.py, features.py and the baselines read from
    RLVLLMReplicaScheduler, from a FrontierFeatureScheduler."""

    tie_break = "arrival"
    decision_mode = "every_batch"

    def __init__(
        self,
        scheduler: FrontierFeatureScheduler,
        num_priority_levels: int,
        first_token_at: Dict[int, float],
        rank_mask: List[bool],
    ) -> None:
        self._scheduler = scheduler
        self._first_token_at = first_token_at
        self.num_priority_levels = num_priority_levels
        self.rank_mask = rank_mask
        self.current_time = float(scheduler._current_schedule_time)
        self.num_blocks = scheduler.num_blocks
        self.num_free_blocks = scheduler.num_free_blocks
        self.block_size = scheduler.block_size
        self.batch_size_cap = scheduler.batch_size_cap
        self.max_num_batched_tokens = scheduler.max_tokens_in_batch
        # requests holding KV cache, as in Vidur
        self.num_running_requests = len(scheduler._allocation_map)

    @property
    def running_requests(self) -> List[_RequestView]:
        return [
            _view(r, False, self._first_token_at)
            for r in self._scheduler.running_requests
        ]

    def num_allocated_blocks_of(self, request_id: int) -> int:
        return self._scheduler._allocation_map.get(request_id, 0)

    def request_history(self, request_id: int) -> RequestHistory:
        return self._scheduler.history.setdefault(request_id, RequestHistory())

    def window_stats(self, metric: str, window: str):
        return self._scheduler.window_stats(metric, window)


class _BridgePolicy(PriorityPolicy):
    """
    Vidur's AgentPriorityPolicy and RLVLLMReplicaScheduler ranking logic on
    Frontier's hook, plus a log of request and token timings for the rewards.
    """

    def __init__(
        self,
        agent: BaseAgent,
        observation_config: ObservationConfig,
        num_priority_levels: int,
    ) -> None:
        self._agent = agent
        self._observation_config = observation_config
        self._num_levels = num_priority_levels
        self._pending = None
        self.transitions: List[Transition] = []
        self.requests: Dict[int, FrontierRequest] = {}
        self.first_token_at: Dict[int, float] = {}
        self.completed_at: Dict[int, float] = {}
        self._last_token_at: Dict[int, float] = {}
        self._emitted: Dict[int, int] = {}
        self.token_times: List[float] = []
        self.token_request_ids: List[int] = []
        self.token_gaps: List[float] = []
        # (end time, duration, number of requests) per batch
        self.batches: List[tuple] = []

    def get_priorities(self, waiting, scheduler):
        self._pending = None
        # Vidur ranks only when there is something to reorder
        if len(waiting) < 2:
            return None
        rank_mask = [True] * len(waiting)
        view = _SchedulerView(scheduler, self._num_levels, self.first_token_at, rank_mask)
        obs = build_observation(
            [_view(r, True, self.first_token_at) for r in waiting],
            view,
            self._observation_config,
        )
        action, info = _act(self._agent, obs, waiting, view)
        assert action.shape == (len(waiting),)
        if self._agent.unbounded_priorities:
            assert action.min() >= 0, action
        else:
            assert action.min() >= 0 and action.max() < self._num_levels, action
        self._pending = (
            obs,
            action,
            info,
            [(r.arrived_at, r.id) for r in waiting],
            [(r, r.priority) for r in waiting],
            scheduler,
        )
        return action

    def on_admission(self, admitted, boundary):
        if self._pending is None:
            return
        obs, action, info, arrival_keys, previous, scheduler = self._pending
        self._pending = None
        ranked = scheduler._ranked_waiting
        self._update_history(scheduler, ranked, admitted)

        if not admitted:
            # no effect on this batch: undo it, so it has none on preemption
            for request, priority in previous:
                request.set_priority(priority)
            return

        admitted_ids = {r.id for r in admitted}
        mask = np.array([i in admitted_ids for i in obs.request_ids], dtype=bool)
        boundary_index = (
            obs.request_ids.index(boundary.id) if boundary is not None else None
        )
        is_prefix = all(r.id in admitted_ids for r in ranked[: len(admitted)])
        if is_prefix:
            low, high = admission_action_bounds(
                action,
                arrival_keys,
                mask,
                boundary_index,
                max(obs.num_priority_levels, int(action.max()) + 1),
            )
        else:
            # admission skipped a request: the exact levels decided it
            low, high = action.copy(), action.copy()
        self.transitions.append(
            Transition(
                obs=obs,
                action=action,
                info=info,
                time=obs.time,
                num_admitted=len(admitted),
                admitted=mask,
                boundary_index=boundary_index,
                action_low=low,
                action_high=high,
            )
        )

    @staticmethod
    def _update_history(scheduler, ranked, admitted) -> None:
        # RLVLLMReplicaScheduler._update_history for a consulted every_batch decision
        admitted_ids = {r.id for r in admitted}
        admitted_arrivals = sorted(r.arrived_at for r in admitted)
        for position, request in enumerate(ranked):
            history = scheduler.history.setdefault(request.id, RequestHistory())
            history.num_decisions_seen += 1
            if admitted:
                history.last_rank_percentile = position / max(1, len(ranked) - 1)
            if request.id in admitted_ids or not admitted:
                continue
            history.times_passed_over += 1
            history.num_overtaken_by += sum(
                1 for arrival in admitted_arrivals if arrival > request.arrived_at
            )

    def on_batch_end(self, batch) -> None:
        time = float(batch.completed_at)
        self.batches.append((time, time - float(batch.scheduled_at), batch.size))
        for request in batch.requests:
            self.requests.setdefault(request.id, request)
            emitted = request.num_emitted_decode_tokens
            if request.id not in self.first_token_at:
                if request.is_prefill_complete:
                    self.first_token_at[request.id] = time
                    self._token(request.id, time, float("nan"))
            elif emitted > self._emitted[request.id]:
                self._token(request.id, time, time - self._last_token_at[request.id])
            self._emitted[request.id] = emitted
            if request.completed and request.id not in self.completed_at:
                self.completed_at[request.id] = float(request.completed_at)

    def _token(self, request_id: int, time: float, gap: float) -> None:
        self._last_token_at[request_id] = time
        self.token_times.append(time)
        self.token_request_ids.append(request_id)
        self.token_gaps.append(gap)


@dataclass
class RequestRecord:
    """The Vidur Request attributes vidur/rl/reward.py and env.py read."""

    id: int
    arrived_at: float
    num_prefill_tokens: int
    num_decode_tokens: int
    # first token (Vidur's restart() moves prefill_completed_at to the end of
    # the recompute; this is the first token the user saw)
    prefill_completed_at: float
    completed_at: float
    completed: bool
    num_restarts: int


def vidur_num_blocks(config: SimulationConfig) -> int:
    """The KV cache size Vidur gives this config (sized from GPU memory when
    num_blocks is None), so Frontier runs hold the same number of tokens."""
    scheduler_config = config.cluster_config.replica_scheduler_config
    if scheduler_config.num_blocks:
        return int(scheduler_config.num_blocks)
    replica_config = config.cluster_config.replica_config
    generator_config = config.request_generator_config
    planner = MemoryPlanner(replica_config, Replica(replica_config, generator_config))
    blocks_per_sequence = generator_config.max_tokens // scheduler_config.block_size
    return int(blocks_per_sequence * planner.get_max_request_slots())


class FrontierEnv(VidurSchedulingEnv):
    """VidurSchedulingEnv with episodes simulated by Frontier (module docstring)."""

    def __init__(
        self,
        config: SimulationConfig,
        frontier: Optional[Dict[str, Any]] = None,
        reward_config: Optional[RewardConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
        num_priority_levels: int = 16,
    ) -> None:
        cluster_config = config.cluster_config
        assert cluster_config.num_replicas == 1, "Only single-replica is supported"
        scheduler_config = cluster_config.replica_scheduler_config
        assert scheduler_config.tie_break == "arrival"
        assert scheduler_config.decision_mode == "every_batch"
        assert not scheduler_config.long_prefill_token_threshold

        self._config = config
        self._reward_config = reward_config or RewardConfig()
        self._observation_config = observation_config or ObservationConfig()
        self._num_priority_levels = num_priority_levels
        self._execution_time_predictor = None
        self._reference_episodes = {}

        settings = dict(FRONTIER_DEFAULTS, **(frontier or {}))
        self._spec_kwargs = dict(
            settings,
            max_tokens_in_batch=scheduler_config.max_num_batched_tokens,
            batch_size_cap=scheduler_config.batch_size_cap,
            num_blocks=vidur_num_blocks(config),
            extra_args=[
                "--vllm_v1_scheduler_config_block_size",
                str(scheduler_config.block_size),
                "--vllm_v1_scheduler_config_watermark_blocks_fraction",
                str(scheduler_config.watermark_blocks_fraction),
            ]
            + list(settings.get("extra_args", [])),
        )
        output_dir = os.path.abspath(config.metrics_config.output_dir)
        self._output_dir = os.path.join(output_dir, "frontier")
        self._trace_dir = os.path.join(output_dir, "frontier_traces")
        os.makedirs(self._trace_dir, exist_ok=True)

        if self._reward_config.needs_calibration:
            self._reward_config = self.calibrate_thresholds(self._reward_config)

    def make_simulator(self, *args, **kwargs):
        raise NotImplementedError("FrontierEnv episodes run in Frontier")

    def workload_trace(
        self, workload_seed: Optional[int] = None, request_generator_config=None
    ) -> str:
        """CSV of the requests Vidur generates for this workload and seed."""
        generator_config = copy.deepcopy(
            request_generator_config or self._config.request_generator_config
        )
        if workload_seed is not None:
            _set_workload_seed(generator_config, workload_seed)
        key = hashlib.sha1(repr(generator_config).encode()).hexdigest()[:16]
        path = os.path.join(self._trace_dir, f"{key}.csv")
        if not os.path.exists(path):
            requests = RequestGeneratorRegistry.get(
                generator_config.get_type(), generator_config
            ).generate()
            frame = pd.DataFrame(
                {
                    "arrived_at": [r.arrived_at for r in requests],
                    "num_prefill_tokens": [r.num_prefill_tokens for r in requests],
                    "num_decode_tokens": [r.num_decode_tokens for r in requests],
                }
            ).sort_values("arrived_at", kind="stable")
            temporary = f"{path}.{os.getpid()}.tmp"
            frame.to_csv(temporary, index=False)
            os.replace(temporary, path)
        return path

    def run_episode(
        self,
        agent: BaseAgent,
        workload_seed: Optional[int] = None,
        request_generator_config=None,
        reward_config: Optional[RewardConfig] = None,
    ) -> Episode:
        reward_config = reward_config or self._reward_config
        context = self.reward_context(
            reward_config, workload_seed, request_generator_config
        )
        trace = self.workload_trace(workload_seed, request_generator_config)
        num_requests = len(pd.read_csv(trace))
        spec = WorkloadSpec(
            name="vidur_workload",
            num_requests=num_requests,
            arrivals="replay",
            replay_trace=trace,
            max_tokens=self._config.request_generator_config.max_tokens,
            **self._spec_kwargs,
        )
        seed = 0 if workload_seed is None else int(workload_seed)
        policy = _BridgePolicy(
            agent, self._observation_config, self._num_priority_levels
        )
        with _working_dir(FRONTIER_ROOT):
            config = frontier_config(spec, seed, self._output_dir, "error")
            set_seeds(seed)
            # ids 0..N-1 in trace order, so they match across episodes
            FrontierRequest._id = -1
            with _feature_scheduler_installed(), _shared_predictors(_engine_key(spec)):
                simulator = _RLSimulator(config)
            [scheduler] = replica_schedulers(simulator)
            scheduler.set_priority_policy(policy)
            simulator.run()
        end_time = float(simulator._time)
        return self._episode(policy, end_time, reward_config, context)

    def _episode(self, policy: _BridgePolicy, end_time, reward_config, context):
        requests = [policy.requests[i] for i in sorted(policy.requests)]
        records = [
            RequestRecord(
                id=r.id,
                arrived_at=float(r.arrived_at),
                num_prefill_tokens=int(r.num_prefill_tokens),
                num_decode_tokens=int(r.num_decode_tokens),
                prefill_completed_at=policy.first_token_at.get(r.id, end_time),
                completed_at=policy.completed_at.get(r.id, end_time),
                completed=r.id in policy.completed_at,
                num_restarts=int(r.get_total_preemption_count()),
            )
            for r in requests
        ]
        transitions = policy.transitions
        starts = np.array([t.time for t in transitions])
        ends = np.append(starts[1:], end_time)
        for transition, end in zip(transitions, ends):
            transition.duration = float(end - transition.time)

        episode = Episode(
            transitions=transitions,
            initial_reward_components=zero_components(),
            requests=records,
            num_decode_tokens={r.id: r.num_decode_tokens for r in records},
            token_times=np.array(policy.token_times),
            token_request_ids=np.array(policy.token_request_ids, dtype=np.int64),
            token_gaps=np.array(policy.token_gaps),
            request_thresholds=context.request_thresholds,
            reward_scale=context.scale,
            reference_costs=dict(context.reference_costs),
        )
        episode.batch_sizes = np.array([size for _, _, size in policy.batches])

        # batch_time, credited like tokens: a batch ending at t belongs to the
        # transition whose interval (start, end] holds t
        batch_time = np.zeros(len(transitions))
        for time, duration, _ in policy.batches:
            row = int(np.searchsorted(starts, time, side="left")) - 1
            if row < 0:
                episode.initial_reward_components["batch_time"] += duration
            else:
                batch_time[row] += duration

        if transitions:
            per_request = per_request_reward_components(episode, reward_config)
            for k, transition in enumerate(transitions):
                components = zero_components()
                for name, values in per_request.items():
                    components[name] = float(values[k].sum())
                components["batch_time"] = float(batch_time[k])
                transition.reward_components = components
                transition.reward = context.scale * reward_config.combine(components)

        episode.cost = episode_cost(episode, reward_config)
        episode.summary = _summarize(episode, end_time)
        episode.summary["num_preemptions"] = int(sum(r.num_restarts for r in records))
        return episode
