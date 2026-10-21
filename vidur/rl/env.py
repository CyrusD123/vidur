import atexit
import copy
import dataclasses
import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from vidur.config import (
    BaseRequestGeneratorConfig,
    ClusterConfig,
    MetricsConfig,
    PoissonRequestIntervalGeneratorConfig,
    ReplicaConfig,
    RlVllmSchedulerConfig,
    SimulationConfig,
    SyntheticRequestGeneratorConfig,
    TraceRequestLengthGeneratorConfig,
    TraceWindowRequestGeneratorConfig,
)
from vidur.config.utils import get_all_subclasses
from vidur.entities import Request
from vidur.entities.base_entity import BaseEntity
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.events.batch_end_event import BatchEndEvent
from vidur.execution_time_predictor import (
    BaseExecutionTimePredictor,
    ExecutionTimePredictorRegistry,
)
from vidur.rl.action import admission_action_bounds
from vidur.rl.agents import BaseAgent, FCFSAgent, OracleAgent, baseline_agent
from vidur.rl.observation import Observation, ObservationConfig, build_observation
from vidur.rl.reward import (
    MEAN_COMPONENT_REFERENCES,
    THRESHOLD_METRICS,
    RewardConfig,
    RewardContext,
    RewardTracker,
    episode_cost,
    episode_cost_components,
    episode_metric_samples,
    window_thresholds,
    zero_components,
)
from vidur.scheduler.replica_scheduler.priority_policy import BasePriorityPolicy
from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
    RLVLLMReplicaScheduler,
)
from vidur.simulator import Simulator
from vidur.types import ReplicaSchedulerType


@dataclass
class Transition:
    obs: Observation
    action: np.ndarray
    # whatever the agent returned alongside the action (e.g. log_prob, value)
    info: Dict[str, Any]
    time: float
    num_admitted: int
    # index of the simulator event during which the decision was made (to replay
    # or fork the simulation at this decision)
    event_index: int = -1
    # [num_requests] whether each waiting request was admitted into the batch
    admitted: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # index (in obs order) of the first request in ranked order that was not
    # admitted; None if all were
    boundary_index: Optional[int] = None
    # inclusive per-request bounds on the priority that would have produced the
    # same admission (see vidur/rl/action.py); the training log-prob is
    # Σ_i log P(action_low_i <= a_i <= action_high_i) over obs.rank_mask
    action_low: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    action_high: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    # filled in at the next decision (or episode end): what accumulated between
    # this decision and the next one
    reward: float = 0.0
    reward_components: Dict[str, float] = field(default_factory=zero_components)
    duration: float = 0.0


@dataclass
class Episode:
    transitions: List[Transition]
    # accumulated before the first decision; no action can be credited with it
    initial_reward_components: Dict[str, float]
    requests: List[Request]
    # original output length per request id (Request.restart() rewrites
    # num_decode_tokens to the number of tokens left)
    num_decode_tokens: Dict[int, int]
    # every generated token: time, request id, gap since the request's previous
    # token (nan for its first token)
    token_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    token_request_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    token_gaps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    summary: Dict[str, float] = field(default_factory=dict)
    # per-request thresholds the rewards used (reference_window mode)
    request_thresholds: Dict[str, Dict[int, float]] = field(default_factory=dict)
    # every reward was multiplied by this (1 / baseline cost or gap when normalizing)
    reward_scale: float = 1.0
    # unscaled episode cost (Σ weight * component, including what accrued before
    # the first decision) and the same for the normalization baselines
    cost: float = float("nan")
    reference_costs: Dict[str, float] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return sum(transition.reward for transition in self.transitions)

    @property
    def normalized_score(self) -> Optional[float]:
        """
        (cost_a - cost) / (cost_a - cost_b) for normalize_by_baselines=(a, b),
        with cost_b = 0 for (a,): 0 = as good as a, 1 = as good as b (or zero
        cost). None when not normalizing.
        """
        if not self.reference_costs:
            return None
        cost_a, cost_b = (list(self.reference_costs.values()) + [0.0])[:2]
        if cost_a == cost_b:
            return float("nan")
        return (cost_a - self.cost) / (cost_a - cost_b)


def _act(
    agent: BaseAgent,
    obs: Observation,
    waiting_requests: List[Request],
    scheduler: RLVLLMReplicaScheduler,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(agent, OracleAgent):
        priorities, info = agent.act_oracle(obs, waiting_requests, scheduler)
    else:
        priorities, info = agent.act(obs)
    return np.asarray(priorities, dtype=np.int64), info


class AgentPolicy(BasePriorityPolicy):
    """Bridges the scheduler and an agent: builds the observation and asks the
    agent for priorities. Records nothing."""

    def __init__(self, agent: BaseAgent, observation_config: ObservationConfig):
        self._agent = agent
        self._observation_config = observation_config
        self.unbounded_priorities = agent.unbounded_priorities

    @property
    def agent(self) -> BaseAgent:
        return self._agent

    def get_priorities(
        self,
        waiting_requests: List[Request],
        scheduler: RLVLLMReplicaScheduler,
    ) -> List[int]:
        obs = build_observation(waiting_requests, scheduler, self._observation_config)
        priorities, _ = _act(self._agent, obs, waiting_requests, scheduler)
        return priorities.tolist()


class AgentPriorityPolicy(BasePriorityPolicy):
    """
    AgentPolicy that also records transitions. In every_batch mode, rankings
    that admitted nothing had no effect (the scheduler undoes them), so they are
    dropped and their reward flows to the previous transition. In
    arrival_and_periodic mode priorities persist, so every decision is kept.
    """

    def __init__(
        self,
        agent: BaseAgent,
        reward_tracker: RewardTracker,
        reward_config: RewardConfig,
        observation_config: ObservationConfig,
        event_index_fn: Callable[[], int] = lambda: -1,
        reward_scale: float = 1.0,
    ) -> None:
        self._agent = agent
        self._reward_scale = reward_scale
        self._reward_tracker = reward_tracker
        self._reward_config = reward_config
        self._observation_config = observation_config
        self._event_index_fn = event_index_fn
        self.unbounded_priorities = agent.unbounded_priorities

        self.transitions: List[Transition] = []
        self.initial_reward_components = zero_components()
        self._pending = None

    def get_priorities(
        self,
        waiting_requests: List[Request],
        scheduler: RLVLLMReplicaScheduler,
    ) -> List[int]:
        obs = build_observation(waiting_requests, scheduler, self._observation_config)
        action, info = _act(self._agent, obs, waiting_requests, scheduler)
        arrival_keys = [(r.arrived_at, r.id) for r in waiting_requests]
        self._pending = (obs, action, info, arrival_keys, scheduler)
        return action.tolist()

    def on_batch_formed(
        self,
        admitted_requests: List[Request],
        boundary_request: Optional[Request],
    ) -> None:
        obs, action, info, arrival_keys, scheduler = self._pending
        self._pending = None

        persistent = scheduler.decision_mode == "arrival_and_periodic"
        if not admitted_requests and not persistent:
            return

        admitted_ids = {r.id for r in admitted_requests}
        admitted = np.array([i in admitted_ids for i in obs.request_ids], dtype=bool)
        boundary_index = (
            obs.request_ids.index(boundary_request.id)
            if boundary_request is not None and boundary_request.id in obs.request_ids
            else None
        )

        if not persistent and scheduler.tie_break == "arrival":
            low, high = admission_action_bounds(
                action,
                arrival_keys,
                admitted,
                boundary_index,
                max(obs.num_priority_levels, int(action.max()) + 1),
            )
        else:
            # exact levels matter beyond this batch
            low, high = action.copy(), action.copy()

        self._credit_reward()
        self.transitions.append(
            Transition(
                obs=obs,
                action=action,
                info=info,
                time=obs.time,
                num_admitted=len(admitted_requests),
                event_index=self._event_index_fn(),
                admitted=admitted,
                boundary_index=boundary_index,
                action_low=low,
                action_high=high,
            )
        )

    def finish(self) -> None:
        self._credit_reward()

    def _credit_reward(self) -> None:
        components = self._reward_tracker.pop()

        if not self.transitions:
            for name, value in components.items():
                self.initial_reward_components[name] += value
            return

        transition = self.transitions[-1]
        transition.reward_components = components
        transition.reward = self._reward_scale * self._reward_config.combine(components)
        transition.duration = self._reward_tracker.time - transition.time


def _id_counter_classes() -> list:
    return [BaseEvent, BaseEntity] + get_all_subclasses(BaseEntity)


def _reset_id_counters() -> None:
    # ids are class-level counters; reset them so episodes are reproducible
    for entity_class in [BaseEntity] + get_all_subclasses(BaseEntity):
        entity_class._id = -1
    BaseEvent._id = 0


class RLSimulator(Simulator):
    """
    Simulator for running many episodes in one process: takes a prebuilt
    execution time predictor, skips writing output at exit, keeps the generated
    requests, and drives a RewardTracker from the event loop.

    It can also be paused (run(stop_at_event=...) / run(until_time=...)) and
    forked: fork() returns an independent deep copy that continues from the same
    state. Entity and event ids come from class-level counters shared by all
    simulators, so each simulator saves its own counters when it pauses and
    restores them when it resumes; forks and the original therefore produce the
    same ids as an uninterrupted run would.
    """

    def __init__(
        self,
        config: SimulationConfig,
        execution_time_predictor: BaseExecutionTimePredictor,
        reward_tracker: RewardTracker,
        reward_context: Optional[RewardContext] = None,
    ) -> None:
        super().__init__(config, execution_time_predictor)
        self.reward_context = reward_context or RewardContext()
        atexit.unregister(self._write_output)
        self._execution_time_predictor = execution_time_predictor
        self._reward_tracker = reward_tracker
        self._events_handled = 0
        self._id_counters = self._read_id_counters()

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def time(self) -> float:
        return self._time

    @property
    def reward_tracker(self) -> RewardTracker:
        return self._reward_tracker

    @property
    def events_handled(self) -> int:
        """Number of events handled so far, i.e. the index of the next event."""
        return self._events_handled

    @property
    def done(self) -> bool:
        return not self._event_queue or self._terminate

    @property
    def next_event_time(self) -> Optional[float]:
        return self._event_queue[0][1].time if self._event_queue else None

    @property
    def replica_schedulers(self) -> List[RLVLLMReplicaScheduler]:
        return [
            self._scheduler.get_replica_scheduler(replica_id)
            for replica_id in self._cluster.replicas
        ]

    def set_priority_policy(self, policy: BasePriorityPolicy) -> None:
        for replica_scheduler in self.replica_schedulers:
            replica_scheduler.set_priority_policy(policy)

    def _init_event_queue(self) -> None:
        self._requests = self._request_generator.generate()

        for request in self._requests:
            self._add_event(RequestArrivalEvent(request.arrived_at, request))

    @staticmethod
    def _read_id_counters() -> Dict[type, int]:
        return {cls: cls._id for cls in _id_counter_classes()}

    def _write_id_counters(self) -> None:
        for cls, value in self._id_counters.items():
            cls._id = value

    def run(
        self,
        stop_at_event: Optional[int] = None,
        until_time: Optional[float] = None,
    ) -> None:
        """
        Handle events until the simulation ends, or until stop_at_event events
        have been handled, or until the next event is after until_time (the
        reward is then integrated up to until_time exactly).
        """
        self._write_id_counters()
        try:
            while not self.done:
                if stop_at_event is not None and self._events_handled >= stop_at_event:
                    return
                if until_time is not None and self.next_event_time > until_time:
                    break
                self._step()
            if until_time is not None:
                self._reward_tracker.advance(until_time)
        finally:
            self._id_counters = self._read_id_counters()

    def _step(self) -> None:
        _, event = heapq.heappop(self._event_queue)
        self._set_time(event._time)

        # integrate the reward up to now before the event changes the state
        self._reward_tracker.advance(event.time)
        new_events = event.handle_event(self._scheduler, self._metric_store)

        if isinstance(event, RequestArrivalEvent):
            self._reward_tracker.on_request_arrival(event._request)
        elif isinstance(event, BatchEndEvent):
            self._reward_tracker.on_batch_end(event._batch)

        self._add_events(new_events)
        self._events_handled += 1

    def fork(self, share: Iterable[Any] = ()) -> "RLSimulator":
        """
        Deep copy of the simulation state. The execution time predictor and any
        objects in `share` (e.g. an agent's model) are shared, not copied. Call
        only while paused, never from inside an event handler.
        """
        memo = {id(obj): obj for obj in share}
        memo[id(self._execution_time_predictor)] = self._execution_time_predictor
        return copy.deepcopy(self, memo)


def _set_workload_seed(
    request_generator_config: BaseRequestGeneratorConfig, seed: int
) -> None:
    # arrival times use the generator seed; trace lengths are shuffled with the
    # length generator seed
    request_generator_config.seed = seed
    for name in ("length_generator_config", "interval_generator_config"):
        sub_config = getattr(request_generator_config, name, None)
        if sub_config is not None:
            sub_config.seed = seed


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def _summarize(episode: Episode, end_time: float) -> Dict[str, float]:
    requests = episode.requests
    num_decode_tokens = episode.num_decode_tokens
    completed = [request for request in requests if request.completed]
    samples = episode_metric_samples(episode)
    e2e, ttft, tbt = samples["e2e"], samples["ttft"], samples["tbt"]
    e2e_normalized = np.array(
        [(r.completed_at - r.arrived_at) / num_decode_tokens[r.id] for r in completed]
    )

    def mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else float("nan")

    return {
        "num_requests": len(requests),
        "num_completed": len(completed),
        "num_decisions": len(episode.transitions),
        "total_reward": float(episode.total_reward),
        "end_time": float(end_time),
        "mean_e2e": mean(e2e),
        "p50_e2e": _quantile(e2e, 0.5),
        "p95_e2e": _quantile(e2e, 0.95),
        "p99_e2e": _quantile(e2e, 0.99),
        "max_e2e": float(np.max(e2e)) if len(e2e) else float("nan"),
        "mean_e2e_normalized": mean(e2e_normalized),
        "p95_e2e_normalized": _quantile(e2e_normalized, 0.95),
        "p99_e2e_normalized": _quantile(e2e_normalized, 0.99),
        "mean_ttft": mean(ttft),
        "p95_ttft": _quantile(ttft, 0.95),
        "p99_ttft": _quantile(ttft, 0.99),
        "mean_tbt": mean(tbt),
        "p95_tbt": _quantile(tbt, 0.95),
        "p99_tbt": _quantile(tbt, 0.99),
        "output_tokens_per_s": (
            sum(num_decode_tokens[r.id] for r in completed) / end_time
            if end_time > 0
            else float("nan")
        ),
        "num_restarts": int(sum(r.num_restarts for r in requests)),
    }


class VidurSchedulingEnv:
    """
    Runs Vidur episodes with an agent ranking the waiting queue of a single
    RL_VLLM replica.

        env = VidurSchedulingEnv(make_simulation_config(qps=10))
        episode = env.run_episode(agent, workload_seed=0)
        episode.transitions  # obs, action, info, reward per decision
        episode.summary      # latency stats for the episode

    The workload (arrival times, request lengths) is fully determined by
    workload_seed, so several rollouts on the same seed differ only through the
    agent's own sampling.
    """

    def __init__(
        self,
        config: SimulationConfig,
        reward_config: Optional[RewardConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
        quiet: bool = True,
    ) -> None:
        cluster_config = config.cluster_config
        assert cluster_config.num_replicas == 1, "Only single-replica is supported"
        assert cluster_config.replica_config.num_pipeline_stages == 1
        assert (
            cluster_config.replica_scheduler_config.get_type()
            == ReplicaSchedulerType.RL_VLLM
        ), "The replica scheduler must be rl_vllm"

        self._config = config
        self._reward_config = reward_config or RewardConfig()
        self._observation_config = observation_config or ObservationConfig()

        if quiet:
            logging.getLogger("vidur").setLevel(logging.WARNING)

        # the expensive part of building a simulator; shared by every episode
        self._execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=cluster_config.replica_config,
            replica_scheduler_config=cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )

        # (baseline name, workload seed, workload config) -> episode
        self._reference_episodes: Dict[Tuple[str, Optional[int], str], Episode] = {}

        if self._reward_config.needs_calibration:
            self._reward_config = self.calibrate_thresholds(self._reward_config)

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def reward_config(self) -> RewardConfig:
        return self._reward_config

    @property
    def observation_config(self) -> ObservationConfig:
        return self._observation_config

    @property
    def execution_time_predictor(self) -> BaseExecutionTimePredictor:
        return self._execution_time_predictor

    def make_simulator(
        self,
        workload_seed: Optional[int] = None,
        request_generator_config: Optional[BaseRequestGeneratorConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> RLSimulator:
        """
        A fresh simulator for one episode, with the default FCFS policy installed.
        request_generator_config overrides the workload (e.g. a different qps or
        trace); workload_seed reseeds it.
        """
        request_generator_config = copy.deepcopy(
            request_generator_config or self._config.request_generator_config
        )
        if workload_seed is not None:
            _set_workload_seed(request_generator_config, workload_seed)

        # shallow copy: dataclasses.replace would rerun __post_init__, which
        # writes a config file
        config = copy.copy(self._config)
        config.request_generator_config = request_generator_config

        reward_config = reward_config or self._reward_config
        context = self.reward_context(
            reward_config, workload_seed, request_generator_config
        )

        _reset_id_counters()
        reward_tracker = RewardTracker(reward_config, context.request_thresholds)
        return RLSimulator(
            config, self._execution_time_predictor, reward_tracker, context
        )

    def reference_episode(
        self,
        baseline: str,
        workload_seed: Optional[int] = None,
        request_generator_config: Optional[BaseRequestGeneratorConfig] = None,
    ) -> Episode:
        """
        A baseline's episode on this workload (see vidur.rl.agents.baseline_agent),
        cached. Baselines do not depend on the reward, so the cached run can be
        rescored under any reward config with episode_cost().
        """
        key = (baseline, workload_seed, repr(request_generator_config))
        if key not in self._reference_episodes:
            # a plain reward config, so building it needs no reference runs
            self._reference_episodes[key] = self.run_episode(
                baseline_agent(baseline, seed=workload_seed),
                workload_seed,
                request_generator_config,
                reward_config=RewardConfig(),
            )
        return self._reference_episodes[key]

    def reward_context(
        self,
        reward_config: RewardConfig,
        workload_seed: Optional[int] = None,
        request_generator_config: Optional[BaseRequestGeneratorConfig] = None,
    ) -> RewardContext:
        """Per-request thresholds and reward scale for this workload."""
        context = RewardContext()
        if reward_config.window_thresholds:
            reference = self.reference_episode(
                reward_config.threshold_reference,
                workload_seed,
                request_generator_config,
            )
            context.request_thresholds = window_thresholds(reference, reward_config)

        if reward_config.normalize_by_baselines is not None:
            for baseline in reward_config.normalize_by_baselines:
                reference = self.reference_episode(
                    baseline, workload_seed, request_generator_config
                )
                context.reference_costs[baseline] = episode_cost(
                    reference, reward_config, context.request_thresholds
                )
            cost_a, cost_b = (list(context.reference_costs.values()) + [0.0])[:2]
            gap = max(
                abs(cost_a - cost_b),
                reward_config.normalization_min_fraction * abs(cost_a),
                1e-9,
            )
            context.scale = 1.0 / gap
        return context

    def run_episode(
        self,
        agent: BaseAgent,
        workload_seed: Optional[int] = None,
        request_generator_config: Optional[BaseRequestGeneratorConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> Episode:
        """
        Run one episode. request_generator_config overrides the workload (e.g. a
        different qps or trace); workload_seed reseeds it.
        """
        reward_config = reward_config or self._reward_config
        simulator = self.make_simulator(
            workload_seed, request_generator_config, reward_config
        )
        policy = AgentPriorityPolicy(
            agent,
            simulator.reward_tracker,
            reward_config,
            self._observation_config,
            event_index_fn=lambda: simulator.events_handled,
            reward_scale=simulator.reward_context.scale,
        )
        simulator.set_priority_policy(policy)

        # Request.restart() rewrites num_decode_tokens, so record the originals
        num_decode_tokens = {r.id: r.num_decode_tokens for r in simulator.requests}

        simulator.run()
        policy.finish()

        tracker = simulator.reward_tracker
        episode = Episode(
            transitions=policy.transitions,
            initial_reward_components=policy.initial_reward_components,
            requests=simulator.requests,
            num_decode_tokens=num_decode_tokens,
            token_times=np.array(tracker.token_times),
            token_request_ids=np.array(tracker.token_request_ids, dtype=np.int64),
            token_gaps=np.array(tracker.token_gaps),
            request_thresholds=simulator.reward_context.request_thresholds,
            reward_scale=simulator.reward_context.scale,
            reference_costs=dict(simulator.reward_context.reference_costs),
        )
        episode.cost = episode_cost(episode, reward_config)
        episode.summary = _summarize(episode, simulator.time)
        return episode

    def mean_gap_closed(
        self,
        episode: Episode,
        workload_seed: Optional[int] = None,
        request_generator_config: Optional[BaseRequestGeneratorConfig] = None,
        mean_term: str = "e2e_normalized",
        reference: Optional[str] = None,
    ) -> float:
        """
        Evaluation metric, not a reward: the fraction of the FCFS-to-SJF gain on
        one mean-latency component that the episode achieved,
            (M_fcfs - M) / (M_fcfs - M_sjf)
        0 = FCFS, 1 = oracle SJF, > 1 = better than SJF (possible: SJF is not
        optimal under continuous batching). The episode must be from the given
        workload. reference defaults to MEAN_COMPONENT_REFERENCES[mean_term].
        Tails are reported separately, against FCFS.
        """
        reference = reference or MEAN_COMPONENT_REFERENCES[mean_term]
        config = RewardConfig(weights={mean_term: 1.0})

        def mean_cost(e: Episode) -> float:
            return episode_cost_components(e, config, {})[mean_term]

        cost_fcfs, cost_sjf = (
            mean_cost(
                self.reference_episode(name, workload_seed, request_generator_config)
            )
            for name in ("fcfs", reference)
        )
        if cost_fcfs == cost_sjf:
            return float("nan")
        return (cost_fcfs - mean_cost(episode)) / (cost_fcfs - cost_sjf)

    def calibrate_thresholds(self, reward_config: RewardConfig) -> RewardConfig:
        """
        Fill the thresholds left as None with calibration_quantile of their
        metric over FCFS episodes on calibration_seeds. Fixed thresholds keep the
        reward stationary while the policy learns. In reference_window mode only
        tbt_slo is calibrated here; the rest are set per workload.
        """
        samples: Dict[str, List[np.ndarray]] = {"ttft": [], "e2e": [], "tbt": []}
        for seed in reward_config.calibration_seeds:
            episode = self.run_episode(
                FCFSAgent(), workload_seed=seed, reward_config=RewardConfig()
            )
            for metric, values in episode_metric_samples(episode).items():
                samples[metric].append(values)

        q = reward_config.calibration_quantile
        updates = {
            name: _quantile(np.concatenate(samples[metric]), q)
            for name, metric in THRESHOLD_METRICS.items()
            if getattr(reward_config, name) is None
            and name not in reward_config.window_thresholds
        }
        return dataclasses.replace(reward_config, **updates)


# trace names make_simulation_config accepts in place of a path
TRACE_FILES = {
    # Azure LLM inference trace 2023, conversation / code (via Splitwise);
    # these have arrival times, so they also work with arrivals="trace"
    "azure_conv": "./data/processed_traces/splitwise_conv.csv",
    "azure_code": "./data/processed_traces/splitwise_code.csv",
    # first-turn lengths, Llama-3 tokenizer (vidur/rl/prepare_traces.py)
    "sharegpt": "./data/processed_traces/sharegpt_llama3_tokenizer.csv",
    "lmsys": "./data/processed_traces/lmsys_llama3_tokenizer.csv",
    "arxiv": "./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv",
}


def make_request_generator_config(
    trace_file: str = "azure_conv",
    arrivals: str = "poisson",
    qps: Optional[float] = 10.0,
    num_requests: int = 300,
    max_tokens: int = 4096,
    seed: int = 42,
) -> BaseRequestGeneratorConfig:
    """
    The workload: request lengths from `trace_file` (a TRACE_FILES name or a
    path) and either
      arrivals="poisson": Poisson arrivals at qps, lengths shuffled by seed;
      arrivals="trace":   a seed-picked window of num_requests consecutive
                          requests replayed with their real arrival times,
                          time scaled so the whole trace averages qps
                          (None = native rate).
    """
    trace_file = TRACE_FILES.get(trace_file, trace_file)
    if arrivals == "trace":
        return TraceWindowRequestGeneratorConfig(
            seed=seed,
            trace_file=trace_file,
            num_requests=num_requests,
            qps=qps,
            max_tokens=max_tokens,
        )
    assert arrivals == "poisson", f"unknown arrivals {arrivals!r}"
    return SyntheticRequestGeneratorConfig(
        seed=seed,
        num_requests=num_requests,
        length_generator_config=TraceRequestLengthGeneratorConfig(
            seed=seed,
            trace_file=trace_file,
            max_tokens=max_tokens,
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(
            seed=seed,
            qps=qps,
        ),
    )


def make_simulation_config(
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    device: str = "a100",
    trace_file: str = "azure_conv",
    arrivals: str = "poisson",
    max_tokens: int = 4096,
    qps: Optional[float] = 10.0,
    num_requests: int = 300,
    batch_size_cap: int = 256,
    max_num_batched_tokens: int = 2048,
    num_blocks: Optional[int] = None,
    num_priority_levels: int = 16,
    tie_break: str = "arrival",
    decision_mode: str = "every_batch",
    rerank_interval_s: float = 0.1,
    seed: int = 42,
    output_dir: str = "simulator_output",
) -> SimulationConfig:
    """
    Single-replica rl_vllm config with the workload from
    make_request_generator_config (Poisson arrivals with trace request lengths
    by default), and metric collection turned off (the env computes its own
    stats).
    num_blocks=None sizes the KV cache from GPU memory; set it lower to force
    memory pressure and preemption.
    """
    return SimulationConfig(
        seed=seed,
        cluster_config=ClusterConfig(
            num_replicas=1,
            replica_config=ReplicaConfig(
                model_name=model_name,
                device=device,
                network_device=f"{device}_pairwise_nvlink",
            ),
            replica_scheduler_config=RlVllmSchedulerConfig(
                batch_size_cap=batch_size_cap,
                max_num_batched_tokens=max_num_batched_tokens,
                num_blocks=num_blocks,
                num_priority_levels=num_priority_levels,
                tie_break=tie_break,
                decision_mode=decision_mode,
                rerank_interval_s=rerank_interval_s,
            ),
        ),
        request_generator_config=make_request_generator_config(
            trace_file=trace_file,
            arrivals=arrivals,
            qps=qps,
            num_requests=num_requests,
            max_tokens=max_tokens,
            seed=seed,
        ),
        metrics_config=MetricsConfig(
            write_metrics=False,
            enable_chrome_trace=False,
            store_plots=False,
            output_dir=output_dir,
        ),
    )
