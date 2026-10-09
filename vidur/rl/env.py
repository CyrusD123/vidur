import atexit
import copy
import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
from vidur.rl.agents import BaseAgent
from vidur.rl.observation import Observation, build_observation
from vidur.rl.reward import RewardConfig, RewardTracker, zero_components
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
    summary: Dict[str, float] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return sum(transition.reward for transition in self.transitions)


class AgentPriorityPolicy(BasePriorityPolicy):
    """
    Bridges the scheduler and an agent: builds the observation, asks the agent
    for priorities, and records a transition whenever the ranking admitted at
    least one request. Rankings that admitted nothing had no effect (the
    scheduler undoes them), so they are dropped and their reward flows to the
    previous transition.
    """

    def __init__(
        self,
        agent: BaseAgent,
        reward_tracker: RewardTracker,
        reward_config: RewardConfig,
    ) -> None:
        self._agent = agent
        self._reward_tracker = reward_tracker
        self._reward_config = reward_config

        self.transitions: List[Transition] = []
        self.initial_reward_components = zero_components()
        self._pending = None

    def get_priorities(
        self,
        waiting_requests: List[Request],
        scheduler: RLVLLMReplicaScheduler,
    ) -> List[int]:
        obs = build_observation(waiting_requests, scheduler, self._reward_tracker.time)
        priorities, info = self._agent.act(obs)
        action = np.asarray(priorities, dtype=np.int64)
        self._pending = (obs, action, info)
        return action.tolist()

    def on_batch_formed(self, admitted_requests: List[Request]) -> None:
        obs, action, info = self._pending
        self._pending = None

        if not admitted_requests:
            return

        self._credit_reward()
        self.transitions.append(
            Transition(
                obs=obs,
                action=action,
                info=info,
                time=obs.time,
                num_admitted=len(admitted_requests),
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
        transition.reward = self._reward_config.combine(components)
        transition.duration = self._reward_tracker.time - transition.time


class RLSimulator(Simulator):
    """
    Simulator for running many episodes in one process: takes a prebuilt
    execution time predictor, skips writing output at exit, keeps the generated
    requests, and drives a RewardTracker from the event loop.
    """

    def __init__(
        self,
        config: SimulationConfig,
        execution_time_predictor: BaseExecutionTimePredictor,
        reward_tracker: RewardTracker,
    ) -> None:
        super().__init__(config, execution_time_predictor)
        atexit.unregister(self._write_output)
        self._reward_tracker = reward_tracker

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def time(self) -> float:
        return self._time

    @property
    def replica_schedulers(self) -> list:
        return [
            self._scheduler.get_replica_scheduler(replica_id)
            for replica_id in self._cluster.replicas
        ]

    def _init_event_queue(self) -> None:
        self._requests = self._request_generator.generate()

        for request in self._requests:
            self._add_event(RequestArrivalEvent(request.arrived_at, request))

    def run(self) -> None:
        while self._event_queue and not self._terminate:
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


def _reset_id_counters() -> None:
    # ids are class-level counters; reset them so episodes are reproducible
    for entity_class in [BaseEntity] + get_all_subclasses(BaseEntity):
        entity_class._id = -1
    BaseEvent._id = 0


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


def _summarize(episode: Episode, end_time: float) -> Dict[str, float]:
    requests = episode.requests
    num_decode_tokens = episode.num_decode_tokens
    completed = [request for request in requests if request.completed]
    e2e = np.array([r.completed_at - r.arrived_at for r in completed])
    ttft = np.array([r.prefill_completed_at - r.arrived_at for r in completed])
    e2e_normalized = np.array(
        [(r.completed_at - r.arrived_at) / num_decode_tokens[r.id] for r in completed]
    )

    def stat(values: np.ndarray, fn) -> float:
        return float(fn(values)) if len(values) else float("nan")

    return {
        "num_requests": len(requests),
        "num_completed": len(completed),
        "num_decisions": len(episode.transitions),
        "total_reward": float(episode.total_reward),
        "end_time": float(end_time),
        "mean_e2e": stat(e2e, np.mean),
        "p50_e2e": stat(e2e, np.median),
        "p99_e2e": stat(e2e, lambda v: np.quantile(v, 0.99)),
        "mean_e2e_normalized": stat(e2e_normalized, np.mean),
        "mean_ttft": stat(ttft, np.mean),
        "p99_ttft": stat(ttft, lambda v: np.quantile(v, 0.99)),
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

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def reward_config(self) -> RewardConfig:
        return self._reward_config

    @property
    def execution_time_predictor(self) -> BaseExecutionTimePredictor:
        return self._execution_time_predictor

    def run_episode(
        self,
        agent: BaseAgent,
        workload_seed: Optional[int] = None,
        request_generator_config: Optional[BaseRequestGeneratorConfig] = None,
    ) -> Episode:
        """
        Run one episode. request_generator_config overrides the workload (e.g. a
        different qps or trace); workload_seed reseeds it.
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

        _reset_id_counters()
        reward_tracker = RewardTracker(self._reward_config)
        simulator = RLSimulator(config, self._execution_time_predictor, reward_tracker)

        policy = AgentPriorityPolicy(agent, reward_tracker, self._reward_config)
        for replica_scheduler in simulator.replica_schedulers:
            replica_scheduler.set_priority_policy(policy)

        # Request.restart() rewrites num_decode_tokens, so record the originals
        num_decode_tokens = {r.id: r.num_decode_tokens for r in simulator.requests}

        simulator.run()
        policy.finish()

        episode = Episode(
            transitions=policy.transitions,
            initial_reward_components=policy.initial_reward_components,
            requests=simulator.requests,
            num_decode_tokens=num_decode_tokens,
        )
        episode.summary = _summarize(episode, simulator.time)
        return episode


def make_simulation_config(
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    device: str = "a100",
    trace_file: str = "./data/processed_traces/splitwise_conv.csv",
    max_tokens: int = 4096,
    qps: float = 10.0,
    num_requests: int = 300,
    batch_size_cap: int = 256,
    max_num_batched_tokens: int = 2048,
    num_blocks: Optional[int] = None,
    num_priority_levels: int = 8,
    seed: int = 42,
    output_dir: str = "simulator_output",
) -> SimulationConfig:
    """
    Single-replica rl_vllm config with Poisson arrivals and trace request
    lengths, and metric collection turned off (the env computes its own stats).
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
            ),
        ),
        request_generator_config=SyntheticRequestGeneratorConfig(
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
        ),
        metrics_config=MetricsConfig(
            write_metrics=False,
            enable_chrome_trace=False,
            store_plots=False,
            output_dir=output_dir,
        ),
    )
