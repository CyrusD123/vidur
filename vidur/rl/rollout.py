"""
Rollout collection for training (docs/rl_plan.md §6 step 7).

Each worker builds its environments once (the execution time predictor is the
expensive part) and then runs (workload, seed, rollout) tasks with whatever
policy weights it is sent. Episodes come back as Trajectory objects: plain
numpy data plus the transitions, so they cross process boundaries cheaply.
"""

import dataclasses
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from vidur.rl.env import Transition, VidurSchedulingEnv, make_simulation_config
from vidur.rl.model import ModelConfig, SchedulerModel
from vidur.rl.observation import ObservationConfig
from vidur.rl.reward import (
    RewardConfig,
    episode_cost,
    per_request_rewards,
)
from vidur.rl.teacher import ImitationAgent, make_teacher
from vidur.rl.torch_agent import TorchAgent, mean_decision_time


@dataclass
class WorkloadConfig:
    """One kind of training/eval workload; kwargs go to make_simulation_config."""

    name: str
    # with arrivals="trace", the trace-average rate (None = native)
    qps: Optional[float] = 10.0
    num_blocks: Optional[int] = None
    num_requests: int = 300
    weight: float = 1.0
    # a TRACE_FILES name (vidur/rl/env.py) or a path
    trace: str = "azure_conv"
    # "poisson", or "trace" to replay seed-picked windows of real arrivals
    arrivals: str = "poisson"
    simulation: Dict[str, Any] = field(default_factory=dict)

    def make_env(
        self,
        reward_config: RewardConfig,
        observation_config: ObservationConfig,
        num_priority_levels: int,
        output_dir: str,
    ) -> VidurSchedulingEnv:
        config = make_simulation_config(
            qps=self.qps,
            trace_file=self.trace,
            arrivals=self.arrivals,
            num_blocks=self.num_blocks,
            num_requests=self.num_requests,
            num_priority_levels=num_priority_levels,
            output_dir=output_dir,
            **self.simulation,
        )
        return VidurSchedulingEnv(config, reward_config, observation_config)


@dataclass
class RolloutTask:
    workload: str
    workload_seed: int
    # index within the group of rollouts on this (workload, seed)
    rollout_index: int
    agent_seed: int
    # TorchAgent decoding: "sample" (training), "argmax" or "rank"
    decoding: str = "sample"
    # oracle teacher (vidur/rl/teacher.py): record its ranking at every
    # decision, and act with it with probability teacher_mix (DAgger)
    teacher: Optional[str] = None
    teacher_mix: float = 0.0


@dataclass
class Trajectory:
    task: RolloutTask
    transitions: List[Transition]
    # [num_transitions, num_requests] weighted, scaled per-request rewards
    # (columns: request_ids); excludes batch_time
    request_rewards: np.ndarray
    request_ids: np.ndarray
    # unscaled episode cost and FCFS's on the same workload, same reward
    cost: float
    fcfs_cost: float
    mean_gap_closed: float
    summary: Dict[str, float]
    decision_time_s: float

    @property
    def improvement_over_fcfs(self) -> float:
        return 1.0 - self.cost / self.fcfs_cost if self.fcfs_cost else float("nan")

    @property
    def total_reward(self) -> float:
        return float(sum(t.reward for t in self.transitions))


class RolloutRunner:
    """Holds one env per workload and runs tasks with a given model."""

    def __init__(
        self,
        workloads: Sequence[WorkloadConfig],
        reward_config: RewardConfig,
        observation_config: ObservationConfig,
        model_config: ModelConfig,
        output_dir: str,
    ) -> None:
        self._workloads = {w.name: w for w in workloads}
        self._reward_config = reward_config
        self._observation_config = observation_config
        self._output_dir = output_dir
        self._envs: Dict[str, VidurSchedulingEnv] = {}
        self.model = SchedulerModel(model_config)
        self.model.eval()

    def env(self, workload: str) -> VidurSchedulingEnv:
        if workload not in self._envs:
            self._envs[workload] = self._workloads[workload].make_env(
                self._reward_config,
                self._observation_config,
                self.model.config.num_priority_levels,
                self._output_dir,
            )
        return self._envs[workload]

    def run(self, task: RolloutTask) -> Trajectory:
        env = self.env(task.workload)
        student = TorchAgent(self.model, task.agent_seed, task.decoding)
        agent = student
        if task.teacher is not None:
            agent = ImitationAgent(
                make_teacher(task.teacher), student, task.teacher_mix, task.agent_seed
            )
        episode = env.run_episode(agent, workload_seed=task.workload_seed)

        fcfs = env.reference_episode("fcfs", task.workload_seed)
        return Trajectory(
            task=task,
            transitions=episode.transitions,
            request_rewards=per_request_rewards(episode, env.reward_config),
            request_ids=np.array([r.id for r in episode.requests], dtype=np.int64),
            cost=episode.cost,
            fcfs_cost=episode_cost(fcfs, env.reward_config, episode.request_thresholds),
            mean_gap_closed=env.mean_gap_closed(episode, task.workload_seed),
            summary=episode.summary,
            decision_time_s=mean_decision_time(student),
        )

    def run_all(
        self,
        state_dict: Optional[Dict[str, torch.Tensor]],
        tasks: Sequence[RolloutTask],
    ) -> List[Trajectory]:
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        return [self.run(task) for task in tasks]


# --- multiprocessing ---------------------------------------------------------

_WORKER: Optional[RolloutRunner] = None


def _init_worker(args: Tuple) -> None:
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = RolloutRunner(*args)


def _run_chunk(payload: Tuple[Dict[str, torch.Tensor], List[RolloutTask]]):
    state_dict, tasks = payload
    return _WORKER.run_all(state_dict, tasks)


class RolloutCollector:
    """Runs tasks in-process (num_workers=0) or on a pool of spawned workers."""

    def __init__(
        self,
        workloads: Sequence[WorkloadConfig],
        reward_config: RewardConfig,
        observation_config: ObservationConfig,
        model_config: ModelConfig,
        output_dir: str,
        num_workers: int = 0,
    ) -> None:
        args = (workloads, reward_config, observation_config, model_config, output_dir)
        self._num_workers = num_workers
        if num_workers > 0:
            # spawn: torch and fork do not mix; workers build their envs once
            context = mp.get_context("spawn")
            self._pool = context.Pool(num_workers, _init_worker, (args,))
            self._runner = None
        else:
            self._pool = None
            self._runner = RolloutRunner(*args)

    def collect(
        self, state_dict: Dict[str, torch.Tensor], tasks: Sequence[RolloutTask]
    ) -> List[Trajectory]:
        state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
        if self._pool is None:
            return self._runner.run_all(state_dict, tasks)
        # keep each (workload, seed) group on one worker, so its reference runs
        # (FCFS, thresholds) are computed once
        groups: Dict[Tuple[str, int], List[RolloutTask]] = {}
        for task in tasks:
            groups.setdefault((task.workload, task.workload_seed), []).append(task)
        chunks = [[] for _ in range(self._num_workers)]
        for i, group in enumerate(groups.values()):
            chunks[i % self._num_workers].extend(group)
        results = self._pool.map(_run_chunk, [(state_dict, c) for c in chunks if c])
        trajectories = [t for chunk in results for t in chunk]
        order = {dataclasses.astuple(t): i for i, t in enumerate(tasks)}
        return sorted(trajectories, key=lambda t: order[dataclasses.astuple(t.task)])

    def env(self, workload: str) -> VidurSchedulingEnv:
        """An in-process env (for evaluation baselines); built on demand."""
        if self._runner is None:
            raise RuntimeError("env() needs num_workers=0")
        return self._runner.env(workload)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()


def default_num_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)
