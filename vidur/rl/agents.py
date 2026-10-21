import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vidur.entities import Request
from vidur.rl.observation import Observation
from vidur.scheduler.replica_scheduler.priority_policy import JOB_SIZE_KEYS, job_size

if TYPE_CHECKING:
    from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
        RLVLLMReplicaScheduler,
    )


class BaseAgent(ABC):
    """
    What the trainer implements. act() returns one priority per waiting request
    (in obs order, ints in [0, obs.num_priority_levels), lower runs first) and an
    info dict that is stored on the transition untouched (e.g. log_prob, value).

    Use a private RNG (np.random.Generator / torch.Generator): Vidur reseeds the
    global `random` and `np.random` state when it generates each workload, so
    sampling from them would give identical rollouts for the same workload seed.
    """

    # may return priorities >= num_priority_levels (oracle baselines only)
    unbounded_priorities: bool = False

    @abstractmethod
    def act(self, obs: Observation) -> Tuple[Sequence[int], Dict[str, Any]]:
        pass


class FCFSAgent(BaseAgent):
    def act(self, obs: Observation) -> Tuple[Sequence[int], Dict[str, Any]]:
        return np.zeros(obs.num_requests, dtype=np.int64), {}


class RandomAgent(BaseAgent):
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: Observation) -> Tuple[Sequence[int], Dict[str, Any]]:
        priorities = self._rng.integers(0, obs.num_priority_levels, obs.num_requests)
        return priorities, {}


class OracleAgent(BaseAgent):
    """
    A baseline that also sees the true waiting Request objects (in obs order),
    including their output lengths, and the scheduler. Never use one as a
    learned policy.
    """

    @abstractmethod
    def act_oracle(
        self,
        obs: Observation,
        waiting_requests: List[Request],
        scheduler: "RLVLLMReplicaScheduler",
    ) -> Tuple[Sequence[int], Dict[str, Any]]:
        pass

    def act(self, obs: Observation) -> Tuple[Sequence[int], Dict[str, Any]]:
        raise TypeError(f"{type(self).__name__} needs the requests: use act_oracle")


class OracleSJFAgent(OracleAgent):
    """
    Shortest job first using true job sizes (see job_size for the keys).

    aging > 0 subtracts aging * (seconds waited) from each job size, so long
    requests eventually move ahead of new short ones (aging is in job-size units
    per second, e.g. tokens/s).

    num_levels=None sorts the queue exactly (priority = rank). With
    num_levels=L the sorted queue is cut into L equal-size groups, which is a
    ranking the RL agent could itself express with L priority levels.

    key="prefill_tokens" ranks by prompt length only, which needs no oracle.
    """

    def __init__(
        self,
        key: str = "total_tokens",
        num_levels: Optional[int] = None,
        aging: float = 0.0,
    ):
        assert key in JOB_SIZE_KEYS, f"Unknown job size key {key}"
        assert aging >= 0
        self._key = key
        self._num_levels = num_levels
        self._aging = aging
        self.unbounded_priorities = num_levels is None

    def ranks(
        self,
        obs: Observation,
        waiting_requests: List[Request],
        scheduler: "RLVLLMReplicaScheduler",
    ) -> np.ndarray:
        """[num_requests] position of each request (obs order) in the exact
        SJF order, 0 = first."""
        sizes = np.array(
            [
                job_size(r, self._key, scheduler.max_num_batched_tokens)
                for r in waiting_requests
            ],
            dtype=float,
        )
        sizes -= self._aging * (obs.time - obs.arrival_times)

        # same order as the scheduler's (priority, arrival, id) sort
        order = np.lexsort((obs.request_ids, obs.arrival_times, sizes))
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(len(order))
        return ranks

    def act_oracle(
        self,
        obs: Observation,
        waiting_requests: List[Request],
        scheduler: "RLVLLMReplicaScheduler",
    ) -> Tuple[Sequence[int], Dict[str, Any]]:
        ranks = self.ranks(obs, waiting_requests, scheduler)
        if self._num_levels is None:
            return ranks, {}
        return bucket_ranks(ranks, min(self._num_levels, obs.num_priority_levels)), {}


def bucket_ranks(ranks: np.ndarray, num_levels: int) -> np.ndarray:
    """Cut a ranking (0 = first) into num_levels equal-size groups."""
    return ranks * num_levels // max(1, len(ranks))


_SJF_NAME = re.compile(
    r"^sjf_(?P<key>[a-z_]+?)(?:_age(?P<aging>[0-9.]+))?(?:_(?P<levels>[0-9]+)lvl)?$"
)


def baseline_agent(name: str, seed: Optional[int] = None) -> BaseAgent:
    """
    A baseline by name:
      fcfs                             stock vLLM order
      random                           uniformly random priorities
      sjf_<key>[_age<rate>][_<L>lvl]   OracleSJFAgent(key, num_levels=L, aging=rate),
                                       e.g. sjf_decode_tokens, sjf_total_tokens_16lvl,
                                       sjf_decode_tokens_age20
    """
    if name == "fcfs":
        return FCFSAgent()
    if name == "random":
        return RandomAgent(seed)
    match = _SJF_NAME.match(name)
    if match and match["key"] in JOB_SIZE_KEYS:
        return OracleSJFAgent(
            match["key"],
            num_levels=int(match["levels"]) if match["levels"] else None,
            aging=float(match["aging"]) if match["aging"] else 0.0,
        )
    raise ValueError(f"Unknown baseline {name}. {baseline_agent.__doc__}")
