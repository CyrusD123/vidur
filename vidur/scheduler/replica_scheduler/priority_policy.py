from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence

from vidur.entities import Request

if TYPE_CHECKING:
    from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
        RLVLLMReplicaScheduler,
    )


class BasePriorityPolicy(ABC):
    """
    Assigns an integer priority to every request in a replica's waiting queue.

    Priorities are in [0, num_priority_levels); lower values are scheduled first,
    matching vLLM's `--scheduling-policy priority`. Ties are broken by arrival
    time, or by the existing queue order with tie_break="previous_order".

    In arrival_and_periodic mode only the requests flagged in
    `scheduler.rank_mask` take the returned priority; the rest keep the one they
    were given earlier.

    Policies with unbounded_priorities = True may return any non-negative int
    (e.g. a job size). Only oracle baselines should; learned policies are held
    to num_priority_levels.
    """

    unbounded_priorities: bool = False

    @abstractmethod
    def get_priorities(
        self,
        waiting_requests: List[Request],
        scheduler: "RLVLLMReplicaScheduler",
    ) -> Optional[Sequence[int]]:
        """
        Return one priority per request in `waiting_requests` (same order), or
        None to leave the queue order unchanged.
        """
        pass

    def on_batch_formed(
        self,
        admitted_requests: List[Request],
        boundary_request: Optional[Request],
    ) -> None:
        """
        Called after every get_priorities() call, once the batch is formed, with
        the waiting requests that were admitted and the first request in ranked
        order that was not (None if all were). Admission takes a prefix of the
        ranking, so the boundary request separates admitted from not admitted.

        In every_batch mode, if nothing was admitted the ranking had no effect and
        the scheduler has already restored the previous queue order.
        """
        pass


class FCFSPriorityPolicy(BasePriorityPolicy):
    """Every request gets the same priority, so the stable sort keeps FCFS order."""

    def get_priorities(
        self,
        waiting_requests: List[Request],
        scheduler: "RLVLLMReplicaScheduler",
    ) -> Sequence[int]:
        return [0] * len(waiting_requests)


JOB_SIZE_KEYS = ("total_tokens", "decode_tokens", "iterations")


def job_size(request: Request, key: str, max_num_batched_tokens: int) -> int:
    """
    Remaining work of a waiting request, using its true output length (an
    oracle; the RL observation never exposes it).

      total_tokens   prefill + decode tokens
      decode_tokens  output tokens still to generate
      iterations     scheduler iterations it needs on its own: chunked prefill
                     iterations plus one per output token

    A preempted request restarts with its generated tokens folded into the
    prompt, so these stay the remaining work after a restart.
    """
    if key == "total_tokens":
        return request.num_prefill_tokens + request.num_decode_tokens
    if key == "decode_tokens":
        return request.num_decode_tokens
    if key == "iterations":
        prefill_iterations = -(-request.num_prefill_tokens // max_num_batched_tokens)
        return prefill_iterations + request.num_decode_tokens
    raise ValueError(f"Unknown job size key {key}. Valid keys: {JOB_SIZE_KEYS}")


class SJFPriorityPolicy(BasePriorityPolicy):
    """
    Oracle shortest-job-first: the priority is the request's true job size, so
    the queue is sorted exactly by it (ties by arrival). Non-preemptive: it only
    orders admission, running requests are never displaced for shorter ones.
    """

    unbounded_priorities = True

    def __init__(self, key: str = "total_tokens") -> None:
        assert key in JOB_SIZE_KEYS, f"Unknown job size key {key}"
        self._key = key

    def get_priorities(
        self,
        waiting_requests: List[Request],
        scheduler: "RLVLLMReplicaScheduler",
    ) -> Sequence[int]:
        return [
            job_size(r, self._key, scheduler.max_num_batched_tokens)
            for r in waiting_requests
        ]


PRIORITY_POLICIES = {
    "fcfs": FCFSPriorityPolicy,
    "sjf": SJFPriorityPolicy,
}


def get_priority_policy(name: str) -> BasePriorityPolicy:
    if name not in PRIORITY_POLICIES:
        raise ValueError(
            f"Unknown priority policy {name}. Valid policies: {list(PRIORITY_POLICIES)}"
        )
    return PRIORITY_POLICIES[name]()
