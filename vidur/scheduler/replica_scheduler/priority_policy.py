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
    matching vLLM's `--scheduling-policy priority`. Ties keep the existing queue
    order (FCFS, with restarted requests at the front).
    """

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


class FCFSPriorityPolicy(BasePriorityPolicy):
    """Every request gets the same priority, so the stable sort keeps FCFS order."""

    def get_priorities(
        self,
        waiting_requests: List[Request],
        scheduler: "RLVLLMReplicaScheduler",
    ) -> Sequence[int]:
        return [0] * len(waiting_requests)


PRIORITY_POLICIES = {
    "fcfs": FCFSPriorityPolicy,
}


def get_priority_policy(name: str) -> BasePriorityPolicy:
    if name not in PRIORITY_POLICIES:
        raise ValueError(
            f"Unknown priority policy {name}. Valid policies: {list(PRIORITY_POLICIES)}"
        )
    return PRIORITY_POLICIES[name]()
