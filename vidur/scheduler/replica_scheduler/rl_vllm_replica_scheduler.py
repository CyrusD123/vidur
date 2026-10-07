from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.priority_policy import (
    BasePriorityPolicy,
    get_priority_policy,
)
from vidur.scheduler.replica_scheduler.vllm_replica_scheduler import (
    VLLMReplicaScheduler,
)


class RLVLLMReplicaScheduler(VLLMReplicaScheduler):
    """
    vLLM scheduler whose waiting queue is re-ranked by a priority policy before
    every batch is formed. Admission (memory watermark, token budget, batch size
    cap, preemption) is inherited unchanged from VLLMReplicaScheduler, so the only
    difference from stock vLLM is the order in which waiting requests are tried.

    With the default FCFS policy this is identical to VLLMReplicaScheduler.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._num_priority_levels = self._config.num_priority_levels
        self._priority_policy: BasePriorityPolicy = get_priority_policy(
            self._config.priority_policy
        )

    @property
    def num_priority_levels(self) -> int:
        return self._num_priority_levels

    @property
    def waiting_requests(self) -> List[Request]:
        return list(self._request_queue)

    def set_priority_policy(self, priority_policy: BasePriorityPolicy) -> None:
        self._priority_policy = priority_policy

    def _rerank_request_queue(self) -> None:
        # nothing to reorder
        if len(self._request_queue) < 2:
            return

        priorities = self._priority_policy.get_priorities(self.waiting_requests, self)

        # policy chose to keep the current (FCFS) order
        if priorities is None:
            return

        assert len(priorities) == len(
            self._request_queue
        ), f"Expected {len(self._request_queue)} priorities, got {len(priorities)}"
        assert all(
            0 <= p < self._num_priority_levels for p in priorities
        ), f"Priorities must be in [0, {self._num_priority_levels}), got {priorities}"

        # sorted() is stable, so ties keep their current queue position
        order = sorted(range(len(priorities)), key=lambda i: priorities[i])
        self._request_queue = [self._request_queue[i] for i in order]

    def _get_next_batch(self) -> Batch:
        self._rerank_request_queue()
        return super()._get_next_batch()
