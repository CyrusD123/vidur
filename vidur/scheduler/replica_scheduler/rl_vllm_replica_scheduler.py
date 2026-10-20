from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.priority_policy import (
    BasePriorityPolicy,
    get_priority_policy,
)
from vidur.scheduler.replica_scheduler.vllm_replica_scheduler import (
    VLLMReplicaScheduler,
)

# Sliding windows (in scheduling iterations) for the system history the policy
# can observe, after Mutant's short/medium/long observation windows.
WINDOW_SIZES = {"short": 10, "medium": 100, "long": 1000}
WINDOW_METRICS = (
    "queue_length",
    "free_block_fraction",
    "batch_num_tokens",
    "batch_time",
)


@dataclass
class RequestHistory:
    """What happened to a request while it waited, relative to other requests."""

    # policy decisions made while the request was waiting
    num_decisions_seen: int = 0
    # iterations in which other waiting requests were admitted but it was not
    times_passed_over: int = 0
    # requests that arrived after it but were admitted while it waited
    num_overtaken_by: int = 0
    # position in the last effective ranking, 0 = front, 1 = back
    last_rank_percentile: Optional[float] = None
    # last priority the policy assigned; persists between decisions in
    # arrival_and_periodic mode
    priority: Optional[int] = None


class SlidingWindow:
    """Mean, min and max of the last `size` samples in O(1) amortized time."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._values: deque = deque()
        self._sum = 0.0
        self._count = 0
        # monotonic queues of (sample index, value)
        self._min_queue: deque = deque()
        self._max_queue: deque = deque()

    def add(self, value: float) -> None:
        index = self._count
        self._count += 1

        self._values.append(value)
        self._sum += value
        if len(self._values) > self._size:
            self._sum -= self._values.popleft()

        while self._min_queue and self._min_queue[-1][1] >= value:
            self._min_queue.pop()
        self._min_queue.append((index, value))
        while self._max_queue and self._max_queue[-1][1] <= value:
            self._max_queue.pop()
        self._max_queue.append((index, value))

        oldest = index - self._size + 1
        while self._min_queue[0][0] < oldest:
            self._min_queue.popleft()
        while self._max_queue[0][0] < oldest:
            self._max_queue.popleft()

    def stats(self) -> Tuple[float, float, float]:
        if not self._values:
            return 0.0, 0.0, 0.0
        return (
            self._sum / len(self._values),
            self._min_queue[0][1],
            self._max_queue[0][1],
        )


class RLVLLMReplicaScheduler(VLLMReplicaScheduler):
    """
    vLLM scheduler whose waiting queue is ranked by a priority policy before
    batches are formed. Admission (memory watermark, token budget, batch size
    cap, preemption) is inherited unchanged from VLLMReplicaScheduler, so the only
    difference from stock vLLM is the order in which waiting requests are tried.

    With the default FCFS policy this is identical to VLLMReplicaScheduler.

    Decision modes:
      every_batch           the policy ranks the whole waiting queue before every
                            batch (when there are at least 2 waiting requests)
      arrival_and_periodic  the policy ranks new requests at the first batch after
                            they arrive and the whole queue every
                            rerank_interval_s; priorities persist in between

    The scheduler also records each request's history (how often it was passed
    over or overtaken) and sliding-window statistics of the system, for policies
    to observe.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._num_priority_levels = self._config.num_priority_levels
        self._priority_policy: BasePriorityPolicy = get_priority_policy(
            self._config.priority_policy
        )

        # The scheduler is not told the time, but it is only invoked at request
        # arrivals and batch ends, so the latest of those is the current time.
        self._current_time = 0.0
        self._last_full_rerank_time: Optional[float] = None

        self._history: Dict[int, RequestHistory] = {}
        self._in_flight_batches: List[Batch] = []
        # which waiting requests (in queue order) the current decision applies to
        self._rank_mask: List[bool] = []

        self._windows = {
            (metric, window): SlidingWindow(size)
            for metric in WINDOW_METRICS
            for window, size in WINDOW_SIZES.items()
        }

    @property
    def num_priority_levels(self) -> int:
        return self._num_priority_levels

    @property
    def tie_break(self) -> str:
        return self._config.tie_break

    @property
    def decision_mode(self) -> str:
        return self._config.decision_mode

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def waiting_requests(self) -> List[Request]:
        return list(self._request_queue)

    @property
    def running_requests(self) -> List[Request]:
        """Requests holding KV cache: decoding, mid-prefill, or in flight."""
        in_flight = [
            request for batch in self._in_flight_batches for request in batch.requests
        ]
        return self._running + in_flight

    @property
    def rank_mask(self) -> List[bool]:
        """
        During get_priorities(): one flag per waiting request (same order) for
        whether this decision sets its priority. All True in every_batch mode and
        on full re-rankings; only new requests otherwise.
        """
        return self._rank_mask

    @property
    def num_running_requests(self) -> int:
        # requests holding KV cache: decoding, mid-prefill, or in an in-flight batch
        return len(self._allocation_map)

    @property
    def num_blocks(self) -> int:
        return self._config.num_blocks

    @property
    def num_free_blocks(self) -> int:
        return self._num_free_blocks

    @property
    def block_size(self) -> int:
        return self._config.block_size

    @property
    def batch_size_cap(self) -> int:
        return self._config.batch_size_cap

    @property
    def max_num_batched_tokens(self) -> int:
        return self._config.max_num_batched_tokens

    def num_allocated_blocks_of(self, request_id: int) -> int:
        return self._allocation_map.get(request_id, 0)

    def request_history(self, request_id: int) -> RequestHistory:
        return self._history[request_id]

    def window_stats(self, metric: str, window: str) -> Tuple[float, float, float]:
        """(mean, min, max) of `metric` over the last `window` iterations."""
        return self._windows[(metric, window)].stats()

    def set_priority_policy(self, priority_policy: BasePriorityPolicy) -> None:
        self._priority_policy = priority_policy

    def add_request(self, request: Request) -> None:
        self._current_time = max(self._current_time, request.arrived_at)
        self._history[request.id] = RequestHistory()
        super().add_request(request)

    def on_batch_end(self, batch: Batch) -> None:
        self._current_time = max(self._current_time, batch.completed_at)
        self._in_flight_batches.remove(batch)
        self._record("batch_time", batch.completed_at - batch.scheduled_at)

        super().on_batch_end(batch)

        for request in batch.requests:
            if request.completed:
                del self._history[request.id]

    def _record(self, metric: str, value: float) -> None:
        for window in WINDOW_SIZES:
            self._windows[(metric, window)].add(value)

    def _sort_key(self, priorities: Dict[int, int]):
        if self._config.tie_break == "arrival":
            return lambda r: (priorities[r.id], r.arrived_at, r.id)
        # sorted() is stable, so ties keep their current queue position
        return lambda r: priorities[r.id]

    def _get_rank_mask(self) -> List[bool]:
        """Which waiting requests the policy should rank now (empty: none)."""
        queue = self._request_queue

        if self._config.decision_mode == "every_batch":
            # nothing to reorder
            if len(queue) < 2:
                return []
            return [True] * len(queue)

        full_rerank_due = (
            self._last_full_rerank_time is None
            or self._current_time - self._last_full_rerank_time
            >= self._config.rerank_interval_s
        )
        if queue and full_rerank_due:
            self._last_full_rerank_time = self._current_time
            return [True] * len(queue)

        mask = [self._history[r.id].priority is None for r in queue]
        return mask if any(mask) else []

    def _rank_request_queue(self) -> bool:
        """Rank the waiting queue. Returns whether the policy was consulted."""
        self._rank_mask = self._get_rank_mask()
        consulted = bool(self._rank_mask)

        priorities = None
        if consulted:
            priorities = self._priority_policy.get_priorities(
                self.waiting_requests, self
            )

        if priorities is not None:
            assert len(priorities) == len(
                self._request_queue
            ), f"Expected {len(self._request_queue)} priorities, got {len(priorities)}"
            if self._priority_policy.unbounded_priorities:
                assert all(p >= 0 for p in priorities), priorities
            else:
                assert all(
                    0 <= p < self._num_priority_levels for p in priorities
                ), f"Priorities must be in [0, {self._num_priority_levels}), got {priorities}"

        if self._config.decision_mode == "every_batch":
            if priorities is None:
                # policy chose to keep the current order
                return consulted
            priority_map = {
                r.id: int(p) for r, p in zip(self._request_queue, priorities)
            }
        else:
            if priorities is not None:
                for request, priority, ranked in zip(
                    self._request_queue, priorities, self._rank_mask
                ):
                    if ranked:
                        self._history[request.id].priority = int(priority)
            # requests the policy never ranked (it returned None) go last
            priority_map = {
                r.id: (
                    self._history[r.id].priority
                    if self._history[r.id].priority is not None
                    else float("inf")
                )
                for r in self._request_queue
            }

        self._request_queue = sorted(
            self._request_queue, key=self._sort_key(priority_map)
        )
        return consulted

    def _update_history(
        self, ranked: List[Request], admitted: List[Request], consulted: bool
    ) -> None:
        admitted_ids = {r.id for r in admitted}
        admitted_arrivals = sorted(r.arrived_at for r in admitted)
        persistent = self._config.decision_mode == "arrival_and_periodic"

        for position, request in enumerate(ranked):
            history = self._history[request.id]
            if consulted:
                history.num_decisions_seen += 1
                # a ranking that admitted nothing is undone in every_batch mode
                if admitted or persistent:
                    history.last_rank_percentile = position / max(1, len(ranked) - 1)

            if request.id in admitted_ids or not admitted:
                continue
            history.times_passed_over += 1
            history.num_overtaken_by += sum(
                1 for arrival in admitted_arrivals if arrival > request.arrived_at
            )

    def _get_next_batch(self) -> Batch:
        queue_before_ranking = list(self._request_queue)
        consulted = self._rank_request_queue()
        ranked = list(self._request_queue)

        self._record("queue_length", len(ranked))
        self._record("free_block_fraction", self._num_free_blocks / self.num_blocks)

        batch = super()._get_next_batch()

        if batch:
            self._in_flight_batches.append(batch)
            self._record("batch_num_tokens", batch.total_num_tokens)

        remaining_ids = {request.id for request in self._request_queue}
        admitted_requests = [r for r in ranked if r.id not in remaining_ids]
        # the first request in ranked order that did not get in; admission is a
        # prefix of the ranking, so everything after it was not considered
        boundary_request = next((r for r in ranked if r.id in remaining_ids), None)

        self._update_history(ranked, admitted_requests, consulted)

        if not consulted:
            return batch

        if not admitted_requests and self._config.decision_mode == "every_batch":
            # Nothing joined the batch, so the ranking had no effect on it. Undo
            # the reorder so it cannot leak into later tie-breaks either. Requests
            # preempted during this iteration were pushed onto the front.
            original_ids = {request.id for request in queue_before_ranking}
            self._request_queue = [
                request
                for request in self._request_queue
                if request.id not in original_ids
            ] + queue_before_ranking

        self._priority_policy.on_batch_formed(admitted_requests, boundary_request)

        return batch
