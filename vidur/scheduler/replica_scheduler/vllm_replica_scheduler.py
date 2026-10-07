from bisect import insort
from itertools import count
from math import ceil
from typing import Dict, List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class VLLMReplicaScheduler(BaseReplicaScheduler):
    """
    Models the vLLM V1 scheduler (vllm/v1/core/sched/scheduler.py): a single
    token budget per iteration with chunked prefill, running requests scheduled
    before waiting ones, and preemption by recomputation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # running requests in admission order; requests that are part of an
        # in-flight batch are temporarily absent from this list
        self._running: List[Request] = []
        self._admission_counter = count()
        self._admission_order: Dict[int, int] = {}
        self._num_running_batches = 0
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    @property
    def _num_free_blocks(self) -> int:
        return self._config.num_blocks - self._num_allocated_blocks

    def _num_required_blocks(self, request: Request, num_tokens: int) -> int:
        num_blocks = ceil(num_tokens / self._config.block_size)
        return max(0, num_blocks - self._allocation_map.get(request.id, 0))

    def _get_num_new_tokens(
        self, request: Request, token_budget: int, num_eligible_requests: int
    ) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        num_new_tokens = request.num_prefill_tokens - request.num_processed_tokens

        # vLLM skips the cap when there is no other request to starve
        threshold = self._config.long_prefill_token_threshold
        if num_eligible_requests > 1 and 0 < threshold < num_new_tokens:
            num_new_tokens = threshold

        return min(num_new_tokens, token_budget)

    def _get_kv_length(self, request: Request, num_new_tokens: int) -> int:
        # once prefill completes, num_processed_tokens already counts the
        # sampled token whose KV is written by the next decode step
        if request.is_prefill_complete:
            return request.num_processed_tokens
        return request.num_processed_tokens + num_new_tokens

    def _preempt(self, request: Request) -> None:
        self.free(request.id)
        del self._admission_order[request.id]
        request.restart()
        self._request_queue.insert(0, request)

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
                del self._admission_order[request.id]
            else:
                insort(
                    self._running,
                    request,
                    key=lambda r: self._admission_order[r.id],
                )

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        token_budget = self._config.max_num_batched_tokens
        preempted = False
        num_eligible_requests = len(self._allocation_map) + len(self._request_queue)

        # first, schedule the running requests (decodes and partial prefills)
        index = 0
        while index < len(self._running) and token_budget > 0:
            request = self._running[index]

            next_num_tokens = self._get_num_new_tokens(
                request, token_budget, num_eligible_requests
            )
            num_required_blocks = self._num_required_blocks(
                request, self._get_kv_length(request, next_num_tokens)
            )

            # preempt the most recently admitted request until this one fits
            while self._num_free_blocks < num_required_blocks:
                victim_request = self._running.pop()
                self._preempt(victim_request)
                preempted = True
                if victim_request is request:
                    break
            else:
                self.allocate(request.id, num_required_blocks)
                requests.append(request)
                num_tokens.append(next_num_tokens)
                token_budget -= next_num_tokens
                index += 1
                continue

            break

        # scheduled requests are in flight until their batch ends
        del self._running[:index]

        # next, admit waiting requests, unless this iteration had to preempt
        while not preempted and self._request_queue and token_budget > 0:
            if len(self._allocation_map) >= self._config.batch_size_cap:
                break

            request = self._request_queue[0]

            # the full sequence must fit, but only the first chunk is allocated
            watermark_blocks = self._watermark_blocks if self._allocation_map else 0
            num_full_sequence_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            if self._num_free_blocks - num_full_sequence_blocks < watermark_blocks:
                break

            next_num_tokens = self._get_num_new_tokens(
                request, token_budget, num_eligible_requests
            )

            self._request_queue.pop(0)
            self.allocate(
                request.id, self._num_required_blocks(request, next_num_tokens)
            )
            self._admission_order[request.id] = next(self._admission_counter)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            token_budget -= next_num_tokens

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)
