from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, List

import numpy as np

from vidur.entities import Request

if TYPE_CHECKING:
    from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
        RLVLLMReplicaScheduler,
    )

# Placeholder features, to be replaced once we design the model. Anything
# derived from num_decode_tokens (the true output length) must stay out: a real
# server does not know it.
REQUEST_FEATURES = (
    "num_prefill_tokens",
    "num_processed_tokens",
    "wait_time",
    "num_restarts",
    "num_blocks_needed",
    "queue_position",
)
GLOBAL_FEATURES = (
    "num_waiting",
    "num_running",
    "num_free_blocks",
    "free_block_fraction",
)


@dataclass
class Observation:
    # [num_waiting, len(REQUEST_FEATURES)], in current queue order
    request_features: np.ndarray
    # [len(GLOBAL_FEATURES)]
    global_features: np.ndarray
    num_priority_levels: int
    time: float
    request_ids: List[int]

    @property
    def num_requests(self) -> int:
        return len(self.request_ids)


def build_observation(
    waiting_requests: List[Request],
    scheduler: "RLVLLMReplicaScheduler",
    time: float,
) -> Observation:
    block_size = scheduler.block_size

    request_features = np.array(
        [
            (
                request.num_prefill_tokens,
                request.num_processed_tokens,
                time - request.arrived_at,
                request.num_restarts,
                ceil(request.num_prefill_tokens / block_size),
                position,
            )
            for position, request in enumerate(waiting_requests)
        ],
        dtype=np.float32,
    ).reshape(len(waiting_requests), len(REQUEST_FEATURES))

    global_features = np.array(
        (
            len(waiting_requests),
            scheduler.num_running_requests,
            scheduler.num_free_blocks,
            scheduler.num_free_blocks / scheduler.num_blocks,
        ),
        dtype=np.float32,
    )

    return Observation(
        request_features=request_features,
        global_features=global_features,
        num_priority_levels=scheduler.num_priority_levels,
        time=time,
        request_ids=[request.id for request in waiting_requests],
    )
