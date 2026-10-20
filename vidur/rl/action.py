from typing import Optional, Sequence, Tuple

import numpy as np


def admission_action_bounds(
    action: np.ndarray,
    arrival_keys: Sequence[Tuple[float, int]],
    admitted: np.ndarray,
    boundary_index: Optional[int],
    num_priority_levels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The part of a ranking that changed which requests were admitted, as inclusive
    per-request bounds [low, high] on the priority.

    The queue is sorted by (priority, arrival key) and admission takes a prefix,
    so given the boundary request b (the first one not admitted, priority c),
    only the side of b each request landed on mattered:
      admitted i:          key_i < key_b, i.e. a_i <= c (or c - 1 if it arrived
                           after b)
      not admitted i != b: key_i > key_b, i.e. a_i >= c (or c + 1 if it arrived
                           before b)
      b itself:            a_b == c

    The training log-probability of the action is then sum_i log P(low_i <= a_i <=
    high_i). If everything was admitted (no boundary), no priority mattered and
    every bound is the full range.

    Only valid with tie_break="arrival" in every_batch mode: with persistent
    priorities or previous-order tie-breaks the exact levels affect later batches.
    """
    num_requests = len(action)
    low = np.zeros(num_requests, dtype=np.int64)
    high = np.full(num_requests, num_priority_levels - 1, dtype=np.int64)

    if boundary_index is None:
        return low, high

    c = int(action[boundary_index])
    boundary_key = arrival_keys[boundary_index]

    for i in range(num_requests):
        if i == boundary_index:
            low[i] = high[i] = c
        elif admitted[i]:
            high[i] = c if arrival_keys[i] < boundary_key else c - 1
        else:
            low[i] = c if arrival_keys[i] > boundary_key else c + 1

    return low, high
