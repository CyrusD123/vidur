from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from vidur.rl.observation import Observation


class BaseAgent(ABC):
    """
    What the trainer implements. act() returns one priority per waiting request
    (in obs order, ints in [0, obs.num_priority_levels), lower runs first) and an
    info dict that is stored on the transition untouched (e.g. log_prob, value).

    Use a private RNG (np.random.Generator / torch.Generator): Vidur reseeds the
    global `random` and `np.random` state when it generates each workload, so
    sampling from them would give identical rollouts for the same workload seed.
    """

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
