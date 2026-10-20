import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from vidur.rl.agents import BaseAgent
from vidur.rl.model import SchedulerModel, collate, full_actions, modeled_indices
from vidur.rl.observation import Observation


class TorchAgent(BaseAgent):
    """
    Samples priorities from a SchedulerModel, one observation at a time.

    Sampling uses a private torch.Generator (Vidur reseeds the global RNGs per
    workload). deterministic=True takes each request's most likely level.

    Nothing about the action distribution is stored: PPO recomputes the
    rollout log-probabilities from the same weights before its first update,
    since the coarsened log-prob (docs/rl_plan.md §3) depends on which requests
    were admitted, which is only known after the batch forms.
    info = {"decision_time_s": wall time of the forward pass}; the value head is
    skipped while acting.
    """

    def __init__(
        self,
        model: SchedulerModel,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        self._model = model
        self._deterministic = deterministic
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        else:
            self._generator.seed()
        self.decision_times = []

    @property
    def model(self) -> SchedulerModel:
        return self._model

    @torch.no_grad()
    def act(self, obs: Observation) -> Tuple[Sequence[int], Dict[str, Any]]:
        start = time.perf_counter()
        modeled = modeled_indices(obs, self._model.config.max_queue_len)
        batch = collate([obs], self._model.config, [modeled])
        out = self._model(batch, with_value=False)
        levels = (
            out.dist.mode() if self._deterministic else out.dist.sample(self._generator)
        )
        action = full_actions(obs, modeled, levels[0].numpy())
        elapsed = time.perf_counter() - start
        self.decision_times.append(elapsed)
        return action, {"decision_time_s": elapsed}


def mean_decision_time(agent: TorchAgent) -> float:
    return (
        float(np.mean(agent.decision_times)) if agent.decision_times else float("nan")
    )
