import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from vidur.rl.agents import BaseAgent
from vidur.rl.model import SchedulerModel, collate, full_actions, modeled_indices
from vidur.rl.observation import Observation

DECODINGS = ("sample", "argmax", "rank")


class TorchAgent(BaseAgent):
    """
    Samples priorities from a SchedulerModel, one observation at a time.

    decoding (docs/rl_plan.md §6 step 5):
    - "sample": each request's level is drawn independently from its
      distribution, with a private torch.Generator (Vidur reseeds the global
      RNGs per workload). Training rollouts use this.
    - "argmax": each request's most likely level. Coarse: scores that move
      within one level's interval change nothing.
    - "rank": sort requests by expected level E[a] (for the ordinal head, the
      same order as the scores; ties by arrival), then cut the ranking into L
      equal groups. Uses every score difference; the default for evaluation.

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
        decoding: str = "sample",
    ) -> None:
        assert decoding in DECODINGS, f"decoding must be one of {DECODINGS}"
        self._model = model
        self._decoding = decoding
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
        if self._decoding == "sample":
            levels = out.dist.sample(self._generator)[0].numpy()
        elif self._decoding == "argmax":
            levels = out.dist.mode()[0].numpy()
        else:
            levels = self._rank_levels(obs, modeled, out.dist)
        action = full_actions(obs, modeled, levels)
        elapsed = time.perf_counter() - start
        self.decision_times.append(elapsed)
        return action, {"decision_time_s": elapsed}

    @staticmethod
    def _rank_levels(obs: Observation, modeled: np.ndarray, dist) -> np.ndarray:
        num_levels = dist.num_levels
        probs = dist.level_log_probs()[0, : len(modeled)].exp()
        expected = (probs * torch.arange(num_levels, dtype=probs.dtype)).sum(-1)
        order = np.lexsort(
            (
                np.asarray(obs.request_ids)[modeled],
                np.asarray(obs.arrival_times)[modeled],
                expected.numpy(),
            )
        )
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(len(order))
        return ranks * num_levels // max(1, len(order))


def mean_decision_time(agent: TorchAgent) -> float:
    return (
        float(np.mean(agent.decision_times)) if agent.decision_times else float("nan")
    )
