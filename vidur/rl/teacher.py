"""
Imitation targets from an oracle teacher (docs/rl_plan.md §8).

The teacher is an OracleSJFAgent (e.g. SJF on total tokens with aging). It reads
the true output lengths, which the student's observation never contains, so the
student has to infer the teacher's ranking from observable features (prompt
length, wait time, ...). That is only possible in simulation, where the oracle
exists; the student itself stays deployable.

ImitationAgent runs inside the env like any agent. At every decision it records
the teacher's ranking in the transition's info, and acts with the teacher
(probability beta) or the student (DAgger's mixture policy). With beta = 0 it
just labels the student's own rollouts, which PPO fine-tuning and evaluation
use for an imitation anchor and agreement metrics.

The imitation loss on the student's distribution has two terms:
- level: -log P(a_i = teacher level_i), the teacher's ranking cut into L equal
  groups. Trains the scores and the cutpoints, so the sampled policy (what PPO
  explores with) follows the teacher.
- rank: pairwise logistic loss on the scores, softplus(s_i - s_j) for every pair
  the teacher puts i before j (lower score runs first). Trains the order that
  "rank" decoding deploys, including within a level.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vidur.entities import Request
from vidur.rl.agents import (
    BaseAgent,
    OracleAgent,
    OracleSJFAgent,
    baseline_agent,
    bucket_ranks,
)
from vidur.rl.model import PriorityDistribution
from vidur.rl.observation import Observation

TEACHER_RANK = "teacher_rank"
TEACHER_LEVEL = "teacher_level"
ACTED_BY_TEACHER = "acted_by_teacher"


def make_teacher(name: str) -> OracleSJFAgent:
    """An SJF baseline by name (see baseline_agent), e.g. sjf_total_tokens_age800.
    Any _<L>lvl suffix is ignored: labels are cut into the env's levels."""
    teacher = baseline_agent(name)
    assert isinstance(teacher, OracleSJFAgent), f"{name} is not an SJF baseline"
    return teacher


class ImitationAgent(OracleAgent):
    """
    DAgger mixture policy: at each decision the teacher's ranking is stored in
    info (teacher_rank, teacher_level), and the action is the teacher's
    bucketed ranking with probability beta, else the student's.
    """

    def __init__(
        self,
        teacher: OracleSJFAgent,
        student: Optional[BaseAgent],
        beta: float,
        seed: Optional[int] = None,
    ) -> None:
        assert 0.0 <= beta <= 1.0
        assert student is not None or beta == 1.0, "beta < 1 needs a student"
        self._teacher = teacher
        self._student = student
        self._beta = beta
        self._rng = np.random.default_rng(seed)

    @property
    def student(self) -> Optional[BaseAgent]:
        return self._student

    def act_oracle(
        self,
        obs: Observation,
        waiting_requests: List[Request],
        scheduler,
    ) -> Tuple[Sequence[int], Dict[str, Any]]:
        ranks = self._teacher.ranks(obs, waiting_requests, scheduler)
        levels = bucket_ranks(ranks, obs.num_priority_levels)
        use_teacher = self._beta == 1.0 or self._rng.random() < self._beta
        if use_teacher:
            action, info = levels, {}
        else:
            action, info = self._student.act(obs)
            info = dict(info)
        info.update(
            {TEACHER_RANK: ranks, TEACHER_LEVEL: levels, ACTED_BY_TEACHER: use_teacher}
        )
        return action, info


def has_labels(transition) -> bool:
    return TEACHER_RANK in transition.info


# --- loss ---------------------------------------------------------------------


def imitation_targets(
    transitions: Sequence, modeled: Sequence[np.ndarray], width: int
) -> Dict[str, torch.Tensor]:
    """Padded [B, width] teacher ranks and levels of the modeled requests, and
    which rows have labels."""
    batch = len(transitions)
    rank = np.zeros((batch, width), np.int64)
    level = np.zeros((batch, width), np.int64)
    labeled = np.zeros(batch, bool)
    for b, (t, idx) in enumerate(zip(transitions, modeled)):
        if not has_labels(t):
            continue
        labeled[b] = True
        rank[b, : len(idx)] = t.info[TEACHER_RANK][idx]
        level[b, : len(idx)] = t.info[TEACHER_LEVEL][idx]
    return {
        "teacher_rank": torch.from_numpy(rank),
        "teacher_level": torch.from_numpy(level),
        "labeled": torch.from_numpy(labeled),
    }


def ranking_scores(dist: PriorityDistribution) -> torch.Tensor:
    """[B, N] the score rank decoding sorts by: the ordinal score, or the
    expected level for the categorical head (same order for the ordinal one)."""
    if dist.is_ordinal:
        return dist.scores
    probs = dist.level_log_probs().exp()
    levels = torch.arange(dist.num_levels, device=probs.device, dtype=probs.dtype)
    return (probs * levels).sum(-1)


def imitation_losses(
    dist: PriorityDistribution,
    targets: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Per-sample [B] losses, each averaged over that sample's requests (level) or
    teacher-ordered pairs (rank), so long queues do not dominate. mask [B, N]
    selects the requests that count (real, ranked by this decision).
    """
    level = targets["teacher_level"]
    level_nll = -dist.interval_log_prob(level, level)
    count = mask.sum(1).clamp(min=1)
    level_loss = (level_nll * mask).sum(1) / count

    scores = ranking_scores(dist)
    rank = targets["teacher_rank"]
    # pairs (i, j) with i ahead of j for the teacher: want s_i < s_j
    before = (rank.unsqueeze(2) < rank.unsqueeze(1)) & (
        mask.unsqueeze(2) & mask.unsqueeze(1)
    )
    pair_loss = F.softplus(scores.unsqueeze(2) - scores.unsqueeze(1))
    num_pairs = before.sum((1, 2))
    rank_loss = (pair_loss * before).sum((1, 2)) / num_pairs.clamp(min=1)
    with torch.no_grad():
        correct = ((scores.unsqueeze(2) < scores.unsqueeze(1)) & before).sum((1, 2))
        pair_accuracy = correct / num_pairs.clamp(min=1)
    return {
        "level_loss": level_loss,
        "rank_loss": rank_loss,
        "pair_accuracy": pair_accuracy,
        "num_pairs": num_pairs,
    }


# --- agreement metrics on rollouts --------------------------------------------


def agreement_metrics(transitions: Sequence) -> Dict[str, float]:
    """
    How closely the actions taken match the teacher, over labeled decisions
    that had an admission boundary (where the ranking mattered):
    - pair_agreement: fraction of request pairs the action ordered the same
      way as the teacher (the action's order: level, then arrival)
    - admitted_overlap: |admitted ∩ teacher's first k| / k, k = number admitted
    - head_agreement: the action's first request is the teacher's first
    """
    pairs, overlap, head = [], [], []
    for t in transitions:
        if not has_labels(t) or t.boundary_index is None:
            continue
        teacher_rank = t.info[TEACHER_RANK]
        keys = t.obs.ranking_keys(t.action)
        order = sorted(range(len(keys)), key=keys.__getitem__)
        action_rank = np.empty(len(order), np.int64)
        action_rank[order] = np.arange(len(order))

        n = len(teacher_rank)
        if n >= 2:
            a = np.sign(action_rank[:, None] - action_rank[None, :])
            b = np.sign(teacher_rank[:, None] - teacher_rank[None, :])
            upper = np.triu_indices(n, 1)
            pairs.append(float(np.mean(a[upper] == b[upper])))
        k = int(t.admitted.sum())
        if k:
            overlap.append(float(np.mean(teacher_rank[t.admitted] < k)))
        head.append(float(order[0] == int(np.argmin(teacher_rank))))

    def mean(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "pair_agreement": mean(pairs),
        "admitted_overlap": mean(overlap),
        "head_agreement": mean(head),
    }
