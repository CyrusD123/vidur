"""
Policy/value network for the queue-ranking scheduler (docs/rl_plan.md §5).

Every waiting request, running request and the global/history features become
one token each; an encoder mixes them; the policy head turns each waiting token
into a distribution over priority levels, and the value head sums per-request
values plus a global value (the decomposed critic of §3).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vidur.rl.observation import Observation

ENCODERS = ("transformer", "deepsets", "mlp_pool")
HEADS = ("ordinal", "categorical")


@dataclass
class ModelConfig:
    num_waiting_features: int
    num_running_features: int
    num_global_features: int
    num_priority_levels: int = 16

    encoder: str = "transformer"
    head: str = "ordinal"
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    # share one encoder between policy and value. Off by default: the value
    # loss's gradients are larger and would dominate a shared encoder
    shared_trunk: bool = False
    # inputs beyond these are dropped: waiting requests by latest arrival (they
    # get the last priority level and no gradient), running requests from the end
    max_queue_len: int = 128
    max_running_len: int = 128
    # How concentrated the untrained policy is. Every request starts with this
    # probability on the middle level (its most likely level, so the untrained
    # deterministic policy is FCFS), the rest spread over nearby levels (ordinal)
    # or all other levels (categorical). High values start training near FCFS,
    # exploring small reorderings; 1 / num_priority_levels is roughly uniform.
    init_middle_level_prob: float = 0.75

    def __post_init__(self):
        assert self.encoder in ENCODERS, f"encoder must be one of {ENCODERS}"
        assert self.head in HEADS, f"head must be one of {HEADS}"
        assert self.num_priority_levels >= 2
        assert self.d_model % self.num_heads == 0
        assert 0 < self.init_middle_level_prob < 1


# ---------------------------------------------------------------------------
# Batching observations


@dataclass
class ObsBatch:
    """Padded tensors for a batch of observations. Waiting tokens follow obs
    order restricted to the modeled requests (see modeled_indices)."""

    waiting: torch.Tensor  # [B, Nw, Fw]
    waiting_mask: torch.Tensor  # [B, Nw] True = real token
    running: torch.Tensor  # [B, Nr, Fr]
    running_mask: torch.Tensor  # [B, Nr]
    global_features: torch.Tensor  # [B, Fg]

    def to(self, device) -> "ObsBatch":
        return ObsBatch(
            *(getattr(self, f).to(device) for f in self.__dataclass_fields__)
        )


def modeled_indices(obs: Observation, max_queue_len: int) -> np.ndarray:
    """Indices (obs order) of the waiting requests the model sees: all of them,
    or the max_queue_len earliest arrivals. The rest keep FCFS order behind."""
    n = obs.num_requests
    if n <= max_queue_len:
        return np.arange(n)
    earliest = np.lexsort((obs.request_ids, obs.arrival_times))[:max_queue_len]
    return np.sort(earliest)


def collate(
    observations: Sequence[Observation],
    config: ModelConfig,
    modeled: Optional[Sequence[np.ndarray]] = None,
    pad_to_multiple: int = 1,
) -> ObsBatch:
    """
    Pad a batch of observations. pad_to_multiple rounds the padded lengths up,
    so a GPU sees few distinct shapes (MPS rebuilds its graph for each one).
    """
    if modeled is None:
        modeled = [modeled_indices(o, config.max_queue_len) for o in observations]
    batch = len(observations)

    def padded(n: int) -> int:
        return -(-max(1, n) // pad_to_multiple) * pad_to_multiple

    nw = padded(max(len(m) for m in modeled))
    nr = padded(max(min(o.num_running, config.max_running_len) for o in observations))

    waiting = np.zeros((batch, nw, config.num_waiting_features), np.float32)
    waiting_mask = np.zeros((batch, nw), bool)
    running = np.zeros((batch, nr, config.num_running_features), np.float32)
    running_mask = np.zeros((batch, nr), bool)
    global_features = np.zeros((batch, config.num_global_features), np.float32)

    for b, (obs, idx) in enumerate(zip(observations, modeled)):
        waiting[b, : len(idx)] = obs.waiting_features[idx]
        waiting_mask[b, : len(idx)] = True
        k = min(obs.num_running, config.max_running_len)
        running[b, :k] = obs.running_features[:k]
        running_mask[b, :k] = True
        global_features[b] = obs.global_features

    return ObsBatch(
        torch.from_numpy(waiting),
        torch.from_numpy(waiting_mask),
        torch.from_numpy(running),
        torch.from_numpy(running_mask),
        torch.from_numpy(global_features),
    )


# ---------------------------------------------------------------------------
# Action distributions


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x < 0, stable."""
    return torch.where(
        x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x))
    )


class PriorityDistribution:
    """
    Independent per-request distributions over levels 0..L-1 for a padded batch
    [B, N]. Both heads support interval log-probabilities
    log P(low <= a <= high), which is what the coarsened PPO objective uses.
    """

    def __init__(
        self,
        num_levels: int,
        mask: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        cutpoints: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        self.num_levels = num_levels
        self.mask = mask
        self.scores = scores  # ordinal: [B, N]
        self.cutpoints = cutpoints  # ordinal: [L-1], increasing
        self.logits = logits  # categorical: [B, N, L]

    @property
    def is_ordinal(self) -> bool:
        return self.scores is not None

    def level_log_probs(self) -> torch.Tensor:
        """[B, N, L] log P(a = k)."""
        if not self.is_ordinal:
            return F.log_softmax(self.logits, dim=-1)
        levels = torch.arange(self.num_levels, device=self.scores.device)
        return self.interval_log_prob(
            levels.expand(*self.scores.shape, -1),
            levels.expand(*self.scores.shape, -1),
            scores=self.scores.unsqueeze(-1),
        )

    def interval_log_prob(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """log P(low <= a <= high), elementwise; low/high are int tensors."""
        top = self.num_levels - 1
        if not self.is_ordinal:
            levels = torch.arange(self.num_levels, device=self.logits.device)
            inside = (levels >= low.unsqueeze(-1)) & (levels <= high.unsqueeze(-1))
            log_probs = F.log_softmax(self.logits, dim=-1)
            return torch.logsumexp(log_probs.masked_fill(~inside, -torch.inf), dim=-1)

        s = self.scores if scores is None else scores
        theta = self.cutpoints
        # F(c) = P(a <= c) = sigmoid(theta_c - s) for c < L-1; F(L-1) = 1
        upper = theta[high.clamp(max=top - 1)] - s  # logit of F(high)
        lower = theta[(low - 1).clamp(min=0)] - s  # logit of F(low - 1)
        has_upper = high < top
        has_lower = low > 0

        log_f_high = F.logsigmoid(upper)
        log_survival_low = F.logsigmoid(-lower)
        # sigma(u) - sigma(l) = sigma(u) sigma(-l) (1 - e^(l - u))
        both = (
            log_f_high
            + log_survival_low
            + _log1mexp(torch.minimum(lower - upper, torch.full_like(upper, -1e-6)))
        )
        result = torch.where(
            has_upper & has_lower,
            both,
            torch.where(
                has_upper,
                log_f_high,
                torch.where(has_lower, log_survival_low, torch.zeros_like(upper)),
            ),
        )
        return result

    def entropy(self) -> torch.Tensor:
        """[B, N] entropy per request (0 on padding)."""
        log_p = self.level_log_probs()
        entropy = -(log_p.exp() * log_p.clamp(min=-1e9)).sum(-1)
        return entropy.masked_fill(~self.mask, 0.0)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """[B, N] sampled levels."""
        if self.is_ordinal:
            u = torch.rand(
                self.scores.shape, generator=generator, device=self.scores.device
            )
            cdf = torch.sigmoid(
                self.cutpoints - self.scores.unsqueeze(-1)
            )  # [B, N, L-1]
            return (cdf < u.unsqueeze(-1)).sum(-1)
        probs = F.softmax(self.logits, dim=-1)
        flat = probs.reshape(-1, self.num_levels)
        return torch.multinomial(flat, 1, generator=generator).reshape(probs.shape[:-1])

    def mode(self) -> torch.Tensor:
        return self.level_log_probs().argmax(-1)


# ---------------------------------------------------------------------------
# Encoders


def _mlp(d_in: int, d_out: int, d_hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )


class TokenEmbedding(nn.Module):
    """Per-type MLPs that map each input to a d_model token."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.waiting = _mlp(config.num_waiting_features, d, d)
        self.running = _mlp(config.num_running_features, d, d)
        self.global_ = _mlp(config.num_global_features, d, d)

    def forward(self, batch: ObsBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokens [B, 1 + Nw + Nr, d] (global first) and padding mask (True = real)."""
        tokens = torch.cat(
            [
                self.global_(batch.global_features).unsqueeze(1),
                self.waiting(batch.waiting),
                self.running(batch.running),
            ],
            dim=1,
        )
        ones = torch.ones_like(batch.waiting_mask[:, :1])
        mask = torch.cat([ones, batch.waiting_mask, batch.running_mask], dim=1)
        return tokens, mask


class TransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.num_heads,
            dim_feedforward=2 * config.d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(
            layer, config.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.norm(self.layers(tokens, src_key_padding_mask=~mask))


class DeepSetsEncoder(nn.Module):
    """Each token sees itself and the masked mean of all tokens (per layer)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.layers = nn.ModuleList(
            _mlp(2 * d, d, 2 * d) for _ in range(config.num_layers)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        for layer in self.layers:
            pooled = (tokens * weights).sum(1, keepdim=True) / weights.sum(
                1, keepdim=True
            )
            tokens = tokens + layer(torch.cat([tokens, pooled.expand_as(tokens)], -1))
        return self.norm(tokens)


class MLPPoolEncoder(nn.Module):
    """Baseline: tokens are processed independently plus one pooled context."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.context = _mlp(d, d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        pooled = (tokens * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True)
        return self.norm(tokens + self.context(pooled))


ENCODER_CLASSES = {
    "transformer": TransformerEncoder,
    "deepsets": DeepSetsEncoder,
    "mlp_pool": MLPPoolEncoder,
}


class Trunk(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(config)
        self.encoder = ENCODER_CLASSES[config.encoder](config)

    def forward(self, batch: ObsBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = self.embedding(batch)
        return self.encoder(tokens, mask), mask


# ---------------------------------------------------------------------------
# Policy/value network


@dataclass
class ModelOutput:
    dist: PriorityDistribution  # over modeled waiting requests [B, Nw]
    # None when called with with_value=False
    value: Optional[torch.Tensor] = None  # [B] V(s) = Σ v_j + v_global
    waiting_values: Optional[torch.Tensor] = None  # [B, Nw] v_j (0 on padding)
    running_values: Optional[torch.Tensor] = None  # [B, Nr]
    global_value: Optional[torch.Tensor] = None  # [B]


class SchedulerModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.policy_trunk = Trunk(config)
        self.value_trunk = None if config.shared_trunk else Trunk(config)

        num_levels = config.num_priority_levels
        if config.head == "ordinal":
            self.score_head = _mlp(d, 1, d)
            # cutpoints: first one free, then positive increments (softplus).
            # Evenly spaced by s with the middle level m between -s/2 and s/2,
            # so a score of 0 puts P(a = m) = tanh(s / 4) = init_middle_level_prob
            middle = (num_levels - 1) // 2
            spacing = 4 * math.atanh(config.init_middle_level_prob)
            self.cutpoint_start = nn.Parameter(torch.tensor((0.5 - middle) * spacing))
            self.cutpoint_increments = nn.Parameter(
                torch.full((num_levels - 2,), math.log(math.expm1(spacing - 1e-3)))
            )
        else:
            self.logit_head = _mlp(d, num_levels, d)
        self.request_value_head = _mlp(d, 1, d)
        self.global_value_head = _mlp(d, 1, d)
        # start every value at 0 (V sums one term per token, so random outputs
        # would add up to a large initial value error) and the policy near its
        # prior (scores ~0, logits ~0)
        last_layers = [self.request_value_head[-1], self.global_value_head[-1]]
        last_layers.append(
            self.score_head[-1] if config.head == "ordinal" else self.logit_head[-1]
        )
        for layer in last_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if config.head == "categorical":
            # softmax puts init_middle_level_prob on the middle level
            p = config.init_middle_level_prob
            with torch.no_grad():
                self.logit_head[-1].bias[(num_levels - 1) // 2] = math.log(
                    p * (num_levels - 1) / (1 - p)
                )

    def cutpoints(self) -> torch.Tensor:
        increments = F.softplus(self.cutpoint_increments) + 1e-3
        return torch.cat(
            [
                self.cutpoint_start.view(1),
                self.cutpoint_start + torch.cumsum(increments, 0),
            ]
        )

    def policy_parameters(self) -> List[nn.Parameter]:
        value_ids = {id(p) for p in self.value_parameters()}
        return [p for p in self.parameters() if id(p) not in value_ids]

    def value_parameters(self) -> List[nn.Parameter]:
        modules = [self.request_value_head, self.global_value_head]
        if self.value_trunk is not None:
            modules.append(self.value_trunk)
        return [p for module in modules for p in module.parameters()]

    def forward(self, batch: ObsBatch, with_value: bool = True) -> ModelOutput:
        nw = batch.waiting.shape[1]
        hidden, mask = self.policy_trunk(batch)
        waiting_hidden = hidden[:, 1 : 1 + nw]

        if self.config.head == "ordinal":
            dist = PriorityDistribution(
                self.config.num_priority_levels,
                batch.waiting_mask,
                scores=self.score_head(waiting_hidden).squeeze(-1),
                cutpoints=self.cutpoints(),
            )
        else:
            dist = PriorityDistribution(
                self.config.num_priority_levels,
                batch.waiting_mask,
                logits=self.logit_head(waiting_hidden),
            )

        if not with_value:
            return ModelOutput(dist=dist)

        value_hidden = (
            hidden if self.value_trunk is None else self.value_trunk(batch)[0]
        )
        request_values = self.request_value_head(value_hidden[:, 1:]).squeeze(-1)
        request_values = request_values.masked_fill(~mask[:, 1:], 0.0)
        global_value = self.global_value_head(value_hidden[:, 0]).squeeze(-1)
        return ModelOutput(
            dist=dist,
            value=request_values.sum(1) + global_value,
            waiting_values=request_values[:, :nw],
            running_values=request_values[:, nw:],
            global_value=global_value,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def config_for(observation_config, num_priority_levels: int, **kwargs) -> ModelConfig:
    """ModelConfig sized for an ObservationConfig."""
    return ModelConfig(
        num_waiting_features=observation_config.num_waiting_features,
        num_running_features=observation_config.num_running_features,
        num_global_features=observation_config.num_global_features,
        num_priority_levels=num_priority_levels,
        **kwargs,
    )


def full_actions(
    obs: Observation, modeled: np.ndarray, modeled_actions: np.ndarray
) -> np.ndarray:
    """Priorities for every waiting request: the model's for modeled ones, the
    last level for the rest (they then keep FCFS order behind)."""
    action = np.full(obs.num_requests, obs.num_priority_levels - 1, dtype=np.int64)
    action[modeled] = modeled_actions[: len(modeled)]
    return action


def interval_targets(
    transitions: Sequence, modeled: Sequence[np.ndarray], width: int
) -> Dict[str, torch.Tensor]:
    """Padded [B, width] action bounds and a mask of which modeled requests'
    log-probs enter the policy loss (ranked by this decision, real tokens)."""
    batch = len(transitions)
    low = np.zeros((batch, width), np.int64)
    high = np.zeros((batch, width), np.int64)
    action = np.zeros((batch, width), np.int64)
    loss_mask = np.zeros((batch, width), bool)
    for b, (t, idx) in enumerate(zip(transitions, modeled)):
        k = len(idx)
        low[b, :k] = t.action_low[idx]
        high[b, :k] = t.action_high[idx]
        action[b, :k] = t.action[idx]
        loss_mask[b, :k] = t.obs.rank_mask[idx] if len(t.obs.rank_mask) else True
    return {
        "low": torch.from_numpy(low),
        "high": torch.from_numpy(high),
        "action": torch.from_numpy(action),
        "loss_mask": torch.from_numpy(loss_mask),
    }
