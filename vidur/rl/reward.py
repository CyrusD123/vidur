from dataclasses import dataclass, field
from typing import Dict

from vidur.entities import Batch, Request

# Each component is a time integral, so summed over an episode it equals a
# per-request metric summed over all requests:
#   e2e             ∫ (#requests in system) dt                 = Σ e2e latency
#   e2e_normalized  ∫ Σ_in-system 1/num_decode_tokens dt       = Σ e2e / output length
#   ttft            ∫ (#requests without a first token) dt     = Σ TTFT
#   e2e_squared     ∫ Σ_in-system 2 * age dt                   = Σ e2e²  (penalizes tails)
#   tbt_violation   Σ_tokens max(0, gap - tbt_slo)             (counted when tokens are produced)
REWARD_COMPONENTS = (
    "e2e",
    "e2e_normalized",
    "ttft",
    "e2e_squared",
    "tbt_violation",
)


@dataclass
class RewardConfig:
    """reward = -Σ weight * component. Components are always recorded per
    transition, so they can be reweighted later without re-simulating."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "e2e": 0.0,
            "e2e_normalized": 1.0,
            "ttft": 0.0,
            "e2e_squared": 0.0,
            "tbt_violation": 0.0,
        }
    )
    # placeholder target gap between tokens, only used by tbt_violation
    tbt_slo: float = 0.1

    def __post_init__(self):
        unknown = set(self.weights) - set(REWARD_COMPONENTS)
        assert not unknown, f"Unknown reward components {unknown}"

    def combine(self, components: Dict[str, float]) -> float:
        return -sum(
            self.weights.get(name, 0.0) * value for name, value in components.items()
        )


def zero_components() -> Dict[str, float]:
    return {name: 0.0 for name in REWARD_COMPONENTS}


class RewardTracker:
    """
    Accumulates the reward components as the simulation advances. The event loop
    calls advance(t) before handling each event and on_request_arrival /
    on_batch_end after it; pop() returns what accumulated since the last pop().
    """

    def __init__(self, reward_config: RewardConfig) -> None:
        self._tbt_slo = reward_config.tbt_slo

        self._time = 0.0
        self._num_in_system = 0
        self._num_without_first_token = 0
        # Σ 1/num_decode_tokens and Σ arrived_at over requests in the system
        self._normalized_weight_sum = 0.0
        self._arrival_time_sum = 0.0

        # output length is captured at arrival: Request.restart() rewrites
        # num_decode_tokens to the number of tokens left
        self._normalized_weights: Dict[int, float] = {}
        self._last_token_time: Dict[int, float] = {}

        self._accumulated = zero_components()

    @property
    def time(self) -> float:
        return self._time

    def advance(self, time: float) -> None:
        dt = time - self._time
        if dt <= 0:
            return

        acc = self._accumulated
        acc["e2e"] += self._num_in_system * dt
        acc["e2e_normalized"] += self._normalized_weight_sum * dt
        acc["ttft"] += self._num_without_first_token * dt
        acc["e2e_squared"] += (
            self._num_in_system * (time * time - self._time * self._time)
            - 2 * self._arrival_time_sum * dt
        )
        self._time = time

    def on_request_arrival(self, request: Request) -> None:
        weight = 1.0 / request.num_decode_tokens
        self._normalized_weights[request.id] = weight

        self._num_in_system += 1
        self._num_without_first_token += 1
        self._normalized_weight_sum += weight
        self._arrival_time_sum += request.arrived_at

    def on_batch_end(self, batch: Batch) -> None:
        for request in batch.requests:
            # a request produces a token in an iteration iff its prefill is
            # complete afterwards (a decode step, or the last prefill chunk)
            if request.is_prefill_complete:
                self._on_token(request)

            if request.completed:
                self._num_in_system -= 1
                self._normalized_weight_sum -= self._normalized_weights.pop(request.id)
                self._arrival_time_sum -= request.arrived_at
                del self._last_token_time[request.id]

    def _on_token(self, request: Request) -> None:
        if request.id not in self._last_token_time:
            self._num_without_first_token -= 1
        else:
            gap = self._time - self._last_token_time[request.id]
            self._accumulated["tbt_violation"] += max(0.0, gap - self._tbt_slo)

        self._last_token_time[request.id] = self._time

    def pop(self) -> Dict[str, float]:
        accumulated, self._accumulated = self._accumulated, zero_components()
        return accumulated
