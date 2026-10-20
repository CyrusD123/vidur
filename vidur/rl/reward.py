import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from vidur.entities import Batch, Request

if TYPE_CHECKING:
    from vidur.rl.env import Episode

# Each component accumulates over simulated time, so summed over an episode it
# equals a per-request metric summed over all requests (τ = thresholds below;
# with threshold_mode="reference_window" each request has its own τ):
#   e2e             ∫ (#requests in system) dt                 = Σ e2e latency
#   e2e_normalized  ∫ Σ_in-system 1/num_decode_tokens dt       = Σ e2e / output length
#   ttft            ∫ (#requests without a first token) dt     = Σ TTFT
#   e2e_squared     ∫ Σ_in-system 2 * age dt                   = Σ e2e²  (penalizes tails)
#   tbt_violation   Σ_tokens max(0, gap - tbt_slo)             (counted when tokens are produced)
#   ttft_hinge      ∫ #(no first token, age > τ_ttft) dt       = Σ max(0, TTFT - τ_ttft)
#   e2e_hinge       ∫ #(in system, age > τ_e2e) dt            = Σ max(0, e2e - τ_e2e)
#   starvation      ∫ Σ_no-first-token 2 max(0, age - τ_s) dt  = Σ max(0, TTFT - τ_s)²
#   batch_time      Σ_batches execution time                   (counted when batches end;
#                                                               not attributable to a request)
REWARD_COMPONENTS = (
    "e2e",
    "e2e_normalized",
    "ttft",
    "e2e_squared",
    "tbt_violation",
    "ttft_hinge",
    "e2e_hinge",
    "starvation",
    "batch_time",
)
# components that are a sum over requests (all but batch_time)
PER_REQUEST_COMPONENTS = REWARD_COMPONENTS[:-1]

# Mean-latency components and the oracle SJF baseline that does best on each
# (vidur/rl/baselines.py). Used to measure how much of the FCFS-to-SJF gain a
# policy achieves; SJF is a strong reference, not a proven optimum.
MEAN_COMPONENT_REFERENCES = {
    "e2e_normalized": "sjf_decode_tokens",
    "e2e": "sjf_total_tokens",
    "ttft": "sjf_total_tokens",
}

# RewardConfig fields that can be calibrated from FCFS episodes, and the
# per-request (or per-token) metric each one is a quantile of
THRESHOLD_METRICS = {
    "tbt_slo": "tbt",
    "ttft_threshold": "ttft",
    "e2e_threshold": "e2e",
    "starvation_threshold": "ttft",
}
# the thresholds that can be set per request (tbt_slo applies to every token)
REQUEST_THRESHOLDS = ("ttft_threshold", "e2e_threshold", "starvation_threshold")
THRESHOLD_MODES = ("fixed", "reference_window")

# threshold name -> request id -> threshold
RequestThresholds = Dict[str, Dict[int, float]]


@dataclass
class RewardConfig:
    """
    reward = -Σ weight * component. Components are always recorded per
    transition, so they can be reweighted later without re-simulating.

    Thresholds (seconds) set to a number are fixed and used as given in every
    mode, e.g. a TTFT SLO: RewardConfig(ttft_threshold=2.0, ...). Fixed and
    calibrated thresholds can be mixed: with threshold_mode="reference_window",
    ttft_threshold=2.0 and e2e_threshold=None, TTFT is judged against the SLO
    and e2e against FCFS's local tail.

    Thresholds left as None are filled in by the environment when
    calibration_quantile is set, depending on threshold_mode:
      fixed             one value: the quantile of the metric over FCFS episodes
                        on calibration_seeds
      reference_window  one value per request: the quantile of the metric over
                        the reference_window_requests requests that arrived
                        closest to it in a run of threshold_reference (a
                        baseline name, see vidur.rl.agents.baseline_agent) on
                        the same workload. Tracks load within the episode: "no
                        worse than FCFS's tail was at this moment". tbt_slo
                        stays a single value.
    A threshold that is still None disables its component (it never crosses).

    normalize_by_baselines puts returns on one scale across seeds and loads by
    dividing every reward by a per-workload cost computed under this same reward:
      ("fcfs",)       FCFS's cost; Episode.normalized_score is then the
                      improvement over FCFS, 1 - cost / cost_fcfs. Recommended:
                      always well defined.
      (a, b)          the cost gap between baselines a and b; the score is the
                      fraction of the a-to-b gap closed. Only meaningful while b
                      beats a under the reward, which tail-weighted rewards can
                      break (see docs/rl_plan.md, weight check).
    The divisor is floored at normalization_min_fraction of baseline a's cost.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {"e2e_normalized": 1.0},
    )
    # target gap between tokens, used by tbt_violation
    tbt_slo: Optional[float] = 0.1
    ttft_threshold: Optional[float] = None
    e2e_threshold: Optional[float] = None
    starvation_threshold: Optional[float] = None

    calibration_quantile: Optional[float] = None
    calibration_seeds: Tuple[int, ...] = (1000, 1001, 1002)

    threshold_mode: str = "fixed"
    threshold_reference: str = "fcfs"
    reference_window_requests: int = 50

    normalize_by_baselines: Optional[Tuple[str, ...]] = None
    normalization_min_fraction: float = 0.05

    def __post_init__(self):
        unknown = set(self.weights) - set(REWARD_COMPONENTS)
        assert not unknown, f"Unknown reward components {unknown}"
        if self.calibration_quantile is not None:
            assert 0 < self.calibration_quantile < 1
        assert (
            self.threshold_mode in THRESHOLD_MODES
        ), f"threshold_mode must be one of {THRESHOLD_MODES}"
        if self.threshold_mode == "reference_window":
            assert (
                self.calibration_quantile is not None
            ), "reference_window thresholds need calibration_quantile"
            assert self.reference_window_requests >= 1
        if self.normalize_by_baselines is not None:
            self.normalize_by_baselines = tuple(self.normalize_by_baselines)
            assert len(self.normalize_by_baselines) in (1, 2)
            assert self.normalization_min_fraction > 0

    @property
    def needs_calibration(self) -> bool:
        """Whether fixed thresholds must be calibrated from FCFS episodes."""
        names = (
            ["tbt_slo"]
            if self.threshold_mode == "reference_window"
            else THRESHOLD_METRICS
        )
        return self.calibration_quantile is not None and any(
            getattr(self, name) is None for name in names
        )

    @property
    def window_thresholds(self) -> Tuple[str, ...]:
        """Thresholds set per request from a reference run of the workload."""
        if self.threshold_mode != "reference_window":
            return ()
        return tuple(name for name in REQUEST_THRESHOLDS if getattr(self, name) is None)

    def combine(self, components: Dict[str, float]) -> float:
        return -sum(
            self.weights.get(name, 0.0) * value for name, value in components.items()
        )


def zero_components() -> Dict[str, float]:
    return {name: 0.0 for name in REWARD_COMPONENTS}


def _threshold(value: Optional[float]) -> float:
    return float("inf") if value is None else value


@dataclass
class RewardContext:
    """Per-workload reward settings derived from baseline runs of that workload."""

    request_thresholds: RequestThresholds = field(default_factory=dict)
    # multiplies every reward (1 / baseline cost or cost gap when normalizing)
    scale: float = 1.0
    # unscaled cost of each normalization baseline on this workload
    reference_costs: Dict[str, float] = field(default_factory=dict)


class HingeIntegral:
    """
    Tracks a set of requests, each with a crossing time c (when its age passes a
    threshold), and integrates over time
      linear:   #(requests past their crossing time)
      squared:  Σ_past-crossing 2 (t - c)
    so that, per request, the integrals are max(0, T - c) and max(0, T - c)² where
    T is when the request leaves the set.
    """

    def __init__(self) -> None:
        self._pending: List[Tuple[float, int]] = []  # heap of (crossing time, id)
        self._crossing: Dict[int, float] = {}
        self._crossed: set = set()
        self._num_crossed = 0
        self._crossing_sum = 0.0

    def add(self, request_id: int, crossing_time: float) -> None:
        self._crossing[request_id] = crossing_time
        heapq.heappush(self._pending, (crossing_time, request_id))

    def remove(self, request_id: int) -> None:
        crossing_time = self._crossing.pop(request_id)
        if request_id in self._crossed:
            self._crossed.remove(request_id)
            self._num_crossed -= 1
            self._crossing_sum -= crossing_time
        # otherwise it stays in the heap and is skipped when popped

    def advance(self, t0: float, t1: float) -> Tuple[float, float]:
        """Integrals over [t0, t1] as (linear, squared)."""
        dt = t1 - t0
        linear = self._num_crossed * dt
        squared = self._num_crossed * (t1 * t1 - t0 * t0) - 2 * self._crossing_sum * dt

        while self._pending and self._pending[0][0] <= t1:
            crossing_time, request_id = heapq.heappop(self._pending)
            if self._crossing.get(request_id) != crossing_time:
                continue  # removed before crossing
            linear += t1 - crossing_time
            squared += (t1 - crossing_time) ** 2
            self._crossed.add(request_id)
            self._num_crossed += 1
            self._crossing_sum += crossing_time

        return linear, squared


class RewardTracker:
    """
    Accumulates the reward components as the simulation advances. The event loop
    calls advance(t) before handling each event and on_request_arrival /
    on_batch_end after it; pop() returns what accumulated since the last pop().

    It also logs every token (time, request id, gap since the request's previous
    token) so per-request rewards can be recomputed offline.
    """

    def __init__(
        self,
        reward_config: RewardConfig,
        request_thresholds: Optional[RequestThresholds] = None,
    ) -> None:
        self._request_thresholds = request_thresholds or {}
        self._tbt_slo = _threshold(reward_config.tbt_slo)
        self._ttft_threshold = _threshold(reward_config.ttft_threshold)
        self._e2e_threshold = _threshold(reward_config.e2e_threshold)
        self._starvation_threshold = _threshold(reward_config.starvation_threshold)

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

        self._ttft_hinge = HingeIntegral()
        self._e2e_hinge = HingeIntegral()
        self._starvation = HingeIntegral()

        self.token_times: List[float] = []
        self.token_request_ids: List[int] = []
        # nan for a request's first token
        self.token_gaps: List[float] = []

        self._accumulated = zero_components()

    @property
    def time(self) -> float:
        return self._time

    def advance(self, time: float) -> None:
        t0 = self._time
        dt = time - t0
        if dt <= 0:
            return

        acc = self._accumulated
        acc["e2e"] += self._num_in_system * dt
        acc["e2e_normalized"] += self._normalized_weight_sum * dt
        acc["ttft"] += self._num_without_first_token * dt
        acc["e2e_squared"] += (
            self._num_in_system * (time * time - t0 * t0)
            - 2 * self._arrival_time_sum * dt
        )
        acc["ttft_hinge"] += self._ttft_hinge.advance(t0, time)[0]
        acc["e2e_hinge"] += self._e2e_hinge.advance(t0, time)[0]
        acc["starvation"] += self._starvation.advance(t0, time)[1]
        self._time = time

    def on_request_arrival(self, request: Request) -> None:
        weight = 1.0 / request.num_decode_tokens
        self._normalized_weights[request.id] = weight

        self._num_in_system += 1
        self._num_without_first_token += 1
        self._normalized_weight_sum += weight
        self._arrival_time_sum += request.arrived_at

        self._ttft_hinge.add(
            request.id, self._crossing("ttft_threshold", self._ttft_threshold, request)
        )
        self._e2e_hinge.add(
            request.id, self._crossing("e2e_threshold", self._e2e_threshold, request)
        )
        self._starvation.add(
            request.id,
            self._crossing("starvation_threshold", self._starvation_threshold, request),
        )

    def _crossing(self, name: str, default: float, request: Request) -> float:
        threshold = self._request_thresholds.get(name, {}).get(request.id, default)
        return request.arrived_at + threshold

    def on_batch_end(self, batch: Batch) -> None:
        self._accumulated["batch_time"] += batch.completed_at - batch.scheduled_at

        for request in batch.requests:
            # a request produces a token in an iteration iff its prefill is
            # complete afterwards (a decode step, or the last prefill chunk)
            if request.is_prefill_complete:
                self._on_token(request)

            if request.completed:
                self._num_in_system -= 1
                self._normalized_weight_sum -= self._normalized_weights.pop(request.id)
                self._arrival_time_sum -= request.arrived_at
                self._e2e_hinge.remove(request.id)
                del self._last_token_time[request.id]

    def _on_token(self, request: Request) -> None:
        if request.id not in self._last_token_time:
            self._num_without_first_token -= 1
            self._ttft_hinge.remove(request.id)
            self._starvation.remove(request.id)
            gap = float("nan")
        else:
            gap = self._time - self._last_token_time[request.id]
            self._accumulated["tbt_violation"] += max(0.0, gap - self._tbt_slo)

        self._last_token_time[request.id] = self._time
        self.token_times.append(self._time)
        self.token_request_ids.append(request.id)
        self.token_gaps.append(gap)

    def pop(self) -> Dict[str, float]:
        accumulated, self._accumulated = self._accumulated, zero_components()
        return accumulated


def _overlap(start, end, t0, t1):
    """Length of [start, end] ∩ [t0, t1], broadcast."""
    return np.maximum(0.0, np.minimum(end, t1) - np.maximum(start, t0))


def _squared_overlap(origin, end, t0, t1):
    """∫ 2 (t - origin) dt over [max(origin, t0), min(end, t1)], broadcast."""
    lo = np.maximum(origin, t0)
    hi = np.minimum(end, t1)
    value = (hi - origin) ** 2 - (lo - origin) ** 2
    return np.where(hi > lo, value, 0.0)


def _request_arrays(
    episode: "Episode",
    reward_config: RewardConfig,
    request_thresholds: Optional[RequestThresholds],
) -> Dict[str, np.ndarray]:
    """Per-request times and thresholds as [1, num_requests] arrays."""
    requests = episode.requests
    if request_thresholds is None:
        request_thresholds = episode.request_thresholds

    def row(values):
        return np.array(values, dtype=float)[None, :]

    arrays = {
        "arrived": row([r.arrived_at for r in requests]),
        "first_token": row([r.prefill_completed_at for r in requests]),
        "completed": row([r.completed_at for r in requests]),
        "output_len": row([episode.num_decode_tokens[r.id] for r in requests]),
    }
    for name in REQUEST_THRESHOLDS:
        default = _threshold(getattr(reward_config, name))
        per_request = request_thresholds.get(name, {})
        arrays[name] = row([per_request.get(r.id, default) for r in requests])
    return arrays


def _overlap_components(
    arrays: Dict[str, np.ndarray], t0: np.ndarray, t1: np.ndarray
) -> Dict[str, np.ndarray]:
    """Per-request components accumulated over [t0, t1] (all but tbt_violation)."""
    arrived, first_token = arrays["arrived"], arrays["first_token"]
    completed = arrays["completed"]
    with np.errstate(invalid="ignore"):
        in_system = _overlap(arrived, completed, t0, t1)
        return {
            "e2e": in_system,
            "e2e_normalized": in_system / arrays["output_len"],
            "ttft": _overlap(arrived, first_token, t0, t1),
            "e2e_squared": _squared_overlap(arrived, completed, t0, t1),
            "ttft_hinge": _overlap(
                arrived + arrays["ttft_threshold"], first_token, t0, t1
            ),
            "e2e_hinge": _overlap(arrived + arrays["e2e_threshold"], completed, t0, t1),
            "starvation": _squared_overlap(
                arrived + arrays["starvation_threshold"], first_token, t0, t1
            ),
        }


def per_request_reward_components(
    episode: "Episode",
    reward_config: RewardConfig,
    request_thresholds: Optional[RequestThresholds] = None,
) -> Dict[str, np.ndarray]:
    """
    Splits each transition's reward components across requests: returns, for
    every per-request component, an array [num_transitions, num_requests] whose
    rows sum to the transition's component (columns follow episode.requests).
    Thresholds come from reward_config, with per-request thresholds from
    request_thresholds (default: the ones the episode was run with), so they can
    differ from the rollout's. Intended as targets for a per-request value head.
    Unscaled: multiply by the episode's reward_scale to match its rewards.
    """
    transitions = episode.transitions
    requests = episode.requests
    num_transitions, num_requests = len(transitions), len(requests)
    if num_transitions == 0:
        return {name: np.zeros((0, num_requests)) for name in PER_REQUEST_COMPONENTS}

    starts = np.array([t.time for t in transitions])
    ends = starts + np.array([t.duration for t in transitions])
    arrays = _request_arrays(episode, reward_config, request_thresholds)
    components = _overlap_components(arrays, starts[:, None], ends[:, None])

    # tokens are credited to the transition whose interval (start, end] holds them
    tbt = np.zeros((num_transitions, num_requests))
    gaps = episode.token_gaps
    has_gap = ~np.isnan(gaps)
    if has_gap.any():
        violation = np.maximum(0.0, gaps[has_gap] - _threshold(reward_config.tbt_slo))
        bounds = np.append(starts, ends[-1])
        rows = np.searchsorted(bounds, episode.token_times[has_gap], side="left") - 1
        column_of = {r.id: column for column, r in enumerate(requests)}
        columns = np.array(
            [column_of[i] for i in episode.token_request_ids[has_gap]], dtype=np.int64
        )
        valid = (rows >= 0) & (rows < num_transitions)
        np.add.at(tbt, (rows[valid], columns[valid]), violation[valid])
    components["tbt_violation"] = tbt

    return components


def episode_cost_components(
    episode: "Episode",
    reward_config: RewardConfig,
    request_thresholds: Optional[RequestThresholds] = None,
) -> Dict[str, float]:
    """
    Every component summed over the whole episode, in closed form from request
    timings, so a finished episode can be rescored under any reward config
    (thresholds and weights) without re-simulating. Matches what the
    RewardTracker accumulates when run with the same thresholds.
    """
    arrays = _request_arrays(episode, reward_config, request_thresholds)
    everything = (np.array([[-np.inf]]), np.array([[np.inf]]))
    totals = {
        name: float(np.sum(value))
        for name, value in _overlap_components(arrays, *everything).items()
    }

    gaps = episode.token_gaps[~np.isnan(episode.token_gaps)]
    totals["tbt_violation"] = float(
        np.maximum(0.0, gaps - _threshold(reward_config.tbt_slo)).sum()
    )
    # recorded on every transition whatever the weights
    totals["batch_time"] = episode.initial_reward_components["batch_time"] + sum(
        t.reward_components["batch_time"] for t in episode.transitions
    )
    return totals


def episode_cost(
    episode: "Episode",
    reward_config: RewardConfig,
    request_thresholds: Optional[RequestThresholds] = None,
) -> float:
    """Σ weight * component over the episode (unscaled negative return)."""
    return -reward_config.combine(
        episode_cost_components(episode, reward_config, request_thresholds)
    )


def window_thresholds(
    reference: "Episode", reward_config: RewardConfig
) -> RequestThresholds:
    """
    Per-request thresholds for reward_config.window_thresholds: for each request,
    calibration_quantile of the metric over the reference_window_requests
    requests of the reference episode that arrived closest to it (a window
    centered on it in arrival order, shifted inward at the ends). Request ids
    match across runs of the same workload.
    """
    requests = sorted(
        (r for r in reference.requests if r.completed), key=lambda r: r.arrived_at
    )
    metrics = {
        "ttft": np.array([r.prefill_completed_at - r.arrived_at for r in requests]),
        "e2e": np.array([r.completed_at - r.arrived_at for r in requests]),
    }
    num_requests = len(requests)
    window = min(reward_config.reference_window_requests, num_requests)
    starts = np.clip(np.arange(num_requests) - window // 2, 0, num_requests - window)

    thresholds: RequestThresholds = {}
    for name in reward_config.window_thresholds:
        values = metrics[THRESHOLD_METRICS[name]]
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        quantiles = np.quantile(windows, reward_config.calibration_quantile, axis=1)
        thresholds[name] = {
            r.id: float(quantiles[start]) for r, start in zip(requests, starts)
        }
    return thresholds


def per_request_rewards(episode: "Episode", reward_config: RewardConfig) -> np.ndarray:
    """[num_transitions, num_requests] weighted per-request rewards, scaled like
    the episode's rewards (batch_time, which belongs to no request, is left
    out)."""
    components = per_request_reward_components(episode, reward_config)
    total = np.zeros_like(components["e2e"])
    for name, value in components.items():
        total -= reward_config.weights.get(name, 0.0) * value
    return total * episode.reward_scale


def episode_metric_samples(episode: "Episode") -> Dict[str, np.ndarray]:
    """Per-request ttft/e2e and per-token tbt samples over completed requests."""
    completed = [r for r in episode.requests if r.completed]
    gaps = episode.token_gaps
    return {
        "ttft": np.array([r.prefill_completed_at - r.arrived_at for r in completed]),
        "e2e": np.array([r.completed_at - r.arrived_at for r in completed]),
        "tbt": gaps[~np.isnan(gaps)],
    }
