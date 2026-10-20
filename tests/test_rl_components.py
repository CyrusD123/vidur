"""
Checks for the RL scheduler extensions: tie-breaking, decision modes, request
history, observation features, reward components, and admission action bounds.

Run from the repo root (so ./data and ./cache resolve):
    python -m pytest tests/test_rl_components.py -v
"""

import copy
import math

import numpy as np
import pytest

from vidur.rl import (
    GLOBAL_FEATURES,
    RUNNING_FEATURES,
    WAITING_FEATURES,
    FCFSAgent,
    RandomAgent,
    RewardConfig,
    VidurSchedulingEnv,
    make_simulation_config,
    per_request_reward_components,
)
from vidur.rl.env import (
    AgentPolicy,
    RLSimulator,
    _reset_id_counters,
    _set_workload_seed,
)
from vidur.rl.features import FeatureContext
from vidur.rl.reward import PER_REQUEST_COMPONENTS, RewardTracker
from vidur.scheduler.replica_scheduler.priority_policy import BasePriorityPolicy
from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import SlidingWindow

WORKLOAD_SEED = 0

THRESHOLDS = dict(
    tbt_slo=0.03,
    ttft_threshold=0.5,
    e2e_threshold=5.0,
    starvation_threshold=0.3,
)


def _config(tmp_path_factory, **kwargs):
    # small KV cache so the queue builds up and requests get preempted
    return make_simulation_config(
        qps=10,
        num_requests=300,
        batch_size_cap=128,
        num_blocks=4000,
        output_dir=str(tmp_path_factory.mktemp("sim")),
        **kwargs,
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return VidurSchedulingEnv(_config(tmp_path_factory), RewardConfig(**THRESHOLDS))


@pytest.fixture(scope="module")
def random_episode(env):
    return env.run_episode(RandomAgent(seed=0), workload_seed=WORKLOAD_SEED)


def _run_with_policy(env, policy, config=None):
    config = copy.copy(config or env.config)
    config.request_generator_config = copy.deepcopy(config.request_generator_config)
    _set_workload_seed(config.request_generator_config, WORKLOAD_SEED)

    _reset_id_counters()
    simulator = RLSimulator(
        config, env.execution_time_predictor, RewardTracker(RewardConfig())
    )
    simulator.set_priority_policy(policy)
    simulator.run()
    return simulator


class RandomPolicy(BasePriorityPolicy):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def get_priorities(self, waiting_requests, scheduler):
        self.scheduler = scheduler
        self.priorities = self.rng.integers(
            0, scheduler.num_priority_levels, len(waiting_requests)
        )
        return self.priorities.tolist()


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


def test_ties_are_broken_by_arrival(env):
    class CheckingPolicy(RandomPolicy):
        num_checked = 0

        def get_priorities(self, waiting_requests, scheduler):
            priorities = super().get_priorities(waiting_requests, scheduler)
            self.key = {
                r.id: (p, r.arrived_at) for r, p in zip(waiting_requests, priorities)
            }
            return priorities

        def on_batch_formed(self, admitted_requests, boundary_request):
            if not admitted_requests:
                return
            # what is left of the ranked queue stays sorted by (priority, arrival)
            remaining = [
                self.key[r.id]
                for r in self.scheduler.waiting_requests
                if r.id in self.key
            ]
            assert remaining == sorted(remaining)
            # and every admitted request ranked ahead of the boundary
            if boundary_request is not None:
                assert all(
                    self.key[r.id] < self.key[boundary_request.id]
                    for r in admitted_requests
                )
            self.num_checked += 1

    policy = CheckingPolicy()
    _run_with_policy(env, policy)
    assert policy.num_checked > 0


def test_request_history_counts(env):
    class CheckingPolicy(RandomPolicy):
        num_checked = 0

        def get_priorities(self, waiting_requests, scheduler):
            self.waiting = list(waiting_requests)
            self.before = {
                r.id: copy.copy(scheduler.request_history(r.id))
                for r in waiting_requests
            }
            return super().get_priorities(waiting_requests, scheduler)

        def on_batch_formed(self, admitted_requests, boundary_request):
            admitted_ids = {r.id for r in admitted_requests}
            for r in self.waiting:
                if r.id in admitted_ids:
                    continue
                before = self.before[r.id]
                after = self.scheduler.request_history(r.id)
                assert after.num_decisions_seen == before.num_decisions_seen + 1
                overtaken = sum(a.arrived_at > r.arrived_at for a in admitted_requests)
                passed_over = int(bool(admitted_requests))
                assert after.times_passed_over == before.times_passed_over + passed_over
                assert after.num_overtaken_by == before.num_overtaken_by + overtaken
                self.num_checked += 1

    policy = CheckingPolicy()
    _run_with_policy(env, policy)
    assert policy.num_checked > 0


def test_sliding_window_matches_brute_force():
    rng = np.random.default_rng(0)
    values = rng.normal(size=500)
    window = SlidingWindow(37)
    for i, value in enumerate(values):
        window.add(value)
        recent = values[max(0, i - 36) : i + 1]
        mean, low, high = window.stats()
        assert math.isclose(mean, recent.mean(), abs_tol=1e-9)
        assert low == recent.min() and high == recent.max()


def test_arrival_and_periodic_mode(tmp_path_factory):
    config = _config(
        tmp_path_factory, decision_mode="arrival_and_periodic", rerank_interval_s=0.5
    )
    env = VidurSchedulingEnv(config)

    class CheckingPolicy(RandomPolicy):
        num_partial = 0

        def get_priorities(self, waiting_requests, scheduler):
            mask = scheduler.rank_mask
            assert len(mask) == len(waiting_requests) and any(mask)
            for r, ranked in zip(waiting_requests, mask):
                # only requests without a priority are ranked between re-rankings
                has_priority = scheduler.request_history(r.id).priority is not None
                assert ranked or has_priority
            if not all(mask):
                self.num_partial += 1
                assert not any(
                    scheduler.request_history(r.id).priority is not None
                    for r, ranked in zip(waiting_requests, mask)
                    if ranked
                )
            return super().get_priorities(waiting_requests, scheduler)

    policy = CheckingPolicy()
    simulator = _run_with_policy(env, policy, config)
    assert all(r.completed for r in simulator.requests)
    assert policy.num_partial > 0

    episode = env.run_episode(RandomAgent(seed=0), workload_seed=WORKLOAD_SEED)
    assert all(r.completed for r in episode.requests)
    # persistent priorities: every decision is kept and bounds are exact
    assert any(t.num_admitted == 0 for t in episode.transitions)
    for t in episode.transitions:
        assert np.array_equal(t.action_low, t.action) and np.array_equal(
            t.action_high, t.action
        )


# ---------------------------------------------------------------------------
# observation features
# ---------------------------------------------------------------------------

FORBIDDEN_REQUEST_ATTRIBUTES = {
    "num_decode_tokens",
    "_num_decode_tokens",
    "total_tokens",
    "pd_ratio",
    "size",
    "completed_at",
    "e2e_time",
    "e2e_time_normalized",
    "execution_time_normalized",
    "model_execution_time_normalized",
}


class OutputLengthGuard:
    """Proxy that fails on any attribute derived from the output length."""

    def __init__(self, request):
        object.__setattr__(self, "_request", request)

    def __getattr__(self, name):
        assert name not in FORBIDDEN_REQUEST_ATTRIBUTES, f"feature read {name}"
        return getattr(object.__getattribute__(self, "_request"), name)


def test_features_never_read_output_length(env):
    class CheckingPolicy(RandomPolicy):
        num_checked = 0

        def get_priorities(self, waiting_requests, scheduler):
            ctx = FeatureContext(
                time=scheduler.current_time,
                scheduler=scheduler,
                waiting_requests=[OutputLengthGuard(r) for r in waiting_requests],
                waiting_histories=[
                    scheduler.request_history(r.id) for r in waiting_requests
                ],
                rank_mask=scheduler.rank_mask,
                running_requests=[
                    OutputLengthGuard(r) for r in scheduler.running_requests
                ],
            )
            for registry in (WAITING_FEATURES, RUNNING_FEATURES):
                for feature in registry.values():
                    values = feature.compute(ctx)
                    assert np.isfinite(values).all(), feature.name
            for feature in GLOBAL_FEATURES.values():
                assert np.isfinite(feature.compute(ctx)), feature.name
            self.num_checked += 1
            return super().get_priorities(waiting_requests, scheduler)

    policy = CheckingPolicy()
    _run_with_policy(env, policy)
    assert policy.num_checked > 0


def test_observation_shapes(env, random_episode):
    config = env.observation_config
    for t in random_episode.transitions:
        obs = t.obs
        assert obs.waiting_features.shape == (
            obs.num_requests,
            config.num_waiting_features,
        )
        assert obs.running_features.shape == (
            obs.num_running,
            config.num_running_features,
        )
        assert obs.global_features.shape == (config.num_global_features,)
        assert obs.rank_mask.all()
        assert math.isclose(obs.time, t.time)
        assert np.isfinite(obs.waiting_features).all()
        assert np.isfinite(obs.global_features).all()


# ---------------------------------------------------------------------------
# admission action bounds
# ---------------------------------------------------------------------------


def test_action_bounds_reproduce_admission(random_episode):
    rng = np.random.default_rng(0)
    transitions = random_episode.transitions
    assert any(t.boundary_index is not None for t in transitions)

    for t in transitions:
        low, high = t.action_low, t.action_high
        assert ((low <= t.action) & (t.action <= high)).all()

        if t.boundary_index is None:
            assert t.admitted.all()
            continue
        assert not t.admitted[t.boundary_index]

        # any action inside the bounds puts exactly the admitted requests ahead
        # of the boundary request in the (priority, arrival) order
        for _ in range(5):
            action = rng.integers(low, high + 1)
            keys = t.obs.ranking_keys(action)
            b = t.boundary_index
            ahead = np.array([keys[i] < keys[b] for i in range(len(action))])
            assert np.array_equal(ahead, t.admitted)


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------


def _episode_totals(episode):
    totals = dict(episode.initial_reward_components)
    for transition in episode.transitions:
        for name, value in transition.reward_components.items():
            totals[name] += value
    return totals


def test_hinge_components_sum_to_request_metrics(random_episode):
    requests = random_episode.requests
    assert all(r.completed for r in requests)
    totals = _episode_totals(random_episode)

    ttft = np.array([r.prefill_completed_at - r.arrived_at for r in requests])
    e2e = np.array([r.completed_at - r.arrived_at for r in requests])
    gaps = random_episode.token_gaps[~np.isnan(random_episode.token_gaps)]
    expected = {
        "ttft_hinge": np.maximum(0, ttft - THRESHOLDS["ttft_threshold"]).sum(),
        "e2e_hinge": np.maximum(0, e2e - THRESHOLDS["e2e_threshold"]).sum(),
        "starvation": (
            np.maximum(0, ttft - THRESHOLDS["starvation_threshold"]) ** 2
        ).sum(),
        "tbt_violation": np.maximum(0, gaps - THRESHOLDS["tbt_slo"]).sum(),
    }
    for name, value in expected.items():
        assert value > 0, name
        assert math.isclose(totals[name], value, rel_tol=1e-9), name


def test_token_log_matches_requests(random_episode):
    for r in random_episode.requests:
        mask = random_episode.token_request_ids == r.id
        times = random_episode.token_times[mask]
        assert times[0] == r.prefill_completed_at
        assert times[-1] == r.completed_at


def test_per_request_rewards_sum_to_transition_rewards(env, random_episode):
    components = per_request_reward_components(random_episode, env.reward_config)
    num_transitions = len(random_episode.transitions)
    for name in PER_REQUEST_COMPONENTS:
        assert components[name].shape == (
            num_transitions,
            len(random_episode.requests),
        )
        per_transition = components[name].sum(axis=1)
        recorded = np.array(
            [t.reward_components[name] for t in random_episode.transitions]
        )
        np.testing.assert_allclose(per_transition, recorded, rtol=1e-7, atol=1e-9)


def test_calibrated_thresholds(tmp_path_factory):
    env = VidurSchedulingEnv(
        _config(tmp_path_factory),
        RewardConfig(
            ttft_threshold=None,
            tbt_slo=None,
            calibration_quantile=0.9,
            calibration_seeds=(1000,),
        ),
    )
    reward_config = env.reward_config
    assert not reward_config.needs_calibration
    # every threshold left as None was filled in
    for name in ("tbt_slo", "ttft_threshold", "e2e_threshold", "starvation_threshold"):
        assert getattr(reward_config, name) > 0, name

    episode = env.run_episode(FCFSAgent(), workload_seed=1000)
    ttft = np.array([r.prefill_completed_at - r.arrived_at for r in episode.requests])
    assert math.isclose((ttft > reward_config.ttft_threshold).mean(), 0.1, abs_tol=0.01)


def test_agent_policy_matches_recording_policy(env, random_episode):
    # the non-recording policy (used by forks) makes the same decisions
    simulator = env.make_simulator(WORKLOAD_SEED)
    simulator.set_priority_policy(
        AgentPolicy(RandomAgent(seed=0), env.observation_config)
    )
    simulator.run()
    outcomes = [(r.id, r.completed_at) for r in simulator.requests]
    assert outcomes == [(r.id, r.completed_at) for r in random_episode.requests]
