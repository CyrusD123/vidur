"""
Checks for the oracle shortest-job-first baselines.

Run from the repo root (so ./data and ./cache resolve):
    python -m pytest tests/test_rl_oracle.py -v
"""

import numpy as np
import pytest

from vidur.rl import (
    REWARD_COMPONENTS,
    FCFSAgent,
    OracleSJFAgent,
    RewardConfig,
    VidurSchedulingEnv,
    baseline_agent,
    episode_cost_components,
    make_simulation_config,
    per_request_rewards,
)
from vidur.scheduler.replica_scheduler.priority_policy import SJFPriorityPolicy

WORKLOAD_SEED = 0


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    config = make_simulation_config(
        qps=10,
        num_requests=150,
        batch_size_cap=128,
        num_blocks=4000,
        output_dir=str(tmp_path_factory.mktemp("sim")),
    )
    return VidurSchedulingEnv(config)


def _outcomes(requests):
    return [(r.id, r.prefill_completed_at, r.completed_at) for r in requests]


def test_oracle_cannot_act_on_observation_alone(env):
    episode = env.run_episode(FCFSAgent(), workload_seed=WORKLOAD_SEED)
    with pytest.raises(TypeError):
        OracleSJFAgent().act(episode.transitions[0].obs)


def test_exact_sjf_admits_shortest_first(env):
    episode = env.run_episode(OracleSJFAgent(), workload_seed=WORKLOAD_SEED)
    requests = {r.id: r for r in episode.requests}
    num_checked = 0
    for t in episode.transitions:
        if t.boundary_index is None:
            continue
        keys = t.obs.ranking_keys(t.action)
        boundary_key = keys[t.boundary_index]
        assert all(keys[i] < boundary_key for i in np.flatnonzero(t.admitted))
        num_checked += 1
    assert num_checked > 0
    assert all(r.completed for r in requests.values())


def test_agent_matches_scheduler_policy(env):
    via_agent = env.run_episode(OracleSJFAgent(), workload_seed=WORKLOAD_SEED)

    simulator = env.make_simulator(WORKLOAD_SEED)
    simulator.set_priority_policy(SJFPriorityPolicy())
    simulator.run()

    assert _outcomes(simulator.requests) == _outcomes(via_agent.requests)
    fcfs = env.run_episode(FCFSAgent(), workload_seed=WORKLOAD_SEED)
    assert _outcomes(fcfs.requests) != _outcomes(via_agent.requests)


def test_bucketed_sjf_stays_in_levels(env):
    num_levels = env.config.cluster_config.replica_scheduler_config.num_priority_levels
    episode = env.run_episode(
        OracleSJFAgent(key="decode_tokens", num_levels=num_levels),
        workload_seed=WORKLOAD_SEED,
    )
    for t in episode.transitions:
        assert ((0 <= t.action) & (t.action < num_levels)).all()
        assert (t.action_low <= t.action).all() and (t.action <= t.action_high).all()


def test_baseline_names():
    agent = baseline_agent("sjf_total_tokens_age20_16lvl")
    assert isinstance(agent, OracleSJFAgent) and not agent.unbounded_priorities
    assert baseline_agent("sjf_decode_tokens").unbounded_priorities
    with pytest.raises(ValueError):
        baseline_agent("sjf_output_tokens")


def test_huge_aging_is_fcfs(env):
    # aging dominates any job size difference, so the order is by arrival
    aged = env.run_episode(
        baseline_agent("sjf_total_tokens_age1000000000"), workload_seed=WORKLOAD_SEED
    )
    fcfs = env.run_episode(FCFSAgent(), workload_seed=WORKLOAD_SEED)
    assert _outcomes(aged.requests) == _outcomes(fcfs.requests)


def _tracked_totals(episode):
    totals = dict(episode.initial_reward_components)
    for t in episode.transitions:
        for name, value in t.reward_components.items():
            totals[name] += value
    return totals


@pytest.mark.parametrize("window", [20, None])
def test_window_thresholds_and_closed_form_cost(env, window):
    reward_config = RewardConfig(
        weights={"e2e": 1.0, "e2e_hinge": 1.0, "ttft_hinge": 1.0, "starvation": 1.0},
        threshold_mode="reference_window",
        calibration_quantile=0.9,
        reference_window_requests=window or 10**9,
    )
    episode = env.run_episode(
        baseline_agent("sjf_decode_tokens"),
        workload_seed=WORKLOAD_SEED,
        reward_config=reward_config,
    )
    fcfs = env.reference_episode("fcfs", WORKLOAD_SEED)
    thresholds = episode.request_thresholds
    assert set(thresholds) == {
        "ttft_threshold",
        "e2e_threshold",
        "starvation_threshold",
    }

    ids = [r.id for r in episode.requests]
    e2e = {r.id: r.completed_at - r.arrived_at for r in fcfs.requests}
    ttft = {r.id: r.prefill_completed_at - r.arrived_at for r in fcfs.requests}
    if window is None:
        # the whole episode: one threshold, FCFS's quantile
        expected = np.quantile(list(e2e.values()), 0.9)
        assert np.allclose([thresholds["e2e_threshold"][i] for i in ids], expected)
    else:
        e2e_thresholds = np.array([thresholds["e2e_threshold"][i] for i in ids])
        assert np.ptp(e2e_thresholds) > 0  # varies with load over the episode
        # first request: window of the first 20 arrivals
        first = sorted(fcfs.requests, key=lambda r: r.arrived_at)[:20]
        assert np.isclose(
            thresholds["ttft_threshold"][first[0].id],
            np.quantile([ttft[r.id] for r in first], 0.9),
        )

    # the tracker, run with per-request thresholds, matches the closed form
    tracked = _tracked_totals(episode)
    closed = episode_cost_components(episode, reward_config)
    for name in REWARD_COMPONENTS:
        assert np.isclose(tracked[name], closed[name], rtol=1e-6, atol=1e-6), name

    # and the per-request split still sums to every transition's reward
    rewards = per_request_rewards(episode, reward_config)
    for row, t in zip(rewards, episode.transitions):
        expected = t.reward + reward_config.combine(
            {"batch_time": t.reward_components["batch_time"]}
        )
        assert np.isclose(row.sum(), expected, rtol=1e-6, atol=1e-6)


def test_gap_normalization(env):
    reward_config = RewardConfig(normalize_by_baselines=("fcfs", "sjf_decode_tokens"))
    scores = {}
    for name in ("fcfs", "sjf_decode_tokens", "random"):
        episode = env.run_episode(
            baseline_agent(name, seed=0),
            workload_seed=WORKLOAD_SEED,
            reward_config=reward_config,
        )
        costs = episode.reference_costs
        assert np.isclose(
            episode.reward_scale, 1 / abs(costs["fcfs"] - costs["sjf_decode_tokens"])
        )
        # rewards are the scaled cost
        initial = reward_config.combine(episode.initial_reward_components)
        assert np.isclose(
            (episode.total_reward / episode.reward_scale) + initial, -episode.cost
        )
        scores[name] = episode.normalized_score
    assert np.isclose(scores["fcfs"], 0.0)
    assert np.isclose(scores["sjf_decode_tokens"], 1.0)


def test_normalize_by_fcfs(env):
    reward_config = RewardConfig(normalize_by_baselines=("fcfs",))
    episode = env.run_episode(
        baseline_agent("sjf_decode_tokens"),
        workload_seed=WORKLOAD_SEED,
        reward_config=reward_config,
    )
    cost_fcfs = episode.reference_costs["fcfs"]
    assert np.isclose(episode.reward_scale, 1 / cost_fcfs)
    assert np.isclose(episode.normalized_score, 1 - episode.cost / cost_fcfs)


def test_mean_gap_closed(env):
    def gap(name, mean_term="e2e_normalized"):
        episode = env.run_episode(baseline_agent(name, seed=0), WORKLOAD_SEED)
        return env.mean_gap_closed(episode, WORKLOAD_SEED, mean_term=mean_term)

    assert np.isclose(gap("fcfs"), 0.0)
    assert np.isclose(gap("sjf_decode_tokens"), 1.0)
    assert np.isclose(gap("sjf_total_tokens", mean_term="e2e"), 1.0)
    assert np.isfinite(gap("random"))


def test_fixed_slo_mixes_with_window_thresholds(env):
    # a user-defined TTFT SLO stays fixed; e2e comes from FCFS windows
    reward_config = RewardConfig(
        weights={"ttft_hinge": 1.0, "e2e_hinge": 1.0},
        ttft_threshold=2.0,
        threshold_mode="reference_window",
        calibration_quantile=0.95,
    )
    episode = env.run_episode(
        baseline_agent("sjf_decode_tokens"),
        workload_seed=WORKLOAD_SEED,
        reward_config=reward_config,
    )
    assert set(episode.request_thresholds) == {"e2e_threshold", "starvation_threshold"}

    ttft = np.array([r.prefill_completed_at - r.arrived_at for r in episode.requests])
    tracked = _tracked_totals(episode)
    assert np.isclose(tracked["ttft_hinge"], np.maximum(0, ttft - 2.0).sum())
    closed = episode_cost_components(episode, reward_config)
    assert np.isclose(tracked["e2e_hinge"], closed["e2e_hinge"])
