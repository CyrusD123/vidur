"""
Checks for the RL environment (vidur/rl).

Run from the repo root (so ./data and ./cache resolve):
    python -m pytest tests/test_rl_env.py -v
"""

import copy
import math

import pytest

from vidur.config import VllmSchedulerConfig
from vidur.rl import (
    FCFSAgent,
    RandomAgent,
    RewardConfig,
    VidurSchedulingEnv,
    make_simulation_config,
)
from vidur.rl.env import RLSimulator, _reset_id_counters, _set_workload_seed
from vidur.rl.reward import RewardTracker

WORKLOAD_SEED = 0


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    # small KV cache so the queue builds up and requests get preempted
    config = make_simulation_config(
        qps=10,
        num_requests=300,
        batch_size_cap=128,
        num_blocks=4000,
        output_dir=str(tmp_path_factory.mktemp("sim")),
    )
    # tbt_slo=0 makes tbt_violation the plain sum of token gaps, which has an
    # exact closed form to check against
    return VidurSchedulingEnv(config, RewardConfig(tbt_slo=0.0))


@pytest.fixture(scope="module")
def fcfs_episode(env):
    return env.run_episode(FCFSAgent(), workload_seed=WORKLOAD_SEED)


@pytest.fixture(scope="module")
def random_episode(env):
    return env.run_episode(RandomAgent(seed=0), workload_seed=WORKLOAD_SEED)


def _request_outcomes(requests):
    return [
        (
            r.id,
            r.arrived_at,
            r.prefill_completed_at,
            r.completed_at,
            r.num_restarts,
        )
        for r in requests
    ]


def test_fcfs_agent_matches_stock_vllm(env, fcfs_episode):
    # same workload through the stock vLLM scheduler
    config = copy.copy(env.config)
    config.cluster_config = copy.deepcopy(env.config.cluster_config)
    rl_scheduler_config = env.config.cluster_config.replica_scheduler_config
    config.cluster_config.replica_scheduler_config = VllmSchedulerConfig(
        batch_size_cap=rl_scheduler_config.batch_size_cap,
        max_num_batched_tokens=rl_scheduler_config.max_num_batched_tokens,
        num_blocks=rl_scheduler_config.num_blocks,
    )
    config.request_generator_config = copy.deepcopy(env.config.request_generator_config)
    _set_workload_seed(config.request_generator_config, WORKLOAD_SEED)

    _reset_id_counters()
    simulator = RLSimulator(
        config, env.execution_time_predictor, RewardTracker(RewardConfig())
    )
    simulator.run()

    assert fcfs_episode.summary["num_restarts"] > 0
    assert _request_outcomes(fcfs_episode.requests) == _request_outcomes(
        simulator.requests
    )


@pytest.mark.parametrize("episode_name", ["fcfs_episode", "random_episode"])
def test_reward_components_sum_to_request_metrics(episode_name, request):
    episode = request.getfixturevalue(episode_name)
    requests = episode.requests
    assert all(r.completed for r in requests)

    totals = dict(episode.initial_reward_components)
    for transition in episode.transitions:
        for name, value in transition.reward_components.items():
            totals[name] += value

    e2e = [r.completed_at - r.arrived_at for r in requests]
    expected = {
        "e2e": sum(e2e),
        "e2e_normalized": sum(
            latency / episode.num_decode_tokens[r.id]
            for latency, r in zip(e2e, requests)
        ),
        "ttft": sum(r.prefill_completed_at - r.arrived_at for r in requests),
        "e2e_squared": sum(latency**2 for latency in e2e),
        # with tbt_slo=0 the gaps telescope from first to last token
        "tbt_violation": sum(r.completed_at - r.prefill_completed_at for r in requests),
    }

    for name, value in expected.items():
        assert math.isclose(totals[name], value, rel_tol=1e-9), name


def test_reward_is_negative_weighted_components(env, random_episode):
    for transition in random_episode.transitions:
        assert transition.reward == env.reward_config.combine(
            transition.reward_components
        )
    assert random_episode.summary["total_reward"] < 0


def test_only_decisions_that_admitted_requests_are_kept(random_episode):
    transitions = random_episode.transitions
    assert len(transitions) > 0
    assert all(t.num_admitted > 0 for t in transitions)
    assert all(t.obs.num_requests >= 2 for t in transitions)
    assert all(len(t.action) == t.obs.num_requests for t in transitions)

    # durations tile the time from the first decision to the end of the episode
    assert math.isclose(
        sum(t.duration for t in transitions),
        random_episode.summary["end_time"] - transitions[0].time,
        rel_tol=1e-9,
    )


def test_random_agent_changes_schedule(fcfs_episode, random_episode):
    assert _request_outcomes(fcfs_episode.requests) != _request_outcomes(
        random_episode.requests
    )


def test_episodes_are_reproducible(env, random_episode):
    rerun = env.run_episode(RandomAgent(seed=0), workload_seed=WORKLOAD_SEED)
    assert rerun.summary == random_episode.summary
    assert _request_outcomes(rerun.requests) == _request_outcomes(
        random_episode.requests
    )

    # same workload, different agent samples: same arrivals, different outcome
    other_agent = env.run_episode(RandomAgent(seed=1), workload_seed=WORKLOAD_SEED)
    assert [r.arrived_at for r in other_agent.requests] == [
        r.arrived_at for r in random_episode.requests
    ]
    assert other_agent.summary != random_episode.summary

    # different workload seed: different arrivals
    other_workload = env.run_episode(FCFSAgent(), workload_seed=WORKLOAD_SEED + 1)
    assert [r.arrived_at for r in other_workload.requests] != [
        r.arrived_at for r in random_episode.requests
    ]


def test_rankings_that_admit_nothing_are_undone(env):
    import numpy as np

    from vidur.scheduler.replica_scheduler.priority_policy import BasePriorityPolicy

    class CheckingPolicy(BasePriorityPolicy):
        # random rankings; after each one that admits nothing, the queue must be
        # back in its pre-ranking order (plus any preempted requests in front)
        def __init__(self):
            self.rng = np.random.default_rng(0)
            self.num_undone = 0

        def get_priorities(self, waiting_requests, scheduler):
            self.scheduler = scheduler
            self.ids_before = [r.id for r in waiting_requests]
            return self.rng.integers(
                0, scheduler.num_priority_levels, len(waiting_requests)
            ).tolist()

        def on_batch_formed(self, admitted_requests):
            if admitted_requests:
                return
            ids_after = [r.id for r in self.scheduler.waiting_requests]
            assert ids_after[len(ids_after) - len(self.ids_before) :] == self.ids_before
            self.num_undone += 1

    config = copy.copy(env.config)
    config.request_generator_config = copy.deepcopy(env.config.request_generator_config)
    _set_workload_seed(config.request_generator_config, WORKLOAD_SEED)

    _reset_id_counters()
    simulator = RLSimulator(
        config, env.execution_time_predictor, RewardTracker(RewardConfig())
    )
    policy = CheckingPolicy()
    for replica_scheduler in simulator.replica_schedulers:
        replica_scheduler.set_priority_policy(policy)
    simulator.run()

    assert policy.num_undone > 0
