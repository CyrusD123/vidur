"""
Checks for forking simulations and the fork experiment (vidur/rl/fork.py).

Run from the repo root (so ./data and ./cache resolve):
    python -m pytest tests/test_rl_fork.py -v
"""

import numpy as np
import pytest

from vidur.rl import FCFSAgent, RandomAgent, VidurSchedulingEnv, make_simulation_config
from vidur.rl.env import AgentPolicy
from vidur.rl.fork import (
    ForkExperimentConfig,
    alternative_action,
    decision_table,
    run_fork_experiment,
    summarize_decisions,
)

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


def _outcomes(simulator):
    return [
        (r.id, r.prefill_completed_at, r.completed_at, r.num_restarts)
        for r in simulator.requests
    ]


def test_fork_continues_identically(env):
    reference = env.make_simulator(WORKLOAD_SEED)
    reference.set_priority_policy(
        AgentPolicy(RandomAgent(seed=0), env.observation_config)
    )
    reference.run()

    simulator = env.make_simulator(WORKLOAD_SEED)
    simulator.set_priority_policy(
        AgentPolicy(RandomAgent(seed=0), env.observation_config)
    )
    simulator.run(stop_at_event=reference.events_handled // 2)

    # the fork copies the agent too (nothing shared), so its RNG continues
    # from the same state as the original's
    fork = simulator.fork()
    fork.run()
    simulator.run()

    assert _outcomes(fork) == _outcomes(reference)
    assert _outcomes(simulator) == _outcomes(reference)
    assert fork.events_handled == simulator.events_handled == reference.events_handled


def test_paused_runs_integrate_reward_exactly(env):
    whole = env.make_simulator(WORKLOAD_SEED)
    whole.run()
    whole_components = whole.reward_tracker.pop()

    pieces = env.make_simulator(WORKLOAD_SEED)
    totals = {name: 0.0 for name in whole_components}
    for until_time in np.arange(0.37, 200, 0.37):
        pieces.run(until_time=until_time)
        for name, value in pieces.reward_tracker.pop().items():
            totals[name] += value
        if pieces.done:
            break
    assert pieces.done

    for name, value in whole_components.items():
        assert np.isclose(totals[name], value, rtol=1e-9), name


def test_fcfs_fork_experiment_has_no_effects(env):
    # FCFS agent, FCFS alternative, FCFS continuation: nothing can differ
    config = ForkExperimentConfig(
        num_decisions=4,
        alternatives=("sampled", "fcfs"),
        horizon_s=1.0,
        num_stochastic=2,
        checkpoint_interval_s=0.1,
    )
    _, records = run_fork_experiment(
        env, lambda seed: FCFSAgent(), WORKLOAD_SEED, config
    )
    table = decision_table(records)
    assert len(records) == 4
    assert table["is_noop"].all()
    assert (table["delta"] == 0).all()
    assert (table["sigma"] == 0).all()


def test_random_fork_experiment(env):
    config = ForkExperimentConfig(
        num_decisions=6, horizon_s=2.0, num_stochastic=3, checkpoint_interval_s=0.1
    )
    episode, records = run_fork_experiment(
        env, lambda seed: RandomAgent(seed), WORKLOAD_SEED, config
    )
    assert len(records) == 6

    for record in records:
        num_checkpoints = len(record.checkpoint_times)
        assert num_checkpoints == 20
        assert set(record.deterministic_costs) == set(config.alternatives)
        assert record.stochastic_costs.shape == (3, num_checkpoints)
        for costs in record.deterministic_costs.values():
            # cost only accumulates
            assert (np.diff(costs) >= -1e-12).all()

        # an alternative identical to the sampled action has exactly no effect
        for name in record.same_as_sampled:
            assert np.array_equal(
                record.deterministic_costs[name],
                record.deterministic_costs["sampled"],
            )

    table = decision_table(records)
    assert (~table["is_noop"]).any()
    assert (table["sigma"] > 0).any()

    summary = summarize_decisions(table)
    assert set(summary) == {"fcfs", "swap", "random"}


def test_swap_puts_boundary_ahead_of_last_admitted(env):
    episode = env.run_episode(RandomAgent(seed=0), workload_seed=WORKLOAD_SEED)
    rng = np.random.default_rng(0)
    num_checked = 0
    for t in episode.transitions:
        b = t.boundary_index
        if b is None:
            continue
        swapped = alternative_action("swap", t, rng)
        assert not np.array_equal(swapped, t.action)
        assert ((0 <= swapped) & (swapped < t.obs.num_priority_levels)).all()

        keys = t.obs.ranking_keys(t.action)
        admitted = np.flatnonzero(t.admitted)
        last = max(admitted, key=lambda i: keys[i])
        new_keys = t.obs.ranking_keys(swapped)
        assert new_keys[b] < new_keys[last]
        num_checked += 1
    assert num_checked > 0
