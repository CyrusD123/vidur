"""
Checks for the model, agent, PPO and training loop (vidur/rl/model.py,
torch_agent.py, ppo.py, rollout.py, train.py, evaluate.py).

Run from the repo root (so ./data and ./cache resolve):
    python -m pytest tests/test_rl_training.py -v
"""

import json
import pickle

import numpy as np
import pytest
import torch

from vidur.rl import (
    FCFSAgent,
    ObservationConfig,
    RewardConfig,
    VidurSchedulingEnv,
    make_simulation_config,
)
from vidur.rl.evaluate import evaluate
from vidur.rl.model import (
    ENCODERS,
    HEADS,
    ModelConfig,
    ObsBatch,
    SchedulerModel,
    collate,
    config_for,
    full_actions,
    modeled_indices,
)
from vidur.rl.ppo import (
    PPOConfig,
    RunningStd,
    _Sample,
    discounted_returns,
    gae,
    group_advantages,
    minibatches,
    normalize_advantages,
    ppo_update,
    return_at,
)
from vidur.rl.rollout import RolloutCollector, RolloutTask, WorkloadConfig
from vidur.rl.torch_agent import TorchAgent
from vidur.rl.train import Trainer, load_config

WORKLOAD = WorkloadConfig("tiny", qps=10, num_blocks=4000, num_requests=80)


def _random_batch(config: ModelConfig, generator: torch.Generator) -> ObsBatch:
    waiting = torch.randn(2, 7, config.num_waiting_features, generator=generator)
    running = torch.randn(2, 3, config.num_running_features, generator=generator)
    waiting_mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    return ObsBatch(
        waiting,
        waiting_mask,
        running,
        torch.ones(2, 3, dtype=torch.bool),
        torch.randn(2, config.num_global_features, generator=generator),
    )


def _randomize(model: SchedulerModel) -> None:
    # the output layers start at zero; give them weights so outputs vary
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))


@pytest.mark.parametrize("head", HEADS)
def test_interval_log_prob_matches_level_sums(head):
    torch.manual_seed(0)
    config = ModelConfig(5, 4, 3, num_priority_levels=6, head=head)
    model = SchedulerModel(config)
    _randomize(model)
    out = model(_random_batch(config, torch.Generator().manual_seed(0)))
    level_log_probs = out.dist.level_log_probs()
    assert torch.allclose(level_log_probs.exp().sum(-1), torch.ones(2, 7), atol=1e-5)

    for low_value in range(6):
        for high_value in range(low_value, 6):
            low = torch.full((2, 7), low_value)
            high = torch.full((2, 7), high_value)
            expected = torch.logsumexp(
                level_log_probs[..., low_value : high_value + 1], dim=-1
            )
            actual = out.dist.interval_log_prob(low, high)
            assert torch.allclose(actual, expected, atol=1e-4), (low_value, high_value)


@pytest.mark.parametrize("head", HEADS)
def test_sampling_matches_probabilities(head):
    torch.manual_seed(0)
    config = ModelConfig(5, 4, 3, num_priority_levels=5, head=head)
    model = SchedulerModel(config)
    _randomize(model)
    with torch.no_grad():
        out = model(_random_batch(config, torch.Generator().manual_seed(1)))
        generator = torch.Generator().manual_seed(0)
        samples = torch.stack([out.dist.sample(generator) for _ in range(20000)])
    probs = out.dist.level_log_probs().exp()[0, 0]
    frequencies = torch.stack(
        [(samples[:, 0, 0] == k).float().mean() for k in range(5)]
    )
    assert torch.allclose(frequencies, probs, atol=0.015)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_encoder_is_permutation_equivariant(encoder):
    torch.manual_seed(0)
    config = ModelConfig(5, 4, 3, num_priority_levels=6, encoder=encoder)
    model = SchedulerModel(config)
    _randomize(model)
    batch = _random_batch(config, torch.Generator().manual_seed(2))
    perm = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    permuted = ObsBatch(
        batch.waiting[:, perm],
        torch.ones(2, 7, dtype=torch.bool),
        batch.running.flip(1),
        batch.running_mask,
        batch.global_features,
    )
    batch.waiting_mask[:] = True
    with torch.no_grad():
        out, out_permuted = model(batch), model(permuted)
    assert torch.allclose(out.dist.scores[:, perm], out_permuted.dist.scores, atol=1e-5)
    assert torch.allclose(out.value, out_permuted.value, atol=1e-4)


def test_truncated_queue_gets_last_level():
    env = VidurSchedulingEnv(
        make_simulation_config(qps=10, num_blocks=4000, num_requests=150)
    )
    episode = env.run_episode(FCFSAgent(), workload_seed=0)
    obs = max((t.obs for t in episode.transitions), key=lambda o: o.num_requests)
    assert obs.num_requests > 10

    modeled = modeled_indices(obs, max_queue_len=10)
    assert len(modeled) == 10
    earliest = np.sort(np.argsort(obs.arrival_times, kind="stable")[:10])
    assert np.array_equal(modeled, earliest)

    action = full_actions(obs, modeled, np.zeros(10, dtype=np.int64))
    rest = np.setdiff1d(np.arange(obs.num_requests), modeled)
    assert (action[modeled] == 0).all()
    assert (action[rest] == obs.num_priority_levels - 1).all()


def test_returns_and_advantages():
    rewards = np.array([-1.0, -2.0, -3.0])
    gammas = np.array([0.5, 0.5, 0.5])
    assert np.allclose(discounted_returns(rewards, gammas), [-2.75, -3.5, -3.0])
    # per-request rewards discount the same way, column by column
    matrix = np.stack([rewards, 2 * rewards], axis=1)
    assert np.allclose(
        discounted_returns(matrix, gammas)[:, 1],
        2 * discounted_returns(rewards, gammas),
    )
    # with a perfect critic, GAE advantages are 0
    values = discounted_returns(rewards, gammas)
    assert np.allclose(gae(rewards, values, gammas, 0.95), 0)

    # identical rollouts: zero group advantages
    times = np.array([0.0, 1.0, 2.0])
    durations = np.array([1.0, 1.0, 1.0])
    rollout = (times, durations, rewards, discounted_returns(rewards, gammas))
    for advantage in group_advantages([rollout, rollout, rollout]):
        assert np.allclose(advantage, 0)
    # the return read off mid-transition loses the accrued part of its reward
    assert np.isclose(return_at(*rollout, 0.5), -2.75 + 0.5)


@pytest.fixture(scope="module")
def collector(tmp_path_factory):
    observation_config = ObservationConfig()
    model_config = config_for(observation_config, 16)
    return RolloutCollector(
        [WORKLOAD],
        RewardConfig(
            weights={"e2e_normalized": 1.0, "e2e_hinge": 0.3},
            threshold_mode="reference_window",
            calibration_quantile=0.95,
            normalize_by_baselines=("fcfs",),
        ),
        observation_config,
        model_config,
        str(tmp_path_factory.mktemp("sim")),
    )


@pytest.mark.parametrize("decoding", ["argmax", "rank"])
def test_initial_deterministic_policy_is_fcfs(collector, decoding):
    model = SchedulerModel(collector._runner.model.config)
    [trajectory] = collector.collect(
        model.state_dict(), [RolloutTask("tiny", 0, 0, 0, decoding)]
    )
    assert trajectory.improvement_over_fcfs == 0
    fcfs = collector.env("tiny").reference_episode("fcfs", 0)
    # identical schedule (total_reward differs: references use a plain reward)
    latency = {k: v for k, v in trajectory.summary.items() if k != "total_reward"}
    assert latency == {k: v for k, v in fcfs.summary.items() if k != "total_reward"}


def test_rank_decoding_follows_expected_level(collector):
    env = collector.env("tiny")
    model = SchedulerModel(collector._runner.model.config)
    _randomize(model)
    episode = env.run_episode(TorchAgent(model, decoding="rank"), workload_seed=0)
    num_levels = model.config.num_priority_levels
    checked = 0
    for t in episode.transitions:
        modeled = modeled_indices(t.obs, model.config.max_queue_len)
        if len(modeled) < 2:
            continue
        with torch.no_grad():
            out = model(collate([t.obs], model.config, [modeled]), with_value=False)
        probs = out.dist.level_log_probs()[0, : len(modeled)].exp()
        expected = (probs * torch.arange(num_levels)).sum(-1).numpy()
        levels = t.action[modeled]
        # higher expected level never gets a lower priority level
        order = np.argsort(expected, kind="stable")
        assert (np.diff(levels[order]) >= 0).all()
        # the ranking is cut into equal groups spanning the levels
        assert levels.min() == 0
        assert levels.max() == (len(modeled) - 1) * num_levels // len(modeled)
        checked += 1
    assert checked > 0


def test_agent_is_reproducible_with_seed(collector):
    env = collector.env("tiny")
    model = SchedulerModel(collector._runner.model.config)
    _randomize(model)
    first = env.run_episode(TorchAgent(model, seed=3), workload_seed=0)
    second = env.run_episode(TorchAgent(model, seed=3), workload_seed=0)
    assert [t.action.tolist() for t in first.transitions] == [
        t.action.tolist() for t in second.transitions
    ]


@pytest.mark.parametrize("advantage", ["group", "gae"])
def test_ppo_update_changes_policy(collector, advantage):
    torch.manual_seed(0)
    model = SchedulerModel(collector._runner.model.config)
    tasks = [
        RolloutTask("tiny", seed, k, 10 * seed + k) for seed in (0, 1) for k in range(2)
    ]
    trajectories = collector.collect(model.state_dict(), tasks)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = ppo_update(
        model,
        optimizer,
        trajectories,
        PPOConfig(advantage=advantage, epochs=2, minibatch_size=32),
        RunningStd(),
        np.random.default_rng(0),
    )
    for name in ("policy_loss", "value_loss", "request_value_loss", "entropy"):
        assert np.isfinite(stats[name]), name
    assert stats["num_policy_samples"] > 0
    changed = [
        k for k, v in model.state_dict().items() if not torch.equal(v, before[k])
    ]
    assert any(k.startswith("score_head") for k in changed)
    assert any(k.startswith("value_trunk") for k in changed)


def test_observation_config_pickles():
    config = ObservationConfig(waiting_features=("wait_time", "num_prefill_tokens"))
    restored = pickle.loads(pickle.dumps(config))
    assert restored.waiting_features == config.waiting_features
    assert restored.num_waiting_features == 2


def test_train_and_evaluate_smoke(tmp_path):
    config = load_config(
        "configs/rl/ppo_default.yaml",
        [
            "iterations=2",
            "seeds_per_iteration=1",
            "group_size=2",
            "num_workers=0",
            "eval_every=2",
            "eval_seeds=[1000000]",
            "workloads=[{name: tiny, qps: 10, num_blocks: 4000, num_requests: 60}]",
            "ppo.epochs=1",
            f"output_dir={tmp_path}",
        ],
    )
    trainer = Trainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert (tmp_path / "best.pt").exists() and (tmp_path / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "iter_0002.pt").exists()
    last = json.loads(lines[-1])
    for decoding in ("rank", "argmax", "sample"):
        assert np.isfinite(last[f"eval/{decoding}/improvement_over_fcfs"])

    table = evaluate(str(tmp_path / "best.pt"), baselines=("fcfs",))
    assert set(table["policy"]) == {"policy_rank", "policy_argmax", "fcfs"}
    assert np.isfinite(table["improvement_over_fcfs"]).all()


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("p", [0.75, 0.9])
def test_initial_concentration(head, p):
    config = ModelConfig(
        5, 4, 3, num_priority_levels=16, head=head, init_middle_level_prob=p
    )
    model = SchedulerModel(config)
    with torch.no_grad():
        out = model(
            _random_batch(config, torch.Generator().manual_seed(0)), with_value=False
        )
    probs = out.dist.level_log_probs().exp()
    assert torch.allclose(probs[..., 7], torch.full_like(probs[..., 7], p), atol=1e-5)
    assert (out.dist.mode() == 7).all()


def test_advantages_normalized_per_workload():
    rng = np.random.default_rng(0)
    samples = [
        _Sample(0, i, np.zeros(0), 0.0, 0.0, np.zeros(0), np.zeros(0))
        for i in range(200)
    ]
    workloads = ["big"] * 100 + ["small"] * 100
    for i, s in enumerate(samples):
        s.advantage = rng.normal(5, 10) if i < 100 else rng.normal(0, 0.1)
    normalize_advantages(samples, np.ones(200, bool), workloads, "workload")
    advantages = np.array([s.advantage for s in samples])
    for part in (advantages[:100], advantages[100:]):
        assert np.isclose(part.mean(), 0, atol=1e-6) and np.isclose(
            part.std(), 1, atol=1e-3
        )


def test_bucketed_minibatches_cover_everything_with_less_padding():
    rng = np.random.default_rng(0)
    lengths = rng.integers(2, 257, 1000).astype(float)
    for bucket in (False, True):
        batches = minibatches(lengths, 64, rng, bucket, jitter=16.0)
        assert sorted(np.concatenate(batches).tolist()) == list(range(1000))
    padding = {
        bucket: np.mean(
            [
                lengths[b].max() - lengths[b].mean()
                for b in minibatches(lengths, 64, rng, bucket, 16.0)
            ]
        )
        for bucket in (False, True)
    }
    assert padding[True] < 0.2 * padding[False]
