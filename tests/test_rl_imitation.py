"""
Checks for imitation learning from an oracle teacher (vidur/rl/teacher.py,
imitation.py) and PPO fine-tuning from it.

Run from the repo root (so ./data and ./cache resolve):
    python -m pytest tests/test_rl_imitation.py -v
"""

import json

import numpy as np
import pytest
import torch

from vidur.rl import ObservationConfig, RewardConfig
from vidur.rl.env import Transition
from vidur.rl.evaluate import evaluate
from vidur.rl.imitation import ImitationConfig, ImitationTrainer
from vidur.rl.model import PriorityDistribution, SchedulerModel, config_for
from vidur.rl.observation import Observation
from vidur.rl.ppo import PPOConfig, RunningStd, ppo_update
from vidur.rl.rollout import RolloutCollector, RolloutTask, WorkloadConfig
from vidur.rl.teacher import (
    ACTED_BY_TEACHER,
    TEACHER_LEVEL,
    TEACHER_RANK,
    ImitationAgent,
    agreement_metrics,
    imitation_losses,
    make_teacher,
)
from vidur.rl.train import Trainer, load_config
from vidur.scheduler.replica_scheduler.priority_policy import job_size

from test_rl_components import OutputLengthGuard

TEACHER = "sjf_total_tokens_age800"
WORKLOAD = WorkloadConfig("tiny", qps=10, num_blocks=4000, num_requests=80)
TINY = "workloads=[{name: tiny, qps: 10, num_blocks: 4000, num_requests: 60}]"


@pytest.fixture(scope="module")
def collector(tmp_path_factory):
    observation_config = ObservationConfig()
    return RolloutCollector(
        [WORKLOAD],
        RewardConfig(
            weights={"e2e_normalized": 1.0, "e2e_hinge": 0.3},
            threshold_mode="reference_window",
            calibration_quantile=0.95,
            normalize_by_baselines=("fcfs",),
        ),
        observation_config,
        config_for(observation_config, 16),
        str(tmp_path_factory.mktemp("sim")),
    )


def test_prefill_tokens_job_size_needs_no_output_length(collector):
    env = collector.env("tiny")
    episode = env.reference_episode("fcfs", 0)
    for request in episode.requests[:20]:
        guarded = OutputLengthGuard(request)
        assert job_size(guarded, "prefill_tokens", 2048) == request.num_prefill_tokens
    assert np.isfinite(env.reference_episode("sjf_prefill_tokens_age800", 0).cost)


def test_teacher_only_agent_is_the_bucketed_teacher(collector):
    env = collector.env("tiny")
    episode = env.run_episode(ImitationAgent(make_teacher(TEACHER), None, 1.0), 0)
    reference = env.reference_episode(f"{TEACHER}_16lvl", 0)
    assert [t.action.tolist() for t in episode.transitions] == [
        t.action.tolist() for t in reference.transitions
    ]
    for t in episode.transitions:
        ranks = t.info[TEACHER_RANK]
        assert sorted(ranks.tolist()) == list(range(t.obs.num_requests))
        assert np.array_equal(t.info[TEACHER_LEVEL], ranks * 16 // len(ranks))
        assert np.array_equal(t.action, t.info[TEACHER_LEVEL])
        assert t.info[ACTED_BY_TEACHER]


def test_student_rollouts_are_labeled(collector):
    model = SchedulerModel(collector._runner.model.config)
    [trajectory] = collector.collect(
        model.state_dict(), [RolloutTask("tiny", 0, 0, 0, "rank", teacher=TEACHER)]
    )
    assert trajectory.transitions
    for t in trajectory.transitions:
        assert not t.info[ACTED_BY_TEACHER]
        assert len(t.info[TEACHER_RANK]) == t.obs.num_requests
    # the untrained rank policy is FCFS, which disagrees with SJF somewhere
    assert agreement_metrics(trajectory.transitions)["pair_agreement"] < 1


def _ordinal(scores):
    scores = torch.tensor(scores, dtype=torch.float32)
    return PriorityDistribution(
        4,
        torch.ones_like(scores, dtype=torch.bool),
        scores=scores,
        cutpoints=torch.tensor([-1.0, 0.0, 1.0]),
    )


def test_imitation_losses():
    targets = {
        "teacher_rank": torch.tensor([[0, 1, 2, 3]]),
        "teacher_level": torch.tensor([[0, 1, 2, 3]]),
    }
    mask = torch.ones(1, 4, dtype=torch.bool)
    agree = imitation_losses(_ordinal([[-3.0, -0.5, 0.5, 3.0]]), targets, mask)
    reverse = imitation_losses(_ordinal([[3.0, 0.5, -0.5, -3.0]]), targets, mask)
    assert agree["pair_accuracy"].item() == 1 and reverse["pair_accuracy"].item() == 0
    assert agree["rank_loss"].item() < np.log(2) < reverse["rank_loss"].item()
    assert agree["level_loss"].item() < reverse["level_loss"].item()
    assert agree["num_pairs"].item() == 6

    # masked requests take no part in either loss
    masked = imitation_losses(
        _ordinal([[-3.0, -0.5, 0.5, -100.0]]),
        targets,
        torch.tensor([[True, True, True, False]]),
    )
    assert masked["num_pairs"].item() == 3 and masked["pair_accuracy"].item() == 1


def _transition(action, teacher_rank, admitted):
    n = len(action)
    obs = Observation(
        waiting_features=np.zeros((n, 1), np.float32),
        running_features=np.zeros((0, 1), np.float32),
        global_features=np.zeros(1, np.float32),
        rank_mask=np.ones(n, bool),
        num_priority_levels=16,
        time=10.0,
        request_ids=list(range(n)),
        running_request_ids=[],
        arrival_times=np.arange(n, dtype=float),
    )
    admitted = np.array(admitted)
    return Transition(
        obs=obs,
        action=np.array(action),
        info={TEACHER_RANK: np.array(teacher_rank)},
        time=10.0,
        num_admitted=int(admitted.sum()),
        admitted=admitted,
        boundary_index=int(np.flatnonzero(~admitted)[0]),
    )


def test_agreement_metrics():
    same = _transition([3, 0, 2, 1], [3, 0, 2, 1], [False, True, False, True])
    assert agreement_metrics([same]) == {
        "pair_agreement": 1.0,
        "admitted_overlap": 1.0,
        "head_agreement": 1.0,
    }
    # FCFS (equal levels) against the reverse order
    fcfs = _transition([0, 0, 0, 0], [3, 2, 1, 0], [True, True, False, False])
    assert agreement_metrics([fcfs]) == {
        "pair_agreement": 0.0,
        "admitted_overlap": 0.0,
        "head_agreement": 0.0,
    }


def test_beta_schedule():
    config = ImitationConfig(beta_start=0.5, beta_decay=0.5)
    assert [config.beta(k) for k in range(4)] == [1.0, 0.5, 0.25, 0.125]


def test_ppo_imitation_anchor(collector):
    torch.manual_seed(0)
    model = SchedulerModel(collector._runner.model.config)
    tasks = [RolloutTask("tiny", 0, k, k, teacher=TEACHER) for k in range(2)]
    trajectories = collector.collect(model.state_dict(), tasks)
    stats = ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        trajectories,
        PPOConfig(epochs=1, minibatch_size=32, imitation_coef=0.1),
        RunningStd(),
        np.random.default_rng(0),
    )
    assert np.isfinite(stats["imitation_loss"]) and stats["imitation_loss"] > 0
    assert np.isfinite(stats["policy_loss"])


def test_dagger_then_finetune_smoke(tmp_path):
    imitation_dir, ppo_dir = tmp_path / "imitation", tmp_path / "ppo"
    common = [TINY, "num_workers=0", "eval_seeds=[1000000]"]
    config = load_config(
        "configs/rl/imitation_sjf_age800.yaml",
        common
        + [
            "imitation.iterations=2",
            "imitation.initial_seeds=2",
            "imitation.seeds_per_iteration=1",
            f"output_dir={imitation_dir}",
        ],
    )
    trainer = ImitationTrainer(config)
    initial = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    try:
        trainer.train()
    finally:
        trainer.close()
    assert trainer.dataset
    assert all(t.boundary_index is not None for t in trainer.dataset)
    assert any(t.info[ACTED_BY_TEACHER] for t in trainer.dataset)
    changed = [
        k
        for k, v in trainer.model.state_dict().items()
        if not torch.equal(v.cpu(), initial[k].cpu())
    ]
    assert any(k.startswith("score_head") for k in changed)
    assert not any(k.startswith("value_trunk") for k in changed)

    lines = [
        json.loads(x)
        for x in (imitation_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert np.isfinite(lines[0]["teacher/improvement_over_fcfs"])
    last = lines[-1]
    for name in ("level_loss", "rank_loss", "pair_accuracy"):
        assert np.isfinite(last[f"imitation/{name}"])
    assert 0 <= last["eval/rank/pair_agreement"] <= 1

    # the checkpoint works with evaluate and as PPO's starting point
    table = evaluate(str(imitation_dir / "best.pt"), baselines=("fcfs",))
    assert np.isfinite(table["improvement_over_fcfs"]).all()

    config = load_config(
        "configs/rl/imitation_sjf_age800.yaml",
        common
        + [
            "iterations=1",
            "seeds_per_iteration=1",
            "group_size=2",
            "ppo.epochs=1",
            f"init_checkpoint={imitation_dir / 'best.pt'}",
            f"output_dir={ppo_dir}",
        ],
    )
    finetune = Trainer(config)
    best = torch.load(imitation_dir / "best.pt", weights_only=False)["model"]
    try:
        for k, v in finetune.model.state_dict().items():
            assert torch.equal(v.cpu(), best[k].cpu()), k
        finetune.train()
    finally:
        finetune.close()
    last = json.loads((ppo_dir / "metrics.jsonl").read_text().splitlines()[-1])
    assert np.isfinite(last["ppo/imitation_loss"]) and last["ppo/imitation_loss"] > 0
    assert np.isfinite(last["eval/rank/pair_agreement"])


def test_imitation_coef_needs_teacher():
    with pytest.raises(AssertionError):
        load_config(None, [TINY, "ppo.imitation_coef=0.1"])
