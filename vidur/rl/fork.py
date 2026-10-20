"""
Fork experiment: measures the true effect of individual scheduling decisions by
forking the simulator just before a decision, applying different actions, and
comparing the cost that accumulates afterwards (docs/rl_plan.md §4).

For each sampled decision d of an episode:
  - every alternative action ("sampled", "fcfs", "swap", "random") is applied in
    its own fork, followed by a deterministic continuation policy (FCFS by
    default); cost differences between alternatives are the decision's clean
    effect Δ_d
  - the sampled action is also followed by K stochastic continuations of the
    agent itself; their spread σ_d is the noise PPO would see

Usage:
    python -m vidur.rl.fork --agent random --num-decisions 50 --horizon 5 \
        --output-dir simulator_output/fork_experiment
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vidur.entities import Request
from vidur.rl.agents import BaseAgent, FCFSAgent, RandomAgent
from vidur.rl.env import (
    AgentPolicy,
    Episode,
    RLSimulator,
    Transition,
    VidurSchedulingEnv,
)
from vidur.rl.reward import RewardConfig
from vidur.scheduler.replica_scheduler.priority_policy import BasePriorityPolicy
from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
    RLVLLMReplicaScheduler,
)

ALTERNATIVES = ("sampled", "fcfs", "swap", "random")

AgentFactory = Callable[[int], BaseAgent]


@dataclass
class ForkExperimentConfig:
    num_decisions: int = 50
    alternatives: Tuple[str, ...] = ALTERNATIVES
    # seconds of simulated time after the decision; None runs to episode end
    horizon_s: Optional[float] = 5.0
    # stochastic continuations of the agent after the sampled action
    num_stochastic: int = 4
    # spacing of the cost-over-time curve
    checkpoint_interval_s: float = 0.05
    # agent seed for the episode; stochastic continuations use seed + 1 + k
    agent_seed: int = 0
    # sampling of decisions and of the "random" alternative
    seed: int = 0

    def __post_init__(self):
        unknown = set(self.alternatives) - set(ALTERNATIVES)
        assert not unknown, f"Unknown alternatives {unknown}"
        assert "sampled" in self.alternatives, "'sampled' is the reference"


@dataclass
class DecisionRecord:
    transition_index: int
    event_index: int
    time: float
    num_waiting: int
    num_running: int
    num_admitted: int
    free_block_fraction: float
    # prompt length of the first request left out (nan if all were admitted)
    boundary_prompt_tokens: float
    # alternatives whose action was identical to the sampled one
    same_as_sampled: List[str]
    # cost-curve sample times, relative to the decision
    checkpoint_times: np.ndarray
    # alternative -> cumulative cost at each checkpoint (deterministic continuation)
    deterministic_costs: Dict[str, np.ndarray] = field(default_factory=dict)
    # [num_stochastic, num_checkpoints] after the sampled action
    stochastic_costs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


class ForcedActionPolicy(BasePriorityPolicy):
    """Returns a fixed action at the first decision, then defers to `then`."""

    def __init__(
        self,
        priorities: Sequence[int],
        expected_request_ids: List[int],
        then: BasePriorityPolicy,
    ) -> None:
        self._priorities = list(priorities)
        self._expected_request_ids = expected_request_ids
        self._then = then
        self._used = False

    def get_priorities(
        self,
        waiting_requests: List[Request],
        scheduler: RLVLLMReplicaScheduler,
    ) -> List[int]:
        if self._used:
            return self._then.get_priorities(waiting_requests, scheduler)

        self._used = True
        request_ids = [r.id for r in waiting_requests]
        assert request_ids == self._expected_request_ids, (
            "The fork did not reach the recorded decision; the replay diverged "
            "(is the agent deterministic given its seed?)"
        )
        return self._priorities


def alternative_action(
    name: str, transition: Transition, rng: np.random.Generator
) -> np.ndarray:
    action = transition.action
    num_levels = transition.obs.num_priority_levels

    if name == "sampled":
        return action.copy()
    if name == "fcfs":
        return np.zeros_like(action)
    if name == "random":
        return rng.integers(0, num_levels, len(action))
    if name == "swap":
        # Put the boundary request (first left out) just ahead of the last
        # admitted one, about the smallest change to the admitted set. Swapping
        # their priorities is not enough when they share a level (common with
        # long queues), since ties go by arrival.
        swapped = action.copy()
        b = transition.boundary_index
        admitted = np.flatnonzero(transition.admitted)
        if b is None or len(admitted) == 0:
            return swapped
        keys = transition.obs.ranking_keys(action)
        last = max(admitted, key=lambda i: keys[i])
        if action[b] > action[last]:
            swapped[last], swapped[b] = action[b], action[last]
        elif action[last] > 0:
            swapped[b] = action[last] - 1
        else:
            swapped[last] = 1
        return swapped
    raise ValueError(f"Unknown alternative {name}")


def _measure_cost(
    simulator: RLSimulator,
    reward_config: RewardConfig,
    start_time: float,
    checkpoint_times: np.ndarray,
) -> np.ndarray:
    """Run `simulator` from start_time; cumulative cost (-reward) at each
    checkpoint (relative to start_time)."""
    tracker = simulator.reward_tracker
    tracker.advance(start_time)
    tracker.pop()  # cost before the decision is the same for every fork

    costs = np.empty(len(checkpoint_times))
    total = 0.0
    for i, offset in enumerate(checkpoint_times):
        simulator.run(until_time=start_time + offset)
        total -= reward_config.combine(tracker.pop())
        costs[i] = total
    return costs


def _checkpoint_times(
    config: ForkExperimentConfig, decision_time: float, episode_end: float
) -> np.ndarray:
    horizon = config.horizon_s
    if horizon is None:
        # past the episode end the cost stops changing; pad so all forks finish
        horizon = 1.5 * (episode_end - decision_time) + config.checkpoint_interval_s
    num = max(1, int(np.ceil(horizon / config.checkpoint_interval_s)))
    return np.linspace(horizon / num, horizon, num)


def run_fork_experiment(
    env: VidurSchedulingEnv,
    agent_factory: AgentFactory,
    workload_seed: int,
    config: Optional[ForkExperimentConfig] = None,
    continuation_factory: Callable[[], BaseAgent] = FCFSAgent,
    share: Sequence[object] = (),
) -> Tuple[Episode, List[DecisionRecord]]:
    """
    agent_factory(seed) must build an agent whose actions are fully determined
    by its seed (it is rebuilt to replay the episode). `share` lists objects the
    forks should share rather than copy, e.g. a torch model.
    """
    config = config or ForkExperimentConfig()
    rng = np.random.default_rng(config.seed)
    reward_config = env.reward_config

    # pass 1: the episode, to pick decisions
    episode = env.run_episode(agent_factory(config.agent_seed), workload_seed)
    transitions = episode.transitions
    num_decisions = min(config.num_decisions, len(transitions))
    chosen = np.sort(rng.choice(len(transitions), num_decisions, replace=False))
    event_indices = [transitions[i].event_index for i in chosen]
    assert len(set(event_indices)) == len(event_indices)
    episode_end = episode.summary["end_time"]

    # pass 2: replay the same episode, forking before each chosen decision
    simulator = env.make_simulator(workload_seed)
    agent = agent_factory(config.agent_seed)
    simulator.set_priority_policy(AgentPolicy(agent, env.observation_config))
    share = [agent, *share]

    records = []
    for transition_index in chosen:
        transition = transitions[transition_index]
        simulator.run(stop_at_event=transition.event_index)
        obs = transition.obs
        checkpoint_times = _checkpoint_times(config, transition.time, episode_end)

        sampled = transition.action
        actions = {
            name: alternative_action(name, transition, rng)
            for name in config.alternatives
        }

        def fork_cost(priorities, then: BasePriorityPolicy) -> np.ndarray:
            fork = simulator.fork(share=share)
            fork.set_priority_policy(
                ForcedActionPolicy(priorities, obs.request_ids, then)
            )
            return _measure_cost(fork, reward_config, transition.time, checkpoint_times)

        deterministic = AgentPolicy(continuation_factory(), env.observation_config)
        deterministic_costs = {
            name: fork_cost(action, deterministic) for name, action in actions.items()
        }
        stochastic_costs = np.array(
            [
                fork_cost(
                    sampled,
                    AgentPolicy(
                        agent_factory(config.agent_seed + 1 + k),
                        env.observation_config,
                    ),
                )
                for k in range(config.num_stochastic)
            ]
        ).reshape(config.num_stochastic, len(checkpoint_times))

        # the paused simulator is in the state the decision observed
        scheduler = simulator.replica_schedulers[0]
        requests_by_id = {r.id: r for r in simulator.requests}
        boundary = transition.boundary_index
        records.append(
            DecisionRecord(
                transition_index=int(transition_index),
                event_index=transition.event_index,
                time=transition.time,
                num_waiting=obs.num_requests,
                num_running=obs.num_running,
                num_admitted=transition.num_admitted,
                free_block_fraction=scheduler.num_free_blocks / scheduler.num_blocks,
                boundary_prompt_tokens=(
                    float(requests_by_id[obs.request_ids[boundary]].num_prefill_tokens)
                    if boundary is not None
                    else float("nan")
                ),
                same_as_sampled=[
                    name
                    for name, action in actions.items()
                    if name != "sampled" and np.array_equal(action, sampled)
                ],
                checkpoint_times=checkpoint_times,
                deterministic_costs=deterministic_costs,
                stochastic_costs=stochastic_costs,
            )
        )

    return episode, records


def decision_table(
    records: List[DecisionRecord],
    reference: str = "sampled",
    effect_fraction: float = 0.9,
    noop_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    One row per (decision, alternative): the clean effect Δ (alternative's cost
    minus the reference's, at the horizon), the noise σ of the reference under
    stochastic continuation, SNR = |Δ|/σ, and the time for |Δ(t)| to first reach
    effect_fraction of its final value.
    """
    rows = []
    for record in records:
        reference_costs = record.deterministic_costs[reference]
        final_stochastic = record.stochastic_costs[:, -1]
        sigma = (
            float(np.std(final_stochastic, ddof=1))
            if len(final_stochastic) > 1
            else float("nan")
        )

        for name, costs in record.deterministic_costs.items():
            if name == reference:
                continue
            delta_curve = costs - reference_costs
            delta = float(delta_curve[-1])
            scale = max(abs(reference_costs[-1]), 1.0)
            is_noop = abs(delta) <= noop_tolerance * scale

            effect_time = float("nan")
            if not is_noop:
                reached = np.abs(delta_curve) >= effect_fraction * abs(delta)
                effect_time = float(record.checkpoint_times[np.argmax(reached)])

            rows.append(
                {
                    "transition_index": record.transition_index,
                    "time": record.time,
                    "alternative": name,
                    "same_action": name in record.same_as_sampled,
                    "num_waiting": record.num_waiting,
                    "num_running": record.num_running,
                    "num_admitted": record.num_admitted,
                    "free_block_fraction": record.free_block_fraction,
                    "boundary_prompt_tokens": record.boundary_prompt_tokens,
                    "reference_cost": float(reference_costs[-1]),
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "sigma": sigma,
                    "snr": (
                        abs(delta) / sigma
                        if sigma and not np.isnan(sigma)
                        else float("nan")
                    ),
                    "is_noop": is_noop,
                    "effect_time": effect_time,
                }
            )
    return pd.DataFrame(rows)


def summarize_decisions(table: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per alternative: SNR quantiles, no-op fraction, effect-time quantiles."""
    summary = {}
    for name, group in table.groupby("alternative"):
        effective = group[~group["is_noop"]]
        snr = effective["snr"].dropna()
        effect_time = effective["effect_time"].dropna()

        def q(series, p):
            return float(series.quantile(p)) if len(series) else float("nan")

        summary[name] = {
            "num_decisions": int(len(group)),
            "noop_fraction": float(group["is_noop"].mean()),
            "mean_abs_delta": float(group["abs_delta"].mean()),
            "median_sigma": float(group["sigma"].median()),
            "snr_p25": q(snr, 0.25),
            "snr_p50": q(snr, 0.5),
            "snr_p75": q(snr, 0.75),
            "fraction_snr_above_1": (
                float((snr > 1).mean()) if len(snr) else float("nan")
            ),
            "effect_time_p50": q(effect_time, 0.5),
            "effect_time_p90": q(effect_time, 0.9),
        }
    return summary


def return_spread(
    env: VidurSchedulingEnv,
    agent_factory: AgentFactory,
    workload_seeds: Sequence[int],
    rollouts_per_seed: int,
) -> Dict[str, float]:
    """
    Spread of episode returns from the agent's own sampling (rollouts on the same
    workload seed) vs from the workload (between seeds). The within-seed part is
    what a per-seed group baseline cannot remove; the between-seed part it can.
    """
    returns = np.array(
        [
            [
                env.run_episode(agent_factory(k), workload_seed=seed).total_reward
                for k in range(rollouts_per_seed)
            ]
            for seed in workload_seeds
        ]
    )
    within = returns.std(axis=1, ddof=1).mean() if rollouts_per_seed > 1 else np.nan
    between = returns.mean(axis=1).std(ddof=1) if len(workload_seeds) > 1 else np.nan
    return {
        "mean_return": float(returns.mean()),
        "within_seed_std": float(within),
        "between_seed_std": float(between),
    }


def _agent_factory(name: str) -> AgentFactory:
    if name == "random":
        return lambda seed: RandomAgent(seed)
    if name == "fcfs":
        return lambda seed: FCFSAgent()
    raise ValueError(f"Unknown agent {name}")


def main() -> None:
    from vidur.rl.env import make_simulation_config

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agent", choices=["random", "fcfs"], default="random")
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--num-requests", type=int, default=300)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--batch-size-cap", type=int, default=128)
    parser.add_argument("--num-priority-levels", type=int, default=16)
    parser.add_argument("--workload-seed", type=int, default=0)
    parser.add_argument("--num-decisions", type=int, default=50)
    parser.add_argument(
        "--horizon", type=float, default=5.0, help="seconds; <= 0 runs to the end"
    )
    parser.add_argument("--num-stochastic", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=float, default=0.05)
    parser.add_argument("--spread-seeds", type=int, default=4)
    parser.add_argument("--spread-rollouts", type=int, default=4)
    parser.add_argument(
        "--output-dir", default="simulator_output/fork_experiment", type=str
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env = VidurSchedulingEnv(
        make_simulation_config(
            qps=args.qps,
            num_requests=args.num_requests,
            num_blocks=args.num_blocks,
            batch_size_cap=args.batch_size_cap,
            num_priority_levels=args.num_priority_levels,
            output_dir=args.output_dir,
        )
    )
    agent_factory = _agent_factory(args.agent)
    config = ForkExperimentConfig(
        num_decisions=args.num_decisions,
        horizon_s=args.horizon if args.horizon > 0 else None,
        num_stochastic=args.num_stochastic,
        checkpoint_interval_s=args.checkpoint_interval,
    )

    episode, records = run_fork_experiment(
        env, agent_factory, args.workload_seed, config
    )
    table = decision_table(records)
    summary = {
        "args": vars(args),
        "episode": episode.summary,
        "alternatives": summarize_decisions(table),
        "return_spread": return_spread(
            env,
            agent_factory,
            range(1, 1 + args.spread_seeds),
            args.spread_rollouts,
        ),
    }

    table.to_csv(os.path.join(args.output_dir, "decisions.csv"), index=False)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["alternatives"], indent=2))
    print(json.dumps(summary["return_spread"], indent=2))


if __name__ == "__main__":
    main()
