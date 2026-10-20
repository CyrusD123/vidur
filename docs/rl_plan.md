# RL Request Scheduler: Implementation Plan

Plan for training a learned queue-ranking policy on top of the RL environment in
`vidur/rl` (commit 854cb77). Sources: the project proposal, our MDP notes, and the
Mutant paper (NSDI '25).

## 1. Starting point

The existing environment already provides:

- **Decision point.** `RLVLLMReplicaScheduler._get_next_batch` re-ranks the whole
  waiting queue before every batch (when it has >= 2 requests). The action is one
  priority per waiting request. Rankings that admit nothing are undone and dropped.
- **Admission is a prefix of the ranking.** `VLLMReplicaScheduler._get_next_batch`
  admits from the head of the ranked queue and stops at the first request that does
  not fit (KV watermark), when the token budget runs out, or at the batch size cap.
  So a ranking affects the system only through the admitted prefix, i.e. which side
  of the first non-admitted request each waiting request lands on.
- **Rewards accumulated over time.** `RewardTracker` adds up each component as
  simulated time passes (e.g. `e2e` = ∫ #requests in system dt). Summed over an
  episode, each component equals its per-request metric summed over requests.
  - Every reward term is already a per-request metric spread over time, so there is
    no per-decode term to exploit by favoring short decodes.
  - The cost of waiting shows up as soon as a request waits, so the reward is dense.
    Delayed effects are carried back with discounting and GAE.
- **Agent interface.** `run_episode(agent, seed)` stores the `info` dict that
  `agent.act(obs)` returns (`log_prob`, `value`) on each transition, which is what
  PPO needs.

### Throughput is not a reward

The workload per episode is fixed, so total tokens generated is fixed. With no
discounting, a per-token reward is a constant. With discounting, it rewards
producing tokens earlier, which favors decodes over prefills (the starvation we want
to avoid). By Little's law, ∫ #in-system dt already captures throughput. Report
throughput in evaluation only.

## 2. MDP

### State

| Group | Features | Notes |
|---|---|---|
| Waiting requests (ranked) | prompt tokens, tokens processed, wait time, restarts, KV blocks needed, `is_new`, queue position, relative-history features (below) | one input per request |
| Running requests (context only) | tokens decoded so far, blocks held, time since first token | one input per request; not ranked |
| Global | free-block fraction, #waiting, #running, short/medium/long window mean/min/max of queue length, batch time, free blocks | Mutant-style windows (their ablation, Fig. 13) |

Features must never use `num_decode_tokens` (a real server does not know the output
length). The reward may use it.

**Relative-history features** (replace the raw "previous priority" feature, see §3):

- `times_passed_over`: decisions where the request was waiting and not admitted
- `num_overtaken_by`: requests that arrived later but were admitted first
- `prev_rank_percentile`: where the request sat in the last ranking, from 0 (front)
  to 1 (back)
- `current_priority` only in `arrival_and_periodic` mode, where priorities persist
  and actually affect future scheduling

### Action

- **Bucketed integer priorities** in [0, `num_priority_levels`) (e.g. 16), one per
  waiting request, sampled independently given the shared encoder output. Lower
  runs first. These map directly onto vLLM's `priority` field for deployment.
- **Head: `ordinal` (default) or `categorical`.**
  - `ordinal`: the network outputs one score s_i per request plus L−1 learned,
    increasing cutpoints θ₁ < … < θ_{L−1} shared by all requests, with
    P(a_i ≤ c) = σ(θ_c − s_i). Levels are ordered, so learning that a request
    belongs at 3 also moves probability toward 2 and 4, and the coarsened
    log-probability in §3 is a single logistic term.
  - `categorical`: independent logits per level; treats levels as unrelated.
- The training log-probability is coarsened to the part of the action that changed
  the outcome (§3).
- `tie_break: arrival | previous_order` (default `arrival`). Today ties keep the
  previous ranking's order, not FCFS.
- `decision_mode: every_batch | arrival_and_periodic` with `rerank_interval_s`.
  `every_batch` is the current behavior. `arrival_and_periodic` ranks new arrivals
  on arrival and re-ranks everything every Δt (closer to the proposal and to a vLLM
  deployment).

### Reward

reward = −Σ weight · component, every component accumulated over time:

| Component | Integrand | Episode sum | Status |
|---|---|---|---|
| `e2e` | #in system | Σ e2e | exists |
| `e2e_normalized` | Σ 1/output_len over in-system | Σ e2e / output_len | exists (default) |
| `ttft` | #without first token | Σ TTFT | exists |
| `e2e_squared` | Σ 2·age | Σ e2e² | exists |
| `tbt_violation` | per token: max(0, gap − `tbt_slo`) | Σ TBT excess | exists |
| `ttft_hinge` | #without first token and age > τ_ttft | Σ max(0, TTFT − τ_ttft) | new |
| `e2e_hinge` | #in system with age > τ_e2e | Σ max(0, e2e − τ_e2e) | new |
| `starvation` | steeper hinge on queue wait | fairness / starvation | new |
| `batch_time` | 1 while a batch runs | per-decode shaping, default weight 0 | new |

- Hinge components are exact if threshold crossings are tracked with a heap.
- Mean latency is already in the reward: the episode sum of `e2e_normalized` is
  N × mean normalized latency, and `e2e`/`ttft` give plain means. The hinges are
  the tail terms.
- Components are stored per transition, so old rollouts can be reweighted without
  re-simulating. `episode_cost_components()` also rescores a finished episode
  under any thresholds and weights in closed form, from request timings alone.

**Thresholds** (`RewardConfig.threshold_mode`, implemented). A threshold set to a
number is a fixed, user-defined SLO and is used as given in every mode (e.g.
`ttft_threshold=2.0`). Fixed SLOs and calibrated thresholds can be mixed, e.g.
TTFT against a 2 s SLO and e2e against FCFS's local tail. Thresholds left as
`None` are filled from FCFS runs at `calibration_quantile`:
- `fixed`: one value per threshold, from FCFS episodes on `calibration_seeds`.
- `reference_window` (recommended): one value per request. θ_i is the p95 of
  FCFS's latency over the `reference_window_requests` (default 50) requests that
  arrived closest to i, from FCFS's run of the same workload. In effect: "no
  worse than FCFS's tail was at this moment".
  - A single episode-wide threshold is too loose early in an episode, when
    queues are short, and too strict late, once backlog builds. A 300-request
    episode is mostly that transient.
  - Each threshold is fixed at arrival, so the hinge integrals stay exact. A
    threshold that moved while a request was in the system would break that.
  - It uses a window of FCFS arrivals, not "FCFS p95 so far". At time t, FCFS has
    only finished the requests that happened to be short, so a p95 over those is
    biased low.
  - `tbt_slo` (for calculating `tbt_violation`) stays a single value.
  - The agent doesn't observe θ_i. Load features such as the sliding-window
    queue length stand in for it.
- Reference runs are cached per workload (`VidurSchedulingEnv.reference_episode`).
  FCFS and SJF don't depend on the agent, so each costs one episode per seed,
  not one per rollout.

**Normalization** (`RewardConfig.normalize_by_baselines`, implemented). Each
reward is divided by a per-workload cost computed under the same reward, so
returns share one scale across seeds and loads. Without it, the FCFS→SJF gap is
~1 s at ample memory and ~6 s at 4000 blocks.
- `("fcfs",)` (recommended): divide by FCFS's cost.
  `Episode.normalized_score` = 1 − cost/cost_FCFS, the improvement over FCFS.
- `("fcfs", "sjf_decode_tokens")`: divide by the FCFS–SJF gap; the score is the
  fraction of that gap closed. This breaks under tail-weighted rewards, where
  SJF can cost more than FCFS (weight check below). It's kept for mean-only
  rewards.
- Subtracting a baseline's cost instead of dividing would change nothing. It's
  action-independent (a baseline), so the optimal policy and the expected
  gradient stay the same, and the per-seed group baseline already removes it.
- What these modes change in training: a single positive scale per workload
  leaves the optimal policy for that workload unchanged. It only reweights seeds
  in the gradient (which per-group advantage normalization already does), puts
  critic targets on one scale across loads, and makes runs comparable. They are
  mostly for reporting and critic stability, not for shaping the policy.

**Possible extension: per-component normalization (not implemented).** Instead
of one divisor for the whole cost, divide each component by its own
per-workload gap between FCFS and the reference SJF:

  reward ∝ −( M / |M_FCFS − M_SJF| + w · T / |T_FCFS − T_SJF| )

where M and T are the mean and tail terms. How it differs from the pair mode
`("fcfs", "sjf_decode_tokens")`:
- **Nothing cancels.** The pair mode divides the combined cost by one gap,
  (M_FCFS − M_SJF) + w·(T_FCFS − T_SJF). The mean gap is positive and the tail
  gap negative, so that combined gap shrinks toward zero and flips sign as w
  grows, right where the reward gets interesting. Per-component gaps can't
  cancel.
- **No component labels needed.** Every component is divided by its own
  |SJF − FCFS| in absolute value. Components where SJF ≈ FCFS (`batch_time`,
  maybe `tbt_violation`) need the same floor as now.
- **w becomes a fixed exchange rate.** SJF scores +1 on each mean term and −1 on
  each tail term relative to FCFS, so w = 1 always makes SJF tie FCFS. The
  weight then means the same thing at every load, seed and choice of mean term.

Unlike the single-divisor modes, this **changes what the policy learns**. The
mean/tail trade-off is now set per workload by how much reordering can gain
there, so the policy becomes sensitive to the workload's characteristics:
- When KV memory is plentiful and SJF barely improves the mean (qps 10 with the
  default cache: mean e2e 23.2 → 22.2 s), the mean gap is small. Each second of
  mean latency then counts for more relative to the tail, but there is little
  to gain by reordering, so the policy is effectively told to leave the tail
  close to FCFS's.
- Under memory pressure, where SJF gains a lot (20.4 → 14.5 s at 4000 blocks),
  the same w tolerates proportionally more tail damage in exchange.
- With raw weights, by contrast, the same w = 2 makes FCFS the best baseline at
  qps 10 / 4000 blocks but aging SJF the best at qps 20. A policy trained on
  both loads would be pushed toward FCFS under one and toward reordering under
  the other.

Offline evidence (`--weight-check` costs, `e2e_hinge`, 50-request windows): the
range of tail weights where aging SJF beats both FCFS and plain SJF at both
loads.

| normalization | mean = `e2e_normalized` | mean = `e2e` |
|---|---|---|
| whole cost ÷ FCFS cost (current recommendation) | 0.3–1 | 10–100 |
| mean ÷ gap, tail ÷ FCFS tail cost | 0.3–2 | 0.3–3 |
| mean ÷ gap, tail ÷ gap (per-component) | 1–30 | 0.3–10 |

Dividing tail terms by FCFS's own tail cost instead of the gap was also tried
and is worse. FCFS's tail cost is small by construction, since thresholds come
from FCFS's own p95. It varies 29% across seeds, versus 15–20% for the gaps. For
TTFT with local windows it is nearly zero.

Trade-offs:
- It departs from absolute units. If the final evaluation is against absolute
  SLOs (e.g. p99 TTFT < 2 s), raw weights tuned for the deployment load are the
  more honest objective. Per-component normalization suits training one policy
  across many loads.
- The reference SJF has to beat FCFS on each mean term: `sjf_decode_tokens` for
  `e2e_normalized`, `sjf_total_tokens` for `e2e`.

How to explore it later, once PPO runs (steps 4–6):
1. Implement it as a mode, e.g. `normalize_by_baselines=("fcfs", ref)` with
   `normalize_per_component=True`. `RewardContext.scale` becomes a dict of
   per-component scales, applied in `AgentPriorityPolicy._credit_reward` and
   `per_request_rewards`, and the weight check gains a matching scoring path.
   About w ≈ 3 is a reasonable start.
2. Train two policies on a mix of loads (e.g. qps 10 with 4000 blocks, qps 10
   with the default cache, qps 20): one with the current recommendation, one
   per-component.
3. Evaluate each load separately on held-out seeds, both on improvement over
   FCFS and on absolute mean and p99 latency. Per-component should give a more
   uniform trade-off across loads. Whether it wins in absolute terms at any
   given load is the open question.

**Recommended start** (from the weight check below):
`weights={"e2e_normalized": 1, "e2e_hinge": 0.3}`,
`threshold_mode="reference_window"`, `calibration_quantile=0.95`,
`normalize_by_baselines=("fcfs",)`. Add `tbt_violation` once a TBT SLO is
chosen. Avoid `ttft_hinge` with a local window (see below).

### Reward weight check (implemented)

`python -m vidur.rl.baselines --weight-check` runs each baseline once per seed
and rescores the episodes offline under a grid of rewards. Each reward is a mean
term plus `tail_weight` × a tail hinge, with thresholds from FCFS. The check asks
two questions:
- **Would the reward push the policy to copy FCFS?** Yes if FCFS scores best.
  A hinge only charges for being worse than FCFS's tail, never for differing from
  FCFS. But FCFS is close to optimal against its own p95, so a large enough tail
  weight always makes FCFS best.
- **Is the tail term too weak to matter?** Yes if plain SJF scores best.

You want a weight where a less extreme ordering wins. The test candidate is SJF
with aging, `sjf_<key>_age<rate>`: each job size is reduced by rate × seconds
waited.

Cost vs FCFS (1 = FCFS) under `e2e_normalized` + `e2e_hinge` with 50-request
windows, seeds 0–4:

| baseline | mean term, qps 10 / 4000 blocks | tail term, qps 10 / 4000 blocks | mean term, qps 20 | tail term, qps 20 |
|---|---|---|---|---|
| random | 0.92 | 3.3 | 0.99 | 1.1 |
| SJF total_tokens | 0.78 | 6.0 | 0.91 | 1.7 |
| SJF decode_tokens | 0.56 | 16.0 | 0.79 | 3.9 |
| SJF total_tokens, aging 400 tok/s | 0.87 | **1.13** | 0.91 | **1.05** |
| SJF decode_tokens, aging 80 tok/s | 0.84 | 4.2 | 0.91 | 1.5 |

Improvement over FCFS (1 − cost/cost_FCFS; higher is better) as the tail weight
varies:

| tail weight | 0 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|
| SJF decode_tokens, qps 10 / 4000 blocks | **0.44** | −0.93 | −3.03 | −7.10 | −10.98 |
| SJF total_tokens aging 400, qps 10 / 4000 blocks | 0.13 | **0.11** | **0.09** | **0.02** | −0.05 (FCFS best) |
| SJF decode_tokens, qps 20 | **0.21** | **0.16** | 0.06 | −0.24 | −0.83 |
| SJF total_tokens aging 400, qps 20 | 0.09 | 0.08 | **0.08** | **0.07** | **0.04** |

Findings:
- **There is a working band.** With `e2e_normalized` as the mean term, a tail
  weight of about 0.3–1 makes aging SJF the best baseline at both loads, ahead of
  both FCFS and plain SJF. 0.3 is the safer choice: at 1, the qps 10 / 4000-block
  setting is close to FCFS being best. With `e2e` as the mean term the band is
  about 3–30.
- **Above the band, the reward does push toward copying FCFS.** At qps 10 / 4000
  blocks, FCFS is best for tail weights ≥ 3. The weight needed depends on load,
  so re-run the check whenever the training workload changes.
- **A real gain exists without hurting the tail.** Aging SJF (total_tokens, 400
  tok/s) gets 9–13% lower mean cost with a tail only 5–13% above FCFS. That's a
  concrete target that a learned policy should at least match.
- **Per-time windows are stricter than one episode-wide threshold.** SJF's tail
  cost relative to FCFS is 16× with windows vs 11× episode-wide. The windows catch
  starvation early in the episode that a single threshold misses.
- **Don't use `ttft_hinge` with a local window.** Under FCFS, TTFT within a local
  window is nearly constant. So FCFS almost never exceeds its own windowed p95,
  while every reordering does: tail costs of 1,000–9,500× FCFS. That term
  effectively says "be FCFS", and FCFS wins from a tail weight of about 0.03–0.1.
  If you want a TTFT tail term, use an episode-wide threshold, a higher quantile,
  or `starvation`.
- **"Gap closed" is unstable under tail-weighted rewards.** Under those rewards
  SJF often costs more than FCFS, which is why FCFS-only normalization is
  recommended.

Caveats:
- Baselines aren't the policy class. A learned policy can reach points that
  none of them do, so the check only bounds the weight from both sides.
- One trace (splitwise_conv), 300-request episodes, 5 seeds, two loads.

## 3. Credit assignment within a decision

The proposal (§3.4) suggests putting past priorities in the state to help credit
assignment. That does not do it on its own:

- **Credit comes from (action, advantage) pairs, not the state.** The policy
  gradient is Σ_t A_t ∇ log π(a_t | s_t). The priority each request got is already
  stored in `a_t`. Putting it in a later state helps the critic (and makes the
  state Markov when priorities persist), but it does not change which action
  receives which reward.
- **The reward is one scalar per decision, not one per request.** Each transition's
  reward sums the cost of every request in the system over the interval. With
  independent per-request categoricals, every request's log-prob is multiplied by
  the same advantage, mostly driven by other requests. The signal-to-noise ratio
  drops roughly as 1/queue length.
- **Most of the priority vector has no effect.** Only the admitted prefix (and the
  first request left out) matters. In `every_batch` mode with arrival tie-breaks, the priorities
  of requests deeper in the queue are discarded next iteration, so their gradient
  terms are pure noise.
- **Absolute levels hide the comparison.** "Priority 3" means "go first" if everyone
  else got 7 and "go last" if everyone got 0. Independent categoricals score each
  level in isolation.

### Fixes

1. **Coarsened log-probability around the admission cutoff.** The queue is sorted
   by key (priority, arrival time), and admission takes a prefix. Let b be the
   first request *not* admitted (the boundary), with key (c, t_b). Given b, the
   outcome depends only on which side of b each request landed, not on its exact
   level:
   - admitted request i: event key_i < key_b, i.e. a_i ≤ c if t_i < t_b, else
     a_i ≤ c − 1
   - non-admitted request i ≠ b: event key_i > key_b, i.e. a_i ≥ c if t_i > t_b,
     else a_i ≥ c + 1
   - the boundary request b: its full log-prob log P(a_b = c)

   The policy loss uses log π = Σ_i log P(event_i), for both the sampling-time and
   update-time probabilities in the PPO ratio. Every request is compared against
   the cutoff set by the others at that moment, which is the comparison we want the
   model to learn, and exact levels that changed nothing contribute no noise. With
   the `ordinal` head each term is one sigmoid, e.g. log σ(θ_c − s_i).
   - If every waiting request was admitted, there is no boundary and the ranking
     had no effect on admission; the transition is skipped in the policy loss (kept
     for the value loss).
   - This is a surrogate: it conditions on the realized boundary rather than
     summing over every action vector that yields the same admitted set.
   - The transition stores the boundary key and an admitted flag per request.
2. **Per-request decomposed critic.** The reward is a sum over requests, so a
   per-request reward stream is free to compute. Use it to train a value head per
   request token, V(s) = Σ_j v_j(s) + v_global(s). The encoder gets per-request
   supervision on which requests end up costly. Do **not** use per-request rewards
   for the policy itself: moving request i up delays others, and a per-request
   objective ignores that (every request would want to go first).
3. **Relative-history features** (§2) instead of raw previous priorities. These say
   how the request fared against others, are meaningful in either decision mode,
   and directly support the starvation penalty.

## 4. Credit assignment across time

§3 handles which priorities *within* a decision mattered. The harder part is
across time: decisions happen every batch (~10–50 ms), but a decision's effects
last seconds (an admitted long prefill slows every token while it runs; a request
left out keeps adding cost until it finishes). Each reward mixes the effects of
hundreds of decisions, and each decision's effect is spread over hundreds of
rewards. Specific risks:

- **Unavoidable background cost.** Each reward sums the cost of every request in
  the system, most of which any policy would pay. The critic must predict that
  background accurately before a single decision's effect stands out.
- **Noise from the policy's own sampling.** An episode's return depends on
  thousands of sampled priority vectors; one decision's effect may be small next
  to that noise.
- **Drain phase.** Arrivals stop after the last request and the queue empties. A
  real server never drains, so the critic cannot (and should not) anticipate it,
  and its errors there leak back through GAE.

### Fork experiment (run before tuning PPO)

The simulator is deterministic and can be forked, so we can measure a decision's
true effect directly instead of guessing whether PPO can find it.

**Feasibility (checked).** `copy.deepcopy` of an `RLSimulator` mid-episode (qps 10,
300 requests, 3000 events in) takes ~4 ms when the execution time predictor is
shared through the memo (`deepcopy(sim, memo={id(predictor): predictor})`). The
fork replays to identical request outcomes. One change is needed:
`VLLMReplicaScheduler._admission_counter` is an `itertools.count`, which cannot be
copied; replace it with an int.

**Procedure.** For a sample of decisions d in rollouts of a given policy:

1. Fork the simulator just before the event that forms the batch, so no call is
   mid-handler. The policy gets a "force the next action" hook for the fork.
2. Run each candidate action at d from a fork:
   - the sampled action
   - FCFS (all priorities equal)
   - a minimal swap: the boundary request is admitted in place of the last
     admitted one
   - a random ranking
3. Continue each fork for a horizon H (seconds of simulated time, or to episode
   end), in two ways:
   - **deterministic continuation** (FCFS, or the policy's most likely action):
     the difference in accumulated cost is the clean effect Δ_d of the decision
   - **K stochastic continuations** with the policy itself, from the same fork and
     first action: their spread σ_d is the noise PPO sees
4. Record cost as a function of time after d for each fork.

**Outputs and what they decide:**

| Measurement | Decides |
|---|---|
| SNR = \|Δ_d\| / σ_d, distribution over decisions | Whether plain PPO + GAE is likely to work, or forked advantages / fewer decisions are needed |
| Time for Δ_d(t) to reach ~90% of its final value | The discount time constant τ (and H for forked advantages) |
| Fraction of decisions with Δ_d ≈ 0 for all alternatives | How many decisions are effectively no-ops; supports `arrival_and_periodic` or skipping decisions where the admitted set equals FCFS's |
| Δ_d vs features of the decision (queue length, KV pressure, prompt size at the boundary) | Which states have consequential decisions; sanity check for features |
| Return spread across rollouts on the same workload seed vs across seeds | How much the group baseline (below) would remove |

Run it first with `RandomAgent` and `FCFSAgent` (before any model exists), then on
checkpoints during training to see whether the problem gets easier or harder as
the policy improves.

Cost: ~4 ms per fork plus the continuation (a full 300-request episode simulates
in ~175 ms; a few seconds of horizon is far less). A few hundred decisions × 4
alternatives × (1 + K) continuations runs in minutes on one machine.

### First results (random agent)

Setup: qps 10, 300 requests, `num_blocks=4000` (memory pressure), 16 levels,
workload seed 0, `RandomAgent`, 8 stochastic continuations, deterministic FCFS
continuation. Alternatives are compared against the sampled action. 100
decisions at a 10 s horizon; 60 decisions running each fork to the end of the
episode (~60 s).

| | 10 s horizon | to episode end |
|---|---|---|
| no-op fraction (fcfs / random / swap) | 0.06 / 0.07 / 0.04 | 0.07 / 0.05 / 0.02 |
| median SNR (fcfs / random / swap) | 0.31 / 0.31 / 0.22 | 0.58 / 0.66 / 0.49 |
| fraction of decisions with SNR > 1 | 0.12–0.15 | 0.29–0.38 |
| time for \|Δ\| to reach 90% (p50 / p90) | 9.1 / 9.7 s (at the horizon) | 28 / 52 s |

Return spread: std of episode return across rollouts on the same seed was 0.7–1.5,
vs 6–12 across workload seeds.

Readings:
- **Almost every decision matters** (≤ 7% no-ops), but **one decision's effect
  is usually smaller than the noise** from the rest of the rollout (median SNR
  0.3–0.7). Plain GAE will be noisy.
- **Effects last tens of seconds.** At 10 s they are still growing; run to the
  end, half take ~28 s to play out. A changed admission shifts the whole later
  schedule. So τ should be long (~20–30 s, about 70–100 decisions at the
  observed ~3.3 decisions/s), or near-undiscounted episode returns.
- **The workload dominates return variance** (between-seed std ~4–16× the
  within-seed std), so a per-seed group baseline (mitigation 1) removes most of
  the variance cheaply. It should be on by default.
- Caveats: one workload seed, one load level, a maximally random agent (σ is an
  upper bound for a trained, lower-entropy policy). Rerun on checkpoints and at
  other QPS levels.
- Cost: the 10 s run (100 decisions × 12 forks) took ~3 min on one core, so
  forked advantages on a small fraction of decisions are affordable.

### Mitigations (enabled based on the experiment)

1. **Group baseline per workload seed (RLOO).** K rollouts per seed; advantages
   relative to the leave-one-out mean return. Removes workload variance exactly,
   which the critic only approximates. Cheap; available as an option alongside
   GAE from the start.
2. **Forked advantages on sampled decisions.** On a fraction of decisions, use the
   measured Δ_d (vs FCFS or the swap) as the advantage, or as an extra target for
   the critic. Only possible in simulation; the main credit-assignment advantage
   of training in Vidur.
3. **Fewer, more meaningful decisions.** `arrival_and_periodic` mode, or consult
   the policy only when its ranking would change the admitted set relative to FCFS.
4. **Discount time constant** τ ≈ the measured effect horizon (likely about a
   request lifetime, seconds), with λ tuned alongside it.
5. **Drain phase.** Either treat episodes as truncated at the last arrival
   (bootstrap from the critic instead of using the drain cost), or run longer
   episodes and drop tail transitions from the loss.

## 5. Network architecture

Train the encoder **together with** the policy. Mutant pre-trains its GRU encoder
separately because its learner is LinUCB, a linear bandit (reward = θᵀz) that
cannot backpropagate into an encoder. With PPO the encoder is the first layers of
the policy network.

- GRU does not fit the queue: Mutant's GRU runs over a time series. The queue is a
  set, and a GRU would make the output depend on input order.
- PCA does not fit either: it needs fixed-length input, and each request has only
  ~8–10 features. The hard part is variable queue length, not dimension.

Default: a small transformer over request inputs (~50–100K parameters).

```
waiting-request inputs ─┐
running-request inputs ─┼─ per-type MLP → d=64 ─→ 2× TransformerEncoder (4 heads, padding mask)
global/history input  ──┘                              │
                        waiting outputs → MLP → score s_i (ordinal) or logits[levels]
                        all request outputs → MLP → v_j      global output → v_global
```

- Encoder registry: `transformer` (default), `deepsets`, `mlp_pool` (baseline),
  later `gru_temporal` (a GRU over time; requires recurrent PPO).
- `shared_trunk: bool` for whether policy and value share the encoder.
- `max_queue_len` truncates the queue tail (kept in FCFS order).

## 6. Implementation steps

Steps 0–9 are implemented (notes at the end of this section). Not yet done:
traces other than splitwise_conv in the workload mix, the Sarathi-Serve/VTC
comparisons, model export for vLLM, and the §4 mitigations beyond the group
baseline.

**0. Dependencies.** `requirements-rl.txt` (`torch>=2.9`, `pyyaml`, `pytest`).
Torch publishes Python 3.14 wheels (2.9–2.14), so the existing venv works.

**1. Scheduler** (`rl_vllm_replica_scheduler.py`, `RlVllmSchedulerConfig`)
- `tie_break` (default `arrival`), `decision_mode`, `rerank_interval_s`.
- `on_batch_formed(admitted_requests, boundary_request)`.
- Per-request `RequestHistory` and sliding-window system stats.
- `VLLMReplicaScheduler._admission_counter` is now an int, so simulators can be
  deep-copied.

**2. Feature registry** (`vidur/rl/features.py`, `observation.py`)

```python
@waiting_feature("wait_time", transform="log1p")
def wait_time(ctx):
    return [ctx.time - r.arrived_at for r in ctx.waiting_requests]
```

Features are computed per feature over all requests at once (one list
comprehension each), which keeps the full default observation at ~0.7 ms per
decision. `ObservationConfig(waiting_features=..., running_features=...,
global_features=...)` selects by name.

**3. Reward** (`reward.py`): hinge components, a per-token log, offline
`per_request_reward_components()` for the decomposed critic, and threshold
calibration (`RewardConfig(calibration_quantile=0.95)` fills thresholds left as
None from FCFS episodes).

**3b. Fork experiment** (`vidur/rl/fork.py`, `python -m vidur.rl.fork`):
`RLSimulator.run(stop_at_event=..., until_time=...)`, `RLSimulator.fork()`,
`ForcedActionPolicy`, `run_fork_experiment`, `decision_table`,
`summarize_decisions`, `return_spread`.

**4. Model** (`vidur/rl/model.py`, implemented)
- `ModelConfig`, `SchedulerModel`: one token per waiting request, per running
  request, and one for global/history features, each from a per-type MLP.
- Encoders: `transformer` (default), `deepsets`, `mlp_pool`.
- Heads: `ordinal` (score + increasing cutpoints) or `categorical`.
  `PriorityDistribution` gives interval log-probs log P(low ≤ a ≤ high),
  sampling, mode and entropy.
- The ordinal interval log-prob is computed stably as
  log σ(u) + log σ(−l) + log(1 − e^(l−u)).
- Size: about 180K parameters (88K policy, 92K value) at d=64 with 2 layers.
- `shared_trunk` defaults to **false**. The value gradients are about 15× the
  policy's, and in a shared encoder they would dominate.
- The value and score output layers start at zero. The untrained deterministic
  policy then puts every request on the same level, which is exactly FCFS, so
  training starts from stock vLLM.
- `max_queue_len` keeps the earliest arrivals. The rest get the last level, so
  they stay in FCFS order behind, and they get no gradient.

**5. `TorchAgent(BaseAgent)`** (`vidur/rl/torch_agent.py`, implemented)
- One forward pass per decision, policy only: about 1 ms on CPU, about 1.5 s per
  300-request episode.
- Sampling uses a private `torch.Generator`. `deterministic=True` takes each
  request's most likely level.
- It doesn't store log-probs. The coarsened log-prob depends on what was
  admitted, so PPO recomputes it from the rollout weights before the first
  update. That's exact for synchronous PPO, where workers act with the same
  weights the learner starts from.

**6. PPO** (`vidur/rl/ppo.py`, implemented)
- Continuous-time discount exp(−duration/τ), with `tau_s` = 25 from the fork
  experiment.
- `advantage: group` (default) is a time-aligned leave-one-out baseline over the
  `group_size` rollouts of the same workload seed. A decision at time t is
  compared against the other rollouts' mean discounted return-to-go from t.
  `gae` uses the critic instead.
- Clipped objective on the coarsened log-prob, summed over the requests the
  decision ranked. Transitions where everything was admitted have log-prob 0
  and drop out.
- Entropy bonus, approximate-KL early stopping (`target_kl`), and returns
  divided by a running std.
- Decomposed critic: each per-request value v_j is fit to that request's own
  discounted return, and v_global to the remainder of the total return.
  Fitting V = Σ v_j + v_global to the total directly multiplies its gradient by
  the token count, giving gradient norms around 400.
- Gradients are clipped separately for the policy and value parameter groups.

**7. Rollouts and training** (`vidur/rl/rollout.py`, `train.py`, implemented)
- A spawned `multiprocessing` pool. Each worker builds one env per workload, once.
  Each (workload, seed) group goes to a single worker, so its FCFS reference runs
  once.
- Workers return `Trajectory` objects: transitions, per-request rewards,
  improvement over FCFS, mean gap closed and summary.
- `WorkloadConfig` mix sampled by weight. Training seeds are below 10⁶; eval
  seeds are 10⁶ and above.
- `python -m vidur.rl.train --config configs/rl/ppo_default.yaml [--set key=value]`.
  One YAML covers observation, reward, model, PPO and workloads.
- Logs go to `<output_dir>/metrics.jsonl`, plus wandb with `wandb: true`.
  Metrics are overall and per workload: return, improvement over FCFS, mean gap
  closed, latency tails, decision time, and the PPO stats.
- `best.pt` (by eval improvement over FCFS) and `latest.pt` are saved every
  `eval_every` iterations.
- Still to do: add ShareGPT/Azure traces to the mix (`simulation.trace_file`
  per workload).

**8. Evaluation** (`vidur/rl/evaluate.py`): held-out seeds and QPS levels; compare
against vLLM FCFS, Sarathi-Serve, VTC; report TTFT, TBT, e2e (mean and tails),
throughput, fairness, and policy inference time per decision. Export via
TorchScript/ONNX for the vLLM deployment.

*Implemented:* `python -m vidur.rl.evaluate --checkpoint run/best.pt [--workload
'{name: ..., qps: ...}'] [--seeds ...] [--sampled]`. It runs the deterministic
(and optionally sampled) policy and the baselines (FCFS, SJF variants, aging SJF)
on held-out seeds. For each workload it reports:
- Improvement over FCFS under the training reward.
- Mean gap closed.
- Mean/p99 e2e and TTFT, and p99 TBT.
- Milliseconds per decision.

Still to do: Sarathi-Serve/VTC comparisons and export.

*Mean latency vs oracle SJF (implemented).* `env.mean_gap_closed(episode,
workload_seed, mean_term)` reports how much of SJF's mean-latency gain over FCFS
a policy achieves on the same workload:

  mean gap closed = (M_FCFS − M_policy) / (M_FCFS − M_SJF)

- M is a single mean component's episode cost (`e2e_normalized`, `e2e` or
  `ttft`), from `episode_cost_components` on the policy's episode and the cached
  FCFS and SJF reference runs.
- The reference SJF per component is in `MEAN_COMPONENT_REFERENCES`:
  `sjf_decode_tokens` for `e2e_normalized`, `sjf_total_tokens` for `e2e` and
  `ttft`. Each is the variant that does best on that metric.
- 0 = FCFS, 1 = oracle SJF. Values above 1 are possible and meaningful: SJF is
  optimal only for one server running one job at a time, not under continuous
  batching (the 16-level bucketed SJF already slightly beats exact SJF on mean
  e2e at qps 20).
- It's an evaluation metric, not a reward term. SJF terms in the reward would
  either do nothing (subtracting an action-independent reference is a baseline)
  or reweight requests by local congestion (dividing by an SJF window turns
  latency into slowdown vs the oracle, discounting the busy periods where
  scheduling matters most).
- Tails are reported separately, against FCFS (p95/p99, and the FCFS-threshold
  hinge costs), never against SJF.

Report the pair "mean gap closed" and "tail cost vs FCFS" per load, so each
policy reads as a point on the mean/tail trade-off. For reference, aging SJF
(total_tokens, 400 tok/s) sits at a 13% lower mean cost with a tail 1.13× FCFS's
(qps 10 / 4000 blocks).

*Oracle SJF baselines (implemented).* `OracleSJFAgent(key, num_levels)` in
`vidur/rl/agents.py` ranks by true job size. It is an `OracleAgent`, so the env
hands it the real `Request` objects, and nothing is added to the observation.
The same ordering is the `sjf` policy in `priority_policy.py` for plain Vidur runs.
The job size `key` is one of:
- `total_tokens`: prompt plus output tokens.
- `decode_tokens`: output tokens still to generate.
- `iterations`: prefill chunks plus output tokens.

`aging=r` subtracts r × seconds waited from the job size (see the weight check in
§2). `baseline_agent(name)` builds any baseline by name, e.g. `fcfs`, `random`,
`sjf_total_tokens_age400` or `sjf_decode_tokens_16lvl`.

With `num_levels=None` the queue is sorted exactly, using `unbounded_priorities`.
With `num_levels=L` it is cut into L equal groups, which is a ranking the agent
could express itself. `python -m vidur.rl.baselines` compares them. Mean of
seeds 0–4 at qps 10 with 4000 KV blocks:

| baseline | reward (e2e_normalized) | mean e2e (s) | p99 e2e (s) | mean TTFT (s) | p99 TTFT (s) |
|---|---|---|---|---|---|
| FCFS | −48.7 | 20.4 | 37.6 | 11.4 | 25.2 |
| random | −44.8 | 18.3 | 42.9 | 8.5 | 35.7 |
| SJF total_tokens | −37.6 | **14.5** | 45.0 | **5.8** | 37.8 |
| SJF decode_tokens | **−27.2** | 18.1 | 52.0 | 9.2 | 42.5 |
| SJF decode_tokens, 16 levels | −28.8 | 18.3 | 52.4 | 9.2 | 42.6 |

What this shows:
- SJF is a strong reference, not an upper bound. SJF is optimal only for a single
  server that runs one job at a time. Here many requests decode concurrently, and
  the limits are KV memory and the batch cap. SJF is also non-preemptive at
  admission. With ample KV cache (qps 10, default blocks) admission order barely
  matters: mean e2e drops only 23.2 → 22.2 s.
- Which job size works best depends on the metric. Ranking by `total_tokens`
  also packs memory (small requests need few blocks), so it gives the best mean
  e2e and TTFT. Ranking by `decode_tokens` gives the best normalized-latency
  reward and TBT. `iterations` ≈ `decode_tokens`.
- Every SJF variant makes the tails worse (p99 e2e/TTFT up 20–70% vs FCFS),
  because long requests starve. The tail and starvation reward terms are where
  a learned policy can beat it.
- 16 bucketed levels lose only 1–6% against an exact sort. The action space is
  not what limits the agent.

**9. Tests**
- Feature registry: input sizes derived correctly; no feature reads output length.
- Hinge components sum exactly to their per-request metrics.
- With per-request (window) thresholds, the running tracker matches the
  closed-form `episode_cost_components`, and the per-request split still sums to
  each transition's reward. FCFS normalization scores FCFS 0, and (fcfs, sjf)
  normalization scores SJF 1 (`tests/test_rl_oracle.py`, implemented).
- `mean_gap_closed` scores FCFS 0 and the reference SJF 1 (implemented).
- Ties broken by arrival time; FCFS still matches stock vLLM.
- Coarsened log-prob: on small queues, every action vector consistent with the
  stored events reproduces the same admitted set (and inconsistent ones don't).
- PPO smoke test: a few updates on a tiny workload, finite loss, policy changes.
- Implemented in `tests/test_rl_training.py`:
  - Interval log-probs equal sums of level probabilities, for both heads.
  - Samples match the probabilities.
  - Every encoder is permutation-equivariant.
  - Truncated requests get the last level.
  - Returns, GAE and the group baseline are checked on hand cases.
  - The untrained deterministic policy reproduces FCFS exactly.
  - A seeded agent is reproducible.
  - PPO updates change the policy and value trunks (both advantage modes).
  - Training and evaluation run end to end.

**Order:** 1–3 (environment side), then 3b (fork experiment), then 4–6, then
7–8. The fork results set τ and decide which §4 mitigations are on before PPO
tuning starts. A first end-to-end run is possible after step 6 with the current
placeholder features.

### Implementation notes

- Only ~200 decisions are recorded per 300-request episode at qps 10 (most
  iterations have < 2 waiting requests or admit nothing), not one per batch.
- The admission bounds (§3) assume only the admitted set matters. The order
  *among* admitted requests also has later effects: vLLM schedules running
  requests and picks preemption victims (most recent first) by admission order.
  So the bounds are an approximation even in `every_batch` mode.
- `make_simulation_config` now defaults to 16 priority levels.
  `RlVllmSchedulerConfig` (the CLI default) still uses 8.
- In `arrival_and_periodic` mode every decision is recorded (even if nothing was
  admitted), and the bounds are exact, because priorities persist.
- `mean_gap_closed` is unstable when SJF barely beats FCFS on the mean term,
  e.g. 100-request episodes, where it reached ~40. Use it on full-size
  episodes, or read it alongside improvement over FCFS.
- `ObservationConfig` pickles by feature names, so it can go to spawned workers.

## 7. Open decisions

1. Changing the tie-break default to arrival order changes current behavior (the
   FCFS-matches-vLLM test should still pass).
2. Start with `every_batch` (works today) or `arrival_and_periodic` (closer to the
   proposal and deployment).
3. Number of priority levels (16 suggested; Sid's queue-length measurements should
   inform it) and `ordinal` vs `categorical` head.
4. Starting reward weights. The weight check (§2) suggests `e2e_normalized: 1`,
   `e2e_hinge: 0.3` with 50-request FCFS windows at p95, normalized by FCFS.
   Still open: whether TBT gets a term, and the window size and quantile.
5. SNR threshold below which forked advantages or fewer decisions become
   required. First results (median SNR 0.3–0.7) suggest at least the group
   baseline is required.
6. Per-component reward normalization (§2, not implemented): one policy with a
   consistent mean/tail trade-off across loads, vs. absolute-unit weights tuned
   for the deployment load. Decide after the first multi-load PPO runs.
