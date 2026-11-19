#!/bin/zsh
# Overnight: DAgger imitation of sjf_prefill_tokens_age200, then PPO fine-tuning
# from its best checkpoint (configs/rl/frontier_imitation.yaml). Run from the
# vidur repo root:  nohup caffeinate -is zsh rl_runs/run_frontier_imitation.sh &
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=../Frontier/.venv/bin/python
CONFIG=configs/rl/frontier_imitation.yaml

mkdir -p rl_runs/frontier_imitation rl_runs/frontier_imitation_ppo
echo "imitation start $(date)"
$PY -m vidur.rl.imitation --config $CONFIG > rl_runs/frontier_imitation/train.log 2>&1
echo "imitation done $(date)"
$PY -m vidur.rl.train --config $CONFIG \
    --set init_checkpoint=rl_runs/frontier_imitation/best.pt \
    --set output_dir=rl_runs/frontier_imitation_ppo \
    > rl_runs/frontier_imitation_ppo/train.log 2>&1 \
    && echo "ppo done $(date)" || echo "ppo FAILED $(date)"
set +e

# CDFs on the eval traces. Baseline samples come from frontier_synthetic_cdfs
# (same workloads and seeds), so only the policy is simulated.
cdfs() {  # <checkpoint> <output dir> <label>
    mkdir -p $2/data
    for f in rl_runs/frontier_synthetic_cdfs/data/*__{fcfs,sjf_prefill_tokens,sjf_prefill_tokens_age200}.npz; do
        cp -n $f $2/data/
    done
    $PY -m vidur.rl.plot_cdfs --backend frontier \
        --workloads rl_runs/frontier_synthetic_cdfs/eval_workloads.yaml \
        --checkpoint $1 --output-dir $2 --workers 4 \
        --policies fcfs sjf_prefill_tokens_age200 sjf_prefill_tokens policy_rank \
        --policy-label "$3" > $2/run.log 2>&1
}
[ -f rl_runs/frontier_imitation_ppo/best.pt ] && \
cdfs rl_runs/frontier_imitation_ppo/best.pt rl_runs/frontier_imitation_ppo_cdfs \
    "Imitation + PPO (rank)"
cdfs rl_runs/frontier_imitation/best.pt rl_runs/frontier_imitation_cdfs \
    "Imitation student (rank)"
echo "cdfs done $(date)"
