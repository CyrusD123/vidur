#!/bin/zsh
# DAgger imitation of sjf_prefill_tokens_age200, then PPO fine-tuning from its
# best checkpoint, then CDF plots (configs/rl/frontier_concurrency.yaml). Run
# from the vidur repo root:
#   nohup caffeinate -is zsh rl_runs/run_frontier_concurrency.sh > rl_runs/frontier_concurrency.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=../Frontier/.venv/bin/python
CONFIG=configs/rl/frontier_concurrency.yaml
IM=rl_runs/frontier_concurrency
FT=rl_runs/frontier_concurrency_ppo

mkdir -p $IM $FT
echo "imitation start $(date)"
$PY -m vidur.rl.imitation --config $CONFIG > $IM/train.log 2>&1 \
    && echo "imitation done $(date)" || { echo "imitation FAILED $(date)"; exit 1; }
$PY -m vidur.rl.train --config $CONFIG \
    --set init_checkpoint=$IM/best.pt --set output_dir=$FT \
    > $FT/train.log 2>&1 \
    && echo "ppo done $(date)" || echo "ppo FAILED $(date)"

# CDFs on the eval workloads (argmax decoding), the 3 eval seeds pooled; baselines are
# simulated once and shared between the two plots
cdfs() {  # <checkpoint> <output dir> <label>
    mkdir -p $2/data
    [ -d rl_runs/frontier_concurrency_cdfs_shared ] && cp -n rl_runs/frontier_concurrency_cdfs_shared/*.npz $2/data/ 2>/dev/null
    $PY -m vidur.rl.plot_cdfs --backend frontier \
        --workloads configs/rl/frontier_concurrency_cdf_workloads.yaml \
        --checkpoint $1 --output-dir $2 --workers 4 --seeds 1000000 1000001 1000002 \
        --policies fcfs sjf_prefill_tokens_age200 sjf_prefill_tokens policy_argmax \
        --policy-label "$3" > $2/run.log 2>&1 && echo "cdfs $2 done $(date)" || echo "cdfs $2 FAILED $(date)"
    mkdir -p rl_runs/frontier_concurrency_cdfs_shared
    for f in $2/data/*__{fcfs,sjf_prefill_tokens,sjf_prefill_tokens_age200}.npz; do
        cp -n $f rl_runs/frontier_concurrency_cdfs_shared/ 2>/dev/null
    done
}
[ -f $FT/best.pt ] && cdfs $FT/best.pt ${FT}_cdfs "Imitation + PPO (argmax)"
cdfs $IM/best.pt ${IM}_cdfs "Imitation student (argmax)"
echo "all done $(date)"
