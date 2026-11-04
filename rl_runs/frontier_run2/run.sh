#!/bin/zsh
# Frontier run2: train, then CDF plots (see configs/rl/frontier_run2.yaml)
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=../Frontier/.venv/bin/python
$PY -m vidur.rl.train --config configs/rl/frontier_run2.yaml > rl_runs/frontier_run2/train.log 2>&1 || { echo TRAIN_FAILED >> rl_runs/frontier_run2/train.log; exit 1; }
$PY -m vidur.rl.plot_cdfs --backend frontier --checkpoint rl_runs/frontier_run2/best.pt \
    --workloads configs/rl/frontier_cdf_workloads.yaml --output-dir rl_runs/frontier_cdf_plots \
    --workers 3 > rl_runs/frontier_cdf_plots.log 2>&1 || { echo PLOT_FAILED >> rl_runs/frontier_cdf_plots.log; exit 1; }
echo ALL_DONE >> rl_runs/frontier_cdf_plots.log
