"""
Checks for RLVLLMReplicaScheduler.

Each simulation runs in its own subprocess because Vidur keeps global state
(entity/event id counters, atexit output hooks) that is not reset between runs.

Run from the repo root (so ./data and ./cache resolve):
    python -m pytest tests/test_rl_vllm_scheduler.py -v

The first run trains the execution time predictor (~5 min); later runs hit ./cache.
"""

import glob
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# High enough load (and small enough KV cache, see _scheduler_args) that the queue
# builds up and vLLM preempts/restarts requests, so every branch of the scheduler
# is exercised.
COMMON_ARGS = [
    "--replica_config_device", "a100",
    "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
    "--cluster_config_num_replicas", "1",
    "--replica_config_tensor_parallel_size", "1",
    "--replica_config_num_pipeline_stages", "1",
    "--request_generator_config_type", "synthetic",
    "--synthetic_request_generator_config_num_requests", "300",
    "--length_generator_config_type", "trace",
    "--trace_request_length_generator_config_max_tokens", "4096",
    "--trace_request_length_generator_config_trace_file",
    "./data/processed_traces/splitwise_conv.csv",
    "--interval_generator_config_type", "poisson",
    "--poisson_request_interval_generator_config_qps", "10",
    "--metrics_config_write_json_trace",
    "--no-metrics_config_enable_chrome_trace",
    "--no-metrics_config_store_plots",
]  # fmt: skip


def _scheduler_args(scheduler_type: str) -> list:
    prefix = {"vllm": "vllm", "rl_vllm": "rl_vllm"}[scheduler_type]
    return [
        "--replica_scheduler_config_type", scheduler_type,
        f"--{prefix}_scheduler_config_batch_size_cap", "128",
        f"--{prefix}_scheduler_config_max_num_batched_tokens", "2048",
        # small KV cache (64k tokens) so the preempt/restart path runs
        f"--{prefix}_scheduler_config_num_blocks", "4000",
    ]  # fmt: skip


def _run_simulation(tmp_path, name: str, args: list, setup_code: str = "") -> str:
    output_dir = str(tmp_path / name)
    argv = [sys.executable, "-m", "vidur.main"]
    full_args = args + ["--metrics_config_output_dir", output_dir]

    if setup_code:
        # register test-only objects in the subprocess before running main()
        script = (
            "import sys\n"
            f"{setup_code}\n"
            "from vidur.main import main\n"
            f"sys.argv = ['vidur'] + {full_args!r}\n"
            "main()\n"
        )
        argv = [sys.executable, "-c", script]
    else:
        argv = argv + full_args

    result = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-5000:]

    # MetricsConfig appends a timestamped subdirectory
    (run_dir,) = glob.glob(f"{output_dir}/*/")
    return run_dir


def _load_outputs(run_dir: str):
    request_metrics = pd.read_csv(f"{run_dir}/request_metrics.csv")
    with open(f"{run_dir}/event_trace.json") as f:
        event_trace = json.load(f)
    return request_metrics, event_trace


@pytest.fixture(scope="module")
def stock_vllm_outputs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("stock")
    run_dir = _run_simulation(tmp_path, "vllm", COMMON_ARGS + _scheduler_args("vllm"))
    return _load_outputs(run_dir)


def test_fcfs_matches_stock_vllm(tmp_path, stock_vllm_outputs):
    run_dir = _run_simulation(
        tmp_path,
        "rl_vllm_fcfs",
        COMMON_ARGS
        + _scheduler_args("rl_vllm")
        + ["--rl_vllm_scheduler_config_priority_policy", "fcfs"],
    )
    rl_metrics, rl_trace = _load_outputs(run_dir)
    stock_metrics, stock_trace = stock_vllm_outputs

    # the workload actually stressed the scheduler (queueing + preemption)
    assert stock_metrics["request_scheduling_delay"].max() > 0
    assert stock_metrics["request_num_restarts"].max() > 0

    # identical event-by-event execution: same batches, same requests, same times
    assert rl_trace == stock_trace
    pd.testing.assert_frame_equal(rl_metrics, stock_metrics, check_exact=True)


REVERSE_POLICY_SETUP = """
from vidur.scheduler.replica_scheduler.priority_policy import (
    BasePriorityPolicy,
    PRIORITY_POLICIES,
)

class ReversePriorityPolicy(BasePriorityPolicy):
    # newest requests get the lowest (best) priority
    def get_priorities(self, waiting_requests, scheduler):
        n = len(waiting_requests)
        k = scheduler.num_priority_levels
        return [max(0, k - 1 - i) for i in range(n)]

PRIORITY_POLICIES["test_reverse"] = ReversePriorityPolicy
"""


def test_non_fcfs_policy_changes_schedule(tmp_path, stock_vllm_outputs):
    run_dir = _run_simulation(
        tmp_path,
        "rl_vllm_reverse",
        COMMON_ARGS
        + _scheduler_args("rl_vllm")
        + ["--rl_vllm_scheduler_config_priority_policy", "test_reverse"],
        setup_code=REVERSE_POLICY_SETUP,
    )
    rl_metrics, rl_trace = _load_outputs(run_dir)
    stock_metrics, stock_trace = stock_vllm_outputs

    # every request still completes
    assert len(rl_metrics) == len(stock_metrics)
    assert rl_metrics["request_e2e_time"].notna().all()

    # but the reranking hook actually changed what got scheduled
    assert rl_trace != stock_trace
    assert not rl_metrics["request_e2e_time"].equals(stock_metrics["request_e2e_time"])
