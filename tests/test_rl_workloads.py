"""
Checks for the trace workloads (make_request_generator_config, the
trace-window replay generator, and the baseline assumption check).

Run from the repo root (so ./data resolves):
    python -m pytest tests/test_rl_workloads.py -v
"""

import numpy as np
import pandas as pd
import pytest

from vidur.config import TraceWindowRequestGeneratorConfig
from vidur.request_generator.request_generator_registry import (
    RequestGeneratorRegistry,
)
from vidur.rl.baselines import check_assumptions, relative_to_fcfs
from vidur.rl.env import TRACE_FILES, _set_workload_seed, make_request_generator_config


def _generate(config):
    return RequestGeneratorRegistry.get(config.get_type(), config).generate()


def _lengths(requests):
    return [(r.num_prefill_tokens, r.num_decode_tokens) for r in requests]


@pytest.mark.parametrize("trace", ["sharegpt", "azure_conv"])
def test_poisson_workloads_are_seeded(trace):
    def requests(seed):
        config = make_request_generator_config(trace, qps=10, num_requests=50)
        _set_workload_seed(config, seed)
        return _generate(config)

    a, b, c = requests(0), requests(0), requests(1)
    assert _lengths(a) == _lengths(b)
    assert [r.arrived_at for r in a] == [r.arrived_at for r in b]
    assert _lengths(a) != _lengths(c)
    assert all(p + d <= 4096 for p, d in _lengths(a))


def test_trace_window_replays_consecutive_rows():
    trace = pd.read_csv(TRACE_FILES["azure_conv"])
    # no clipping, so lengths match the trace rows exactly
    config = TraceWindowRequestGeneratorConfig(
        trace_file=TRACE_FILES["azure_conv"],
        num_requests=100,
        qps=None,
        max_tokens=10**6,
    )
    windows = []
    for seed in (0, 0, 1):
        config.seed = seed
        windows.append(_generate(config))
    assert _lengths(windows[0]) == _lengths(windows[1])
    assert _lengths(windows[0]) != _lengths(windows[2])

    requests = windows[0]
    arrived_at = np.array([r.arrived_at for r in requests])
    assert arrived_at[0] == 0 and np.all(np.diff(arrived_at) >= 0)
    # the window is a run of consecutive trace rows with the native gaps
    lengths = _lengths(requests)
    rows = list(zip(trace["num_prefill_tokens"], trace["num_decode_tokens"]))
    starts = [
        i
        for i in range(len(rows) - len(lengths) + 1)
        if rows[i : i + len(lengths)] == lengths
    ]
    assert starts
    native = trace["arrived_at"].to_numpy()[starts[0] : starts[0] + len(lengths)]
    np.testing.assert_allclose(arrived_at, native - native[0], atol=1e-9)


def test_trace_window_qps_scales_time_globally():
    def arrivals(qps):
        config = TraceWindowRequestGeneratorConfig(
            seed=3, trace_file=TRACE_FILES["azure_conv"], num_requests=200, qps=qps
        )
        return np.array([r.arrived_at for r in _generate(config)])

    trace = pd.read_csv(TRACE_FILES["azure_conv"])["arrived_at"]
    native_qps = (len(trace) - 1) / (trace.iloc[-1] - trace.iloc[0])
    native, doubled = arrivals(None), arrivals(2 * native_qps)
    np.testing.assert_allclose(doubled, native / 2, rtol=1e-9)


def test_trace_window_clips_to_max_tokens():
    config = TraceWindowRequestGeneratorConfig(
        seed=0,
        trace_file=TRACE_FILES["azure_conv"],
        num_requests=2000,
        max_tokens=1024,
    )
    lengths = _lengths(_generate(config))
    assert all(p + d <= 1024 and p >= 1 and d >= 1 for p, d in lengths)


def _table(fcfs, sjf, aged):
    """Baseline summaries for one workload and seed: (mean_e2e, p99_e2e)."""
    rows = []
    names = {"fcfs": fcfs, "sjf_total_tokens": sjf, "sjf_decode_tokens": sjf}
    for name in ("sjf_total_tokens_age100", "sjf_total_tokens_age400"):
        names[name] = aged
    for name in ("sjf_decode_tokens_age20", "sjf_decode_tokens_age80"):
        names[name] = aged
    for name, (mean, tail) in names.items():
        rows.append(
            {
                "workload": "w",
                "seed": 0,
                "baseline": name,
                "mean_e2e": mean,
                "mean_e2e_normalized": mean,
                "p99_e2e": tail,
                "max_e2e": tail,
                "p99_ttft": tail,
                "p99_e2e_normalized": tail,
            }
        )
    return pd.DataFrame(rows)


def test_check_assumptions():
    holds = check_assumptions(_table(fcfs=(10, 20), sjf=(6, 30), aged=(8, 25)))["w"]
    assert holds.all()

    # within tolerance: a 0.5% miss still holds, and the spread is "no spread"
    ties = check_assumptions(_table((10, 20), (9.96, 20.1), (9.98, 20.05)))["w"]
    assert ties["spread_mean_e2e"] == 0.0
    assert ties["fcfs_best_p99_e2e"] == 1.0

    # aging worse than both ends on the tail, and FCFS no longer the best tail
    fails = check_assumptions(_table(fcfs=(10, 20), sjf=(6, 30), aged=(8, 35)))["w"]
    assert fails["fcfs_best_p99_e2e"] == 1.0  # FCFS 20 is still the minimum
    assert fails["sjf_total_tokens_age100_between_p99_e2e"] == 0.0
    assert fails["sjf_total_tokens_age100_between_mean_e2e"] == 1.0

    ratios = relative_to_fcfs(_table((10, 20), (6, 30), (8, 25)), ["mean_e2e"])
    assert ratios.loc[("w", "sjf_total_tokens"), "mean_e2e"] == pytest.approx(0.6)
