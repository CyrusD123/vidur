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


@pytest.mark.parametrize("trace", ["tracelab", "sharegpt_multiturn"])
def test_session_workloads_carry_prefix_cache_keys(trace):
    from vidur.rl.env import SESSION_TRACE_FILES, _session_trace_file

    import os

    if not os.path.exists(_session_trace_file(trace)):
        pytest.skip(f"{SESSION_TRACE_FILES[trace]} not built in Frontier")

    def requests(seed):
        config = make_request_generator_config(
            trace, arrivals="sessions", qps=2, num_requests=100, max_tokens=131072
        )
        _set_workload_seed(config, seed)
        return _generate(config)

    a, b, c = requests(0), requests(0), requests(1)
    assert len(a) == 100
    assert _lengths(a) == _lengths(b) and _lengths(a) != _lengths(c)
    times = [r.arrived_at for r in a]
    assert times == sorted(times) and times[0] == 0
    assert 100 / times[-1] == pytest.approx(2, rel=0.3)
    # each request names its prompt's and output's blocks:
    # (num_prefill_tokens + num_decode_tokens) // 16 ids (or one placeholder
    # id for a call under a block)
    from frontier.utils.block_hash_ids import decode_block_hash_ids

    for r in a:
        ids = decode_block_hash_ids(r.block_hash_ids)
        assert len(ids) == max((r.num_prefill_tokens + r.num_decode_tokens) // 16, 1)
        assert r.ids_cover_output == 1
        assert r.num_prefill_tokens + r.num_decode_tokens <= 131072
    assert len({r.session_id for r in a}) < 100  # calls share sessions


def test_job_size_skips_prefix_cached_prompt_tokens():
    from types import SimpleNamespace

    from vidur.scheduler.replica_scheduler.priority_policy import job_size

    plain = SimpleNamespace(num_prefill_tokens=120_000, num_decode_tokens=200)
    cached = SimpleNamespace(num_prefill_tokens=120_000, num_decode_tokens=200,
                             num_cached_prefill_tokens=119_000)
    assert job_size(plain, "total_tokens", 2048) == 120_200
    assert job_size(cached, "total_tokens", 2048) == 1_200
    assert job_size(cached, "iterations", 2048) == 1 + 200
    assert job_size(cached, "decode_tokens", 2048) == 200


def test_frontier_oracle_requests_carry_cache_hits():
    from types import SimpleNamespace

    from frontier.entities import Request as FrontierRequest
    from frontier.kv_cache.base_kv_cache_manager import KVCacheManager
    from frontier.rl.observation import cached_prefix_tokens
    from vidur.rl.frontier_env import _OracleRequest

    manager = KVCacheManager(block_size=16, num_gpu_blocks=8, enable_caching=True,
                             caching_hash_algo="builtin", num_preallocate_tokens=0)
    manager.block_pool.cached_block_hash_to_block[5] = {0: manager.block_pool.blocks[0]}
    request = FrontierRequest(0.0, 48, 7, block_hash_ids=[5, 6, 7])
    scheduler = SimpleNamespace(_kv_cache_manager=manager)
    view = _OracleRequest(request, cached_prefix_tokens(scheduler, request))
    assert view.num_cached_prefill_tokens == 16
    assert (view.id, view.num_prefill_tokens, view.num_decode_tokens) == (request.id, 48, 7)


def test_prefix_cache_state_counts_idle_blocks_and_caps_last_token():
    from frontier.entities import Request as FrontierRequest
    from frontier.kv_cache.base_kv_cache_manager import KVCacheManager

    manager = KVCacheManager(block_size=16, num_gpu_blocks=8, enable_caching=True,
                             caching_hash_algo="builtin", num_preallocate_tokens=0)
    blocks = manager.block_pool.blocks
    for i, block_hash in enumerate((5, 6, 7)):
        manager.block_pool.cached_block_hash_to_block[block_hash] = {i: blocks[i]}
    blocks[0].incr_ref()  # a running request holds the chain's first block
    # 48 tokens, all three blocks cached: admission still computes the last
    # token, so only two blocks count; one of them is idle
    request = FrontierRequest(0.0, 48, 7, block_hash_ids=[5, 6, 7])
    assert manager.peek_prefix_cache(request) == (32, 1)
    assert manager.peek_prefix_cache(request, chained=True) == (32, 1)
    # a miss in the middle ends the hit (unchained ids: the scan only)
    other = FrontierRequest(0.0, 64, 7, block_hash_ids=[5, 9, 7, 8])
    assert manager.peek_prefix_cache(other) == (16, 0)
    # chained ids that diverge after block 1
    branch = FrontierRequest(0.0, 64, 7, block_hash_ids=[5, 6, 70, 80])
    assert manager.peek_prefix_cache(branch, chained=True) == (32, 1)
    assert manager.prefix_cache_stats.queries == 0  # a peek is not a lookup


def test_prefix_cache_features():
    from types import SimpleNamespace

    from vidur.rl.features import FeatureContext
    from vidur.rl.observation import (
        DEFAULT_GLOBAL_FEATURES,
        PREFIX_CACHE_GLOBAL_FEATURES,
        PREFIX_CACHE_WAITING_FEATURES,
        ObservationConfig,
        build_observation,
    )

    assert not set(PREFIX_CACHE_GLOBAL_FEATURES) & set(DEFAULT_GLOBAL_FEATURES)
    config = ObservationConfig(prefix_cache=True)
    assert config.waiting_features[-len(PREFIX_CACHE_WAITING_FEATURES):] == PREFIX_CACHE_WAITING_FEATURES
    assert ObservationConfig(**{"prefix_cache": True, "waiting_features": config.waiting_features}).waiting_features == config.waiting_features
    import pickle

    assert pickle.loads(pickle.dumps(config)).global_features == config.global_features

    def request(i, prompt, cached, idle):
        return SimpleNamespace(
            id=i, arrived_at=float(i), num_prefill_tokens=prompt, num_processed_tokens=0,
            num_restarts=0, num_cached_prefill_tokens=cached, num_idle_cached_blocks=idle,
        )

    # a 120k prompt with 119k cached against a cold 4k one
    waiting = [request(0, 120_000, 118_992, 7_000), request(1, 4_000, 0, 0)]
    scheduler = SimpleNamespace(
        current_time=10.0, block_size=16, num_free_blocks=10_000, num_blocks=28_318,
        max_num_batched_tokens=2048, num_priority_levels=16, num_running_requests=0,
        rank_mask=[True, True], running_requests=[], recent_prefix_hit_rate=0.5,
        request_history=lambda _: __import__(
            "vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler",
            fromlist=["RequestHistory"],
        ).RequestHistory(),
        window_stats=lambda metric, window: (0.0, 0.0, 0.0),
    )
    obs = build_observation(waiting, scheduler, config)
    w = {name: obs.waiting_features[:, i] for i, name in enumerate(config.waiting_features)}
    g = dict(zip(config.global_features, obs.global_features))
    np.testing.assert_allclose(w["num_uncached_prefill_tokens"], np.log1p([1_008, 4_000]), rtol=1e-6)
    np.testing.assert_allclose(w["cached_fraction"], [118_992 / 120_000, 0], rtol=1e-6)
    # 7500 blocks, 7437 cached -> 63 new; the cold one needs all 250
    np.testing.assert_allclose(w["num_uncached_blocks_needed"], np.log1p([63, 250]), rtol=1e-6)
    np.testing.assert_allclose(w["alloc_blocks_over_free"], np.log1p([(63 + 7_000) / 10_000, 0.025]), rtol=1e-5)
    np.testing.assert_allclose(w["uncached_rank"], [0, 1])
    np.testing.assert_allclose(g["waiting_uncached_tokens"], np.log1p(5_008), rtol=1e-6)
    assert g["recent_prefix_hit_rate"] == pytest.approx(0.5)
    # without a prefix cache (Vidur requests) they read as uncached
    plain = SimpleNamespace(**{k: v for k, v in vars(waiting[0]).items() if "cached" not in k})
    obs = build_observation([plain, waiting[1]], scheduler, config)
    assert obs.waiting_features[0, config.waiting_features.index("cached_fraction")] == 0


def test_synthetic_session_workloads_are_seeded_and_vary():
    def requests(seed):
        config = make_request_generator_config(
            "synthetic:random", arrivals="sessions", qps=1, num_requests=80, max_tokens=131072
        )
        _set_workload_seed(config, seed)
        return _generate(config)

    a, b, c = requests(0), requests(0), requests(1)
    assert len(a) == len(c) == 80
    assert _lengths(a) == _lengths(b) and _lengths(a) != _lengths(c)
    assert all(r.ids_cover_output == 1 and r.block_hash_ids for r in a)
    assert len({r.session_id for r in a}) < 80
