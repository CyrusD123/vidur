"""
Named observation features. A feature is a function of a FeatureContext that
returns one value per request (waiting or running features) or a single value
(global features); register new ones with the decorators below and select them
by name in ObservationConfig.

    @waiting_feature("wait_time", transform="log1p")
    def wait_time(ctx):
        return [ctx.time - r.arrived_at for r in ctx.waiting_requests]

Features must not read a request's output length (num_decode_tokens and anything
derived from it, such as total_tokens or pd_ratio): a real server does not know
it. Rewards may use it.
"""

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Callable, Dict, List

import numpy as np

from vidur.entities import Request
from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
    WINDOW_METRICS,
    WINDOW_SIZES,
    RequestHistory,
)

if TYPE_CHECKING:
    from vidur.scheduler.replica_scheduler.rl_vllm_replica_scheduler import (
        RLVLLMReplicaScheduler,
    )


@dataclass
class FeatureContext:
    time: float
    scheduler: "RLVLLMReplicaScheduler"
    # in queue order, which is the order of the action
    waiting_requests: List[Request]
    waiting_histories: List[RequestHistory]
    rank_mask: List[bool]
    running_requests: List[Request]


@dataclass(frozen=True)
class Feature:
    name: str
    fn: Callable[[FeatureContext], object]
    # "none" or "log1p", applied after scaling
    transform: str = "none"
    scale: float = 1.0

    def compute(self, ctx: FeatureContext) -> np.ndarray:
        values = np.asarray(self.fn(ctx), dtype=np.float32) * self.scale
        if self.transform == "log1p":
            values = np.log1p(values)
        return values


WAITING_FEATURES: Dict[str, Feature] = {}
RUNNING_FEATURES: Dict[str, Feature] = {}
GLOBAL_FEATURES: Dict[str, Feature] = {}


def _register(registry: Dict[str, Feature], name: str, transform: str, scale: float):
    assert transform in ("none", "log1p"), f"Unknown transform {transform}"

    def decorator(fn):
        assert name not in registry, f"Feature {name} is already registered"
        registry[name] = Feature(name, fn, transform, scale)
        return fn

    return decorator


def waiting_feature(name: str, transform: str = "none", scale: float = 1.0):
    """fn(ctx) -> one value per waiting request, in queue order."""
    return _register(WAITING_FEATURES, name, transform, scale)


def running_feature(name: str, transform: str = "none", scale: float = 1.0):
    """fn(ctx) -> one value per running request."""
    return _register(RUNNING_FEATURES, name, transform, scale)


def global_feature(name: str, transform: str = "none", scale: float = 1.0):
    """fn(ctx) -> a single value."""
    return _register(GLOBAL_FEATURES, name, transform, scale)


def _percentiles(values: List[float]) -> np.ndarray:
    """Rank of each value among all of them, 0 = smallest, 1 = largest."""
    n = len(values)
    ranks = np.empty(n, dtype=np.float32)
    ranks[np.argsort(values, kind="stable")] = np.arange(n)
    return ranks / max(1, n - 1)


# ---------------------------------------------------------------------------
# waiting requests (the ones being ranked)
# ---------------------------------------------------------------------------


@waiting_feature("num_prefill_tokens", transform="log1p")
def _num_prefill_tokens(ctx):
    # after a restart this includes the tokens decoded before preemption
    return [r.num_prefill_tokens for r in ctx.waiting_requests]


@waiting_feature("num_processed_tokens", transform="log1p")
def _num_processed_tokens(ctx):
    return [r.num_processed_tokens for r in ctx.waiting_requests]


@waiting_feature("wait_time", transform="log1p")
def _wait_time(ctx):
    return [ctx.time - r.arrived_at for r in ctx.waiting_requests]


@waiting_feature("num_restarts")
def _num_restarts(ctx):
    return [r.num_restarts for r in ctx.waiting_requests]


@waiting_feature("num_blocks_needed", transform="log1p")
def _num_blocks_needed(ctx):
    block_size = ctx.scheduler.block_size
    return [ceil(r.num_prefill_tokens / block_size) for r in ctx.waiting_requests]


@waiting_feature("blocks_needed_over_free", transform="log1p")
def _blocks_needed_over_free(ctx):
    # > 1 means the request cannot be admitted until memory frees up
    block_size = ctx.scheduler.block_size
    free = max(1, ctx.scheduler.num_free_blocks)
    return [
        ceil(r.num_prefill_tokens / block_size) / free for r in ctx.waiting_requests
    ]


# --- prefix caching (Frontier backend; all 0 without a prefix cache) -------
#
# A waiting request's cache hit is read the way vLLM's admission reads it:
# the leading prompt blocks whose chained hashes are in the cache now, capped
# so at least the last prompt token is computed
# (KVCacheManager.peek_prefix_cache, set on the request view by
# vidur/rl/frontier_env.py). On a real server this is the lookup
# get_computed_blocks already does for the head of the queue; done for every
# waiting request it is one dict lookup per block (or O(log blocks) with a
# binary search over the hash chain, since LRU eviction of tail-first freed
# blocks leaves cached prefixes contiguous).


def _cached_tokens(r) -> int:
    return getattr(r, "num_cached_prefill_tokens", 0)


def _uncached_tokens(r) -> int:
    # the prefill admission would actually compute (a restart recomputes its
    # generated tokens too: num_prefill_tokens covers them on the view)
    return max(r.num_prefill_tokens - _cached_tokens(r), 0)


def _uncached_blocks(r, block_size: int) -> int:
    return max(ceil(r.num_prefill_tokens / block_size) - _cached_tokens(r) // block_size, 0)


@waiting_feature("num_cached_prefill_tokens", transform="log1p")
def _num_cached_prefill_tokens(ctx):
    return [_cached_tokens(r) for r in ctx.waiting_requests]


@waiting_feature("num_uncached_prefill_tokens", transform="log1p")
def _num_uncached_prefill_tokens(ctx):
    return [_uncached_tokens(r) for r in ctx.waiting_requests]


@waiting_feature("cached_fraction")
def _cached_fraction(ctx):
    return [_cached_tokens(r) / max(1, r.num_prefill_tokens) for r in ctx.waiting_requests]


@waiting_feature("num_uncached_blocks_needed", transform="log1p")
def _num_uncached_blocks_needed(ctx):
    # new KV blocks the whole prompt needs beyond its cache hit
    block_size = ctx.scheduler.block_size
    return [_uncached_blocks(r, block_size) for r in ctx.waiting_requests]


@waiting_feature("alloc_blocks_over_free", transform="log1p")
def _alloc_blocks_over_free(ctx):
    # what admitting the whole prompt takes from the free pool, over the free
    # pool: new blocks plus the hit's idle blocks, which are free-queue
    # entries until admission takes them (vLLM's can_allocate_slots counts
    # both). > 1: it cannot fully fit until memory frees up
    block_size = ctx.scheduler.block_size
    free = max(1, ctx.scheduler.num_free_blocks)
    return [
        (_uncached_blocks(r, block_size) + getattr(r, "num_idle_cached_blocks", 0)) / free
        for r in ctx.waiting_requests
    ]


@waiting_feature("uncached_rank")
def _uncached_rank(ctx):
    # the shortest-uncached-prompt-first order, as a percentile
    return _percentiles([_uncached_tokens(r) for r in ctx.waiting_requests])


@waiting_feature("uncached_over_budget", transform="log1p")
def _uncached_over_budget(ctx):
    # prefill steps it needs at the full token budget: its service time in
    # steps, the unit the policy trades waits in
    budget = max(1, ctx.scheduler.max_num_batched_tokens)
    return [_uncached_tokens(r) / budget for r in ctx.waiting_requests]


# Eviction distance (RLVLLMv1EngineReplicaScheduler.cache_state): how many
# blocks will be allocated before a waiting request's cached prefix loses a
# block. vLLM evicts the free queue's head first, so this is the position of
# the hit's first-evicted idle block in the free queue; a Fenwick tree over
# queue sequence numbers keeps it at O(log blocks) per queue operation and
# per query, and with chained hashes that block is the hit's last.


def _evict_distance(r) -> int:
    return getattr(r, "evict_distance_blocks", -1)


@waiting_feature("has_evict_distance")
def _has_evict_distance(ctx):
    # its hit has blocks no running request holds (they can be evicted)
    return [_evict_distance(r) >= 0 for r in ctx.waiting_requests]


@waiting_feature("evict_distance_k", transform="log1p", scale=1e-3)
def _evict_distance_k(ctx):
    # thousands of blocks (0 when nothing can be evicted)
    return [max(_evict_distance(r), 0) for r in ctx.waiting_requests]


@waiting_feature("evict_over_queued")
def _evict_over_queued(ctx):
    # that distance over the blocks the queue's uncached prompts need, capped
    # at 4 and scaled to [0, 1]: < 1/4 means serving the rest of the queue
    # first would evict part of its hit
    block_size = ctx.scheduler.block_size
    queued = max(1, sum(_uncached_blocks(r, block_size) for r in ctx.waiting_requests))
    return [
        min(_evict_distance(r) / queued, 4.0) / 4.0 if _evict_distance(r) >= 0 else 0.0
        for r in ctx.waiting_requests
    ]


@waiting_feature("queue_position")
def _queue_position(ctx):
    n = len(ctx.waiting_requests)
    return np.arange(n) / max(1, n - 1)


@waiting_feature("arrival_rank")
def _arrival_rank(ctx):
    return _percentiles([r.arrived_at for r in ctx.waiting_requests])


@waiting_feature("is_new")
def _is_new(ctx):
    return [h.num_decisions_seen == 0 for h in ctx.waiting_histories]


@waiting_feature("times_passed_over", transform="log1p")
def _times_passed_over(ctx):
    return [h.times_passed_over for h in ctx.waiting_histories]


@waiting_feature("num_overtaken_by", transform="log1p")
def _num_overtaken_by(ctx):
    return [h.num_overtaken_by for h in ctx.waiting_histories]


@waiting_feature("prev_rank_percentile")
def _prev_rank_percentile(ctx):
    return [
        h.last_rank_percentile if h.last_rank_percentile is not None else 0.0
        for h in ctx.waiting_histories
    ]


@waiting_feature("has_prev_rank")
def _has_prev_rank(ctx):
    return [h.last_rank_percentile is not None for h in ctx.waiting_histories]


@waiting_feature("current_priority")
def _current_priority(ctx):
    # only persists between decisions in arrival_and_periodic mode
    top = max(1, ctx.scheduler.num_priority_levels - 1)
    return [
        h.priority / top if h.priority is not None else 0.0
        for h in ctx.waiting_histories
    ]


@waiting_feature("has_priority")
def _has_priority(ctx):
    return [h.priority is not None for h in ctx.waiting_histories]


@waiting_feature("to_rank")
def _to_rank(ctx):
    return ctx.rank_mask


# ---------------------------------------------------------------------------
# running requests (context only, not ranked)
# ---------------------------------------------------------------------------


@running_feature("num_prefill_tokens", transform="log1p")
def _running_num_prefill_tokens(ctx):
    return [r.num_prefill_tokens for r in ctx.running_requests]


@running_feature("num_decoded_tokens", transform="log1p")
def _running_num_decoded_tokens(ctx):
    return [r.num_processed_decode_tokens for r in ctx.running_requests]


@running_feature("prefill_progress")
def _running_prefill_progress(ctx):
    return [
        r.num_processed_prefill_tokens / r.num_prefill_tokens
        for r in ctx.running_requests
    ]


@running_feature("is_prefill_complete")
def _running_is_prefill_complete(ctx):
    return [r.is_prefill_complete for r in ctx.running_requests]


@running_feature("blocks_held", transform="log1p")
def _running_blocks_held(ctx):
    return [ctx.scheduler.num_allocated_blocks_of(r.id) for r in ctx.running_requests]


@running_feature("time_since_arrival", transform="log1p")
def _running_time_since_arrival(ctx):
    return [ctx.time - r.arrived_at for r in ctx.running_requests]


@running_feature("time_since_first_token", transform="log1p")
def _running_time_since_first_token(ctx):
    return [
        ctx.time - r.prefill_completed_at if r.is_prefill_complete else 0.0
        for r in ctx.running_requests
    ]


# ---------------------------------------------------------------------------
# global
# ---------------------------------------------------------------------------


@global_feature("num_waiting", transform="log1p")
def _num_waiting(ctx):
    return len(ctx.waiting_requests)


@global_feature("num_running", transform="log1p")
def _num_running(ctx):
    return ctx.scheduler.num_running_requests


@global_feature("free_block_fraction")
def _free_block_fraction(ctx):
    return ctx.scheduler.num_free_blocks / ctx.scheduler.num_blocks


@global_feature("waiting_uncached_tokens", transform="log1p")
def _waiting_uncached_tokens(ctx):
    # prefill work queued (cache hits excluded)
    return sum(_uncached_tokens(r) for r in ctx.waiting_requests)


@global_feature("waiting_cached_fraction")
def _waiting_cached_fraction(ctx):
    prompt = sum(r.num_prefill_tokens for r in ctx.waiting_requests)
    return sum(_cached_tokens(r) for r in ctx.waiting_requests) / max(1, prompt)


@global_feature("recent_prefix_hit_rate")
def _recent_prefix_hit_rate(ctx):
    # cached / prompt tokens of recently admitted requests; falls as sessions
    # evict each other's prefixes (the cache cliff)
    return getattr(ctx.scheduler, "recent_prefix_hit_rate", 0.0)


# sliding-window history: {metric}_{mean,min,max}_{short,medium,long}
_WINDOW_FEATURE_TRANSFORMS = {
    "queue_length": ("log1p", 1.0),
    "free_block_fraction": ("none", 1.0),
    "batch_num_tokens": ("log1p", 1.0),
    # seconds -> 10 ms units
    "batch_time": ("log1p", 100.0),
}


def _register_window_features() -> None:
    for metric in WINDOW_METRICS:
        transform, scale = _WINDOW_FEATURE_TRANSFORMS[metric]
        for window in WINDOW_SIZES:
            for index, stat in enumerate(("mean", "min", "max")):

                def fn(ctx, metric=metric, window=window, index=index):
                    return ctx.scheduler.window_stats(metric, window)[index]

                global_feature(f"{metric}_{stat}_{window}", transform, scale)(fn)


_register_window_features()
