import os
import sys
from typing import List

from vidur.config import SessionTraceRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator

# Frontier next to this repo, or $FRONTIER_ROOT (as in vidur/rl/frontier_env.py)
FRONTIER_ROOT = os.path.abspath(
    os.environ.get(
        "FRONTIER_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "Frontier"),
    )
)


# session_concurrency: calls generated per scored call (warm-up included);
# the simulation stops once the scored ones finish, so extras are cheap
CONCURRENT_OVERPROVISION = 4


def _frontier_sessions():
    if FRONTIER_ROOT not in sys.path:
        sys.path.insert(0, FRONTIER_ROOT)
    try:
        from frontier.trace_prep import sessions
    except ImportError as error:
        raise ImportError(
            f"session traces are sampled by Frontier's frontier.trace_prep; "
            f"none found at {FRONTIER_ROOT} (set FRONTIER_ROOT)"
        ) from error
    return sessions


class SessionTraceRequestGenerator(BaseRequestGenerator):
    """
    A window of calls sampled from a session trace by Frontier's
    frontier/trace_prep/sessions.py, so Vidur and Frontier workloads with the
    same config are the same calls. Each Request also gets `session_id` and
    `block_hash_ids` (runs text, frontier/utils/block_hash_ids.py), which
    vidur/rl/frontier_env.py hands to Frontier's prefix cache.
    """

    def __init__(self, config: SessionTraceRequestGeneratorConfig):
        super().__init__(config)
        assert config.trace_file, "session_trace needs a trace_file"

    def generate_requests(self) -> List[Request]:
        sessions = _frontier_sessions()
        config = self.config
        # a session trace file, or "synthetic:<preset>" / "synthetic:random"
        # sessions generated for this seed (frontier/trace_prep/synthetic.py)
        if config.session_concurrency:
            # closed system: over-provisioned calls; FrontierEnv scores the
            # first num_requests arriving after the warm-up and stops there
            workload = sessions.sample_named_workload(
                config.trace_file,
                CONCURRENT_OVERPROVISION * config.num_requests,
                config.qps,
                config.seed,
                config.max_tokens,
                concurrency=config.session_concurrency,
                session_time_scale=config.session_time_scale,
                turn_gap=config.turn_gap,
                session_gap=config.session_gap,
            )
        else:
            workload = sessions.sample_named_workload(
                config.trace_file,
                config.num_requests,
                config.qps,
                config.seed,
                config.max_tokens,
                session_time_scale=config.session_time_scale,
                turn_gap=config.turn_gap,
                phase=config.session_phase,
                closed_loop=config.session_closed_loop,
                **({"warmup": config.session_warmup} if config.session_warmup else {}),
            )
        requests = []
        for row in workload.itertuples(index=False):
            request = Request(
                arrived_at=float(row.arrived_at),
                num_prefill_tokens=int(row.num_prefill_tokens),
                num_decode_tokens=int(row.num_decode_tokens),
            )
            request.session_id = int(row.session_id)
            request.block_hash_ids = str(row.block_hash_ids)
            # the ids name the output's blocks too (frontier/trace_prep)
            request.ids_cover_output = int(getattr(row, "ids_cover_output", 0) or 0)
            think_time = getattr(row, "think_time", float("nan"))
            request.think_time = None if think_time != think_time else float(think_time)
            # warm-up call (session_warmup): simulated, not scored
            request.warmup = int(getattr(row, "warmup", 0) or 0)
            requests.append(request)
        return requests
