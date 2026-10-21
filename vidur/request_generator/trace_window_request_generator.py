import functools
import logging
from typing import List

import numpy as np
import pandas as pd

from vidur.config import TraceWindowRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _load_trace(trace_file: str, max_tokens: int) -> pd.DataFrame:
    trace_df = pd.read_csv(trace_file).sort_values("arrived_at", kind="stable")
    prefill = trace_df["num_prefill_tokens"].to_numpy(dtype=float)
    decode = trace_df["num_decode_tokens"].to_numpy(dtype=float)

    # same as TraceRequestLengthGenerator: deduct any excess over max_tokens
    # from prefill and decode proportionally, keeping at least one of each
    total = prefill + decode
    excess = np.clip(total - max_tokens, 0, None)
    prefill = np.clip(prefill - np.ceil(excess * prefill / total), 1, None)
    decode = np.clip(decode - np.ceil(excess * decode / total), 1, None)

    return pd.DataFrame(
        {
            "arrived_at": trace_df["arrived_at"].to_numpy(dtype=float),
            "num_prefill_tokens": prefill.astype(int),
            "num_decode_tokens": decode.astype(int),
        }
    )


class TraceWindowRequestGenerator(BaseRequestGenerator):
    """
    Replays a window of consecutive requests from an arrival trace (e.g. the
    Azure LLM inference traces), keeping the real arrival pattern and the real
    pairing of arrival times with request lengths. The seed picks where the
    window starts, so different seeds are different stretches of the trace.
    """

    def __init__(self, config: TraceWindowRequestGeneratorConfig):
        super().__init__(config)
        self.trace_df = _load_trace(config.trace_file, config.max_tokens)
        assert config.num_requests <= len(self.trace_df), (
            f"{config.trace_file} has only {len(self.trace_df)} requests"
        )

        arrived_at = self.trace_df["arrived_at"].to_numpy()
        native_qps = (len(arrived_at) - 1) / (arrived_at[-1] - arrived_at[0])
        self.time_scale_factor = native_qps / config.qps if config.qps else 1.0

    def generate_requests(self) -> List[Request]:
        rng = np.random.default_rng(self.config.seed)
        num_requests = self.config.num_requests
        start = int(rng.integers(0, len(self.trace_df) - num_requests + 1))
        window = self.trace_df.iloc[start : start + num_requests]

        arrived_at = window["arrived_at"].to_numpy()
        arrived_at = (arrived_at - arrived_at[0]) * self.time_scale_factor
        return [
            Request(
                arrived_at=float(t),
                num_prefill_tokens=int(p),
                num_decode_tokens=int(d),
            )
            for t, p, d in zip(
                arrived_at,
                window["num_prefill_tokens"],
                window["num_decode_tokens"],
            )
        ]
