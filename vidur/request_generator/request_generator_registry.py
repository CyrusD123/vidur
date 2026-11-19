from vidur.request_generator.session_trace_request_generator import (
    SessionTraceRequestGenerator,
)
from vidur.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from vidur.request_generator.trace_replay_request_generator import (
    TraceReplayRequestGenerator,
)
from vidur.request_generator.trace_window_request_generator import (
    TraceWindowRequestGenerator,
)
from vidur.types import RequestGeneratorType
from vidur.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.TRACE_REPLAY, TraceReplayRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.TRACE_WINDOW, TraceWindowRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.SESSION_TRACE, SessionTraceRequestGenerator
)
