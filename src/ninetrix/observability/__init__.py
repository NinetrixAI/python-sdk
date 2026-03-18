"""
ninetrix.observability
======================

Observability primitives for the ninetrix SDK.

    from ninetrix.observability import NinetrixLogger, enable_debug, get_logger
"""

from ninetrix.observability.logger import (
    NinetrixLogger as NinetrixLogger,
    enable_debug as enable_debug,
    get_logger as get_logger,
)
from ninetrix.observability.telemetry import (
    TelemetryEvent as TelemetryEvent,
    TelemetryCollector as TelemetryCollector,
    record_event as record_event,
)

__all__ = [
    "NinetrixLogger",
    "enable_debug",
    "get_logger",
    "TelemetryEvent",
    "TelemetryCollector",
    "record_event",
]
