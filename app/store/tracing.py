"""Per-node tracing: times each graph node, streams a `step` event to the UI and records it for SQLite.

Lightweight on purpose (no Langfuse/OTel stack to run on a laptop); the step records have the same shape
as OTel spans (name, duration, attributes) so exporting them later is mechanical (docs/ROADMAP.md).
"""

import contextlib
import functools
import time
from collections.abc import Awaitable, Callable

from langgraph.config import get_stream_writer


def emit(event: dict) -> None:
    """Send a custom event to the stream; silently no-op outside a streaming run."""
    with contextlib.suppress(Exception):  # observability must never break a request
        get_stream_writer()(event)


def traced(name: str) -> Callable:
    def deco(fn: Callable[[dict], Awaitable[dict]]) -> Callable[[dict], Awaitable[dict]]:
        @functools.wraps(fn)
        async def wrapper(state: dict) -> dict:
            t0 = time.perf_counter()
            update = await fn(state) or {}
            step = {"node": name, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "detail": update.pop("_detail", None)}
            emit({"type": "step", **step})
            update["steps"] = [step]
            return update

        return wrapper

    return deco
