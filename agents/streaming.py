"""Keep failed stream attempts out of committed answers and learning data."""
from __future__ import annotations

import asyncio
from collections import deque
from contextvars import ContextVar
import logging
import time

from .ui import print_info

_REQUEST = ContextVar("model_request_diagnostic", default=None)


class StreamAttempts:
    def _runtime_event(self, event, **fields):
        """Bounded, content-free diagnostics shared by CLI and Web."""
        if not hasattr(self, "runtime_events"):
            self.runtime_events = deque(maxlen=128)
        row = {"event": event, "time": time.monotonic(), **fields}
        self.runtime_events.append(row)
        logging.getLogger("bear.runtime").info("%s", row)
        state = _REQUEST.get()
        if state is not None and event in {"request_started", "first_chunk", "model_finished", "stream_closed"}:
            state["phase"] = event

    async def _run_model_request(self, request, *, retry):
        state = {"phase": "request_started"}
        token = _REQUEST.set(state)
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.model_timeout) as deadline:
                return await self._with_stream_retry(request, retry=retry)
        except TimeoutError:
            if not deadline.expired():
                raise
            self._runtime_event("model_timeout", phase=state["phase"], elapsed=time.monotonic() - started)
            raise TimeoutError(f"Model request exceeded {self.model_timeout:g}s (last phase: {state['phase']}).") from None
        except asyncio.CancelledError:
            self._runtime_event("model_cancelled", phase=state["phase"])
            raise
        except Exception as exc:
            self._runtime_event("model_error", phase=state["phase"], error_type=type(exc).__name__)
            raise
        finally:
            _REQUEST.reset(token)

    def _capture_stream_output(self):
        return {"turn": len(self._turn_output_buffer or []),
                "output": len(self._output_buffer or [])}

    def _discard_stream_output(self, state) -> None:
        emitted = self._turn_output_buffer is not None and len(self._turn_output_buffer) > state["turn"]
        if self._turn_output_buffer is not None:
            del self._turn_output_buffer[state["turn"]:]
        if self._output_buffer is not None:
            del self._output_buffer[state["output"]:]
        elif emitted:
            # Terminal text cannot be recalled; explicitly invalidate the partial
            # attempt. Browser adapters replace their provisional entries.
            print_info("Response interrupted; the partial answer above was discarded.")

    async def _with_stream_retry(self, request, *, retry):
        async def attempt():
            state = self._capture_stream_output()
            try:
                return await request()
            except BaseException:
                self._discard_stream_output(state)
                raise
        return await retry(attempt)
