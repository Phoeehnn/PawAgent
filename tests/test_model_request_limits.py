"""Offline fake streams: no model calls, including the real Web run boundary."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from agents.agent import Agent
from agents.web import WebApp
from agents import session


class FakeStream:
    def __init__(self, mode="ok", protocol="openai"):
        self.mode, self.protocol, self.closed = mode, protocol, False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        self.closed = True

    async def get_final_message(self):
        return NS(content=[], stop_reason="end_turn")

    def __aiter__(self):
        return self.events()

    async def events(self):
        if self.mode == "silent":
            await asyncio.Event().wait()
        if self.protocol == "openai":
            yield NS(usage=None, choices=[NS(delta=NS(content="OK", tool_calls=None),
                                           finish_reason="stop" if self.mode in {"ok", "finish_stall"} else None)])
        else:
            yield NS(type="content_block_delta", delta=NS(text="OK"))
        if self.mode == "broken":
            raise RuntimeError("ECONNRESET")
        if self.mode in {"stall", "finish_stall"}:
            await asyncio.Event().wait()


class ModelRequestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for patcher in (patch("pathlib.Path.home", return_value=root),
                        patch.object(session, "SESSION_DIR", root / "sessions"),
                        patch("agents.agent.start_spinner"), patch("agents.agent.stop_spinner"),
                        patch("agents.agent.print_divider")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def agent(self, protocol="openai", **kwargs):
        agent = Agent(api_key="offline", api_protocol=protocol, custom_system_prompt="test",
                      enable_mcp=False, enable_memory_prefetch=False, enable_online_evolution=False,
                      model_timeout=.1, **kwargs)
        agent._output_buffer = []
        self.addAsyncCleanup(agent.aclose)
        return agent

    def install(self, agent, stream):
        if agent.use_openai:
            client, method, factory = agent._openai_client.chat.completions, "create", AsyncMock
        else:
            client, method, factory = agent._anthropic_client.messages, "stream", Mock
        patcher = patch.object(client, method, factory(return_value=stream))
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    async def call(self, agent):
        method = agent._call_openai_stream if agent.use_openai else agent._call_anthropic_stream
        return await asyncio.wait_for(method(), 2)

    async def test_success_finish_and_close_for_both_protocols(self):
        for protocol in ("openai", "anthropic"):
            agent = self.agent(protocol)
            stream = FakeStream(protocol=protocol)
            request = self.install(agent, stream)
            await self.call(agent)
            self.assertEqual([row["event"] for row in agent.runtime_events],
                             ["request_started", "first_chunk", "model_finished", "stream_closed"])
            self.assertTrue(stream.closed)
            self.assertEqual(request.call_count, 1)
            client = agent._openai_client if agent.use_openai else agent._anthropic_client
            self.assertEqual(client.max_retries, 0)
            self.assertEqual(client.timeout.connect, 10)

    async def test_timeout_before_and_after_first_chunk_closes_and_discards(self):
        for protocol in ("openai", "anthropic"):
            for mode, phase in (("silent", "request_started"), ("stall", "first_chunk")):
                agent = self.agent(protocol)
                stream = FakeStream(mode, protocol)
                request = self.install(agent, stream)
                with self.assertRaisesRegex(TimeoutError, phase):
                    await self.call(agent)
                self.assertTrue(stream.closed)
                self.assertEqual(agent._output_buffer, [])
                self.assertEqual(request.call_count, 1)
                self.assertEqual(agent.runtime_events[-1]["event"], "model_timeout")

    async def test_connection_wait_is_bounded(self):
        agent = self.agent()
        async def wait(**kwargs):
            await asyncio.Event().wait()
        with patch.object(agent._openai_client.chat.completions, "create", side_effect=wait) as request:
            with self.assertRaisesRegex(TimeoutError, "request_started"):
                await self.call(agent)
            self.assertEqual(request.call_count, 1)

    async def test_stop_without_transport_eof_reports_the_actual_phase(self):
        agent = self.agent()
        stream = FakeStream("finish_stall")
        self.install(agent, stream)
        with self.assertRaisesRegex(TimeoutError, "model_finished"):
            await self.call(agent)
        self.assertTrue(stream.closed)

    async def test_broken_stream_with_zero_retries(self):
        agent = self.agent(model_max_retries=0)
        stream = FakeStream("broken")
        request = self.install(agent, stream)
        with self.assertRaisesRegex(RuntimeError, "ECONNRESET"):
            await self.call(agent)
        self.assertEqual(request.call_count, 1)
        self.assertTrue(stream.closed)
        self.assertEqual(agent._output_buffer, [])

    async def test_retry_backoff_shares_the_deadline(self):
        agent = self.agent(model_max_retries=2)
        request = self.install(agent, FakeStream("broken"))
        with patch("agents.agent.print_retry"), self.assertRaises(TimeoutError):
            await self.call(agent)
        # Deadline expires during the first backoff, before a second API attempt.
        self.assertEqual(request.call_count, 1)

    async def test_cancellation_propagates_and_closes(self):
        agent = self.agent()
        stream = FakeStream("silent")
        self.install(agent, stream)
        task = asyncio.create_task(agent._call_openai_stream())
        await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(stream.closed)
        self.assertEqual(agent.runtime_events[-1]["event"], "model_cancelled")

    async def test_diagnostics_are_bounded_and_do_not_record_content(self):
        agent = self.agent()
        for _ in range(200):
            agent._runtime_event("chat_started")
        self.assertEqual(len(agent.runtime_events), 128)
        self.assertNotIn("offline", repr(agent.runtime_events))

    async def test_real_read_file_records_execution_without_file_content(self):
        agent = self.agent()
        path = Path(self.temp.name) / "sample.txt"
        path.write_text("private sample 中文", encoding="utf-8")
        with patch("agents.tool_runtime.print_tool_call"):
            result = await asyncio.wait_for(agent._run_checked_tool("read_file", {"file_path": str(path)}), 2)
        self.assertIn("private sample", result)
        self.assertEqual([row["event"] for row in agent.runtime_events], ["tool_started", "tool_finished"])
        self.assertNotIn("private sample", repr(agent.runtime_events))
        self.assertNotIn(str(path), repr(agent.runtime_events))

    async def test_web_timeout_clears_busy_and_publishes_error(self):
        app = WebApp(options={"api_key": "offline", "api_protocol": "openai", "model": "test",
                              "custom_system_prompt": "test", "enable_mcp": False,
                              "enable_memory_prefetch": False, "enable_online_evolution": False,
                              "model_timeout": .1, "model_max_retries": 0})
        self.addCleanup(app.close)
        state = app.call(app.dispatch("POST", "/api/sessions", {}))
        view = app.conversations[state["id"]]
        async def run():
            view.ensure_agent()
            stream = FakeStream("silent")
            with patch.object(view.agent._openai_client.chat.completions, "create", AsyncMock(return_value=stream)):
                view.busy = True
                app.active = view.id
                await asyncio.wait_for(view.run("hello"), 2)
            self.assertFalse(view.busy)
            self.assertIsNone(app.active)
            self.assertIn("Model request exceeded", view.error)
            self.assertTrue(stream.closed)
            self.assertIsNone(view.agent._current_task)
            self.assertEqual(view.snapshot["busy"], False)
        app.call(run())
