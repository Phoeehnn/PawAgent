"""Offline preflight invariants: reuse sizing without dropping context or folds."""
import asyncio
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from agents.agent import Agent
from agents.tool_results import compact_tool_results, SNIP_PLACEHOLDER


MEMORY = {"episode_memory": {"task_description": "保留原始任务😀", "key_events": [], "current_progress": ""},
          "working_memory": {"immediate_goal": "继续回答", "current_challenges": "", "next_actions": []},
          "tool_memory": {"tools_used": [], "derived_rules": []}}


class ContextPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for patcher in (patch("pathlib.Path.home", return_value=Path(self.temp.name)),
                        patch("agents.agent.print_info"), patch("agents.agent.save_folded_session_memory")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def agent(self, protocol):
        agent = Agent(api_key="offline", api_protocol=protocol, custom_system_prompt="离线系统提示😀",
                      context_window=32000, enable_mcp=False, enable_memory_prefetch=False,
                      enable_online_evolution=False)
        self.addAsyncCleanup(agent.aclose)
        return agent

    def messages(self, agent):
        return agent._openai_messages if agent.use_openai else agent._anthropic_messages

    def history(self, agent, count=1):
        messages = self.messages(agent)
        messages.append({"role": "user", "content": "保留原始任务😀"})
        for index in range(count):
            call_id = str(index)
            if agent.use_openai:
                messages.extend([
                    {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "read_file", "arguments": '{"file_path":"中文.txt"}'}}]},
                    {"role": "tool", "tool_call_id": call_id, "content": "文件内容😀" * 100},
                ])
            else:
                messages.extend([
                    {"role": "assistant", "content": [{"type": "tool_use", "id": call_id,
                        "name": "read_file", "input": {"file_path": "中文.txt"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id,
                        "content": "文件内容😀" * 100}]},
                ])

    async def prepare(self, agent):
        await asyncio.wait_for(agent._prepare_model_context(), 2)

    async def test_no_fold_serializes_once_and_preserves_complete_payload(self):
        for protocol in ("openai", "anthropic"):
            agent = self.agent(protocol)
            self.history(agent)
            before = deepcopy((self.messages(agent), agent.tools, agent._system_prompt))
            measure = agent._context_request_bytes
            with patch.object(agent, "_context_request_bytes", wraps=measure) as sizing:
                await self.prepare(agent)
                self.assertEqual(sizing.call_count, 1)
            self.assertEqual((self.messages(agent), agent.tools, agent._system_prompt), before)
            self.assertEqual(agent._last_request_bytes, measure())
            row = agent.runtime_events[-1]
            self.assertEqual(row["event"], "context_prepared")
            self.assertEqual(row["estimated_context_bytes"], measure())
            self.assertNotIn("原始任务", repr(row))

    async def test_existing_compression_precedes_sizing_and_keeps_protocol_policy(self):
        for protocol in ("openai", "anthropic"):
            agent = self.agent(protocol)
            self.history(agent, 6)
            agent.last_input_token_count = int(agent.effective_window * .65)
            expected = deepcopy(self.messages(agent))
            compact_tool_results(expected, use_openai=agent.use_openai, utilization=.65)
            with patch.object(agent, "_context_request_bytes", wraps=agent._context_request_bytes) as sizing:
                await self.prepare(agent)
                self.assertEqual(sizing.call_count, 1)
            self.assertEqual(self.messages(agent), expected)
            self.assertIn(SNIP_PLACEHOLDER, repr(expected))
            self.assertEqual(agent._last_request_bytes, agent._context_request_bytes())

    async def test_successful_fold_recounts_new_history_and_system(self):
        for protocol in ("openai", "anthropic"):
            agent = self.agent(protocol)
            self.history(agent, 2)
            agent.last_input_token_count = agent.effective_window
            before = agent._context_request_bytes()
            with patch.object(agent, "_generate_folded_session_memory", AsyncMock(return_value=deepcopy(MEMORY))), \
                    patch.object(agent, "_context_request_bytes", wraps=agent._context_request_bytes) as sizing:
                await self.prepare(agent)
                self.assertEqual(sizing.call_count, 2)
            self.assertEqual(agent._fold_count, 1)
            self.assertEqual(agent._last_request_bytes, agent._context_request_bytes())
            self.assertLess(agent._last_request_bytes, before)
            self.assertIn("保留原始任务", repr(self.messages(agent)))

    async def test_failed_fold_recounts_but_retains_original_messages(self):
        for protocol in ("openai", "anthropic"):
            agent = self.agent(protocol)
            self.history(agent, 2)
            agent.last_input_token_count = agent.effective_window
            before = deepcopy(self.messages(agent))
            with patch.object(agent, "_generate_folded_session_memory", AsyncMock(return_value=None)), \
                    patch.object(agent, "_context_request_bytes", wraps=agent._context_request_bytes) as sizing:
                await self.prepare(agent)
                self.assertEqual(sizing.call_count, 2)
            self.assertEqual(self.messages(agent), before)
            self.assertEqual(agent._fold_count, 0)
            self.assertEqual(agent._last_request_bytes, agent._context_request_bytes())

    async def test_large_new_input_still_triggers_fold_with_small_previous_usage(self):
        for protocol in ("openai", "anthropic"):
            agent = self.agent(protocol)
            self.history(agent)
            agent.last_input_token_count = 1
            agent._last_request_bytes = agent._context_request_bytes()
            self.messages(agent).append({"role": "user", "content": "新增内容😀" * 20000})
            with patch.object(agent, "_generate_folded_session_memory", AsyncMock(return_value=deepcopy(MEMORY))):
                await self.prepare(agent)
            self.assertEqual(agent._fold_count, 1)
            self.assertEqual(agent._last_request_bytes, agent._context_request_bytes())

    async def test_disabled_aborted_and_exhausted_folds_still_size_once(self):
        for protocol in ("openai", "anthropic"):
            for option in ("disabled", "aborted", "limit"):
                agent = self.agent(protocol)
                self.history(agent)
                agent.last_input_token_count = agent.effective_window
                if option == "disabled":
                    agent.enable_folding = False
                elif option == "aborted":
                    agent._aborted = True
                else:
                    agent.max_fold_limit = 0
                with patch.object(agent, "_compact_conversation", AsyncMock()) as compact, \
                        patch.object(agent, "_context_request_bytes", wraps=agent._context_request_bytes) as sizing:
                    await self.prepare(agent)
                    compact.assert_not_awaited()
                    self.assertEqual(sizing.call_count, 1)

    async def test_next_preflight_does_not_reuse_size_after_history_changes(self):
        for protocol in ("openai", "anthropic"):
            agent = self.agent(protocol)
            await self.prepare(agent)
            before = agent._last_request_bytes
            self.history(agent)
            await self.prepare(agent)
            self.assertGreater(agent._last_request_bytes, before)
            self.assertEqual(agent._last_request_bytes, agent._context_request_bytes())

    async def test_cancellation_during_fold_keeps_previous_size_anchor(self):
        for protocol in ("openai", "anthropic"):
            agent = self.agent(protocol)
            self.history(agent, 2)
            agent.last_input_token_count = agent.effective_window
            agent._last_request_bytes = 123
            entered = asyncio.Event()
            async def fold(*args):
                entered.set()
                await asyncio.Event().wait()
            with patch.object(agent, "_generate_folded_session_memory", fold):
                task = asyncio.create_task(agent._prepare_model_context())
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 2)
            self.assertEqual(agent._last_request_bytes, 123)
