"""Offline persistence regressions. Set TEMP/TMP to a D: directory on Windows."""
import tempfile
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents import session, storage
from agents.web import WebAgent
from unittest.mock import AsyncMock


class PersistenceRegressionTests(unittest.TestCase):
    def test_real_web_save_writes_once_without_readback(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(session, "SESSION_DIR", Path(directory)), \
                patch("pathlib.Path.home", return_value=Path(directory)):
            view = SimpleNamespace(title="中文会话", entries=[{"text": "你好😀"}],
                                   confirm=AsyncMock(), approve_plan=AsyncMock())
            agent = WebAgent(view, api_key="offline", api_protocol="openai", custom_system_prompt="test",
                             enable_mcp=False, enable_memory_prefetch=False, enable_online_evolution=False)
            self.addCleanup(lambda: asyncio.run(agent.aclose()))
            # Use the real Agent serializer and storage primitive; only count I/O.
            with patch("agents.session.write_json", wraps=storage.write_json) as write, \
                    patch.object(Path, "read_text", side_effect=AssertionError("unexpected readback")):
                agent._openai_messages.append({"role": "user", "content": "你好😀"})
                agent._auto_save()
                self.assertEqual(write.call_count, 1)
            saved = session.load_session(agent.session_id)
            self.assertEqual(saved["webEntries"], view.entries)
            self.assertEqual(saved["openaiMessages"][-1]["content"], "你好😀")

    def test_legacy_utf8_load_and_list_under_gbk_default(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(session, "SESSION_DIR", Path(directory)):
                data = {"metadata": {"cwd": directory, "title": "中文😀"},
                        "openaiMessages": [{"role": "user", "content": "你好😀"}]}
                session.save_session("legacy", data)
                original = Path.read_text

                def gbk_default(path, *args, **kwargs):
                    kwargs.setdefault("encoding", "gbk")
                    return original(path, *args, **kwargs)

                with patch.object(Path, "read_text", gbk_default):
                    self.assertEqual(session.load_session("legacy")["openaiMessages"], data["openaiMessages"])
                    self.assertEqual(len(session.list_sessions()), 1)

    def test_web_save_propagates_failure_and_supplies_entries(self):
        agent = object.__new__(WebAgent)
        agent.view = SimpleNamespace(title="中文", entries=[{"text": "内容"}])
        with patch("agents.agent.Agent._auto_save", side_effect=OSError("disk full")) as save:
            with self.assertRaises(OSError):
                agent._auto_save()
            save.assert_called_once_with(strict=True, extra_data={"webEntries": agent.view.entries})

    def test_atomic_replace_failure_preserves_existing_session(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            storage.write_json(path, {"old": "中文"})
            before = path.read_bytes()
            with patch("agents.storage.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    storage.write_json(path, {"new": "内容"})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
