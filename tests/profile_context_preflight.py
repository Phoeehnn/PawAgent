"""Offline CPU comparison; no provider clients, user sessions, or exact token counts.

Run with the project venv: python tests/profile_context_preflight.py
Outputs synthetic composition/timing metrics to .bear/tmp/context-profile-comparison.json.
"""
import asyncio
from copy import deepcopy
import faulthandler
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / ".bear" / "tmp"
os.environ["TEMP"] = os.environ["TMP"] = str(TEMP)
tempfile.tempdir = str(TEMP)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from agents.agent import Agent, _to_openai_tools
from agents.prompt import SYSTEM_PROMPT_TEMPLATE
from agents.tools import tool_definitions, get_active_tool_definitions


def json_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def sample(protocol, rounds):
    # Only exercise context methods, without constructing SDK clients or loading user data.
    agent = object.__new__(Agent)
    agent.use_openai = protocol == "openai"
    agent.tools = deepcopy(tool_definitions)
    agent._system_prompt = SYSTEM_PROMPT_TEMPLATE
    agent._openai_messages = [{"role": "system", "content": agent._system_prompt}]
    agent._anthropic_messages = []
    messages = agent._openai_messages if agent.use_openai else agent._anthropic_messages
    messages.append({"role": "user", "content": "根据已读取文件总结主要功能。"})
    text = ("函数说明：这是离线构造的文件内容，用于测量请求准备开销。\n" * 300)[:8000]
    for index in range(rounds):
        call_id, arguments = str(index), {"file_path": f"sample-{index}.txt"}
        if agent.use_openai:
            messages.extend([
                {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps(arguments)}}]},
                {"role": "tool", "tool_call_id": call_id, "content": text},
            ])
        else:
            messages.extend([
                {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": "read_file", "input": arguments}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": text}]},
            ])
    agent.effective_window = 16_000_000  # Isolate CPU sizing, without triggering model-based folds.
    agent.last_input_token_count = agent.last_api_call_time = agent._fold_count = 0
    agent.auto_compact_threshold = .8
    agent.max_fold_limit = agent._last_request_bytes = None
    agent.enable_folding, agent._aborted = True, False
    return agent


def breakdown(agent):
    active = get_active_tool_definitions(agent.tools)
    tools = _to_openai_tools(active) if agent.use_openai else active
    messages = agent._openai_messages if agent.use_openai else agent._anthropic_messages
    request = {"model": "offline-profile", "tools": tools, "messages": messages, "stream": True}
    if agent.use_openai:
        request["stream_options"] = {"include_usage": True}
        output = [m["content"] for m in messages if m.get("role") == "tool"]
    else:
        request.update(system=agent._system_prompt, max_tokens=16384)
        output = [b["content"] for m in messages if isinstance(m.get("content"), list)
                  for b in m["content"] if b.get("type") == "tool_result"]
    return {"synthetic_request_json_bytes": json_bytes(request), "active_tool_schema_bytes": json_bytes(tools),
            "active_tool_count": len(tools), "system_text_utf8_bytes": len(agent._system_prompt.encode("utf-8")),
            "tool_result_text_utf8_bytes": sum(len(t.encode("utf-8")) for t in output)}


async def reference_prepare(agent):
    """Source path before this optimization, for the no-fold comparison below."""
    agent._run_compression_pipeline()
    if (not agent._aborted and agent.enable_folding and not agent._fold_limit_reached()
            and agent._estimated_input_tokens() > agent.effective_window * agent.auto_compact_threshold):
        raise AssertionError("profiling must not trigger model-based folding")
    agent._last_request_bytes = agent._context_request_bytes()


async def compare():
    rows = []
    for protocol in ("openai", "anthropic"):
        for rounds in (0, 10, 100):
            before, after = sample(protocol, rounds), sample(protocol, rounds)
            original = deepcopy(before._openai_messages if before.use_openai else before._anthropic_messages)
            operations = {"before": lambda: reference_prepare(before), "after": after._prepare_model_context}
            counts = {}
            for name, agent in (("before", before), ("after", after)):
                agent._context_request_bytes = Mock(wraps=agent._context_request_bytes)
                await operations[name]()  # Warm up and count a complete preflight.
                counts[name] = agent._context_request_bytes.call_count
            timings = {"before": [], "after": []}
            for iteration in range(25):
                for name in (("before", "after") if iteration % 2 == 0 else ("after", "before")):
                    started = time.perf_counter()
                    await operations[name]()
                    timings[name].append((time.perf_counter() - started) * 1000)
            for agent in (before, after):
                assert (agent._openai_messages if agent.use_openai else agent._anthropic_messages) == original
                assert agent._fold_count == 0
            assert breakdown(before) == breakdown(after)
            before_ms, after_ms = (statistics.median(timings[name]) for name in ("before", "after"))
            rows.append({"protocol": protocol, "tool_rounds": rounds, **breakdown(after),
                         "serializations": counts, "before_median_ms": before_ms, "after_median_ms": after_ms,
                         "reduction_percent": (1 - after_ms / before_ms) * 100})
    return {"method": "25 alternating pairs per case; identical contents; no provider clients",
            "context_window_note": "Synthetic 16M effective window isolates no-fold CPU cost; not the configured model limit.",
            "payload_note": "Static prompt template and synthetic read_file histories; not user sessions or a captured HTTP request.",
            "model_api_requests": 0, "exact_tokens_measured": False, "samples": rows}


if __name__ == "__main__":
    TEMP.mkdir(parents=True, exist_ok=True)
    faulthandler.dump_traceback_later(45, exit=True)
    try:
        report = asyncio.run(compare())
        target = TEMP / "context-profile-comparison.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
    finally:
        faulthandler.cancel_dump_traceback_later()
