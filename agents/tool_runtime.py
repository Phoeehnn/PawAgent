"""Shared tool validation and permission checks for both model protocols."""
from __future__ import annotations

import asyncio
from .tools import check_permission
from .ui import print_tool_call


class ToolExecution:
    async def _run_checked_tool(self, name: str, inp) -> str:
        """Validate and authorize at execution time for both model protocols."""
        from .validation import validate_json
        definition = next((tool for tool in self.tools if tool["name"] == name), None)
        if definition is None:
            return f"Tool not available in this run: {name}"
        error = validate_json(inp, definition["input_schema"])
        if error:
            return f"Error: invalid arguments for {name}: {error}"
        print_tool_call(name, inp)
        permission = check_permission(name, inp, self.permission_mode, self._plan_file_path)
        if permission["action"] == "deny":
            return f"Action denied: {permission.get('message', '')}"
        if permission["action"] == "confirm" and not await self._confirm_tool_call(name, permission):
            return "User denied this action."
        permission = check_permission(name, inp, self.permission_mode, self._plan_file_path)
        if permission["action"] == "deny":
            return f"Action denied: {permission.get('message', '')}"
        try:
            self._runtime_event("tool_started", tool=name)
            result = await self._execute_tool_call(name, inp)
            self._runtime_event("tool_finished", tool=name)
            return result
        except asyncio.CancelledError:
            self._runtime_event("tool_cancelled", tool=name)
            raise
        except Exception as exc:
            self._runtime_event("tool_error", tool=name, error_type=type(exc).__name__)
            return f"Error executing tool: {exc}"
