"""Utility helpers for coordinating MCP tool calls within chat loops."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, MutableSequence, Sequence

from .mcp_client import MCPClient, MCPInvocationError


LOGGER = logging.getLogger(__name__)


class ToolBus:
    """Routes tool calls from the model to MCP servers."""

    def __init__(self, client: MCPClient, *, max_depth: int = 3) -> None:
        self._client = client
        self._max_depth = max_depth

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def has_tools(self) -> bool:
        return self._client.has_tools()

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return self._client.tools

    def augment_messages(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.has_tools():
            return list(messages)

        instruction = (
            "You may use the following tools via function calling when helpful: "
            + ", ".join(self._client.tool_names)
            + ". Return tool calls using the OpenAI tool_calls schema."
        )

        updated = [dict(m) for m in messages]
        for msg in updated:
            if msg.get("role") == "system":
                msg["content"] = msg.get("content", "") + "\n\n" + instruction
                break
        else:
            updated.insert(0, {"role": "system", "content": instruction})
        return updated

    # ------------------------------------------------------------------
    # Tool loop
    # ------------------------------------------------------------------
    def run_chat_loop(
        self,
        llm,
        base_messages: Sequence[Dict[str, Any]],
        *,
        llm_kwargs: Mapping[str, Any],
        original_prompt: str | None = None,
    ) -> Dict[str, Any]:
        """Iteratively query the model while resolving tool calls."""

        if not self.has_tools():
            return llm.chat(base_messages, **dict(llm_kwargs))

        messages: MutableSequence[Dict[str, Any]] = list(base_messages)
        tools = self.tool_schemas()
        response: Dict[str, Any] | None = None

        for depth in range(self._max_depth):
            LOGGER.info("Invoking LLM turn %s with %s messages", depth + 1, len(messages))
            response = llm.chat(
                messages,
                tools=tools,
                tool_choice="auto",
                **dict(llm_kwargs),
            )

            choice = (response or {}).get("choices", [{}])[0]
            assistant_message = dict(choice.get("message", {}))
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                LOGGER.info("No tool calls produced on turn %s", depth + 1)
                break

            LOGGER.info("Processing %s tool call(s) from LLM", len(tool_calls))
            for call in tool_calls:
                call_id = call.get("id") or call.get("tool_call_id") or call.get("function", {}).get("name", "call")
                function_info = call.get("function", {})
                tool_name = function_info.get("name")
                if not tool_name:
                    LOGGER.warning("Tool call missing function name: %s", call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": "unknown",
                            "content": json.dumps(
                                {
                                    "error": {
                                        "type": "ToolInvocationError",
                                        "tool": None,
                                        "message": "Tool call missing function name",
                                    }
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                raw_args = function_info.get("arguments", "{}")
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                    if not isinstance(parsed_args, dict):
                        raise ValueError("Tool arguments must decode to an object")
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Invalid tool arguments for %s: %s", tool_name, raw_args)
                    error_payload = {
                        "error": {
                            "type": "ToolArgumentsError",
                            "tool": tool_name,
                            "message": str(exc),
                        }
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name or "unknown",
                            "content": json.dumps(error_payload, ensure_ascii=False),
                        }
                    )
                    continue

                invocation_args = dict(parsed_args)
                if original_prompt and "prompt" not in invocation_args:
                    invocation_args["prompt"] = original_prompt

                try:
                    result = self._client.invoke(tool_name, invocation_args)
                except MCPInvocationError as exc:
                    LOGGER.exception("Tool invocation failed for %s", tool_name)
                    result = {
                        "error": {
                            "type": "ToolInvocationError",
                            "tool": tool_name,
                            "message": str(exc),
                        }
                    }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        if response is None:
            raise RuntimeError("LLM did not return a response")
        return response


__all__ = ["ToolBus"]

