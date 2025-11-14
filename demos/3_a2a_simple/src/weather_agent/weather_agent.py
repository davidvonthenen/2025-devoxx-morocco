"""Weather agent that proxies requests to an OpenAI chat model."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterable
from typing import Any, Callable, Coroutine
import json

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

ToolFn = Callable[..., Coroutine[Any, Any, str]]

class WeatherAgent:
    """Interact with an OpenAI chat model to answer weather questions."""

    SYSTEM_INSTRUCTION = (
        # "You are a weather assistant. Use the provided tools to gather real data from "
        # "weather.gov. When you respond to the user, cite the data you used and keep the "
        # "answer concise and actionable."
        "You are a weather assistant. "
        "Create fictional but realistic weather forecasts based on user queries. "
        "Assume Seattle, WA is experiencing persistent rain while Long Beach, CA "
        "enjoys sunny skies, and weave those conditions into your replies."
    )

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model or os.getenv(
            "OPENAI_WEATHER_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano")
        )
        self._session_history: dict[str, list[ChatCompletionMessageParam]] = {}
        # Exmples of tool functions that could be integrated.
        # self._tools: dict[str, ToolFn] = {
        #     "get_alerts": get_alerts,
        #     "get_forecast": get_forecast,
        #     "get_forecast_by_city": get_forecast_by_city,
        # }
        self._function_specs: list[dict[str, Any]] | None = None
        # Example: enable function calling with two mock functions.
        # self._function_specs = [
        #     {
        #         "name": "mock_weather_lookup",
        #         "description": "Return pretend forecast data for a city.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "city": {"type": "string"},
        #                 "state": {"type": "string"},
        #             },
        #             "required": ["city", "state"],
        #         },
        #     },
        #     {
        #         "name": "mock_weather_alerts",
        #         "description": "Return pretend weather alerts for a region.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "region": {"type": "string"},
        #             },
        #             "required": ["region"],
        #         },
        #     },
        # ]
        logger.info("WeatherAgent initialized with model %s", self._model)

    def _history_for_session(self, session_id: str) -> list[ChatCompletionMessageParam]:
        return self._session_history.setdefault(session_id, [])

    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        logger.debug("WeatherAgent.ainvoke query=%s session_id=%s", query, session_id)
        history = self._history_for_session(session_id)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            *history,
            {"role": "user", "content": query},
        ]

        while True:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                # functions=self._function_specs,
                # function_call="auto",
                # temperature=0.2, # BAD
            )
            message = response.choices[0].message
            content = message.content or "I was unable to generate a response."

            # if message.function_call:
            #     tool_name = message.function_call.name
            #     arguments_json = message.function_call.arguments or "{}"
            #     try:
            #         arguments = json.loads(arguments_json)
            #     except json.JSONDecodeError:
            #         logger.error("Failed to parse tool arguments: %s", arguments_json)
            #         tool_output = "Unable to parse tool arguments."
            #     else:
            #         tool_output = await self._invoke_tool(tool_name, arguments)

            #     assistant_message: ChatCompletionMessageParam = {
            #         "role": "assistant",
            #         "content": message.content,
            #         "function_call": message.function_call,
            #     }
            #     messages.append(assistant_message)
            #     history.append(assistant_message)

            #     tool_response: ChatCompletionMessageParam = {
            #         "role": "function",
            #         "name": tool_name,
            #         "content": tool_output,
            #     }
            #     messages.append(tool_response)
            #     history.append(tool_response)
            #     continue

            assistant_message = {"role": "assistant", "content": content}
            messages.append(assistant_message)
            history.extend([
                {"role": "user", "content": query},
                assistant_message,
            ])

            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": content,
            }

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        yield await self.ainvoke(query, session_id)

    async def _invoke_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Placeholder retained for future tool integrations."""

        # Example: connect the mock functions defined in ``self._function_specs``.
        # tool = self._tools.get(name)
        # if tool is None:
        #     logger.error("Unknown tool requested: %s", name)
        #     return f"Tool '{name}' is not supported."

        # try:
        #     return await tool(**arguments)
        # except Exception as exc:  # pragma: no cover - defensive logging
        #     logger.exception("Error executing tool %s", name)
        #     return f"An error occurred while executing {name}: {exc}"

        logger.debug("Tool calling is disabled. name=%s arguments=%s", name, arguments)
        return "Tool calling is currently disabled."
