"""Simple agent that proxies conversation to an OpenAI chat model."""

from __future__ import annotations

import logging
import os

from collections.abc import AsyncIterable
from typing import Any, Callable, Coroutine

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

ToolFn = Callable[..., Coroutine[Any, Any, str]]

class NewsAgent:
    """Simple conversational agent that relies on an OpenAI chat model."""

    SYSTEM_INSTRUCTION = (
        "You are a fact-focused assistant. Use only the provided context snippets. "
        "If the answer is not grounded in the snippets, respond with 'I don't know.' "
        "Provide concise answers."
    )

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(
            base_url="http://127.0.0.1:8001/v1",
        )
        self._model = model or os.getenv(
            "OPENAI_NEWS_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano")
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
        #         "name": "mock_property_search",
        #         "description": "Return pretend rental listings.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "city": {"type": "string"},
        #                 "guests": {"type": "integer", "minimum": 1},
        #             },
        #             "required": ["city"],
        #         },
        #     },
        #     {
        #         "name": "mock_price_estimate",
        #         "description": "Return a pretend nightly price estimate.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "city": {"type": "string"},
        #                 "season": {"type": "string"},
        #             },
        #         },
        #     },
        # ]
        logger.info("NewsAgent initialized with model %s", self._model)

    def _history_for_session(self, session_id: str) -> list[ChatCompletionMessageParam]:
        return self._session_history.setdefault(session_id, [])

    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        """Invoke the agent for the provided query.

        Args:
            query: User prompt text.
            session_id: Identifier for maintaining conversation context.

        Returns:
            Structured payload used by the executor layer.
        """
        logger.debug(
            "NewsAgent.ainvoke query=%s session_id=%s", query, session_id
        )
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
            choice = response.choices[0].message
            content = choice.content or "I'm sorry, I was unable to generate a response."


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

            history.extend(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": content},
                ]
            )

            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": content,
            }

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """Asynchronous generator that yields a single response event."""
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
