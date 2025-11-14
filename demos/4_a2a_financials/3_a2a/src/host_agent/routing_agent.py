"""Routing agent orchestrated by a LangGraph policy (News/Financial)."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, TypedDict

import httpx
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    DataPart,
    FilePart,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TextPart,
)
from dotenv import load_dotenv

try:  # pragma: no cover
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover
    from .langgraph_stub import END, StateGraph  # type: ignore[no-redef]

from typing_extensions import NotRequired

from .policy_manager import PolicyClassification, NewsFinancePolicyManager
from .remote_agent_connection import RemoteAgentConnections

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

DEFAULT_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=30.0)


class HostGraphState(TypedDict, total=False):
    """State container passed between LangGraph nodes."""

    user_message: str
    session_id: str

    response_chunks: list[str]
    policy_notes: list[str]

    # Policy outcomes
    policy_route: NotRequired[str]
    tickers: NotRequired[list[str]]

    # Outputs from specialists
    news_output: NotRequired[str | None]
    finance_output: NotRequired[str | None]


class RoutingAgent:
    """Delegates user requests to remote agents using a LangGraph policy."""

    def __init__(
        self,
        *,
        policy_manager: NewsFinancePolicyManager | None = None,
    ) -> None:
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self._session_history: dict[str, list[dict[str, str]]] = {}
        self._session_context_ids: dict[tuple[str, str], str] = {}

        self._policy_manager = policy_manager or NewsFinancePolicyManager()
        self._graph = self._build_graph()

        # Discovered specialists
        self._news_agent_name: str | None = None
        self._finance_agent_name: str | None = None

        logger.info("RoutingAgent initialized with LangGraph policy orchestrator")

    async def _async_init_components(
        self, remote_agent_addresses: list[str]
    ) -> None:
        async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                except Exception as exc:
                    logger.error("Failed to load agent card from %s: %s", address, exc)
                    continue

                remote_connection = RemoteAgentConnections(
                    agent_card=card, agent_url=address
                )
                self.remote_agent_connections[card.name] = remote_connection
                self.cards[card.name] = card
                self._maybe_track_specialist(card)

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
    ) -> RoutingAgent:
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def list_remote_agents(self) -> list[dict[str, str]]:
        return [{"name": c.name, "description": c.description or ""} for c in self.cards.values()]

    async def handle_user_message(self, message: str, session_id: str) -> list[str]:
        history = self._history_for_session(session_id)
        history.append({"role": "user", "content": message})

        initial_state: HostGraphState = {"user_message": message, "session_id": session_id}
        final_state = await self._graph.ainvoke(initial_state)
        responses = final_state.get("response_chunks", [])

        if not responses:
            fallback = "I don't know."
            history.append({"role": "assistant", "content": fallback})
            return [fallback]

        history.extend({"role": "assistant", "content": text} for text in responses)
        return responses

    def _build_graph(self):
        graph = StateGraph(HostGraphState)

        graph.add_node("classify_request", self._classify_request)
        graph.add_node("evaluate_policy", self._evaluate_policy)
        graph.add_node("fetch_news", self._fetch_news)
        graph.add_node("fetch_finance", self._fetch_finance)
        graph.add_node("compose_response", self._compose_response)

        graph.set_entry_point("classify_request")
        graph.add_edge("classify_request", "evaluate_policy")
        graph.add_conditional_edges(
            "evaluate_policy",
            self._route_policy,
            {
                "fetch_news": "fetch_news",
                "fetch_finance": "fetch_finance",
                "respond": "compose_response",
            },
        )
        graph.add_edge("fetch_news", "compose_response")
        graph.add_edge("fetch_finance", "compose_response")
        graph.add_edge("compose_response", END)

        return graph.compile()

    async def _send_message(self, agent_name: str, task: str, session_id: str) -> Task | None:
        connection = self.remote_agent_connections.get(agent_name)
        if connection is None:
            logger.error("Unknown agent requested: %s", agent_name)
            return None

        context_key = (session_id, agent_name)
        context_id = self._session_context_ids.get(context_key)
        message_id = uuid.uuid4().hex
        payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
            }
        }
        if context_id:
            payload["message"]["contextId"] = context_id

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await connection.send_message(
            message_request=message_request
        )

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            logger.error("Received non-success response from %s", agent_name)
            return None

        result = send_response.root.result
        if not isinstance(result, Task):
            logger.error("Received non-task response from %s", agent_name)
            return None

        self._session_context_ids[context_key] = result.context_id
        return result

    def _extract_task_output(self, task: Task | None) -> str:
        """Collect text from both status.message.parts and any artifacts.parts, without duplicates."""
        if task is None:
            return ""

        def parts_to_text(parts: list[Part]) -> list[str]:
            out: list[str] = []
            for p in parts or []:
                try:
                    if isinstance(p.root, TextPart):
                        out.append(p.root.text)
                    elif isinstance(p.root, DataPart):
                        out.append(json.dumps(p.root.data, indent=2))
                    elif isinstance(p.root, FilePart):
                        mt = p.root.file.mime_type or "unknown mime type"
                        out.append(f"Received file content ({mt}).")
                except Exception:
                    # Defensive: ignore malformed parts
                    continue
            return [t for t in out if t is not None]

        status_texts: list[str] = []
        artifact_texts: list[str] = []

        # Status message parts (often a terse single line)
        try:
            if getattr(task, "status", None) and getattr(task.status, "message", None):
                if getattr(task.status.message, "parts", None):
                    status_texts.extend(parts_to_text(task.status.message.parts))
        except Exception:
            pass

        # Artifact parts (often a full report that repeats the status line)
        try:
            if getattr(task, "artifacts", None):
                for art in (task.artifacts or []):
                    if getattr(art, "parts", None):
                        artifact_texts.extend(parts_to_text(art.parts))
        except Exception:
            pass

        # If artifacts fully subsume status snippets, prefer artifacts-only.
        artifact_blob = "\n".join(artifact_texts)
        if artifact_texts and status_texts:
            if all((s or "").strip() in artifact_blob for s in status_texts):
                status_texts = []

        merged_blocks = status_texts + artifact_texts

        # Final pass: remove exact duplicate lines while preserving order.
        seen: set[str] = set()
        dedup_lines: list[str] = []
        for block in merged_blocks:
            for line in (block or "").splitlines():
                line_norm = line.strip()
                if not line_norm:
                    # collapse multiple blank lines down to a single blank line
                    if dedup_lines and dedup_lines[-1] != "":
                        dedup_lines.append("")
                    continue
                if line_norm not in seen:
                    seen.add(line_norm)
                    dedup_lines.append(line_norm)
            # paragraph break between blocks
            if dedup_lines and dedup_lines[-1] != "":
                dedup_lines.append("")

        # trim trailing blank
        if dedup_lines and dedup_lines[-1] == "":
            dedup_lines.pop()

        return "\n".join(dedup_lines)


    def _maybe_track_specialist(self, card: AgentCard) -> None:
        """Record which remote cards correspond to news or financial experts."""
        lowered_name = card.name.lower()
        lowered_desc = (card.description or "").lower()

        if self._news_agent_name is None and ("news" in lowered_name or "news" in lowered_desc):
            self._news_agent_name = card.name

        if self._finance_agent_name is None and (
            "financial" in lowered_name
            or "finance" in lowered_name
            or "stock" in lowered_name
            or "market" in lowered_name
            or "financial" in lowered_desc
            or "finance" in lowered_desc
            or "stock" in lowered_desc
            or "market" in lowered_desc
        ):
            self._finance_agent_name = card.name

    def _history_for_session(self, session_id: str) -> list[dict[str, str]]:
        return self._session_history.setdefault(session_id, [])

    def _classify_request(self, state: HostGraphState) -> HostGraphState:
        classification: PolicyClassification = self._policy_manager.classify_request(
            state["user_message"]
        )
        response_chunks: list[str] = []
        policy_notes: list[str] = []
        if classification.note:
            policy_notes.append(classification.note)

        if classification.route == "finance":
            response_chunks.append(
                f"Policy check: financial intent with ticker(s) {', '.join(classification.tickers)} → routing to Financial specialist."
            )
        else:
            response_chunks.append("Policy check: routing to News specialist.")

        return {
            "session_id": state["session_id"],
            "user_message": state["user_message"],
            "response_chunks": response_chunks,
            "policy_notes": policy_notes,
            "tickers": classification.tickers,
            "policy_route": classification.route,
        }

    def _evaluate_policy(self, state: HostGraphState) -> HostGraphState:
        """Translate the classification into a concrete step, considering availability."""
        policy_notes = list(state.get("policy_notes", []))
        route = state.get("policy_route", "news")

        if route == "finance":
            if self._finance_agent_name:
                return {**state, "policy_notes": policy_notes, "policy_route": "fetch_finance"}
            if self._news_agent_name:
                policy_notes.append("Policy fallback: Financial specialist unavailable; falling back to News.")
                return {**state, "policy_notes": policy_notes, "policy_route": "fetch_news"}
            policy_notes.append("Policy fallback: No specialists available; responding directly.")
            return {**state, "policy_notes": policy_notes, "policy_route": "respond"}

        # route == "news"
        if self._news_agent_name:
            return {**state, "policy_notes": policy_notes, "policy_route": "fetch_news"}

        if self._finance_agent_name and state.get("tickers"):
            policy_notes.append("Policy fallback: News specialist unavailable; using Financial specialist due to provided ticker(s).")
            return {**state, "policy_notes": policy_notes, "policy_route": "fetch_finance"}

        policy_notes.append("Policy fallback: No specialists available; responding directly.")
        return {**state, "policy_notes": policy_notes, "policy_route": "respond"}

    def _route_policy(self, state: HostGraphState) -> str:
        return state.get("policy_route", "respond")

    async def _fetch_news(self, state: HostGraphState) -> HostGraphState:
        responses = list(state.get("response_chunks", []))
        session_id = state["session_id"]
        agent_name = self._news_agent_name
        if not agent_name:
            responses.append("News specialist is offline right now.")
            return {**state, "response_chunks": responses, "news_output": ""}

        task_prompt = (
            "You are the News specialist.\n"
            "Provide a concise, up-to-date news summary relevant to the user's request. "
            "If you reference articles, include short citations or source names.\n"
            f"User request:\n{state['user_message']}"
        )
        task = await self._send_message(agent_name, task_prompt, session_id)
        news_output = self._extract_task_output(task)

        if news_output:
            responses.append(news_output.strip())
        else:
            responses.append("The News specialist did not return a summary.")

        return {**state, "response_chunks": responses, "news_output": news_output}

    async def _fetch_finance(self, state: HostGraphState) -> HostGraphState:
        responses = list(state.get("response_chunks", []))
        session_id = state["session_id"]
        agent_name = self._finance_agent_name
        if not agent_name:
            responses.append("Financial specialist is offline right now.")
            return {**state, "response_chunks": responses, "finance_output": ""}

        tickers = state.get("tickers") or []
        tickers_text = ", ".join(tickers) if tickers else "N/A"

        task_prompt = (
            "You are the Financial specialist.\n"
            "Analyze the user's financial question focusing on the specified ticker symbols. "
            "Prioritize recent results, guidance, valuation context, and material news. "
            "If data is unknown, respond with 'I don't know'.\n"
            f"Tickers: {tickers_text}\n"
            f"User request:\n{state['user_message']}"
        )
        task = await self._send_message(agent_name, task_prompt, session_id)
        finance_output = self._extract_task_output(task)

        if finance_output:
            responses.append(finance_output.strip())
        else:
            responses.append("The Financial specialist did not return an analysis.")

        return {**state, "response_chunks": responses, "finance_output": finance_output}

    def _compose_response(self, state: HostGraphState) -> HostGraphState:
        responses = list(state.get("response_chunks", []))
        policy_notes = state.get("policy_notes", [])

        if policy_notes:
            responses.append("Policy summary:")
            responses.extend(f"- {note}" for note in policy_notes)

        return {**state, "response_chunks": responses}


async def initialize_routing_agent() -> RoutingAgent:
    # Card discovery is dynamic; we provide two URLs and let _maybe_track_specialist map them.
    return await RoutingAgent.create(
        remote_agent_addresses=[
            os.getenv("NEWS_AGENT_URL", "http://localhost:10001"),
            os.getenv("FIN_AGENT_URL", "http://localhost:10002"),
        ]
    )
