"""Workflow regression tests for the host routing agent."""

from __future__ import annotations

import asyncio
from a2a.types import Message, Part, Role, Task, TaskState, TaskStatus, TextPart

from host_agent.policy_manager import TravelPolicyManager
from host_agent.routing_agent import RoutingAgent


class StubRoutingAgent(RoutingAgent):
    """Routing agent variant that returns canned responses for remote calls."""

    def __init__(self, responses: dict[str, list[str]]) -> None:
        super().__init__(policy_manager=TravelPolicyManager())
        self._weather_agent_name = "Weather Specialist"
        self._hotel_agent_name = "Hotel Specialist"
        self._responses: dict[str, list[str]] = {
            name: list(outputs) for name, outputs in responses.items()
        }

    async def _send_message(  # type: ignore[override]
        self, agent_name: str, task: str, session_id: str
    ) -> Task:
        if agent_name not in self._responses:
            raise AssertionError(f"No stubbed responses for agent {agent_name!r}")
        queue = self._responses[agent_name]
        if not queue:
            raise AssertionError(
                f"Stubbed responses for agent {agent_name!r} were exhausted"
            )

        output = queue.pop(0)
        context_key = (session_id, agent_name)
        context_id = f"ctx-{agent_name}"
        self._session_context_ids[context_key] = context_id

        message = Message(
            messageId=f"msg-{agent_name}-{len(queue)}",
            role=Role.agent,
            parts=[Part(root=TextPart(text=output))],
        )
        return Task(
            id=f"task-{agent_name}-{len(queue)}",
            contextId=context_id,
            status=TaskStatus(state=TaskState.completed, message=message),
        )


def _run_agent(agent: RoutingAgent, message: str, session_id: str) -> list[str]:
    return asyncio.run(agent.handle_user_message(message, session_id=session_id))


def _has_hotel_suggestions(chunks: list[str]) -> bool:
    return any(chunk.startswith("hotel ideas:") for chunk in chunks)


def test_hotel_blocked_for_rainy_seattle() -> None:
    agent = StubRoutingAgent(
        {
            "Weather Specialist": [
                "Severe storm warning with dangerous flooding expected for Seattle.",
            ]
        }
    )

    responses = _run_agent(
        agent,
        "I'd like to book a hotel in Seattle, WA this weekend.",
        session_id="session-block",
    )

    assert not _has_hotel_suggestions(
        responses
    ), "Hazardous weather should block rental suggestions."
    assert "hazardous conditions" in responses[-1]


def test_hotel_allowed_for_sunny_long_beach() -> None:
    agent = StubRoutingAgent(
        {
            "Weather Specialist": [
                "Warm sunshine and gentle breezes over Long Beach, CA all weekend.",
            ],
            "Hotel Specialist": [
                "Consider the Ocean Breeze loft a short walk from the Long Beach pier.",
            ],
        }
    )

    responses = _run_agent(
        agent,
        "Help me secure a hotel in Long Beach, CA.",
        session_id="session-allow",
    )

    assert _has_hotel_suggestions(
        responses
    ), "Clear weather should allow the Hotel specialist to respond."
    assert "Here are some rental ideas" in responses[-1]
    assert "Ocean Breeze loft" in responses[-1]


def test_session_recovers_after_blocked_destination() -> None:
    agent = StubRoutingAgent(
        {
            "Weather Specialist": [
                "Severe storm warning with dangerous flooding expected for Seattle.",
                "Bright sunshine and calm skies over Long Beach, CA.",
            ],
            "Hotel Specialist": [
                "Try the Sunny Surf bungalow near Long Beach's waterfront.",
            ],
        }
    )

    seattle_responses = _run_agent(
        agent,
        "Book a hotel in Seattle, WA for next Friday.",
        session_id="session-switch",
    )
    assert "hazardous conditions" in seattle_responses[-1]

    long_beach_responses = _run_agent(
        agent,
        "Okay, how about a hotel in Long Beach, CA instead?",
        session_id="session-switch",
    )
    assert _has_hotel_suggestions(
        long_beach_responses
    ), "Once a safe forecast arrives, the agent should surface rentals."
    assert "Here are some rental ideas" in long_beach_responses[-1]
    assert "Sunny Surf bungalow" in long_beach_responses[-1]
