from collections.abc import Callable

import httpx

from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from dotenv import load_dotenv


load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    # Remote agents may perform long-running tasks; allow ample time for their
    # responses before surfacing a timeout to the caller.
    DEFAULT_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=30.0)

    def __init__(self, agent_card: AgentCard, agent_url: str):
        print(f'agent_card: {agent_card}')
        print(f'agent_url: {agent_url}')
        self._httpx_client = httpx.AsyncClient(timeout=self.DEFAULT_HTTP_TIMEOUT)
        self.agent_client = A2AClient(
            self._httpx_client, agent_card, url=agent_url
        )
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(
        self, message_request: SendMessageRequest
    ) -> SendMessageResponse:
        return await self.agent_client.send_message(message_request)
