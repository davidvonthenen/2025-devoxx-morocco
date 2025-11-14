"""A2A executor wrapper for the News agent."""

from __future__ import annotations

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact

from .news_agent import NewsAgent

logger = logging.getLogger(__name__)


class NewsAgentExecutor(AgentExecutor):
    """Executes the News agent for a2a requests."""

    def __init__(self) -> None:
        super().__init__()
        self.agent = NewsAgent()

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        if not context.message:
            raise RuntimeError("No message provided")

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        result = await self.agent.ainvoke(query, task.context_id)

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                append=False,
                context_id=task.context_id,
                task_id=task.id,
                last_chunk=True,
                artifact=new_text_artifact(
                    name="current_result",
                    description="Result of request to agent.",
                    text=result["content"],
                ),
            )
        )

        status_state = (
            TaskState.input_required
            if result.get("require_user_input")
            else TaskState.completed
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=status_state,
                    message=new_agent_text_message(
                        result["content"], task.context_id, task.id
                    ),
                ),
                final=True,
                context_id=task.context_id,
                task_id=task.id,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise RuntimeError("cancel not supported")
