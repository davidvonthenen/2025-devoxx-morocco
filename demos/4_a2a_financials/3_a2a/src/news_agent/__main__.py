"""Command line entrypoint for running the News agent service."""

from __future__ import annotations

import os

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from .agent_executor import NewsAgentExecutor
from .news_agent import NewsAgent

load_dotenv(override=True)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 10001
DEFAULT_LOG_LEVEL = "info"


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, log_level: str = DEFAULT_LOG_LEVEL) -> None:
    agent_executor = NewsAgentExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    a2a_server = A2AStarletteApplication(
        agent_card=get_agent_card(host, port), http_handler=request_handler
    )

    uvicorn.run(
        a2a_server.build(),
        host=host,
        port=port,
        log_level=log_level.lower(),
    )


def get_agent_card(host: str, port: int) -> AgentCard:
    capabilities = AgentCapabilities(streaming=True, push_notifications=True)
    skill = AgentSkill(
        id="news_search",
        name="Search news",
        description="Helps with news search",
        tags=["news search"],
        examples=[
            "Find recent news articles about Google."
        ],
    )
    app_url = os.environ.get("APP_URL", f"http://{host}:{port}")

    return AgentCard(
        name="News Agent",
        description="Helps with searching accommodation",
        url=app_url,
        version="1.0.0",
        default_input_modes=NewsAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=NewsAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )


@click.command()
@click.option("--host", "host", default=DEFAULT_HOST)
@click.option("--port", "port", default=DEFAULT_PORT, type=int)
@click.option("--log-level", "log_level", default=DEFAULT_LOG_LEVEL)
def cli(host: str, port: int, log_level: str) -> None:
    main(host, port, log_level)


if __name__ == "__main__":
    main()
