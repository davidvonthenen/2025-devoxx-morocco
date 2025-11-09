"""Command line entrypoint for running the Hotel agent service."""

from __future__ import annotations

import os

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from .agent_executor import HotelAgentExecutor
from .hotel_agent import HotelAgent

load_dotenv(override=True)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 10002
DEFAULT_LOG_LEVEL = "info"


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, log_level: str = DEFAULT_LOG_LEVEL) -> None:
    agent_executor = HotelAgentExecutor()
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
        id="hotel_search",
        name="Search hotel accommodation",
        description="Helps with accommodation search",
        tags=["hotel accommodation"],
        examples=[
            "Please find a room in LA, CA, April 15, 2025, checkout date is april 18, 2 adults"
        ],
    )
    app_url = os.environ.get("APP_URL", f"http://{host}:{port}")

    return AgentCard(
        name="Hotel Agent",
        description="Helps with searching accommodation",
        url=app_url,
        version="1.0.0",
        default_input_modes=HotelAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=HotelAgent.SUPPORTED_CONTENT_TYPES,
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
