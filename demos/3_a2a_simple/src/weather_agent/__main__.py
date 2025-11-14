"""Command line entrypoint for the weather agent service."""

from __future__ import annotations

import os

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from .weather_executor import WeatherExecutor

load_dotenv()

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 10001


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    agent_card = build_agent_card(host, port)
    agent_executor = WeatherExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    uvicorn.run(a2a_app.build(), host=host, port=port)


def build_agent_card(host: str, port: int) -> AgentCard:
    skill = AgentSkill(
        id="weather_search",
        name="Search weather",
        description="Helps with weather in city, or states",
        tags=["weather"],
        examples=["weather in LA, CA"],
    )
    app_url = os.environ.get("APP_URL", f"http://{host}:{port}")

    return AgentCard(
        name="Weather Agent",
        description="Helps with weather",
        url=app_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


@click.command()
@click.option("--host", "host", default=DEFAULT_HOST)
@click.option("--port", "port", default=DEFAULT_PORT)
def cli(host: str, port: int) -> None:
    main(host, port)


if __name__ == "__main__":
    main()
