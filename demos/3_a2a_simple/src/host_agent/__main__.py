"""Gradio interface for the host routing agent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import gradio as gr

from .routing_agent import RoutingAgent, initialize_routing_agent

SESSION_ID = "default_session"

ROUTING_AGENT: RoutingAgent | None = None


async def get_response_from_agent(
    message: str, history: list[gr.ChatMessage]
) -> AsyncIterator[gr.ChatMessage]:
    if ROUTING_AGENT is None:
        raise RuntimeError("Routing agent is not initialized")

    responses = await ROUTING_AGENT.handle_user_message(message, SESSION_ID)
    for response in responses:
        yield gr.ChatMessage(role="assistant", content=response)


async def main() -> None:
    global ROUTING_AGENT
    ROUTING_AGENT = await initialize_routing_agent()

    with gr.Blocks(title="A2A Host Agent") as demo:
        gr.Image(
            "https://a2a-protocol.org/latest/assets/a2a-logo-black.svg",
            width=100,
            height=100,
            scale=0,
            show_label=False,
            show_download_button=False,
            container=False,
            show_fullscreen_button=False,
        )
        gr.ChatInterface(
            get_response_from_agent,
            title="A2A Host Agent",
            description="This assistant can help you check the weather and find Airbnb accommodation.",
        )

    demo.queue().launch(server_name="127.0.0.1", server_port=11000)


if __name__ == "__main__":
    asyncio.run(main())
