"""CLI client for interacting with the local MCP-enabled LLM agent."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

import requests

from ..common.config import load_settings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local agent server")
    parser.add_argument("--question", required=True, help="User question to send to the model")
    parser.add_argument(
        "--host", default=None, help="Server host (defaults to SERVER_HOST from settings)"
    )
    parser.add_argument(
        "--port", default=None, help="Server port (defaults to SERVER_PORT from settings)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    settings = load_settings()
    host = args.host or settings.server_host
    port = int(args.port or settings.server_port)
    url = f"http://{host}:{port}/v1/chat/completions"

    payload = {
        "model": "local-llama",
        "messages": [{"role": "user", "content": args.question}],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 512,
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    answer = message.get("content", "").strip()
    print("\n\nAssistant:\n")
    print(answer)
    print("\n")

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        print("Tool calls:")
        for call in tool_calls:
            name = call.get("function", {}).get("name", "unknown")
            arguments = call.get("function", {}).get("arguments", "{}")
            print(f"  - {name}: {arguments}")
        print("\n")

    print("Raw response:\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
