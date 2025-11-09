"""Simple client to query the local RAG server."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

import requests

from ..common.config import load_settings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local RAG server")
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
        "max_tokens": 2048,
        "rag": {"k": settings.rag_top_k, "num_candidates": settings.rag_num_candidates},
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
    print("RAG Hits:")
    hits = data.get("rag_context", {}).get("hits", [])
    if not hits:
        print("  (none)")
    else:
        for hit in hits:
            print(
                f"  - {hit.get('title', 'unknown')} | {hit.get('category', 'unknown')} | score={hit.get('score', 0.0):.4f}"
            )

    print("\nRaw response:\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
