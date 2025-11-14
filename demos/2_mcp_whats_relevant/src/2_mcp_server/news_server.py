"""Mock MCP server providing deterministic BBC-style news articles."""
from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List

try:  # pragma: no cover - optional dependency loading
    import spacy
    from spacy.language import Language
    from spacy.matcher import Matcher
    from spacy.tokens import Span
except Exception:  # noqa: BLE001 - spaCy may be unavailable in some environments
    spacy = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment]
    Matcher = None  # type: ignore[assignment]
    Span = None  # type: ignore[assignment]


LOGGER = logging.getLogger("mock-mcp-server")

TOOLS = [
    {
        "name": "fetch_mock_news",
        "description": "Fetches deterministic mock BBC-style news articles.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Optional keyword to filter articles by topic, category, or summary.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of articles to return (1-50).",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                },
                "prompt": {
                    "type": "string",
                    "description": "Original user prompt used for named entity extraction.",
                },
            },
            "required": [],
        },
    }
]


def _log_prompt_details(prompt: str | None) -> None:
    if prompt:
        LOGGER.info("Original prompt received: %s", prompt)
        print(f"Original prompt received: {prompt}", file=sys.stderr, flush=True)


SPACY_NLP: Language | None = None
_SPACY_ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LOC"}


def _ensure_spacy_model() -> Language | None:
    global SPACY_NLP
    if spacy is None:
        LOGGER.warning("spaCy is not installed; server-side entity extraction disabled.")
        return None
    if SPACY_NLP is not None:
        return SPACY_NLP

    try:
        SPACY_NLP = spacy.load("en_core_web_sm")
        LOGGER.info("Loaded spaCy model 'en_core_web_sm' for server entity extraction.")
        return SPACY_NLP
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Failed to load spaCy model 'en_core_web_sm' (%s); falling back to heuristic matcher.",
            exc,
        )
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        matcher = Matcher(nlp.vocab)
        patterns = [
            [{"IS_TITLE": True, "IS_STOP": False}],
            [
                {"IS_TITLE": True, "IS_STOP": False},
                {"IS_TITLE": True, "IS_STOP": False},
            ],
            [
                {"IS_TITLE": True, "IS_STOP": False},
                {"IS_TITLE": True, "IS_STOP": False},
                {"IS_TITLE": True, "IS_STOP": False},
            ],
        ]
        matcher.add("PROPER_NOUN", patterns)

        @Language.component("entity_matcher")  # type: ignore[misc]
        def entity_matcher(doc):  # type: ignore[override]
            matches = matcher(doc)
            spans = list(doc.ents)
            for match_id, start, end in matches:
                spans.append(Span(doc, start, end, label=match_id))
            doc.ents = tuple(spans)
            return doc

        nlp.add_pipe("entity_matcher", after="sentencizer")
        SPACY_NLP = nlp
        return SPACY_NLP


def _extract_named_entities(text: str | None) -> List[str]:
    if not text:
        return []
    nlp = _ensure_spacy_model()
    if nlp is None:
        return []
    doc = nlp(text)
    seen = set()
    entities: List[str] = []
    for ent in doc.ents:
        label = getattr(ent, "label_", "") or getattr(ent, "label", "")
        if label and _SPACY_ALLOWED_LABELS and label not in _SPACY_ALLOWED_LABELS:
            continue
        value = ent.text.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(value)
    return entities

OPENAI_WINDSURF_CONTENT = (
    "Google Is Said to Pay $2.4 Billion for Windsurf Assets, Talent\n\n"
    "OpenAI, the generative AI company backed by Microsoft (NASDAQ:MSFT), saw its planned $3B acquisition of AI-assisted coding tool Windsurf go into the scrap heap on Friday, according to multiple reports.\n\n"
    "In addition, Windsurf CEO Varun Mohan, along with co-founder Douglas Chen, and other top talent are joining Google (NASDAQ:GOOG) DeepMind, which operates as an AI research laboratory, The Verge reported. The tech news outlet noted that the former Windsurf employees will focus efforts on Gemini.\n\n"
    "Google (NASDAQ:GOOGL) will have neither a stake nor control in Windsurf, but will have a nonexclusive license to some Windsurf technology, The Verge added.\n\n"
    "Late Friday night, The Wall Street Journal reported that Google will pay ~$2.4B to license Windsurf's technology.\n\n"
    "\"We're excited to welcome some top AI coding talent from Windsurf's team to Google DeepMind to advance our work in agentic coding,\" Google spokesperson Chris Pappas told TechCrunch.\n\n"
    "Windsurf Head of Business Jeff Wang is becoming its interim CEO, effective immediately, while VP, Global Sales, Graham Moreno, is becoming president.\n\n"
    "In a post on X, Wang portrayed the development as \"an agreement to kickstart the next phase of the company.\" He added that \"we find ourselves at a crossroads where the company needs to evolve on two, slightly divergent axes. On one axis, we have a responsibility to continue pushing the frontier of model capabilities. On the other hand, we have a responsibility to make the technology more approachable, secure, and reliable for enterprise workloads, the most critical and consequential workloads for society.\"\n\n"
    "\"Given the rapid pace of innovation, we see an advantage to double down our focus on the enterprise problems, which has long been our primary focus, and we will be continuing to devote resources to taking the wide range of product innovations in the broader market and making them work for enterprise workloads, the most impactful workloads to society,\" a Windsurf blog post reads.\n\n"
)

_ARTICLES: List[Dict[str, str]] = [
    {
        "id": "mock-001",
        "title": "UK inflation edges lower in August",
        "content": "Consumer prices eased slightly as energy costs retreated, according to the ONS.",
        "category": "business",
        "url": "https://example.com/articles/mock-001",
        "published_at": "2025-09-21T10:05:00Z",
        "entities": ["uk", "ons" ],
    },
    {
        "id": "mock-002",
        "title": "OpenAI deal for Windsurf falls apart; Google paying $2.4B for Windsurf tech - reports",
        "content": OPENAI_WINDSURF_CONTENT,
        "category": "tech",
        "url": "https://example.com/articles/mock-002",
        "published_at": "2025-09-20T07:40:00Z",
        "entities": ["openai_windsurf", "openai_windsurf_google", "windsurf_google", "openai", "google", "windsurf"],
    },
]

def _build_handshake(session_id: str, *, include_url: bool = False) -> Dict[str, object]:
    handshake: Dict[str, object] = {
        "type": "ready",
        "session_id": session_id,
        "alias": "news",
        "tools": TOOLS,
    }
    if include_url:
        handshake["invoke_url"] = "/mcp/invoke"
    return handshake


def _handle_invoke(tool: str, arguments: Dict[str, object]) -> Dict[str, object]:
    if tool != "fetch_mock_news":
        LOGGER.warning("Unknown tool requested: %s", tool)
        return {
            "ok": False,
            "error": {
                "type": "UnknownTool",
                "message": f"Tool '{tool}' is not available.",
            },
        }

    limit_arg = arguments.get("limit", 5) if isinstance(arguments, dict) else 5
    prompt = (
        str(arguments.get("prompt", "")).strip()
        if isinstance(arguments, dict)
        else ""
    )

    LOGGER.info("Invoking fetch_mock_news with limit=%s", limit_arg)
    _log_prompt_details(prompt)

    try:
        limit = max(1, min(int(limit_arg), 50))
    except Exception:  # noqa: BLE001 - fall back to default range if parsing fails
        limit = 5

    extracted_entities = _extract_named_entities(prompt)
    if not extracted_entities:
        LOGGER.info("No named entities extracted from the prompt; returning empty result set.")
        return {"ok": True, "content": {"articles": []}}

    LOGGER.info("Server extracted entities from prompt: %s", extracted_entities)
    print(f"Extracted entities: {extracted_entities}", file=sys.stderr, flush=True)

    def _normalize(value: str) -> str:
        return value.strip().lower().replace(" ", "_")

    normalized_query_entities = {
        _normalize(entity)
        for entity in extracted_entities
        if isinstance(entity, str) and entity.strip()
    }

    matched_articles: List[Dict[str, str]] = []
    for article in _ARTICLES:
        raw_entities = article.get("entities", [])
        if not isinstance(raw_entities, list):
            continue
        normalized_article_entities = {
            _normalize(str(item))
            for item in raw_entities
            if isinstance(item, str) and item.strip()
        }
        if normalized_query_entities.intersection(normalized_article_entities):
            matched_articles.append(article)

    LOGGER.info("Returning %s article(s) that match extracted entities.", len(matched_articles))
    return {"ok": True, "content": {"articles": matched_articles[:limit]}}


def run_stdio_server() -> None:
    session_id = str(uuid.uuid4())
    handshake = _build_handshake(session_id)
    print(json.dumps(handshake), flush=True)
    LOGGER.info("Mock MCP stdio server ready (session=%s)", session_id)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            LOGGER.warning("Received non-JSON payload: %s", raw_line)
            continue

        if message.get("type") == "shutdown":
            LOGGER.info("Shutdown requested; exiting.")
            break

        if message.get("type") != "invoke":
            LOGGER.warning("Unsupported message type: %s", message.get("type"))
            continue

        call_id = message.get("id", str(uuid.uuid4()))
        tool = message.get("tool", "")
        arguments = message.get("arguments", {}) if isinstance(message, dict) else {}
        LOGGER.info("STDIO invocation received (call_id=%s, tool=%s, arguments=%s)", call_id, tool, arguments)
        result = _handle_invoke(tool, arguments if isinstance(arguments, dict) else {})
        response = {
            "type": "result",
            "id": call_id,
            **result,
        }
        LOGGER.info("STDIO response (call_id=%s): %s", call_id, response)
        print(json.dumps(response), flush=True)


@dataclass
class _ClientInfo:
    session_id: str
    queue: "queue.Queue[Dict[str, object]]"


class _MCPHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.clients: Dict[str, _ClientInfo] = {}
        self._lock = threading.Lock()

    def register_client(self, session_id: str) -> queue.Queue:
        q: "queue.Queue[Dict[str, object]]" = queue.Queue()
        with self._lock:
            self.clients[session_id] = _ClientInfo(session_id, q)
        return q

    def push_event(self, session_id: str, payload: Dict[str, object]) -> bool:
        with self._lock:
            info = self.clients.get(session_id)
        if not info:
            return False
        info.queue.put(payload)
        return True

    def remove_client(self, session_id: str) -> None:
        with self._lock:
            self.clients.pop(session_id, None)


class MCPRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - inherited signature
        LOGGER.info("%s - %s", self.client_address[0], format % args)

    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - HTTP verb name
        if self.path != "/mcp":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        session_id = str(uuid.uuid4())
        q = self.server.register_client(session_id)  # type: ignore[attr-defined]

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        handshake = _build_handshake(session_id, include_url=True)
        payload = json.dumps(handshake)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()
        LOGGER.info("SSE client connected (session=%s)", session_id)

        try:
            while True:
                try:
                    message = q.get(timeout=1)
                except queue.Empty:
                    continue
                if message is None:
                    break
                self.wfile.write(("data: " + json.dumps(message) + "\n\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("SSE client disconnected (session=%s)", session_id)
        finally:
            self.server.remove_client(session_id)  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802 - HTTP verb name
        if self.path != "/mcp/invoke":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
            return
        
        print("-------------- DATA BEGIN")
        print(payload)
        print("-------------- DATA END")
        # {
        #     "id": "2e24d90a-97d5-4675-b7ac-9dfe8ebb4168",
        #     "session_id": "e178a76f-be13-4b6c-af43-ea86d0ecce67",
        #     "tool": "fetch_mock_news",
        #     "arguments": {
        #         "limit": 5,
        #         "prompt": "Tell me about the connection between Ernie Wise and Vodafone."
        #     }
        # }

        session_id = payload.get("session_id")
        if not isinstance(session_id, str):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing session_id"})
            return

        call_id = payload.get("id", str(uuid.uuid4()))
        tool = payload.get("tool", "")
        arguments = payload.get("arguments", {}) if isinstance(payload.get("arguments"), dict) else {}
        LOGGER.info(
            "SSE invocation received (session=%s, call_id=%s, tool=%s, arguments=%s)",
            session_id,
            call_id,
            tool,
            arguments,
        )
        result = _handle_invoke(tool, arguments)
        event = {
            "type": "result",
            "id": call_id,
            **result,
        }
        pushed = self.server.push_event(session_id, event)  # type: ignore[attr-defined]
        if not pushed:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
            return
        LOGGER.info("SSE response dispatched (session=%s, call_id=%s)", session_id, call_id)
        self._send_json(HTTPStatus.OK, {"status": "ok"})


def run_sse_server(port: int) -> None:
    server = _MCPHTTPServer(("0.0.0.0", port), MCPRequestHandler)
    LOGGER.info("Mock MCP SSE server listening on port %s", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        LOGGER.info("SSE server interrupted; shutting down")
    finally:
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock MCP server for news retrieval")
    parser.add_argument("--stdio", action="store_true", help="Run using stdio transport")
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Run using SSE/HTTP transport (default)",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for SSE mode")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.stdio and args.sse:
        parser.error("Cannot enable both stdio and SSE modes")

    if args.stdio:
        run_stdio_server()
    else:
        run_sse_server(args.port)


if __name__ == "__main__":
    main()

