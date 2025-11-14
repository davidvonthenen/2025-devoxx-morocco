"""Flask server exposing an OpenAI-style chat completions endpoint with MCP tools."""
from __future__ import annotations

import atexit
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple

from flask import Flask, Response, jsonify, request
from opensearchpy.exceptions import RequestError

from ..common.config import Settings, load_settings
from ..common.embeddings import EmbeddingModel, to_list
from ..common.logging import get_logger
from ..common.opensearch_client import create_client, ensure_index, knn_search
from .mcp_client import MCPClient, MCPClientError

try:  # pragma: no cover - optional dependency at runtime
    import spacy
    from spacy.language import Language
    from spacy.matcher import Matcher
    from spacy.tokens import Span
except Exception:  # noqa: BLE001 - optional dependency fallback
    spacy = None
    Language = Any  # type: ignore[assignment]
    Matcher = Any  # type: ignore[assignment]
    Span = Any  # type: ignore[assignment]
from .tool_bus import ToolBus

LOGGER = get_logger(__name__)
APP = Flask(__name__)

_GPU_OOM_SIGNS = (
    "Insufficient Memory",
    "kIOGPUCommandBufferCallbackErrorOutOfMemory",
    "ggml_metal_graph_compute",
    "llama_decode returned -3",
)


class LLMBackend:
    """Abstract interface for language model backends."""

    def chat(self, messages: Sequence[Dict[str, str]], **gen_kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class LlamaCppBackend(LLMBackend):
    """llama-cpp-python implementation with automatic Metal->CPU fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llama = None
        self._init_mode = "uninitialized"

    def _build_kwargs(
        self,
        *,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        low_vram: bool | None = None,
    ):
        from llama_cpp import Llama  # noqa: F401

        kw = dict(
            model_path=self.settings.llama_model_path,
            n_ctx=int(n_ctx if n_ctx is not None else self.settings.llama_ctx),
            n_threads=int(self.settings.llama_n_threads),
            n_gpu_layers=int(n_gpu_layers if n_gpu_layers is not None else self.settings.llama_n_gpu_layers),
            n_batch=int(getattr(self.settings, "llama_n_batch", 256)),
            low_vram=bool(self.settings.llama_low_vram if low_vram is None else low_vram),
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        if hasattr(self.settings, "llama_n_ubatch") and self.settings.llama_n_ubatch:
            kw["n_ubatch"] = int(self.settings.llama_n_ubatch)
        return kw

    def _load_model(self, *, mode: str) -> None:
        from llama_cpp import Llama

        if mode == "gpu":
            kwargs = self._build_kwargs()
        elif mode == "cpu":
            kwargs = self._build_kwargs(n_gpu_layers=0, n_ctx=min(self.settings.llama_ctx, 4096))
        else:
            raise ValueError(f"Unknown init mode: {mode}")
        LOGGER.info(
            "Loading llama.cpp model (%s) with kwargs: %s",
            mode,
            {k: v for k, v in kwargs.items() if k != "model_path"},
        )
        self._llama = Llama(**kwargs)
        self._init_mode = mode

    def _ensure_loaded(self) -> None:
        if self._llama is not None:
            return
        try:
            self._load_model(mode="gpu")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(marker in msg for marker in _GPU_OOM_SIGNS):
                LOGGER.warning(
                    "Metal init failed due to memory pressure (%s). Falling back to CPU...",
                    msg,
                )
                self._load_model(mode="cpu")
            else:
                raise

    def chat(self, messages: Sequence[Dict[str, str]], **gen_kwargs: Any) -> Dict[str, Any]:
        self._ensure_loaded()
        kwargs = dict(gen_kwargs)
        try:
            return self._llama.create_chat_completion(messages=list(messages), **kwargs)  # type: ignore
        except TypeError as exc:
            unsupported = {"tools", "tool_choice", "functions", "function_call"}
            fallback_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported}
            if len(fallback_kwargs) != len(kwargs):
                LOGGER.warning(
                    "llama.cpp backend does not recognise tool parameters directly; retrying without them."
                )
                return self._llama.create_chat_completion(messages=list(messages), **fallback_kwargs)  # type: ignore
            raise
        except RuntimeError as e:
            msg = str(e)
            if self._init_mode == "gpu" and any(marker in msg for marker in _GPU_OOM_SIGNS):
                LOGGER.warning(
                    "llama.cpp decode failed under GPU (%s). Reinitializing on CPU with smaller context and retrying...",
                    msg,
                )
                self._llama = None
                self._load_model(mode="cpu")
                if "max_tokens" not in kwargs or int(kwargs["max_tokens"]) > 256:
                    kwargs["max_tokens"] = 256
                return self._llama.create_chat_completion(messages=list(messages), **kwargs)  # type: ignore
            raise

    def warm_up(self) -> None:
        """Eagerly load the underlying llama.cpp model."""

        if self._llama is not None:
            return
        start = time.time()
        LOGGER.info("Preloading llama.cpp model during server startup...")
        self._ensure_loaded()
        elapsed = time.time() - start
        LOGGER.info("llama.cpp model ready (%.2fs)", elapsed)


SETTINGS = load_settings()
EMBEDDER = EmbeddingModel(SETTINGS)
OPENSEARCH_CLIENT = create_client(SETTINGS)
ensure_index(SETTINGS, EMBEDDER.dimension)
LLM = LlamaCppBackend(SETTINGS)

try:
    LLM.warm_up()
except Exception:  # noqa: BLE001
    LOGGER.exception("Failed to preload llama.cpp model during startup")
    raise

MCP_CLIENT: MCPClient | None = None
TOOL_BUS: ToolBus | None = None
SPACY_NLP: Language | None = None
_SPACY_MATCHER: Matcher | None = None
_SPACY_ALLOWED_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "NORP",
    "EVENT",
    "PRODUCT",
    "WORK_OF_ART",
    "FAC",
    "LOC",
    "LAW",
    "LANGUAGE",
    "PROPER_NOUN",
}

if SETTINGS.mcp_enabled:
    LOGGER.info("MCP integration enabled; discovering tools...")
    try:
        MCP_CLIENT = MCPClient(
            connect_timeout=SETTINGS.mcp_connect_timeout_sec,
            invocation_timeout=SETTINGS.mcp_invocation_timeout_sec,
        )
        if SETTINGS.mcp_targets:
            MCP_CLIENT.discover(SETTINGS.mcp_targets)
        else:
            LOGGER.warning("MCP_ENABLED=1 but MCP_TARGETS is empty")
        if MCP_CLIENT.has_tools():
            TOOL_BUS = ToolBus(MCP_CLIENT, max_depth=SETTINGS.mcp_max_tool_call_depth)
            LOGGER.info("Registered MCP tools: %s", ", ".join(MCP_CLIENT.tool_names))
        else:
            LOGGER.warning("MCP client initialised but no tools discovered")
    except MCPClientError:
        LOGGER.exception("Failed to initialise MCP client; continuing without tools")
        MCP_CLIENT = None
        TOOL_BUS = None
    except Exception:  # noqa: BLE001
        LOGGER.exception("Unexpected error initialising MCP client; continuing without tools")
        MCP_CLIENT = None
        TOOL_BUS = None
else:
    LOGGER.info("MCP integration disabled")

if MCP_CLIENT:
    atexit.register(MCP_CLIENT.close)


def _ensure_spacy_model() -> Language | None:
    """Lazily load a spaCy pipeline for named entity extraction."""

    global SPACY_NLP, _SPACY_MATCHER
    if spacy is None:
        LOGGER.warning("spaCy is not installed; skipping MCP article prefetch.")
        return None
    if SPACY_NLP is not None:
        return SPACY_NLP

    try:
        SPACY_NLP = spacy.load("en_core_web_sm")
        LOGGER.info("Loaded spaCy model 'en_core_web_sm' for entity extraction.")
        return SPACY_NLP
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "spaCy model 'en_core_web_sm' unavailable (%s); using heuristic matcher.",
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
        _SPACY_MATCHER = matcher
        return SPACY_NLP


def _extract_named_entities(text: str) -> List[str]:
    nlp = _ensure_spacy_model()
    if nlp is None:
        return []
    doc = nlp(text)
    seen = set()
    entities: List[str] = []
    for ent in doc.ents:
        label = getattr(ent, "label_", "") or getattr(ent, "label", "")
        label = str(label) if label else ""
        if label and label not in _SPACY_ALLOWED_LABELS:
            continue
        value = ent.text.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(value)
    if entities:
        LOGGER.info("spaCy extracted entities for MCP prefetch: %s", entities)
    else:
        LOGGER.info("spaCy did not identify entities for MCP prefetch.")
    return entities


def _collect_mcp_articles(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    if MCP_CLIENT is None or not MCP_CLIENT.has_tools():
        return "", []

    entities = _extract_named_entities(question)
    if not entities:
        return "", []

    tool_name = next((name for name in MCP_CLIENT.tool_names if name.endswith("fetch_mock_news")), None)
    if not tool_name:
        LOGGER.info("No MCP tool capable of fetching mock news discovered; skipping prefetch.")
        return "", []

    try:
        LOGGER.info("Prefetching MCP articles for prompt via tool '%s'", tool_name)
        result = MCP_CLIENT.invoke(
            tool_name,
            {
                "limit": 5,
                "prompt": question,
            },
        )
    except MCPClientError as exc:
        LOGGER.exception("Failed MCP prefetch for question")
        return f"[MCP] Error fetching articles: {exc}", []

    aggregated_text: List[str] = []
    article_records: List[Dict[str, Any]] = []
    primary_articles: List[Dict[str, Any]] = []
    entity_payloads: List[Mapping[str, Any]] = []

    if isinstance(result, Mapping):
        raw_articles = result.get("articles")
        if isinstance(raw_articles, list):
            for article in raw_articles:
                if isinstance(article, Mapping):
                    primary_articles.append(dict(article))

        raw_entity_articles = result.get("entity_articles")
        if isinstance(raw_entity_articles, list):
            for payload in raw_entity_articles:
                if isinstance(payload, Mapping):
                    entity_payloads.append(dict(payload))

    if entity_payloads:
        for payload in entity_payloads:
            entity = str(payload.get("entity", "")).strip() or "(unknown entity)"
            articles = payload.get("articles") if isinstance(payload.get("articles"), list) else []
            if not articles:
                LOGGER.info("MCP returned no articles for entity '%s'", entity)
                aggregated_text.append(f"[MCP {entity}] No new articles returned.")
                continue
            article_records.append({"entity": entity, "articles": articles})
            aggregated_text.append(f"[MCP {entity}] Latest articles:")
            for article in articles:
                title = article.get("title", "(untitled)")
                summary = article.get("summary", "")
                published = article.get("published_at", "")
                category = article.get("category", "")
                aggregated_text.append(
                    f"- {title} ({category}) [{published}] :: {summary}"
                )

    if not aggregated_text and primary_articles:
        aggregated_text.append("[MCP] Latest articles:")
        article_records.append({"entity": "*", "articles": primary_articles})
        for article in primary_articles:
            title = article.get("title", "(untitled)")
            summary = article.get("summary", "")
            published = article.get("published_at", "")
            category = article.get("category", "")
            aggregated_text.append(
                f"- {title} ({category}) [{published}] :: {summary}"
            )

    if not aggregated_text:
        aggregated_text.append("[MCP] No new articles returned.")

    return "\n".join(aggregated_text).strip(), article_records


def _augment_context_with_mcp_articles(context_block: str, question: str) -> Tuple[str, List[Dict[str, Any]]]:
    supplemental_text, records = _collect_mcp_articles(question)
    if not supplemental_text:
        return context_block, []
    if context_block:
        combined = context_block + "\n\n" + supplemental_text
    else:
        combined = supplemental_text
    return combined, records


def _extract_user_question(messages: Sequence[Dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "").strip()
            if content:
                return content
    raise ValueError("No user message found in request")


def _trim_snippet(text: str, max_length: int = 900) -> str:
    if len(text) <= max_length:
        return text
    trimmed = text[:max_length]
    last_space = trimmed.rfind(" ")
    if last_space == -1:
        return trimmed + "..."
    return trimmed[:last_space] + "..."


def _build_context_block(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for idx, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        snippet = _trim_snippet(source.get("text", ""))
        parts.append(
            "[DOC {idx} | source: {category}/{title} | path: {path}]\n{snippet}".format(
                idx=idx,
                category=source.get("category", "unknown"),
                title=source.get("title", "unknown"),
                path=source.get("path", ""),
                snippet=snippet,
            )
        )
    return "\n\n".join(parts)


def _rag_hits_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        hits.append(
            {
                "path": source.get("path", ""),
                "title": source.get("title", ""),
                "category": source.get("category", ""),
                "score": float(hit.get("_score", 0.0)),
            }
        )
    return hits


def _compose_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a fact-focused assistant. "
        "Only use the context provided below to create a response. "
        "If the answer is not grounded in the context, respond with 'I don't know.' "
        "Cite sources inline like [source: ] when drawing from a snippet."
    )
    user_prompt = (
        f"Question:\n{question}\n\nContext:\n{context_block if context_block else 'No context available.'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _normalize_rag_params(payload: Dict[str, Any], settings: Settings) -> tuple[int, int]:
    rag_config = payload.get("rag", {}) if isinstance(payload.get("rag"), dict) else {}
    k = int(rag_config.get("k", settings.rag_top_k))
    num_candidates = int(rag_config.get("num_candidates", settings.rag_num_candidates))
    return k, num_candidates


def _make_stream_response(choice: Dict[str, Any], *, response_id: str, model: str) -> Response:
    created = int(time.time())
    assistant_message = choice.get("message", {})
    content = assistant_message.get("content", "") or ""

    def generate():
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": assistant_message.get("role", "assistant"),
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


@APP.post("/v1/chat/completions")
def chat_completions():
    payload = request.get_json(force=True, silent=False)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        question = _extract_user_question(payload.get("messages", []))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    k, num_candidates = _normalize_rag_params(payload, SETTINGS)

    embedding = EMBEDDER.encode([question])[0]
    try:
        search_response = knn_search(
            OPENSEARCH_CLIENT,
            SETTINGS.opensearch_index,
            to_list(embedding),
            k=k,
            num_candidates=num_candidates,
        )
    except RequestError as exc:
        detail = getattr(exc, "info", None) or str(exc)
        LOGGER.exception("OpenSearch query failed: %s", detail)
        return jsonify({"error": "OpenSearch query failed", "details": detail}), 400

    hits = search_response.get("hits", {}).get("hits", [])
    context_block = _build_context_block(hits)
    mcp_prefetch_records: List[Dict[str, Any]] = []
    if MCP_CLIENT and MCP_CLIENT.has_tools():
        context_block, mcp_prefetch_records = _augment_context_with_mcp_articles(context_block, question)
    messages = _compose_messages(question, context_block)

    if TOOL_BUS and TOOL_BUS.has_tools():
        messages = TOOL_BUS.augment_messages(messages)

    default_max_tokens = min(1024, int(payload.get("max_tokens", 1024)))
    llm_kwargs = dict(
        temperature=float(payload.get("temperature", 0.2)),
        top_p=float(payload.get("top_p", 0.95)),
        max_tokens=default_max_tokens,
    )

    stream_requested = bool(payload.get("stream"))

    if TOOL_BUS and TOOL_BUS.has_tools():
        llm_response = TOOL_BUS.run_chat_loop(
            LLM,
            messages,
            llm_kwargs=llm_kwargs,
            original_prompt=question,
        )
    else:
        llm_response = LLM.chat(messages, **llm_kwargs)

    usage = llm_response.get("usage", {})
    choice = llm_response.get("choices", [{}])[0]
    assistant_message = choice.get("message", {})

    response_id = str(uuid.uuid4())
    response_body = {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "local-llama"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": assistant_message.get("role", "assistant"),
                    "content": assistant_message.get("content", ""),
                },
                "finish_reason": choice.get("finish_reason", "stop"),
                "tool_calls": assistant_message.get("tool_calls"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "rag_context": {
            "index": SETTINGS.opensearch_index,
            "k": k,
            "num_candidates": num_candidates,
            "hits": _rag_hits_from_response(search_response),
        },
        "mcp_prefetch": mcp_prefetch_records,
        "llama_runtime": {
            "init_mode": getattr(LLM, "_init_mode", "unknown"),
            "ctx": getattr(SETTINGS, "llama_ctx", None),
            "n_gpu_layers": getattr(SETTINGS, "llama_n_gpu_layers", None),
            "n_batch": getattr(SETTINGS, "llama_n_batch", None),
        },
    }

    if stream_requested:
        return _make_stream_response(choice, response_id=response_id, model=response_body["model"])
    return jsonify(response_body)


@APP.get("/__healthz")
def healthz():
    return jsonify({"status": "ok"})


def main() -> None:
    LOGGER.info("Starting server on %s:%s", SETTINGS.server_host, SETTINGS.server_port)
    APP.run(host=SETTINGS.server_host, port=SETTINGS.server_port)


if __name__ == "__main__":
    main()

