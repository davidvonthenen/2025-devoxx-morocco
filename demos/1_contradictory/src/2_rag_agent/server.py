"""Flask server exposing an OpenAI-style chat completions endpoint with resilient llama.cpp init."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Sequence

from flask import Flask, jsonify, request
from opensearchpy.exceptions import RequestError

from ..common.config import Settings, load_settings
from ..common.embeddings import EmbeddingModel, to_list
from ..common.logging import get_logger
from ..common.opensearch_client import create_client, ensure_index, knn_search

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

    def chat(self, messages: Sequence[Dict[str, str]], **gen_kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


class LlamaCppBackend(LLMBackend):
    """llama-cpp-python implementation with automatic Metal->CPU fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llama = None
        self._init_mode = "uninitialized"

    def warm_up(self) -> None:
        """Eagerly load the underlying llama.cpp model."""
        self._ensure_loaded()

    def _build_kwargs(self, *, n_ctx: int | None = None, n_gpu_layers: int | None = None, low_vram: bool | None = None):
        # Defer import so that unit tests don't require llama_cpp
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
        LOGGER.info("Loading llama.cpp model (%s) with kwargs: %s", mode, {k: v for k, v in kwargs.items() if k != "model_path"})
        self._llama = Llama(**kwargs)
        self._init_mode = mode

    def _ensure_loaded(self) -> None:
        """Try GPU init first; fall back to CPU if Metal runs out of memory."""
        if self._llama is not None:
            return
        try:
            self._load_model(mode="gpu")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(marker in msg for marker in _GPU_OOM_SIGNS):
                LOGGER.warning("Metal init failed due to memory pressure (%s). Falling back to CPU...", msg)
                self._load_model(mode="cpu")
            else:
                raise

    def chat(self, messages: Sequence[Dict[str, str]], **gen_kwargs: Any) -> Dict[str, Any]:
        self._ensure_loaded()
        try:
            return self._llama.create_chat_completion(messages=list(messages), **gen_kwargs)  # type: ignore
        except RuntimeError as e:
            # If decode fails under GPU due to KV cache / Metal OOM, drop to CPU with smaller context and retry once.
            msg = str(e)
            if self._init_mode == "gpu" and any(marker in msg for marker in _GPU_OOM_SIGNS):
                LOGGER.warning("llama.cpp decode failed under GPU (%s). Reinitializing on CPU with smaller context and retrying...", msg)
                self._llama = None
                self._load_model(mode="cpu")
                if "max_tokens" not in gen_kwargs or int(gen_kwargs["max_tokens"]) > 256:
                    gen_kwargs["max_tokens"] = 256
                return self._llama.create_chat_completion(messages=list(messages), **gen_kwargs)  # type: ignore
            raise


SETTINGS = load_settings()
EMBEDDER = EmbeddingModel(SETTINGS)
OPENSEARCH_CLIENT = create_client(SETTINGS)
ensure_index(SETTINGS, EMBEDDER.dimension)
LLM = LlamaCppBackend(SETTINGS)
try:
    LLM.warm_up()
    LOGGER.info("llama.cpp model preloaded successfully")
except Exception:  # noqa: BLE001
    LOGGER.exception("Failed to preload llama.cpp model during startup")
    raise


def _extract_user_question(messages: Sequence[Dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "").strip()
            if content:
                return content
    raise ValueError("No user message found in request")


def _trim_snippet(text: str, max_length: int = 900) -> str:
    """Trim long snippets; keep memory footprint reasonable on M1."""
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
        "If the answer is not grounded in the snippets, respond with 'I don't know.' "
        "Provide concise answers."
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

    # Embed query and fetch RAG context
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
    messages = _compose_messages(question, context_block)

    # Keep generation limits conservative on laptops
    default_max_tokens = min(1024, int(payload.get("max_tokens", 1024)))
    llm_response = LLM.chat(
        messages,
        temperature=float(payload.get("temperature", 0.2)),
        top_p=float(payload.get("top_p", 0.95)),
        max_tokens=default_max_tokens,
    )

    usage = llm_response.get("usage", {})
    choice = llm_response.get("choices", [{}])[0]
    assistant_message = choice.get("message", {})

    response_body = {
        "id": str(uuid.uuid4()),
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
        "llama_runtime": {
            "init_mode": getattr(LLM, "_init_mode", "unknown"),
            "ctx": getattr(SETTINGS, "llama_ctx", None),
            "n_gpu_layers": getattr(SETTINGS, "llama_n_gpu_layers", None),
            "n_batch": getattr(SETTINGS, "llama_n_batch", None),
        },
    }
    return jsonify(response_body)


@APP.get("/__healthz")
def healthz():
    return jsonify({"status": "ok"})


def main() -> None:
    LOGGER.info("Starting server on %s:%s", SETTINGS.server_host, SETTINGS.server_port)
    APP.run(host=SETTINGS.server_host, port=SETTINGS.server_port)


if __name__ == "__main__":
    main()
