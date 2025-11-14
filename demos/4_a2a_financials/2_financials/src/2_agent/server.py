"""Flask server exposing an OpenAI-style chat completions endpoint with MCP tools."""
from __future__ import annotations

import atexit
import json
import time
import uuid
from collections.abc import Sequence
from typing import Any, Dict, List, Mapping

from flask import Flask, Response, jsonify, request

from ..common.config import Settings, load_settings
from ..common.logging import get_logger
from .financial_queries import (
    IMPROVEMENT_TOOL_SUFFIX,
    RESULTS_TOOL_SUFFIX,
    detect_financial_tool_request,
    format_improvement_summary,
    format_results_summary,
)
from .mcp_client import MCPClient, MCPClientError, MCPInvocationError
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

    def chat(self, messages: Sequence[Dict[str, Any]], **gen_kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - interface
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
    ) -> Dict[str, Any]:
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

    def chat(self, messages: Sequence[Dict[str, Any]], **gen_kwargs: Any) -> Dict[str, Any]:
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
LLM = LlamaCppBackend(SETTINGS)

try:
    LLM.warm_up()
except Exception:  # noqa: BLE001
    LOGGER.exception("Failed to preload llama.cpp model during startup")
    raise

MCP_CLIENT: MCPClient | None = None
TOOL_BUS: ToolBus | None = None

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


def _resolve_tool_name(suffix: str) -> str | None:
    if not MCP_CLIENT:
        return None
    for name in MCP_CLIENT.tool_names:
        if name.endswith(f".{suffix}") or name == suffix:
            return name
    return None


def _build_direct_completion(content: str, *, model: str, stream: bool):
    response_id = str(uuid.uuid4())
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": "stop",
    }
    response_body = {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "mcp": {
            "enabled": bool(TOOL_BUS and TOOL_BUS.has_tools()),
            "tools": MCP_CLIENT.tool_names if MCP_CLIENT else [],
        },
        "llama_runtime": {
            "init_mode": getattr(LLM, "_init_mode", "unknown"),
            "ctx": getattr(SETTINGS, "llama_ctx", None),
            "n_gpu_layers": getattr(SETTINGS, "llama_n_gpu_layers", None),
            "n_batch": getattr(SETTINGS, "llama_n_batch", None),
        },
    }

    if stream:
        return _make_stream_response(choice, response_id=response_id, model=model)
    return jsonify(response_body)


def _maybe_handle_direct_financial_query(question: str, *, stream: bool, model: str):
    detection = detect_financial_tool_request(question)
    if not detection:
        return None
    tool_suffix, symbol = detection
    tool_name = _resolve_tool_name(tool_suffix)
    if not tool_name:
        LOGGER.info("Financial tool '%s' requested but not registered", tool_suffix)
        return None

    arguments = {"symbol": symbol}
    if tool_suffix == RESULTS_TOOL_SUFFIX:
        arguments["limit"] = 1
    elif tool_suffix == IMPROVEMENT_TOOL_SUFFIX:
        arguments["limit"] = 2
    try:
        payload = MCP_CLIENT.invoke(tool_name, arguments) if MCP_CLIENT else None
    except MCPInvocationError as exc:
        LOGGER.exception("MCP invocation failed for %s", tool_name)
        content = f"Unable to retrieve financial data for {symbol}: {exc}"
    else:
        if tool_suffix == RESULTS_TOOL_SUFFIX:
            content = format_results_summary(symbol, payload)
        else:
            content = format_improvement_summary(symbol, payload)
    return _build_direct_completion(content, model=model, stream=stream)


def _extract_user_question(messages: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                content = content.strip()
            if content:
                return str(content)
    raise ValueError("No user message found in request")


def _ensure_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("'messages' must be a non-empty list")

    normalised: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise ValueError(f"Message at index {idx} is not an object")
        normalised.append(dict(item))
    return normalised


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
        messages = _ensure_messages(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        question = _extract_user_question(messages)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    llm_messages: List[Dict[str, Any]] = list(messages)
    model_name = str(payload.get("model", "local-llama"))

    if TOOL_BUS and TOOL_BUS.has_tools():
        llm_messages = TOOL_BUS.augment_messages(llm_messages)

    default_max_tokens = min(1024, int(payload.get("max_tokens", 1024)))
    llm_kwargs = dict(
        temperature=float(payload.get("temperature", 0.2)),
        top_p=float(payload.get("top_p", 0.95)),
        max_tokens=default_max_tokens,
    )

    stream_requested = bool(payload.get("stream"))

    if MCP_CLIENT:
        direct_response = _maybe_handle_direct_financial_query(
            question,
            stream=stream_requested,
            model=model_name,
        )
        if direct_response is not None:
            return direct_response

    if TOOL_BUS and TOOL_BUS.has_tools():
        llm_response = TOOL_BUS.run_chat_loop(
            LLM,
            llm_messages,
            llm_kwargs=llm_kwargs,
            original_prompt=question,
        )
    else:
        llm_response = LLM.chat(llm_messages, **llm_kwargs)

    usage = llm_response.get("usage", {})
    choice = llm_response.get("choices", [{}])[0]
    assistant_message = choice.get("message", {})

    response_id = str(uuid.uuid4())
    response_body = {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
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
        "mcp": {
            "enabled": bool(TOOL_BUS and TOOL_BUS.has_tools()),
            "tools": MCP_CLIENT.tool_names if MCP_CLIENT else [],
        },
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
