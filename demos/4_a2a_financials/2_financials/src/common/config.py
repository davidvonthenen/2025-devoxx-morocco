"""Configuration loading utilities for the MCP-enabled agent service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime configuration parameters for the application."""

    # Llama.cpp runtime options
    llama_model_path: str = "neural-chat-7b-v3-3.Q4_K_M.gguf"
    llama_ctx: int = 32768
    llama_n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    llama_n_gpu_layers: int = 20
    llama_n_batch: int = 256
    llama_n_ubatch: Optional[int] = 256
    llama_low_vram: bool = True

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8002

    # MCP integration
    mcp_enabled: bool = True
    mcp_targets: str = ""
    mcp_connect_timeout_sec: float = 5.0
    mcp_invocation_timeout_sec: float = 15.0
    mcp_max_tool_call_depth: int = 3


def _get_int(name: str, default_val: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default_val
    try:
        return int(v)
    except ValueError:
        return default_val


def _get_bool(name: str, default_val: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default_val
    return v.lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default_val: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default_val
    try:
        return float(v)
    except ValueError:
        return default_val


def load_settings(env_file: str | None = None) -> Settings:
    """Load settings from environment (.env) with sane defaults."""
    if env_file:
        load_dotenv(env_file)
    else:
        for candidate in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
            if candidate.exists():
                load_dotenv(str(candidate))
                break

    return Settings(
        llama_model_path=os.getenv(
            "LLAMA_MODEL_PATH",
            str(Path.home() / "models" / Settings.llama_model_path),
        ),
        llama_ctx=_get_int("LLAMA_CTX", Settings.llama_ctx),
        llama_n_threads=_get_int("LLAMA_N_THREADS", Settings.llama_n_threads),
        llama_n_gpu_layers=_get_int("LLAMA_N_GPU_LAYERS", Settings.llama_n_gpu_layers),
        llama_n_batch=_get_int("LLAMA_N_BATCH", Settings.llama_n_batch),
        llama_n_ubatch=_get_int("LLAMA_N_UBATCH", Settings.llama_n_ubatch or 0) or None,
        llama_low_vram=_get_bool("LLAMA_LOW_VRAM", Settings.llama_low_vram),
        server_host=os.getenv("SERVER_HOST", Settings.server_host),
        server_port=_get_int("SERVER_PORT", Settings.server_port),
        mcp_enabled=_get_bool("MCP_ENABLED", Settings.mcp_enabled),
        mcp_targets=os.getenv("MCP_TARGETS", Settings.mcp_targets),
        mcp_connect_timeout_sec=_get_float("MCP_CONNECT_TIMEOUT_SEC", Settings.mcp_connect_timeout_sec),
        mcp_invocation_timeout_sec=_get_float(
            "MCP_INVOCATION_TIMEOUT_SEC", Settings.mcp_invocation_timeout_sec
        ),
        mcp_max_tool_call_depth=_get_int("MCP_MAX_TOOL_CALL_DEPTH", Settings.mcp_max_tool_call_depth),
    )


__all__ = ["Settings", "load_settings"]
