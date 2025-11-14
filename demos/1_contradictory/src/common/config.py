"""Configuration loading utilities for the RAG service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime configuration parameters for the application."""

    # OpenSearch
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_index: str = "bbc"

    # Embeddings
    embedding_model: str = "thenlper/gte-small"

    # Llama.cpp
    llama_model_path: str = "neural-chat-7b-v3-3.Q4_K_M.gguf"
    llama_ctx: int = 32768                    # keep conservative by default on laptops
    llama_n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    llama_n_gpu_layers: int = 20             # modest offload; fallback logic drops to CPU if needed
    llama_n_batch: int = 256                 # prompt processing batch
    llama_n_ubatch: Optional[int] = 256      # physical micro-batch; None to let llama.cpp choose
    llama_low_vram: bool = True              # reduce Metal VRAM usage

    # RAG
    rag_top_k: int = 3
    rag_num_candidates: int = 50

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000


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


def load_settings(env_file: str | None = None) -> Settings:
    """Load settings from environment (.env) with sane defaults."""
    # Load a .env if present (project root or provided explicit path)
    if env_file:
        load_dotenv(env_file)
    else:
        # Probe common locations
        for candidate in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
            if candidate.exists():
                load_dotenv(str(candidate))
                break

    return Settings(
        opensearch_host=os.getenv("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", Settings.opensearch_index),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
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
        rag_top_k=_get_int("RAG_TOP_K", Settings.rag_top_k),
        rag_num_candidates=_get_int("RAG_NUM_CANDIDATES", Settings.rag_num_candidates),
        server_host=os.getenv("SERVER_HOST", Settings.server_host),
        server_port=_get_int("SERVER_PORT", Settings.server_port),
    )


__all__ = ["Settings", "load_settings"]
