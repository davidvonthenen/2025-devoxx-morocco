"""Embedding utilities using sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Union, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformers model."""
    return SentenceTransformer(model_name)


@dataclass
class EmbeddingModel:
    """Wrapper around a sentence-transformers embedding model with lazy init."""
    settings: Settings
    _cached_model: Optional[SentenceTransformer] = None

    @property
    def model_name(self) -> str:
        """
        Prefer `embedding_model_name` if present for backward-compat,
        otherwise use `embedding_model` (the current field name).
        """
        name = getattr(self.settings, "embedding_model_name", None)
        if not name:
            name = getattr(self.settings, "embedding_model", None)
        if not name:
            # Sensible default for this project
            name = "thenlper/gte-small"
        return name

    @property
    def model(self) -> SentenceTransformer:
        if self._cached_model is None:
            self._cached_model = _load_model(self.model_name)
        return self._cached_model

    @property
    def dimension(self) -> int:
        """
        Returns the embedding dimension. If `Settings` declares an explicit
        `embedding_dimension` override, use it; otherwise ask the model.
        """
        dim = getattr(self.settings, "embedding_dimension", None)
        if dim is not None:
            return int(dim)

        # Most SentenceTransformer models expose this method.
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            # Fallback: run a tiny encode to infer dimensionality.
            vec = self.encode(["_probe_"])[0]
            return int(vec.shape[-1])

    def encode(self, texts: Iterable[str]) -> List[np.ndarray]:
        """
        Encode a list/iterable of texts to a list of 1-D numpy arrays
        (L2-normalized) for direct use with cosine-similarity kNN.
        """
        items: List[str] = list(texts)
        if len(items) == 0:
            return []

        arr = self.model.encode(
            items,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Ensure we always return List[np.ndarray] of shape (D,)
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return [arr[i] for i in range(arr.shape[0])]
        if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(items) == 1:
            return [arr]
        return [np.asarray(v, dtype=float) for v in arr]


def to_list(vec: Union[np.ndarray, Sequence[float], List[float]]) -> List[float]:
    """Convert a vector-like object to a Python list of floats."""
    if isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()
    return [float(x) for x in vec]


__all__ = ["EmbeddingModel", "to_list"]
