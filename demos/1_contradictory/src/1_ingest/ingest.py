"""BBC dataset ingestion script."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator, Sequence

from tqdm import tqdm

from ..common.config import Settings, load_settings
from ..common.embeddings import EmbeddingModel, to_list
from ..common.logging import get_logger
from ..common.opensearch_client import create_client, ensure_index

LOGGER = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest dataset into OpenSearch")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--index-name", type=str, required=True, help="Target OpenSearch index")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    return parser.parse_args(argv)


def _iter_documents(data_dir: Path) -> Iterator[dict[str, str]]:
    categories = [p for p in data_dir.iterdir() if p.is_dir()]
    if not categories:
        raise FileNotFoundError(f"No category folders found in {data_dir}")

    for category_dir in categories:
        for file_path in category_dir.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8")
            rel_path = file_path.relative_to(data_dir).as_posix()
            yield {
                "path": rel_path,
                "title": file_path.stem,
                "category": category_dir.name,
                "text": text,
            }


def _doc_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def ingest(data_dir: Path, index_name: str, batch_size: int, settings: Settings) -> None:
    client = create_client(settings)
    embedder = EmbeddingModel(settings)
    ensure_index(settings, embedder.dimension)

    _ = batch_size  # Retained for CLI compatibility; ingestion is single-document.
    total_indexed = 0
    progress = tqdm(desc="Indexing", unit="docs")
    try:
        for doc in _iter_documents(data_dir):
            embedding = embedder.encode([doc["text"]])
            embedding_list = to_list(embedding[0])
            doc_id = _doc_id(doc["path"])
            response = client.index(
                index=index_name,
                id=doc_id,
                body={**doc, "embedding": embedding_list},
            )
            status = response.get("result", "unknown")
            LOGGER.info("Indexed %s with status '%s'", doc["path"], status)
            progress.update(1)
            total_indexed += 1
    finally:
        progress.close()

    if total_indexed == 0:
        LOGGER.warning("No documents were ingested.")
    else:
        client.indices.refresh(index=index_name)
        LOGGER.info("Ingested %d documents into index '%s'", total_indexed, index_name)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    settings = load_settings()
    settings = Settings(
        **{
            **settings.__dict__,
            "opensearch_index": args.index_name,
        }
    )
    ingest(data_dir=data_dir, index_name=args.index_name, batch_size=args.batch_size, settings=settings)


if __name__ == "__main__":
    main()
