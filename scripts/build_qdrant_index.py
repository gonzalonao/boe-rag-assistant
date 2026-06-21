"""Upsert precomputed E5 passage embeddings into a Qdrant collection.

The dense leg of the retriever can be served either from the in-memory NumPy
index or from Qdrant (Arc 5). This script populates the Qdrant side once: it
reads the same ``.npz`` produced by ``scripts/precompute_embeddings.py`` and
upserts each vector into a Cosine collection, so :class:`QdrantSearcher` ranks
identically to the NumPy index over the same vectors.

Qdrant point ids must be unsigned integers or UUIDs, so the row index is used as
the point id and the stable BOE ``chunk_id`` rides in the payload (recovered at
search time via :data:`boe_rag.eval.qdrant_store.CHUNK_ID_FIELD`).

Requires the ``qdrant`` extra (``pip install -e .[qdrant]``) and a running
Qdrant (``docker run -p 6333:6333 qdrant/qdrant``).

Example:
    python scripts/build_qdrant_index.py \
        --embeddings data/corpus/boe-2015-present-embeddings.npz \
        --url http://localhost:6333 \
        --collection boe_chunks
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

from boe_rag.eval.qdrant_store import CHUNK_ID_FIELD
from boe_rag.eval.retriever import load_embeddings

logger = logging.getLogger(__name__)

#: Default upsert batch size — large enough to be fast, small enough to keep the
#: request payload well within Qdrant's limits for 384-dim vectors.
DEFAULT_BATCH_SIZE = 256


def build_index(
    embeddings: Path,
    *,
    url: str | None,
    path: str | None,
    collection: str,
    api_key: str | None,
    batch_size: int,
) -> int:
    """Create (or recreate) the collection and upsert every embedding.

    Connects to a Qdrant server (``url``) or a local embedded on-disk instance
    (``path``); exactly one must be given. The client is closed before returning
    so an embedded ``path`` instance releases its lock for the next process.

    Args:
        embeddings: Path to the precomputed ``.npz`` (ids + matrix).
        url: Base URL of a running Qdrant server.
        path: Directory for a local embedded Qdrant instance.
        collection: Target collection name; recreated if it already exists.
        api_key: Optional API key for a secured server.
        batch_size: Number of points per upsert request.

    Returns:
        The number of points in the collection after indexing.
    """
    chunk_ids, matrix = load_embeddings(embeddings)
    num_points, dim = matrix.shape
    logger.info("Loaded %d x %d embeddings from %s", num_points, dim, embeddings)

    client = (
        QdrantClient(path=path)
        if path is not None
        else QdrantClient(url=url, api_key=api_key)
    )
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    logger.info("Created collection '%s' (size=%d, COSINE)", collection, dim)

    for start in range(0, num_points, batch_size):
        stop = min(start + batch_size, num_points)
        points = [
            models.PointStruct(
                id=i,
                vector=matrix[i].tolist(),
                payload={CHUNK_ID_FIELD: chunk_ids[i]},
            )
            for i in range(start, stop)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        logger.info("Upserted %d / %d points", stop, num_points)

    count = client.count(collection_name=collection, exact=True).count
    logger.info("Collection '%s' now holds %d points", collection, count)
    client.close()  # release the embedded on-disk lock for the next process
    return count


def main(argv: list[str] | None = None) -> int:
    """Run the Qdrant index build.

    Returns:
        Process exit code (0 on success, 1 on a mismatched point count).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build a Qdrant index from a .npz.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/corpus/boe-2015-present-embeddings.npz"),
        help="Precomputed embeddings .npz (ids + matrix).",
    )
    location = parser.add_mutually_exclusive_group()
    location.add_argument(
        "--url", default=None, help="Qdrant server URL (default localhost:6333)."
    )
    location.add_argument(
        "--path",
        default=None,
        help="Directory for a local embedded Qdrant (no server/Docker).",
    )
    parser.add_argument(
        "--collection", default="boe_chunks", help="Target collection name."
    )
    parser.add_argument(
        "--api-key", default=None, help="API key for a secured Qdrant (optional)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Points per upsert request.",
    )
    args = parser.parse_args(argv)

    if not args.embeddings.is_file():
        print(f"ERROR: embeddings not found: {args.embeddings}", file=sys.stderr)
        return 1

    url = args.url
    if url is None and args.path is None:
        url = "http://localhost:6333"  # default to a local server when unspecified

    _, matrix = load_embeddings(args.embeddings)
    expected = matrix.shape[0]
    count = build_index(
        args.embeddings,
        url=url,
        path=args.path,
        collection=args.collection,
        api_key=args.api_key,
        batch_size=args.batch_size,
    )
    if count != expected:
        print(
            f"ERROR: indexed {count} points but expected {expected}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
