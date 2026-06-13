"""BOE ingestion: fetch daily sumarios and documents, parse, and chunk them."""

from boe_rag.ingest.chunker import chunk_document
from boe_rag.ingest.client import BoeApiError, BoeClient
from boe_rag.ingest.parser import parse_document, parse_sumario

__all__ = [
    "BoeApiError",
    "BoeClient",
    "chunk_document",
    "parse_document",
    "parse_sumario",
]
