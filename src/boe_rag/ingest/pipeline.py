"""End-to-end ingestion orchestration.

Ties the client, parser, and chunker together: given a range of dates it walks
each daily sumario, fetches and parses every in-scope document, and yields the
resulting chunks as a single stream. Generators keep memory flat regardless of
how many days are ingested.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from datetime import date, timedelta

from boe_rag.config import IngestionConfig
from boe_rag.ingest.chunker import chunk_document
from boe_rag.ingest.client import BoeApiError, BoeClient
from boe_rag.ingest.parser import parse_document, parse_sumario
from boe_rag.models import Chunk, SumarioItem

logger = logging.getLogger(__name__)


def date_range(start: date, end: date) -> Iterator[date]:
    """Yield each date from ``start`` to ``end`` inclusive.

    Args:
        start: First date to yield.
        end: Last date to yield.

    Returns:
        An iterator over consecutive dates.

    Raises:
        ValueError: If ``end`` precedes ``start``.
    """
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def ingest_dates(
    dates: Iterable[date],
    config: IngestionConfig | None = None,
    client: BoeClient | None = None,
) -> Iterator[Chunk]:
    """Ingest every in-scope document published on the given dates.

    Args:
        dates: Dates to ingest.
        config: Ingestion configuration; defaults are used when omitted.
        client: An open client to reuse; one is created and closed if omitted.

    Yields:
        Chunks for each document, in publication order.
    """
    cfg = config or IngestionConfig()
    owned_client = client is None
    active = client or BoeClient(cfg)
    try:
        for day in dates:
            yield from _ingest_single_date(day, cfg, active)
    finally:
        if owned_client:
            active.close()


def _ingest_single_date(
    day: date,
    config: IngestionConfig,
    client: BoeClient,
) -> Iterator[Chunk]:
    """Ingest all in-scope documents for one date, skipping days with no issue."""
    date_str = day.strftime("%Y%m%d")
    try:
        payload = client.fetch_sumario(date_str)
    except BoeApiError as err:
        logger.info("No BOE issue for %s (%s)", date_str, err)
        return
    items = parse_sumario(payload, config.sections)
    logger.info("%s: %d in-scope documents", date_str, len(items))
    for item in items:
        yield from _ingest_item(item, config, client)


def _ingest_item(
    item: SumarioItem,
    config: IngestionConfig,
    client: BoeClient,
) -> Iterator[Chunk]:
    """Fetch, parse, and chunk a single document, tolerating per-doc failures."""
    try:
        xml_text = client.fetch_document_xml(item.identifier)
        document = parse_document(xml_text)
    except (BoeApiError, ValueError) as err:
        logger.warning("Skipping %s: %s", item.identifier, err)
        return
    yield from chunk_document(document, config)
