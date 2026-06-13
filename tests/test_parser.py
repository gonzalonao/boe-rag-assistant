"""Tests for the sumario and document parsers."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from boe_rag.ingest.parser import parse_document, parse_sumario


def test_parse_sumario_filters_to_requested_sections(
    sumario_payload: dict[str, Any],
) -> None:
    """Only items in the requested sections are returned."""
    items = parse_sumario(sumario_payload, frozenset({"1"}))
    identifiers = {item.identifier for item in items}
    assert identifiers == {"BOE-A-2024-714", "BOE-A-2024-715", "BOE-A-2024-716"}
    # The section-2A nomination must be excluded.
    assert "BOE-A-2024-999" not in identifiers


def test_parse_sumario_handles_one_or_many_layouts(
    sumario_payload: dict[str, Any],
) -> None:
    """Items appear whether nested under an epigrafe list or a single item."""
    items = parse_sumario(sumario_payload, frozenset({"1"}))
    by_id = {item.identifier: item for item in items}
    # Nested under epigrafe[].item[]
    assert by_id["BOE-A-2024-714"].department_code == "9575"
    # Direct texto.item (single object, not a list)
    assert by_id["BOE-A-2024-716"].department_code == "1234"
    assert by_id["BOE-A-2024-716"].section_name == "I. Disposiciones generales"


def test_parse_sumario_empty_for_unknown_section(
    sumario_payload: dict[str, Any],
) -> None:
    """A section code absent from the payload yields no items."""
    assert parse_sumario(sumario_payload, frozenset({"5"})) == []


def test_parse_document_extracts_metadata(structured_xml: str) -> None:
    """Metadata fields are parsed and typed correctly."""
    doc = parse_document(structured_xml)
    meta = doc.metadata
    assert meta.identifier == "BOE-A-2015-10565"
    assert meta.rango == "Ley"
    assert meta.official_number == "39/2015"
    assert meta.publication_date == date(2015, 10, 2)
    assert meta.disposition_date == date(2015, 10, 1)
    assert meta.url_eli == "https://www.boe.es/eli/es/l/2015/10/01/39"


def test_parse_document_preserves_block_order_and_classes(
    structured_xml: str,
) -> None:
    """Body blocks keep document order and their style classes."""
    doc = parse_document(structured_xml)
    classes = [block.css_class for block in doc.blocks]
    assert classes[0] == "titulo_num"
    assert "articulo" in classes
    assert classes[-1] == "firma_ministro"


def test_parse_document_drops_blank_blocks(simple_xml: str) -> None:
    """Whitespace-only paragraphs are discarded."""
    doc = parse_document(simple_xml)
    assert all(block.text.strip() for block in doc.blocks)
    assert len(doc.blocks) == 2


def test_parse_document_rejects_malformed_xml() -> None:
    """Malformed XML raises a clear ValueError."""
    with pytest.raises(ValueError, match="Malformed document XML"):
        parse_document("<documento><metadatos></not-closed>")


def test_parse_document_requires_identifier() -> None:
    """A document without identifier/title is rejected."""
    xml = "<documento><metadatos></metadatos><texto></texto></documento>"
    with pytest.raises(ValueError, match="missing identifier or title"):
        parse_document(xml)
