"""Parsers turning raw BOE payloads into typed models.

Two responsibilities:

* :func:`parse_sumario` flattens the deeply nested daily-index JSON into a flat
  list of document references, filtered to the configured sections.
* :func:`parse_document` turns a single document's XML into a
  :class:`~boe_rag.models.BoeDocument` with structured metadata and ordered
  body blocks.

The BOE JSON uses the common "one object or a list of objects" pattern at every
level (a day with one section serialises that section as an object, not a
one-element list), so every traversal goes through :func:`_as_list`.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any
from xml.etree import ElementTree as ET

from boe_rag.config import DEFAULT_SECTIONS
from boe_rag.models import BoeDocument, DocumentMetadata, SumarioItem, TextBlock

logger = logging.getLogger(__name__)

# `\s` matches Unicode whitespace, including the non-breaking spaces (U+00A0)
# the BOE XML is littered with, so a single class collapses them all.
_WHITESPACE_RE = re.compile(r"\s+")


def _as_list(value: Any) -> list[Any]:
    """Normalise a "one object or many" BOE field into a list.

    Args:
        value: A dict, a list of dicts, or ``None``.

    Returns:
        A list: empty for ``None``, the value itself if already a list, or a
        single-element list otherwise.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_text(raw: str) -> str:
    """Collapse whitespace and strip a text fragment."""
    return _WHITESPACE_RE.sub(" ", raw).strip()


def parse_sumario(
    payload: dict[str, Any],
    sections: frozenset[str] = DEFAULT_SECTIONS,
) -> list[SumarioItem]:
    """Extract document references from a daily sumario payload.

    Args:
        payload: Parsed JSON returned by the sumario endpoint.
        sections: Section codes to keep; items in other sections are skipped.

    Returns:
        Document references in publication order, filtered to ``sections``.
    """
    sumario = payload.get("data", {}).get("sumario", {})
    items: list[SumarioItem] = []
    for diario in _as_list(sumario.get("diario")):
        for seccion in _as_list(diario.get("seccion")):
            section_code = str(seccion.get("codigo", ""))
            if section_code not in sections:
                continue
            section_name = str(seccion.get("nombre", ""))
            items.extend(_parse_section(seccion, section_code, section_name))
    return items


def _parse_section(
    seccion: dict[str, Any],
    section_code: str,
    section_name: str,
) -> list[SumarioItem]:
    """Extract items from one section of a sumario."""
    items: list[SumarioItem] = []
    for dep in _as_list(seccion.get("departamento")):
        dep_code = str(dep.get("codigo", ""))
        dep_name = str(dep.get("nombre", ""))
        for raw_item in _iter_department_items(dep):
            parsed = _build_item(
                raw_item, section_code, section_name, dep_code, dep_name
            )
            if parsed is not None:
                items.append(parsed)
    return items


def _iter_department_items(dep: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield raw item dicts under a department, across both layout variants.

    Documents may sit directly under ``texto.item`` or be grouped under
    ``texto.epigrafe[].item``; this flattens both.
    """
    texto = dep.get("texto")
    if not isinstance(texto, dict):
        return []
    collected: list[dict[str, Any]] = list(_as_list(texto.get("item")))
    for epigrafe in _as_list(texto.get("epigrafe")):
        collected.extend(_as_list(epigrafe.get("item")))
    return collected


def _build_item(
    raw: dict[str, Any],
    section_code: str,
    section_name: str,
    dep_code: str,
    dep_name: str,
) -> SumarioItem | None:
    """Build a :class:`SumarioItem` from a raw item dict, or ``None`` if invalid."""
    identifier = raw.get("identificador")
    title = raw.get("titulo")
    url_xml = raw.get("url_xml")
    if not (identifier and title and url_xml):
        logger.warning("Skipping malformed sumario item: %r", raw)
        return None
    return SumarioItem(
        identifier=str(identifier),
        title=_clean_text(str(title)),
        url_xml=str(url_xml),
        url_html=str(raw.get("url_html", "")),
        section_code=section_code,
        section_name=section_name,
        department_code=dep_code,
        department_name=dep_name,
    )


def _parse_boe_date(value: str | None) -> date | None:
    """Parse a ``YYYYMMDD`` BOE date string, returning ``None`` on failure."""
    if not value or not value.isdigit() or len(value) != 8:
        return None
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except ValueError:
        return None


def _text_of(element: ET.Element | None) -> str | None:
    """Return the cleaned text of an element, or ``None`` if empty/missing."""
    if element is None or element.text is None:
        return None
    cleaned = _clean_text(element.text)
    return cleaned or None


def parse_document(xml_text: str) -> BoeDocument:
    """Parse a document's XML into a structured :class:`BoeDocument`.

    Args:
        xml_text: Raw document XML from :meth:`BoeClient.fetch_document_xml`.

    Returns:
        The parsed document with metadata and ordered body blocks.

    Raises:
        ValueError: If the XML is malformed or missing the document identifier.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as err:
        raise ValueError(f"Malformed document XML: {err}") from err

    metadata = _parse_metadata(root)
    blocks = _parse_blocks(root)
    return BoeDocument(metadata=metadata, blocks=tuple(blocks))


def _parse_metadata(root: ET.Element) -> DocumentMetadata:
    """Extract :class:`DocumentMetadata` from the ``<metadatos>`` block."""
    meta = root.find("metadatos")
    if meta is None:
        raise ValueError("Document XML has no <metadatos> block")
    identifier = _text_of(meta.find("identificador"))
    title = _text_of(meta.find("titulo"))
    if not identifier or not title:
        raise ValueError("Document XML missing identifier or title")
    return DocumentMetadata(
        identifier=identifier,
        title=title,
        rango=_text_of(meta.find("rango")),
        department=_text_of(meta.find("departamento")),
        publication_date=_parse_boe_date(_text_of(meta.find("fecha_publicacion"))),
        disposition_date=_parse_boe_date(_text_of(meta.find("fecha_disposicion"))),
        official_number=_text_of(meta.find("numero_oficial")),
        section_code=_text_of(meta.find("seccion")),
        url_eli=_text_of(meta.find("url_eli")),
        url_pdf=_text_of(meta.find("url_pdf")),
    )


def _parse_blocks(root: ET.Element) -> list[TextBlock]:
    """Extract ordered, non-empty body blocks from the ``<texto>`` block."""
    texto = root.find("texto")
    if texto is None:
        return []
    blocks: list[TextBlock] = []
    for para in texto.findall("p"):
        text = _clean_text("".join(para.itertext()))
        if not text:
            continue
        css_class = para.get("class", "parrafo")
        blocks.append(TextBlock(css_class=css_class, text=text))
    return blocks
