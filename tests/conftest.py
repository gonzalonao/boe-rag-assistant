"""Shared pytest fixtures: loaders for the on-disk sample payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sumario_payload() -> dict[str, Any]:
    """Return the parsed sample sumario JSON payload."""
    data: dict[str, Any] = json.loads(
        (_FIXTURES / "sumario_sample.json").read_text(encoding="utf-8")
    )
    return data


@pytest.fixture
def structured_xml() -> str:
    """Return the XML of a richly structured law (títulos, capítulos, artículos)."""
    return (_FIXTURES / "document_structured.xml").read_text(encoding="utf-8")


@pytest.fixture
def simple_xml() -> str:
    """Return the XML of a flat resolution with no article structure."""
    return (_FIXTURES / "document_simple.xml").read_text(encoding="utf-8")
