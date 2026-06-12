"""Tests for package metadata."""

from boe_rag import __version__


def test_version_is_semver() -> None:
    """Package version follows MAJOR.MINOR.PATCH."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
