"""Tests for the typed runtime settings and the ``.env`` loader.

Verifies that ``.env`` values are read and exported, that a real environment
variable always wins over the file, and that ``load_environment`` never clobbers
an already-set variable — the property the LLM provider tests rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boe_rag.settings import Settings, load_environment


def _write_env(tmp_path: Path, body: str) -> Path:
    """Write a ``.env`` file and return its path."""
    env_file = tmp_path / ".env"
    env_file.write_text(body, encoding="utf-8")
    return env_file


def _settings(env_file: Path | None) -> Settings:
    """Build Settings reading ``env_file`` (None disables file loading).

    Wraps the pydantic-settings ``_env_file`` init kwarg, which the mypy plugin
    does not surface, in one typed place.
    """
    return Settings(_env_file=env_file)  # type: ignore[call-arg]


#: Every environment variable the settings model reads.
_ALL_ENV_VARS = (
    "BOE_CORPUS_PATH",
    "BOE_EMBEDDINGS_PATH",
    "BOE_REPORTS_DIR",
    "QDRANT_URL",
    "QDRANT_PATH",
    "QDRANT_COLLECTION",
    "QDRANT_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GOOGLE_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
)


def test_settings_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing configured, every field is None and as_env is empty."""
    for name in _ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    settings = _settings(env_file=None)
    assert settings.groq_api_key is None
    assert settings.corpus_path is None
    assert settings.as_env() == {}


def test_reads_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A value present only in the .env file is read into the typed field."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env_file = _write_env(tmp_path, "GROQ_API_KEY=from-file\n")
    settings = _settings(env_file=env_file)
    assert settings.groq_api_key == "from-file"


def test_environment_wins_over_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real environment variable overrides the .env file value."""
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    env_file = _write_env(tmp_path, "GROQ_API_KEY=from-file\n")
    assert _settings(env_file=env_file).groq_api_key == "from-env"


def test_gemini_accepts_google_api_key_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GOOGLE_API_KEY populates the gemini credential as an alias."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    assert _settings(env_file=None).gemini_api_key == "g-key"


def test_path_fields_are_parsed_as_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path-typed settings come back as Path objects."""
    monkeypatch.delenv("BOE_CORPUS_PATH", raising=False)
    env_file = _write_env(tmp_path, "BOE_CORPUS_PATH=data/corpus/x.parquet\n")
    corpus = _settings(env_file=env_file).corpus_path
    assert isinstance(corpus, Path)
    assert corpus == Path("data/corpus/x.parquet")


def test_as_env_uses_canonical_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """as_env keys values by their canonical environment-variable name."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("BOE_REPORTS_DIR", "reports")
    exported = _settings(env_file=None).as_env()
    assert exported["GEMINI_API_KEY"] == "g"
    assert exported["BOE_REPORTS_DIR"] == "reports"


def test_load_environment_exports_env_file_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_environment copies .env values into os.environ for legacy readers."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env_file = _write_env(tmp_path, "GROQ_API_KEY=exported\n")
    settings = load_environment(env_file)
    assert settings.groq_api_key == "exported"
    import os

    assert os.environ["GROQ_API_KEY"] == "exported"


def test_load_environment_does_not_override_real_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An already-set environment variable is never clobbered by the .env file."""
    monkeypatch.setenv("GROQ_API_KEY", "real")
    env_file = _write_env(tmp_path, "GROQ_API_KEY=from-file\n")
    load_environment(env_file)
    import os

    assert os.environ["GROQ_API_KEY"] == "real"


def test_load_environment_missing_file_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing .env file is fine: nothing is exported, no error."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = load_environment(tmp_path / "does-not-exist.env")
    assert settings.groq_api_key is None
