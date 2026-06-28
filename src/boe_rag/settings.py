"""Centralised, typed runtime settings via pydantic-settings.

One declarative source for every environment-driven knob — corpus/embeddings/report
paths, the LLM provider credentials and model overrides, and the optional Langfuse
keys — instead of ``os.environ.get`` calls scattered across the service and provider
layers. (Ingestion-pipeline constants live separately in :mod:`boe_rag.config`.)

Values come from the process environment and, for local development, an optional
``.env`` file at the repository root (copy ``.env.example`` to ``.env``). **The real
environment always wins over the file.**

Loading the ``.env`` file is done *only at application/CLI entrypoints* via
:func:`load_environment` — never as an import side effect of library code — so the
unit tests, which drive the providers with ``monkeypatch``-ed environment variables,
are never perturbed by a developer's local ``.env``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

#: Repository root: ``src/boe_rag/settings.py`` → parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Default ``.env`` location, resolved against the repo root so it is found
#: regardless of the working directory the app or a script is launched from.
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Typed view of every environment-driven runtime setting.

    Each field is optional: an unset knob stays ``None`` and the consuming code
    keeps its own default, preserving the previous ``os.environ.get(name, default)``
    behaviour. Field values are read case-insensitively from the environment and an
    optional ``.env`` file (environment wins).

    Attributes:
        corpus_path: Corpus Parquet path (``BOE_CORPUS_PATH``).
        embeddings_path: Precomputed-embeddings ``.npz`` (``BOE_EMBEDDINGS_PATH``).
        reports_dir: Directory of eval report JSON (``BOE_REPORTS_DIR``).
        cors_origins: Comma-separated browser origins allowed to call the API
            cross-origin (``BOE_CORS_ORIGINS``); see :attr:`cors_origins_list`.
        qdrant_url: Qdrant server URL (``QDRANT_URL``); when set (or
            ``qdrant_path``), the dense leg is served from Qdrant instead of the
            in-memory NumPy index.
        qdrant_path: Local embedded-Qdrant directory (``QDRANT_PATH``); a
            server-free alternative to ``qdrant_url``.
        qdrant_collection: Qdrant collection name (``QDRANT_COLLECTION``).
        qdrant_api_key: Optional Qdrant API key (``QDRANT_API_KEY``).
        openrouter_api_key: OpenRouter credential (``OPENROUTER_API_KEY``).
        openrouter_model: OpenRouter model override (``OPENROUTER_MODEL``).
        groq_api_key: Groq credential (``GROQ_API_KEY``).
        groq_model: Groq model override (``GROQ_MODEL``).
        gemini_api_key: Gemini credential (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``).
        gemini_model: Gemini model override (``GEMINI_MODEL``).
        langfuse_public_key: Langfuse public key (``LANGFUSE_PUBLIC_KEY``).
        langfuse_secret_key: Langfuse secret key (``LANGFUSE_SECRET_KEY``).
        langfuse_host: Langfuse host override (``LANGFUSE_HOST``).
    """

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    corpus_path: Path | None = Field(default=None, validation_alias="BOE_CORPUS_PATH")
    embeddings_path: Path | None = Field(
        default=None, validation_alias="BOE_EMBEDDINGS_PATH"
    )
    reports_dir: Path | None = Field(default=None, validation_alias="BOE_REPORTS_DIR")

    cors_origins: str | None = Field(default=None, validation_alias="BOE_CORS_ORIGINS")

    qdrant_url: str | None = Field(default=None, validation_alias="QDRANT_URL")
    qdrant_path: str | None = Field(default=None, validation_alias="QDRANT_PATH")
    qdrant_collection: str | None = Field(
        default=None, validation_alias="QDRANT_COLLECTION"
    )
    qdrant_api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")

    openrouter_api_key: str | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    openrouter_model: str | None = Field(
        default=None, validation_alias="OPENROUTER_MODEL"
    )
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_model: str | None = Field(default=None, validation_alias="GROQ_MODEL")
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model: str | None = Field(default=None, validation_alias="GEMINI_MODEL")

    langfuse_public_key: str | None = Field(
        default=None, validation_alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key: str | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")

    @property
    def cors_origins_list(self) -> list[str]:
        """Allowed CORS origins, parsed from the comma-separated ``cors_origins``.

        Returns the configured browser origins (e.g. the deployed frontend URL)
        that may call the JSON API cross-origin; empty when unset, in which case
        no CORS middleware is added and the API is same-origin only.
        """
        if not self.cors_origins:
            return []
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    def as_env(self) -> dict[str, str]:
        """Return the set values keyed by their canonical environment-variable name.

        Only non-``None`` fields are included. Used by :func:`load_environment` to
        bridge ``.env`` values to the parts of the codebase (LLM providers, the
        Langfuse client) that still read ``os.environ`` directly.

        Returns:
            A mapping of environment-variable name to string value for every
            configured setting.
        """
        candidates: dict[str, str | Path | None] = {
            "BOE_CORPUS_PATH": self.corpus_path,
            "BOE_EMBEDDINGS_PATH": self.embeddings_path,
            "BOE_REPORTS_DIR": self.reports_dir,
            "QDRANT_URL": self.qdrant_url,
            "QDRANT_PATH": self.qdrant_path,
            "QDRANT_COLLECTION": self.qdrant_collection,
            "QDRANT_API_KEY": self.qdrant_api_key,
            "OPENROUTER_API_KEY": self.openrouter_api_key,
            "OPENROUTER_MODEL": self.openrouter_model,
            "GROQ_API_KEY": self.groq_api_key,
            "GROQ_MODEL": self.groq_model,
            "GEMINI_API_KEY": self.gemini_api_key,
            "GEMINI_MODEL": self.gemini_model,
            "LANGFUSE_PUBLIC_KEY": self.langfuse_public_key,
            "LANGFUSE_SECRET_KEY": self.langfuse_secret_key,
            "LANGFUSE_HOST": self.langfuse_host,
        }
        return {k: str(v) for k, v in candidates.items() if v is not None}


def load_environment(env_file: Path | None = None) -> Settings:
    """Load settings from the environment and ``.env``, exporting to ``os.environ``.

    Call this once at an application or CLI entrypoint (never from imported library
    code). It parses the environment plus the ``.env`` file into a :class:`Settings`,
    then copies every configured value into ``os.environ`` **without overriding a
    variable that is already set** — so a real environment variable always wins, and
    the downstream code that still reads ``os.environ`` directly (the LLM providers,
    the Langfuse client) transparently picks up ``.env`` values.

    Args:
        env_file: Override the ``.env`` path (mainly for tests); defaults to
            :data:`DEFAULT_ENV_FILE`.

    Returns:
        The populated :class:`Settings`, for callers that prefer the typed view.
    """
    path = env_file if env_file is not None else DEFAULT_ENV_FILE
    settings = Settings(_env_file=path)  # type: ignore[call-arg]
    exported: list[str] = []
    for name, value in settings.as_env().items():
        if name not in os.environ:
            os.environ[name] = value
            exported.append(name)
    if exported:
        logger.info(
            "Loaded %d setting(s) from %s into the environment.", len(exported), path
        )
    return settings
