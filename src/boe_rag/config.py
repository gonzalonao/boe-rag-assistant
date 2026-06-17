"""Configuration constants and the ingestion settings model.

Centralises every tunable value for the ingestion pipeline so the rest of the
package never hard-codes URLs, timeouts, or scoping rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Base URL of the BOE open-data sumario (daily index) endpoint.
SUMARIO_API_URL = "https://www.boe.es/datosabiertos/api/boe/sumario"

#: URL template for a single document's XML, formatted with its identifier.
DOCUMENT_XML_URL = "https://www.boe.es/diario_boe/xml.php?id={identifier}"

#: Public HTML view of a document, used for human-facing citations.
DOCUMENT_HTML_URL = "https://www.boe.es/diario_boe/txt.php?id={identifier}"

#: Section code of "I. Disposiciones generales" — the highest-value, most
#: durable content in the BOE and the corpus scope for this project.
SECTION_DISPOSICIONES_GENERALES = "1"

#: Default set of section codes to ingest.
DEFAULT_SECTIONS: frozenset[str] = frozenset({SECTION_DISPOSICIONES_GENERALES})


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    """Tunable parameters for a corpus ingestion run.

    Attributes:
        sections: BOE section codes to keep. Documents in other sections are
            skipped during sumario parsing.
        request_timeout_s: Per-request timeout in seconds.
        min_request_interval_s: Minimum delay between consecutive HTTP requests,
            used to stay polite to the public BOE servers.
        max_retries: Number of retry attempts for transient HTTP failures.
        max_chunk_chars: Soft upper bound on a chunk's character length before
            it is split. Article-level chunks below this are kept whole.
        min_chunk_chars: Chunks shorter than this are merged with their
            neighbour to avoid fragments that pollute retrieval.
    """

    sections: frozenset[str] = field(default=DEFAULT_SECTIONS)
    request_timeout_s: float = 30.0
    min_request_interval_s: float = 0.5
    max_retries: int = 4
    max_chunk_chars: int = 1800
    min_chunk_chars: int = 120
