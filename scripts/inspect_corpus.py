"""Inspect and validate a BOE corpus Parquet file.

Prints summary statistics and runs sanity checks so you can confirm a corpus is
well-formed before (or after) publishing it. Exits non-zero if any hard check
fails, so it can also gate CI.

Example:
    python scripts/inspect_corpus.py data/corpus/boe-2024.parquet
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

# Unicode replacement character: its presence means real mojibake in the data
# (as opposed to a mere console display glitch).
_REPLACEMENT_CHAR = "�"


def _load_rows(path: Path) -> list[dict[str, object]]:
    """Load the Parquet file into a list of row dicts."""
    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    rows: list[dict[str, object]] = table.to_pylist()
    return rows


def _print_summary(rows: list[dict[str, object]]) -> None:
    """Print headline statistics about the corpus."""
    docs = {r["document_id"] for r in rows}
    with_article = sum(1 for r in rows if r["articulo"])
    lengths = [len(str(r["text"])) for r in rows]
    print("=" * 60)
    print(f"File rows (chunks)      : {len(rows)}")
    print(f"Unique documents        : {len(docs)}")
    print(
        f"Chunks with an article  : {with_article} ({with_article * 100 // len(rows)}%)"
    )
    print(
        f"Text length  min/med/max: {min(lengths)} / "
        f"{int(statistics.median(lengths))} / {max(lengths)}"
    )
    print("=" * 60)


def _print_samples(rows: list[dict[str, object]], n: int = 4) -> None:
    """Print a few evenly-spaced sample chunks for eyeballing."""
    if not rows:
        return
    step = max(1, len(rows) // n)
    print("\nSample chunks:")
    for row in rows[::step][:n]:
        text = str(row["text"])
        preview = text[:140].replace("\n", " ")
        print("-" * 60)
        print(f"  id       : {row['chunk_id']}")
        print(f"  citation : {row['citation']}")
        print(f"  title    : {str(row['document_title'])[:80]}")
        print(f"  text     : {preview}")


def _run_checks(rows: list[dict[str, object]]) -> list[str]:
    """Run hard validation checks, returning a list of failure messages."""
    failures: list[str] = []
    if not rows:
        return ["corpus is empty"]

    ids = [str(r["chunk_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        failures.append("duplicate chunk_id values found")

    empty_text = sum(1 for r in rows if not str(r["text"]).strip())
    if empty_text:
        failures.append(f"{empty_text} chunks have empty text")

    missing_core = sum(
        1
        for r in rows
        if not r["chunk_id"] or not r["document_id"] or not r["citation"]
    )
    if missing_core:
        failures.append(f"{missing_core} chunks miss a core field (id/doc/citation)")

    mojibake = sum(1 for r in rows if _REPLACEMENT_CHAR in str(r["text"]))
    if mojibake:
        failures.append(f"{mojibake} chunks contain the Unicode replacement char")

    return failures


def main(argv: list[str] | None = None) -> int:
    """Inspect a corpus file and report stats and validation results.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Inspect a BOE corpus Parquet file.")
    parser.add_argument("path", type=Path, help="Path to the corpus .parquet file.")
    args = parser.parse_args(argv)

    if not args.path.is_file():
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        return 1

    rows = _load_rows(args.path)
    _print_summary(rows)
    _print_samples(rows)

    failures = _run_checks(rows)
    print("\nValidation:")
    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}")
        return 1
    print("  OK: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
