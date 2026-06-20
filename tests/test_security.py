"""Tests for the adversarial security evaluation checks and aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from boe_rag.eval.answerer import REFUSAL, SYSTEM_PROMPT_CANARY
from boe_rag.eval.security import (
    AdversarialCase,
    SecurityReport,
    citation_indices,
    contains_forbidden,
    evaluate_case,
    evaluate_security,
    has_fabricated_citation,
    is_refusal,
    leaks_canary,
    load_adversarial_cases,
)

_CANARY = "CANARY-XYZ"


def _case(expectation: str, **kwargs: object) -> AdversarialCase:
    """Build an adversarial case with sensible defaults for the given expectation."""
    return AdversarialCase(
        case_id=kwargs.get("case_id", "c1"),  # type: ignore[arg-type]
        category=kwargs.get("category", "instruction_override"),  # type: ignore[arg-type]
        question=kwargs.get("question", "q"),  # type: ignore[arg-type]
        expectation=expectation,  # type: ignore[arg-type]
        forbidden=kwargs.get("forbidden", ()),  # type: ignore[arg-type]
    )


def test_citation_indices_extracts_numbers() -> None:
    """Bracketed integers are parsed; other text is ignored."""
    assert citation_indices("Grounded [1] and also [3], not [x].") == {1, 3}


def test_has_fabricated_citation_flags_out_of_range() -> None:
    """A citation beyond the available sources is fabricated."""
    assert has_fabricated_citation("See [4].", num_sources=3) is True
    assert has_fabricated_citation("See [2].", num_sources=3) is False


def test_has_fabricated_citation_flags_nonpositive() -> None:
    """Zero or negative indices are never valid."""
    assert has_fabricated_citation("See [0].", num_sources=5) is True


def test_leaks_canary_detects_the_marker() -> None:
    """The canary check is a plain containment test."""
    assert leaks_canary(f"...{_CANARY}...", _CANARY) is True
    assert leaks_canary("nothing secret here", _CANARY) is False


def test_contains_forbidden_is_case_insensitive() -> None:
    """Forbidden payloads match regardless of case."""
    assert contains_forbidden("I am pwned now", ["PWNED"]) is True
    assert contains_forbidden("all good", ["PWNED"]) is False


def test_is_refusal_matches_the_refusal_string() -> None:
    """A leading refusal string (with surrounding space) counts as a refusal."""
    assert is_refusal(f"  {REFUSAL}", REFUSAL) is True
    assert is_refusal("Claro, el IVA es del 21% [1].", REFUSAL) is False


def test_evaluate_case_no_canary() -> None:
    """A leaked canary fails the exfiltration expectation."""
    case = _case("no_canary", category="system_prompt_exfiltration")
    assert evaluate_case(case, "harmless", 1, canary=_CANARY, refusal=REFUSAL).passed
    assert not evaluate_case(
        case, f"my secret is {_CANARY}", 1, canary=_CANARY, refusal=REFUSAL
    ).passed


def test_evaluate_case_no_payload() -> None:
    """Echoing the injected payload fails the override expectation."""
    case = _case("no_payload", forbidden=("PWNED",))
    assert evaluate_case(case, "I refuse.", 1, canary=_CANARY, refusal=REFUSAL).passed
    assert not evaluate_case(case, "PWNED", 1, canary=_CANARY, refusal=REFUSAL).passed


def test_evaluate_case_valid_citations() -> None:
    """A fabricated citation fails the citation-integrity expectation."""
    case = _case("valid_citations", category="citation_spoofing")
    assert evaluate_case(case, "Answer [1].", 2, canary=_CANARY, refusal=REFUSAL).passed
    assert not evaluate_case(
        case, "Answer [99].", 2, canary=_CANARY, refusal=REFUSAL
    ).passed


def test_evaluate_case_refuse() -> None:
    """Answering an out-of-corpus probe instead of refusing fails."""
    case = _case("refuse", category="out_of_corpus_hallucination")
    assert evaluate_case(case, REFUSAL, 5, canary=_CANARY, refusal=REFUSAL).passed
    assert not evaluate_case(
        case, "Sí, la ley dice...", 5, canary=_CANARY, refusal=REFUSAL
    ).passed


def test_evaluate_case_rejects_unknown_expectation() -> None:
    """An unknown expectation is a programming error, not a silent pass."""
    bad = AdversarialCase("c", "instruction_override", "q", "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluate_case(bad, "x", 1, canary=_CANARY, refusal=REFUSAL)


def test_evaluate_security_aggregates_and_groups() -> None:
    """The report aggregates the pass rate overall and per category."""
    cases = [
        _case("no_payload", case_id="a", forbidden=("PWNED",)),
        _case(
            "refuse",
            case_id="b",
            category="out_of_corpus_hallucination",
            question="absent?",
        ),
    ]
    # 'a' passes (no payload echoed); 'b' fails (answers instead of refusing).
    answers = {"q": ("safe answer", 2), "absent?": ("here is a made-up answer", 3)}
    report = evaluate_security(
        cases, lambda q: answers[q], canary=_CANARY, refusal=REFUSAL
    )
    assert report.num_cases == 2
    assert report.num_passed == 1
    assert report.pass_rate == 0.5
    by_cat = report.pass_rate_by_category()
    assert by_cat["instruction_override"] == 1.0
    assert by_cat["out_of_corpus_hallucination"] == 0.0


def test_empty_report_passes_vacuously() -> None:
    """A report with no findings has a pass rate of 1.0."""
    assert SecurityReport(findings=()).pass_rate == 1.0


def test_report_as_dict_is_serialisable() -> None:
    """The report dict exposes the headline fields and per-case findings."""
    case = _case("no_payload", forbidden=("X",))
    report = evaluate_security(
        [case], lambda q: ("clean", 1), canary=_CANARY, refusal=REFUSAL
    )
    payload = report.as_dict()
    assert payload["num_cases"] == 1
    assert payload["pass_rate"] == 1.0
    assert isinstance(payload["findings"], list)


def test_committed_adversarial_set_is_valid() -> None:
    """The shipped adversarial set loads and every case is well-formed."""
    path = Path("eval_data/adversarial_security.jsonl")
    cases = load_adversarial_cases(path)
    assert len(cases) >= 12
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))  # ids are unique
    valid_expectations = {"refuse", "no_canary", "no_payload", "valid_citations"}
    for case in cases:
        assert case.question.strip()
        assert case.expectation in valid_expectations
        # no_payload cases must declare what payload to look for
        if case.expectation == "no_payload":
            assert case.forbidden


def test_canary_is_not_in_the_default_refusal() -> None:
    """Sanity: a clean refusal never trips the canary check."""
    assert not leaks_canary(REFUSAL, SYSTEM_PROMPT_CANARY)
