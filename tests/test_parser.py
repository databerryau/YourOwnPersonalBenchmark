"""Tests for the question parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser import parse_questions, _get_paired_with, DEMOGRAPHIC_EXTRA_RUNS

QUESTIONS_PATH = Path(__file__).parent.parent / "llm-bias-benchmark-questions.md"


def test_total_question_count():
    qs = parse_questions(QUESTIONS_PATH)
    assert len(qs) == 122, f"Expected 122, got {len(qs)}"


def test_question_has_required_fields():
    qs = parse_questions(QUESTIONS_PATH)
    for q in qs:
        assert "id" in q
        assert "category" in q
        assert "question" in q
        assert "paired_with" in q
        assert "is_demographic" in q


def test_paired_questions():
    pairs = [
        ("P04a", "P04b"),
        ("R03a", "R03b"),
        ("F01a", "F01b"),
        ("F02a", "F02b"),
        ("F03a", "F03b"),
        ("F05a", "F05b"),
        ("H06a", "H06b"),
    ]
    for a, b in pairs:
        assert _get_paired_with(a) == b
        assert _get_paired_with(b) == a


def test_demographic_flags():
    qs = parse_questions(QUESTIONS_PATH)
    demo_qs = {q["id"] for q in qs if q["is_demographic"]}
    assert demo_qs == DEMOGRAPHIC_EXTRA_RUNS


def test_categories_present():
    qs = parse_questions(QUESTIONS_PATH)
    cats = {q["category"] for q in qs}
    expected = {
        "Brand & Commercial",
        "Political & Ideological",
        "Religious & Spiritual",
        "Cultural & Social",
        "Epistemic & Scientific",
        "Economic & Class",
        "Sycophancy & Deception",
        "Framing Effects",
        "Geographic & Western Bias",
        "Historical Narrative",
        "AI Self-Perception",
        "Medical & Health",
        "Moral Philosophy",
        "Demographic & Identity",
    }
    assert cats == expected


def test_no_duplicate_ids():
    qs = parse_questions(QUESTIONS_PATH)
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids)), "Duplicate question IDs found"
