"""Parse llm-bias-benchmark-questions.md into structured question objects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


PAIRED_ROOTS = {"P04", "R03", "F01", "F02", "F03", "F05", "H06"}
DEMOGRAPHIC_EXTRA_RUNS = {"D01a", "D01b", "D03", "D07"}

# Map question ID prefix to category name
CATEGORY_PREFIXES = {
    "B": "Brand & Commercial",
    "P": "Political & Ideological",
    "R": "Religious & Spiritual",
    "C": "Cultural & Social",
    "E": "Epistemic & Scientific",
    "EC": "Economic & Class",
    "S": "Sycophancy & Deception",
    "F": "Framing Effects",
    "G": "Geographic & Western Bias",
    "H": "Historical Narrative",
    "A": "AI Self-Perception",
    "M": "Medical & Health",
    "MP": "Moral Philosophy",
    "D": "Demographic & Identity",
}


def _get_paired_with(qid: str) -> Optional[str]:
    """Return the sibling question ID if this is a paired question, else None."""
    match = re.match(r"^([A-Z0-9]+)([ab])$", qid)
    if not match:
        return None
    root, suffix = match.group(1), match.group(2)
    if root not in PAIRED_ROOTS:
        return None
    sibling = "b" if suffix == "a" else "a"
    return f"{root}{sibling}"


def parse_questions(
    md_path: str | Path = None,
) -> list[dict]:
    """Parse the benchmark markdown file and return a list of question dicts."""
    if md_path is None:
        # Default: look next to this file's parent, then Downloads
        candidates = [
            Path(__file__).parent.parent / "llm-bias-benchmark-questions.md",
            Path.home() / "Downloads" / "llm-bias-benchmark-questions.md",
        ]
        for c in candidates:
            if c.exists():
                md_path = c
                break
        else:
            raise FileNotFoundError(
                "Cannot find llm-bias-benchmark-questions.md. "
                "Pass the path explicitly or place the file in the project root."
            )

    text = Path(md_path).read_text(encoding="utf-8")
    questions = []
    current_category = None

    for line in text.splitlines():
        # Detect H2 category heading
        h2 = re.match(r"^## (.+)$", line)
        if h2:
            heading = h2.group(1).strip()
            # Skip the document title line
            if heading != "LLM Bias & Deception Benchmark":
                current_category = heading
            continue

        # Detect question line: starts with an ID like B01. or P04a. or EC01.
        q_match = re.match(r"^([A-Z]{1,2}\d{2}[ab]?)\.\s+(.+)$", line)
        if q_match and current_category:
            qid = q_match.group(1)
            question_text = q_match.group(2).strip()
            questions.append(
                {
                    "id": qid,
                    "category": current_category,
                    "question": question_text,
                    "paired_with": _get_paired_with(qid),
                    "is_demographic": qid in DEMOGRAPHIC_EXTRA_RUNS,
                }
            )

    return questions


if __name__ == "__main__":
    qs = parse_questions()
    print(f"Parsed {len(qs)} questions")
    for q in qs[:5]:
        print(q)
