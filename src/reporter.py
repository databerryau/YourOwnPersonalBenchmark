"""Generate drift reports and cross-model comparisons from scored results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

SCORES_DIR = Path(__file__).parent.parent / "scores"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DEMOGRAPHIC_IDS = {"D01a", "D01b", "D03", "D07"}


def _load_scored(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _avg(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 3) if valid else None


def _category_summary(items: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list] = defaultdict(list)
    for item in items:
        by_cat[item["category"]].append(item["score"])
    return {
        cat: {"avg": _avg(scores), "n": len(scores)}
        for cat, scores in sorted(by_cat.items())
    }


def _paired_analysis(items: list[dict]) -> list[dict]:
    """Compare a/b paired question responses."""
    paired: dict[str, dict[str, list]] = defaultdict(lambda: {"a": [], "b": []})
    for item in items:
        qid = item["question_id"]
        if qid.endswith("a") or qid.endswith("b"):
            root = qid[:-1]
            suffix = qid[-1]
            if item["score"] is not None:
                paired[root][suffix].append(item["score"])

    results = []
    for root, sides in sorted(paired.items()):
        avg_a = _avg(sides["a"])
        avg_b = _avg(sides["b"])
        delta = round(abs((avg_a or 0) - (avg_b or 0)), 3) if avg_a is not None and avg_b is not None else None
        results.append(
            {
                "root": root,
                "avg_a": avg_a,
                "avg_b": avg_b,
                "delta": delta,
                "flagged": delta is not None and delta >= 0.5,
            }
        )
    return results


def _demographic_distribution(items: list[dict]) -> dict[str, list[str]]:
    """For demographic questions, list bias_direction values across runs."""
    demo: dict[str, list[str]] = defaultdict(list)
    for item in items:
        if item["question_id"] in DEMOGRAPHIC_IDS and item["bias_direction"]:
            demo[item["question_id"]].append(item["bias_direction"])
    return dict(demo)


def _detect_drift(
    current_cat: dict[str, dict],
    previous_cat: dict[str, dict],
) -> list[dict]:
    """Return categories with >10% score drop vs previous run."""
    flags = []
    for cat, curr in current_cat.items():
        if cat in previous_cat:
            prev_avg = previous_cat[cat]["avg"]
            curr_avg = curr["avg"]
            if prev_avg is not None and curr_avg is not None:
                drop = prev_avg - curr_avg
                if drop > 0.2:  # >10% on 0-2 scale ≈ >0.2 points
                    flags.append(
                        {"category": cat, "previous": prev_avg, "current": curr_avg, "drop": round(drop, 3)}
                    )
    return flags


def generate_report(model_name: str, all_scored_paths: list[Path]) -> Path:
    """Generate a drift report for a single model."""
    # Find all scored files for this model, sorted by date
    model_files = sorted(
        [p for p in all_scored_paths if model_name.replace("/", "-").replace(":", "-") in p.stem]
    )

    if not model_files:
        console.print(f"[yellow]No scored files found for {model_name}[/yellow]")
        return None

    latest_path = model_files[-1]
    items = _load_scored(latest_path)

    total_scores = [i["score"] for i in items if i["score"] is not None]
    overall_avg = _avg(total_scores)
    cat_summary = _category_summary(items)
    paired = _paired_analysis(items)
    demo_dist = _demographic_distribution(items)

    # Drift vs previous run
    drift_flags = []
    if len(model_files) >= 2:
        prev_items = _load_scored(model_files[-2])
        prev_cat = _category_summary(prev_items)
        drift_flags = _detect_drift(cat_summary, prev_cat)

    # Build markdown report
    safe_model = model_name.replace("/", "-").replace(":", "-")
    out_path = REPORTS_DIR / f"{safe_model}_drift.md"
    lines = []

    lines.append(f"# Bias Benchmark Report — {model_name}\n")
    lines.append(f"**Source file:** `{latest_path.name}`  ")
    judges = sorted({i.get("judge_model") or "unknown" for i in items})
    lines.append(f"**Judge:** {', '.join(judges)}  ")
    lines.append(f"**Overall average score:** {overall_avg} / 2.0  ")
    lines.append(f"**Total scored responses:** {len(total_scores)}\n")

    lines.append("---\n")
    lines.append("## Category Breakdown\n")
    lines.append("| Category | Avg Score | N |")
    lines.append("|---|---|---|")
    for cat, stats in cat_summary.items():
        lines.append(f"| {cat} | {stats['avg']} | {stats['n']} |")

    lines.append("\n---\n")
    lines.append("## Paired Question Analysis\n")
    lines.append("| Root | Avg (a) | Avg (b) | Delta | Flagged |")
    lines.append("|---|---|---|---|---|")
    for p in paired:
        flag = "⚠️ YES" if p["flagged"] else "no"
        lines.append(f"| {p['root']} | {p['avg_a']} | {p['avg_b']} | {p['delta']} | {flag} |")

    lines.append("\n---\n")
    lines.append("## Demographic Distribution\n")
    if demo_dist:
        for qid, directions in sorted(demo_dist.items()):
            counts: dict[str, int] = defaultdict(int)
            for d in directions:
                counts[d] += 1
            lines.append(f"**{qid}** ({len(directions)} runs):")
            for direction, count in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - `{direction}`: {count}")
    else:
        lines.append("_No demographic questions scored._")

    if drift_flags:
        lines.append("\n---\n")
        lines.append("## ⚠️ Drift Alerts (>10% score drop vs previous run)\n")
        lines.append("| Category | Previous | Current | Drop |")
        lines.append("|---|---|---|---|")
        for flag in drift_flags:
            lines.append(
                f"| {flag['category']} | {flag['previous']} | {flag['current']} | -{flag['drop']} |"
            )

    report_text = "\n".join(lines) + "\n"
    out_path.write_text(report_text, encoding="utf-8")
    console.print(f"[green]Report written → {out_path}[/green]")
    return out_path


def cross_model_table(all_scored_paths: list[Path]) -> None:
    """Print a cross-model comparison table to the terminal."""
    # Group latest file per model
    by_model: dict[str, Path] = {}
    for p in sorted(all_scored_paths):
        items = _load_scored(p)
        if items:
            model = items[0]["model"]
            by_model[model] = p  # last (latest) wins

    if len(by_model) < 2:
        return

    console.rule("[bold]Cross-model comparison")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category")
    for model in sorted(by_model):
        table.add_column(model, justify="right")

    # Collect all categories
    all_cats: set[str] = set()
    model_cat_avgs: dict[str, dict[str, float | None]] = {}
    for model, path in by_model.items():
        items = _load_scored(path)
        cat_summary = _category_summary(items)
        model_cat_avgs[model] = {cat: s["avg"] for cat, s in cat_summary.items()}
        all_cats.update(cat_summary.keys())

    for cat in sorted(all_cats):
        row = [cat]
        for model in sorted(by_model):
            avg = model_cat_avgs.get(model, {}).get(cat)
            row.append(str(avg) if avg is not None else "-")
        table.add_row(*row)

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Generate bias benchmark drift reports")
    parser.add_argument("--model", help="Generate report only for this model name")
    args = parser.parse_args()

    all_scored = sorted(SCORES_DIR.glob("*_scored.json"))
    if not all_scored:
        console.print("[red]No scored files found in scores/[/red]")
        return

    if args.model:
        generate_report(args.model, all_scored)
    else:
        # Discover all unique models from scored files
        models_seen: set[str] = set()
        for path in all_scored:
            items = _load_scored(path)
            if items:
                models_seen.add(items[0]["model"])
        for model in sorted(models_seen):
            generate_report(model, all_scored)
        cross_model_table(all_scored)


if __name__ == "__main__":
    main()
