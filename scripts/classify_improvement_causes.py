#!/usr/bin/env python3
"""
Classify improved-to-correct cases into coarse improvement causes.

Categories:
  - graph_driven
  - graph_assisted_but_not_decisive
  - non_graph_or_runtime_recovery
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORY_META = {
    "graph_driven": {
        "label": "graph_driven",
        "description": "Graph retrieval appears to provide the decisive localization or candidate narrowing before the final visual confirmation.",
    },
    "graph_assisted_but_not_decisive": {
        "label": "graph_assisted_but_not_decisive",
        "description": "Graph is present in the successful trace, but the observed graph evidence is weak/mixed and the final win likely comes from later clip search or frame inspection.",
    },
    "non_graph_or_runtime_recovery": {
        "label": "non_graph_or_runtime_recovery",
        "description": "The improvement does not rely on graph retrieval, or it mainly reflects recovery from baseline runtime/termination failures.",
    },
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _to_int(row: dict[str, str], key: str) -> int:
    try:
        return int(str(row.get(key, "")).strip() or 0)
    except Exception:
        return 0


def _to_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def _tags(row: dict[str, str]) -> set[str]:
    value = str(row.get("tags", "")).strip()
    return {part for part in value.split("|") if part}


def _max_scores(graph_evidence: str) -> tuple[float, float]:
    score_vals = [float(x) for x in re.findall(r"score=([0-9.]+)", graph_evidence or "")]
    lex_vals = [float(x) for x in re.findall(r"lex=([0-9.]+)", graph_evidence or "")]
    max_score = max(score_vals) if score_vals else 0.0
    max_lex = max(lex_vals) if lex_vals else 0.0
    return max_score, max_lex


def _starts_with_graph(tool_path: str) -> bool:
    return tool_path.startswith("SG") or tool_path.startswith("FG")


def _classify(row: dict[str, str]) -> tuple[str, list[str]]:
    tags = _tags(row)
    new_sg = _to_int(row, "new_sg")
    new_fg = _to_int(row, "new_fg")
    graph_used = (new_sg + new_fg) > 0
    baseline_failure = row.get("baseline_verdict") == "UNKNOWN"
    baseline_missing_trace = row.get("baseline_end_status") in {"N/A", "llm_call_returned_none"}
    max_score, max_lex = _max_scores(row.get("new_graph_evidence", ""))
    graph_then_fi = _to_bool(row, "graph_then_frame_inspect")
    graph_best_temporal_hit = _to_bool(row, "new_graph_best_temporal_hit")
    direct_graph_path = _starts_with_graph(str(row.get("new_tool_path", "")))

    reasons: list[str] = []

    if not graph_used:
        if baseline_failure:
            reasons.append("baseline_failed_but_new_finished_without_graph")
        else:
            reasons.append("no_graph_used_in_success_trace")
        return "non_graph_or_runtime_recovery", reasons

    strong_graph = (
        (
            max_lex >= 1.0
            and (
                graph_then_fi
                or direct_graph_path
                or "replaced_broad_search_with_graph" in tags
            )
        )
        or graph_best_temporal_hit
        or "graph_temporal_hit" in tags
        or (
            max_score >= 4.0
            and graph_then_fi
            and (
                direct_graph_path
                or "temporal_question_helped_by_segment_graph" in tags
            )
        )
    )
    weak_graph = (
        max_lex < 1.0
        and max_score < 4.0
        and not graph_best_temporal_hit
        and "graph_temporal_hit" not in tags
    )

    if strong_graph and (
        graph_then_fi
        or direct_graph_path
        or "replaced_broad_search_with_graph" in tags
        or "new_segment_graph_retrieval" in tags
        or "new_frame_graph_retrieval" in tags
    ):
        if max_lex >= 1.0:
            reasons.append(f"strong_graph_lexical_match={max_lex:.3f}")
        elif max_score >= 4.0:
            reasons.append(f"strong_graph_score={max_score:.3f}")
        if graph_best_temporal_hit or "graph_temporal_hit" in tags:
            reasons.append("graph_hit_target_time")
        if graph_then_fi:
            reasons.append("graph_followed_by_visual_confirmation")
        return "graph_driven", reasons

    if weak_graph:
        if baseline_failure:
            reasons.append("baseline_failed")
        if baseline_missing_trace:
            reasons.append("baseline_missing_trace_end")
        reasons.append("graph_present_but_weak")
        if graph_then_fi:
            reasons.append("later_visual_confirmation_likely_decisive")
        return "graph_assisted_but_not_decisive", reasons

    if baseline_failure and not strong_graph:
        reasons.append("baseline_failed")
        reasons.append("graph_present_but_not_clearly_decisive")
        return "graph_assisted_but_not_decisive", reasons

    reasons.append("graph_used_with_mixed_signal")
    return "graph_assisted_but_not_decisive", reasons


def _render_report(output_path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    categorized: dict[str, list[dict[str, str]]] = defaultdict(list)
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        category, reasons = _classify(row)
        row["improvement_category"] = category
        row["improvement_reasons"] = "|".join(reasons)
        row["transition"] = f"{row['baseline_verdict']}->{row['new_verdict']}"
        categorized[category].append(row)
        transition_counts[category][row["transition"]] += 1

    total = len(rows)
    summary: dict[str, Any] = {
        "total_improved_to_correct": total,
        "categories": {},
    }

    lines: list[str] = []
    lines.append("# Improved-To-Correct Cause Split")
    lines.append("")
    lines.append(f"This report partitions the {total} `non-correct -> correct` cases into one coarse improvement cause.")
    lines.append("")
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Count | Share | UNKNOWN->CORRECT | WRONG->CORRECT |")
    lines.append("|---|---:|---:|---:|---:|")

    for category in CATEGORY_META:
        bucket = categorized.get(category, [])
        count = len(bucket)
        share = (count / total) if total else 0.0
        u2c = transition_counts[category]["UNKNOWN->CORRECT"]
        w2c = transition_counts[category]["WRONG->CORRECT"]
        summary["categories"][category] = {
            "label": CATEGORY_META[category]["label"],
            "description": CATEGORY_META[category]["description"],
            "count": count,
            "share": share,
            "transition_counts": dict(transition_counts[category]),
        }
        lines.append(f"| `{category}` | {count} | {share:.1%} | {u2c} | {w2c} |")

    for category in CATEGORY_META:
        bucket = categorized.get(category, [])
        lines.append("")
        lines.append(f"## {category}")
        lines.append("")
        lines.append(CATEGORY_META[category]["description"])
        lines.append("")
        lines.append("| qid | transition | video | baseline path | new path | reasons | graph evidence |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in bucket[:40]:
            lines.append(
                f"| {row['qid']} | {row['transition']} | {row['video_id']} | "
                f"{row['baseline_tool_path'].replace('|', '/')} | {row['new_tool_path'].replace('|', '/')} | "
                f"{row['improvement_reasons'].replace('|', ', ')} | {row['new_graph_evidence'].replace('|', '; ')} |"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify improved-to-correct cases by likely cause.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_csv)
    rows.sort(key=lambda row: (row.get("video_id", ""), int(row.get("qid", "0"))))

    summary = _render_report(output_dir / "improvement_3way_report.md", rows)
    _write_json(output_dir / "improvement_3way_summary.json", summary)

    categorized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        categorized[row["improvement_category"]].append(row)

    all_rows: list[dict[str, Any]] = []
    for category, bucket in categorized.items():
        bucket = sorted(bucket, key=lambda row: (row["transition"], row["video_id"], int(row["qid"])))
        categorized[category] = bucket
        _write_csv(output_dir / f"improvement_category_{category}.csv", bucket)
        all_rows.extend(bucket)

    all_rows.sort(key=lambda row: (row["improvement_category"], row["transition"], row["video_id"], int(row["qid"])))
    _write_csv(output_dir / "improvement_3way_all.csv", all_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
