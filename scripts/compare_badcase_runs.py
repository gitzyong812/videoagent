#!/usr/bin/env python3
"""
Compare two badcase benchmark runs and export changed-question analysis.

Outputs:
  - summary.json
  - all_changes.csv
  - improved_to_correct.csv
  - regressed_from_correct.csv
  - comparison_report.md

This script is intentionally heuristic. It does not try to prove causality; it
surfaces review-friendly evidence and likely causes so changed traces can be
triaged quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


QUESTION_TAG_PATTERNS: list[tuple[str, str]] = [
    ("counting", r"\bhow many\b|\bnumber of\b"),
    ("color", r"\bwhat color\b|\bcolour\b"),
    ("appearance", r"\bwear\b|\bwearing\b|\bclothes\b|\bearrings\b|\bglasses\b|\bhat\b|\bshirt\b|\bdress\b"),
    ("identity", r"^\s*who\b|\bname\b"),
    ("temporal_reasoning", r"\bbefore\b|\bafter\b|\bfirst\b|\blast\b|\bwhen\b|\bwhile\b|\bwhat happens\b"),
    ("relationship", r"\brelationship\b"),
]

TOOL_ABBR = {
    "global_browse_tool": "GB",
    "clip_search_tool": "CS",
    "frame_inspect_tool": "FI",
    "frame_graph_query_tool": "FG",
    "segment_graph_query_tool": "SG",
    "finish": "FIN",
}


@dataclass
class TraceSummary:
    trace_id: str
    path: str
    verdict: str
    predicted: str
    expected: str
    question: str
    has_trace_end: bool
    trace_end_status: str
    step_count: int
    llm_calls: int
    tool_counts: dict[str, int]
    tool_sequence: list[str]
    tool_path: str
    graph_calls: int
    graph_then_frame_inspect: bool
    frame_inspect_calls: int
    global_browse_calls: int
    clip_search_calls: int
    frame_graph_calls: int
    segment_graph_calls: int
    graph_fallback_used: bool
    graph_best_temporal_hit: bool
    graph_best_match_count: int
    graph_evidence_notes: list[str]


def _safe_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _verdict_from_is_correct(value: Any) -> str:
    if value is True:
        return "CORRECT"
    if value is False:
        return "WRONG"
    return "UNKNOWN"


def _normalize_pred(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _trace_id_from_qid(video_id: str, qid: str) -> str:
    return f"{video_id}__{qid}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _tool_path(tool_sequence: list[str]) -> str:
    if not tool_sequence:
        return ""
    compact: list[str] = []
    prev = None
    run = 0
    for tool in tool_sequence:
        if tool == prev:
            run += 1
            compact[-1] = f"{TOOL_ABBR.get(tool, tool)}x{run}"
        else:
            prev = tool
            run = 1
            compact.append(TOOL_ABBR.get(tool, tool))
    return " > ".join(compact)


def _parse_graph_preview(preview: str) -> dict[str, Any] | None:
    text = (preview or "").strip()
    if not text or not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_trace_summary(path: Path) -> TraceSummary:
    events = _read_jsonl(path)
    trace_id = path.stem
    question = ""
    has_trace_end = False
    trace_end_status = ""
    step_count = 0
    llm_calls = 0
    answer_check: dict[str, Any] | None = None

    tool_rows: list[tuple[int, int, int, str, dict[str, Any]]] = []
    graph_fallback_used = False
    graph_best_temporal_hit = False
    graph_best_match_count = 0
    graph_evidence_notes: list[str] = []

    for event in events:
        event_type = event.get("event_type")
        if event_type == "trace_start":
            trace_id = str(event.get("trace_id", trace_id))
            question = str(event.get("question", ""))
        elif event_type == "trace_end":
            has_trace_end = True
            trace_end_status = str(event.get("status", ""))
        elif event_type == "answer_check":
            answer_check = event
        elif event_type == "llm_call":
            llm_calls += 1

        step_idx = event.get("step_idx")
        if isinstance(step_idx, int):
            step_count = max(step_count, step_idx)

        if event_type != "tool_call":
            continue

        step_i = event.get("step_idx", 0) if isinstance(event.get("step_idx"), int) else 0
        tool_call_idx = event.get("tool_call_idx", 0) if isinstance(event.get("tool_call_idx"), int) else 0
        event_idx = event.get("event_idx", 0) if isinstance(event.get("event_idx"), int) else 0
        tool_name = str(event.get("tool_name", "unknown"))
        tool_rows.append((step_i, tool_call_idx, event_idx, tool_name, event))

        if tool_name not in {"frame_graph_query_tool", "segment_graph_query_tool"}:
            continue

        preview = str(((event.get("tool_output_summary") or {}).get("preview")) or "")
        parsed = _parse_graph_preview(preview)
        if not parsed:
            continue

        graph_fallback_used = graph_fallback_used or bool(parsed.get("fallback_used"))
        matched = parsed.get("matched_frames") or parsed.get("matched_segments") or []
        if isinstance(matched, list):
            graph_best_match_count = max(graph_best_match_count, len(matched))
            if matched:
                top = matched[0]
                if bool(top.get("temporal_in_target")) or float(top.get("temporal_distance_sec", 1.0) or 1.0) == 0.0:
                    graph_best_temporal_hit = True
                top_score = float(top.get("score", 0.0) or 0.0)
                top_lex = float(top.get("lexical_score", 0.0) or 0.0)
                top_dist = float(top.get("temporal_distance_sec", 0.0) or 0.0)
                graph_evidence_notes.append(
                    f"{TOOL_ABBR.get(tool_name, tool_name)} score={top_score:.3f} lex={top_lex:.3f} dist={top_dist:.1f}"
                )

    verdict = "UNKNOWN"
    predicted = ""
    expected = ""
    if isinstance(answer_check, dict):
        verdict = _verdict_from_is_correct(answer_check.get("is_correct"))
        predicted = _normalize_pred(
            answer_check.get("predicted_answer_normalized") or answer_check.get("predicted_answer_raw")
        )
        expected = _normalize_pred(
            answer_check.get("expected_answer_normalized") or answer_check.get("expected_answer_raw")
        )

    tool_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    tool_sequence = [row[3] for row in tool_rows]
    tool_counts = Counter(tool_sequence)
    graph_calls = tool_counts.get("frame_graph_query_tool", 0) + tool_counts.get("segment_graph_query_tool", 0)

    first_graph_idx = None
    graph_then_frame_inspect = False
    for idx, tool_name in enumerate(tool_sequence):
        if tool_name in {"frame_graph_query_tool", "segment_graph_query_tool"} and first_graph_idx is None:
            first_graph_idx = idx
        if first_graph_idx is not None and idx > first_graph_idx and tool_name == "frame_inspect_tool":
            graph_then_frame_inspect = True
            break

    return TraceSummary(
        trace_id=trace_id,
        path=str(path),
        verdict=verdict,
        predicted=predicted,
        expected=expected,
        question=question,
        has_trace_end=has_trace_end,
        trace_end_status=trace_end_status or "N/A",
        step_count=step_count,
        llm_calls=llm_calls,
        tool_counts=dict(sorted(tool_counts.items())),
        tool_sequence=tool_sequence,
        tool_path=_tool_path(tool_sequence),
        graph_calls=graph_calls,
        graph_then_frame_inspect=graph_then_frame_inspect,
        frame_inspect_calls=tool_counts.get("frame_inspect_tool", 0),
        global_browse_calls=tool_counts.get("global_browse_tool", 0),
        clip_search_calls=tool_counts.get("clip_search_tool", 0),
        frame_graph_calls=tool_counts.get("frame_graph_query_tool", 0),
        segment_graph_calls=tool_counts.get("segment_graph_query_tool", 0),
        graph_fallback_used=graph_fallback_used,
        graph_best_temporal_hit=graph_best_temporal_hit,
        graph_best_match_count=graph_best_match_count,
        graph_evidence_notes=graph_evidence_notes[:8],
    )


def _load_meta(meta_path: Path) -> dict[str, dict[str, Any]]:
    by_qid: dict[str, dict[str, Any]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            video_id = str(row.get("key", ""))
            for qa in row.get("qa", []):
                qid = str(qa.get("uid", ""))
                by_qid[qid] = {
                    "qid": qid,
                    "video_id": video_id,
                    "question": str(qa.get("question", "")),
                    "answer": _normalize_pred(qa.get("answer")),
                    "question_type": list(qa.get("question_type", [])),
                    "time_reference": str(qa.get("time_reference", "")),
                    "trace_id": _trace_id_from_qid(video_id, qid),
                }
    return by_qid


def _load_baseline_trace_index(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace_id = str(row.get("trace_id", ""))
            qid = trace_id.split("__", 1)[1] if "__" in trace_id else trace_id
            rows[qid] = row
    return rows


def _load_trace_summaries(trace_dir: Path) -> dict[str, TraceSummary]:
    out: dict[str, TraceSummary] = {}
    for path in sorted(trace_dir.glob("*.jsonl")):
        summary = _parse_trace_summary(path)
        qid = summary.trace_id.split("__", 1)[1] if "__" in summary.trace_id else summary.trace_id
        out[qid] = summary
    return out


def _load_new_benchmark_results(path: Path) -> dict[str, dict[str, Any]]:
    data = _safe_load_json(path)
    out: dict[str, dict[str, Any]] = {}
    for qid, row in data.items():
        out[str(qid)] = row if isinstance(row, dict) else {}
    return out


def _question_tags(question: str) -> list[str]:
    q = (question or "").strip()
    low = q.lower()
    tags = [name for name, pattern in QUESTION_TAG_PATTERNS if re.search(pattern, low)]
    return tags


def _change_bucket(base: TraceSummary, new: TraceSummary) -> str | None:
    if base.verdict != "CORRECT" and new.verdict == "CORRECT":
        return "improved_to_correct"
    if base.verdict == "CORRECT" and new.verdict != "CORRECT":
        return "regressed_from_correct"
    if base.verdict == "UNKNOWN" and new.verdict == "WRONG":
        return "unknown_to_wrong"
    if base.verdict == "WRONG" and new.verdict == "UNKNOWN":
        return "wrong_to_unknown"
    return None


def _tags_for_change(base: TraceSummary, new: TraceSummary, meta: dict[str, Any], bucket: str) -> list[str]:
    tags: list[str] = []
    question_tags = _question_tags(meta.get("question", ""))

    if bucket == "improved_to_correct":
        if base.verdict == "UNKNOWN":
            tags.append("recovered_from_unknown")
        if not base.has_trace_end or base.trace_end_status == "N/A":
            tags.append("baseline_missing_trace_end")
        if new.segment_graph_calls > 0 and base.segment_graph_calls == 0:
            tags.append("new_segment_graph_retrieval")
        if new.frame_graph_calls > 0 and base.frame_graph_calls == 0:
            tags.append("new_frame_graph_retrieval")
        if new.graph_then_frame_inspect:
            tags.append("graph_then_visual_confirm")
        if new.frame_inspect_calls > base.frame_inspect_calls:
            tags.append("more_visual_confirmation")
        if new.graph_best_temporal_hit:
            tags.append("graph_temporal_hit")
        if "temporal_reasoning" in question_tags and new.segment_graph_calls > 0:
            tags.append("temporal_question_helped_by_segment_graph")
        if new.global_browse_calls <= base.global_browse_calls and new.graph_calls > base.graph_calls:
            tags.append("replaced_broad_search_with_graph")
    elif bucket == "regressed_from_correct":
        if new.verdict == "UNKNOWN":
            tags.append("new_runtime_failure")
        if new.graph_calls > base.graph_calls:
            tags.append("more_graph_dependence")
        if new.graph_fallback_used:
            tags.append("weak_graph_fallback")
        if new.graph_calls > 0 and new.frame_inspect_calls == 0:
            tags.append("graph_without_visual_confirmation")
        if new.frame_inspect_calls < base.frame_inspect_calls:
            tags.append("less_visual_confirmation")
        if new.llm_calls > base.llm_calls + 2:
            tags.append("longer_reasoning_but_wrong")
        if any(tag in question_tags for tag in ("counting", "color", "appearance")):
            tags.append("fine_visual_detail_question")
        if "temporal_reasoning" in question_tags and not new.graph_best_temporal_hit and new.segment_graph_calls > 0:
            tags.append("segment_graph_no_temporal_hit")
        if new.segment_graph_calls > 0 and any(tag in question_tags for tag in ("counting", "appearance")):
            tags.append("segment_graph_may_be_too_coarse")
    elif bucket == "unknown_to_wrong":
        tags.append("answered_after_baseline_unknown_but_still_wrong")
        if new.graph_calls > 0:
            tags.append("graph_involved")
    elif bucket == "wrong_to_unknown":
        tags.append("lost_answer")

    tags.extend(question_tags)
    deduped: list[str] = []
    seen = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def _tool_count_get(summary: TraceSummary, name: str) -> int:
    return int(summary.tool_counts.get(name, 0))


def _contrast_line(base: TraceSummary, new: TraceSummary) -> str:
    return (
        f"baseline={base.verdict}({base.trace_end_status}; {base.tool_path or 'no-tools'}) | "
        f"new={new.verdict}({new.trace_end_status}; {new.tool_path or 'no-tools'})"
    )


def _row_for_case(
    qid: str,
    meta: dict[str, Any],
    base: TraceSummary,
    new: TraceSummary,
    baseline_index_row: dict[str, str] | None,
    new_report_dir: Path,
    bucket: str,
) -> dict[str, Any]:
    baseline_report_path = ""
    if baseline_index_row:
        baseline_report_path = str(
            Path(base.path).parents[1] / "trace_reports" / f"{base.trace_id}.md"
        )

    new_report_path = str(new_report_dir / f"{new.trace_id}.md")
    tags = _tags_for_change(base=base, new=new, meta=meta, bucket=bucket)

    return {
        "qid": qid,
        "trace_id": meta.get("trace_id", ""),
        "video_id": meta.get("video_id", ""),
        "bucket": bucket,
        "question_type": "|".join(meta.get("question_type", [])),
        "time_reference": meta.get("time_reference", ""),
        "expected": meta.get("answer", "") or new.expected or base.expected,
        "baseline_verdict": base.verdict,
        "baseline_pred": base.predicted,
        "baseline_end_status": base.trace_end_status,
        "baseline_step_count": base.step_count,
        "baseline_llm_calls": base.llm_calls,
        "baseline_tool_path": base.tool_path,
        "baseline_behavior": (baseline_index_row or {}).get("behavior", ""),
        "baseline_gb": base.global_browse_calls,
        "baseline_cs": base.clip_search_calls,
        "baseline_fi": base.frame_inspect_calls,
        "baseline_fg": base.frame_graph_calls,
        "baseline_sg": base.segment_graph_calls,
        "new_verdict": new.verdict,
        "new_pred": new.predicted,
        "new_end_status": new.trace_end_status,
        "new_step_count": new.step_count,
        "new_llm_calls": new.llm_calls,
        "new_tool_path": new.tool_path,
        "new_gb": new.global_browse_calls,
        "new_cs": new.clip_search_calls,
        "new_fi": new.frame_inspect_calls,
        "new_fg": new.frame_graph_calls,
        "new_sg": new.segment_graph_calls,
        "new_graph_fallback_used": new.graph_fallback_used,
        "new_graph_best_temporal_hit": new.graph_best_temporal_hit,
        "new_graph_best_match_count": new.graph_best_match_count,
        "graph_then_frame_inspect": new.graph_then_frame_inspect,
        "tag_count": len(tags),
        "tags": "|".join(tags),
        "contrast": _contrast_line(base, new),
        "new_graph_evidence": " | ".join(new.graph_evidence_notes),
        "question": meta.get("question", "").replace("\n", " "),
        "baseline_trace_path": base.path,
        "new_trace_path": new.path,
        "baseline_report_path": baseline_report_path,
        "new_report_path": new_report_path,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _render_report(
    output_path: Path,
    baseline_summary: dict[str, Any],
    new_summary: dict[str, Any],
    change_counts: dict[str, int],
    improved_tag_counts: Counter[str],
    regressed_tag_counts: Counter[str],
    improved_rows: list[dict[str, Any]],
    regressed_rows: list[dict[str, Any]],
    notes: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Badcase Run Comparison")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | New | Delta |")
    lines.append("|---|---:|---:|---:|")

    metric_keys = [
        "completion_rate",
        "answered_rate",
        "answered_acc",
        "e2e_acc",
        "correct",
        "wrong",
        "unknown",
    ]
    for key in metric_keys:
        b = baseline_summary["metrics"].get(key)
        n = new_summary["metrics"].get(key)
        if isinstance(b, float):
            lines.append(f"| `{key}` | {b:.4f} | {n:.4f} | {n - b:+.4f} |")
        else:
            lines.append(f"| `{key}` | {b} | {n} | {n - b:+d} |")

    lines.append("")
    lines.append("## Change Counts")
    lines.append("")
    for key in sorted(change_counts):
        lines.append(f"- `{key}`: {change_counts[key]}")
    lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Heuristic Buckets")
    lines.append("")
    lines.append("### Improved To Correct")
    lines.append("")
    for tag, count in improved_tag_counts.most_common(20):
        lines.append(f"- `{tag}`: {count}")
    lines.append("")
    lines.append("### Regressed From Correct")
    lines.append("")
    for tag, count in regressed_tag_counts.most_common(20):
        lines.append(f"- `{tag}`: {count}")
    lines.append("")

    def add_case_table(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| qid | video | baseline | new | tags | contrast |")
        lines.append("|---|---|---|---|---|---|")
        for row in rows[:40]:
            tags = row["tags"].replace("|", ", ")
            contrast = row["contrast"].replace("|", " / ")
            lines.append(
                f"| {row['qid']} | {row['video_id']} | {row['baseline_verdict']}:{row['baseline_pred']} | "
                f"{row['new_verdict']}:{row['new_pred']} | {tags} | {contrast} |"
            )
        lines.append("")

    add_case_table("Improved Samples", improved_rows)
    add_case_table("Regressed Samples", regressed_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two badcase benchmark runs.")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--baseline-trace-index-csv", default="")
    parser.add_argument("--new-run-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline_run_dir = Path(args.baseline_run_dir).expanduser().resolve()
    new_run_dir = Path(args.new_run_dir).expanduser().resolve()
    meta_path = Path(args.meta).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    baseline_summary_path = baseline_run_dir / "run_summary.generated.json"
    new_summary_path = new_run_dir / "run_summary.json"
    baseline_trace_dir = baseline_run_dir / "traces"
    new_trace_dir = new_run_dir / "traces"
    baseline_index_csv = (
        Path(args.baseline_trace_index_csv).expanduser().resolve()
        if args.baseline_trace_index_csv
        else baseline_trace_dir / "index.csv"
    )
    new_benchmark_results = new_run_dir / "benchmark_results.json"
    new_trace_report_dir = output_dir / "new_trace_reports"

    meta_by_qid = _load_meta(meta_path)
    baseline_trace_index = _load_baseline_trace_index(baseline_index_csv)
    baseline_traces = _load_trace_summaries(baseline_trace_dir)
    new_traces = _load_trace_summaries(new_trace_dir)
    new_results = _load_new_benchmark_results(new_benchmark_results)
    baseline_summary = _safe_load_json(baseline_summary_path)
    new_summary = _safe_load_json(new_summary_path)

    improved_rows: list[dict[str, Any]] = []
    regressed_rows: list[dict[str, Any]] = []
    other_changed_rows: list[dict[str, Any]] = []
    all_changed_rows: list[dict[str, Any]] = []
    improved_tag_counts: Counter[str] = Counter()
    regressed_tag_counts: Counter[str] = Counter()
    change_counts: Counter[str] = Counter()

    qids = sorted(set(meta_by_qid) & set(baseline_traces) & set(new_traces), key=lambda x: int(x))
    for qid in qids:
        meta = meta_by_qid[qid]
        base = baseline_traces[qid]
        new = new_traces[qid]

        # Prefer benchmark_results verdict for new run if present.
        result_row = new_results.get(qid, {})
        if result_row:
            new.verdict = _verdict_from_is_correct(result_row.get("is_correct"))
            new.predicted = _normalize_pred(result_row.get("answer"))
            new.expected = _normalize_pred(result_row.get("expected_answer") or new.expected or meta.get("answer"))

        bucket = _change_bucket(base, new)
        if bucket is None:
            continue

        row = _row_for_case(
            qid=qid,
            meta=meta,
            base=base,
            new=new,
            baseline_index_row=baseline_trace_index.get(qid),
            new_report_dir=new_trace_report_dir,
            bucket=bucket,
        )
        all_changed_rows.append(row)
        change_counts[bucket] += 1

        tags = row["tags"].split("|") if row["tags"] else []
        if bucket == "improved_to_correct":
            improved_rows.append(row)
            improved_tag_counts.update(tags)
        elif bucket == "regressed_from_correct":
            regressed_rows.append(row)
            regressed_tag_counts.update(tags)
        else:
            other_changed_rows.append(row)

    improved_rows.sort(key=lambda x: (x["video_id"], int(x["qid"])))
    regressed_rows.sort(key=lambda x: (x["video_id"], int(x["qid"])))
    other_changed_rows.sort(key=lambda x: (x["video_id"], int(x["qid"])))
    all_changed_rows.sort(key=lambda x: (x["video_id"], int(x["qid"])))

    correct_to_wrong_rows = [row for row in regressed_rows if row["new_verdict"] == "WRONG"]
    correct_to_unknown_rows = [row for row in regressed_rows if row["new_verdict"] == "UNKNOWN"]

    notes = [
        (
            "Strict changed-to-correct count is "
            f"{len(improved_rows)}. The net `correct` gain is "
            f"{new_summary['metrics']['correct'] - baseline_summary['metrics']['correct']} "
            "because some baseline-correct questions also regressed."
        ),
        (
            "Strict regressed-from-correct count is "
            f"{len(regressed_rows)}. This excludes `UNKNOWN -> WRONG` style changes, "
            f"which are exported separately ({len(other_changed_rows)} rows across other buckets)."
        ),
    ]

    summary = {
        "baseline_run_dir": str(baseline_run_dir),
        "new_run_dir": str(new_run_dir),
        "output_dir": str(output_dir),
        "baseline_metrics": baseline_summary["metrics"],
        "new_metrics": new_summary["metrics"],
        "delta": {
            "completion_rate": new_summary["metrics"]["completion_rate"] - baseline_summary["metrics"]["completion_rate"],
            "answered_rate": new_summary["metrics"]["answered_rate"] - baseline_summary["metrics"]["answered_rate"],
            "answered_acc": new_summary["metrics"]["answered_acc"] - baseline_summary["metrics"]["answered_acc"],
            "e2e_acc": new_summary["metrics"]["e2e_acc"] - baseline_summary["metrics"]["e2e_acc"],
            "correct": new_summary["metrics"]["correct"] - baseline_summary["metrics"]["correct"],
            "wrong": new_summary["metrics"]["wrong"] - baseline_summary["metrics"]["wrong"],
            "unknown": new_summary["metrics"]["unknown"] - baseline_summary["metrics"]["unknown"],
        },
        "change_counts": dict(sorted(change_counts.items())),
        "improved_to_correct_count": len(improved_rows),
        "regressed_from_correct_count": len(regressed_rows),
        "other_changed_count": len(other_changed_rows),
        "improved_tag_counts": dict(improved_tag_counts.most_common()),
        "regressed_tag_counts": dict(regressed_tag_counts.most_common()),
        "notes": notes,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "all_changes.csv", all_changed_rows)
    _write_csv(output_dir / "improved_to_correct.csv", improved_rows)
    _write_csv(output_dir / "regressed_from_correct.csv", regressed_rows)
    _write_csv(output_dir / "correct_to_wrong.csv", correct_to_wrong_rows)
    _write_csv(output_dir / "correct_to_unknown.csv", correct_to_unknown_rows)
    _write_csv(output_dir / "other_changed.csv", other_changed_rows)
    _render_report(
        output_path=output_dir / "comparison_report.md",
        baseline_summary=baseline_summary,
        new_summary=new_summary,
        change_counts=dict(change_counts),
        improved_tag_counts=improved_tag_counts,
        regressed_tag_counts=regressed_tag_counts,
        improved_rows=improved_rows,
        regressed_rows=regressed_rows,
        notes=notes,
    )

    changed_trace_ids = [row["trace_id"] for row in all_changed_rows]
    (output_dir / "changed_trace_ids.txt").write_text("\n".join(changed_trace_ids) + ("\n" if changed_trace_ids else ""), encoding="utf-8")

    print(f"[OK] Wrote comparison artifacts to: {output_dir}")
    print(
        json.dumps(
            {
                "improved_to_correct": len(improved_rows),
                "regressed_from_correct": len(regressed_rows),
                "other_changed": len(other_changed_rows),
                "net_correct_delta": summary["delta"]["correct"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
