#!/usr/bin/env python3
"""
Build a reusable run summary JSON for DVD benchmark runs.

Works for both smoke and large-scale runs as long as traces are available.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("missing_database_arg", r"missing 1 required positional argument:\s*'database'"),
    ("openai_520", r"OpenAI API returned status 520"),
    ("openai_timeout", r"timed out|timeout"),
    ("content_filter", r"content\s*filter"),
    ("no_result_warn", r"\[WARN\]\s+No result for qid="),
    ("parallel_worker_error", r"Error processing question:"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _safe_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _verdict_from_is_correct(value: Any) -> str:
    if value is True:
        return "CORRECT"
    if value is False:
        return "WRONG"
    return "UNKNOWN"


def _to_float(n: int, d: int) -> float | None:
    if d <= 0:
        return None
    return n / d


def _parse_error_patterns(
    items: list[str],
) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --error-pattern '{item}'. Expected format: name=regex")
        name, regex = item.split("=", 1)
        name = name.strip()
        regex = regex.strip()
        if not name or not regex:
            raise ValueError(f"Invalid --error-pattern '{item}'. Name/regex cannot be empty")
        parsed.append((name, regex))
    return parsed


@dataclass
class TraceRecord:
    path: str
    file: str
    trace_id: str
    question: str
    has_trace_end: bool
    trace_end_status: str
    verdict: str
    predicted: str
    expected: str
    llm_calls: int
    tool_calls: int
    tool_counts: dict[str, int]
    parse_error_lines: int
    failure_bucket: str | None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    parse_errors = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
    return events, parse_errors


def _classify_failure_bucket(
    verdict: str,
    has_trace_end: bool,
    trace_end_status: str,
    predicted: str,
) -> str | None:
    if verdict in {"CORRECT", "WRONG"}:
        return None
    if not has_trace_end:
        return "missing_trace_end"
    if trace_end_status == "llm_call_returned_none":
        return "llm_call_returned_none"
    if trace_end_status == "max_iterations_reached":
        return "max_iterations_reached"
    if trace_end_status == "finished_by_finish_tool" and not str(predicted).strip():
        return "empty_answer_after_finish"
    return "unknown_other"


def _parse_trace(path: Path) -> TraceRecord:
    events, parse_error_lines = _read_jsonl(path)
    trace_id = path.stem
    question = ""
    has_trace_end = False
    trace_end_status = ""
    answer_check: dict[str, Any] | None = None
    llm_calls = 0
    tool_calls = 0
    tool_counter: Counter[str] = Counter()

    for e in events:
        event_type = e.get("event_type")
        if event_type == "trace_start":
            trace_id = str(e.get("trace_id", trace_id))
            question = str(e.get("question", ""))
        elif event_type == "trace_end":
            has_trace_end = True
            trace_end_status = str(e.get("status", ""))
        elif event_type == "answer_check":
            answer_check = e
        elif event_type == "llm_call":
            llm_calls += 1
        elif event_type == "tool_call":
            tool_calls += 1
            tool_counter[str(e.get("tool_name", "unknown"))] += 1

    verdict = "UNKNOWN"
    predicted = ""
    expected = ""
    if isinstance(answer_check, dict):
        verdict = _verdict_from_is_correct(answer_check.get("is_correct"))
        predicted = str(answer_check.get("predicted_answer_normalized") or answer_check.get("predicted_answer_raw") or "")
        expected = str(answer_check.get("expected_answer_normalized") or answer_check.get("expected_answer_raw") or "")

    failure_bucket = _classify_failure_bucket(
        verdict=verdict,
        has_trace_end=has_trace_end,
        trace_end_status=trace_end_status,
        predicted=predicted,
    )

    return TraceRecord(
        path=str(path),
        file=path.name,
        trace_id=trace_id,
        question=question,
        has_trace_end=has_trace_end,
        trace_end_status=trace_end_status,
        verdict=verdict,
        predicted=predicted,
        expected=expected,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        tool_counts=dict(sorted(tool_counter.items())),
        parse_error_lines=parse_error_lines,
        failure_bucket=failure_bucket,
    )


def _parse_trace_index_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    behavior = Counter()
    finish_early = 0
    rows = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            behavior[str(row.get("behavior", "UNKNOWN"))] += 1
            finish_early_raw = str(row.get("finish_early", "")).strip().lower()
            if finish_early_raw in {"true", "1", "yes"}:
                finish_early += 1
    return {
        "rows": rows,
        "behavior_counts": dict(sorted(behavior.items())),
        "finish_early_count": finish_early,
    }


def _parse_benchmark_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = _safe_load_json(path)
    except Exception:
        return {"error": "failed_to_parse_benchmark_results"}
    if not isinstance(data, dict):
        return {"error": "benchmark_results_not_dict"}

    total = 0
    correct = 0
    wrong = 0
    unknown = 0
    for _qid, row in data.items():
        total += 1
        is_correct = None
        if isinstance(row, dict):
            is_correct = row.get("is_correct")
        verdict = _verdict_from_is_correct(is_correct)
        if verdict == "CORRECT":
            correct += 1
        elif verdict == "WRONG":
            wrong += 1
        else:
            unknown += 1
    return {
        "total_questions": total,
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "answered": correct + wrong,
        "answered_acc": _to_float(correct, correct + wrong),
        "e2e_acc": _to_float(correct, total),
    }


def _parse_stage_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = _safe_load_json(path)
    except Exception:
        return {"error": "failed_to_parse_stage_report"}
    if not isinstance(data, dict):
        return {"error": "stage_report_not_dict"}
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return {"error": "stage_report_missing_summary"}
    return summary


def _discover_run_log(run_dir: Path, run_log: str) -> Path | None:
    if run_log:
        p = Path(run_log).expanduser().resolve()
        return p if p.exists() else None

    candidates = [
        run_dir / "run_stage12_pipeline.log",
        run_dir / "run_benchmark_all.log",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _discover_stage_report(run_dir: Path, stage_report: str, run_log_path: Path | None) -> Path | None:
    if stage_report:
        p = Path(stage_report).expanduser().resolve()
        return p if p.exists() else None

    if run_log_path and run_log_path.exists():
        text = _safe_read_text(run_log_path)
        matches = re.findall(r"Saved report to:\s*(.+?\.json)", text)
        for m in reversed(matches):
            p = Path(m.strip()).expanduser().resolve()
            if p.exists():
                return p
    # fallback: none
    return None


def _scan_log_errors(
    run_log_path: Path | None,
    patterns: list[tuple[str, str]],
) -> dict[str, int]:
    if run_log_path is None or not run_log_path.exists():
        return {}
    text = _safe_read_text(run_log_path)
    out: dict[str, int] = {}
    for name, regex in patterns:
        out[name] = len(re.findall(regex, text, flags=re.IGNORECASE))
    return out


def _metrics_from_traces(records: list[TraceRecord]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    total = len(records)
    completed = sum(1 for r in records if r.has_trace_end)
    correct = sum(1 for r in records if r.verdict == "CORRECT")
    wrong = sum(1 for r in records if r.verdict == "WRONG")
    unknown = sum(1 for r in records if r.verdict == "UNKNOWN")
    answered = correct + wrong

    end_status_counter = Counter(r.trace_end_status or "N/A" for r in records)
    failure_counter = Counter(r.failure_bucket for r in records if r.failure_bucket)

    total_llm_calls = sum(r.llm_calls for r in records)
    total_tool_calls = sum(r.tool_calls for r in records)
    calls_by_tool = Counter()
    traces_using_tool: defaultdict[str, int] = defaultdict(int)
    for r in records:
        for t, c in r.tool_counts.items():
            calls_by_tool[t] += c
            traces_using_tool[t] += 1

    unknown_samples: list[dict[str, Any]] = []
    for r in records:
        if r.verdict == "UNKNOWN":
            unknown_samples.append(
                {
                    "trace_id": r.trace_id,
                    "trace_end_status": r.trace_end_status or "N/A",
                    "failure_bucket": r.failure_bucket,
                    "predicted": r.predicted,
                    "expected": r.expected,
                    "path": r.path,
                }
            )

    metrics = {
        "total_traces": total,
        "completed_traces": completed,
        "completion_rate": _to_float(completed, total),
        "answered": answered,
        "answered_rate": _to_float(answered, total),
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "unknown_rate": _to_float(unknown, total),
        "answered_acc": _to_float(correct, answered),
        "e2e_acc": _to_float(correct, total),
    }
    llm_usage = {
        "total_llm_calls": total_llm_calls,
        "avg_llm_calls_per_trace": _to_float(total_llm_calls, total),
    }
    tool_usage = {
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls_per_trace": _to_float(total_tool_calls, total),
        "calls_by_tool": dict(sorted(calls_by_tool.items())),
        "traces_using_tool": dict(sorted(traces_using_tool.items())),
    }
    return (
        metrics,
        dict(sorted(end_status_counter.items())),
        dict(sorted(failure_counter.items())),
        {"llm_usage": llm_usage, "tool_usage": tool_usage},
        unknown_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build run_summary.json from DVD run artifacts.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory (e.g. .../logs/<run_name_timestamp>).",
    )
    parser.add_argument(
        "--trace-dir",
        default="",
        help="Trace dir override. Default: <run-dir>/traces",
    )
    parser.add_argument(
        "--trace-glob",
        default="*.jsonl",
        help="Trace glob pattern under trace dir. Default: *.jsonl",
    )
    parser.add_argument(
        "--trace-index-csv",
        default="",
        help="trace_index.csv path override. Default: <run-dir>/trace_index.csv",
    )
    parser.add_argument(
        "--benchmark-results",
        default="",
        help="benchmark_results.json path override. Default: <run-dir>/benchmark_results.json",
    )
    parser.add_argument(
        "--run-log",
        default="",
        help="Run log path override. Defaults: run_stage12_pipeline.log / run_benchmark_all.log",
    )
    parser.add_argument(
        "--stage-report",
        default="",
        help="Stage report JSON path override. If omitted, try to parse from run log.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output run_summary.json path. Default: <run-dir>/run_summary.json",
    )
    parser.add_argument(
        "--include-per-trace",
        action="store_true",
        help="Include per-trace rows in output (can be large on full benchmark).",
    )
    parser.add_argument(
        "--unknown-sample-limit",
        type=int,
        default=30,
        help="Max number of unknown samples to keep. Default: 30",
    )
    parser.add_argument(
        "--error-pattern",
        action="append",
        default=[],
        help="Extra log signature matcher: name=regex. Can be repeated.",
    )
    parser.add_argument(
        "--no-default-error-patterns",
        action="store_true",
        help="Disable built-in log signature patterns.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    trace_dir = Path(args.trace_dir).expanduser().resolve() if args.trace_dir else (run_dir / "traces")
    if not trace_dir.exists():
        raise SystemExit(f"Trace directory does not exist: {trace_dir}")

    trace_files = sorted(Path(p) for p in glob.glob(str(trace_dir / args.trace_glob)))
    if not trace_files:
        raise SystemExit(f"No trace files found in {trace_dir} with pattern {args.trace_glob}")

    trace_index_csv = Path(args.trace_index_csv).expanduser().resolve() if args.trace_index_csv else (run_dir / "trace_index.csv")
    benchmark_results = Path(args.benchmark_results).expanduser().resolve() if args.benchmark_results else (run_dir / "benchmark_results.json")

    run_log_path = _discover_run_log(run_dir=run_dir, run_log=args.run_log)
    stage_report_path = _discover_stage_report(
        run_dir=run_dir,
        stage_report=args.stage_report,
        run_log_path=run_log_path,
    )
    output_path = Path(args.output).expanduser().resolve() if args.output else (run_dir / "run_summary.json")

    records = [_parse_trace(path) for path in trace_files]
    metrics, end_status_counts, failure_bucket_counts, usage, unknown_samples = _metrics_from_traces(records)

    patterns: list[tuple[str, str]] = []
    if not args.no_default_error_patterns:
        patterns.extend(DEFAULT_ERROR_PATTERNS)
    patterns.extend(_parse_error_patterns(args.error_pattern))
    log_signature_counts = _scan_log_errors(run_log_path=run_log_path, patterns=patterns)

    trace_index_summary = _parse_trace_index_csv(trace_index_csv)
    benchmark_results_summary = _parse_benchmark_results(benchmark_results)
    stage_summary = _parse_stage_report(stage_report_path) if stage_report_path else {}

    notes: list[str] = []
    if benchmark_results_summary:
        br_total = benchmark_results_summary.get("total_questions")
        if isinstance(br_total, int) and br_total != metrics["total_traces"]:
            notes.append(
                f"benchmark_results total_questions ({br_total}) != trace total_traces ({metrics['total_traces']})"
            )

    out: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "inputs": {
            "run_dir": str(run_dir),
            "trace_dir": str(trace_dir),
            "trace_glob": args.trace_glob,
            "trace_count": len(trace_files),
            "trace_index_csv": str(trace_index_csv) if trace_index_csv.exists() else None,
            "benchmark_results": str(benchmark_results) if benchmark_results.exists() else None,
            "run_log": str(run_log_path) if run_log_path else None,
            "stage_report": str(stage_report_path) if stage_report_path else None,
        },
        "metrics": metrics,
        "trace_end_status_counts": end_status_counts,
        "failure_bucket_counts": failure_bucket_counts,
        "llm_usage": usage["llm_usage"],
        "tool_usage": usage["tool_usage"],
        "log_signature_counts": log_signature_counts,
        "trace_index_summary": trace_index_summary,
        "benchmark_results_summary": benchmark_results_summary,
        "stage_summary": stage_summary,
        "unknown_samples": unknown_samples[: max(0, args.unknown_sample_limit)],
        "notes": notes,
    }

    if args.include_per_trace:
        out["per_trace"] = [
            {
                "trace_id": r.trace_id,
                "file": r.file,
                "path": r.path,
                "question": r.question,
                "has_trace_end": r.has_trace_end,
                "trace_end_status": r.trace_end_status,
                "verdict": r.verdict,
                "predicted": r.predicted,
                "expected": r.expected,
                "failure_bucket": r.failure_bucket,
                "llm_calls": r.llm_calls,
                "tool_calls": r.tool_calls,
                "tool_counts": r.tool_counts,
                "parse_error_lines": r.parse_error_lines,
            }
            for r in records
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote run summary: {output_path}")
    print(
        json.dumps(
            {
                "total_traces": metrics["total_traces"],
                "completion_rate": metrics["completion_rate"],
                "answered_rate": metrics["answered_rate"],
                "answered_acc": metrics["answered_acc"],
                "e2e_acc": metrics["e2e_acc"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
