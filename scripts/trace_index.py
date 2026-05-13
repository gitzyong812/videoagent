#!/usr/bin/env python3
"""
Build a compact CSV index for DVD traces and write bucket lists.

Supports both:
  - Raw traces: traces/*.jsonl
  - Rendered reports: trace_reports/*.md
"""

import argparse
import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Markdown Trace Report patterns
TRACE_ID_RE = re.compile(r"# Trace Report:\s*`([^`]+)`")
QUESTION_RE = re.compile(r"- Question:\s*`(.+?)`")
ANSWERCHK_RE = re.compile(r"- Answer check:\s*`([^`]+)`")
PRED_RE = re.compile(r"- Predicted:\s*`([^`]+)`")
EXP_RE = re.compile(r"- Expected:\s*`([^`]+)`")
EVENTS_RE = re.compile(r"- Total events:\s*`?(\d+)`?")
STEP_RE = re.compile(r"^## Step\s+(\d+)", re.M)
TOOL_RE = re.compile(r"#### Tool\s+\d+\.\d+:\s+`([^`]+)`")
VERDICT_RE = re.compile(r"- Verdict:\s*`([^`]+)`")
PRED2_RE = re.compile(r"- Predicted \(normalized\):\s*`([^`]+)`")
EXP2_RE = re.compile(r"- Expected \(normalized\):\s*`([^`]+)`")
STEP_HEADER_RE = re.compile(r"^## Step\s+(\d+)")
STEP_TOOL_RE = re.compile(r"^#### Tool\s+\d+\.\d+:\s+`([^`]+)`")


def _first_group(pattern: re.Pattern, text: str, default: str = "") -> str:
    match = pattern.search(text)
    return match.group(1) if match else default


def _normalize_verdict(value: Optional[str]) -> str:
    if value is None:
        return "UNKNOWN"
    upper = str(value).strip().upper()
    if upper in {"CORRECT", "WRONG", "UNKNOWN"}:
        return upper
    if upper in {"TRUE", "1"}:
        return "CORRECT"
    if upper in {"FALSE", "0"}:
        return "WRONG"
    return "UNKNOWN"


def _verdict_from_is_correct(is_correct) -> str:
    if is_correct is True:
        return "CORRECT"
    if is_correct is False:
        return "WRONG"
    return "UNKNOWN"


def _behavior_label(
    cs: int,
    fi: int,
    gb: int,
    max_cs_run: int,
    max_fi_run: int,
    cs_fi_transitions: int,
) -> str:
    # DVD-inspired heuristic buckets.
    if cs == 0 and fi == 0 and gb >= 1:
        return "GBO"  # Global Browse Only
    if max_fi_run >= 4:
        return "FIT"  # Frame Inspect Trap
    if max_cs_run >= 4:
        return "CST"  # Clip Search Trap

    action_calls = cs + fi
    if 1 <= action_calls <= 2:
        return "SA"  # Simple Action
    if cs >= 2 and fi >= 2 and cs_fi_transitions >= 2:
        return "IS"  # Iterative Search
    return "OTHER"


def _sequence_stats(tool_sequence: Sequence[str]) -> Tuple[int, int, int]:
    max_cs_run = 0
    max_fi_run = 0
    cs_fi_transitions = 0
    prev = None
    csfi_prev = None
    run_len = 0

    for tool in tool_sequence:
        if tool == prev:
            run_len += 1
        else:
            run_len = 1
            prev = tool
        if tool == "clip_search_tool":
            max_cs_run = max(max_cs_run, run_len)
        elif tool == "frame_inspect_tool":
            max_fi_run = max(max_fi_run, run_len)

        if tool in {"clip_search_tool", "frame_inspect_tool"}:
            if csfi_prev and csfi_prev != tool:
                cs_fi_transitions += 1
            csfi_prev = tool

    return max_cs_run, max_fi_run, cs_fi_transitions


def _parse_md_trace(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    trace_id = _first_group(TRACE_ID_RE, text, default=path.stem)
    question = _first_group(QUESTION_RE, text, default="")
    answer_check = _first_group(ANSWERCHK_RE, text, default="")
    verdict = _first_group(VERDICT_RE, text, default=answer_check)
    verdict = _normalize_verdict(verdict)

    pred = _first_group(PRED2_RE, text, default=_first_group(PRED_RE, text, default=""))
    exp = _first_group(EXP2_RE, text, default=_first_group(EXP_RE, text, default=""))

    total_events_raw = _first_group(EVENTS_RE, text, default="")
    total_events = int(total_events_raw) if total_events_raw.isdigit() else None

    step_vals = [int(x) for x in STEP_RE.findall(text)]
    step_count = max(step_vals) if step_vals else 0

    tool_counter: Counter = Counter()
    tool_sequence: List[str] = []
    finish_step_min: Optional[int] = None
    step_idx = 0

    for line in text.splitlines():
        step_match = STEP_HEADER_RE.match(line)
        if step_match:
            step_idx = int(step_match.group(1))
            continue

        tool_match = STEP_TOOL_RE.match(line)
        if not tool_match:
            continue

        tool_name = tool_match.group(1)
        tool_counter[tool_name] += 1
        tool_sequence.append(tool_name)
        if tool_name == "finish" and step_idx:
            finish_step_min = (
                step_idx
                if finish_step_min is None
                else min(finish_step_min, step_idx)
            )

    # Fallback to broad regex if line-level parse found nothing.
    if not tool_sequence:
        tools = TOOL_RE.findall(text)
        tool_counter = Counter(tools)
        tool_sequence = tools

    max_cs_run, max_fi_run, cs_fi_transitions = _sequence_stats(tool_sequence)
    cs = tool_counter.get("clip_search_tool", 0)
    fi = tool_counter.get("frame_inspect_tool", 0)
    gb = tool_counter.get("global_browse_tool", 0)

    behavior = _behavior_label(cs, fi, gb, max_cs_run, max_fi_run, cs_fi_transitions)
    unknown_or_none = verdict == "UNKNOWN" or pred in {"", "None", "null", "NULL"}

    return {
        "path": str(path),
        "file": path.name,
        "trace_id": trace_id,
        "verdict": verdict,
        "pred": pred,
        "exp": exp,
        "behavior": behavior,
        "step_count": step_count,
        "total_events": total_events,
        "gb": gb,
        "cs": cs,
        "fi": fi,
        "fg": tool_counter.get("frame_graph_query_tool", 0),
        "finish": tool_counter.get("finish", 0),
        "finish_step_min": finish_step_min,
        "max_cs_run": max_cs_run,
        "max_fi_run": max_fi_run,
        "unknown_or_none": unknown_or_none,
        "question": question[:2000],
    }


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
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


def _parse_jsonl_trace(path: Path) -> Dict[str, object]:
    events = _read_jsonl(path)
    trace_id = path.stem
    question = ""
    total_events = len(events)

    trace_start = None
    answer_check = None
    step_count = 0
    tool_rows: List[Tuple[int, int, int, str]] = []

    for e in events:
        event_type = e.get("event_type")
        if event_type == "trace_start" and trace_start is None:
            trace_start = e
        if event_type == "answer_check":
            answer_check = e

        step_idx = e.get("step_idx")
        if isinstance(step_idx, int):
            step_count = max(step_count, step_idx)

        if event_type == "tool_call":
            eidx = e.get("event_idx", 0)
            step_i = e.get("step_idx", 0) if isinstance(e.get("step_idx"), int) else 0
            tool_call_idx = (
                e.get("tool_call_idx", 0) if isinstance(e.get("tool_call_idx"), int) else 0
            )
            tool_name = str(e.get("tool_name", "unknown"))
            tool_rows.append((step_i, tool_call_idx, eidx if isinstance(eidx, int) else 0, tool_name))

    if trace_start:
        trace_id = str(trace_start.get("trace_id", trace_id))
        question = str(trace_start.get("question", ""))

    verdict = "UNKNOWN"
    pred = ""
    exp = ""
    if isinstance(answer_check, dict):
        verdict = _verdict_from_is_correct(answer_check.get("is_correct"))
        pred_norm = answer_check.get("predicted_answer_normalized")
        exp_norm = answer_check.get("expected_answer_normalized")
        pred_raw = answer_check.get("predicted_answer_raw")
        exp_raw = answer_check.get("expected_answer_raw")
        pred = "" if pred_norm is None else str(pred_norm)
        exp = "" if exp_norm is None else str(exp_norm)
        if not pred and pred_raw is not None:
            pred = str(pred_raw)
        if not exp and exp_raw is not None:
            exp = str(exp_raw)

    tool_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    tool_sequence = [t[3] for t in tool_rows]
    tool_counter = Counter(tool_sequence)

    finish_steps = [t[0] for t in tool_rows if t[3] == "finish" and t[0] > 0]
    finish_step_min = min(finish_steps) if finish_steps else None

    max_cs_run, max_fi_run, cs_fi_transitions = _sequence_stats(tool_sequence)
    cs = tool_counter.get("clip_search_tool", 0)
    fi = tool_counter.get("frame_inspect_tool", 0)
    gb = tool_counter.get("global_browse_tool", 0)
    behavior = _behavior_label(cs, fi, gb, max_cs_run, max_fi_run, cs_fi_transitions)
    unknown_or_none = verdict == "UNKNOWN" or pred in {"", "None", "null", "NULL"}

    return {
        "path": str(path),
        "file": path.name,
        "trace_id": trace_id,
        "verdict": verdict,
        "pred": pred,
        "exp": exp,
        "behavior": behavior,
        "step_count": step_count,
        "total_events": total_events,
        "gb": gb,
        "cs": cs,
        "fi": fi,
        "fg": tool_counter.get("frame_graph_query_tool", 0),
        "finish": tool_counter.get("finish", 0),
        "finish_step_min": finish_step_min,
        "max_cs_run": max_cs_run,
        "max_fi_run": max_fi_run,
        "unknown_or_none": unknown_or_none,
        "question": question[:2000],
    }


def _discover_files(input_path: Path, pattern: Optional[str] = None) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if pattern:
        return sorted(Path(p) for p in glob.glob(str(input_path / pattern)))

    jsonl_files = sorted(input_path.glob("*.jsonl"))
    md_files = sorted(
        p for p in input_path.glob("*.md") if p.name.lower() != "index.md"
    )
    if jsonl_files:
        return jsonl_files
    if md_files:
        return md_files
    return []


def _write_list(paths: Iterable[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"{p}\n")


def _print_summary(rows: Sequence[Dict[str, object]]) -> None:
    total = len(rows)
    verdict_counter = Counter(r["verdict"] for r in rows)
    behavior_counter = Counter(r["behavior"] for r in rows)
    wrong_behavior_counter = Counter(r["behavior"] for r in rows if r["verdict"] == "WRONG")

    print(f"rows: {total}")
    print("verdict:", dict(sorted(verdict_counter.items())))
    print("behavior_all:", dict(sorted(behavior_counter.items())))
    print("behavior_wrong:", dict(sorted(wrong_behavior_counter.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description="Index DVD traces into CSV and bucket files.")
    parser.add_argument("input_path", help="Trace dir or file (jsonl or md).")
    parser.add_argument("out_csv", help="Output CSV path.")
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help="Optional glob under input dir, e.g. '*.jsonl' or '*.md'.",
    )
    parser.add_argument(
        "--finish-early-step",
        type=int,
        default=2,
        help="finish called at step <= N is considered early (default: 2).",
    )
    parser.add_argument("--wrong-list", default=None, help="Output path for wrong list.")
    parser.add_argument("--broken-list", default=None, help="Output path for broken list.")
    parser.add_argument(
        "--finish-early-list", default=None, help="Output path for finish-early list."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    files = _discover_files(input_path, args.glob_pattern)
    if not files:
        raise SystemExit(f"No trace files found in {input_path}")

    rows: List[Dict[str, object]] = []
    for path in files:
        if path.suffix.lower() == ".md":
            row = _parse_md_trace(path)
        else:
            row = _parse_jsonl_trace(path)
        finish_step = row.get("finish_step_min")
        row["finish_early"] = (
            isinstance(finish_step, int)
            and finish_step <= args.finish_early_step
            and int(row.get("finish", 0)) > 0
        )
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "path",
        "file",
        "trace_id",
        "verdict",
        "pred",
        "exp",
        "behavior",
        "step_count",
        "total_events",
        "gb",
        "cs",
        "fi",
        "fg",
        "finish",
        "finish_step_min",
        "finish_early",
        "max_cs_run",
        "max_fi_run",
        "unknown_or_none",
        "question",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    out_prefix = out_csv.with_suffix("")
    wrong_list = (
        Path(args.wrong_list).expanduser().resolve()
        if args.wrong_list
        else out_prefix.parent / f"{out_prefix.name}.wrong.list"
    )
    broken_list = (
        Path(args.broken_list).expanduser().resolve()
        if args.broken_list
        else out_prefix.parent / f"{out_prefix.name}.broken.list"
    )
    finish_early_list = (
        Path(args.finish_early_list).expanduser().resolve()
        if args.finish_early_list
        else out_prefix.parent / f"{out_prefix.name}.finish_early.list"
    )

    wrong_paths = [str(r["path"]) for r in rows if r["verdict"] == "WRONG"]
    broken_paths = [str(r["path"]) for r in rows if bool(r["unknown_or_none"])]
    finish_early_paths = [str(r["path"]) for r in rows if bool(r["finish_early"])]

    _write_list(wrong_paths, wrong_list)
    _write_list(broken_paths, broken_list)
    _write_list(finish_early_paths, finish_early_list)

    print(f"wrote csv: {out_csv} ({len(rows)} rows)")
    print(f"wrote wrong list: {wrong_list} ({len(wrong_paths)})")
    print(f"wrote broken list: {broken_list} ({len(broken_paths)})")
    print(f"wrote finish-early list: {finish_early_list} ({len(finish_early_paths)})")
    _print_summary(rows)


if __name__ == "__main__":
    main()
