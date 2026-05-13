#!/usr/bin/env python3
"""
Render DVD trace JSONL files into step-by-step Markdown reports.

Examples:
  python3 scripts/render_trace_report.py \
    --trace-dir logs/test_20260210_191641/traces

  python3 scripts/render_trace_report.py \
    --trace-dir logs/test_20260210_191641/traces \
    --output-dir logs/test_20260210_191641/trace_reports \
    --trace-glob "Aiem1w_TvaA__*.jsonl"

  python3 scripts/render_trace_report.py \
    --trace-file logs/test_20260210_191641/traces/Aiem1w_TvaA__4144.jsonl
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def json_dumps_limited(value, max_chars: int) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return truncate_text(rendered, max_chars)


def read_jsonl(path: Path):
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                events.append(
                    {
                        "event_type": "parse_error",
                        "line_no": line_no,
                        "error": str(exc),
                        "raw_line": truncate_text(line, 400),
                    }
                )
    return events


def unwrap_maybe_truncated(value):
    if isinstance(value, dict):
        if "value" in value and "truncated" in value and len(value.keys()) <= 3:
            text = str(value.get("value", ""))
            if value.get("truncated"):
                original_chars = value.get("original_chars", "unknown")
                text += f"\n...[truncated in trace logger, original_chars={original_chars}]"
            return text
        return {k: unwrap_maybe_truncated(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unwrap_maybe_truncated(v) for v in value]
    return value


def extract_token_summary(tokens):
    if not isinstance(tokens, dict):
        return "N/A"
    prompt_tokens = tokens.get("prompt_tokens", "N/A")
    completion_tokens = tokens.get("completion_tokens", "N/A")
    total_tokens = tokens.get("total_tokens", "N/A")
    return f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"


def extract_answer_check(events):
    for event in reversed(events):
        if event.get("event_type") == "answer_check":
            return event
    return {}


def answer_label(is_correct):
    if is_correct is True:
        return "CORRECT"
    if is_correct is False:
        return "WRONG"
    return "UNKNOWN"


def render_message_brief(msg, idx, max_chars):
    role = msg.get("role", "unknown")
    lines = [f"{idx}. role: `{role}`"]

    if msg.get("name"):
        lines.append(f"   name: `{msg.get('name')}`")
    if msg.get("tool_call_id"):
        lines.append(f"   tool_call_id: `{msg.get('tool_call_id')}`")

    content = unwrap_maybe_truncated(msg.get("content"))
    if isinstance(content, str) and content.strip():
        lines.append("   content:")
        lines.append("```text")
        lines.append(truncate_text(content, max_chars))
        lines.append("```")

    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        lines.append(f"   tool_calls: {len(tool_calls)}")
        for tc in tool_calls:
            tc_id = tc.get("id", "unknown")
            func = tc.get("function", {})
            lines.append(f"   - `{tc_id}` -> `{func.get('name', 'unknown')}`")

    return "\n".join(lines)


def render_trace_markdown(trace_file: Path, events, max_chars: int):
    trace_start = next((e for e in events if e.get("event_type") == "trace_start"), {})
    trace_end = next((e for e in reversed(events) if e.get("event_type") == "trace_end"), {})
    answer_check = extract_answer_check(events)
    trace_id = trace_start.get("trace_id") or (events[0].get("trace_id") if events else trace_file.stem)

    step_map = defaultdict(lambda: {"llm_call": None, "tool_calls": []})
    parse_errors = []
    for e in events:
        event_type = e.get("event_type")
        if event_type == "llm_call":
            step_map[e.get("step_idx")]["llm_call"] = e
        elif event_type == "tool_call":
            step_map[e.get("step_idx")]["tool_calls"].append(e)
        elif event_type == "parse_error":
            parse_errors.append(e)

    lines = []
    lines.append(f"# Trace Report: `{trace_id}`")
    lines.append("")
    lines.append(f"- Source JSONL: `{trace_file}`")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    if trace_start:
        lines.append(f"- Max iterations: `{trace_start.get('max_iterations', 'N/A')}`")
        lines.append(f"- Question: `{trace_start.get('question', '').replace(chr(10), ' ')}`")
    if trace_end:
        lines.append(f"- End status: `{trace_end.get('status', 'N/A')}`")
    if answer_check:
        verdict = answer_label(answer_check.get("is_correct"))
        lines.append(f"- Answer check: `{verdict}`")
        lines.append(f"- Predicted: `{answer_check.get('predicted_answer_raw')}`")
        lines.append(f"- Expected: `{answer_check.get('expected_answer_raw')}`")
    else:
        lines.append("- Answer check: `UNKNOWN` (no `answer_check` event in trace)")
    lines.append(f"- Total events: `{len(events)}`")
    lines.append("")

    if parse_errors:
        lines.append("## Parse Errors")
        lines.append("")
        for err in parse_errors:
            lines.append(f"- line `{err.get('line_no')}`: `{err.get('error')}`")
        lines.append("")

    step_keys = sorted(k for k in step_map.keys() if isinstance(k, int))
    for step_idx in step_keys:
        step = step_map[step_idx]
        llm_event = step["llm_call"]
        tool_events = sorted(step["tool_calls"], key=lambda x: x.get("tool_call_idx", 0))

        lines.append(f"## Step {step_idx}")
        lines.append("")

        if llm_event:
            model = llm_event.get("model", "N/A")
            latency_ms = llm_event.get("latency_ms", "N/A")
            token_summary = extract_token_summary(llm_event.get("tokens"))
            endpoint = llm_event.get("endpoint", "N/A")
            lines.append("### LLM Call")
            lines.append(f"- Model: `{model}`")
            lines.append(f"- Latency: `{latency_ms} ms`")
            lines.append(f"- Tokens: `{token_summary}`")
            lines.append(f"- Endpoint: `{endpoint}`")
            lines.append("")

            llm_input = unwrap_maybe_truncated(llm_event.get("llm_input", {}))
            messages = llm_input.get("messages", []) if isinstance(llm_input, dict) else []
            lines.append("### LLM Input")
            if messages:
                for idx, msg in enumerate(messages, start=1):
                    lines.append(render_message_brief(msg, idx, max_chars))
                    lines.append("")
            else:
                lines.append("```json")
                lines.append(json_dumps_limited(llm_input, max_chars))
                lines.append("```")
                lines.append("")

            llm_output = unwrap_maybe_truncated(llm_event.get("llm_output", {}))
            lines.append("### LLM Output")
            if isinstance(llm_output, dict):
                content = llm_output.get("content")
                if isinstance(content, str) and content.strip():
                    lines.append("```text")
                    lines.append(truncate_text(content, max_chars))
                    lines.append("```")
                    lines.append("")
                tool_calls = llm_output.get("tool_calls") or []
                if tool_calls:
                    lines.append(f"- tool_calls: `{len(tool_calls)}`")
                    for tc in tool_calls:
                        tc_id = tc.get("id", "unknown")
                        func = tc.get("function", {})
                        func_name = func.get("name", "unknown")
                        func_args = func.get("arguments", "")
                        lines.append(f"- `{tc_id}` -> `{func_name}`")
                        if func_args:
                            lines.append("```json")
                            lines.append(truncate_text(func_args, max_chars))
                            lines.append("```")
                    lines.append("")

            lines.append("<details>")
            lines.append("<summary>Raw LLM Input JSON</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(json_dumps_limited(llm_input, max_chars))
            lines.append("```")
            lines.append("</details>")
            lines.append("")

            lines.append("<details>")
            lines.append("<summary>Raw LLM Output JSON</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(json_dumps_limited(llm_output, max_chars))
            lines.append("```")
            lines.append("</details>")
            lines.append("")

        if tool_events:
            lines.append("### Tool Calls")
            lines.append("")
            for t in tool_events:
                tool_no = t.get("tool_call_idx", "N/A")
                lines.append(f"#### Tool {step_idx}.{tool_no}: `{t.get('tool_name', 'unknown')}`")
                lines.append(f"- status: `{t.get('status', 'N/A')}`")
                lines.append(f"- tool_call_id: `{t.get('tool_call_id', 'N/A')}`")
                lines.append("")
                lines.append("Tool args:")
                lines.append("```json")
                lines.append(truncate_text(str(t.get("tool_args", "")), max_chars))
                lines.append("```")
                lines.append("")
                preview = t.get("tool_output_summary", {}).get("preview", "")
                output_chars = t.get("tool_output_summary", {}).get("output_chars", "N/A")
                truncated = t.get("tool_output_summary", {}).get("truncated", False)
                lines.append(f"Tool output summary: chars=`{output_chars}`, truncated=`{truncated}`")
                lines.append("```text")
                lines.append(truncate_text(str(preview), max_chars))
                lines.append("```")
                lines.append("")

    if answer_check:
        lines.append("## Answer Check")
        lines.append("")
        lines.append(f"- Verdict: `{answer_label(answer_check.get('is_correct'))}`")
        lines.append(f"- Predicted (raw): `{answer_check.get('predicted_answer_raw')}`")
        lines.append(f"- Predicted (normalized): `{answer_check.get('predicted_answer_normalized')}`")
        lines.append(f"- Expected (raw): `{answer_check.get('expected_answer_raw')}`")
        lines.append(f"- Expected (normalized): `{answer_check.get('expected_answer_normalized')}`")
        lines.append("")

    return "\n".join(lines), {
        "trace_id": trace_id,
        "end_status": trace_end.get("status", "N/A"),
        "answer_check": answer_check,
    }


def collect_trace_files(trace_dir, trace_glob, trace_files):
    collected = []
    if trace_dir:
        collected.extend(sorted(Path(trace_dir).glob(trace_glob)))
    for item in trace_files:
        collected.append(Path(item))

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for path in collected:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Render DVD trace JSONL into Markdown reports.")
    parser.add_argument("--trace-dir", default=None, help="Directory that contains trace JSONL files.")
    parser.add_argument("--trace-glob", default="*.jsonl", help="Glob pattern when --trace-dir is set.")
    parser.add_argument("--trace-file", action="append", default=[],
                        help="Single trace JSONL file path. Can be used multiple times.")
    parser.add_argument("--output-dir", default=None, help="Output directory for Markdown reports.")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max chars per rendered block.")
    args = parser.parse_args()

    if not args.trace_dir and not args.trace_file:
        raise ValueError("Provide at least one of --trace-dir or --trace-file.")
    if args.max_chars < 200:
        raise ValueError("--max-chars must be >= 200")

    trace_files = collect_trace_files(args.trace_dir, args.trace_glob, args.trace_file)
    if not trace_files:
        raise FileNotFoundError("No trace files found with current input options.")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.trace_dir).parent / "trace_reports" if args.trace_dir else Path(trace_files[0]).parent / "trace_reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    index_lines = ["# Trace Reports", ""]
    generated = 0
    stats = {"correct": 0, "wrong": 0, "unknown": 0}
    index_rows = []
    for trace_file in trace_files:
        if not trace_file.exists():
            print(f"[WARN] Missing trace file: {trace_file}")
            continue
        events = read_jsonl(trace_file)
        report_md, report_meta = render_trace_markdown(trace_file, events, max_chars=args.max_chars)
        out_file = output_dir / f"{trace_file.stem}.md"
        out_file.write_text(report_md, encoding="utf-8")
        generated += 1

        answer_event = report_meta.get("answer_check", {})
        verdict = answer_label(answer_event.get("is_correct"))
        if verdict == "CORRECT":
            stats["correct"] += 1
        elif verdict == "WRONG":
            stats["wrong"] += 1
        else:
            stats["unknown"] += 1

        predicted = answer_event.get("predicted_answer_raw", "N/A")
        expected = answer_event.get("expected_answer_raw", "N/A")
        index_rows.append(
            f"- `{trace_file.name}` -> `{out_file.name}` "
            f"(status: `{report_meta.get('end_status', 'N/A')}`, "
            f"answer: `{verdict}`, predicted: `{predicted}`, expected: `{expected}`)"
        )

    checked = stats["correct"] + stats["wrong"]
    accuracy_text = "N/A"
    if checked > 0:
        accuracy_text = f"{(stats['correct'] / checked) * 100:.2f}%"

    index_lines.append(f"- Total traces: `{generated}`")
    index_lines.append(
        f"- Answer checks: correct=`{stats['correct']}`, wrong=`{stats['wrong']}`, unknown=`{stats['unknown']}`"
    )
    index_lines.append(f"- Accuracy (excluding unknown): `{accuracy_text}`")
    index_lines.append("")
    index_lines.extend(index_rows)

    index_path = output_dir / "INDEX.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"[OK] Generated {generated} report(s) in: {output_dir}")
    print(f"[OK] Index: {index_path}")


if __name__ == "__main__":
    main()
