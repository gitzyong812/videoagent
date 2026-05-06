import json
import os
import re
from datetime import datetime, timezone


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sanitize_trace_id(trace_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(trace_id))


def _truncate_string(value: str, max_chars: int):
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars] + "...(truncated)", True


def _make_jsonable(value, max_chars: int):
    if isinstance(value, str):
        trimmed, truncated = _truncate_string(value, max_chars)
        if truncated:
            return {
                "value": trimmed,
                "truncated": True,
                "original_chars": len(value),
            }
        return value

    if isinstance(value, dict):
        return {str(k): _make_jsonable(v, max_chars) for k, v in value.items()}

    if isinstance(value, list):
        return [_make_jsonable(v, max_chars) for v in value]

    if isinstance(value, tuple):
        return [_make_jsonable(v, max_chars) for v in value]

    if value is None or isinstance(value, (int, float, bool)):
        return value

    fallback = str(value)
    trimmed, truncated = _truncate_string(fallback, max_chars)
    if truncated:
        return {
            "value": trimmed,
            "truncated": True,
            "original_chars": len(fallback),
        }
    return trimmed


class TraceLogger:
    def __init__(self, trace_id: str, log_dir: str, max_field_chars: int = 12000):
        self.trace_id = str(trace_id)
        self.max_field_chars = max_field_chars
        safe_trace_id = _sanitize_trace_id(self.trace_id)
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{safe_trace_id}.jsonl")
        self.event_idx = 0

    def _write_event(self, event_type: str, payload: dict):
        self.event_idx += 1
        event = {
            "event_idx": self.event_idx,
            "event_type": event_type,
            "trace_id": self.trace_id,
            "ts_utc": _utc_now_iso(),
        }
        event.update(_make_jsonable(payload, self.max_field_chars))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_trace_start(self, question: str, max_iterations: int):
        self._write_event(
            "trace_start",
            {
                "question": question,
                "max_iterations": max_iterations,
            },
        )

    def log_llm_call(
        self,
        step_idx: int,
        llm_input,
        llm_output,
        model: str,
        latency_ms: float,
        tokens,
        endpoint: str,
    ):
        self._write_event(
            "llm_call",
            {
                "step_idx": step_idx,
                "llm_input": llm_input,
                "llm_output": llm_output,
                "model": model,
                "latency_ms": latency_ms,
                "tokens": tokens,
                "endpoint": endpoint,
            },
        )

    def log_tool_call(
        self,
        step_idx: int,
        tool_name: str,
        tool_args,
        tool_output,
        tool_call_id: str = None,
        tool_call_idx: int = None,
        status: str = "ok",
    ):
        output_text = ""
        try:
            if isinstance(tool_output, str):
                output_text = tool_output
            else:
                output_text = json.dumps(tool_output, ensure_ascii=False, default=str)
        except Exception:
            output_text = str(tool_output)

        preview, truncated = _truncate_string(output_text, min(4000, self.max_field_chars))
        self._write_event(
            "tool_call",
            {
                "step_idx": step_idx,
                "tool_call_idx": tool_call_idx,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_output_summary": {
                    "preview": preview,
                    "truncated": truncated,
                    "output_chars": len(output_text),
                },
                "status": status,
            },
        )

    def log_trace_end(self, status: str):
        self._write_event("trace_end", {"status": status})
