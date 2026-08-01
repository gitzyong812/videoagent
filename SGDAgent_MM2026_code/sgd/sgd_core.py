from __future__ import annotations

import copy
import inspect
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated as A
from typing import Any
from typing import Literal as L

import sgd.config as config
from sgd.build_database import (caption_backoff_tool, frame_graph_query_tool,
                                frame_inspect_tool, global_graph_query_tool,
                                init_single_video_db, segment_graph_query_tool)
from sgd.func_call_shema import as_json_schema
from sgd.func_call_shema import doc as D
from sgd.utils import call_openai_model_with_tools
from concurrent.futures import ThreadPoolExecutor, as_completed

TOPK = 16
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gold",
    "golden",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}
GENERIC_GRAPH_TERMS = {
    "adult",
    "audience",
    "boy",
    "camera",
    "chair",
    "child",
    "court",
    "crowd",
    "desk",
    "floor",
    "girl",
    "group",
    "man",
    "men",
    "people",
    "person",
    "player",
    "room",
    "screen",
    "stage",
    "subject",
    "team",
    "woman",
    "women",
    "worker",
}


@dataclass
class QuestionPolicy:
    question_types: tuple[str, ...]
    time_reference: str
    video_type: str
    requires_targeted_fi: bool
    requires_repeated_targeted_fi: bool
    requires_time_valid_terminal: bool
    targeted_label: str
    notes: list[str] = field(default_factory=list)


@dataclass
class PolicyState:
    best_graph_quality: float = float("-inf")
    best_clip_overlap: float = 0.0
    graph_backoff_active: bool = False
    graph_backoff_reason: str = ""
    graph_rejected_count: int = 0
    non_improving_streak: int = 0
    stagnation_prompted: bool = False
    recent_progress: list[bool] = field(default_factory=list)
    recent_tools: list[str] = field(default_factory=list)
    finish_rejections: int = 0
    fi_records: list[dict[str, Any]] = field(default_factory=list)
    clip_records: list[dict[str, Any]] = field(default_factory=list)
    last_graph_requires_followup: bool = False
    last_graph_recommended_action: str = ""

class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


def finish(answer: A[str, D("Answer to the user's question.")]) -> None:
    """Call this function after confirming the answer of the user's question, and finish the conversation."""
    raise StopException(answer)


class SGDCoreAgent:
    def __init__(self, video_db_path, video_caption_path, max_iterations, video_file_root: str | None = None):
        self.tools = [
            global_graph_query_tool,
            segment_graph_query_tool,
            frame_graph_query_tool,
            frame_inspect_tool,
            caption_backoff_tool,
            finish,
        ]
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self.video_db = init_single_video_db(
            video_caption_path,
            video_db_path,
            config.AOAI_EMBEDDING_LARGE_DIM,
            video_file_root_override=video_file_root,
        )
        self.max_iterations = max_iterations
        self.messages = self._construct_messages()

    def _tool_requires_database(self, func) -> bool:
        try:
            sig = inspect.signature(func)
            return "database" in sig.parameters
        except Exception:
            return False

    def _call_orchestrator_with_retries(self, msgs, step_idx, trace_logger=None, tool_choice="auto"):
        attempts = int(getattr(config, "ORCHESTRATOR_RETRY_ATTEMPTS", 3))
        delay = float(getattr(config, "ORCHESTRATOR_RETRY_BASE_DELAY_SEC", 1.0))
        last_none = False

        for attempt in range(1, max(1, attempts) + 1):
            response = call_openai_model_with_tools(
                msgs,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                tools=self.function_schemas,
                temperature=0.0,
                api_key=config.OPENAI_API_KEY,
                tool_choice=tool_choice,
                trace_logger=trace_logger,
                trace_context={
                    "step_idx": step_idx,
                    "attempt_idx": attempt,
                },
            )
            if response is not None:
                return response

            last_none = True
            if attempt < attempts:
                sleep_sec = delay * (2 ** (attempt - 1))
                print(
                    f"[WARN] Orchestrator call returned None at step {step_idx}, "
                    f"retrying attempt {attempt + 1}/{attempts} after {sleep_sec:.1f}s."
                )
                time.sleep(sleep_sec)

        if trace_logger is not None and last_none:
            trace_logger._write_event(
                "orchestrator_retries_exhausted",
                {
                    "step_idx": step_idx,
                    "attempts": attempts,
                },
            )
        return None

    def _construct_messages(self):
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the THINK → ACT → OBSERVE loop:
  • THOUGHT Reason step-by-step about which function to call next.
  • ACTION   Call exactly one function that moves you closer to the final answer.
  • OBSERVATION Summarize the function's output.
You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls.
Only pass arguments that come verbatim from the user or from earlier function outputs—never invent them. Continue the loop until the user's query is fully resolved, then call the `finish` function with the final answer. Do not answer directly in assistant text when you are ready to finish. If you are uncertain about code structure or video content, use the available tools to inspect rather than guessing. Plan carefully before each call and reflect on every result. Do not rely solely on blind function calls, as this degrades reasoning quality. Timestamps may be formatted as 'HH:MM:SS' or 'MM:SS'."""
            },
            {
                "role": "user",
                "content": """Carefully read the timestamps and narration in the following script, paying attention to the causal order of events, object details and movements, and people's actions and poses.

Here are tools you can use to reveal your reasoning process whenever the provided information is insufficient.

• For plot-level, causal, or video-spanning questions, start with `global_graph_query_tool`.
• To search caption wording when graph retrieval is weak, use `caption_backoff_tool`.
• To retrieve structured segment-level temporal evidence, use `segment_graph_query_tool`.
• To retrieve structured frame-level evidence (entities and relations), use `frame_graph_query_tool`.
• If the retrieved material lacks precise, question-relevant detail (e.g., an unknown name), call `frame_inspect_tool` with a list of time ranges (list[tuple[HH:MM:SS, HH:MM:SS]]).
• Whenever you are uncertain of an answer after searching, inspect frames in the relevant intervals with `frame_inspect_tool`.
• After locating an answer in the script, always make a **CONFIRM** with `frame_inspect_tool` query.
• If visual evidence is ambiguous or yields a non-option attribute, do NOT force the nearest answer option. Gather more evidence first.
• When a tool accepts `time_reference`, preserve the explicit time range exactly; do not collapse a range into a single timestamp.
• Your final step must be a call to `finish`; do not reply with the option letter directly in assistant text.

**Reading `controller_signals` from `segment_graph_query_tool` results — REQUIRED:**
Every result from `segment_graph_query_tool` includes a `controller_signals` field. You MUST check it before deciding what to do next:
• `localize_only: true` → The graph result tells you WHERE to look, not WHAT the answer is. Use the returned time range as a starting point only. You MUST call `frame_inspect_tool` to visually confirm the answer before calling `finish`. Do NOT call `finish` based on graph text alone.
• `question_is_fine_detail: true` → The question asks for something the graph cannot reliably answer (color, count, appearance, on-screen text, exact identity). Graph = location hint only. Always follow up with `frame_inspect_tool`.
• `coarse_window: true` → The matched segment spans ~30 seconds. This is too wide for precise visual verification. Do NOT limit your `frame_inspect_tool` call to only that segment's time range — inspect multiple sub-windows or a broader range.
• `no_temporal_hit: true` → The graph had no direct keyword match; results are speculative. Re-anchor with `caption_backoff_tool` before proceeding.
• `recommend_frame_inspect: true` → You must call `frame_inspect_tool` before calling `finish`.
• `weak_match: true` → Low-confidence match. Treat as a rough hint; verify independently.

Use the hierarchy progressively when appropriate: global graph → segment graph → frame graph → frame inspection.

Based on your observations and tool outputs, provide a concise answer that directly addresses the question. \n

Total video length: VIDEO_LENGTH seconds.

Question: QUESTION_PLACEHOLDER"""
            },
        ]
        add_data = self.video_db.get_additional_data()
        video_length = add_data['video_length']
        messages[-1]['content'] = messages[-1]['content'].replace("VIDEO_LENGTH", str(video_length))

        return messages

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        msgs.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": name,
                "content": content,
            }
        )

    def _build_trace_logger(self, question, trace_id=None):
        return None

    def _normalized_tool_calls(self, response) -> list[dict]:
        tool_calls = response.get("tool_calls")
        if isinstance(tool_calls, list):
            return tool_calls
        return []

    def _has_direct_answer(self, response) -> bool:
        content = response.get("content")
        return isinstance(content, str) and bool(content.strip()) and not self._normalized_tool_calls(response)

    def _has_tool_observation(self, msgs) -> bool:
        return any(msg.get("role") == "tool" for msg in msgs)

    def _is_single_choice_question(self, question: str) -> bool:
        if not question:
            return False
        return all(token in question for token in ("(A)", "(B)", "(C)", "(D)"))

    def _extract_strict_choice_letter(self, answer: str | None) -> str | None:
        if answer is None:
            return None
        value = str(answer).strip().strip("`'\" ")
        upper = value.upper()
        if upper in {"A", "B", "C", "D"}:
            return upper

        patterns = [
            r"^\(?([A-D])\)?[\.\):]?$",
            r"^(?:OPTION|ANSWER|CHOICE)\s*[:\-]?\s*\(?([A-D])\)?[\.\):]?$",
            r"(?:CORRECT\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-D])\)?",
            r"\(([A-D])\)",
            r"(?:^|[^A-Z])\(?([A-D])\)?\s*$",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, upper))
            if matches:
                return matches[-1].group(1)
        return None

    def _recover_single_choice_from_messages(self, msgs) -> str | None:
        for msg in reversed(msgs):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue

            for tool_call in reversed(self._normalized_tool_calls(msg)):
                if tool_call.get("function", {}).get("name") != "finish":
                    continue
                try:
                    args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                except Exception:
                    continue
                strict = self._extract_strict_choice_letter(args.get("answer"))
                if strict is not None:
                    return strict

            strict = self._extract_strict_choice_letter(msg.get("content"))
            if strict is not None:
                return strict
        return None

    def _append_guidance_once(self, msgs, content: str):
        for msg in msgs[-5:]:
            if msg.get("role") == "user" and msg.get("content") == content:
                return
        msgs.append({"role": "user", "content": content})

    def _normalize_question_context(self, question_context: dict[str, Any] | None) -> dict[str, Any]:
        context = question_context if isinstance(question_context, dict) else {}
        qtypes = context.get("question_type") or []
        if not isinstance(qtypes, list):
            qtypes = [qtypes]
        time_reference = self._canonical_time_reference(str(context.get("time_reference", "")).strip())
        return {
            "question_type": [str(item).strip() for item in qtypes if str(item).strip()],
            "time_reference": time_reference,
            "qid": str(context.get("qid", "")).strip(),
            "video_id": str(context.get("video_id", "")).strip(),
            "video_type": str(context.get("video_type", "")).strip(),
        }

    def _extract_time_marks_sec(self, text: str) -> list[float]:
        marks: list[float] = []
        for match in re.finditer(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)", text or ""):
            first = int(match.group(1))
            second = int(match.group(2))
            third = match.group(3)
            if third is None:
                marks.append(float(first * 60 + second))
            else:
                marks.append(float(first * 3600 + second * 60 + int(third)))
        return marks

    def _parse_time_reference_ranges(self, time_reference: str) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []
        for chunk in re.split(r"[;,/]", time_reference or ""):
            marks = self._extract_time_marks_sec(chunk)
            if not marks:
                continue
            if len(marks) == 1:
                ranges.append((marks[0], marks[0]))
            else:
                ranges.append((min(marks), max(marks)))
        return ranges

    def _format_time_reference_sec(self, total_sec: float) -> str:
        seconds = max(0, int(round(total_sec)))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:02}"

    def _canonical_time_reference(self, time_reference: str) -> str:
        raw = (time_reference or "").strip()
        if not raw:
            return ""
        ranges = self._parse_time_reference_ranges(raw)
        if not ranges:
            return raw
        normalized = []
        for start_sec, end_sec in ranges:
            start_text = self._format_time_reference_sec(start_sec)
            end_text = self._format_time_reference_sec(end_sec)
            if abs(end_sec - start_sec) < 0.5:
                normalized.append(start_text)
            else:
                normalized.append(f"{start_text}-{end_text}")
        return "; ".join(normalized)

    def _range_overlap_ratio(self, start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        left = max(start_a, start_b)
        right = min(end_a, end_b)
        if right <= left:
            return 0.0
        inter = right - left
        union = max(end_a, end_b) - min(start_a, start_b)
        return inter / union if union > 0 else 0.0

    def _span_overlaps_reference(
        self,
        start_sec: float,
        end_sec: float,
        target_ranges: list[tuple[float, float]],
    ) -> tuple[bool, float]:
        if not target_ranges:
            return True, 0.0
        best = 0.0
        for ref_start, ref_end in target_ranges:
            best = max(best, self._range_overlap_ratio(start_sec, end_sec, ref_start, ref_end))
            if best > 0.0:
                return True, best
            if ref_start <= start_sec <= ref_end or ref_start <= end_sec <= ref_end:
                return True, best
        return False, best

    def _question_token_flags(self, question: str) -> dict[str, bool]:
        q = (question or "").strip().lower()
        return {
            "count": bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q)),
            "color": bool(re.search(r"\bwhat color\b|\bcolour\b|\bcolor of\b", q)),
            "appearance": bool(
                re.search(r"\bwear\b|\bwearing\b|\bclothes\b|\bhat\b|\bshirt\b|\bdress\b|\bhair\b|\blook like\b", q)
            ),
            "identity": bool(re.search(r"^\s*who\b|\bname\b|\bwhich person\b|\bwho is on\b", q)),
            "text": bool(
                re.search(
                    r"\bword\b|\bletter\b|\bnumber\b|\bcaption\b|\btext\b|\bon the screen\b|\blicense plate\b|\bsubtitle\b",
                    q,
                )
            ),
            "temporal_event": bool(re.search(r"\bbefore\b|\bafter\b|\bwhen\b|\bwhat happens\b|\bnext\b|\bthen\b", q)),
        }

    def _initial_tool_name(self, question: str) -> str:
        q = (question or "").lower()
        flags = self._question_token_flags(question)
        if flags["count"] or flags["color"] or flags["text"] or flags["appearance"]:
            return "frame_inspect_tool"
        if re.search(r"\bwhy\b|\bcause\b|\bcausal\b|\bplot\b|\bstory\b|\boverall\b|\bthroughout\b", q):
            return "global_graph_query_tool"
        if re.search(r"\bwhen\b|\bbefore\b|\bafter\b|\bwhich part\b|\bwhat happens next\b", q):
            return "segment_graph_query_tool"
        if re.search(r"\bwhere\b|\bholding\b|\bwearing\b|\bleft of\b|\bright of\b|\bobject\b", q):
            return "frame_graph_query_tool"
        return "global_graph_query_tool"

    def _tool_choice_for_name(self, name: str) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name}}

    def _build_question_policy(self, question: str, question_context: dict[str, Any] | None) -> QuestionPolicy:
        context = self._normalize_question_context(question_context)
        qtypes = tuple(q.lower() for q in context["question_type"])
        flags = self._question_token_flags(question)
        fine_detail_like = bool(
            flags["count"]
            or flags["color"]
            or flags["appearance"]
            or flags["identity"]
            or flags["text"]
        )

        entity_like = any(
            qtype in {"entity recognition", "key information retrieval"}
            for qtype in qtypes
        )
        temporal_like = any(
            qtype in {"event understanding", "temporal grounding", "summarization"}
            for qtype in qtypes
        ) or flags["temporal_event"]

        requires_targeted_fi = bool(
            entity_like
            or fine_detail_like
        )
        requires_repeated_targeted_fi = bool(
            flags["count"]
            or (
                fine_detail_like
                and bool(getattr(config, "POLICY_REPEAT_TARGETED_FI_FOR_FINE_DETAIL", True))
            )
        )

        targeted_label = "frame-level detail"
        if flags["count"]:
            targeted_label = "count"
        elif flags["color"]:
            targeted_label = "color"
        elif flags["text"]:
            targeted_label = "on-screen text"
        elif flags["identity"]:
            targeted_label = "identity"
        elif flags["appearance"]:
            targeted_label = "appearance"
        elif temporal_like:
            targeted_label = "time-localized event"

        notes = [
            "Scene graph can localize candidates, but it is not decisive terminal evidence.",
        ]
        if requires_targeted_fi:
            notes.append(
                "Before `finish`, localize a coarse span, shrink to a narrower window, then use targeted `frame_inspect_tool` on the discriminative attribute."
            )
        if requires_repeated_targeted_fi:
            notes.append(
                "This question type needs two narrow targeted `frame_inspect_tool` confirmations that agree before `finish`."
            )
        if targeted_label == "count":
            notes.append(
                "For count questions, keep the counted group explicit and stable across repeated confirmations. Re-check the same localized moment instead of sweeping across adjacent instants that can change visibility."
            )
        if context["time_reference"]:
            notes.append(
                "Treat the explicit time reference as a hard admissibility constraint for graph hits and terminal evidence."
            )

        return QuestionPolicy(
            question_types=qtypes,
            time_reference=context["time_reference"],
            video_type=context["video_type"],
            requires_targeted_fi=requires_targeted_fi,
            requires_repeated_targeted_fi=requires_repeated_targeted_fi,
            requires_time_valid_terminal=bool(context["time_reference"]),
            targeted_label=targeted_label,
            notes=notes,
        )

    def _question_policy_prompt(self, policy: QuestionPolicy) -> str:
        lines = []
        if policy.question_types:
            lines.append(f"Question type hints: {', '.join(policy.question_types)}.")
        if policy.time_reference:
            lines.append(f"Explicit time reference: {policy.time_reference}.")
            lines.append("Pass the explicit time reference unchanged into graph-tool calls; do not shorten it to a single timestamp.")
        lines.extend(policy.notes)
        if policy.requires_targeted_fi:
            lines.append(
                f"HARD RULE: This question requires visual confirmation. "
                f"Use graph or clip search ONLY to localize the time range, "
                f"then call `frame_inspect_tool` with a targeted question about the {policy.targeted_label}. "
                f"Calling `finish` based solely on graph text is not allowed for this question type."
            )
        elif policy.requires_time_valid_terminal:
            lines.append(
                "Your terminal evidence must stay inside the explicit time reference, via `caption_backoff_tool` or `frame_inspect_tool`."
            )
        return "\n".join(f"- {line}" for line in lines if line)

    def _parse_tool_json(self, tool_output):
        if not isinstance(tool_output, str):
            return None
        try:
            parsed = json.loads(tool_output)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _policy_target_ranges(self, policy: QuestionPolicy) -> list[tuple[float, float]]:
        return self._parse_time_reference_ranges(policy.time_reference)

    def _targeted_window_threshold_sec(self, policy: QuestionPolicy) -> float:
        if not policy.time_reference:
            return float(getattr(config, "POLICY_TARGETED_FI_MAX_WINDOW_SEC", 18.0))
        spans = self._policy_target_ranges(policy)
        total = sum(max(0.0, end - start) for start, end in spans)
        if total <= 0:
            return float(getattr(config, "POLICY_TARGETED_FI_MAX_WINDOW_SEC", 18.0))
        max_window = float(getattr(config, "POLICY_TARGETED_FI_MAX_WINDOW_SEC", 18.0))
        min_window = float(getattr(config, "POLICY_TARGETED_FI_MIN_WINDOW_SEC", 6.0))
        scale = float(getattr(config, "POLICY_TARGETED_FI_WINDOW_MULTIPLIER", 0.6))
        return min(max_window, max(min_window, total * scale))

    def _extract_clip_ranges(self, tool_output: str) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []
        pattern = re.compile(
            r"\[From\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\s+to\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\s+seconds\]",
            re.IGNORECASE,
        )
        for match in pattern.finditer(tool_output or ""):
            starts = self._extract_time_marks_sec(match.group(1))
            ends = self._extract_time_marks_sec(match.group(2))
            if starts and ends:
                ranges.append((starts[0], ends[0]))
        return ranges

    def _query_matches_target(self, policy: QuestionPolicy, query: str) -> bool:
        q = (query or "").lower()
        label = policy.targeted_label
        if label == "count":
            return bool(re.search(r"\bcount\b|\bhow many\b|\bnumber of\b|\bdistinct\b", q))
        if label == "color":
            return bool(re.search(r"\bcolor\b|\bcolour\b", q))
        if label == "on-screen text":
            return bool(re.search(r"\bword\b|\bletter\b|\bnumber\b|\bcaption\b|\btext\b|\blicense plate\b", q))
        if label == "identity":
            return bool(re.search(r"\bwho\b|\bname\b|\bwhich person\b", q))
        if label == "appearance":
            return bool(re.search(r"\bwear\b|\bwearing\b|\blook\b|\bhair\b|\bclothes\b", q))
        if label == "time-localized event":
            return bool(re.search(r"\bbefore\b|\bafter\b|\bwhen\b|\bwhat happens\b|\bnext\b", q))
        if label == "frame-level detail":
            return bool(
                re.search(
                    r"\bdescribe\b|\baction\b|\bactions\b|\bdoing\b|\bhappening\b|\binteraction\b|\bvisible\b|\bshown\b|\bcaptured\b|\bwhat\b|\bwhich\b",
                    q,
                )
            ) or bool(q.strip())
        return bool(q.strip())

    def _normalize_output_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _extract_fi_signature(self, policy: QuestionPolicy, output: str) -> str | None:
        text = self._normalize_output_text(output)
        if not text:
            return None
        if policy.targeted_label == "count":
            digit = re.search(r"\b(\d{1,3})\b", text)
            if digit:
                return digit.group(1)
            for word, value in NUMBER_WORDS.items():
                if re.search(rf"\b{word}\b", text):
                    return value
        if policy.targeted_label in {"color", "appearance"}:
            for color in sorted(COLOR_WORDS):
                if re.search(rf"\b{re.escape(color)}\b", text):
                    return color
        if policy.targeted_label == "on-screen text":
            quoted = re.search(r"'([^']+)'|\"([^\"]+)\"|`([^`]+)`", output or "")
            if quoted:
                return next(group for group in quoted.groups() if group)
            token = re.search(r"\b([A-Z0-9]{1,8})\b", output or "")
            if token:
                return token.group(1)
        if policy.targeted_label == "identity":
            phrase = text
            match = re.search(r"\b(?:is|are|appears to be|appear to be|looks like|look like)\s+([a-z0-9][a-z0-9\s'/-]{0,80})", text)
            if match:
                phrase = match.group(1)
            phrase = re.split(r"[.;,]| because | while | when ", phrase, maxsplit=1)[0]
            phrase = re.sub(r"^(the|a|an)\s+", "", phrase).strip()
            tokens = [
                tok
                for tok in re.findall(r"[a-z0-9]+", phrase)
                if tok not in {"person", "people", "individual", "individuals"}
            ]
            if tokens:
                return " ".join(tokens[:5])
        return None

    def _text_token_similarity(self, left: str, right: str) -> float:
        left_tokens = {tok for tok in re.findall(r"[a-z0-9]+", left.lower()) if len(tok) > 1}
        right_tokens = {tok for tok in re.findall(r"[a-z0-9]+", right.lower()) if len(tok) > 1}
        if not left_tokens or not right_tokens:
            return 0.0
        inter = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return inter / union if union > 0 else 0.0

    def _fi_records_consistent(self, policy: QuestionPolicy, first: dict[str, Any], second: dict[str, Any]) -> bool:
        sig_a = first.get("signature")
        sig_b = second.get("signature")
        if sig_a and sig_b:
            return sig_a == sig_b
        similarity = self._text_token_similarity(first.get("output", ""), second.get("output", ""))
        threshold = float(getattr(config, "POLICY_FI_OUTPUT_SIMILARITY_THRESHOLD", 0.55))
        return similarity >= threshold

    def _ranges_duration(self, ranges: list[tuple[float, float]]) -> float:
        return sum(max(0.0, end - start) for start, end in ranges)

    def _fi_records_overlap_ratio(self, first: dict[str, Any], second: dict[str, Any]) -> float:
        first_ranges = first.get("ranges") or []
        second_ranges = second.get("ranges") or []
        if not first_ranges or not second_ranges:
            return 0.0

        overlap = 0.0
        for start_a, end_a in first_ranges:
            for start_b, end_b in second_ranges:
                overlap += max(0.0, min(end_a, end_b) - max(start_a, start_b))

        min_duration = min(self._ranges_duration(first_ranges), self._ranges_duration(second_ranges))
        if min_duration <= 0.0:
            return 0.0
        return overlap / min_duration

    def _recent_repeated_fi_status(
        self,
        policy: QuestionPolicy,
        qualifying_fi: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        if len(qualifying_fi) < 2:
            return (
                False,
                f"Before `finish`, run a second narrow targeted `frame_inspect_tool` confirmation for the {policy.targeted_label}.",
            )

        latest = qualifying_fi[-1]
        previous = qualifying_fi[-2]
        min_overlap = float(getattr(config, "POLICY_REPEATED_FI_MIN_OVERLAP_RATIO", 0.5))
        overlap_ratio = self._fi_records_overlap_ratio(previous, latest)
        if overlap_ratio < min_overlap:
            return (
                False,
                f"The latest narrow targeted `frame_inspect_tool` confirmations for the {policy.targeted_label} inspect different instants. Re-check the same localized window before `finish`.",
            )
        if not self._fi_records_consistent(policy, previous, latest):
            return (
                False,
                f"The latest narrow targeted `frame_inspect_tool` confirmations disagree on the {policy.targeted_label}; shrink the time window and confirm again.",
            )
        return True, ""

    def _qualifying_fi_records(self, policy: QuestionPolicy, state: PolicyState) -> list[dict[str, Any]]:
        records = []
        for record in state.fi_records:
            if policy.requires_time_valid_terminal and not record.get("time_valid"):
                continue
            if policy.requires_targeted_fi and not (record.get("targeted") and record.get("narrow")):
                continue
            records.append(record)
        return records

    def _finish_policy_status(
        self,
        question: str,
        policy: QuestionPolicy | None,
        state: PolicyState | None,
    ) -> tuple[bool, str]:
        if policy is None or state is None:
            return True, ""

        max_rej = int(getattr(config, "POLICY_MAX_FINISH_REJECTIONS", 3))
        if state.finish_rejections >= max_rej:
            return True, ""

        if state.last_graph_requires_followup:
            next_action = state.last_graph_recommended_action or "caption_backoff_tool or frame_inspect_tool"
            return (
                False,
                f"The latest graph result is localization-only or weak. Call `{next_action}` before `finish`.",
            )

        time_valid_clip = any(record.get("time_valid") for record in state.clip_records)
        qualifying_fi = self._qualifying_fi_records(policy, state)
        if policy.requires_targeted_fi:
            if not qualifying_fi:
                return (
                    False,
                    f"Before `finish`, gather a narrow targeted `frame_inspect_tool` confirmation for the {policy.targeted_label}.",
                )
            if policy.requires_repeated_targeted_fi and len(qualifying_fi) < 2:
                return (
                    False,
                    f"Before `finish`, run a second narrow targeted `frame_inspect_tool` confirmation for the {policy.targeted_label}.",
                )
            if policy.requires_repeated_targeted_fi:
                return self._recent_repeated_fi_status(policy, qualifying_fi)

        if policy.requires_time_valid_terminal and not (qualifying_fi or time_valid_clip):
            return (
                False,
                "The final answer still lacks time-valid terminal evidence inside the explicit time reference. Use `caption_backoff_tool` or `frame_inspect_tool` inside that span before `finish`.",
            )
        return True, ""

    def _entity_is_generic(self, entity: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(entity.get("canonical_name", "")),
                str(entity.get("type", "")),
            ]
        ).lower()
        tokens = {tok for tok in re.findall(r"[a-z0-9_]+", text) if tok}
        return not tokens or tokens <= GENERIC_GRAPH_TERMS

    def _graph_result_analysis(
        self,
        tool_name: str,
        tool_output,
        policy: QuestionPolicy | None,
    ) -> dict[str, Any]:
        parsed = self._parse_tool_json(tool_output)
        if not parsed:
            return {
                "reject": False,
                "quality": float("-inf"),
                "reason": "",
                "reasons": [],
                "top_score": 0.0,
                "top_lexical": 0.0,
            }
        if parsed.get("error"):
            return {
                "reject": True,
                "quality": float("-inf"),
                "reason": str(parsed.get("error")),
                "reasons": ["graph_error"],
                "top_score": 0.0,
                "top_lexical": 0.0,
            }

        if tool_name == "global_graph_query_tool":
            matched = parsed.get("matched_regions", []) or []
        elif tool_name == "segment_graph_query_tool":
            matched = parsed.get("matched_segments", []) or []
        elif tool_name == "frame_graph_query_tool":
            matched = parsed.get("matched_frames", []) or []
        else:
            matched = []

        if not matched:
            return {
                "reject": True,
                "quality": float("-inf"),
                "reason": "graph returned no matches",
                "reasons": ["no_match"],
                "top_score": 0.0,
                "top_lexical": 0.0,
            }

        top = matched[0]
        top_score = float(top.get("score", 0.0) or 0.0)
        top_lexical = float(top.get("lexical_score", 0.0) or 0.0)
        top_relations = len(top.get("matched_relations", []) or [])
        top_entities = top.get("matched_entities", []) or []
        generic_entities = bool(top_entities) and all(self._entity_is_generic(entity) for entity in top_entities)
        fallback_used = bool(parsed.get("fallback_used"))
        time_valid = True
        overlap = 0.0
        reasons: list[str] = []

        if policy is not None and policy.time_reference:
            target_ranges = self._policy_target_ranges(policy)
            if tool_name in {"global_graph_query_tool", "segment_graph_query_tool"}:
                seg_start = float(top.get("t_start", 0.0) or 0.0)
                seg_end = float(top.get("t_end", seg_start) or seg_start)
                time_valid, overlap = self._span_overlaps_reference(seg_start, seg_end, target_ranges)
            else:
                point = float(top.get("t_sec", 0.0) or 0.0)
                time_valid, overlap = self._span_overlaps_reference(point, point, target_ranges)
            if not time_valid:
                reasons.append("target_span_miss")

        if fallback_used:
            reasons.append("fallback_only")
        if top_lexical < float(getattr(config, "POLICY_GRAPH_MIN_LEXICAL_SCORE", 0.15)):
            reasons.append("lexical_low")
        if generic_entities and top_relations == 0:
            reasons.append("generic_entity_only")
        if len(matched) >= 2:
            second_score = float(matched[1].get("score", 0.0) or 0.0)
            margin = top_score - second_score
            if margin < float(getattr(config, "POLICY_GRAPH_MIN_TOP_MARGIN", 0.12)):
                reasons.append("top_margin_low")

        quality = (
            top_score
            + top_lexical * 2.0
            + float(len(top_entities))
            + float(top_relations) * 1.5
            + (1.5 if time_valid else overlap)
        )
        reject = bool(reasons)
        reason_text = ", ".join(reasons)
        return {
            "reject": reject,
            "quality": quality,
            "reason": reason_text,
            "reasons": reasons,
            "top_score": top_score,
            "top_lexical": top_lexical,
        }

    def _graph_reject_guidance(self, tool_name: str, analysis: dict[str, Any], policy: QuestionPolicy | None) -> str:
        detail = analysis.get("reason") or "weak graph evidence"
        if tool_name == "global_graph_query_tool":
            base = "Reject the global graph hit as a terminal anchor."
        elif tool_name == "segment_graph_query_tool":
            base = "Reject the segment graph hit as a terminal anchor."
        else:
            base = "Reject the frame graph hit as a terminal anchor."
        if policy is not None and policy.requires_targeted_fi:
            return (
                f"{base} Reason: {detail}. Use graph only for coarse localization, then switch to "
                f"`caption_backoff_tool` or a narrower targeted `frame_inspect_tool` for the {policy.targeted_label}."
            )
        return (
            f"{base} Reason: {detail}. Fall back to `caption_backoff_tool` or `frame_inspect_tool` inside the target time span before answering."
        )

    def _record_clip_search(
        self,
        state: PolicyState,
        policy: QuestionPolicy,
        tool_output: str,
    ) -> bool:
        ranges = self._extract_clip_ranges(tool_output)
        best_overlap = 0.0
        time_valid = False
        target_ranges = self._policy_target_ranges(policy)
        for start_sec, end_sec in ranges:
            overlaps, overlap = self._span_overlaps_reference(start_sec, end_sec, target_ranges)
            time_valid = time_valid or overlaps
            best_overlap = max(best_overlap, overlap)
        state.clip_records.append(
            {
                "ranges": ranges,
                "time_valid": time_valid or not target_ranges,
                "best_overlap": best_overlap,
            }
        )
        improved = (time_valid and best_overlap >= state.best_clip_overlap) or (not target_ranges and bool(ranges))
        if improved:
            state.best_clip_overlap = max(state.best_clip_overlap, best_overlap)
        state.graph_backoff_active = False
        return improved

    def _record_frame_inspect(
        self,
        state: PolicyState,
        policy: QuestionPolicy,
        args: dict[str, Any],
        tool_output: str,
    ) -> bool:
        ranges = args.get("time_ranges_hhmmss") or []
        converted: list[tuple[float, float]] = []
        for item in ranges:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                starts = self._extract_time_marks_sec(str(item[0]))
                ends = self._extract_time_marks_sec(str(item[1]))
                if starts and ends:
                    converted.append((starts[0], ends[0]))
        duration = sum(max(0.0, end - start) for start, end in converted)
        target_ranges = self._policy_target_ranges(policy)
        time_valid = True
        best_overlap = 0.0
        if target_ranges and converted:
            time_valid = False
            for start_sec, end_sec in converted:
                overlaps, overlap = self._span_overlaps_reference(start_sec, end_sec, target_ranges)
                time_valid = time_valid or overlaps
                best_overlap = max(best_overlap, overlap)
        targeted = self._query_matches_target(policy, str(args.get("question", "")))
        threshold = self._targeted_window_threshold_sec(policy)
        narrow = duration > 0.0 and duration <= threshold
        signature = self._extract_fi_signature(policy, tool_output)
        record = {
            "duration_sec": duration,
            "narrow": narrow,
            "targeted": targeted,
            "time_valid": time_valid,
            "output": tool_output,
            "signature": signature,
            "ranges": converted,
        }
        state.fi_records.append(record)
        state.graph_backoff_active = False

        qualifying = self._qualifying_fi_records(policy, state)
        if record in qualifying:
            if len(qualifying) == 1:
                return True
            if policy.requires_repeated_targeted_fi:
                ok, _ = self._recent_repeated_fi_status(policy, qualifying)
                return ok
            return True
        return time_valid and not policy.requires_targeted_fi

    def _stagnation_guidance(self, policy: QuestionPolicy) -> str:
        if policy.requires_targeted_fi:
            return (
                "Recent actions did not narrow the time window or improve evidence quality. "
                f"Stop broad graph paraphrases. Either run one final narrow targeted `frame_inspect_tool` for the {policy.targeted_label}, "
                "or finish only if the existing narrow targeted confirmations are already sufficient."
            )
        return (
            "Recent actions did not improve evidence quality. Stop broad graph paraphrases and use the strongest time-valid `caption_backoff_tool` or `frame_inspect_tool` evidence to converge."
        )

    def _narrow_window_guidance(self, policy: QuestionPolicy) -> str:
        if policy.requires_repeated_targeted_fi:
            if policy.targeted_label == "count":
                return (
                    "Use the current evidence only to localize. Next, keep the counted group explicit, stay on the same localized moment, and run two overlapping narrow `frame_inspect_tool` checks for the count; only finish if the latest two agree."
                )
            return (
                f"Use the current evidence only to localize. Next, shrink to a narrower time window and run two overlapping targeted `frame_inspect_tool` checks for the {policy.targeted_label}; only finish if the latest two agree."
            )
        return (
            f"Use the current evidence only to localize. Next, shrink to a narrower time window and run a targeted `frame_inspect_tool` check for the {policy.targeted_label} before finishing."
        )

    def _note_progress(self, state: PolicyState, tool_name: str, improved: bool) -> None:
        state.recent_progress.append(bool(improved))
        state.recent_tools.append(tool_name)
        window = int(getattr(config, "POLICY_STAGNATION_WINDOW", 3))
        if len(state.recent_progress) > window:
            state.recent_progress = state.recent_progress[-window:]
            state.recent_tools = state.recent_tools[-window:]
        state.non_improving_streak = 0 if improved else state.non_improving_streak + 1
        if improved:
            state.stagnation_prompted = False

    def _logged_tool_args_json(self, args: dict[str, Any]) -> str:
        logged_args: dict[str, Any] = {}
        for key, value in args.items():
            if key == "database":
                logged_args[key] = "<NanoVectorDB>"
            else:
                logged_args[key] = value
        return json.dumps(logged_args, ensure_ascii=False, default=str)

    def _is_retryable_tool_error(self, tool_name: str, err: Exception) -> bool:
        if tool_name not in {"frame_inspect_tool", "caption_backoff_tool"}:
            return False
        low = str(err).lower()
        retry_keywords = (
            "no response from the model",
            "timeout",
            "timed out",
            "rate limit",
            "too many requests",
            "service unavailable",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "gateway timeout",
            "bad gateway",
            "returned none",
        )
        return any(keyword in low for keyword in retry_keywords)

    def _call_tool_with_retries(self, tool_name: str, target_func, args: dict[str, Any]):
        attempts = 1
        if tool_name in {"frame_inspect_tool", "caption_backoff_tool"}:
            attempts = int(getattr(config, "TOOL_CALL_RETRY_ATTEMPTS", 1))
        delay = float(getattr(config, "TOOL_CALL_RETRY_BASE_DELAY_SEC", 2.0))

        for attempt in range(1, max(1, attempts) + 1):
            try:
                return target_func(**args)
            except Exception as exc:
                if attempt >= attempts or not self._is_retryable_tool_error(tool_name, exc):
                    raise
                sleep_sec = delay * (2 ** (attempt - 1))
                print(
                    f"[WARN] Tool `{tool_name}` failed with retryable error on attempt "
                    f"{attempt}/{attempts}: {exc}. Retrying after {sleep_sec:.1f}s."
                )
                time.sleep(sleep_sec)

    def _finish_tool_choice(self):
        return {
            "type": "function",
            "function": {
                "name": "finish",
            },
        }

    def _exec_tool(
        self,
        tool_call,
        msgs,
        step_idx=None,
        tool_call_idx=None,
        trace_logger=None,
        question=None,
        question_policy: QuestionPolicy | None = None,
        policy_state: PolicyState | None = None,
    ):
        name = tool_call["function"]["name"]
        args_str = tool_call.get("function", {}).get("arguments", "{}")
        tool_call_id = tool_call.get("id")

        if name not in self.name_to_function_map:
            error_msg = f"Invalid function name: {name!r}"
            self._append_tool_msg(tool_call["id"], name, error_msg, msgs)
            if trace_logger is not None:
                trace_logger.log_tool_call(
                    step_idx=step_idx,
                    tool_call_idx=tool_call_idx,
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    tool_args=args_str,
                    tool_output=error_msg,
                    status="error",
                )
            return

        # Parse arguments
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError as exc:
            if trace_logger is not None:
                trace_logger.log_tool_call(
                    step_idx=step_idx,
                    tool_call_idx=tool_call_idx,
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    tool_args=args_str,
                    tool_output=f"Error decoding arguments: {exc!s}",
                    status="error",
                )
            raise StopException(f"Error decoding arguments: {exc!s}")

        if not isinstance(args, dict):
            err = f"Tool arguments must be a JSON object, got: {type(args).__name__}"
            self._append_tool_msg(tool_call["id"], name, err, msgs)
            if trace_logger is not None:
                trace_logger.log_tool_call(
                    step_idx=step_idx,
                    tool_call_idx=tool_call_idx,
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    tool_args=args_str,
                    tool_output=err,
                    status="error",
                )
            return

        target_func = self.name_to_function_map[name]
        if name == "finish" and self._is_single_choice_question(question or ""):
            strict_answer = self._extract_strict_choice_letter(args.get("answer"))
            if strict_answer is None:
                err = (
                    "Finish answer must be exactly one option letter (A/B/C/D) for this multiple-choice question."
                )
                self._append_tool_msg(tool_call["id"], name, err, msgs)
                self._append_guidance_once(
                    msgs,
                    "Your final answer for this question must be one capital option letter only. "
                    "Call `finish` with exactly one of A, B, C, or D.",
                )
                if trace_logger is not None:
                    trace_logger.log_tool_call(
                        step_idx=step_idx,
                        tool_call_idx=tool_call_idx,
                        tool_call_id=tool_call_id,
                        tool_name=name,
                        tool_args=args_str,
                        tool_output=err,
                        status="error",
                    )
                return
            args["answer"] = strict_answer

        graph_tools = {"global_graph_query_tool", "segment_graph_query_tool", "frame_graph_query_tool"}
        if name in graph_tools and policy_state is not None:
            if policy_state.graph_backoff_active:
                err = (
                    f"Policy gate rejected `{name}` because the last graph hit was weak ({policy_state.graph_backoff_reason or 'no stronger graph anchor yet'})."
                )
                self._append_tool_msg(tool_call["id"], name, err, msgs)
                if question_policy is not None:
                    self._append_guidance_once(
                        msgs,
                        self._graph_reject_guidance(
                            name,
                            {"reason": policy_state.graph_backoff_reason or "graph backoff is active"},
                            question_policy,
                        ),
                    )
                if trace_logger is not None:
                    trace_logger.log_tool_call(
                        step_idx=step_idx,
                        tool_call_idx=tool_call_idx,
                        tool_call_id=tool_call_id,
                        tool_name=name,
                        tool_args=args_str,
                        tool_output=err,
                        status="error",
                    )
                self._note_progress(policy_state, name, improved=False)
                return

        if name == "finish":
            ok, reason = self._finish_policy_status(question or "", question_policy, policy_state)
            if not ok:
                if policy_state is not None:
                    policy_state.finish_rejections += 1
                err = f"Policy gate blocked `finish`: {reason}"
                self._append_tool_msg(tool_call["id"], name, err, msgs)
                self._append_guidance_once(msgs, reason)
                if trace_logger is not None:
                    trace_logger.log_tool_call(
                        step_idx=step_idx,
                        tool_call_idx=tool_call_idx,
                        tool_call_id=tool_call_id,
                        tool_name=name,
                        tool_args=args_str,
                        tool_output=err,
                        status="error",
                    )
                return

        if self._tool_requires_database(target_func):
            args["database"] = self.video_db

        try:
            sig = inspect.signature(target_func)
        except Exception:
            sig = None

        if (
            sig is not None
            and question_policy is not None
            and question_policy.time_reference
            and "time_reference" in sig.parameters
            and name in graph_tools
        ):
            if bool(getattr(config, "POLICY_STRICT_GRAPH_TIME_REFERENCE", True)):
                args["time_reference"] = question_policy.time_reference
            elif not str(args.get("time_reference", "")).strip():
                args["time_reference"] = question_policy.time_reference
        
        if "topk" in args and config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
            args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK
        if "top_k" in args and config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
            args["top_k"] = config.OVERWRITE_CLIP_SEARCH_TOPK
        if "top_k_frames" in args and getattr(config, "SCENE_GRAPH_TOPK_FRAMES", 0) > 0:
            args["top_k_frames"] = config.SCENE_GRAPH_TOPK_FRAMES
        if "top_k_segments" in args and getattr(config, "SCENE_GRAPH_TOPK_SEGMENTS", 0) > 0:
            args["top_k_segments"] = config.SCENE_GRAPH_TOPK_SEGMENTS
        if "top_k_regions" in args and getattr(config, "SCENE_GRAPH_TOPK_GLOBAL_REGIONS", 0) > 0:
            args["top_k_regions"] = config.SCENE_GRAPH_TOPK_GLOBAL_REGIONS
        if "max_triples" in args and getattr(config, "SCENE_GRAPH_MAX_TRIPLES", 0) > 0:
            args["max_triples"] = config.SCENE_GRAPH_MAX_TRIPLES

        logged_args = self._logged_tool_args_json(args)

        # Call the tool
        try:
            print(f"Calling function `{name}` with args: {args}")
            result = self._call_tool_with_retries(name, target_func, args)
            self._append_tool_msg(tool_call["id"], name, result, msgs)
            improved = False
            _cs_guidance_fired = False
            if policy_state is not None and question_policy is not None:
                if name in graph_tools:
                    analysis = self._graph_result_analysis(name, result, question_policy)
                    parsed_graph = self._parse_tool_json(result) or {}
                    controller = parsed_graph.get("controller_signals", {}) or {}
                    policy_state.last_graph_requires_followup = bool(
                        controller.get("weak_match") or controller.get("localize_only")
                    )
                    policy_state.last_graph_recommended_action = str(
                        controller.get("recommended_next_action", "")
                    )
                    improved = (
                        not analysis.get("reject")
                        and float(analysis.get("quality", float("-inf"))) > policy_state.best_graph_quality
                    )
                    if improved:
                        policy_state.best_graph_quality = float(analysis["quality"])
                    if analysis.get("reject"):
                        policy_state.graph_backoff_active = True
                        policy_state.graph_backoff_reason = analysis.get("reason", "")
                        policy_state.graph_rejected_count += 1
                        self._append_guidance_once(
                            msgs,
                            self._graph_reject_guidance(name, analysis, question_policy),
                        )
                        if trace_logger is not None:
                            trace_logger._write_event(
                                "policy_graph_reject",
                                {
                                    "step_idx": step_idx,
                                    "tool_name": name,
                                    "reason": analysis.get("reason", ""),
                                },
                            )
                    elif question_policy.requires_targeted_fi:
                        # Merge narrow_window_guidance with G6 controller_signals:
                        # prefer the more specific controller signal when available.
                        if name == "segment_graph_query_tool":
                            _parsed_sg = self._parse_tool_json(result)
                            _cs = (_parsed_sg or {}).get("controller_signals", {})
                            _width = float(_cs.get("top_segment_width_sec", 0.0) or 0.0)
                            _targeted_label = question_policy.targeted_label
                            if _cs.get("question_is_fine_detail") and _cs.get("coarse_window"):
                                self._append_guidance_once(
                                    msgs,
                                    f"Controller signal: question needs fine visual detail ({_targeted_label}). "
                                    f"Matched segment is ~{_width:.0f}s wide — too coarse. "
                                    f"Use the time range as a starting point only. "
                                    f"Call `frame_inspect_tool` with a targeted, narrow window to verify the {_targeted_label}.",
                                )
                                _cs_guidance_fired = True
                            elif _cs.get("question_is_fine_detail"):
                                self._append_guidance_once(
                                    msgs,
                                    f"Controller signal: visual confirmation needed ({_targeted_label}). "
                                    f"Call `frame_inspect_tool` before `finish`.",
                                )
                                _cs_guidance_fired = True
                        if not _cs_guidance_fired:
                            self._append_guidance_once(msgs, self._narrow_window_guidance(question_policy))

                    # G6: weak_match guidance for non-fine-detail accepted segment_graph results
                    if name == "segment_graph_query_tool" and not analysis.get("reject") and not _cs_guidance_fired:
                        _parsed_sg = locals().get("_parsed_sg") or self._parse_tool_json(result)
                        _cs = (_parsed_sg or {}).get("controller_signals", {})
                        if _cs.get("weak_match") and not _cs.get("question_is_fine_detail"):
                            self._append_guidance_once(
                                msgs,
                                "Controller signal: weak graph match. "
                                "Treat returned segments as rough hints. "
                                "Verify with `caption_backoff_tool` or `frame_inspect_tool` before answering.",
                            )
                elif name == "caption_backoff_tool":
                    improved = self._record_clip_search(policy_state, question_policy, str(result))
                    policy_state.last_graph_requires_followup = False
                    if question_policy.requires_targeted_fi:
                        self._append_guidance_once(msgs, self._narrow_window_guidance(question_policy))
                elif name == "frame_inspect_tool":
                    improved = self._record_frame_inspect(policy_state, question_policy, args, str(result))
                    if "Error: Cannot find corresponding result" not in str(result):
                        policy_state.last_graph_requires_followup = False
                    last_record = policy_state.fi_records[-1] if policy_state.fi_records else None
                    if (
                        last_record is not None
                        and question_policy.requires_targeted_fi
                        and not (
                            last_record.get("targeted")
                            and last_record.get("narrow")
                            and (
                                last_record.get("time_valid")
                                or not question_policy.requires_time_valid_terminal
                            )
                        )
                    ):
                        self._append_guidance_once(msgs, self._narrow_window_guidance(question_policy))
                self._note_progress(policy_state, name, improved)
                window = int(getattr(config, "POLICY_STAGNATION_WINDOW", 3))
                if (
                    len(policy_state.recent_progress) >= window
                    and not any(policy_state.recent_progress[-window:])
                    and not policy_state.stagnation_prompted
                ):
                    policy_state.stagnation_prompted = True
                    policy_state.graph_backoff_active = True
                    policy_state.graph_backoff_reason = "recent actions did not improve evidence quality"
                    self._append_guidance_once(msgs, self._stagnation_guidance(question_policy))
                    if trace_logger is not None:
                        trace_logger._write_event(
                            "policy_stagnation",
                            {
                                "step_idx": step_idx,
                                "recent_tools": policy_state.recent_tools[-window:],
                            },
                        )
            if trace_logger is not None:
                trace_logger.log_tool_call(
                    step_idx=step_idx,
                    tool_call_idx=tool_call_idx,
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    tool_args=logged_args,
                    tool_output=result,
                    status="ok",
                )
        except StopException as exc:  # graceful stop
            print(f"Finish task with message: '{exc!s}'")
            if trace_logger is not None:
                trace_logger.log_tool_call(
                    step_idx=step_idx,
                    tool_call_idx=tool_call_idx,
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    tool_args=logged_args,
                    tool_output=str(exc),
                    status="finish",
                )
            raise
        except Exception as exc:
            err = f"Tool execution error in `{name}`: {exc!s}"
            print(f"[WARN] {err}")
            self._append_tool_msg(tool_call["id"], name, err, msgs)
            if trace_logger is not None:
                trace_logger.log_tool_call(
                    step_idx=step_idx,
                    tool_call_idx=tool_call_idx,
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    tool_args=logged_args,
                    tool_output=err,
                    status="error",
                )
            return

    def _force_finish_from_context(
        self,
        msgs,
        question,
        trace_logger=None,
        step_idx=None,
        question_policy: QuestionPolicy | None = None,
        policy_state: PolicyState | None = None,
    ):
        prompt = (
            "Do not call any more retrieval tools. Based only on the evidence already gathered, "
            "call `finish` now with exactly one option letter (A/B/C/D)."
            if self._is_single_choice_question(question)
            else "Do not call any more retrieval tools. Based only on the evidence already gathered, call `finish` now."
        )
        self._append_guidance_once(msgs, prompt)
        response = self._call_orchestrator_with_retries(
            msgs=msgs,
            step_idx=step_idx or self.max_iterations + 1,
            trace_logger=trace_logger,
            tool_choice=self._finish_tool_choice(),
        )
        if response is None:
            return None

        response.setdefault("role", "assistant")
        msgs.append(response)

        if self._has_direct_answer(response):
            ok, reason = self._finish_policy_status(question, question_policy, policy_state)
            if not ok:
                if policy_state is not None:
                    policy_state.finish_rejections += 1
                self._append_guidance_once(msgs, reason)
                return None
            if self._is_single_choice_question(question):
                strict = self._extract_strict_choice_letter(response.get("content"))
                if strict is None:
                    return None
                response["content"] = strict
            if trace_logger is not None:
                trace_logger.log_trace_end(status="finished_by_direct_answer")
            return msgs

        for tool_call_idx, tool_call in enumerate(self._normalized_tool_calls(response), start=1):
            if tool_call.get("function", {}).get("name") != "finish":
                continue
            try:
                self._exec_tool(
                    tool_call,
                    msgs,
                    step_idx=step_idx or self.max_iterations + 1,
                    tool_call_idx=tool_call_idx,
                    trace_logger=trace_logger,
                    question=question,
                    question_policy=question_policy,
                    policy_state=policy_state,
                )
            except StopException:
                if trace_logger is not None:
                    trace_logger.log_trace_end(status="finished_by_finish_tool")
                return msgs
        return None

    def _hard_force_single_choice_answer(self, msgs, question, trace_logger=None, step_idx=None):
        if not self._is_single_choice_question(question):
            return None

        prompt = (
            "Final forced-answer step. Use only the evidence already collected. "
            "Do not call any tools. Do not explain. Reply with exactly one capital option letter only: A, B, C, or D."
        )
        invalid_prompt = "Invalid format. Reply now with exactly one capital option letter only: A, B, C, or D."
        base_step = step_idx or self.max_iterations + 2

        for attempt in range(2):
            self._append_guidance_once(msgs, prompt if attempt == 0 else invalid_prompt)
            response = self._call_orchestrator_with_retries(
                msgs=msgs,
                step_idx=base_step + attempt,
                trace_logger=trace_logger,
                tool_choice="none",
            )
            if response is None:
                return None

            response.setdefault("role", "assistant")
            response["tool_calls"] = []
            msgs.append(response)

            strict = self._extract_strict_choice_letter(response.get("content"))
            if strict is None:
                continue

            response["content"] = strict
            if trace_logger is not None:
                trace_logger.log_trace_end(status="finished_by_hard_choice_answer")
            return msgs

        return None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, question, trace_id=None, question_context: dict[str, Any] | None = None) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        """
        msgs = copy.deepcopy(self.messages)
        policy = self._build_question_policy(question, question_context)
        policy_state = PolicyState()
        question_prompt = question
        policy_prompt = self._question_policy_prompt(policy)
        if policy_prompt:
            question_prompt = f"{question}\n\nRuntime policy:\n{policy_prompt}"
        msgs[-1]["content"] = msgs[-1]["content"].replace("QUESTION_PLACEHOLDER", question_prompt)
        trace_logger = self._build_trace_logger(question=question, trace_id=trace_id)
        if trace_logger is not None:
            trace_logger._write_event(
                "policy_start",
                {
                    "question_type": list(policy.question_types),
                    "time_reference": policy.time_reference,
                    "requires_targeted_fi": policy.requires_targeted_fi,
                    "requires_repeated_targeted_fi": policy.requires_repeated_targeted_fi,
                    "requires_time_valid_terminal": policy.requires_time_valid_terminal,
                    "targeted_label": policy.targeted_label,
                },
            )

        try:
            for i in range(self.max_iterations):
                q_hint = question.replace("\n", " ")
                if len(q_hint) > 80:
                    q_hint = q_hint[:80] + "..."
                print(f"[ITER {i+1}/{self.max_iterations}] {q_hint}")
                # Force a final `finish` on the last iteration to avoid hanging
                if i == self.max_iterations - 1:
                    msgs.append(
                        {
                            "role": "user",
                            "content": "Please call the `finish` function to finish the task.",
                        }
                    )

                if i == self.max_iterations - 1:
                    tool_choice = self._finish_tool_choice()
                elif i == 0:
                    tool_choice = self._tool_choice_for_name(self._initial_tool_name(question))
                else:
                    tool_choice = "auto"
                response = self._call_orchestrator_with_retries(
                    msgs=msgs,
                    step_idx=i + 1,
                    trace_logger=trace_logger,
                    tool_choice=tool_choice,
                )
                if response is None:
                    if trace_logger is not None:
                        trace_logger.log_trace_end(status="llm_call_returned_none")
                    return None

                response.setdefault("role", "assistant")
                msgs.append(response)

                if self._has_direct_answer(response):
                    if not self._has_tool_observation(msgs) and i < self.max_iterations - 1:
                        msgs.append(
                            {
                                "role": "user",
                                "content": (
                                    "Do not answer directly yet. Use the retrieval/inspection tools first, "
                                    "and only call `finish` after collecting evidence."
                                ),
                            }
                        )
                        continue
                    ok, reason = self._finish_policy_status(question, policy, policy_state)
                    if not ok:
                        if policy_state is not None:
                            policy_state.finish_rejections += 1
                        self._append_guidance_once(msgs, reason)
                        continue
                    if self._is_single_choice_question(question):
                        strict = self._extract_strict_choice_letter(response.get("content"))
                        if strict is None:
                            self._append_guidance_once(
                                msgs,
                                "Do not answer with prose. Call `finish` with exactly one option letter (A/B/C/D).",
                            )
                            continue
                        response["content"] = strict
                    if trace_logger is not None:
                        trace_logger.log_trace_end(status="finished_by_direct_answer")
                    return msgs

                # Execute any requested tool calls
                try:
                    for tool_call_idx, tool_call in enumerate(self._normalized_tool_calls(response), start=1):
                        self._exec_tool(
                            tool_call,
                            msgs,
                            step_idx=i + 1,
                            tool_call_idx=tool_call_idx,
                            trace_logger=trace_logger,
                            question=question,
                            question_policy=policy,
                            policy_state=policy_state,
                        )
                except StopException:
                    if trace_logger is not None:
                        trace_logger.log_trace_end(status="finished_by_finish_tool")
                    return msgs
        except Exception as exc:
            print(f"[ERROR] Agent run failed: {exc}")
            if trace_logger is not None:
                trace_logger._write_event("run_exception", {"error": str(exc)})
                trace_logger.log_trace_end(status="run_exception")
            return None

        forced = self._force_finish_from_context(
            msgs=msgs,
            question=question,
            trace_logger=trace_logger,
            step_idx=self.max_iterations + 1,
            question_policy=policy,
            policy_state=policy_state,
        )
        if forced is not None:
            return forced
        hard_forced = self._hard_force_single_choice_answer(
            msgs=msgs,
            question=question,
            trace_logger=trace_logger,
            step_idx=self.max_iterations + 2,
        )
        if hard_forced is not None:
            return hard_forced
        if self._is_single_choice_question(question):
            strict = self._recover_single_choice_from_messages(msgs)
            if strict is not None:
                msgs.append({"role": "assistant", "content": strict, "tool_calls": []})
                if trace_logger is not None:
                    trace_logger.log_trace_end(status="recovered_after_max_iterations")
                return msgs
        if trace_logger is not None:
            trace_logger.log_trace_end(status="max_iterations_reached")
        return msgs

    def parallel_run(self, questions, max_workers=4, trace_ids=None, question_contexts=None) -> list[list[dict]]:
        """
        Run multiple questions in parallel.
        """
        if trace_ids is not None and len(trace_ids) != len(questions):
            raise ValueError("trace_ids length must match questions length.")
        if question_contexts is not None and len(question_contexts) != len(questions):
            raise ValueError("question_contexts length must match questions length.")
        results = []
        results = [None] * len(questions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self.run,
                    q,
                    trace_id=(trace_ids[idx] if trace_ids is not None else None),
                    question_context=(question_contexts[idx] if question_contexts is not None else None),
                ): idx
                for idx, q in enumerate(questions)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error processing question: {e}")
                    retry_attempts = int(getattr(config, "PARALLEL_RUN_RETRY_ATTEMPTS", 1))
                    recovered = None
                    for attempt in range(1, max(1, retry_attempts) + 1):
                        print(
                            f"[WARN] Retrying failed worker idx={idx} "
                            f"attempt={attempt}/{retry_attempts}"
                        )
                        recovered = self.run(
                            questions[idx],
                            trace_id=(trace_ids[idx] if trace_ids is not None else None),
                            question_context=(question_contexts[idx] if question_contexts is not None else None),
                        )
                        if recovered is not None:
                            break
                    results[idx] = recovered
        return results

    # ------------------------------------------------------------------ #
    # Streaming (generator) loop
    # ------------------------------------------------------------------ #
    def stream_run(self, question, trace_id=None, question_context: dict[str, Any] | None = None):
        """
        A generator version of `run`.  
        Yields:
            dict: every assistant / tool message produced during reasoning.
        """
        msgs = copy.deepcopy(self.messages)
        policy = self._build_question_policy(question, question_context)
        policy_state = PolicyState()
        question_prompt = question
        policy_prompt = self._question_policy_prompt(policy)
        if policy_prompt:
            question_prompt = f"{question}\n\nRuntime policy:\n{policy_prompt}"
        msgs[-1]["content"] = msgs[-1]["content"].replace("QUESTION_PLACEHOLDER", question_prompt)
        trace_logger = self._build_trace_logger(question=question, trace_id=trace_id)
        if trace_logger is not None:
            trace_logger._write_event(
                "policy_start",
                {
                    "question_type": list(policy.question_types),
                    "time_reference": policy.time_reference,
                    "requires_targeted_fi": policy.requires_targeted_fi,
                    "requires_repeated_targeted_fi": policy.requires_repeated_targeted_fi,
                    "requires_time_valid_terminal": policy.requires_time_valid_terminal,
                    "targeted_label": policy.targeted_label,
                },
            )

        for i in range(self.max_iterations):
            q_hint = question.replace("\n", " ")
            if len(q_hint) > 80:
                q_hint = q_hint[:80] + "..."
            print(f"[ITER {i+1}/{self.max_iterations}] {q_hint}")
            # Force a final `finish` on the last iteration
            if i == self.max_iterations - 1:
                final_usr_msg = {
                    "role": "user",
                    "content": "Please call the `finish` function to finish the task.",
                }
                msgs.append(final_usr_msg)
                # Don't yield user messages to the UI

            if i == self.max_iterations - 1:
                tool_choice = self._finish_tool_choice()
            elif i == 0:
                tool_choice = self._tool_choice_for_name(self._initial_tool_name(question))
            else:
                tool_choice = "auto"
            response = self._call_orchestrator_with_retries(
                msgs=msgs,
                step_idx=i + 1,
                trace_logger=trace_logger,
                tool_choice=tool_choice,
            )
            if response is None:
                if trace_logger is not None:
                    trace_logger.log_trace_end(status="llm_call_returned_none")
                return

            response.setdefault("role", "assistant")
            msgs.append(response)
            yield response  # ← stream assistant reply

            if self._has_direct_answer(response):
                if not self._has_tool_observation(msgs) and i < self.max_iterations - 1:
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                "Do not answer directly yet. Use the retrieval/inspection tools first, "
                                "and only call `finish` after collecting evidence."
                            ),
                        }
                    )
                    continue
                ok, reason = self._finish_policy_status(question, policy, policy_state)
                if not ok:
                    if policy_state is not None:
                        policy_state.finish_rejections += 1
                    self._append_guidance_once(msgs, reason)
                    continue
                if self._is_single_choice_question(question):
                    strict = self._extract_strict_choice_letter(response.get("content"))
                    if strict is None:
                        self._append_guidance_once(
                            msgs,
                            "Do not answer with prose. Call `finish` with exactly one option letter (A/B/C/D).",
                        )
                        continue
                    response["content"] = strict
                if trace_logger is not None:
                    trace_logger.log_trace_end(status="finished_by_direct_answer")
                return

            # Execute any requested tool calls
            try:
                for tool_call_idx, tool_call in enumerate(self._normalized_tool_calls(response), start=1):
                    # Yield a formatted message about the tool being called
                    tool_name = tool_call.get("function", {}).get("name", "unknown")
                    tool_args = tool_call.get("function", {}).get("arguments", "{}")
                    yield {
                        "role": "tool_call",
                        "name": tool_name,
                        "arguments": tool_args
                    }
                    
                    self._exec_tool(
                        tool_call,
                        msgs,
                        step_idx=i + 1,
                        tool_call_idx=tool_call_idx,
                        trace_logger=trace_logger,
                        question=question,
                        question_policy=policy,
                        policy_state=policy_state,
                    )
                    # Only yield the tool result message
                    if msgs[-1].get("role") == "tool":
                        yield msgs[-1]  # ← stream tool observation
            except StopException:
                if trace_logger is not None:
                    trace_logger.log_trace_end(status="finished_by_finish_tool")
                return

        forced = self._force_finish_from_context(
            msgs=msgs,
            question=question,
            trace_logger=trace_logger,
            step_idx=self.max_iterations + 1,
            question_policy=policy,
            policy_state=policy_state,
        )
        if forced is not None:
            return
        hard_forced = self._hard_force_single_choice_answer(
            msgs=msgs,
            question=question,
            trace_logger=trace_logger,
            step_idx=self.max_iterations + 2,
        )
        if hard_forced is not None:
            yield hard_forced[-1]
            return
        if self._is_single_choice_question(question):
            strict = self._recover_single_choice_from_messages(msgs)
            if strict is not None:
                recovery_msg = {"role": "assistant", "content": strict, "tool_calls": []}
                msgs.append(recovery_msg)
                yield recovery_msg
                if trace_logger is not None:
                    trace_logger.log_trace_end(status="recovered_after_max_iterations")
                return
        if trace_logger is not None:
            trace_logger.log_trace_end(status="max_iterations_reached")
