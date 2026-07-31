"""Lightweight retrieval over the heuristic Global Event Index.

This module retrieves coarse event regions for navigation.  It does not expose
causal, state-transition, or VLM-abstracted global-graph semantics.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from sgd.scene_graph.global_event_index import build_global_event_index_from_path


MAX_TOP_K = 5
WEAK_MATCH_THRESHOLD = 0.35
AMBIGUITY_MARGIN = 0.10
COARSE_WINDOW_SEC = 60.0

BASIC_GR_BACKEND = {
    "name": "heuristic_gei_retrieval",
    "version": "basic_gr_v1",
    "is_vlm_global_graph": False,
    "supports_causal_edges": False,
    "supports_state_edges": False,
}

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "with",
}
_START_HINTS = (
    "at the beginning",
    "in the beginning",
    "at first",
    "initially",
    "start of the video",
    "beginning of the video",
)
_END_HINTS = (
    "at the end",
    "in the end",
    "finally",
    "last part",
    "end of the video",
    "near the end",
)
_MIDDLE_HINTS = ("in the middle", "middle of the video", "midway")
_OTHER_TEMPORAL_HINTS = ("before", "after", "then", "earlier", "later", "next")
_SEQUENCE_CUE_SUFFIX = (
    r"(?=\s*(?:[?.!,]|$|\b(?:before|after|during|in|when|then)\b))"
)
_FIRST_SEQUENCE_PATTERN = re.compile(
    r"\b(?:happens?|happened|does|do|did|appears?|appeared|shown?|seen|"
    r"comes?|came|occurs?|occurred)\b[^?]{0,60}\bfirst\b"
    + _SEQUENCE_CUE_SUFFIX
    + r"|"
    r"\bfirst\s+(?:event|action|scene|thing|step|part|moment)\b",
    flags=re.IGNORECASE,
)
_LAST_SEQUENCE_PATTERN = re.compile(
    r"\b(?:happens?|happened|does|do|did|appears?|appeared|shown?|seen|"
    r"comes?|came|occurs?|occurred)\b[^?]{0,60}\blast\b"
    + _SEQUENCE_CUE_SUFFIX
    + r"|"
    r"\blast\s+(?:event|action|scene|thing|step|part|moment)\b",
    flags=re.IGNORECASE,
)
_CAUSAL_PATTERN = re.compile(
    r"\bwhy\b|\bbecause\b|\bcause(?:d|s)?\b|\bresult(?:ed|s)? in\b|"
    r"\bleads? to\b|\bconsequence\b",
    flags=re.IGNORECASE,
)


def _tokenize(text: str) -> set[str]:
    normalized = (text or "").lower().replace("_", " ")
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _parse_time_value(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid timestamps.")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Timestamp must not be empty.")
        if ":" not in text:
            seconds = float(text)
        else:
            parts = text.split(":")
            if len(parts) == 2:
                minutes, secs = map(float, parts)
                if minutes < 0 or not 0 <= secs < 60:
                    raise ValueError(f"Invalid timestamp: {value!r}")
                seconds = minutes * 60 + secs
            elif len(parts) == 3:
                hours, minutes, secs = map(float, parts)
                if hours < 0 or not 0 <= minutes < 60 or not 0 <= secs < 60:
                    raise ValueError(f"Invalid timestamp: {value!r}")
                seconds = hours * 3600 + minutes * 60 + secs
            else:
                raise ValueError(f"Invalid timestamp: {value!r}")
    if seconds < 0:
        raise ValueError("Timestamp must be non-negative.")
    return seconds


def _normalize_scope(scope: Any) -> tuple[float, float] | None:
    if scope is None:
        return None
    if isinstance(scope, str):
        parts = [part.strip() for part in scope.split("-")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "scope must use the HH:MM:SS-HH:MM:SS format."
            )
        start_raw, end_raw = parts
    elif isinstance(scope, (list, tuple)) and len(scope) == 2:
        # Internal compatibility for existing callers.  The model-facing
        # function schema uses the string form above.
        start_raw, end_raw = scope
    else:
        raise ValueError(
            "scope must use the HH:MM:SS-HH:MM:SS format."
        )
    start = _parse_time_value(start_raw)
    end = _parse_time_value(end_raw)
    if end <= start:
        raise ValueError("scope end must be greater than scope start.")
    return start, end


def _format_hhmmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_scope(scope: tuple[float, float]) -> str:
    return f"{_format_hhmmss(scope[0])}-{_format_hhmmss(scope[1])}"


def _extract_query_time_marks(query: str) -> list[float]:
    marks: list[float] = []
    pattern = r"(?<!\d)(\d{1,3}):(\d{2})(?::(\d{2}))?(?!\d)"
    for match in re.finditer(pattern, query or ""):
        first = int(match.group(1))
        second = int(match.group(2))
        third = match.group(3)
        if third is None:
            if second < 60:
                marks.append(float(first * 60 + second))
        elif second < 60 and int(third) < 60:
            marks.append(float(first * 3600 + second * 60 + int(third)))
    return marks


def _contains_temporal_phrase(text: str, hints: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(hint)}(?!\w)", text)
        for hint in hints
    )


def _infer_temporal_hint(query: str) -> dict[str, Any]:
    query_lower = (query or "").lower()
    marks = _extract_query_time_marks(query)
    anchor = None
    if (
        _contains_temporal_phrase(query_lower, _START_HINTS)
        or _FIRST_SEQUENCE_PATTERN.search(query_lower)
    ):
        anchor = "start"
    elif (
        _contains_temporal_phrase(query_lower, _END_HINTS)
        or _LAST_SEQUENCE_PATTERN.search(query_lower)
    ):
        anchor = "end"
    elif _contains_temporal_phrase(query_lower, _MIDDLE_HINTS):
        anchor = "middle"
    explicit_span = None
    explicit_point = None
    if len(marks) >= 2:
        explicit_span = (min(marks), max(marks))
    elif marks:
        explicit_point = marks[0]
    return {
        "anchor": anchor,
        "explicit_span": explicit_span,
        "explicit_point": explicit_point,
        "has_temporal_cue": bool(
            anchor
            or marks
            or _contains_temporal_phrase(query_lower, _OTHER_TEMPORAL_HINTS)
        ),
    }


def _temporal_score(
    event_start: float,
    event_end: float,
    hint: dict[str, Any],
    video_start: float,
    video_end: float,
) -> float:
    explicit_span = hint.get("explicit_span")
    if explicit_span is not None:
        target_start, target_end = explicit_span
        intersection = max(
            0.0,
            min(event_end, target_end) - max(event_start, target_start),
        )
        target_width = max(target_end - target_start, 1.0)
        return min(1.0, intersection / target_width)

    explicit_point = hint.get("explicit_point")
    if explicit_point is not None:
        if event_start <= explicit_point <= event_end:
            return 1.0
        distance = min(abs(explicit_point - event_start), abs(explicit_point - event_end))
        return 1.0 / (1.0 + distance / 60.0)

    anchor = hint.get("anchor")
    if not anchor:
        return 0.0
    video_width = max(video_end - video_start, 1.0)
    midpoint = (event_start + event_end) * 0.5
    position = min(1.0, max(0.0, (midpoint - video_start) / video_width))
    if anchor == "start":
        return 1.0 - position
    if anchor == "end":
        return position
    return max(0.0, 1.0 - abs(position - 0.5) * 2.0)


def _event_text(event: dict[str, Any]) -> str:
    keywords = event.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = [str(keywords)]
    return " ".join(
        [
            str(event.get("entities", "")),
            str(event.get("actions", "")),
            " ".join(str(keyword) for keyword in keywords),
        ]
    )


def _score_event(
    event: dict[str, Any],
    query_tokens: set[str],
    temporal_hint: dict[str, Any],
    video_start: float,
    video_end: float,
    scope: tuple[float, float] | None,
) -> dict[str, Any]:
    event_tokens = _tokenize(_event_text(event))
    exact = query_tokens & event_tokens
    lexical = len(exact) / max(len(query_tokens), 1)

    fuzzy_total = 0.0
    for query_token in query_tokens - exact:
        best = max(
            (
                SequenceMatcher(None, query_token, event_token).ratio()
                for event_token in event_tokens
            ),
            default=0.0,
        )
        if best >= 0.7:
            fuzzy_total += best
    soft = fuzzy_total / max(len(query_tokens), 1)

    event_start = float(event.get("t_start", 0.0))
    event_end = float(event.get("t_end", event_start))
    temporal = _temporal_score(
        event_start,
        event_end,
        temporal_hint,
        video_start,
        video_end,
    )

    lexical = round(lexical, 4)
    soft = round(soft, 4)
    temporal = round(temporal, 4)
    score = round(lexical + soft + temporal, 4)

    payload = {
        "event_id": event.get("event_id"),
        "t_start": event_start,
        "t_end": event_end,
        "entities": event.get("entities", "—"),
        "actions": event.get("actions", "—"),
        "keywords": list(event.get("keywords", []) or []),
        "num_segments": int(event.get("num_segments", 0) or 0),
        "score": score,
        "score_components": {
            "lexical": lexical,
            "soft": soft,
            "temporal": temporal,
        },
        "evidence_links": {
            "segment_ids": list(event.get("support_segment_ids", []) or []),
        },
    }
    if scope is not None:
        intersection = max(
            0.0,
            min(event_end, scope[1]) - max(event_start, scope[0]),
        )
        payload["scope_overlap"] = round(
            intersection / max(event_end - event_start, 1.0),
            4,
        )
    return payload


def _select_overview(
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: (item["t_start"], item["t_end"]))
    if len(ordered) <= top_k:
        return ordered
    if top_k == 1:
        return [ordered[len(ordered) // 2]]
    indices = [
        int(round(i * (len(ordered) - 1) / (top_k - 1)))
        for i in range(top_k)
    ]
    return [ordered[index] for index in indices]


def _recommended_scope(
    event: dict[str, Any] | None,
    scope: tuple[float, float] | None,
) -> str | None:
    if event is None:
        return None
    start = float(event.get("t_start", 0.0))
    end = float(event.get("t_end", start))
    if scope is not None:
        start = max(start, scope[0])
        end = min(end, scope[1])
    if end <= start:
        return None
    return _format_scope((start, end))


def query_global_event_index(
    index: dict[str, Any],
    query: str,
    scope: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve coarse event regions from a built heuristic GEI.

    Scope is an optional ``HH:MM:SS-HH:MM:SS`` hard filter.  The returned
    regions are navigation candidates only and always require verification
    through a finer-level tool.
    """
    normalized_scope = _normalize_scope(scope)
    requested_top_k = int(top_k)
    effective_top_k = min(MAX_TOP_K, max(1, requested_top_k))
    events = sorted(
        list(index.get("event_timeline", []) or []),
        key=lambda event: (
            float(event.get("t_start", 0.0)),
            float(event.get("t_end", event.get("t_start", 0.0))),
            str(event.get("event_id", "")),
        ),
    )
    video_start = min(
        (float(event.get("t_start", 0.0)) for event in events),
        default=0.0,
    )
    video_end = max(
        (float(event.get("t_end", event.get("t_start", 0.0))) for event in events),
        default=video_start,
    )

    if normalized_scope is not None:
        candidates = [
            event
            for event in events
            if float(event.get("t_end", event.get("t_start", 0.0))) > normalized_scope[0]
            and float(event.get("t_start", 0.0)) < normalized_scope[1]
        ]
    else:
        candidates = events

    query_tokens = _tokenize(query)
    temporal_hint = _infer_temporal_hint(query)
    scored = [
        _score_event(
            event,
            query_tokens,
            temporal_hint,
            video_start,
            video_end,
            normalized_scope,
        )
        for event in candidates
    ]
    ranked = sorted(
        scored,
        key=lambda item: (item["score"], -item["t_start"], -item["t_end"]),
        reverse=True,
    )

    semantic_hit = any(
        item["score_components"]["lexical"] > 0.0
        or item["score_components"]["soft"] > 0.0
        for item in ranked
    )
    temporal_hit = any(
        item["score_components"]["temporal"] > 0.0
        for item in ranked
    )
    fallback_used = not semantic_hit
    overview_fallback = fallback_used and not temporal_hit
    if overview_fallback:
        matched = _select_overview(ranked, effective_top_k)
    else:
        matched = ranked[:effective_top_k]

    top = matched[0] if matched else None
    second = matched[1] if len(matched) > 1 else None
    top_score = float(top.get("score", 0.0)) if top else 0.0
    top_components = top.get("score_components", {}) if top else {}
    top_has_semantic_hit = bool(
        float(top_components.get("lexical", 0.0)) > 0.0
        or float(top_components.get("soft", 0.0)) > 0.0
    )
    weak_match = (
        top is None
        or not top_has_semantic_hit
        or top_score < WEAK_MATCH_THRESHOLD
    )
    ambiguous_top = bool(
        top is not None
        and second is not None
        and top_score - float(second.get("score", 0.0)) < AMBIGUITY_MARGIN
    )
    coarse_window = bool(
        top is not None
        and float(top.get("t_end", 0.0)) - float(top.get("t_start", 0.0))
        > COARSE_WINDOW_SEC
    )
    scope_miss = normalized_scope is not None and not candidates
    causal_query = bool(_CAUSAL_PATTERN.search(query or ""))

    controller_signals = {
        "localize_only": True,
        "verification_required": True,
        "weak_match": weak_match,
        "ambiguous_top": ambiguous_top,
        "coarse_window": coarse_window,
        "scope_applied": normalized_scope is not None,
        "scope_miss": scope_miss,
        "fallback_used": fallback_used,
        "overview_fallback": overview_fallback,
        "no_temporal_hit": bool(
            temporal_hint["has_temporal_cue"] and not temporal_hit
        ),
        "causal_query": causal_query,
        "causal_relation_available": False,
        "recommended_next_tool": (
            "segment_graph_query_tool" if top is not None else "clip_search_tool"
        ),
        "recommended_scope": _recommended_scope(top, normalized_scope),
        "candidate_count": len(candidates),
        "top_k_capped": requested_top_k > MAX_TOP_K,
    }

    return {
        "query": query,
        "scope": _format_scope(normalized_scope) if normalized_scope else None,
        "top_k": effective_top_k,
        "index_version": index.get("version"),
        "video_id": index.get("video_id"),
        "backend": dict(BASIC_GR_BACKEND),
        "matched_events": matched,
        "fallback_used": fallback_used,
        "controller_signals": controller_signals,
        "limitations": [
            "Heuristic event-region retrieval only.",
            "No learned global abstraction, causal edges, state transitions, or relation paths.",
        ],
    }


def query_global_event_index_by_path(
    stage2_graph_path: str,
    query: str,
    scope: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    index = build_global_event_index_from_path(stage2_graph_path)
    return query_global_event_index(
        index=index,
        query=query,
        scope=scope,
        top_k=top_k,
    )
