from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from sgd.scene_graph.schema import safe_load_stage1_graph

START_HINTS = (
    "at the beginning",
    "in the beginning",
    "at first",
    "initially",
    "start of the video",
    "beginning of the video",
    "early in the video",
)
END_HINTS = (
    "at the end",
    "in the end",
    "finally",
    "last part",
    "end of the video",
    "towards the end",
    "near the end",
)
MIDDLE_HINTS = (
    "in the middle",
    "middle of the video",
    "midway",
)
TEMPORAL_CUES = (
    "first",
    "last",
    "before",
    "after",
    "then",
    "while",
    "during",
    "earlier",
    "later",
    "previous",
    "next",
)


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(tok) > 1}


def _extract_time_marks_sec(text: str) -> list[float]:
    marks: list[float] = []
    for m in re.finditer(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)", text or ""):
        a = int(m.group(1))
        b = int(m.group(2))
        c = m.group(3)
        if c is None:
            marks.append(float(a * 60 + b))
        else:
            marks.append(float(a * 3600 + b * 60 + int(c)))
    return marks


def _query_time_span(query: str) -> tuple[float, float] | None:
    marks = _extract_time_marks_sec(query)
    if len(marks) < 2:
        return None
    return (min(marks), max(marks))


def _infer_temporal_hint(query: str, max_t_sec: float) -> dict[str, Any]:
    q = (query or "").lower()
    span = _query_time_span(query)
    anchor = None
    if any(k in q for k in START_HINTS):
        anchor = "start"
    elif any(k in q for k in END_HINTS):
        anchor = "end"
    elif any(k in q for k in MIDDLE_HINTS):
        anchor = "middle"

    has_temporal_cue = any(k in q for k in TEMPORAL_CUES)
    return {
        "span": span,
        "anchor": anchor,
        "has_temporal_cue": has_temporal_cue,
        "max_t_sec": float(max_t_sec),
    }


def _point_distance_to_span(t_sec: float, span: tuple[float, float]) -> float:
    s0, s1 = span
    if s0 <= t_sec <= s1:
        return 0.0
    if t_sec < s0:
        return s0 - t_sec
    return t_sec - s1


def _anchor_target_sec(anchor: str, max_t_sec: float) -> float:
    if max_t_sec <= 0:
        return 0.0
    if anchor == "start":
        return 0.0
    if anchor == "end":
        return max_t_sec
    return max_t_sec * 0.5


def _temporal_adjustment(t_sec: float, hint: dict[str, Any]) -> tuple[float, float, bool]:
    """
    Returns:
      temporal_score_delta, temporal_distance_sec, temporal_in_target
    """
    span = hint.get("span")
    if span is not None:
        d = _point_distance_to_span(t_sec, span)
        if d <= 0:
            return 3.0, 0.0, True
        # Strongly penalize far-away timestamps for explicit time queries.
        delta = 0.7 / (1.0 + d / 20.0) - min(2.5, d / 80.0)
        return delta, d, False

    anchor = hint.get("anchor")
    max_t_sec = float(hint.get("max_t_sec", 0.0))
    if anchor:
        target = _anchor_target_sec(anchor, max_t_sec)
        d = abs(t_sec - target)
        norm = d / max(max_t_sec, 1.0)
        # Mild anchor preference; keep semantic score dominant.
        delta = 0.8 - min(1.2, norm * 1.6)
        return delta, d, False

    return 0.0, 0.0, False


def _fuzzy_token_overlap(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    score = 0.0
    for q in query_tokens:
        best = 0.0
        for t in text_tokens:
            if q == t:
                best = 1.0
                break
            sim = SequenceMatcher(None, q, t).ratio()
            if sim > best:
                best = sim
        if best >= 0.9:
            score += 1.0
        elif best >= 0.8:
            score += 0.6
        elif best >= 0.7:
            score += 0.3
    return score


def _entity_text(ent: dict[str, Any]) -> str:
    parts = [
        ent.get("canonical_name", ""),
        ent.get("type", ""),
        ent.get("desc_text", ""),
        " ".join(ent.get("aliases", [])),
        " ".join(ent.get("attributes", [])),
    ]
    return " ".join(str(p) for p in parts if p)


def _frame_text(frame: dict[str, Any], entities_by_id: dict[str, dict[str, Any]]) -> str:
    parts = [str(frame.get("caption_context", ""))]
    for item in frame.get("entities", []):
        ent = entities_by_id.get(item.get("entity_id"), {})
        parts.append(_entity_text(ent))
    for rel in frame.get("relations", []):
        parts.extend([str(rel.get("s", "")), str(rel.get("p", "")), str(rel.get("o", ""))])
    return " ".join(p for p in parts if p)


def _temporal_prior(query_span: tuple[float, float], t_sec: float) -> float:
    q0, q1 = query_span
    if q0 <= t_sec <= q1:
        return 2.5
    dist = min(abs(t_sec - q0), abs(t_sec - q1))
    return 0.8 / (1.0 + dist / 20.0)


def _select_with_time_diversity(
    candidates: list[dict[str, Any]],
    top_k: int,
    key_name: str = "t_sec",
    diversity_window_sec: float = 45.0,
    diversity_penalty: float = 0.35,
) -> list[dict[str, Any]]:
    if top_k <= 0 or not candidates:
        return []

    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < top_k:
        best_idx = 0
        best_val = float("-inf")
        for idx, item in enumerate(remaining):
            score = float(item.get("score", 0.0))
            t = float(item.get(key_name, 0.0))
            penalty = 0.0
            for picked in selected:
                pt = float(picked.get(key_name, 0.0))
                dt = abs(t - pt)
                if dt < diversity_window_sec:
                    penalty = max(penalty, (diversity_window_sec - dt) / diversity_window_sec)
            adjusted = score - diversity_penalty * penalty
            if adjusted > best_val:
                best_val = adjusted
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def query_stage1_graph(
    graph: dict[str, Any],
    query: str,
    time_reference: str | None = None,
    top_k_frames: int = 8,
    max_triples: int = 24,
    min_soft_score: float = 0.35,
    min_fallback_score: float = 0.45,
    fallback_relative_floor: float = 0.35,
) -> dict[str, Any]:
    query_tokens = _tokenize(query)
    entities_by_id = {ent["entity_id"]: ent for ent in graph.get("entities", [])}
    frame_times = [float(frame.get("t_sec", 0.0)) for frame in graph.get("frames", [])]
    max_t_sec = max(frame_times) if frame_times else 0.0
    temporal_source = query if not time_reference else f"{query} {time_reference}"
    temporal_hint = _infer_temporal_hint(query=temporal_source, max_t_sec=max_t_sec)
    query_span = temporal_hint.get("span")

    ranked_frames = []
    fallback_candidates = []
    all_soft_candidates = []
    for frame in graph.get("frames", []):
        lexical_score = 0.0
        soft_score = 0.0
        match_entities = []
        for item in frame.get("entities", []):
            ent = entities_by_id.get(item.get("entity_id"))
            if ent is None:
                continue
            ent_tokens = _tokenize(_entity_text(ent))
            overlap = len(query_tokens & ent_tokens)
            if overlap > 0:
                lexical_score += float(overlap) + float(item.get("confidence", 0.0)) * 0.25
                match_entities.append(
                    {
                        "entity_id": ent["entity_id"],
                        "canonical_name": ent.get("canonical_name", ""),
                        "type": ent.get("type", ""),
                    }
                )

        match_relations = []
        for rel in frame.get("relations", []):
            p_tokens = _tokenize(rel.get("p", ""))
            overlap = len(query_tokens & p_tokens)
            if overlap > 0:
                lexical_score += float(overlap) + float(rel.get("confidence", 0.0)) * 0.2
                match_relations.append(rel)

        caption_tokens = _tokenize(frame.get("caption_context", ""))
        lexical_score += float(len(query_tokens & caption_tokens)) * 0.6

        f_text = _frame_text(frame, entities_by_id)
        f_tokens = _tokenize(f_text)
        soft_score += _fuzzy_token_overlap(query_tokens, f_tokens) * 0.35
        soft_score += SequenceMatcher(None, query.lower(), f_text.lower()).ratio() * 0.6
        base_score = lexical_score + soft_score
        temporal_delta, temporal_distance_sec, temporal_in_target = _temporal_adjustment(
            t_sec=float(frame.get("t_sec", 0.0)),
            hint=temporal_hint,
        )
        final_score = base_score + temporal_delta

        payload = {
            "frame_id": frame.get("frame_id"),
            "t_sec": frame.get("t_sec"),
            "score": round(final_score, 4),
            "base_score": round(base_score, 4),
            "temporal_score": round(temporal_delta, 4),
            "temporal_distance_sec": round(temporal_distance_sec, 3),
            "temporal_in_target": temporal_in_target,
            "lexical_score": round(lexical_score, 4),
            "soft_score": round(soft_score, 4),
            "matched_entities": match_entities,
            "matched_relations": match_relations,
            "caption_context": frame.get("caption_context", ""),
        }
        all_soft_candidates.append(payload)
        if lexical_score > 0:
            ranked_frames.append(payload)
        elif soft_score >= min_soft_score:
            fallback_candidates.append(payload)

    def _sort_key(x: dict[str, Any]) -> tuple[float, float]:
        if query_span is not None:
            return (
                float(1.0 if x.get("temporal_in_target") else 0.0),
                -float(x.get("temporal_distance_sec", 0.0)),
                float(x.get("score", 0.0)),
            )
        anchor = temporal_hint.get("anchor")
        if anchor:
            return (
                -float(x.get("temporal_distance_sec", 0.0)),
                float(x.get("score", 0.0)),
            )
        t = float(x.get("t_sec", 0.0))
        return (float(x.get("score", 0.0)), -t)

    ranked_frames.sort(key=_sort_key, reverse=True)
    if not ranked_frames:
        fallback_candidates.sort(key=_sort_key, reverse=True)
        all_soft_candidates.sort(key=_sort_key, reverse=True)
        candidates = fallback_candidates if fallback_candidates else all_soft_candidates

        # For explicit time ranges, keep fallback within a sane distance if possible.
        if query_span is not None and candidates:
            span_len = max(1.0, query_span[1] - query_span[0])
            dist_cap = max(45.0, span_len * 2.0)
            near = [c for c in candidates if float(c.get("temporal_distance_sec", 1e9)) <= dist_cap]
            if near:
                candidates = near

        if candidates:
            max_score = float(candidates[0].get("score", 0.0))
            floor = max(min_fallback_score, max_score * fallback_relative_floor)
            ranked_frames = [x for x in candidates if float(x.get("score", 0.0)) >= floor]
            if not ranked_frames:
                ranked_frames = candidates[:1]
            # For non-explicit-time queries, diversify top-k over time.
            if query_span is None and temporal_hint.get("anchor") is None:
                ranked_frames = _select_with_time_diversity(ranked_frames, top_k_frames)
            else:
                ranked_frames = ranked_frames[:top_k_frames]
    else:
        if query_span is None and temporal_hint.get("anchor") is None:
            ranked_frames = _select_with_time_diversity(ranked_frames, top_k_frames)
        else:
            ranked_frames = ranked_frames[:top_k_frames]

    flat_rel = []
    for item in ranked_frames:
        for rel in item["matched_relations"]:
            flat_rel.append(
                {
                    "frame_id": item["frame_id"],
                    "t_sec": item["t_sec"],
                    "s": rel.get("s"),
                    "p": rel.get("p"),
                    "o": rel.get("o"),
                    "confidence": rel.get("confidence", 0.0),
                }
            )
    flat_rel = sorted(flat_rel, key=lambda x: x["confidence"], reverse=True)[:max_triples]

    return {
        "query": query,
        "time_reference": time_reference,
        "graph_version": graph.get("version"),
        "video_id": graph.get("video_id"),
        "temporal_hint": {
            "span": list(query_span) if query_span is not None else None,
            "anchor": temporal_hint.get("anchor"),
            "has_temporal_cue": temporal_hint.get("has_temporal_cue", False),
        },
        "matched_frames": ranked_frames,
        "top_triples": flat_rel,
        "fallback_used": bool(ranked_frames and all(float(f.get("lexical_score", 0.0)) <= 0 for f in ranked_frames)),
    }


def query_stage1_graph_by_path(
    graph_path: str,
    query: str,
    time_reference: str | None = None,
    top_k_frames: int = 8,
    max_triples: int = 24,
) -> dict[str, Any]:
    graph = safe_load_stage1_graph(graph_path)
    if graph is None:
        raise ValueError(f"Invalid or missing stage1 graph: {graph_path}")
    return query_stage1_graph(
        graph=graph,
        query=query,
        time_reference=time_reference,
        top_k_frames=top_k_frames,
        max_triples=max_triples,
    )


def result_to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
