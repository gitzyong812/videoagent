from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from typing import Any

import sgd.config as config
from sgd.scene_graph.schema import safe_load_stage2_segment_graph
# AzureOpenAIEmbeddingService is imported lazily in _get_query_embedding() to avoid
# pulling in heavy deps (cv2 etc.) at module load time.

START_HINTS = (
    "at the beginning",
    "in the beginning",
    "at first",
    "initially",
    "opening",
    "in the opening",
    "opening scene",
    "opening caption",
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


# Patterns that indicate the question needs fine-grained visual verification.
_FINE_DETAIL_RE = re.compile(
    r"\bhow many\b|\bnumber of\b|\bcount\b"
    r"|\bwhat colou?r\b|\bcolou?r of\b|\bcolou?red\b"
    r"|\bwear(?:ing)?\b|\bclothes\b|\boutfit\b|\bhat\b|\bshirt\b|\bdress\b"
    r"|\bhair(?:style|color|colour)?\b|\blook(?:ing)? like\b"
    r"|\bon[- ]?screen\b|\bcaption\b|\bsubtitle\b|\blicense[- ]?plate\b"
    r"|\btext\b|\bsign\b|\bword[s]?\b|\bletter[s]?\b"
    r"|\bwho is\b|\bwho are\b|\bwhose\b"
)


def _detect_fine_detail_query(query: str) -> bool:
    """Return True if the query asks for fine visual detail that graph text cannot answer directly."""
    return bool(_FINE_DETAIL_RE.search((query or "").lower()))


def _compute_controller_signals(
    query: str,
    ranked_segments: list[dict[str, Any]],
    fallback_used: bool,
) -> dict[str, Any]:
    """
    Compute controller signals that guide the agent on how much to trust the graph result.

    Signals are written into the query result JSON so the orchestrator LLM can read them
    and adjust its subsequent tool-call strategy.
    """
    top = ranked_segments[0] if ranked_segments else {}
    top_lexical = float(top.get("lexical_score", 0.0) or 0.0)
    top_score = float(top.get("score", 0.0) or 0.0)
    top_soft = float(top.get("soft_score", 0.0) or 0.0)
    top_embedding_used = bool(top.get("embedding_used"))
    matched_relations = top.get("matched_relations", []) or []
    matched_entities = top.get("matched_entities", []) or []

    # ── query-type detection ──────────────────────────────────────────────────
    question_is_fine_detail = _detect_fine_detail_query(query)

    # ── match-quality signals ─────────────────────────────────────────────────
    # no_temporal_hit: true when the retrieval fell back to soft-only matching
    # with zero lexical score — graph found nothing direct.
    # If embedding similarity produced a strong match (soft_score >= 0.9), treat
    # it as a semantic hit rather than a miss.
    embedding_strong_hit = top_embedding_used and top_soft >= 0.9
    no_temporal_hit = fallback_used and top_lexical <= 0.0 and not embedding_strong_hit

    # low_lexical_score: top segment matched mostly via soft/embedding similarity.
    low_lexical_score = top_lexical < 0.5

    # relation_empty_on_top: top segment carries no temporal-relation evidence.
    relation_empty_on_top = len(matched_relations) == 0

    # weak_match: overall match confidence is low.
    weak_match = (
        no_temporal_hit
        or (low_lexical_score and relation_empty_on_top and len(matched_entities) == 0)
        or top_score < 0.5
    )

    # ── spatial coarseness signal ─────────────────────────────────────────────
    t_start = float(top.get("t_start", 0.0) or 0.0)
    t_end = float(top.get("t_end", t_start) or t_start)
    segment_width_sec = t_end - t_start
    # Stage-2 windows are nominally 30 s; anything >= 25 s is "coarse".
    coarse_window = segment_width_sec >= 25.0

    # ── derived action signals ────────────────────────────────────────────────
    # recommend_frame_inspect: agent should call frame_inspect_tool before finish.
    recommend_frame_inspect = (
        question_is_fine_detail
        or no_temporal_hit
        or (low_lexical_score and relation_empty_on_top)
    )

    # localize_only: graph result is useful for time-localization only, not as a
    # terminal answer — the agent must confirm with frame_inspect_tool.
    localize_only = (
        weak_match
        or question_is_fine_detail
        or (coarse_window and recommend_frame_inspect)
    )

    return {
        # match quality
        "no_temporal_hit": no_temporal_hit,
        "low_lexical_score": low_lexical_score,
        "relation_empty_on_top": relation_empty_on_top,
        "weak_match": weak_match,
        # spatial
        "coarse_window": coarse_window,
        "top_segment_width_sec": round(segment_width_sec, 1),
        # query type
        "question_is_fine_detail": question_is_fine_detail,
        # action guidance
        "recommend_frame_inspect": recommend_frame_inspect,
        "localize_only": localize_only,
        # raw scores for transparency
        "top_segment_score": round(top_score, 4),
        "top_segment_lexical": round(top_lexical, 4),
        # embedding info
        "embedding_used": top_embedding_used,
        "embedding_strong_hit": embedding_strong_hit,
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_query_embedding(query: str) -> list[float] | None:
    """Embed query text via the configured embedding service. Returns None on failure."""
    try:
        from sgd.utils import AzureOpenAIEmbeddingService  # lazy import

        results = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=[query],
            api_key=config.OPENAI_API_KEY,
        )
        if results and results[0].get("embedding"):
            return results[0]["embedding"]
    except Exception as exc:
        print(f"[WARN] Query embedding failed, falling back to fuzzy match: {exc}")
    return None


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


def _anchor_target_sec(anchor: str, max_t_sec: float) -> float:
    if max_t_sec <= 0:
        return 0.0
    if anchor == "start":
        return 0.0
    if anchor == "end":
        return max_t_sec
    return max_t_sec * 0.5


def _span_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(a0, b0)
    right = min(a1, b1)
    if right <= left:
        return 0.0
    inter = right - left
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _span_distance_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    if _span_overlap(a0, a1, b0, b1) > 0:
        return 0.0
    return min(abs(a0 - b1), abs(b0 - a1))


def _temporal_prior_score(query_span: tuple[float, float], seg_span: tuple[float, float]) -> float:
    q0, q1 = query_span
    s0, s1 = seg_span
    iou = _span_overlap(q0, q1, s0, s1)
    if iou > 0:
        # Favor true overlap strongly for explicit time-range questions.
        return 1.0 + iou * 4.0
    dist = _span_distance_sec(q0, q1, s0, s1)
    # Keep non-overlap bonus small to avoid over-promoting far segments.
    return 0.4 / (1.0 + dist / 25.0)


def _temporal_adjustment(
    seg_start: float,
    seg_end: float,
    hint: dict[str, Any],
) -> tuple[float, float, bool]:
    span = hint.get("span")
    if span is not None:
        iou = _span_overlap(seg_start, seg_end, span[0], span[1])
        if iou > 0:
            return 3.2 + iou * 2.0, 0.0, True
        d = _span_distance_sec(seg_start, seg_end, span[0], span[1])
        delta = 0.6 / (1.0 + d / 20.0) - min(2.8, d / 70.0)
        return delta, d, False

    anchor = hint.get("anchor")
    max_t_sec = float(hint.get("max_t_sec", 0.0))
    if anchor:
        target = _anchor_target_sec(anchor, max_t_sec)
        mid = 0.5 * (seg_start + seg_end)
        d = abs(mid - target)
        norm = d / max(max_t_sec, 1.0)
        delta = 0.9 - min(1.2, norm * 1.6)
        return delta, d, False

    return 0.0, 0.0, False


def _select_with_time_diversity(
    candidates: list[dict[str, Any]],
    top_k: int,
    diversity_window_sec: float = 90.0,
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
            mid = 0.5 * (float(item.get("t_start", 0.0)) + float(item.get("t_end", 0.0)))
            penalty = 0.0
            for picked in selected:
                pmid = 0.5 * (float(picked.get("t_start", 0.0)) + float(picked.get("t_end", 0.0)))
                dt = abs(mid - pmid)
                if dt < diversity_window_sec:
                    penalty = max(penalty, (diversity_window_sec - dt) / diversity_window_sec)
            adjusted = score - diversity_penalty * penalty
            if adjusted > best_val:
                best_val = adjusted
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def _candidate_unique_key(item: dict[str, Any]) -> tuple[Any, ...]:
    seg_id = item.get("segment_id")
    if seg_id is not None:
        return ("segment_id", seg_id)
    return (
        "span",
        float(item.get("t_start", 0.0)),
        float(item.get("t_end", 0.0)),
        str(item.get("summary", "")),
    )


def _reorder_top1_candidate(
    ranked_segments: list[dict[str, Any]],
    temporal_hint: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(ranked_segments) <= 1:
        return ranked_segments

    # Respect explicit time span sorting.
    if temporal_hint.get("span") is not None:
        return ranked_segments

    anchor = temporal_hint.get("anchor")
    if anchor == "start":
        idx = min(range(len(ranked_segments)), key=lambda i: float(ranked_segments[i].get("t_start", 0.0)))
    elif anchor == "end":
        idx = max(range(len(ranked_segments)), key=lambda i: float(ranked_segments[i].get("t_end", 0.0)))
    elif anchor == "middle":
        max_t = max(float(s.get("t_end", 0.0)) for s in ranked_segments)
        target = 0.5 * max_t
        idx = min(
            range(len(ranked_segments)),
            key=lambda i: abs(
                0.5
                * (
                    float(ranked_segments[i].get("t_start", 0.0))
                    + float(ranked_segments[i].get("t_end", 0.0))
                )
                - target
            ),
        )
    else:
        idx = 0
        top = ranked_segments[0]
        top_lex = float(top.get("lexical_score", 0.0) or 0.0)
        top_ent = len(top.get("matched_entities", []) or [])
        top_rel = len(top.get("matched_relations", []) or [])
        # Soft-only ties are unstable; prefer earlier candidates within small score margin.
        if top_lex <= 0.0 and top_ent == 0 and top_rel == 0:
            top_score = float(top.get("score", 0.0) or 0.0)
            tie_margin = 0.03
            tie_indices = [
                i
                for i, seg in enumerate(ranked_segments)
                if top_score - float(seg.get("score", 0.0) or 0.0) <= tie_margin
            ]
            if tie_indices:
                idx = min(tie_indices, key=lambda i: float(ranked_segments[i].get("t_start", 0.0)))

    if idx <= 0:
        return ranked_segments
    return [ranked_segments[idx], *ranked_segments[:idx], *ranked_segments[idx + 1 :]]


def _segment_text(seg: dict[str, Any]) -> str:
    fields = [str(seg.get("summary", ""))]
    for cap in seg.get("caption_snippets", []) or []:
        fields.append(str(cap))
    ent_name_by_id: dict[str, str] = {}
    for ent in seg.get("active_entities", []):
        ent_id = str(ent.get("entity_id", ""))
        ent_name = str(ent.get("canonical_name", ""))
        if ent_id and ent_name:
            ent_name_by_id[ent_id] = ent_name
        fields.append(ent_name)
        fields.append(str(ent.get("type", "")))
    for rel in seg.get("temporal_relations", []):
        s_id = str(rel.get("s", ""))
        o_id = str(rel.get("o", ""))
        s_name = str(rel.get("s_name", ent_name_by_id.get(s_id, "")))
        o_name = str(rel.get("o_name", ent_name_by_id.get(o_id, "")))
        fields.extend(
            [
                s_name,
                str(rel.get("p", "")),
                o_name,
            ]
        )
    return " ".join(f for f in fields if f)


def _fuzzy_token_overlap(query_tokens: set[str], seg_tokens: set[str]) -> float:
    if not query_tokens or not seg_tokens:
        return 0.0
    score = 0.0
    for q in query_tokens:
        best = 0.0
        for s in seg_tokens:
            if q == s:
                best = 1.0
                break
            sim = SequenceMatcher(None, q, s).ratio()
            if sim > best:
                best = sim
        if best >= 0.9:
            score += 1.0
        elif best >= 0.8:
            score += 0.6
        elif best >= 0.7:
            score += 0.3
    return score


def query_stage2_segment_graph(
    graph: dict[str, Any],
    query: str,
    time_reference: str | None = None,
    top_k_segments: int = 6,
    max_triples: int = 24,
    min_soft_score: float = 0.35,
    min_fallback_score: float = 0.3,
    fallback_relative_floor: float = 0.25,
    fallback_top_k: int | None = None,
) -> dict[str, Any]:
    query_tokens = _tokenize(query)
    segments = graph.get("segments", [])
    segment_bounds = [
        (
            float(seg.get("t_start", 0.0)),
            float(seg.get("t_end", float(seg.get("t_start", 0.0)))),
        )
        for seg in segments
    ]
    max_t_sec = max((b[1] for b in segment_bounds), default=0.0)
    temporal_source = query if not time_reference else f"{query} {time_reference}"
    temporal_hint = _infer_temporal_hint(query=temporal_source, max_t_sec=max_t_sec)
    query_span = temporal_hint.get("span")
    ranked_segments = []
    fallback_candidates = []
    all_soft_candidates = []

    # ── embedding retrieval: compute query embedding once if segments have embeddings ──
    _has_embeddings = any(seg.get("embedding") for seg in segments)
    _query_embedding: list[float] | None = None
    if _has_embeddings:
        _query_embedding = _get_query_embedding(query)

    for seg in segments:
        lexical_score = 0.0
        soft_score = 0.0
        matched_entities = []
        matched_relations = []
        ent_name_by_id: dict[str, str] = {}

        seg_text = _segment_text(seg)
        summary_tokens = _tokenize(seg.get("summary", ""))
        caption_tokens = _tokenize(" ".join(str(x) for x in (seg.get("caption_snippets", []) or [])))
        lexical_score += float(len(query_tokens & summary_tokens)) * 0.8
        lexical_score += float(len(query_tokens & caption_tokens)) * 0.25

        for ent in seg.get("active_entities", []):
            ent_id = str(ent.get("entity_id", ""))
            ent_name = str(ent.get("canonical_name", ""))
            if ent_id and ent_name:
                ent_name_by_id[ent_id] = ent_name
            ent_text = " ".join(
                [
                    ent_name,
                    str(ent.get("type", "")),
                ]
            )
            overlap = len(query_tokens & _tokenize(ent_text))
            if overlap > 0:
                lexical_score += float(overlap) + float(ent.get("confidence", 0.0)) * 0.2
                matched_entities.append(
                    {
                        "entity_id": ent.get("entity_id"),
                        "canonical_name": ent.get("canonical_name", ""),
                        "type": ent.get("type", ""),
                        "confidence": ent.get("confidence", 0.0),
                    }
                )

        for rel in seg.get("temporal_relations", []):
            s_id = str(rel.get("s", ""))
            o_id = str(rel.get("o", ""))
            s_name = str(rel.get("s_name", ent_name_by_id.get(s_id, "")))
            o_name = str(rel.get("o_name", ent_name_by_id.get(o_id, "")))
            rel_text = " ".join(
                [
                    str(rel.get("p", "")),
                    s_name,
                    o_name,
                ]
            )
            overlap = len(query_tokens & _tokenize(rel_text))
            if overlap > 0:
                lexical_score += float(overlap) + float(rel.get("confidence", 0.0)) * 0.2
                rel_out = dict(rel)
                rel_out["s_name"] = s_name
                rel_out["o_name"] = o_name
                matched_relations.append(rel_out)

        seg_embedding = seg.get("embedding")
        if _query_embedding and seg_embedding:
            # Embedding-based semantic similarity (much stronger than SequenceMatcher)
            cos_sim = _cosine_similarity(_query_embedding, seg_embedding)
            # Scale to comparable range: cosine [0,1] → soft_score ~[0, 1.5]
            soft_score += max(0.0, cos_sim) * 1.5
        else:
            # Fallback to fuzzy string matching when no embeddings
            seg_tokens = _tokenize(seg_text)
            soft_score += _fuzzy_token_overlap(query_tokens, seg_tokens) * 0.35
            soft_score += SequenceMatcher(None, query.lower(), seg_text.lower()).ratio() * 0.6

        s0 = float(seg.get("t_start", 0.0))
        s1 = float(seg.get("t_end", s0))
        base_score = lexical_score + soft_score
        temporal_delta, temporal_distance_sec, temporal_in_target = _temporal_adjustment(
            seg_start=s0,
            seg_end=s1,
            hint=temporal_hint,
        )
        final_score = base_score + temporal_delta

        payload = {
            "segment_id": seg.get("segment_id"),
            "t_start": seg.get("t_start"),
            "t_end": seg.get("t_end"),
            "score": round(final_score, 4),
            "base_score": round(base_score, 4),
            "temporal_score": round(temporal_delta, 4),
            "temporal_distance_sec": round(temporal_distance_sec, 3),
            "temporal_in_target": temporal_in_target,
            "lexical_score": round(lexical_score, 4),
            "soft_score": round(soft_score, 4),
            "summary": seg.get("summary", ""),
            "matched_entities": matched_entities,
            "matched_relations": matched_relations,
            "support_frame_ids": seg.get("support_frame_ids", []),
            "embedding_used": bool(_query_embedding and seg_embedding),
        }

        all_soft_candidates.append(payload)
        if lexical_score > 0:
            ranked_segments.append(payload)
        elif soft_score >= min_soft_score:
            fallback_candidates.append(payload)

    def _sort_key(x: dict[str, Any]) -> tuple[float, float, float]:
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
                -float(x.get("t_start", 0.0)),
            )
        return (
            float(x.get("score", 0.0)),
            -float(x.get("t_start", 0.0)),
            -float(x.get("t_end", 0.0)),
        )

    ranked_segments.sort(key=_sort_key, reverse=True)
    if not ranked_segments:
        fallback_candidates.sort(key=_sort_key, reverse=True)
        all_soft_candidates.sort(key=_sort_key, reverse=True)
        use_top_k = fallback_top_k if fallback_top_k is not None else top_k_segments
        if fallback_candidates:
            candidates = fallback_candidates
        else:
            candidates = all_soft_candidates

        if query_span is not None and candidates:
            span_len = max(1.0, query_span[1] - query_span[0])
            dist_cap = max(60.0, span_len * 2.5)
            near = [c for c in candidates if float(c.get("temporal_distance_sec", 1e9)) <= dist_cap]
            if near:
                candidates = near

        if candidates:
            max_score = float(candidates[0].get("score", 0.0))
            floor = max(min_fallback_score, max_score * fallback_relative_floor)
            ranked_segments = [c for c in candidates if float(c.get("score", 0.0)) >= floor]
            if not ranked_segments:
                ranked_segments = candidates[:1]
            if query_span is None and temporal_hint.get("anchor") is None:
                ranked_segments = _select_with_time_diversity(ranked_segments, use_top_k)
            else:
                ranked_segments = ranked_segments[:use_top_k]
    else:
        if query_span is None and temporal_hint.get("anchor") is None:
            ranked_segments = _select_with_time_diversity(ranked_segments, top_k_segments)
        else:
            ranked_segments = ranked_segments[:top_k_segments]

    # Keep lexical-first ordering, but always backfill to top-k when possible.
    # This raises recall@k for sparse segment graphs where lexical matches are few.
    if top_k_segments > 0 and len(ranked_segments) < top_k_segments:
        backup_pool = fallback_candidates + all_soft_candidates
        backup_pool.sort(key=_sort_key, reverse=True)
        if query_span is not None and backup_pool:
            span_len = max(1.0, query_span[1] - query_span[0])
            dist_cap = max(60.0, span_len * 2.5)
            near = [c for c in backup_pool if float(c.get("temporal_distance_sec", 1e9)) <= dist_cap]
            if near:
                backup_pool = near
        if query_span is None and temporal_hint.get("anchor") is None:
            backup_pool = _select_with_time_diversity(backup_pool, len(backup_pool))

        seen = {_candidate_unique_key(item) for item in ranked_segments}
        for cand in backup_pool:
            key = _candidate_unique_key(cand)
            if key in seen:
                continue
            ranked_segments.append(cand)
            seen.add(key)
            if len(ranked_segments) >= top_k_segments:
                break

    ranked_segments = _reorder_top1_candidate(ranked_segments=ranked_segments, temporal_hint=temporal_hint)

    flat_rel = []
    for seg in ranked_segments:
        for rel in seg["matched_relations"]:
            flat_rel.append(
                {
                    "segment_id": seg["segment_id"],
                    "t_start": rel.get("t_start"),
                    "t_end": rel.get("t_end"),
                    "s": rel.get("s"),
                    "p": rel.get("p"),
                    "o": rel.get("o"),
                    "confidence": rel.get("confidence", 0.0),
                }
            )
    flat_rel = sorted(flat_rel, key=lambda x: x["confidence"], reverse=True)[:max_triples]

    fallback_used = bool(
        ranked_segments and all(float(s.get("lexical_score", 0.0)) <= 0 for s in ranked_segments)
    )
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
        "matched_segments": ranked_segments,
        "top_temporal_triples": flat_rel,
        "fallback_used": fallback_used,
        "controller_signals": _compute_controller_signals(
            query=query,
            ranked_segments=ranked_segments,
            fallback_used=fallback_used,
        ),
    }


def query_stage2_segment_graph_by_path(
    graph_path: str,
    query: str,
    time_reference: str | None = None,
    top_k_segments: int = 6,
    max_triples: int = 24,
) -> dict[str, Any]:
    graph = safe_load_stage2_segment_graph(graph_path)
    if graph is None:
        raise ValueError(f"Invalid or missing Stage-2 segment graph: {graph_path}")
    return query_stage2_segment_graph(
        graph=graph,
        query=query,
        time_reference=time_reference,
        top_k_segments=top_k_segments,
        max_triples=max_triples,
    )


def result_to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
