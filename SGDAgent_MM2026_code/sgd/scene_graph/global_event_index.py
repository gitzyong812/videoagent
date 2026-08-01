"""Legacy compact timeline formatter kept for artifact compatibility.

The paper-aligned global graph is constructed by ``global_builder`` and queried
through ``query_global``. New runs do not inject this legacy summary into the
agent prompt.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Any

from sgd.scene_graph.schema import safe_load_stage2_segment_graph


# ---------------------------------------------------------------------------
# Configuration defaults (overridable via sgd.config)
# ---------------------------------------------------------------------------
_DEFAULT_MAX_EVENTS = 12  # target scene count; aggressive compression
_DEFAULT_MAX_CHARS = 2400  # hard cap on the text summary length
_DEFAULT_MERGE_GAP_SEC = 5.0  # merge adjacent segments sharing entities within this gap
_DEFAULT_MAX_ENTITIES_DISPLAY = 8  # entity continuity entries

# Stop words to filter from discriminative keyword extraction
_STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "and", "but", "or", "if", "this",
    "that", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "what", "which", "who", "whom",
    # generic video/scene terms
    "scene", "video", "frame", "clip", "segment", "person", "people",
    "man", "woman", "one", "two", "time", "camera", "screen", "image",
    "background", "foreground", "left", "right", "side", "area",
}


def _format_ts(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = max(0, int(round(seconds)))
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _tokenize_for_keywords(text: str) -> list[str]:
    """Extract non-stop-word tokens >= 3 chars from text.

    Filters out entity IDs (ent_XXXXXX pattern) and graph-schema noise words
    like 'none', 'entities', 'predicates'.
    """
    tokens = re.findall(r"[a-z][a-z0-9_]{2,}", (text or "").lower())
    return [
        tok for tok in tokens
        if tok not in _STOP_WORDS
        and not re.match(r"ent_\d+", tok)  # filter entity IDs
        and tok not in {"none", "entities", "predicates", "segment", "seg"}
    ]


# ---------------------------------------------------------------------------
# Entity continuity map
# ---------------------------------------------------------------------------

def _build_entity_timeline(segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Map each entity_id to its sorted list of segment appearances."""
    timeline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        t_start = float(seg.get("t_start", 0))
        t_end = float(seg.get("t_end", t_start))
        for ent in seg.get("active_entities", []):
            eid = ent.get("entity_id", "")
            if not eid:
                continue
            timeline[eid].append({
                "t_start": t_start,
                "t_end": t_end,
                "canonical_name": ent.get("canonical_name", eid),
                "type": ent.get("type", ""),
            })
    for eid in timeline:
        timeline[eid].sort(key=lambda x: x["t_start"])
    return dict(timeline)


def _merge_entity_spans(
    appearances: list[dict[str, Any]],
    merge_gap_sec: float,
) -> list[tuple[float, float]]:
    """Merge consecutive appearances into contiguous spans."""
    if not appearances:
        return []
    spans: list[tuple[float, float]] = []
    cur_start = appearances[0]["t_start"]
    cur_end = appearances[0]["t_end"]
    for app in appearances[1:]:
        if app["t_start"] <= cur_end + merge_gap_sec:
            cur_end = max(cur_end, app["t_end"])
        else:
            spans.append((cur_start, cur_end))
            cur_start = app["t_start"]
            cur_end = app["t_end"]
    spans.append((cur_start, cur_end))
    return spans


# ---------------------------------------------------------------------------
# Segment → Scene block clustering (aggressive compression)
# ---------------------------------------------------------------------------

def _collect_segment_text(seg: dict[str, Any]) -> str:
    """Gather all text signals from a segment for keyword extraction."""
    parts: list[str] = []
    summary = seg.get("summary", "")
    if summary:
        parts.append(summary)
    for cap in seg.get("caption_snippets", [])[:3]:
        if isinstance(cap, str) and cap.strip():
            parts.append(cap.strip())
    for ent in seg.get("active_entities", [])[:6]:
        name = ent.get("canonical_name", "")
        if name:
            parts.append(name)
    for rel in seg.get("temporal_relations", [])[:6]:
        s_name = rel.get("s_name", rel.get("s", ""))
        p = rel.get("p", "")
        o_name = rel.get("o_name", rel.get("o", ""))
        if s_name and p and o_name:
            parts.append(f"{s_name} {p} {o_name}")
    return " ".join(parts)


def _entity_set(seg: dict[str, Any]) -> set[str]:
    return {
        ent.get("entity_id", "")
        for ent in seg.get("active_entities", [])
        if ent.get("entity_id")
    }


def _predicate_set(seg: dict[str, Any]) -> set[str]:
    return {
        rel.get("p", "")
        for rel in seg.get("temporal_relations", [])
        if rel.get("p")
    }


def _cluster_similarity(seg_a: dict[str, Any], seg_b: dict[str, Any]) -> float:
    """Compute a 0-1 similarity between two segments based on entity + predicate overlap."""
    ent_a, ent_b = _entity_set(seg_a), _entity_set(seg_b)
    pred_a, pred_b = _predicate_set(seg_a), _predicate_set(seg_b)

    ent_union = ent_a | ent_b
    pred_union = pred_a | pred_b

    ent_sim = len(ent_a & ent_b) / len(ent_union) if ent_union else 0.0
    pred_sim = len(pred_a & pred_b) / len(pred_union) if pred_union else 0.0

    # Entity overlap matters more (0.7) than predicate overlap (0.3)
    return ent_sim * 0.7 + pred_sim * 0.3


def _identify_ubiquitous_entities(
    segments: list[dict[str, Any]],
    threshold: float = 0.4,
) -> set[str]:
    """Find entity_ids that appear in more than `threshold` fraction of segments.

    These are too generic to drive clustering (e.g. "person", "buildings")
    and would chain all segments into one mega-cluster.
    """
    if not segments:
        return set()
    entity_seg_count: dict[str, int] = defaultdict(int)
    for seg in segments:
        seen: set[str] = set()
        for ent in seg.get("active_entities", []):
            eid = ent.get("entity_id", "")
            if eid and eid not in seen:
                entity_seg_count[eid] += 1
                seen.add(eid)
    cutoff = len(segments) * threshold
    return {eid for eid, count in entity_seg_count.items() if count >= cutoff}


def _discriminative_entity_set(seg: dict[str, Any], ubiquitous: set[str]) -> set[str]:
    """Entity IDs from segment, excluding ubiquitous ones."""
    return {
        ent.get("entity_id", "")
        for ent in seg.get("active_entities", [])
        if ent.get("entity_id") and ent.get("entity_id") not in ubiquitous
    }


def _extract_event_timeline(
    segments: list[dict[str, Any]],
    merge_gap_sec: float,
    max_events: int,
) -> list[dict[str, Any]]:
    """Extract a compressed event timeline from Stage 2 segments.

    Strategy: sweep left-to-right, merging adjacent segments when they share
    DISCRIMINATIVE entities (not ubiquitous ones like "person") AND are
    temporally close.  Then iteratively merge the most similar neighboring
    pair until we hit max_events.

    This achieves real compression (120+ segs → ~12 scene blocks) while
    avoiding the mega-cluster problem where a single generic entity chains
    everything together.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: float(s.get("t_start", 0)))

    # Identify entities too common to be discriminative
    ubiquitous = _identify_ubiquitous_entities(sorted_segs)

    # Target max span per cluster — prevents runaway chaining
    total_span = float(sorted_segs[-1].get("t_end", 0)) - float(sorted_segs[0].get("t_start", 0))
    max_cluster_span = max(60.0, total_span / max(max_events, 1) * 2.0)

    # Phase 1: initial clustering — greedily merge adjacent segments
    # that share DISCRIMINATIVE entities, are temporally close,
    # and don't exceed max_cluster_span.
    clusters: list[list[dict[str, Any]]] = [[sorted_segs[0]]]
    for seg in sorted_segs[1:]:
        prev_cluster = clusters[-1]
        prev_start = min(float(s.get("t_start", 0)) for s in prev_cluster)
        prev_end = max(float(s.get("t_end", 0)) for s in prev_cluster)
        seg_start = float(seg.get("t_start", 0))
        seg_end = float(seg.get("t_end", 0))

        # Check temporal proximity
        if seg_start > prev_end + merge_gap_sec:
            clusters.append([seg])
            continue

        # Check cluster wouldn't exceed max span
        if seg_end - prev_start > max_cluster_span:
            clusters.append([seg])
            continue

        # Check discriminative entity overlap (fall back to predicate overlap)
        disc_overlap = _discriminative_entity_set(seg, ubiquitous) & _discriminative_entity_set(prev_cluster[-1], ubiquitous)
        pred_overlap = _predicate_set(seg) & _predicate_set(prev_cluster[-1])
        if disc_overlap or pred_overlap:
            prev_cluster.append(seg)
        else:
            clusters.append([seg])

    # Phase 2: iteratively merge the most-similar neighboring pair
    # until we reach target count.  Enforce max_cluster_span to prevent
    # mega-clusters that destroy navigational value.
    while len(clusters) > max_events and len(clusters) > 1:
        best_idx = -1
        best_score = -1.0
        for i in range(len(clusters) - 1):
            # Check combined span wouldn't exceed limit
            left_start = min(float(s.get("t_start", 0)) for s in clusters[i])
            right_end = max(float(s.get("t_end", 0)) for s in clusters[i + 1])
            if right_end - left_start > max_cluster_span:
                continue  # skip: would create a mega-cluster

            # Score = content similarity, penalized by temporal gap
            last_of_left = clusters[i][-1]
            first_of_right = clusters[i + 1][0]
            sim = _cluster_similarity(last_of_left, first_of_right)
            gap = float(first_of_right.get("t_start", 0)) - max(
                float(s.get("t_end", 0)) for s in clusters[i]
            )
            gap_penalty = 1.0 / (1.0 + gap / 60.0)  # decay over 1 minute
            score = sim * 0.6 + gap_penalty * 0.4
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break  # no mergeable pair found (all would exceed span limit)
        clusters[best_idx].extend(clusters[best_idx + 1])
        clusters.pop(best_idx + 1)

    # Phase 3: summarize each cluster
    events: list[dict[str, Any]] = []
    # Collect global keyword frequencies to find discriminative ones per block
    global_kw_freq: dict[str, int] = defaultdict(int)
    cluster_keywords: list[dict[str, int]] = []
    for cluster in clusters:
        kw_freq: dict[str, int] = defaultdict(int)
        for seg in cluster:
            text = _collect_segment_text(seg)
            for tok in _tokenize_for_keywords(text):
                kw_freq[tok] += 1
                global_kw_freq[tok] += 1
        cluster_keywords.append(dict(kw_freq))

    num_clusters = len(clusters)
    for idx, cluster in enumerate(clusters):
        t_start = min(float(s.get("t_start", 0)) for s in cluster)
        t_end = max(float(s.get("t_end", 0)) for s in cluster)

        # Top entities by frequency within cluster
        entity_counts: dict[str, tuple[str, int]] = {}
        for seg in cluster:
            for ent in seg.get("active_entities", []):
                eid = ent.get("entity_id", "")
                name = ent.get("canonical_name", eid)
                if eid:
                    prev = entity_counts.get(eid, (name, 0))
                    entity_counts[eid] = (name, prev[1] + 1)
        top_entities = sorted(entity_counts.values(), key=lambda x: -x[1])[:3]

        # Top predicates
        pred_counts: dict[str, int] = defaultdict(int)
        for seg in cluster:
            for rel in seg.get("temporal_relations", []):
                p = rel.get("p", "")
                if p:
                    pred_counts[p] += 1
        top_preds = sorted(pred_counts, key=lambda k: -pred_counts[k])[:3]

        # Discriminative keywords: high local TF, lower global DF
        kw_freq = cluster_keywords[idx]
        scored_kw: list[tuple[str, float]] = []
        for kw, local_count in kw_freq.items():
            global_count = global_kw_freq.get(kw, 1)
            # TF-IDF-like: local frequency * inverse document frequency
            idf = math.log(max(num_clusters, 1) / max(global_count, 1) + 1.0)
            scored_kw.append((kw, local_count * idf))
        scored_kw.sort(key=lambda x: -x[1])
        # Pick top keywords not already covered by entity names or predicates
        existing = set()
        for name, _ in top_entities:
            existing.update(_tokenize_for_keywords(name))
        for p in top_preds:
            existing.update(_tokenize_for_keywords(p))
        disc_keywords = [kw for kw, _ in scored_kw if kw not in existing][:4]

        entity_str = ", ".join(name for name, _ in top_entities) if top_entities else "—"
        pred_str = ", ".join(top_preds) if top_preds else "—"

        events.append({
            "t_start": t_start,
            "t_end": t_end,
            "entities": entity_str,
            "actions": pred_str,
            "keywords": disc_keywords,
            "num_segments": len(cluster),
        })

    return events


# ---------------------------------------------------------------------------
# Entity continuity summary
# ---------------------------------------------------------------------------

def _entity_continuity_summary(
    entity_timeline: dict[str, list[dict[str, Any]]],
    merge_gap_sec: float,
    max_entities: int = 8,
) -> list[dict[str, str]]:
    """Summarize when each major entity appears across the video."""
    ranked = sorted(entity_timeline.items(), key=lambda kv: -len(kv[1]))[:max_entities]
    results = []
    for eid, appearances in ranked:
        if not appearances:
            continue
        name = appearances[0].get("canonical_name", eid)
        etype = appearances[0].get("type", "")
        spans = _merge_entity_spans(appearances, merge_gap_sec)
        span_strs = [f"{_format_ts(s)}-{_format_ts(e)}" for s, e in spans]
        results.append({
            "name": name,
            "type": etype,
            "spans": ", ".join(span_strs),
        })
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_global_event_index(
    stage2_graph: dict[str, Any],
    *,
    max_events: int | None = None,
    max_chars: int | None = None,
    merge_gap_sec: float | None = None,
) -> dict[str, Any]:
    """Build a Global Event Index from a Stage 2 segment graph.

    Returns a dict with:
      - "event_timeline": list of simplified events
      - "entity_continuity": entity appearance spans
      - "prompt_text": pre-formatted text for prompt injection
    """
    try:
        import sgd.config as cfg
    except ImportError:
        cfg = None  # type: ignore[assignment]

    if max_events is None:
        max_events = int(getattr(cfg, "GEI_MAX_EVENTS", _DEFAULT_MAX_EVENTS))
    if max_chars is None:
        max_chars = int(getattr(cfg, "GEI_MAX_CHARS", _DEFAULT_MAX_CHARS))
    if merge_gap_sec is None:
        merge_gap_sec = float(getattr(cfg, "GEI_MERGE_GAP_SEC", _DEFAULT_MERGE_GAP_SEC))

    segments = stage2_graph.get("segments", [])
    if not segments:
        return {"event_timeline": [], "entity_continuity": [], "prompt_text": ""}

    entity_timeline = _build_entity_timeline(segments)
    events = _extract_event_timeline(segments, merge_gap_sec, max_events)
    continuity = _entity_continuity_summary(entity_timeline, merge_gap_sec)

    prompt_text = _format_prompt_text(events, continuity, max_chars)

    return {
        "event_timeline": events,
        "entity_continuity": continuity,
        "prompt_text": prompt_text,
    }


def build_global_event_index_from_path(
    stage2_graph_path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a Stage 2 graph from disk and build the GEI."""
    graph = safe_load_stage2_segment_graph(stage2_graph_path)
    if graph is None:
        return {"event_timeline": [], "entity_continuity": [], "prompt_text": ""}
    return build_global_event_index(graph, **kwargs)


# ---------------------------------------------------------------------------
# Prompt text formatting
# ---------------------------------------------------------------------------

def _format_prompt_text(
    events: list[dict[str, Any]],
    continuity: list[dict[str, str]],
    max_chars: int,
) -> str:
    """Format the GEI as concise text for prompt injection.

    The output is designed to maximise the agent's ability to match question
    keywords to time regions, so every event line includes discriminative
    keywords alongside entities and predicates.
    """
    if not events and not continuity:
        return ""

    lines: list[str] = []

    if events:
        lines.append("VIDEO TIMELINE (scene index — use to orient your tool calls):")
        for i, ev in enumerate(events, 1):
            ts = f"{_format_ts(ev['t_start'])}-{_format_ts(ev['t_end'])}"
            parts = [ev["entities"]]
            if ev.get("actions") and ev["actions"] != "—":
                parts.append(ev["actions"])
            kws = ev.get("keywords", [])
            if kws:
                parts.append(" ".join(kws))
            desc = " | ".join(parts)
            segs = ev.get("num_segments", 0)
            line = f"  {i}. [{ts}] ({segs}seg) {desc}"
            lines.append(line)

    if continuity:
        lines.append("RECURRING ENTITIES:")
        for ent in continuity:
            label = ent["name"]
            if ent.get("type"):
                label += f" ({ent['type']})"
            lines.append(f"  - {label}: {ent['spans']}")

    text = "\n".join(lines)

    # Enforce character budget — trim from the bottom (entity section first)
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."

    return text
