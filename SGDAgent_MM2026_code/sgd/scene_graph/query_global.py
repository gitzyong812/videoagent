from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from typing import Any

from sgd.scene_graph.schema import safe_load_global_event_graph


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(token) > 1}


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def _parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if len(parts) == 2:
        parts = ["0", *parts]
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {value}")
    hours, minutes, seconds = (float(part) for part in parts)
    return hours * 3600 + minutes * 60 + seconds


def normalize_scope(scope: list[tuple] | None) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for item in scope or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("scope must be a list of (start, end) timestamp pairs")
        start, end = _parse_timestamp(item[0]), _parse_timestamp(item[1])
        if end < start:
            start, end = end, start
        normalized.append((start, end))
    return normalized


def span_in_scope(start: float, end: float, scope: list[tuple[float, float]]) -> bool:
    return not scope or any(max(start, left) <= min(end, right) for left, right in scope)


def _time_reference_score(start: float, end: float, time_reference: str | None) -> float:
    text = (time_reference or "").lower()
    if not text:
        return 0.0
    marks = re.findall(r"(?:(\d+):)?(\d{1,2}):(\d{2})", text)
    if marks:
        seconds = [int(h or 0) * 3600 + int(m) * 60 + int(s) for h, m, s in marks]
        target_start, target_end = min(seconds), max(seconds)
        return 1.0 if max(start, target_start) <= min(end, target_end) else -1.0
    return 0.0


def query_global_event_graph(
    graph: dict[str, Any],
    query: str,
    scope: list[tuple] | None = None,
    time_reference: str | None = None,
    top_k_regions: int = 5,
    max_relations: int = 16,
) -> dict[str, Any]:
    query_tokens = _tokenize(query)
    scope_seconds = normalize_scope(scope)
    entities = {
        str(entity.get("entity_id")): entity
        for entity in graph.get("entities", [])
        if entity.get("entity_id")
    }
    relations_by_region: dict[str, list[dict[str, Any]]] = {}
    for relation in graph.get("relations", []):
        relations_by_region.setdefault(str(relation.get("region_id")), []).append(relation)

    query_embedding: list[float] | None = None
    if any(region.get("embedding") for region in graph.get("regions", [])):
        try:
            from sgd.scene_graph.query_segment import _get_query_embedding

            query_embedding = _get_query_embedding(query)
        except Exception:
            query_embedding = None

    ranked: list[dict[str, Any]] = []
    for region in graph.get("regions", []):
        start = float(region.get("t_start", 0.0))
        end = float(region.get("t_end", start))
        if not span_in_scope(start, end, scope_seconds):
            continue
        region_relations = relations_by_region.get(str(region.get("region_id")), [])
        text_parts: list[str] = []
        matched_relations: list[dict[str, Any]] = []
        matched_entities: dict[str, dict[str, Any]] = {}
        for relation in region_relations:
            participant_names = [
                str(entities.get(entity_id, {}).get("canonical_name", entity_id))
                for entity_id in relation.get("participant_entity_ids", [])
            ]
            relation_text = " ".join(
                [
                    str(relation.get("relation_type", "")),
                    str(relation.get("predicate", "")),
                    str(relation.get("description", "")),
                    *participant_names,
                ]
            )
            text_parts.append(relation_text)
            if query_tokens & _tokenize(relation_text):
                matched_relations.append(relation)
                for entity_id in relation.get("participant_entity_ids", []):
                    if entity_id in entities:
                        matched_entities[entity_id] = entities[entity_id]
        region_text = " ".join(text_parts)
        lexical_score = float(len(query_tokens & _tokenize(region_text)))
        if query_embedding and region.get("embedding"):
            soft_score = max(0.0, _cosine(query_embedding, region["embedding"]))
            embedding_used = True
        else:
            soft_score = SequenceMatcher(None, query.lower(), region_text.lower()).ratio()
            embedding_used = False
        temporal_score = _time_reference_score(start, end, time_reference)
        score = lexical_score + soft_score + temporal_score
        ranked.append(
            {
                "region_id": region.get("region_id"),
                "t_start": start,
                "t_end": end,
                "score": round(score, 4),
                "lexical_score": round(lexical_score, 4),
                "soft_score": round(soft_score, 4),
                "temporal_score": round(temporal_score, 4),
                "matched_entities": list(matched_entities.values()),
                "matched_relations": matched_relations,
                "support_segment_ids": region.get("support_segment_ids", []),
                "embedding_used": embedding_used,
            }
        )
    ranked.sort(key=lambda item: (float(item["score"]), -float(item["t_start"])), reverse=True)
    ranked = ranked[: max(0, top_k_regions)]
    top_relations = sorted(
        [relation for region in ranked for relation in region.get("matched_relations", [])],
        key=lambda relation: float(relation.get("confidence", 0.0))
        - float(relation.get("uncertainty", 0.0)),
        reverse=True,
    )[: max(0, max_relations)]
    top = ranked[0] if ranked else {}
    weak_match = not ranked or float(top.get("score", 0.0)) < 0.5 or not top.get("matched_relations")
    return {
        "query": query,
        "scope": scope,
        "time_reference": time_reference,
        "graph_version": graph.get("version"),
        "video_id": graph.get("video_id"),
        "matched_regions": ranked,
        "top_relations": top_relations,
        "fallback_used": bool(ranked and float(top.get("lexical_score", 0.0)) <= 0.0),
        "controller_signals": {
            "weak_match": weak_match,
            "localize_only": True,
            "recommend_frame_inspect": False,
            "recommended_next_action": "caption_backoff_tool" if weak_match else "segment_graph_query_tool",
            "top_region_score": round(float(top.get("score", 0.0)), 4),
        },
    }


def query_global_event_graph_by_path(
    graph_path: str,
    query: str,
    scope: list[tuple] | None = None,
    time_reference: str | None = None,
    top_k_regions: int = 5,
    max_relations: int = 16,
) -> dict[str, Any]:
    graph = safe_load_global_event_graph(graph_path)
    if graph is None:
        raise ValueError(f"Invalid or missing global event graph: {graph_path}")
    return query_global_event_graph(
        graph=graph,
        query=query,
        scope=scope,
        time_reference=time_reference,
        top_k_regions=top_k_regions,
        max_relations=max_relations,
    )


def result_to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
