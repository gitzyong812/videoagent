from __future__ import annotations

from collections import Counter
import json
import math
import os
from typing import Any

import sgd.config as config
from sgd.scene_graph.schema import (
    empty_global_event_graph,
    safe_load_global_event_graph,
    safe_load_stage1_graph,
    safe_load_stage2_segment_graph,
    validate_global_event_graph,
)


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _entity_ids(segment: dict[str, Any]) -> set[str]:
    return {
        str(entity.get("entity_id"))
        for entity in segment.get("active_entities", [])
        if entity.get("entity_id")
    }


def _relation_signatures(segment: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (str(rel.get("s")), str(rel.get("p")), str(rel.get("o")))
        for rel in segment.get("temporal_relations", [])
        if rel.get("s") and rel.get("p") and rel.get("o")
    }


def _predicates(segment: dict[str, Any]) -> set[str]:
    return {
        str(rel.get("p"))
        for rel in segment.get("temporal_relations", [])
        if rel.get("p")
    }


def _ubiquitous_generic_entity_ids(segments: list[dict[str, Any]]) -> set[str]:
    if not segments:
        return set()
    generic_names = {
        "person", "people", "man", "woman", "someone", "somebody",
        "object", "item", "thing",
    }
    counts = Counter(
        entity_id
        for segment in segments
        for entity_id in _entity_ids(segment)
    )
    names = {
        str(entity.get("entity_id")): str(
            entity.get("canonical_name", entity.get("name", ""))
        ).strip().lower()
        for segment in segments
        for entity in segment.get("active_entities", [])
        if entity.get("entity_id")
    }
    min_occurrences = max(3, math.ceil(len(segments) * 0.4))
    return {
        entity_id
        for entity_id, count in counts.items()
        if count >= min_occurrences and names.get(entity_id) in generic_names
    }


def group_candidate_regions(
    segments: list[dict[str, Any]],
    max_gap_sec: float,
    relation_overlap_threshold: float,
    predicate_overlap_threshold: float,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(segments, key=lambda item: float(item.get("t_start", 0.0)))
    if not ordered:
        return []
    ignored_entity_ids = _ubiquitous_generic_entity_ids(ordered)
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for segment in ordered[1:]:
        previous = groups[-1][-1]
        gap = float(segment.get("t_start", 0.0)) - float(previous.get("t_end", 0.0))
        relation_overlap = _jaccard(_relation_signatures(previous), _relation_signatures(segment))
        predicate_overlap = _jaccard(_predicates(previous), _predicates(segment))
        shares_entity = bool(
            (_entity_ids(previous) & _entity_ids(segment)) - ignored_entity_ids
        )
        should_merge = gap <= max_gap_sec and (
            relation_overlap >= relation_overlap_threshold
            or predicate_overlap >= predicate_overlap_threshold
            or shares_entity
        )
        if should_merge:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    return groups


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    value = (text or "").strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(value[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _region_messages(
    region_id: str,
    segments: list[dict[str, Any]],
    entity_bank: dict[str, dict[str, Any]],
    frame_bank: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    t_start = min(float(seg.get("t_start", 0.0)) for seg in segments)
    t_end = max(float(seg.get("t_end", t_start)) for seg in segments)
    segment_ids = [str(seg.get("segment_id")) for seg in segments if seg.get("segment_id")]
    active_ids = set().union(*(_entity_ids(seg) for seg in segments)) if segments else set()
    support_frame_ids = list(
        dict.fromkeys(
            str(frame_id)
            for segment in segments
            for frame_id in segment.get("support_frame_ids", [])
            if str(frame_id) in frame_bank
        )
    )
    if len(support_frame_ids) > 24:
        support_frame_ids = [
            support_frame_ids[round(index * (len(support_frame_ids) - 1) / 23)]
            for index in range(24)
        ]
    context = {
        "region_id": region_id,
        "time_range": [t_start, t_end],
        "entity_registry": [entity_bank[eid] for eid in sorted(active_ids) if eid in entity_bank],
        "supporting_frame_graphs": [
            {
                "frame_id": frame_bank[frame_id].get("frame_id"),
                "t_sec": frame_bank[frame_id].get("t_sec"),
                "entities": frame_bank[frame_id].get("entities", []),
                "relations": frame_bank[frame_id].get("relations", []),
            }
            for frame_id in support_frame_ids
        ],
        "segments": [
            {
                "segment_id": seg.get("segment_id"),
                "t_start": seg.get("t_start"),
                "t_end": seg.get("t_end"),
                "summary": seg.get("summary", ""),
                "caption_snippets": seg.get("caption_snippets", []),
                "active_entities": seg.get("active_entities", []),
                "temporal_relations": seg.get("temporal_relations", []),
            }
            for seg in segments
        ],
    }
    schema = {
        "relations": [
            {
                "relation_type": "event",
                "predicate": "prepares_meal",
                "participant_entity_ids": ["ent_000001"],
                "description": "A concise high-order event description.",
                "t_start": t_start,
                "t_end": t_end,
                "support_segment_ids": segment_ids,
                "confidence": 0.0,
                "uncertainty": 0.0,
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": "You construct a global video event graph and return strict JSON only.",
        },
        {
            "role": "user",
            "content": (
                "Abstract the supplied frame states and segment actions into high-order event, "
                "state_evolution, temporal, and causal relations. Candidate regions are evidence "
                "scopes, not event nodes. Reuse only supplied entity and segment IDs. Ground every "
                "relation in a bounded interval and supporting segments. Confidence and uncertainty "
                "must be numbers in [0,1]. Use snake_case predicates. "
                f"Return JSON matching: {json.dumps(schema, separators=(',', ':'))}\n"
                f"Context: {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]


def _call_global_vlm(
    region_id: str,
    segments: list[dict[str, Any]],
    entity_bank: dict[str, dict[str, Any]],
    frame_bank: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from sgd.utils import call_openai_model_with_tools

    attempts = max(1, int(getattr(config, "SCENE_GRAPH_GLOBAL_VLM_RETRIES", 3)))
    for _ in range(attempts):
        response = call_openai_model_with_tools(
            messages=_region_messages(region_id, segments, entity_bank, frame_bank),
            endpoints=config.SCENE_GRAPH_GLOBAL_VLM_ENDPOINT_LIST,
            model_name=config.SCENE_GRAPH_GLOBAL_VLM_MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0,
            max_tokens=int(getattr(config, "SCENE_GRAPH_GLOBAL_VLM_MAX_TOKENS", 1800)),
            return_json=True,
        )
        parsed = _extract_json_obj((response or {}).get("content", ""))
        if parsed is not None and isinstance(parsed.get("relations"), list) and parsed["relations"]:
            return parsed
    raise RuntimeError(f"Global VLM extraction failed for candidate region {region_id}.")


def _clip_unit(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _average_embedding(segments: list[dict[str, Any]]) -> list[float]:
    embeddings = [seg.get("embedding") for seg in segments if seg.get("embedding")]
    if not embeddings:
        return []
    dim = len(embeddings[0])
    compatible = [emb for emb in embeddings if len(emb) == dim]
    if not compatible:
        return []
    return [sum(float(emb[idx]) for emb in compatible) / len(compatible) for idx in range(dim)]


def _normalize_region_relations(
    raw_relations: list[dict[str, Any]],
    region_id: str,
    segments: list[dict[str, Any]],
    entity_bank: dict[str, dict[str, Any]],
    relation_offset: int,
) -> list[dict[str, Any]]:
    allowed_types = {"event", "state_evolution", "temporal", "causal"}
    t_start = min(float(seg.get("t_start", 0.0)) for seg in segments)
    t_end = max(float(seg.get("t_end", t_start)) for seg in segments)
    valid_segments = {
        str(seg.get("segment_id")) for seg in segments if seg.get("segment_id")
    }
    normalized: list[dict[str, Any]] = []
    for raw in raw_relations:
        if not isinstance(raw, dict):
            continue
        relation_type = str(raw.get("relation_type", "event")).strip().lower()
        if relation_type not in allowed_types:
            continue
        predicate = str(raw.get("predicate", "")).strip().lower().replace(" ", "_")
        description = str(raw.get("description", "")).strip()
        participants = [
            str(entity_id)
            for entity_id in raw.get("participant_entity_ids", [])
            if str(entity_id) in entity_bank
        ]
        support_segments = [
            str(segment_id)
            for segment_id in raw.get("support_segment_ids", [])
            if str(segment_id) in valid_segments
        ]
        if not predicate or not description or not participants or not support_segments:
            continue
        try:
            rel_start = max(t_start, float(raw.get("t_start", t_start)))
            rel_end = min(t_end, float(raw.get("t_end", t_end)))
        except (TypeError, ValueError):
            rel_start, rel_end = t_start, t_end
        if rel_end < rel_start:
            rel_start, rel_end = rel_end, rel_start
        normalized.append(
            {
                "relation_id": f"grel_{relation_offset + len(normalized) + 1:05d}",
                "region_id": region_id,
                "relation_type": relation_type,
                "predicate": predicate,
                "participant_entity_ids": list(dict.fromkeys(participants)),
                "description": description,
                "t_start": round(rel_start, 3),
                "t_end": round(rel_end, 3),
                "support_segment_ids": list(dict.fromkeys(support_segments)),
                "confidence": _clip_unit(raw.get("confidence"), 0.5),
                "uncertainty": _clip_unit(raw.get("uncertainty"), 0.5),
            }
        )
    return normalized


def build_global_event_graph(
    stage1_graph_path: str,
    stage2_graph_path: str,
    output_path: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    if not bool(getattr(config, "SCENE_GRAPH_GLOBAL_ENABLED", True)):
        raise RuntimeError("Global event graph construction is disabled.")
    if not force_rebuild and os.path.exists(output_path):
        cached = safe_load_global_event_graph(output_path)
        if cached is not None:
            return cached

    stage1 = safe_load_stage1_graph(stage1_graph_path)
    stage2 = safe_load_stage2_segment_graph(stage2_graph_path)
    if stage1 is None or stage2 is None:
        raise ValueError("Valid Stage-1 and Stage-2 graphs are required for global construction.")

    snapshot = {
        "max_gap_sec": float(config.SCENE_GRAPH_GLOBAL_MAX_GAP_SEC),
        "relation_overlap_threshold": float(config.SCENE_GRAPH_GLOBAL_RELATION_OVERLAP_THRESHOLD),
        "predicate_overlap_threshold": float(config.SCENE_GRAPH_GLOBAL_PREDICATE_OVERLAP_THRESHOLD),
        "vlm_model": config.SCENE_GRAPH_GLOBAL_VLM_MODEL_NAME,
    }
    graph = empty_global_event_graph(stage1.get("video_id", "unknown_video"), snapshot)
    entity_bank = {
        str(entity["entity_id"]): entity
        for entity in stage1.get("entities", [])
        if isinstance(entity, dict) and entity.get("entity_id")
    }
    frame_bank = {
        str(frame["frame_id"]): frame
        for frame in stage1.get("frames", [])
        if isinstance(frame, dict) and frame.get("frame_id")
    }
    groups = group_candidate_regions(
        stage2.get("segments", []),
        max_gap_sec=snapshot["max_gap_sec"],
        relation_overlap_threshold=snapshot["relation_overlap_threshold"],
        predicate_overlap_threshold=snapshot["predicate_overlap_threshold"],
    )
    participating_ids: set[str] = set()
    for index, segments in enumerate(groups, start=1):
        region_id = f"region_{index:04d}"
        region_start = min(float(seg.get("t_start", 0.0)) for seg in segments)
        region_end = max(float(seg.get("t_end", region_start)) for seg in segments)
        support_segment_ids = [
            str(seg.get("segment_id")) for seg in segments if seg.get("segment_id")
        ]
        vlm_output = _call_global_vlm(region_id, segments, entity_bank, frame_bank)
        relations = _normalize_region_relations(
            vlm_output.get("relations", []),
            region_id=region_id,
            segments=segments,
            entity_bank=entity_bank,
            relation_offset=len(graph["relations"]),
        )
        if not relations:
            raise RuntimeError(
                f"Global VLM returned no valid grounded relations for candidate region {region_id}."
            )
        graph["regions"].append(
            {
                "region_id": region_id,
                "t_start": region_start,
                "t_end": region_end,
                "support_segment_ids": support_segment_ids,
                "relation_ids": [relation["relation_id"] for relation in relations],
                "embedding": _average_embedding(segments),
            }
        )
        graph["relations"].extend(relations)
        for relation in relations:
            participating_ids.update(relation["participant_entity_ids"])
            for segment_id in relation["support_segment_ids"]:
                graph["contains_links"].append(
                    {"relation_id": relation["relation_id"], "segment_id": segment_id}
                )

    graph["entities"] = [entity_bank[eid] for eid in sorted(participating_ids)]
    graph["stats"].update(
        {
            "num_entities": len(graph["entities"]),
            "num_regions": len(graph["regions"]),
            "num_relations": len(graph["relations"]),
            "num_contains_links": len(graph["contains_links"]),
        }
    )
    ok, reason = validate_global_event_graph(graph)
    if not ok:
        raise ValueError(f"Global event graph validation failed: {reason}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    return graph


def ensure_global_event_graph(
    stage1_graph_path: str,
    stage2_graph_path: str,
    output_path: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    return build_global_event_graph(
        stage1_graph_path=stage1_graph_path,
        stage2_graph_path=stage2_graph_path,
        output_path=output_path,
        force_rebuild=force_rebuild,
    )
