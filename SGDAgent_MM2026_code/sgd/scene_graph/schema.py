from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


STAGE1_GRAPH_VERSION = "stage1_v1"
STAGE2_SEGMENT_GRAPH_VERSION = "stage2_segment_v1"
GLOBAL_EVENT_GRAPH_VERSION = "global_event_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def empty_stage1_graph(video_id: str, config_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": STAGE1_GRAPH_VERSION,
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "config": config_snapshot,
        "entities": [],
        "frames": [],
        "stats": {
            "num_entities": 0,
            "num_frames": 0,
            "num_relations": 0,
        },
    }


def validate_stage1_graph(graph: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(graph, dict):
        return False, "Graph must be a dict."

    required_keys = {"version", "video_id", "entities", "frames", "stats"}
    missing = required_keys - set(graph.keys())
    if missing:
        return False, f"Missing top-level keys: {sorted(missing)}"

    if graph.get("version") != STAGE1_GRAPH_VERSION:
        return False, f"Unsupported version: {graph.get('version')}"

    entities = graph.get("entities")
    if not isinstance(entities, list):
        return False, "`entities` must be a list."
    for ent in entities:
        if not isinstance(ent, dict):
            return False, "Each entity must be a dict."
        if "entity_id" not in ent or "canonical_name" not in ent:
            return False, "Entity missing `entity_id` or `canonical_name`."

    frames = graph.get("frames")
    if not isinstance(frames, list):
        return False, "`frames` must be a list."
    for frame in frames:
        if not isinstance(frame, dict):
            return False, "Each frame entry must be a dict."
        for key in ("frame_id", "t_sec", "entities", "relations"):
            if key not in frame:
                return False, f"Frame missing key `{key}`."
        if not isinstance(frame["entities"], list) or not isinstance(frame["relations"], list):
            return False, "Frame `entities` and `relations` must be lists."

    return True, "ok"


def safe_load_stage1_graph(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        ok, _ = validate_stage1_graph(graph)
        if not ok:
            return None
        return graph
    except Exception:
        return None


def empty_stage2_segment_graph(video_id: str, config_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": STAGE2_SEGMENT_GRAPH_VERSION,
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "config": config_snapshot,
        "segments": [],
        "stats": {
            "num_segments": 0,
            "num_temporal_relations": 0,
            "num_segment_entities": 0,
        },
    }


def validate_stage2_segment_graph(graph: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(graph, dict):
        return False, "Graph must be a dict."

    required_keys = {"version", "video_id", "segments", "stats"}
    missing = required_keys - set(graph.keys())
    if missing:
        return False, f"Missing top-level keys: {sorted(missing)}"

    if graph.get("version") != STAGE2_SEGMENT_GRAPH_VERSION:
        return False, f"Unsupported version: {graph.get('version')}"

    segments = graph.get("segments")
    if not isinstance(segments, list):
        return False, "`segments` must be a list."

    for segment in segments:
        if not isinstance(segment, dict):
            return False, "Each segment must be a dict."
        for key in ("segment_id", "t_start", "t_end", "active_entities", "temporal_relations"):
            if key not in segment:
                return False, f"Segment missing key `{key}`."
        if not isinstance(segment["active_entities"], list):
            return False, "`active_entities` must be a list."
        if not isinstance(segment["temporal_relations"], list):
            return False, "`temporal_relations` must be a list."
        if float(segment["t_end"]) < float(segment["t_start"]):
            return False, "Segment has invalid time span."

    return True, "ok"


def safe_load_stage2_segment_graph(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        ok, _ = validate_stage2_segment_graph(graph)
        if not ok:
            return None
        return graph
    except Exception:
        return None


def empty_global_event_graph(video_id: str, config_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": GLOBAL_EVENT_GRAPH_VERSION,
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "config": config_snapshot,
        "entities": [],
        "regions": [],
        "relations": [],
        "contains_links": [],
        "stats": {
            "num_entities": 0,
            "num_regions": 0,
            "num_relations": 0,
            "num_contains_links": 0,
        },
    }


def validate_global_event_graph(graph: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(graph, dict):
        return False, "Graph must be a dict."
    required = {
        "version", "video_id", "entities", "regions", "relations",
        "contains_links", "stats",
    }
    missing = required - set(graph)
    if missing:
        return False, f"Missing top-level keys: {sorted(missing)}"
    if graph.get("version") != GLOBAL_EVENT_GRAPH_VERSION:
        return False, f"Unsupported version: {graph.get('version')}"

    region_ids: set[str] = set()
    for region in graph.get("regions", []):
        if not isinstance(region, dict):
            return False, "Each region must be a dict."
        for key in ("region_id", "t_start", "t_end", "support_segment_ids"):
            if key not in region:
                return False, f"Region missing key `{key}`."
        if float(region["t_end"]) < float(region["t_start"]):
            return False, "Region has invalid time span."
        if not isinstance(region["support_segment_ids"], list):
            return False, "Region `support_segment_ids` must be a list."
        region_ids.add(str(region["region_id"]))

    relation_ids: set[str] = set()
    allowed_types = {"event", "state_evolution", "temporal", "causal"}
    for relation in graph.get("relations", []):
        if not isinstance(relation, dict):
            return False, "Each global relation must be a dict."
        for key in (
            "relation_id", "region_id", "relation_type", "predicate",
            "participant_entity_ids", "description", "t_start", "t_end",
            "support_segment_ids", "confidence", "uncertainty",
        ):
            if key not in relation:
                return False, f"Global relation missing key `{key}`."
        if relation["relation_type"] not in allowed_types:
            return False, f"Unsupported global relation type: {relation['relation_type']}"
        if str(relation["region_id"]) not in region_ids:
            return False, "Global relation references an unknown region."
        if float(relation["t_end"]) < float(relation["t_start"]):
            return False, "Global relation has invalid time span."
        relation_ids.add(str(relation["relation_id"]))

    for link in graph.get("contains_links", []):
        if not isinstance(link, dict):
            return False, "Each contains link must be a dict."
        if str(link.get("relation_id", "")) not in relation_ids:
            return False, "Contains link references an unknown relation."
        if not link.get("segment_id"):
            return False, "Contains link is missing segment_id."
    return True, "ok"


def safe_load_global_event_graph(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        ok, _ = validate_global_event_graph(graph)
        return graph if ok else None
    except Exception:
        return None
