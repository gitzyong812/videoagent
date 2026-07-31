from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


STAGE1_GRAPH_VERSION = "stage1_v1"
STAGE2_SEGMENT_GRAPH_VERSION = "stage2_segment_v1"


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
