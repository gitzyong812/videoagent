"""Inference-artifact pipeline (M8).

Builds the strict-safe v2 graph artifacts that Agent-facing tools read at
inference time, per Alg-plan §0.10 + dev-plan M8.

Construction vs inference contract:

    Construction artifacts (e.g. legacy Stage1/Stage2 JSON files produced by
    ``stage1_builder.py`` and ``segment_builder.py``) may contain caption
    snippets, transcripts, VLM narrations, and other free-text signals.
    Those are NEVER touched by Agent-facing tools.

    Inference artifacts (the files written here) only contain structured
    graph records. They are the canonical read surface for M9 wrapper tools
    and M10 agent loops.

Layout written by ``build_inference_artifacts``:

    <inference_dir>/
        entity_registry.json
        frame_facts.json
        interval_facts.json
        segment_events.json
        temporal_units.json
        inference_graph_stripped.json     # assembled index pointing at the others

The assembled ``inference_graph_stripped.json`` is the single entry point
loaded by Agent tools.

Audit hook:
    ``audit_inference_artifacts`` performs a structural strict-mode check on
    a directory and raises if any caption-like field, non-strict-safe alias,
    or string-valued ``event_signature`` is found.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from sgd.scene_graph.entity_registry import build_entity_registry_dict
from sgd.scene_graph.frame_facts import build_frame_facts, build_frame_facts_dict
from sgd.scene_graph.interval_facts import consolidate_facts_dict
from sgd.scene_graph.records import GraphFact
from sgd.scene_graph.records import frame_facts_to_dict
from sgd.scene_graph.schema import safe_load_stage1_graph, utc_now_iso
from sgd.scene_graph.segment_events import induce_segment_events_dict
from sgd.scene_graph.stage3_builder import build_global_graph_v1
from sgd.scene_graph.temporal_units import build_temporal_units


INFERENCE_GRAPH_VERSION = "inference_graph_stripped_v1"

ARTIFACT_FILES = {
    "entity_registry": "entity_registry.json",
    "frame_facts": "frame_facts.json",
    "interval_facts": "interval_facts.json",
    "segment_events": "segment_events.json",
    "global_graph": "global_graph.json",
    "temporal_units": "temporal_units.json",
    "stripped_index": "inference_graph_stripped.json",
}

# Field names that must never appear in any inference artifact.
FORBIDDEN_INFERENCE_FIELDS: set[str] = {
    "caption",
    "caption_context",
    "caption_snippets",
    "transcript",
    "narration",
    "script",
    "free_form_summary",
    "summary",
}


@dataclass
class InferenceArtifactPaths:
    base_dir: str
    entity_registry: str
    frame_facts: str
    interval_facts: str
    segment_events: str
    global_graph: str
    temporal_units: str
    stripped_index: str

    @classmethod
    def under(cls, base_dir: str) -> "InferenceArtifactPaths":
        return cls(
            base_dir=base_dir,
            entity_registry=os.path.join(base_dir, ARTIFACT_FILES["entity_registry"]),
            frame_facts=os.path.join(base_dir, ARTIFACT_FILES["frame_facts"]),
            interval_facts=os.path.join(base_dir, ARTIFACT_FILES["interval_facts"]),
            segment_events=os.path.join(base_dir, ARTIFACT_FILES["segment_events"]),
            global_graph=os.path.join(base_dir, ARTIFACT_FILES["global_graph"]),
            temporal_units=os.path.join(base_dir, ARTIFACT_FILES["temporal_units"]),
            stripped_index=os.path.join(base_dir, ARTIFACT_FILES["stripped_index"]),
        )


def _frame_facts_payload_to_records(payload: dict[str, Any]) -> list[GraphFact]:
    facts: list[GraphFact] = []
    for fact in payload.get("facts", []) or []:
        time_span = fact.get("time_span", ["00:00:00.000", "00:00:00.000"])
        facts.append(
            GraphFact(
                fact_id=str(fact.get("fact_id", "")),
                fact_type=str(fact.get("fact_type", "relation")),  # type: ignore[arg-type]
                subject=str(fact.get("subject", "")),
                predicate=str(fact.get("predicate", "")),
                object=fact.get("object"),
                time_span=(str(time_span[0]), str(time_span[1])),
                evidence=list(fact.get("evidence", []) or []),
                confidence=float(fact.get("confidence", 0.0)),
                source=list(fact.get("source", []) or []),
                source_provenance=str(fact.get("source_provenance", "vlm_structured")),  # type: ignore[arg-type]
                status=str(fact.get("status", "unverified")),  # type: ignore[arg-type]
            )
        )
    return facts


def build_inference_artifacts(
    stage1_graph: dict[str, Any],
    *,
    output_dir: str,
    interval_max_gap_sec: float = 2.0,
    interval_min_duration_sec: float = 0.0,
    segment_min_event_duration_sec: float = 0.0,
    segment_min_event_confidence: float = 0.0,
    segment_transfer_max_gap_sec: float = 5.0,
    temporal_window_sec: float = 30.0,
    temporal_overlap_sec: float = 5.0,
    include_frame_paths: bool = True,
) -> InferenceArtifactPaths:
    """Build the strict-safe inference artifact bundle from a Stage1 graph."""
    os.makedirs(output_dir, exist_ok=True)
    paths = InferenceArtifactPaths.under(output_dir)
    video_id = str(stage1_graph.get("video_id", ""))

    entity_registry = build_entity_registry_dict(stage1_graph, include_unsafe_aliases=False)
    frame_facts_payload = build_frame_facts_dict(stage1_graph)
    facts = _frame_facts_payload_to_records(frame_facts_payload)
    interval_facts_payload = consolidate_facts_dict(
        facts,
        video_id=video_id,
        max_gap_sec=interval_max_gap_sec,
        min_duration_sec=interval_min_duration_sec,
    )
    interval_records = _frame_facts_payload_to_records(interval_facts_payload)
    segment_events_payload = induce_segment_events_dict(
        interval_records,
        video_id=video_id,
        min_event_duration_sec=segment_min_event_duration_sec,
        min_event_confidence=segment_min_event_confidence,
        transfer_max_gap_sec=segment_transfer_max_gap_sec,
    )
    temporal_units_payload = build_temporal_units(
        stage1_graph,
        window_sec=temporal_window_sec,
        overlap_sec=temporal_overlap_sec,
        include_frame_paths=include_frame_paths,
    )
    global_graph_payload = build_global_graph_v1(
        entity_registry=entity_registry,
        segment_events=segment_events_payload,
        video_id=video_id,
    )

    _write_json(paths.entity_registry, entity_registry)
    _write_json(paths.frame_facts, frame_facts_payload)
    _write_json(paths.interval_facts, interval_facts_payload)
    _write_json(paths.segment_events, segment_events_payload)
    _write_json(paths.global_graph, global_graph_payload)
    _write_json(paths.temporal_units, temporal_units_payload)

    stripped_index = {
        "version": INFERENCE_GRAPH_VERSION,
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "components": {
            "entity_registry": ARTIFACT_FILES["entity_registry"],
            "frame_facts": ARTIFACT_FILES["frame_facts"],
            "interval_facts": ARTIFACT_FILES["interval_facts"],
            "segment_events": ARTIFACT_FILES["segment_events"],
            "global_graph": ARTIFACT_FILES["global_graph"],
            "temporal_units": ARTIFACT_FILES["temporal_units"],
        },
        "stats": {
            "num_entities": entity_registry["stats"]["num_entities"],
            "num_frame_facts": frame_facts_payload["stats"]["num_facts"],
            "num_interval_facts": interval_facts_payload["stats"]["num_facts"],
            "num_events": segment_events_payload["stats"]["num_events"],
            "num_state_transitions": segment_events_payload["stats"]["num_state_transitions"],
            "num_global_events": global_graph_payload["stats"]["num_global_events"],
            "num_temporal_edges": global_graph_payload["stats"]["num_temporal_edges"],
            "num_frames": temporal_units_payload["stats"]["num_frames"],
            "num_segment_windows": temporal_units_payload["stats"]["num_segment_windows"],
        },
    }
    _write_json(paths.stripped_index, stripped_index)

    return paths


def build_inference_artifacts_from_stage1_path(
    stage1_path: str,
    *,
    output_dir: str,
    **kwargs: Any,
) -> InferenceArtifactPaths | None:
    graph = safe_load_stage1_graph(stage1_path)
    if graph is None:
        return None
    return build_inference_artifacts(graph, output_dir=output_dir, **kwargs)


def load_inference_artifacts(base_dir: str) -> dict[str, Any]:
    """Load the assembled stripped artifact bundle from a directory."""
    paths = InferenceArtifactPaths.under(base_dir)
    bundle: dict[str, Any] = {}
    bundle["stripped_index"] = _read_json(paths.stripped_index)
    bundle["entity_registry"] = _read_json(paths.entity_registry)
    bundle["frame_facts"] = _read_json(paths.frame_facts)
    bundle["interval_facts"] = _read_json(paths.interval_facts)
    bundle["segment_events"] = _read_json(paths.segment_events)
    bundle["global_graph"] = (
        _read_json(paths.global_graph) if os.path.exists(paths.global_graph) else {}
    )
    bundle["temporal_units"] = _read_json(paths.temporal_units)
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Strict-mode audit
# ─────────────────────────────────────────────────────────────────────────────


class StrictModeViolation(ValueError):
    """Raised when an inference artifact contains a forbidden field or value."""


def _walk_forbidden_keys(value: Any, *, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key in FORBIDDEN_INFERENCE_FIELDS:
                hits.append(f"{path}.{key}")
            hits.extend(_walk_forbidden_keys(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            hits.extend(_walk_forbidden_keys(child, path=f"{path}[{i}]"))
    return hits


def audit_inference_artifacts(base_dir: str) -> dict[str, Any]:
    """Strict-mode structural audit of an inference artifact directory.

    Raises StrictModeViolation on the first failure. Returns an OK report on
    success so callers can log the audited file set.
    """
    bundle = load_inference_artifacts(base_dir)

    # 1. No forbidden field names anywhere in the bundle.
    forbidden_hits = _walk_forbidden_keys(bundle)
    if forbidden_hits:
        raise StrictModeViolation(
            f"Forbidden inference fields present: {forbidden_hits[:5]}"
        )

    # 2. Entity aliases must all be strict_safe.
    registry = bundle.get("entity_registry") or {}
    for ent in registry.get("entities", []) or []:
        for alias in ent.get("aliases", []) or []:
            if not alias.get("strict_safe", False):
                raise StrictModeViolation(
                    f"Non-strict-safe alias surfaced in entity_registry: "
                    f"entity={ent.get('entity_id')} alias={alias.get('text')!r}"
                )

    # 3. event_signature must be a structured object (dict), not a string.
    seg = bundle.get("segment_events") or {}
    for event in seg.get("events", []) or []:
        sig = event.get("event_signature")
        if not isinstance(sig, dict):
            raise StrictModeViolation(
                f"event_signature must be a dict, got {type(sig).__name__} "
                f"for event_id={event.get('event_id')}"
            )

    # 4. frame_facts must not carry caption_context.
    facts_payload = bundle.get("frame_facts") or {}
    for fact in facts_payload.get("facts", []) or []:
        if "caption_context" in fact:
            raise StrictModeViolation(
                f"frame_fact carries caption_context: fact_id={fact.get('fact_id')}"
            )

    return {
        "status": "ok",
        "audited_files": sorted(ARTIFACT_FILES.values()),
        "stats": (bundle.get("stripped_index") or {}).get("stats", {}),
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
