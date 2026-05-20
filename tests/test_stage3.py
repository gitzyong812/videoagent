"""Tests for Stage3 v1 global graph builder + query (M6)."""

from __future__ import annotations

import pytest

from sgd.agent import GraphAgent
from sgd.scene_graph.entity_registry import build_entity_registry_dict
from sgd.scene_graph.frame_facts import build_frame_facts
from sgd.scene_graph.interval_facts import consolidate_facts
from sgd.scene_graph.pipeline import (
    ARTIFACT_FILES,
    audit_inference_artifacts,
    build_inference_artifacts,
    load_inference_artifacts,
)
from sgd.scene_graph.query_global import (
    events_after,
    events_before,
    first_event_of_type,
    first_holder,
    holder_at,
    last_event_of_type,
    last_holder,
)
from sgd.scene_graph.records import frame_facts_to_dict
from sgd.scene_graph.segment_events import induce_segment_events_dict
from sgd.scene_graph.stage3_builder import build_global_graph_v1


def _stage1() -> dict:
    return {
        "version": "stage1_v1",
        "video_id": "vid_stage3",
        "entities": [
            {
                "entity_id": "person_a",
                "canonical_name": "person",
                "aliases": ["man with mustache"],
                "type": "person",
                "first_seen": 10.0,
                "last_seen": 12.0,
                "confidence": 0.8,
                "evidence": ["f_001", "f_002"],
            },
            {
                "entity_id": "person_b",
                "canonical_name": "person",
                "aliases": ["woman in red"],
                "type": "person",
                "first_seen": 14.0,
                "last_seen": 16.0,
                "confidence": 0.8,
                "evidence": ["f_003", "f_004"],
            },
            {
                "entity_id": "lollipop_01",
                "canonical_name": "lollipop",
                "type": "object",
                "first_seen": 10.0,
                "last_seen": 16.0,
                "confidence": 0.9,
                "evidence": ["f_001", "f_002", "f_003", "f_004"],
            },
        ],
        "frames": [
            {
                "frame_id": "f_001",
                "frame_idx": 20,
                "t_sec": 10.0,
                "path": "/p/f_001.jpg",
                "entities": [],
                "relations": [{"s": "person_a", "p": "holds", "o": "lollipop_01", "confidence": 0.9}],
            },
            {
                "frame_id": "f_002",
                "frame_idx": 22,
                "t_sec": 11.0,
                "path": "/p/f_002.jpg",
                "entities": [],
                "relations": [{"s": "person_a", "p": "holds", "o": "lollipop_01", "confidence": 0.9}],
            },
            {
                "frame_id": "f_003",
                "frame_idx": 28,
                "t_sec": 14.0,
                "path": "/p/f_003.jpg",
                "entities": [],
                "relations": [{"s": "person_b", "p": "holds", "o": "lollipop_01", "confidence": 0.9}],
            },
            {
                "frame_id": "f_004",
                "frame_idx": 30,
                "t_sec": 15.0,
                "path": "/p/f_004.jpg",
                "entities": [],
                "relations": [{"s": "person_b", "p": "holds", "o": "lollipop_01", "confidence": 0.9}],
            },
        ],
        "stats": {},
    }


def _build_global_graph() -> dict:
    stage1 = _stage1()
    registry = build_entity_registry_dict(stage1)
    raw_facts = build_frame_facts(stage1)
    interval = consolidate_facts(raw_facts, max_gap_sec=2.0)
    interval_payload = frame_facts_to_dict(interval, video_id=stage1["video_id"], created_at="t")
    events = induce_segment_events_dict([f for f in interval], video_id=stage1["video_id"])
    return build_global_graph_v1(
        entity_registry=registry,
        segment_events=events,
        video_id=stage1["video_id"],
    )


def test_global_graph_has_lifecycle_and_events():
    g = _build_global_graph()
    assert g["version"] == "global_graph_v1"
    assert len(g["entity_lifecycle_index"]) == 3
    assert g["stats"]["num_global_events"] >= 1


def test_first_and_last_holder_on_lollipop():
    g = _build_global_graph()
    fh = first_holder(g, "lollipop_01")
    lh = last_holder(g, "lollipop_01")
    assert fh and fh["entity_id"] == "person_a"
    assert lh and lh["entity_id"] == "person_b"


def test_first_and_last_event_of_type():
    g = _build_global_graph()
    first = first_event_of_type(g, "holds")
    last = last_event_of_type(g, "holds")
    assert first and last
    assert first["time_span"][0] <= last["time_span"][0]


def test_temporal_edges_before_and_overlap():
    g = _build_global_graph()
    edges = g["temporal_edges"]
    assert any(e["predicate"] == "before" for e in edges)


def test_events_before_and_after():
    g = _build_global_graph()
    transfer = [e for e in g["global_events"] if e["event_type"] == "transfer"]
    assert transfer
    transfer_id = transfer[0]["event_id"]
    before = events_before(g, transfer_id)
    after = events_after(g, transfer_id)
    # Some "holds" events come before the transfer, and possibly after as well.
    assert isinstance(before, list)
    assert isinstance(after, list)


def test_holder_at_time_uses_state_carry_over():
    g = _build_global_graph()
    # Before transfer, holder should be person_a.
    assert holder_at(g, "lollipop_01", 10.5) == "person_a"
    # After transfer, holder should be person_b.
    assert holder_at(g, "lollipop_01", 20.0) == "person_b"


def test_pipeline_writes_global_graph_artifact(tmp_path):
    build_inference_artifacts(_stage1(), output_dir=str(tmp_path))
    bundle = load_inference_artifacts(str(tmp_path))
    assert bundle["global_graph"]["version"] == "global_graph_v1"
    audit_inference_artifacts(str(tmp_path))


def test_agent_uses_global_graph_for_temporal_order(tmp_path):
    build_inference_artifacts(_stage1(), output_dir=str(tmp_path))
    agent = GraphAgent.from_inference_dir(str(tmp_path))
    out = agent.run("Who gets the lollipop last?")
    history = out["belief_state"]["tool_history"]
    temporal_steps = [h for h in history if h["tool_name"] == "TemporalOrderQuery"]
    assert temporal_steps
    # The Stage3 fast path tags itself in warnings.
    flat_warnings = []
    for step in temporal_steps:
        flat_warnings.extend(step["result_summary"].get("warnings") or [])
    assert any("global_graph_v1" in w for w in flat_warnings)
    assert out["answer"]["answer"] == "person_b"
