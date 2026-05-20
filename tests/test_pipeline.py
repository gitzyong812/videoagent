"""Tests for the inference-artifact pipeline (M8)."""

from __future__ import annotations

import json
import os

import pytest

from sgd.scene_graph.pipeline import (
    ARTIFACT_FILES,
    InferenceArtifactPaths,
    StrictModeViolation,
    audit_inference_artifacts,
    build_inference_artifacts,
    load_inference_artifacts,
)


def _stage1_graph_with_holders() -> dict:
    return {
        "version": "stage1_v1",
        "video_id": "vid_pipeline",
        "entities": [
            {
                "entity_id": "person_a",
                "canonical_name": "person",
                "aliases": ["man with mustache"],
                "type": "person",
                "first_seen": 10.0,
                "last_seen": 20.0,
                "confidence": 0.8,
                "evidence": ["f_001", "f_002"],
            },
            {
                "entity_id": "person_b",
                "canonical_name": "person",
                "aliases": ["woman in red"],
                "type": "person",
                "first_seen": 14.0,
                "last_seen": 24.0,
                "confidence": 0.8,
                "evidence": ["f_003", "f_004"],
            },
            {
                "entity_id": "cup_01",
                "canonical_name": "cup",
                "type": "object",
                "first_seen": 10.0,
                "last_seen": 24.0,
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
                "entities": [
                    {"entity_id": "person_a", "name": "person", "type": "person", "confidence": 0.9},
                    {"entity_id": "cup_01", "name": "cup", "type": "object", "confidence": 0.9},
                ],
                "relations": [
                    {"s": "person_a", "p": "holds", "o": "cup_01", "confidence": 0.9},
                ],
            },
            {
                "frame_id": "f_002",
                "frame_idx": 22,
                "t_sec": 11.0,
                "path": "/p/f_002.jpg",
                "entities": [
                    {"entity_id": "person_a", "name": "person", "type": "person", "confidence": 0.9},
                ],
                "relations": [
                    {"s": "person_a", "p": "holds", "o": "cup_01", "confidence": 0.9},
                ],
            },
            {
                "frame_id": "f_003",
                "frame_idx": 28,
                "t_sec": 14.0,
                "path": "/p/f_003.jpg",
                "entities": [
                    {"entity_id": "person_b", "name": "person", "type": "person", "confidence": 0.9},
                ],
                "relations": [
                    {"s": "person_b", "p": "holds", "o": "cup_01", "confidence": 0.9},
                ],
            },
            {
                "frame_id": "f_004",
                "frame_idx": 30,
                "t_sec": 15.0,
                "path": "/p/f_004.jpg",
                "entities": [
                    {"entity_id": "person_b", "name": "person", "type": "person", "confidence": 0.9},
                ],
                "relations": [
                    {"s": "person_b", "p": "holds", "o": "cup_01", "confidence": 0.9},
                ],
            },
        ],
        "stats": {},
    }


def test_pipeline_writes_all_artifact_files(tmp_path):
    paths = build_inference_artifacts(
        _stage1_graph_with_holders(), output_dir=str(tmp_path)
    )
    for name in ARTIFACT_FILES.values():
        assert os.path.exists(os.path.join(tmp_path, name)), name
    assert paths.base_dir == str(tmp_path)


def test_stripped_index_carries_component_pointers_and_stats(tmp_path):
    build_inference_artifacts(_stage1_graph_with_holders(), output_dir=str(tmp_path))
    with open(os.path.join(tmp_path, ARTIFACT_FILES["stripped_index"])) as f:
        idx = json.load(f)
    assert idx["version"] == "inference_graph_stripped_v1"
    assert idx["video_id"] == "vid_pipeline"
    assert set(idx["components"].keys()) == {
        "entity_registry",
        "frame_facts",
        "interval_facts",
        "segment_events",
        "global_graph",
        "temporal_units",
    }
    stats = idx["stats"]
    assert stats["num_entities"] == 3
    assert stats["num_events"] >= 1
    assert stats["num_state_transitions"] >= 1


def test_pipeline_emits_transfer_event_end_to_end(tmp_path):
    build_inference_artifacts(_stage1_graph_with_holders(), output_dir=str(tmp_path))
    with open(os.path.join(tmp_path, ARTIFACT_FILES["segment_events"])) as f:
        seg = json.load(f)
    types = [e["event_type"] for e in seg["events"]]
    assert "transfer" in types


def test_load_inference_artifacts_round_trip(tmp_path):
    build_inference_artifacts(_stage1_graph_with_holders(), output_dir=str(tmp_path))
    bundle = load_inference_artifacts(str(tmp_path))
    assert bundle["stripped_index"]["video_id"] == "vid_pipeline"
    assert bundle["entity_registry"]["version"] == "entity_registry_v1"
    assert bundle["frame_facts"]["version"] == "frame_facts_v1"
    assert bundle["interval_facts"]["version"] == "interval_facts_v1"
    assert bundle["segment_events"]["version"] == "segment_events_v1"
    assert bundle["temporal_units"]["version"] == "temporal_units_v1"


def test_audit_passes_on_clean_artifacts(tmp_path):
    build_inference_artifacts(_stage1_graph_with_holders(), output_dir=str(tmp_path))
    report = audit_inference_artifacts(str(tmp_path))
    assert report["status"] == "ok"
    assert "entity_registry.json" in report["audited_files"]


def test_pipeline_strips_unsafe_aliases_before_inference_audit(tmp_path):
    graph = _stage1_graph_with_holders()
    graph["entities"][0]["aliases"].append("the villain who later breaks the mirror")
    build_inference_artifacts(graph, output_dir=str(tmp_path))
    report = audit_inference_artifacts(str(tmp_path))
    assert report["status"] == "ok"
    reg_path = os.path.join(tmp_path, ARTIFACT_FILES["entity_registry"])
    with open(reg_path) as f:
        reg = json.load(f)
    texts = {
        alias["text"]
        for entity in reg["entities"]
        for alias in entity.get("aliases", [])
    }
    assert "the villain who later breaks the mirror" not in texts


def test_audit_rejects_caption_context_in_frame_facts(tmp_path):
    build_inference_artifacts(_stage1_graph_with_holders(), output_dir=str(tmp_path))
    facts_path = os.path.join(tmp_path, ARTIFACT_FILES["frame_facts"])
    with open(facts_path) as f:
        facts = json.load(f)
    # Tamper: inject a forbidden field.
    facts["facts"][0]["caption_context"] = "a hidden caption sentence"
    with open(facts_path, "w") as f:
        json.dump(facts, f)
    with pytest.raises(StrictModeViolation):
        audit_inference_artifacts(str(tmp_path))


def test_audit_rejects_non_strict_safe_alias(tmp_path):
    build_inference_artifacts(_stage1_graph_with_holders(), output_dir=str(tmp_path))
    reg_path = os.path.join(tmp_path, ARTIFACT_FILES["entity_registry"])
    with open(reg_path) as f:
        reg = json.load(f)
    # Tamper: smuggle a non-strict-safe alias into the registry.
    reg["entities"][0]["aliases"].append(
        {
            "text": "the villain who later breaks the mirror",
            "source": "stage1_entity_extraction",
            "source_provenance": "vlm_structured",
            "strict_safe": False,
            "support_frame_ids": [],
        }
    )
    with open(reg_path, "w") as f:
        json.dump(reg, f)
    with pytest.raises(StrictModeViolation):
        audit_inference_artifacts(str(tmp_path))


def test_audit_rejects_string_event_signature(tmp_path):
    build_inference_artifacts(_stage1_graph_with_holders(), output_dir=str(tmp_path))
    seg_path = os.path.join(tmp_path, ARTIFACT_FILES["segment_events"])
    with open(seg_path) as f:
        seg = json.load(f)
    seg["events"][0]["event_signature"] = "a natural-language summary"
    with open(seg_path, "w") as f:
        json.dump(seg, f)
    with pytest.raises(StrictModeViolation):
        audit_inference_artifacts(str(tmp_path))


def test_audit_rejects_summary_field_anywhere(tmp_path):
    build_inference_artifacts(_stage1_graph_with_holders(), output_dir=str(tmp_path))
    reg_path = os.path.join(tmp_path, ARTIFACT_FILES["entity_registry"])
    with open(reg_path) as f:
        reg = json.load(f)
    reg["entities"][0]["summary"] = "free-form text"
    with open(reg_path, "w") as f:
        json.dump(reg, f)
    with pytest.raises(StrictModeViolation):
        audit_inference_artifacts(str(tmp_path))


def test_inference_artifact_paths_under():
    paths = InferenceArtifactPaths.under("/tmp/x")
    assert paths.base_dir == "/tmp/x"
    assert paths.entity_registry.endswith("entity_registry.json")
    assert paths.stripped_index.endswith("inference_graph_stripped.json")
