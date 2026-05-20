"""End-to-end tests for the GraphAgent loop (M10)."""

from __future__ import annotations

import pytest

from sgd.agent import GraphAgent, RUN_MODE_STRICT
from sgd.scene_graph.pipeline import build_inference_artifacts


def _stage1_with_transfer() -> dict:
    return {
        "version": "stage1_v1",
        "video_id": "vid_agent",
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
                "entity_id": "lollipop_01",
                "canonical_name": "lollipop",
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
                    {"entity_id": "lollipop_01", "name": "lollipop", "type": "object", "confidence": 0.9},
                ],
                "relations": [
                    {"s": "person_a", "p": "holds", "o": "lollipop_01", "confidence": 0.9},
                ],
            },
            {
                "frame_id": "f_002",
                "frame_idx": 22,
                "t_sec": 11.0,
                "path": "/p/f_002.jpg",
                "entities": [
                    {"entity_id": "person_a", "name": "person", "type": "person", "confidence": 0.9}
                ],
                "relations": [
                    {"s": "person_a", "p": "holds", "o": "lollipop_01", "confidence": 0.9},
                ],
            },
            {
                "frame_id": "f_003",
                "frame_idx": 28,
                "t_sec": 14.0,
                "path": "/p/f_003.jpg",
                "entities": [
                    {"entity_id": "person_b", "name": "person", "type": "person", "confidence": 0.9}
                ],
                "relations": [
                    {"s": "person_b", "p": "holds", "o": "lollipop_01", "confidence": 0.9},
                ],
            },
            {
                "frame_id": "f_004",
                "frame_idx": 30,
                "t_sec": 15.0,
                "path": "/p/f_004.jpg",
                "entities": [
                    {"entity_id": "person_b", "name": "person", "type": "person", "confidence": 0.9}
                ],
                "relations": [
                    {"s": "person_b", "p": "holds", "o": "lollipop_01", "confidence": 0.9},
                ],
            },
        ],
        "stats": {},
    }


@pytest.fixture
def agent(tmp_path) -> GraphAgent:
    build_inference_artifacts(_stage1_with_transfer(), output_dir=str(tmp_path))
    return GraphAgent.from_inference_dir(str(tmp_path))


def test_agent_default_run_mode_is_strict_graph_only(agent):
    assert agent.run_mode == RUN_MODE_STRICT


def test_who_gets_the_lollipop_last(agent):
    out = agent.run("Who gets the lollipop last?")
    assert out["run_mode"] == RUN_MODE_STRICT
    chain = out["answer"]["evidence_chain"]
    assert chain["frame_facts"]
    assert chain["support_frames"]
    assert chain["visual_evidence"]
    # Belief state captures the intent and the picked answer.
    belief = out["belief_state"]
    assert belief["parsed_intent"]["intent"] == "temporal_order_query"
    assert belief["parsed_intent"]["temporal_operator"] == "last"
    assert out["answer"]["answer"] == "person_b"


def test_who_first_holds_lollipop(agent):
    out = agent.run("Who first holds the lollipop?")
    assert out["belief_state"]["parsed_intent"]["temporal_operator"] == "first"
    assert out["answer"]["answer"] == "person_a"


def test_what_happens_with_lollipop_returns_events(agent):
    out = agent.run("What happens with the lollipop?")
    chain = out["answer"]["evidence_chain"]
    assert chain["segment_events"]
    event_types = {e["event_type"] for e in chain["segment_events"]}
    assert "transfer" in event_types or "holds" in event_types


def test_tool_history_excludes_legacy_tools(agent):
    out = agent.run("Who gets the lollipop last?")
    used = {entry["tool_name"] for entry in out["belief_state"]["tool_history"]}
    # Only strict-safe Alg-plan tool names appear.
    legacy = {"frame_inspect_tool", "clip_search_tool", "global_browse_tool", "finish"}
    assert used.isdisjoint(legacy)
    # Strict-mode tool surface contract: every step is one of the Alg-plan tools.
    allowed = {
        "QuestionDecompose",
        "EntityResolve",
        "RelationQuery",
        "EventSearch",
        "TemporalOrderQuery",
        "EvidenceTrace",
        "VisualVerify",
        "AnswerWithEvidence",
    }
    assert used <= allowed


def test_question_with_no_evidence_returns_unsupported(agent):
    out = agent.run("Who repairs the spaceship at sunset?")
    assert out["answer"]["confidence"] == 0.0
    assert out["answer"].get("status") == "unsupported"
    assert "no_supporting_evidence_in_graph" in out["belief_state"]["uncertain_items"]
