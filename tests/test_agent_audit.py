"""Tests for the agent-run audit and run-report (M0 + M11)."""

from __future__ import annotations

import pytest

from sgd.agent import GraphAgent
from sgd.agent.audit import AgentRunAuditError, audit_agent_run
from sgd.agent.run_report import build_run_report, evidence_chain_validity
from sgd.scene_graph.pipeline import build_inference_artifacts


def _stage1() -> dict:
    return {
        "version": "stage1_v1",
        "video_id": "vid_audit",
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
                "evidence": ["f_003"],
            },
            {
                "entity_id": "lollipop_01",
                "canonical_name": "lollipop",
                "type": "object",
                "first_seen": 10.0,
                "last_seen": 16.0,
                "confidence": 0.9,
                "evidence": ["f_001", "f_002", "f_003"],
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
        ],
        "stats": {},
    }


@pytest.fixture
def agent(tmp_path) -> GraphAgent:
    build_inference_artifacts(_stage1(), output_dir=str(tmp_path))
    return GraphAgent.from_inference_dir(str(tmp_path))


def test_clean_strict_run_passes_audit(agent):
    out = agent.run("Who gets the lollipop last?")
    report = audit_agent_run(out)
    assert report["status"] == "ok"
    assert report["run_mode"] == "strict_graph_only"
    # All tool names should be Alg-plan §12 names.
    assert set(report["used_tools"]) <= {
        "QuestionDecompose",
        "EntityResolve",
        "RelationQuery",
        "EventSearch",
        "TemporalOrderQuery",
        "EvidenceTrace",
        "VisualVerify",
        "AnswerWithEvidence",
    }


def test_audit_rejects_legacy_tool_in_history():
    forged = {
        "run_mode": "strict_graph_only",
        "answer": {
            "answer": "x",
            "confidence": 0.5,
            "evidence_chain": {
                "segment_events": [{"event_id": "e1"}],
                "frame_facts": [],
                "support_frames": ["f_001"],
                "visual_evidence": [],
                "global": [],
                "verification": [],
            },
        },
        "belief_state": {
            "tool_history": [
                {"tool_name": "frame_inspect_tool", "query": {}, "result_summary": {}},
            ],
        },
    }
    with pytest.raises(AgentRunAuditError):
        audit_agent_run(forged)


def test_audit_rejects_caption_field_in_chain():
    forged = {
        "run_mode": "strict_graph_only",
        "answer": {
            "answer": "x",
            "confidence": 0.5,
            "evidence_chain": {
                "segment_events": [{"event_id": "e1", "caption_context": "leak"}],
                "frame_facts": [],
                "support_frames": ["f_001"],
                "visual_evidence": [],
                "global": [],
                "verification": [],
            },
        },
        "belief_state": {"tool_history": []},
    }
    with pytest.raises(AgentRunAuditError):
        audit_agent_run(forged)


def test_audit_rejects_hybrid_run_mode():
    forged = {
        "run_mode": "hybrid",
        "answer": {"status": "unsupported", "evidence_chain": {}},
        "belief_state": {"tool_history": []},
    }
    with pytest.raises(AgentRunAuditError):
        audit_agent_run(forged)


def test_audit_allows_unsupported_answer_with_empty_chain():
    out = {
        "run_mode": "strict_graph_only",
        "answer": {
            "status": "unsupported",
            "answer": "unknown",
            "confidence": 0.0,
            "evidence_chain": {
                "global": [],
                "segment_events": [],
                "frame_facts": [],
                "support_frames": [],
                "visual_evidence": [],
                "verification": [],
            },
        },
        "belief_state": {"tool_history": [{"tool_name": "QuestionDecompose", "query": {}, "result_summary": {}}]},
    }
    report = audit_agent_run(out)
    assert report["status"] == "ok"
    assert report["answer_status"] == "unsupported"


# ── M11 run report ────────────────────────────────────────────────────────


def test_evidence_chain_validity_supported(agent):
    out = agent.run("Who gets the lollipop last?")
    validity = evidence_chain_validity(out["answer"])
    assert validity["is_supported"] is True
    assert validity["has_support_frame"] is True


def test_evidence_chain_validity_unsupported(agent):
    out = agent.run("Who repairs the spaceship at sunset?")
    validity = evidence_chain_validity(out["answer"])
    assert validity["is_supported"] is False
    assert validity["answer_status"] == "unsupported"


def test_run_report_summary_counts_supported_and_unsupported(agent):
    runs = [
        agent.run("Who gets the lollipop last?"),
        agent.run("Who repairs the spaceship at sunset?"),
    ]
    report = build_run_report(runs)
    assert report["version"] == "agent_run_report_v1"
    assert report["report_status"] == "ok"
    summary = report["summary"]
    assert summary["total_runs"] == 2
    assert summary["supported_runs"] == 1
    assert summary["unsupported_runs"] == 1
    assert summary["audit_failures"] == 0


def test_run_report_records_audit_failures_without_raising():
    forged = {
        "run_mode": "hybrid",
        "answer": {"status": "unsupported", "evidence_chain": {}},
        "belief_state": {"tool_history": []},
    }
    report = build_run_report([forged])
    assert report["report_status"] == "failed"
    assert report["summary"]["audit_failures"] == 1
