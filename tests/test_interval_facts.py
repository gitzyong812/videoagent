"""Tests for the v2 interval-fact consolidator (M4)."""

from __future__ import annotations

import pytest

from sgd.scene_graph.interval_facts import consolidate_facts, consolidate_facts_dict
from sgd.scene_graph.records import GraphFact


def _frame_fact(
    fact_id: str,
    subject: str,
    predicate: str,
    obj: str | None,
    t_sec: float,
    *,
    fact_type: str = "relation",
    confidence: float = 0.8,
    evidence: list[str] | None = None,
    provenance: str = "vlm_structured",
) -> GraphFact:
    ts = f"00:00:{t_sec:06.3f}"
    return GraphFact(
        fact_id=fact_id,
        fact_type=fact_type,  # type: ignore[arg-type]
        subject=subject,
        predicate=predicate,
        object=obj,
        time_span=(ts, ts),
        evidence=evidence or [f"f_{int(t_sec * 100):04d}"],
        confidence=confidence,
        source=["stage1_relation_extraction"],
        source_provenance=provenance,  # type: ignore[arg-type]
        status="unverified",
    )


def test_adjacent_facts_merge_into_interval():
    facts = [
        _frame_fact("fact_000001", "p", "holds", "cup", 10.0),
        _frame_fact("fact_000002", "p", "holds", "cup", 10.5),
        _frame_fact("fact_000003", "p", "holds", "cup", 11.0),
    ]
    intervals = consolidate_facts(facts, max_gap_sec=2.0)
    assert len(intervals) == 1
    iv = intervals[0]
    assert iv.fact_id.startswith("ifact_")
    assert iv.subject == "p"
    assert iv.predicate == "holds"
    assert iv.object == "cup"
    assert iv.time_span == ("00:00:10.000", "00:00:11.000")
    assert "interval_consolidation" in iv.source
    assert iv.source_provenance == "vlm_structured"


def test_large_gap_splits_intervals():
    facts = [
        _frame_fact("fact_000001", "p", "holds", "cup", 10.0),
        _frame_fact("fact_000002", "p", "holds", "cup", 30.0),
    ]
    intervals = consolidate_facts(facts, max_gap_sec=2.0)
    assert len(intervals) == 2


def test_min_duration_filter_drops_short_intervals():
    facts = [
        _frame_fact("fact_000001", "p", "holds", "cup", 10.0),
    ]
    intervals = consolidate_facts(facts, min_duration_sec=1.0)
    assert intervals == []


def test_different_predicate_keeps_separate_groups():
    facts = [
        _frame_fact("fact_000001", "p", "holds", "cup", 10.0),
        _frame_fact("fact_000002", "p", "looking_at", "cup", 10.5),
    ]
    intervals = consolidate_facts(facts)
    keys = {(iv.subject, iv.predicate, iv.object) for iv in intervals}
    assert keys == {("p", "holds", "cup"), ("p", "looking_at", "cup")}


def test_evidence_frames_are_deduplicated_and_concatenated():
    facts = [
        _frame_fact("fact_000001", "p", "holds", "cup", 10.0, evidence=["f_a"]),
        _frame_fact("fact_000002", "p", "holds", "cup", 10.5, evidence=["f_a", "f_b"]),
        _frame_fact("fact_000003", "p", "holds", "cup", 11.0, evidence=["f_c"]),
    ]
    iv = consolidate_facts(facts)[0]
    assert iv.evidence == ["f_a", "f_b", "f_c"]


def test_max_confidence_aggregation():
    facts = [
        _frame_fact("fact_000001", "p", "holds", "cup", 10.0, confidence=0.5),
        _frame_fact("fact_000002", "p", "holds", "cup", 10.5, confidence=0.9),
        _frame_fact("fact_000003", "p", "holds", "cup", 11.0, confidence=0.7),
    ]
    iv = consolidate_facts(facts)[0]
    assert iv.confidence == 0.9


def test_mixed_provenance_raises():
    facts = [
        _frame_fact("fact_000001", "p", "holds", "cup", 10.0, provenance="vlm_structured"),
        _frame_fact("fact_000002", "p", "holds", "cup", 10.5, provenance="caption_fallback"),
    ]
    with pytest.raises(ValueError):
        consolidate_facts(facts)


def test_consolidate_facts_dict_top_level_shape():
    facts = [
        _frame_fact("fact_000001", "p", "holds", "cup", 10.0),
        _frame_fact("fact_000002", "p", "holds", "cup", 10.5),
    ]
    payload = consolidate_facts_dict(facts, video_id="vid_test", max_gap_sec=2.0)
    assert payload["version"] == "interval_facts_v1"
    assert payload["video_id"] == "vid_test"
    assert payload["stats"]["num_facts"] == 1
    assert payload["config"]["max_gap_sec"] == 2.0
