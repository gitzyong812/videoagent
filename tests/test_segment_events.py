"""Tests for the v2 segment event inducer (M5)."""

from __future__ import annotations

from sgd.scene_graph.records import EventNode, GraphFact, StateTransition
from sgd.scene_graph.segment_events import (
    induce_segment_events,
    induce_segment_events_dict,
)


def _interval(
    fact_id: str,
    subject: str,
    predicate: str,
    obj: str | None,
    t_start: float,
    t_end: float,
    *,
    confidence: float = 0.8,
    provenance: str = "vlm_structured",
) -> GraphFact:
    return GraphFact(
        fact_id=fact_id,
        fact_type="relation",
        subject=subject,
        predicate=predicate,
        object=obj,
        time_span=(f"00:00:{t_start:06.3f}", f"00:00:{t_end:06.3f}"),
        evidence=[f"f_{int(t_start * 100):04d}", f"f_{int(t_end * 100):04d}"],
        confidence=confidence,
        source=["stage1_relation_extraction", "interval_consolidation"],
        source_provenance=provenance,  # type: ignore[arg-type]
        status="unverified",
    )


def test_continuous_event_for_each_interval_fact():
    intervals = [
        _interval("ifact_000001", "p", "holds", "cup", 10.0, 12.0),
        _interval("ifact_000002", "p", "looking_at", "cup", 10.0, 12.0),
    ]
    events, transitions = induce_segment_events(intervals)
    assert {e.event_type for e in events} >= {"holds", "looking_at"}
    assert all(isinstance(e, EventNode) for e in events)
    assert all(e.event_signature.event_type == e.event_type for e in events)
    assert transitions == []


def test_possession_transfer_emits_transfer_event_and_transition():
    intervals = [
        _interval("ifact_000001", "person_a", "holds", "cup_01", 10.0, 12.0),
        _interval("ifact_000002", "person_b", "holds", "cup_01", 13.0, 16.0),
    ]
    events, transitions = induce_segment_events(intervals)
    transfer = [e for e in events if e.event_type == "transfer"]
    assert len(transfer) == 1

    t_event = transfer[0]
    assert t_event.participants == {
        "agent": "person_a",
        "patient": "cup_01",
        "recipient": "person_b",
    }
    assert t_event.event_signature.preconditions == ["holder(cup_01)=person_a"]
    assert t_event.event_signature.effects == ["holder(cup_01)=person_b"]
    assert t_event.core_facts == ["ifact_000001", "ifact_000002"]

    assert len(transitions) == 1
    trans = transitions[0]
    assert isinstance(trans, StateTransition)
    assert trans.entity == "cup_01"
    assert trans.attribute == "holder"
    assert trans.from_value == "person_a"
    assert trans.to_value == "person_b"
    assert trans.trigger_event == t_event.event_id


def test_same_holder_does_not_emit_transfer():
    intervals = [
        _interval("ifact_000001", "person_a", "holds", "cup_01", 10.0, 12.0),
        _interval("ifact_000002", "person_a", "holds", "cup_01", 14.0, 16.0),
    ]
    _, transitions = induce_segment_events(intervals)
    assert transitions == []


def test_large_gap_skips_transfer():
    intervals = [
        _interval("ifact_000001", "person_a", "holds", "cup_01", 10.0, 12.0),
        _interval("ifact_000002", "person_b", "holds", "cup_01", 60.0, 65.0),
    ]
    _, transitions = induce_segment_events(intervals, transfer_max_gap_sec=5.0)
    assert transitions == []


def test_state_transition_dict_uses_from_to_keys():
    intervals = [
        _interval("ifact_000001", "person_a", "holds", "cup_01", 10.0, 12.0),
        _interval("ifact_000002", "person_b", "holds", "cup_01", 13.0, 16.0),
    ]
    payload = induce_segment_events_dict(intervals, video_id="vid_test")
    assert payload["version"] == "segment_events_v1"
    transitions = payload["state_transitions"]
    assert len(transitions) == 1
    t = transitions[0]
    assert "from" in t and "to" in t
    # `from_value` / `to_value` are internal names and must not leak.
    assert "from_value" not in t and "to_value" not in t


def test_min_duration_filter_drops_short_continuous_events():
    intervals = [
        _interval("ifact_000001", "p", "looking_at", "cup", 10.0, 10.1),
    ]
    events, _ = induce_segment_events(intervals, min_event_duration_sec=1.0)
    assert events == []


def test_min_confidence_filter_drops_low_confidence_events():
    intervals = [
        _interval("ifact_000001", "p", "looking_at", "cup", 10.0, 12.0, confidence=0.2),
        _interval("ifact_000002", "p", "holds", "cup", 10.0, 12.0, confidence=0.9),
    ]
    events, _ = induce_segment_events(intervals, min_event_confidence=0.5)
    types = {e.event_type for e in events}
    assert "looking_at" not in types
    assert "holds" in types


def test_event_signature_is_structured_object_not_string():
    intervals = [
        _interval("ifact_000001", "p", "holds", "cup", 10.0, 12.0),
    ]
    payload = induce_segment_events_dict(intervals, video_id="vid_test")
    sig = payload["events"][0]["event_signature"]
    assert isinstance(sig, dict)
    assert {"event_type", "participants", "preconditions", "effects", "time_span"} <= set(
        sig.keys()
    )


def test_payload_stats_count_event_types():
    intervals = [
        _interval("ifact_000001", "person_a", "holds", "cup_01", 10.0, 12.0),
        _interval("ifact_000002", "person_b", "holds", "cup_01", 13.0, 16.0),
    ]
    payload = induce_segment_events_dict(intervals, video_id="vid_test")
    by_type = payload["stats"]["by_event_type"]
    assert by_type.get("holds", 0) == 2
    assert by_type.get("transfer", 0) == 1
