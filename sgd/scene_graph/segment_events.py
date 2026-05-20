"""Segment-level event induction (M5).

Consumes consolidated interval facts (produced by ``interval_facts.py``) and
emits structured EventNode + StateTransition records per Alg-plan §6 +
dev-plan M5.

MVP scope:
    - ``continuous_relation`` events: every interval fact with sufficient
      duration and confidence becomes an EventNode whose ``event_type``
      mirrors the predicate. This guarantees EventSearch returns something
      for any video.
    - ``possession_transfer`` events: when the holder of an object changes
      across two adjacent intervals, induce a ``transfer`` event with agent /
      patient / recipient participants, ``holder(X)=A`` precondition, and
      ``holder(X)=B`` effect. The original "A holds X" / "B holds X" interval
      facts remain as continuous_relation events; the transfer is added on
      top so EventSearch by event_type="transfer" works without filtering.

Out of MVP:
    - LLM / VLM structured induction (deferred to v2.0 + builder-assisted).
    - State transitions for open/close/break (Stage1 currently does not emit
      key/value attribute facts; once it does, M5 will add them).
    - Segment-window grouping (events carry full time_spans; M6 Stage3 can
      cluster them later if needed).

All emitted EventNodes carry ``status="unverified"`` until M7 verifies
support.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from sgd.scene_graph.records import (
    EventNode,
    EventSignature,
    FactStatus,
    GraphFact,
    SourceProvenance,
    StateTransition,
    hhmmss_to_secs,
    secs_to_hhmmss,
    segment_event_graph_to_dict,
)
from sgd.scene_graph.schema import utc_now_iso


# Predicates that indicate possession; any of these is treated as "holder(X)=S".
POSSESSION_PREDICATES: set[str] = {
    "holds",
    "holding",
    "has",
    "possess",
    "possesses",
    "carrying",
    "carries",
    "taking",
    "takes",
    "giving",
    "gives",
    "grabs",
    "grabbing",
    "clutches",
}

DEFAULT_MIN_EVENT_DURATION_SEC = 0.0
DEFAULT_MIN_EVENT_CONFIDENCE = 0.0
DEFAULT_TRANSFER_MAX_GAP_SEC = 5.0


def _event_id(counter: int) -> str:
    return f"event_{counter:06d}"


def _transition_id(counter: int) -> str:
    return f"trans_{counter:06d}"


def _duration_sec(time_span: tuple[str, str]) -> float:
    return max(0.0, hhmmss_to_secs(time_span[1]) - hhmmss_to_secs(time_span[0]))


def _normalize_predicate(pred: str) -> str:
    return (pred or "").strip().lower()


def _build_continuous_event(
    *,
    counter: int,
    fact: GraphFact,
) -> EventNode:
    event_type = _normalize_predicate(fact.predicate)
    participants: dict[str, str] = {"agent": fact.subject}
    if fact.object:
        participants["patient"] = fact.object

    signature = EventSignature(
        event_type=event_type,
        participants=participants,
        preconditions=[],
        effects=[],
        time_span=fact.time_span,
    )
    return EventNode(
        event_id=_event_id(counter),
        event_type=event_type,
        time_span=fact.time_span,
        participants=participants,
        event_signature=signature,
        preconditions=[],
        effects=[],
        core_facts=[fact.fact_id],
        support_facts=[fact.fact_id],
        evidence=list(fact.evidence),
        confidence=fact.confidence,
        source=["continuous_relation_induction"],
        source_provenance=fact.source_provenance,
        status="unverified",
    )


def _build_transfer_event(
    *,
    counter: int,
    fact_a: GraphFact,
    fact_b: GraphFact,
) -> EventNode:
    """Emit a give/transfer event from two adjacent holder intervals."""
    obj = fact_a.object or fact_b.object or ""
    agent = fact_a.subject
    recipient = fact_b.subject

    a_end = hhmmss_to_secs(fact_a.time_span[1])
    b_start = hhmmss_to_secs(fact_b.time_span[0])
    t_start = min(a_end, b_start)
    t_end = max(a_end, b_start)
    time_span = (secs_to_hhmmss(t_start), secs_to_hhmmss(t_end))

    participants = {"agent": agent, "patient": obj, "recipient": recipient}
    preconditions_facts = [{"fact": f"holder({obj})={agent}", "support_facts": [fact_a.fact_id]}]
    effects_facts = [{"state": f"holder({obj})={recipient}", "support_facts": [fact_b.fact_id]}]

    signature = EventSignature(
        event_type="transfer",
        participants=participants,
        preconditions=[f"holder({obj})={agent}"],
        effects=[f"holder({obj})={recipient}"],
        time_span=time_span,
    )

    evidence: list[str] = []
    for fid in [*fact_a.evidence, *fact_b.evidence]:
        if fid not in evidence:
            evidence.append(fid)

    # Provenance: both inputs must share provenance; caller groups by holder fact.
    provenance: SourceProvenance = fact_a.source_provenance

    return EventNode(
        event_id=_event_id(counter),
        event_type="transfer",
        time_span=time_span,
        participants=participants,
        event_signature=signature,
        preconditions=preconditions_facts,
        effects=effects_facts,
        core_facts=[fact_a.fact_id, fact_b.fact_id],
        support_facts=[fact_a.fact_id, fact_b.fact_id],
        evidence=evidence,
        confidence=min(fact_a.confidence, fact_b.confidence),
        source=["possession_transfer_induction"],
        source_provenance=provenance,
        status="unverified",
    )


def _build_holder_transition(
    *,
    counter: int,
    obj: str,
    fact_a: GraphFact,
    fact_b: GraphFact,
    trigger_event_id: str | None,
) -> StateTransition:
    a_end = hhmmss_to_secs(fact_a.time_span[1])
    b_start = hhmmss_to_secs(fact_b.time_span[0])
    t_start = min(a_end, b_start)
    t_end = max(a_end, b_start)
    return StateTransition(
        transition_id=_transition_id(counter),
        entity=obj,
        attribute="holder",
        from_value=fact_a.subject,
        to_value=fact_b.subject,
        time_span=(secs_to_hhmmss(t_start), secs_to_hhmmss(t_end)),
        support_facts=[fact_a.fact_id, fact_b.fact_id],
        confidence=min(fact_a.confidence, fact_b.confidence),
        trigger_event=trigger_event_id,
        source=["possession_transfer_induction"],
        status="unverified",
    )


def _possession_intervals(facts: Iterable[GraphFact]) -> list[GraphFact]:
    return [
        f
        for f in facts
        if f.fact_type == "relation"
        and f.object
        and _normalize_predicate(f.predicate) in POSSESSION_PREDICATES
    ]


def induce_segment_events(
    interval_facts: list[GraphFact],
    *,
    min_event_duration_sec: float = DEFAULT_MIN_EVENT_DURATION_SEC,
    min_event_confidence: float = DEFAULT_MIN_EVENT_CONFIDENCE,
    transfer_max_gap_sec: float = DEFAULT_TRANSFER_MAX_GAP_SEC,
) -> tuple[list[EventNode], list[StateTransition]]:
    """Induce EventNode + StateTransition records from interval facts."""
    events: list[EventNode] = []
    transitions: list[StateTransition] = []
    event_counter = 0
    transition_counter = 0

    # ── continuous_relation events ──
    for fact in interval_facts:
        if fact.fact_type != "relation":
            continue
        if _duration_sec(fact.time_span) < min_event_duration_sec:
            continue
        if fact.confidence < min_event_confidence:
            continue
        event_counter += 1
        events.append(_build_continuous_event(counter=event_counter, fact=fact))

    # ── possession_transfer events ──
    possession = _possession_intervals(interval_facts)
    by_object: dict[str, list[GraphFact]] = defaultdict(list)
    for f in possession:
        by_object[str(f.object)].append(f)

    for obj, group in by_object.items():
        group.sort(key=lambda f: hhmmss_to_secs(f.time_span[0]))
        for prev, cur in zip(group, group[1:]):
            if prev.subject == cur.subject:
                continue
            if prev.source_provenance != cur.source_provenance:
                continue
            gap = hhmmss_to_secs(cur.time_span[0]) - hhmmss_to_secs(prev.time_span[1])
            if gap < 0:
                gap = 0.0
            if gap > transfer_max_gap_sec:
                continue
            event_counter += 1
            transfer_event = _build_transfer_event(
                counter=event_counter, fact_a=prev, fact_b=cur
            )
            events.append(transfer_event)
            transition_counter += 1
            transitions.append(
                _build_holder_transition(
                    counter=transition_counter,
                    obj=obj,
                    fact_a=prev,
                    fact_b=cur,
                    trigger_event_id=transfer_event.event_id,
                )
            )

    return events, transitions


def induce_segment_events_dict(
    interval_facts: list[GraphFact],
    *,
    video_id: str,
    min_event_duration_sec: float = DEFAULT_MIN_EVENT_DURATION_SEC,
    min_event_confidence: float = DEFAULT_MIN_EVENT_CONFIDENCE,
    transfer_max_gap_sec: float = DEFAULT_TRANSFER_MAX_GAP_SEC,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events, transitions = induce_segment_events(
        interval_facts,
        min_event_duration_sec=min_event_duration_sec,
        min_event_confidence=min_event_confidence,
        transfer_max_gap_sec=transfer_max_gap_sec,
    )
    return segment_event_graph_to_dict(
        events,
        transitions,
        video_id=video_id,
        created_at=utc_now_iso(),
        config_snapshot={
            **(config_snapshot or {}),
            "min_event_duration_sec": min_event_duration_sec,
            "min_event_confidence": min_event_confidence,
            "transfer_max_gap_sec": transfer_max_gap_sec,
        },
    )


def event_status_unsupported() -> FactStatus:
    return "unsupported"
