"""Alg-plan §12 Agent-facing semantic tools (M9).

This module exposes the strict-safe tool surface for SGD-Agent v2.0:
QuestionDecompose, EntityResolve, RelationQuery, EventSearch,
TemporalOrderQuery, EvidenceTrace, VisualVerify, AnswerWithEvidence.

Every tool returns an envelope of the form::

    {
        "tool_name": <str>,
        "query": <dict>,
        "result": <tool-specific dict>,
    }

The ``GraphTools`` facade groups them so the Agent (M10) can hold one object
that owns the GraphStore and dispatches tool calls.
"""

from __future__ import annotations

from typing import Any

from sgd.agent.graph_store import (
    GraphStore,
    parse_time_constraint,
    span_to_secs,
    time_overlaps,
)
from sgd.agent.question_decompose import question_decompose
from sgd.agent.visual_verify import VisualVerifyAdapter
from sgd.scene_graph.query_global import (
    first_holder as global_first_holder,
    last_holder as global_last_holder,
)


_REQUIRED_ANSWER_KEYS = {"answer", "confidence", "evidence_chain"}
_REQUIRED_EVIDENCE_CHAIN_KEYS = {
    "global",
    "segment_events",
    "frame_facts",
    "support_frames",
    "visual_evidence",
    "verification",
}


class AnswerWithEvidenceError(ValueError):
    """Raised when AnswerWithEvidence is built without sufficient support."""


class GraphTools:
    """Facade exposing the strict-safe tool surface to the Agent."""

    def __init__(
        self,
        store: GraphStore,
        *,
        visual_verify: VisualVerifyAdapter | None = None,
    ) -> None:
        self._store = store
        self._visual_verify = visual_verify or VisualVerifyAdapter(store)

    # ── Tool 1: QuestionDecompose ────────────────────────────────────────

    def question_decompose(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return question_decompose(question, metadata=metadata)

    # ── Tool 2: EntityResolve ────────────────────────────────────────────

    def entity_resolve(
        self,
        referring_expression: str,
        *,
        time_constraint: Any = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        constraint = parse_time_constraint(time_constraint)
        matches = self._store.find_entities_by_alias(
            referring_expression,
            top_k=top_k,
            time_constraint=constraint,
        )
        candidates: list[dict[str, Any]] = []
        for m in matches:
            ent = self._store.entity_by_id(m.entity_id) or {}
            candidates.append(
                {
                    "entity_id": m.entity_id,
                    "score": round(m.score, 4),
                    "matched_alias": m.alias_text,
                    "aliases": [a.get("text") for a in m.aliases],
                    "category": ent.get("category"),
                    "first_seen": ent.get("first_seen"),
                    "last_seen": ent.get("last_seen"),
                    "support_frame_ids": m.support_frame_ids,
                    "confidence": ent.get("confidence"),
                }
            )
        return {
            "tool_name": "EntityResolve",
            "query": {
                "referring_expression": referring_expression,
                "time_constraint": time_constraint,
                "top_k": top_k,
            },
            "result": {
                "candidates": candidates,
                "missing_slots": ["entity"] if not candidates else [],
                "warnings": [],
            },
        }

    # ── Tool 3: RelationQuery ────────────────────────────────────────────

    def relation_query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        time_constraint: Any = None,
        prefer_interval: bool = True,
    ) -> dict[str, Any]:
        constraint = parse_time_constraint(time_constraint)
        facts = self._store.select_facts(
            subject=subject,
            predicate=predicate,
            obj=object,
            time_constraint=constraint,
            prefer_interval=prefer_interval,
        )
        return {
            "tool_name": "RelationQuery",
            "query": {
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "time_constraint": time_constraint,
            },
            "result": {
                "facts": facts,
                "missing_slots": [] if facts else ["matching_fact"],
                "warnings": [],
            },
        }

    # ── Tool 4: EventSearch ──────────────────────────────────────────────

    def event_search(
        self,
        *,
        event_type: str | None = None,
        participants: dict[str, str] | None = None,
        time_constraint: Any = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        constraint = parse_time_constraint(time_constraint)
        events = self._store.select_events(
            event_type=event_type,
            participants=participants,
            time_constraint=constraint,
        )
        events_sorted = sorted(events, key=lambda e: e.get("confidence", 0.0), reverse=True)
        return {
            "tool_name": "EventSearch",
            "query": {
                "event_type": event_type,
                "participants": participants,
                "time_constraint": time_constraint,
                "top_k": top_k,
            },
            "result": {
                "events": events_sorted[: max(1, top_k)],
                "missing_slots": [] if events_sorted else ["matching_event"],
                "warnings": [],
            },
        }

    # ── Tool 5: TemporalOrderQuery ───────────────────────────────────────

    def temporal_order_query(
        self,
        *,
        operator: str,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        event_type: str | None = None,
        participants: dict[str, str] | None = None,
        time_constraint: Any = None,
    ) -> dict[str, Any]:
        if operator not in {"first", "last", "before", "after"}:
            raise ValueError(f"unsupported operator: {operator!r}")
        constraint = parse_time_constraint(time_constraint)

        # Fast path: Stage3 v1 global graph indexes first/last holder per object.
        if (
            operator in {"first", "last"}
            and object is not None
            and constraint is None
            and self._store.has_global_graph()
        ):
            holder_lookup = global_last_holder if operator == "last" else global_first_holder
            picked = holder_lookup(self._store.global_graph(), object)
            if picked is not None and picked.get("entity_id"):
                event = picked.get("event") or {}
                result_payload = {
                    "kind": "global_holder",
                    "id": picked.get("event_id"),
                    "subject": picked.get("entity_id"),
                    "predicate": "holds",
                    "object": object,
                    "time_span": event.get("time_span"),
                    "confidence": event.get("confidence"),
                    "event_type": event.get("event_type"),
                    "participants": event.get("participants"),
                }
                return {
                    "tool_name": "TemporalOrderQuery",
                    "query": {
                        "operator": operator,
                        "subject": subject,
                        "predicate": predicate,
                        "object": object,
                        "event_type": event_type,
                        "participants": participants,
                        "time_constraint": time_constraint,
                    },
                    "result": {
                        "result": result_payload,
                        "missing_slots": [],
                        "warnings": ["resolved_via=global_graph_v1.first_last_index"],
                    },
                }

        # Choose the candidate pool: events if event_type/participants given,
        # otherwise facts.
        pool: list[dict[str, Any]]
        kind: str
        if event_type or participants:
            pool = self._store.select_events(
                event_type=event_type,
                participants=participants,
                time_constraint=constraint,
            )
            kind = "event"
        else:
            pool = self._store.select_facts(
                subject=subject,
                predicate=predicate,
                obj=object,
                time_constraint=constraint,
                prefer_interval=True,
            )
            kind = "fact"

        candidates_sorted = sorted(pool, key=lambda x: span_to_secs(x.get("time_span", ["", ""]))[0])
        warnings: list[str] = []
        if not candidates_sorted:
            return {
                "tool_name": "TemporalOrderQuery",
                "query": {
                    "operator": operator,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "event_type": event_type,
                    "participants": participants,
                    "time_constraint": time_constraint,
                },
                "result": {
                    "result": None,
                    "missing_slots": ["matching_candidate"],
                    "warnings": warnings,
                },
            }
        if operator == "first":
            pick = candidates_sorted[0]
        elif operator == "last":
            pick = candidates_sorted[-1]
        elif operator == "before":
            pick = candidates_sorted[0]
            warnings.append(
                "MVP 'before' returns the earliest candidate within the constraint; "
                "Stage3 v1 will refine this once available."
            )
        else:  # after
            pick = candidates_sorted[-1]
            warnings.append(
                "MVP 'after' returns the latest candidate within the constraint; "
                "Stage3 v1 will refine this once available."
            )

        result: dict[str, Any] = {
            "kind": kind,
            "id": pick.get("event_id") if kind == "event" else pick.get("fact_id"),
            "time_span": pick.get("time_span"),
            "confidence": pick.get("confidence"),
        }
        if kind == "event":
            result["event_type"] = pick.get("event_type")
            result["participants"] = pick.get("participants")
        else:
            result["subject"] = pick.get("subject")
            result["predicate"] = pick.get("predicate")
            result["object"] = pick.get("object")

        return {
            "tool_name": "TemporalOrderQuery",
            "query": {
                "operator": operator,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "event_type": event_type,
                "participants": participants,
                "time_constraint": time_constraint,
            },
            "result": {
                "result": result,
                "missing_slots": [],
                "warnings": warnings,
            },
        }

    # ── Tool 6: EvidenceTrace ────────────────────────────────────────────

    def evidence_trace(
        self,
        target_id: str,
        target_type: str,
    ) -> dict[str, Any]:
        if target_type not in {"fact", "event", "entity"}:
            raise ValueError(f"unsupported target_type: {target_type!r}")

        support_facts: list[str] = []
        support_frame_ids: list[str] = []
        time_span: list[str] | None = None
        confidence: float | None = None

        if target_type == "fact":
            fact = self._store.fact_by_id(target_id)
            if fact:
                support_facts = [fact["fact_id"]]
                support_frame_ids = list(fact.get("evidence") or [])
                time_span = fact.get("time_span")
                confidence = float(fact.get("confidence", 0.0))
        elif target_type == "event":
            event = self._store.event_by_id(target_id)
            if event:
                support_facts = list(event.get("support_facts") or [])
                support_frame_ids = list(event.get("evidence") or [])
                time_span = event.get("time_span")
                confidence = float(event.get("confidence", 0.0))
        else:  # entity
            ent = self._store.entity_by_id(target_id)
            if ent:
                support_frame_ids = list(ent.get("support_frame_ids") or [])
                time_span = [ent.get("first_seen"), ent.get("last_seen")]
                confidence = float(ent.get("confidence", 0.0))

        visual_evidence = self._store.frames_to_evidence_pointers(support_frame_ids)
        return {
            "tool_name": "EvidenceTrace",
            "query": {"target_id": target_id, "target_type": target_type},
            "result": {
                "target_id": target_id,
                "target_type": target_type,
                "time_span": time_span,
                "support_facts": support_facts,
                "support_frame_ids": support_frame_ids,
                "visual_evidence": visual_evidence,
                "confidence": confidence,
                "missing_slots": [] if (support_facts or support_frame_ids) else ["support"],
                "warnings": [],
            },
        }

    # ── Tool 7: VisualVerify ─────────────────────────────────────────────

    def visual_verify(
        self,
        *,
        claim: dict[str, Any],
        evidence_pointers: list[str],
    ) -> dict[str, Any]:
        return self._visual_verify(claim=claim, evidence_pointers=evidence_pointers)

    # ── Tool 8: AnswerWithEvidence ───────────────────────────────────────

    def answer_with_evidence(
        self,
        *,
        candidate_answer: str,
        supporting_subgraph: dict[str, Any],
        confidence: float | None = None,
        verification: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        events_ids: list[str] = list(supporting_subgraph.get("events") or [])
        facts_ids: list[str] = list(supporting_subgraph.get("facts") or [])
        global_paths: list[str] = list(supporting_subgraph.get("global") or [])

        if not events_ids and not facts_ids:
            raise AnswerWithEvidenceError(
                "AnswerWithEvidence requires at least one supporting event or fact"
            )

        segment_events: list[dict[str, Any]] = []
        for eid in events_ids:
            ev = self._store.event_by_id(eid)
            if ev is None:
                raise AnswerWithEvidenceError(f"unknown event_id in supporting_subgraph: {eid!r}")
            segment_events.append(
                {
                    "event_id": ev["event_id"],
                    "event_type": ev.get("event_type"),
                    "time_span": ev.get("time_span"),
                    "participants": ev.get("participants"),
                    "confidence": ev.get("confidence"),
                }
            )

        frame_facts_summary: list[dict[str, Any]] = []
        for fid in facts_ids:
            fact = self._store.fact_by_id(fid)
            if fact is None:
                raise AnswerWithEvidenceError(f"unknown fact_id in supporting_subgraph: {fid!r}")
            frame_facts_summary.append(
                {
                    "fact_id": fact["fact_id"],
                    "subject": fact.get("subject"),
                    "predicate": fact.get("predicate"),
                    "object": fact.get("object"),
                    "time_span": fact.get("time_span"),
                    "confidence": fact.get("confidence"),
                }
            )

        support_frame_ids = self._store.collect_support_frames(events_ids + facts_ids)
        if not support_frame_ids:
            raise AnswerWithEvidenceError(
                "AnswerWithEvidence requires at least one support frame in the evidence chain"
            )
        visual_evidence = self._store.frames_to_evidence_pointers(support_frame_ids)

        # Confidence defaults to the minimum of contributing confidences.
        if confidence is None:
            contributors = [
                *(e.get("confidence", 0.0) for e in segment_events),
                *(f.get("confidence", 0.0) for f in frame_facts_summary),
            ]
            confidence = float(min(contributors)) if contributors else 0.0

        envelope = {
            "tool_name": "AnswerWithEvidence",
            "query": {
                "candidate_answer": candidate_answer,
                "supporting_subgraph": supporting_subgraph,
            },
            "result": {
                "answer": candidate_answer,
                "confidence": float(confidence),
                "evidence_chain": {
                    "global": global_paths,
                    "segment_events": segment_events,
                    "frame_facts": frame_facts_summary,
                    "support_frames": support_frame_ids,
                    "visual_evidence": visual_evidence,
                    "verification": list(verification or []),
                },
            },
        }
        _validate_answer_envelope(envelope)
        return envelope


def _validate_answer_envelope(envelope: dict[str, Any]) -> None:
    result = envelope.get("result") or {}
    missing = _REQUIRED_ANSWER_KEYS - set(result.keys())
    if missing:
        raise AnswerWithEvidenceError(f"AnswerWithEvidence result missing keys: {sorted(missing)}")
    chain = result["evidence_chain"]
    missing_chain = _REQUIRED_EVIDENCE_CHAIN_KEYS - set(chain.keys())
    if missing_chain:
        raise AnswerWithEvidenceError(
            f"evidence_chain missing keys: {sorted(missing_chain)}"
        )
