"""v2 record dataclasses for the SGD-Agent graph memory.

These records are the canonical inference-time data shapes for SGD-Agent v2.0.
They replace the loose dict schemas in `schema.py`. Every record carries enough
provenance and evidence linkage to satisfy Strict Graph-only inference:

    - EntityRecord aliases must declare source_provenance and strict_safe.
    - GraphFact must declare source_provenance, status, and support_frame_ids.

The legacy `schema.py` validators are retained only for the construction-time
Stage1 / Stage2 dicts produced by the old builders. New code reads/writes these
records.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal


SourceProvenance = Literal[
    "vlm_structured",
    "graph_attribute",
    "caption_fallback",
    "transcript_fallback",
    "rule_induction",
    "manual_annotation",
]

FactType = Literal[
    "entity_presence",
    "attribute",
    "relation",
    "local_action",
]

FactStatus = Literal[
    "unverified",
    "verified",
    "rejected",
    "unsupported",
]

EntityCategory = Literal[
    "person",
    "object",
    "scene",
    "animal",
    "vehicle",
    "food",
    "text",
    "location",
    "building",
    "clothing",
    "instrument",
    "plant",
]


def secs_to_hhmmss(t: float) -> str:
    if t is None:
        return "00:00:00.000"
    t = max(0.0, float(t))
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = t - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def hhmmss_to_secs(s: str) -> float:
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS.sss, got {s!r}")
    h, m, sec = parts
    return int(h) * 3600 + int(m) * 60 + float(sec)


@dataclass
class Alias:
    """A single referring expression for an entity.

    Strict graph-only inference may only read aliases with strict_safe=true.
    Captions, narrations, and other free-text-derived aliases must mark
    strict_safe=false even when they remain in the construction artifact.
    """

    text: str
    source: str
    source_provenance: SourceProvenance
    strict_safe: bool
    support_frame_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class EntityRecord:
    """A persistent entity in E_video.

    MVP fields: entity_id, category, aliases, time spans, support_frame_ids,
    source, confidence. v2.0 full adds bbox-level tracklets, masks, crops, and
    visual embeddings — those live in tracklets/appearance fields kept empty
    in MVP.
    """

    entity_id: str
    category: EntityCategory
    aliases: list[Alias]
    first_seen: str  # HH:MM:SS.sss
    last_seen: str
    support_frame_ids: list[str]
    source: list[str]
    confidence: float
    # MVP-optional state/relation timelines (populated by M5 / M6).
    states_over_time: list[dict[str, Any]] = field(default_factory=list)
    relations_over_time: list[dict[str, Any]] = field(default_factory=list)
    # v2.0 full (kept empty in MVP).
    appearance: dict[str, Any] = field(default_factory=dict)
    tracklets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["aliases"] = [a.to_dict() if isinstance(a, Alias) else a for a in self.aliases]
        return out

    def strict_safe_aliases(self) -> list[Alias]:
        return [a for a in self.aliases if a.strict_safe]


@dataclass
class GraphFact:
    """An atomic, temporally grounded fact.

    Every fact carries fact_id, time_span, support_frame_ids (via `evidence`),
    confidence, source_provenance, and status. RelationQuery, AttributeQuery,
    and EvidenceTrace return GraphFact records.
    """

    fact_id: str
    fact_type: FactType
    subject: str
    predicate: str
    object: str | None
    time_span: tuple[str, str]
    evidence: list[str]
    confidence: float
    source: list[str]
    source_provenance: SourceProvenance
    status: FactStatus = "unverified"

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["time_span"] = [self.time_span[0], self.time_span[1]]
        return out


@dataclass
class EventSignature:
    """Structured event signature — replaces free-form Stage2 ``summary``.

    Strict graph-only inference may surface event_signature objects but
    never free-form natural-language event sentences. Always emit this as a
    nested object, not a stringified description.
    """

    event_type: str
    participants: dict[str, str]
    preconditions: list[str]
    effects: list[str]
    time_span: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "participants": dict(self.participants),
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "time_span": [self.time_span[0], self.time_span[1]],
        }


@dataclass
class EventNode:
    """A segment-level event node per Alg-plan §6.3."""

    event_id: str
    event_type: str
    time_span: tuple[str, str]
    participants: dict[str, str]
    event_signature: EventSignature
    preconditions: list[dict[str, Any]]
    effects: list[dict[str, Any]]
    core_facts: list[str]
    support_facts: list[str]
    evidence: list[str]
    confidence: float
    source: list[str]
    source_provenance: SourceProvenance
    status: FactStatus = "unverified"

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["event_signature"] = self.event_signature.to_dict()
        out["time_span"] = [self.time_span[0], self.time_span[1]]
        return out


@dataclass
class StateTransition:
    """A state-transition record per Alg-plan §6.4.

    ``from_value`` and ``to_value`` use the safe attribute names (``from`` is
    a Python reserved word). On dict serialization we emit ``from`` and ``to``
    to match Alg-plan §6.4.
    """

    transition_id: str
    entity: str
    attribute: str
    from_value: str
    to_value: str
    time_span: tuple[str, str]
    support_facts: list[str]
    confidence: float
    trigger_event: str | None = None
    source: list[str] = field(default_factory=list)
    status: FactStatus = "unverified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "entity": self.entity,
            "attribute": self.attribute,
            "from": self.from_value,
            "to": self.to_value,
            "trigger_event": self.trigger_event,
            "time_span": [self.time_span[0], self.time_span[1]],
            "support_facts": list(self.support_facts),
            "confidence": self.confidence,
            "source": list(self.source),
            "status": self.status,
        }


def segment_event_graph_to_dict(
    events: list[EventNode],
    transitions: list[StateTransition],
    *,
    video_id: str,
    created_at: str,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": "segment_events_v1",
        "video_id": video_id,
        "created_at": created_at,
        "config": config_snapshot or {},
        "events": [e.to_dict() for e in events],
        "state_transitions": [t.to_dict() for t in transitions],
        "stats": {
            "num_events": len(events),
            "num_state_transitions": len(transitions),
            "by_event_type": _count_by(events, lambda e: e.event_type),
        },
    }


def entity_registry_to_dict(
    entities: list[EntityRecord],
    *,
    video_id: str,
    created_at: str,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": "entity_registry_v1",
        "video_id": video_id,
        "created_at": created_at,
        "config": config_snapshot or {},
        "entities": [e.to_dict() for e in entities],
        "stats": {
            "num_entities": len(entities),
            "num_strict_safe_aliases": sum(len(e.strict_safe_aliases()) for e in entities),
        },
    }


def frame_facts_to_dict(
    facts: list[GraphFact],
    *,
    video_id: str,
    created_at: str,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": "frame_facts_v1",
        "video_id": video_id,
        "created_at": created_at,
        "config": config_snapshot or {},
        "facts": [f.to_dict() for f in facts],
        "stats": {
            "num_facts": len(facts),
            "by_type": _count_by(facts, lambda f: f.fact_type),
            "by_status": _count_by(facts, lambda f: f.status),
            "by_provenance": _count_by(facts, lambda f: f.source_provenance),
        },
    }


def _count_by(items, key) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        k = str(key(item))
        out[k] = out.get(k, 0) + 1
    return out
