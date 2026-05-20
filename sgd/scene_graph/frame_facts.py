"""Frame-fact extractor (M3).

Converts a Stage1 scene graph dict into a list of GraphFact records per
Alg-plan §5 + dev-plan M3. Each fact carries:

    fact_id, fact_type, subject, predicate, object, time_span, evidence,
    confidence, source, source_provenance, status.

MVP scope:
    - entity_presence facts: one per (entity_id, frame_id) appearance.
    - relation facts: one per (subject, predicate, object, frame_id) tuple.
    - attribute and local_action facts are deferred to v2.0 (Stage1 currently
      does not emit attribute key/value pairs or distinct local actions).

Temporal granularity:
    Each fact's time_span is the single-frame instant ``(t_sec, t_sec)``.
    Temporal consolidation into interval facts is M4's responsibility, not
    M3's. RelationQuery + EvidenceTrace consumers must therefore tolerate
    instantaneous facts.

Provenance:
    All facts produced from Stage1 carry source_provenance="vlm_structured",
    on the assumption that Stage1 caption fallback is disabled in strict
    construction. Callers should validate this upstream; the audit script
    (M0) will catch leaks at the artifact boundary.
"""

from __future__ import annotations

from typing import Any

from sgd.scene_graph.records import (
    FactStatus,
    GraphFact,
    SourceProvenance,
    frame_facts_to_dict,
    secs_to_hhmmss,
)
from sgd.scene_graph.schema import safe_load_stage1_graph, utc_now_iso


def _fact_id(counter: int) -> str:
    return f"fact_{counter:06d}"


def _build_presence_fact(
    *,
    counter: int,
    entity_id: str,
    frame_id: str,
    t_sec: float,
    confidence: float,
    provenance: SourceProvenance,
    status: FactStatus,
) -> GraphFact:
    ts = secs_to_hhmmss(t_sec)
    return GraphFact(
        fact_id=_fact_id(counter),
        fact_type="entity_presence",
        subject=entity_id,
        predicate="present",
        object=None,
        time_span=(ts, ts),
        evidence=[frame_id],
        confidence=confidence,
        source=["stage1_entity_extraction"],
        source_provenance=provenance,
        status=status,
    )


def _build_relation_fact(
    *,
    counter: int,
    subject: str,
    predicate: str,
    obj: str,
    frame_id: str,
    t_sec: float,
    confidence: float,
    provenance: SourceProvenance,
    status: FactStatus,
) -> GraphFact:
    ts = secs_to_hhmmss(t_sec)
    return GraphFact(
        fact_id=_fact_id(counter),
        fact_type="relation",
        subject=subject,
        predicate=predicate,
        object=obj,
        time_span=(ts, ts),
        evidence=[frame_id],
        confidence=confidence,
        source=["stage1_relation_extraction"],
        source_provenance=provenance,
        status=status,
    )


def build_frame_facts(
    stage1_graph: dict[str, Any],
    *,
    include_presence: bool = True,
    default_provenance: SourceProvenance = "vlm_structured",
    default_status: FactStatus = "unverified",
) -> list[GraphFact]:
    """Convert Stage1 graph dict into a flat list of GraphFact records.

    Facts are emitted in temporal order (sorted by frame timestamp, then by
    extraction order within the frame). Fact IDs are assigned monotonically.

    Args:
        stage1_graph: a stage1_v1 dict with `frames` containing per-frame
            entities and relations.
        include_presence: emit entity_presence facts in addition to relations.
        default_provenance: provenance tag applied to every emitted fact.
            Callers must override only when Stage1's source is known to differ
            (e.g. transcript_fallback or manual_annotation).
        default_status: status applied to every emitted fact. Verification
            (M7) flips facts to verified/rejected/unsupported.
    """
    frames = list(stage1_graph.get("frames", []) or [])
    frames.sort(key=lambda f: float(f.get("t_sec", 0.0)))

    facts: list[GraphFact] = []
    counter = 0

    for frame in frames:
        frame_id = str(frame.get("frame_id", ""))
        if not frame_id:
            continue
        t_sec = float(frame.get("t_sec", 0.0))

        if include_presence:
            for ent in frame.get("entities", []) or []:
                ent_id = str(ent.get("entity_id", ""))
                if not ent_id:
                    continue
                counter += 1
                facts.append(
                    _build_presence_fact(
                        counter=counter,
                        entity_id=ent_id,
                        frame_id=frame_id,
                        t_sec=t_sec,
                        confidence=float(ent.get("confidence", 0.0)),
                        provenance=default_provenance,
                        status=default_status,
                    )
                )

        for rel in frame.get("relations", []) or []:
            s = str(rel.get("s", ""))
            p = str(rel.get("p", ""))
            o = str(rel.get("o", ""))
            if not s or not p or not o:
                continue
            counter += 1
            facts.append(
                _build_relation_fact(
                    counter=counter,
                    subject=s,
                    predicate=p,
                    obj=o,
                    frame_id=frame_id,
                    t_sec=t_sec,
                    confidence=float(rel.get("confidence", 0.0)),
                    provenance=default_provenance,
                    status=default_status,
                )
            )

    return facts


def build_frame_facts_dict(
    stage1_graph: dict[str, Any],
    *,
    include_presence: bool = True,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    facts = build_frame_facts(stage1_graph, include_presence=include_presence)
    return frame_facts_to_dict(
        facts,
        video_id=str(stage1_graph.get("video_id", "")),
        created_at=utc_now_iso(),
        config_snapshot=config_snapshot,
    )


def build_frame_facts_from_path(stage1_path: str) -> dict[str, Any] | None:
    graph = safe_load_stage1_graph(stage1_path)
    if graph is None:
        return None
    return build_frame_facts_dict(graph)
