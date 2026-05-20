"""Entity registry builder (M2).

Consumes a Stage1 scene graph dict and emits a strict-mode EntityRecord
registry per Alg-plan §4 + dev-plan M2.

Strict-mode rule:
    Only aliases marked ``strict_safe=true`` may surface to the Agent. Aliases
    that look like free-form narrative phrases ("the man who argues with the
    woman", "the villain who later breaks the mirror") are kept on the record
    but marked ``strict_safe=false`` so they cannot leak into inference.

Provenance assignment for MVP:
    canonical_name -> Alias(source_provenance="graph_attribute") if the name
      matches a controlled category vocabulary (person, object, ...), else
      ``vlm_structured``.
    extra aliases -> ``vlm_structured`` unless a heuristic flags them as
      narrative phrases, in which case they remain on the record but with
      ``strict_safe=false``. We never fabricate a ``caption_fallback`` tag
      from this side — Stage1 is expected to disable caption fallback in
      strict mode (``SCENE_GRAPH_STAGE1_CAPTION_FALLBACK = False``). Any
      caption-derived alias must be tagged at the source.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from sgd.scene_graph.records import (
    Alias,
    EntityCategory,
    EntityRecord,
    SourceProvenance,
    entity_registry_to_dict,
    secs_to_hhmmss,
)
from sgd.scene_graph.schema import safe_load_stage1_graph
from sgd.scene_graph.schema import utc_now_iso


# Controlled category vocabulary mirrored from Stage1's ALLOWED_ENTITY_TYPES.
ALLOWED_CATEGORIES: set[str] = {
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
}

# Lowercase tokens that indicate a free-form narrative alias.
NARRATIVE_TOKENS: set[str] = {
    "who",
    "whom",
    "whose",
    "which",
    "later",
    "earlier",
    "after",
    "before",
    "when",
    "while",
    "until",
    "because",
    "since",
    "then",
    "afterward",
    "afterwards",
}

MAX_STRICT_SAFE_TOKENS = 6


def _is_narrative_phrase(text: str) -> bool:
    """Return True if the alias text looks like a narrative phrase.

    Narrative aliases are blocked from strict inference because they encode
    storyline information that captures arrived through caption / VLM text.
    """
    lowered = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z]+", lowered)
    if not tokens:
        return True
    if any(tok in NARRATIVE_TOKENS for tok in tokens):
        return True
    if len(tokens) > MAX_STRICT_SAFE_TOKENS:
        return True
    return False


def _normalize_category(raw_type: str) -> EntityCategory:
    t = (raw_type or "").strip().lower()
    if t in ALLOWED_CATEGORIES:
        return t  # type: ignore[return-value]
    return "object"


def _build_alias(
    text: str,
    *,
    source: str,
    provenance: SourceProvenance,
    support_frame_ids: list[str],
    force_unsafe: bool = False,
) -> Alias:
    strict_safe = (not force_unsafe) and (not _is_narrative_phrase(text))
    # Caption- or transcript-derived aliases are never strict-safe regardless
    # of phrase shape.
    if provenance in {"caption_fallback", "transcript_fallback"}:
        strict_safe = False
    return Alias(
        text=text.strip(),
        source=source,
        source_provenance=provenance,
        strict_safe=strict_safe,
        support_frame_ids=list(support_frame_ids),
    )


def build_entity_record(
    stage1_entity: dict[str, Any],
    *,
    frame_time_lookup: dict[str, float],
) -> EntityRecord:
    """Convert a single Stage1 entity dict into an EntityRecord."""
    entity_id = str(stage1_entity["entity_id"])
    canonical_name = str(stage1_entity.get("canonical_name", "")).strip()
    category = _normalize_category(str(stage1_entity.get("type", "object")))
    raw_aliases = stage1_entity.get("aliases", []) or []
    evidence_frames = list(stage1_entity.get("evidence", []) or [])
    first_seen_secs = float(stage1_entity.get("first_seen", 0.0))
    last_seen_secs = float(stage1_entity.get("last_seen", first_seen_secs))
    confidence = float(stage1_entity.get("confidence", 0.0))

    # Sort evidence frames by their timestamp to make support_frame_ids stable.
    evidence_frames_sorted = sorted(
        set(evidence_frames),
        key=lambda fid: frame_time_lookup.get(fid, 0.0),
    )

    aliases: list[Alias] = []

    if canonical_name:
        # Canonical name itself is exposed as an alias so the agent can refer
        # to entities by name. If the canonical happens to BE the category
        # noun (e.g. "person"), label its provenance as graph_attribute.
        canon_prov: SourceProvenance = (
            "graph_attribute"
            if canonical_name.lower() in ALLOWED_CATEGORIES
            else "vlm_structured"
        )
        aliases.append(
            _build_alias(
                canonical_name,
                source="stage1_entity_extraction",
                provenance=canon_prov,
                support_frame_ids=evidence_frames_sorted,
            )
        )

    seen_texts = {canonical_name.lower()} if canonical_name else set()
    for raw in raw_aliases:
        text = str(raw).strip()
        if not text or text.lower() in seen_texts:
            continue
        seen_texts.add(text.lower())
        aliases.append(
            _build_alias(
                text,
                source="stage1_entity_extraction",
                provenance="vlm_structured",
                support_frame_ids=evidence_frames_sorted,
            )
        )

    return EntityRecord(
        entity_id=entity_id,
        category=category,
        aliases=aliases,
        first_seen=secs_to_hhmmss(first_seen_secs),
        last_seen=secs_to_hhmmss(last_seen_secs),
        support_frame_ids=evidence_frames_sorted,
        source=["stage1_entity_extraction"],
        confidence=confidence,
    )


def build_entity_registry(
    stage1_graph: dict[str, Any],
) -> list[EntityRecord]:
    """Build the full EntityRecord list from a Stage1 graph dict."""
    frames = stage1_graph.get("frames", []) or []
    frame_time_lookup: dict[str, float] = {}
    for frame in frames:
        fid = str(frame.get("frame_id", ""))
        if fid:
            frame_time_lookup[fid] = float(frame.get("t_sec", 0.0))

    records: list[EntityRecord] = []
    for ent in stage1_graph.get("entities", []) or []:
        if not isinstance(ent, dict):
            continue
        if "entity_id" not in ent or "canonical_name" not in ent:
            continue
        records.append(build_entity_record(ent, frame_time_lookup=frame_time_lookup))
    return records


def build_entity_registry_dict(
    stage1_graph: dict[str, Any],
    *,
    include_unsafe_aliases: bool = True,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = build_entity_registry(stage1_graph)
    if not include_unsafe_aliases:
        records = [
            EntityRecord(
                entity_id=rec.entity_id,
                category=rec.category,
                aliases=rec.strict_safe_aliases(),
                first_seen=rec.first_seen,
                last_seen=rec.last_seen,
                support_frame_ids=rec.support_frame_ids,
                source=rec.source,
                confidence=rec.confidence,
                states_over_time=rec.states_over_time,
                relations_over_time=rec.relations_over_time,
                appearance=rec.appearance,
                tracklets=rec.tracklets,
            )
            for rec in records
        ]
    return entity_registry_to_dict(
        records,
        video_id=str(stage1_graph.get("video_id", "")),
        created_at=utc_now_iso(),
        config_snapshot=config_snapshot,
    )


def build_entity_registry_from_path(stage1_path: str) -> dict[str, Any] | None:
    graph = safe_load_stage1_graph(stage1_path)
    if graph is None:
        return None
    return build_entity_registry_dict(graph)


def strict_safe_aliases(entity_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the strict-safe aliases of an EntityRecord dict.

    Useful for downstream tool wrappers (EntityResolve) to enforce that strict
    inference never surfaces aliases with strict_safe=false.
    """
    return [
        a for a in (entity_dict.get("aliases") or [])
        if isinstance(a, dict) and a.get("strict_safe", False)
    ]


def iter_strict_safe_aliases(
    records: Iterable[EntityRecord],
) -> Iterable[tuple[str, Alias]]:
    for rec in records:
        for alias in rec.strict_safe_aliases():
            yield rec.entity_id, alias
