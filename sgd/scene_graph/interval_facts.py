"""Temporal fact consolidation (M4).

Consolidates instantaneous frame facts (time_span ``(t, t)``) into interval
facts (time_span ``(t_start, t_end)``) per Alg-plan §9.5 + dev-plan M4.

Algorithm:
    1. Group frame facts by (fact_type, subject, predicate, object).
    2. Sort each group by start time.
    3. Merge adjacent observations whose temporal gap is at most
       ``max_gap_sec``.
    4. Drop intervals shorter than ``min_duration_sec``.
    5. Aggregate confidence (max), concatenate support frame ids, and
       propagate ``source_provenance``. Caller is responsible for ensuring all
       facts in a group share a provenance — mixing provenances inside one
       interval is rejected.

The output is a new flat list of GraphFact records with fresh fact_ids in the
``ifact_*`` namespace, ``source`` extended with ``"interval_consolidation"``,
and ``status="unverified"`` until M7 verifies them.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sgd.scene_graph.records import (
    FactStatus,
    GraphFact,
    SourceProvenance,
    frame_facts_to_dict,
    hhmmss_to_secs,
    secs_to_hhmmss,
)
from sgd.scene_graph.schema import utc_now_iso


DEFAULT_MAX_GAP_SEC = 2.0
DEFAULT_MIN_DURATION_SEC = 0.0


def _interval_id(counter: int) -> str:
    return f"ifact_{counter:06d}"


def _group_key(fact: GraphFact) -> tuple[str, str, str, str]:
    return (
        fact.fact_type,
        fact.subject,
        fact.predicate,
        fact.object or "",
    )


def consolidate_facts(
    facts: list[GraphFact],
    *,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
) -> list[GraphFact]:
    """Consolidate instantaneous facts into interval facts."""
    grouped: dict[tuple[str, str, str, str], list[GraphFact]] = defaultdict(list)
    for fact in facts:
        grouped[_group_key(fact)].append(fact)

    intervals: list[GraphFact] = []
    counter = 0

    for key, group in grouped.items():
        fact_type, subject, predicate, obj_str = key
        obj = obj_str or None

        group.sort(key=lambda f: hhmmss_to_secs(f.time_span[0]))

        cur_start: float | None = None
        cur_end: float | None = None
        cur_conf: float = 0.0
        cur_provenance: SourceProvenance | None = None
        cur_evidence: list[str] = []
        cur_sources: set[str] = set()

        def flush():
            nonlocal counter
            if cur_start is None or cur_end is None or cur_provenance is None:
                return
            if (cur_end - cur_start) < min_duration_sec:
                return
            counter += 1
            sources = sorted(cur_sources | {"interval_consolidation"})
            intervals.append(
                GraphFact(
                    fact_id=_interval_id(counter),
                    fact_type=fact_type,  # type: ignore[arg-type]
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    time_span=(secs_to_hhmmss(cur_start), secs_to_hhmmss(cur_end)),
                    evidence=list(cur_evidence),
                    confidence=cur_conf,
                    source=sources,
                    source_provenance=cur_provenance,
                    status="unverified",
                )
            )

        for fact in group:
            t_start = hhmmss_to_secs(fact.time_span[0])
            t_end = hhmmss_to_secs(fact.time_span[1])

            if cur_start is None:
                cur_start = t_start
                cur_end = t_end
                cur_conf = fact.confidence
                cur_provenance = fact.source_provenance
                cur_evidence = list(fact.evidence)
                cur_sources = set(fact.source)
                continue

            if fact.source_provenance != cur_provenance:
                raise ValueError(
                    "consolidate_facts cannot merge facts with different "
                    f"source_provenance ({cur_provenance!r} vs "
                    f"{fact.source_provenance!r}). Group by provenance first."
                )

            gap = t_start - (cur_end or t_start)
            if gap <= max_gap_sec:
                # Extend the current interval.
                cur_end = max(cur_end or t_start, t_end)
                cur_conf = max(cur_conf, fact.confidence)
                for fid in fact.evidence:
                    if fid not in cur_evidence:
                        cur_evidence.append(fid)
                cur_sources.update(fact.source)
            else:
                flush()
                cur_start = t_start
                cur_end = t_end
                cur_conf = fact.confidence
                cur_provenance = fact.source_provenance
                cur_evidence = list(fact.evidence)
                cur_sources = set(fact.source)

        flush()

    return intervals


def consolidate_facts_dict(
    facts: list[GraphFact],
    *,
    video_id: str,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intervals = consolidate_facts(
        facts,
        max_gap_sec=max_gap_sec,
        min_duration_sec=min_duration_sec,
    )
    payload = frame_facts_to_dict(
        intervals,
        video_id=video_id,
        created_at=utc_now_iso(),
        config_snapshot={
            **(config_snapshot or {}),
            "max_gap_sec": max_gap_sec,
            "min_duration_sec": min_duration_sec,
        },
    )
    payload["version"] = "interval_facts_v1"
    return payload


def fact_status_for_unsupported() -> FactStatus:
    """Marker for downstream M7 verifier; kept here so consumers share a constant."""
    return "unsupported"
