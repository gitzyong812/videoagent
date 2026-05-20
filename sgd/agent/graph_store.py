"""GraphStore — the canonical read surface for Agent-facing tools.

Loads the strict-safe inference artifact bundle produced by M8 and serves
lookups used by EntityResolve, RelationQuery, EventSearch, TemporalOrderQuery,
EvidenceTrace, and AnswerWithEvidence.

Strict-mode invariants enforced at load time:
    - Entity aliases must have ``strict_safe == true`` to surface from
      ``find_entities_by_alias``; non-strict-safe aliases are silently
      filtered (callers never see them).
    - All facts and events are loaded with their full provenance fields so
      downstream tools can refuse to use rejected / unsupported records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sgd.scene_graph.pipeline import audit_inference_artifacts, load_inference_artifacts
from sgd.scene_graph.records import hhmmss_to_secs, secs_to_hhmmss


TimeSpanSecs = tuple[float, float]
TimeSpanStr = tuple[str, str]


def parse_time_constraint(value: Any) -> TimeSpanSecs | None:
    """Accept ``None``, ``"HH:MM:SS"`` / ``"HH:MM:SS-HH:MM:SS"``, ``[a, b]``."""
    if value is None or value == "" or value == "whole_video":
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (_to_secs(value[0]), _to_secs(value[1]))
    if isinstance(value, str):
        if "-" in value:
            a, b = value.split("-", 1)
            return (_to_secs(a.strip()), _to_secs(b.strip()))
        # Single timestamp -> point span (caller must broaden if needed).
        t = _to_secs(value)
        return (t, t)
    raise ValueError(f"Unrecognized time_constraint shape: {value!r}")


def _to_secs(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return hhmmss_to_secs(str(v))


def time_overlaps(a: TimeSpanSecs | None, b: TimeSpanSecs) -> bool:
    if a is None:
        return True
    return not (a[1] < b[0] or b[1] < a[0])


def span_to_secs(time_span: list[str] | tuple[str, str]) -> TimeSpanSecs:
    return (_to_secs(time_span[0]), _to_secs(time_span[1]))


@dataclass
class AliasMatch:
    entity_id: str
    alias_text: str
    score: float
    aliases: list[dict[str, Any]]  # all strict_safe aliases for the entity
    support_frame_ids: list[str]


class GraphStore:
    """In-memory index over the inference artifact bundle."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self._bundle = bundle
        self._entities: list[dict[str, Any]] = list(
            (bundle.get("entity_registry") or {}).get("entities") or []
        )
        self._entity_by_id: dict[str, dict[str, Any]] = {
            str(e.get("entity_id")): e for e in self._entities if e.get("entity_id")
        }
        self._frame_facts: list[dict[str, Any]] = list(
            (bundle.get("frame_facts") or {}).get("facts") or []
        )
        self._interval_facts: list[dict[str, Any]] = list(
            (bundle.get("interval_facts") or {}).get("facts") or []
        )
        self._events: list[dict[str, Any]] = list(
            (bundle.get("segment_events") or {}).get("events") or []
        )
        self._state_transitions: list[dict[str, Any]] = list(
            (bundle.get("segment_events") or {}).get("state_transitions") or []
        )
        self._global_graph: dict[str, Any] = bundle.get("global_graph") or {}
        self._temporal_units: dict[str, Any] = bundle.get("temporal_units") or {}
        self._frame_lookup: dict[str, dict[str, Any]] = {
            str(f.get("frame_id")): f for f in self._temporal_units.get("frames") or []
        }
        self._fact_by_id: dict[str, dict[str, Any]] = {}
        for fact in self._frame_facts:
            self._fact_by_id[str(fact.get("fact_id"))] = fact
        for fact in self._interval_facts:
            self._fact_by_id[str(fact.get("fact_id"))] = fact
        self._event_by_id: dict[str, dict[str, Any]] = {
            str(e.get("event_id")): e for e in self._events if e.get("event_id")
        }

    # ── Loaders ────────────────────────────────────────────────────────────

    @classmethod
    def from_dir(cls, base_dir: str, *, audit: bool = True) -> "GraphStore":
        if audit:
            audit_inference_artifacts(base_dir)
        bundle = load_inference_artifacts(base_dir)
        return cls(bundle)

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> "GraphStore":
        return cls(bundle)

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def video_id(self) -> str:
        return str((self._bundle.get("stripped_index") or {}).get("video_id", ""))

    def entities(self) -> list[dict[str, Any]]:
        return list(self._entities)

    def entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        return self._entity_by_id.get(entity_id)

    def frame_facts(self) -> list[dict[str, Any]]:
        return list(self._frame_facts)

    def interval_facts(self) -> list[dict[str, Any]]:
        return list(self._interval_facts)

    def fact_by_id(self, fact_id: str) -> dict[str, Any] | None:
        return self._fact_by_id.get(fact_id)

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def event_by_id(self, event_id: str) -> dict[str, Any] | None:
        return self._event_by_id.get(event_id)

    def state_transitions(self) -> list[dict[str, Any]]:
        return list(self._state_transitions)

    def global_graph(self) -> dict[str, Any]:
        return dict(self._global_graph)

    def has_global_graph(self) -> bool:
        return bool(self._global_graph and self._global_graph.get("version"))

    def frame_record(self, frame_id: str) -> dict[str, Any] | None:
        return self._frame_lookup.get(frame_id)

    # ── Strict-safe alias lookup ──────────────────────────────────────────

    def find_entities_by_alias(
        self,
        text: str,
        *,
        top_k: int = 5,
        time_constraint: TimeSpanSecs | None = None,
    ) -> list[AliasMatch]:
        """Score entity candidates by alias text similarity.

        Only strict_safe aliases participate in the match. Non-strict-safe
        aliases never surface from this method, so Agent observations cannot
        be polluted by narrative-style aliases.
        """
        normalized = _normalize_text(text)
        if not normalized:
            return []
        matches: list[AliasMatch] = []
        for ent in self._entities:
            safe_aliases = [
                a
                for a in (ent.get("aliases") or [])
                if isinstance(a, dict) and a.get("strict_safe")
            ]
            if not safe_aliases:
                continue
            # Optional time filter via entity's first_seen / last_seen.
            if time_constraint is not None:
                entity_span = (
                    _to_secs(ent.get("first_seen", "00:00:00.000")),
                    _to_secs(ent.get("last_seen", "00:00:00.000")),
                )
                if not time_overlaps(time_constraint, entity_span):
                    continue
            best_score = 0.0
            best_alias = ""
            for alias in safe_aliases:
                score = _alias_score(normalized, _normalize_text(alias.get("text", "")))
                if score > best_score:
                    best_score = score
                    best_alias = alias.get("text", "")
            if best_score <= 0.0:
                continue
            matches.append(
                AliasMatch(
                    entity_id=str(ent["entity_id"]),
                    alias_text=best_alias,
                    score=best_score,
                    aliases=safe_aliases,
                    support_frame_ids=list(ent.get("support_frame_ids") or []),
                )
            )
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[: max(1, top_k)]

    # ── Filtered fact / event lookups ─────────────────────────────────────

    def select_facts(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        time_constraint: TimeSpanSecs | None = None,
        prefer_interval: bool = True,
    ) -> list[dict[str, Any]]:
        pool = self._interval_facts if prefer_interval else self._frame_facts
        if prefer_interval and not pool:
            pool = self._frame_facts
        return [
            f
            for f in pool
            if _fact_matches(f, subject, predicate, obj)
            and time_overlaps(time_constraint, span_to_secs(f.get("time_span", ["", ""])))
        ]

    def select_events(
        self,
        *,
        event_type: str | None = None,
        participants: dict[str, str] | None = None,
        time_constraint: TimeSpanSecs | None = None,
    ) -> list[dict[str, Any]]:
        return [
            e
            for e in self._events
            if _event_matches(e, event_type, participants)
            and time_overlaps(time_constraint, span_to_secs(e.get("time_span", ["", ""])))
        ]

    # ── Evidence collection ──────────────────────────────────────────────

    def collect_support_frames(self, fact_or_event_ids: Iterable[str]) -> list[str]:
        out: list[str] = []
        for tid in fact_or_event_ids:
            fact = self._fact_by_id.get(tid)
            if fact:
                for fid in fact.get("evidence") or []:
                    if fid not in out:
                        out.append(fid)
            event = self._event_by_id.get(tid)
            if event:
                for fid in event.get("evidence") or []:
                    if fid not in out:
                        out.append(fid)
        return out

    def frames_to_evidence_pointers(self, frame_ids: Iterable[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for fid in frame_ids:
            rec = self._frame_lookup.get(fid) or {}
            out.append(
                {
                    "frame_id": fid,
                    "timestamp": rec.get("t") or secs_to_hhmmss(rec.get("t_sec", 0.0)),
                }
            )
        return out


def _fact_matches(
    fact: dict[str, Any],
    subject: str | None,
    predicate: str | None,
    obj: str | None,
) -> bool:
    if subject and str(fact.get("subject")) != subject:
        return False
    if predicate and str(fact.get("predicate")) != predicate:
        return False
    if obj and (fact.get("object") or "") != obj:
        return False
    return True


def _event_matches(
    event: dict[str, Any],
    event_type: str | None,
    participants: dict[str, str] | None,
) -> bool:
    if event_type and str(event.get("event_type")) != event_type:
        return False
    if participants:
        e_parts = event.get("participants") or {}
        for role, ent_id in participants.items():
            if str(e_parts.get(role, "")) != str(ent_id):
                return False
    return True


def _normalize_text(s: str) -> str:
    return " ".join((s or "").lower().split())


def _alias_score(query: str, alias: str) -> float:
    if not query or not alias:
        return 0.0
    if query == alias:
        return 1.0
    if query in alias or alias in query:
        return 0.85
    q_tokens = set(query.split())
    a_tokens = set(alias.split())
    if not q_tokens or not a_tokens:
        return 0.0
    overlap = len(q_tokens & a_tokens)
    if not overlap:
        return 0.0
    return 0.5 * overlap / max(len(q_tokens), len(a_tokens))
