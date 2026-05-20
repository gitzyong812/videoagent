"""Stage3 v1 — Global Graph builder (M6).

Builds the minimal global graph required for long-range temporal reasoning per
Alg-plan §7 + §0.7 + dev-plan M6.

G_global_v1 components (MVP scope only):
    - entity_lifecycle_index: entity_id -> {first_seen, last_seen, support_frame_ids}
    - global_events: list of event refs (same shape as segment_events.events)
    - temporal_edges: pairwise before / overlap edges between events
    - first_last_index:
        by_event_type: {event_type: {"first": event_id, "last": event_id}}
        by_object:    {object_id:  {"first_holder": event_id,
                                    "last_holder":  event_id}}
    - state_carry_over: simple holder / location records derived from
      state_transitions

Out of MVP (deferred to v2.0):
    - Causal edges
    - Episode / narrative graph
    - Reappears_as entity continuity edges with appearance embeddings

This is a separate module from ``global_event_index.py``. GEI-MVP is kept as a
prompt-injected navigation ablation in MVP Graph-guided mode and is NEVER part
of the strict graph-only inference path.
"""

from __future__ import annotations

from typing import Any

from sgd.scene_graph.records import hhmmss_to_secs
from sgd.scene_graph.schema import utc_now_iso


GLOBAL_GRAPH_VERSION = "global_graph_v1"
POSSESSION_EVENT_TYPES = {
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
    "transfer",
}


def _span_secs(span: list[str] | tuple[str, str]) -> tuple[float, float]:
    return (hhmmss_to_secs(span[0]), hhmmss_to_secs(span[1]))


def _build_entity_lifecycle_index(
    entity_registry: dict[str, Any],
) -> list[dict[str, Any]]:
    lifecycle: list[dict[str, Any]] = []
    for ent in entity_registry.get("entities", []) or []:
        lifecycle.append(
            {
                "entity_id": str(ent.get("entity_id", "")),
                "category": ent.get("category"),
                "first_seen": ent.get("first_seen"),
                "last_seen": ent.get("last_seen"),
                "support_frame_ids": list(ent.get("support_frame_ids") or []),
                "confidence": ent.get("confidence"),
            }
        )
    return lifecycle


def _build_temporal_edges(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    events_sorted = sorted(events, key=lambda e: _span_secs(e["time_span"])[0])
    for i, e1 in enumerate(events_sorted):
        e1_start, e1_end = _span_secs(e1["time_span"])
        for e2 in events_sorted[i + 1 :]:
            e2_start, e2_end = _span_secs(e2["time_span"])
            if e1_end <= e2_start:
                edges.append(
                    {
                        "source": e1["event_id"],
                        "target": e2["event_id"],
                        "predicate": "before",
                        "time_gap_secs": round(e2_start - e1_end, 3),
                    }
                )
            elif e1_start <= e2_start <= e1_end:
                edges.append(
                    {
                        "source": e1["event_id"],
                        "target": e2["event_id"],
                        "predicate": "overlap",
                        "time_gap_secs": 0.0,
                    }
                )
    return edges


def _build_first_last_by_event_type(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        by_type.setdefault(str(ev.get("event_type", "")), []).append(ev)
    out: dict[str, dict[str, str]] = {}
    for event_type, evs in by_type.items():
        if not event_type:
            continue
        evs_sorted = sorted(evs, key=lambda e: _span_secs(e["time_span"])[0])
        out[event_type] = {
            "first": evs_sorted[0]["event_id"],
            "last": evs_sorted[-1]["event_id"],
        }
    return out


def _build_first_last_by_object(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """First / last holder of each object, derived from possession events."""
    by_obj: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        if str(ev.get("event_type", "")).lower() not in POSSESSION_EVENT_TYPES:
            continue
        parts = ev.get("participants") or {}
        patient = parts.get("patient")
        if not patient:
            continue
        by_obj.setdefault(str(patient), []).append(ev)
    out: dict[str, dict[str, str]] = {}
    for obj_id, evs in by_obj.items():
        evs_sorted = sorted(evs, key=lambda e: _span_secs(e["time_span"])[0])
        # We want the "first holder" and "last holder" rather than first/last
        # event id; pick events whose agent / recipient represents that role.
        first_ev = evs_sorted[0]
        last_ev = evs_sorted[-1]
        first_holder = (first_ev.get("participants") or {}).get("agent")
        # For the last event, prefer the recipient (post-transfer holder) if
        # present; otherwise fall back to agent.
        last_parts = last_ev.get("participants") or {}
        last_holder = last_parts.get("recipient") or last_parts.get("agent")
        out[obj_id] = {
            "first_holder_event": first_ev["event_id"],
            "first_holder_entity": str(first_holder) if first_holder else "",
            "last_holder_event": last_ev["event_id"],
            "last_holder_entity": str(last_holder) if last_holder else "",
        }
    return out


def _build_state_carry_over(
    state_transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trans in state_transitions:
        out.append(
            {
                "entity": trans.get("entity"),
                "attribute": trans.get("attribute"),
                "from": trans.get("from"),
                "to": trans.get("to"),
                "time_span": trans.get("time_span"),
                "trigger_event": trans.get("trigger_event"),
                "support_facts": list(trans.get("support_facts") or []),
                "confidence": trans.get("confidence"),
            }
        )
    return out


def build_global_graph_v1(
    *,
    entity_registry: dict[str, Any],
    segment_events: dict[str, Any],
    video_id: str | None = None,
) -> dict[str, Any]:
    """Build the G_global_v1 dict from inference artifacts."""
    events = list(segment_events.get("events") or [])
    state_transitions = list(segment_events.get("state_transitions") or [])

    return {
        "version": GLOBAL_GRAPH_VERSION,
        "video_id": video_id or str(segment_events.get("video_id", "")),
        "created_at": utc_now_iso(),
        "entity_lifecycle_index": _build_entity_lifecycle_index(entity_registry),
        "global_events": [
            {
                "event_id": ev.get("event_id"),
                "event_type": ev.get("event_type"),
                "time_span": ev.get("time_span"),
                "participants": ev.get("participants"),
                "confidence": ev.get("confidence"),
            }
            for ev in events
        ],
        "temporal_edges": _build_temporal_edges(events),
        "first_last_index": {
            "by_event_type": _build_first_last_by_event_type(events),
            "by_object": _build_first_last_by_object(events),
        },
        "state_carry_over": _build_state_carry_over(state_transitions),
        "stats": {
            "num_global_events": len(events),
            "num_temporal_edges": len(_build_temporal_edges(events)),
            "num_state_carry_over": len(state_transitions),
        },
    }
