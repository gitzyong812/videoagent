"""Query API over G_global_v1 (M6).

Strict-safe lookups for first / last / before / after / holder questions.
Used by ``GraphTools.temporal_order_query`` as a high-precision backend when a
global graph is available; the naive fact-sort fallback in tools.py is used
when the global graph is absent.
"""

from __future__ import annotations

from typing import Any

from sgd.scene_graph.records import hhmmss_to_secs


def _span_secs(span: Any) -> tuple[float, float]:
    return (hhmmss_to_secs(span[0]), hhmmss_to_secs(span[1]))


def first_event_of_type(
    global_graph: dict[str, Any], event_type: str
) -> dict[str, Any] | None:
    idx = (global_graph.get("first_last_index") or {}).get("by_event_type") or {}
    pick = (idx.get(event_type) or {}).get("first")
    return _event_by_id(global_graph, pick)


def last_event_of_type(
    global_graph: dict[str, Any], event_type: str
) -> dict[str, Any] | None:
    idx = (global_graph.get("first_last_index") or {}).get("by_event_type") or {}
    pick = (idx.get(event_type) or {}).get("last")
    return _event_by_id(global_graph, pick)


def first_holder(global_graph: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    idx = (global_graph.get("first_last_index") or {}).get("by_object") or {}
    entry = idx.get(object_id) or {}
    if not entry.get("first_holder_event"):
        return None
    return {
        "entity_id": entry.get("first_holder_entity"),
        "event_id": entry.get("first_holder_event"),
        "event": _event_by_id(global_graph, entry["first_holder_event"]),
    }


def last_holder(global_graph: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    idx = (global_graph.get("first_last_index") or {}).get("by_object") or {}
    entry = idx.get(object_id) or {}
    if not entry.get("last_holder_event"):
        return None
    return {
        "entity_id": entry.get("last_holder_entity"),
        "event_id": entry.get("last_holder_event"),
        "event": _event_by_id(global_graph, entry["last_holder_event"]),
    }


def events_before(
    global_graph: dict[str, Any], anchor_event_id: str
) -> list[dict[str, Any]]:
    edges = global_graph.get("temporal_edges") or []
    sources = [
        e["source"]
        for e in edges
        if e.get("predicate") == "before" and e.get("target") == anchor_event_id
    ]
    return [ev for ev in (global_graph.get("global_events") or []) if ev.get("event_id") in sources]


def events_after(
    global_graph: dict[str, Any], anchor_event_id: str
) -> list[dict[str, Any]]:
    edges = global_graph.get("temporal_edges") or []
    targets = [
        e["target"]
        for e in edges
        if e.get("predicate") == "before" and e.get("source") == anchor_event_id
    ]
    return [ev for ev in (global_graph.get("global_events") or []) if ev.get("event_id") in targets]


def holder_at(
    global_graph: dict[str, Any], object_id: str, t_sec: float
) -> str | None:
    """Return the entity that holds object_id at time t_sec (seconds)."""
    carry = global_graph.get("state_carry_over") or []
    matching = [
        c
        for c in carry
        if c.get("entity") == object_id and c.get("attribute") == "holder"
    ]
    if not matching:
        # No transitions recorded — fall back to first holder of the object.
        fh = first_holder(global_graph, object_id)
        return fh["entity_id"] if fh else None
    matching.sort(key=lambda c: _span_secs(c["time_span"])[1])
    holder = None
    for c in matching:
        end = _span_secs(c["time_span"])[1]
        if t_sec >= end:
            holder = c.get("to")
        else:
            if holder is None:
                holder = c.get("from")
            break
    if holder is None:
        holder = matching[-1].get("to")
    return holder


def _event_by_id(global_graph: dict[str, Any], event_id: str | None) -> dict[str, Any] | None:
    if not event_id:
        return None
    for ev in global_graph.get("global_events") or []:
        if ev.get("event_id") == event_id:
            return ev
    return None
