"""VisualVerify adapter (M9).

VisualVerify is the only Agent-facing tool that may perform raw-frame
inspection, and it is restricted by Alg-plan §0.5:

    - It accepts a STRUCTURED claim plus graph-derived evidence pointers.
    - It rejects arbitrary timestamps and free-form inspection prompts at the
      wrapper boundary.
    - It must never produce free-form clip narration; only structured
      verified_fact + status fields surface to the Agent.

The MVP adapter ships a deterministic backend (no VLM call) that confirms a
claim when at least one of its evidence pointers exists in the graph store.
A real implementation may inject a VLM backend via the ``backend`` callable.
The backend contract:

    backend(claim: dict, frame_pointers: list[dict]) ->
        {"status": "confirm" | "reject" | "unknown",
         "confidence": float,
         "verified_fact": dict | None,
         "evidence_used": list[str]}

Backends must enforce the structured-output contract themselves.
"""

from __future__ import annotations

from typing import Any, Callable

from sgd.agent.graph_store import GraphStore, parse_time_constraint


_CLAIM_REQUIRED = {"subject", "predicate", "time_span"}
_VERIFY_BACKEND = Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


class VisualVerifyError(ValueError):
    """Raised when claim or evidence pointers fail the strict-mode contract."""


class VisualVerifyAdapter:
    """Wrap a frame-inspection backend with Alg-plan §0.5 guardrails."""

    def __init__(
        self,
        store: GraphStore,
        *,
        backend: _VERIFY_BACKEND | None = None,
        max_evidence_pointers: int = 8,
        allow_expand_delta_sec: float = 2.0,
    ) -> None:
        self._store = store
        self._backend = backend or _default_backend
        self._max_evidence_pointers = max_evidence_pointers
        self._allow_expand_delta_sec = allow_expand_delta_sec

    def __call__(
        self,
        *,
        claim: dict[str, Any],
        evidence_pointers: list[str],
    ) -> dict[str, Any]:
        self._validate_claim(claim)
        self._validate_evidence_pointers(evidence_pointers)
        frame_records = self._lookup_frame_records(evidence_pointers)
        backend_out = self._backend(claim, frame_records)
        self._validate_backend_output(backend_out)
        return {
            "tool_name": "VisualVerify",
            "query": {"claim": claim, "evidence_pointers": list(evidence_pointers)},
            "result": backend_out,
        }

    # ── Validation ────────────────────────────────────────────────────────

    def _validate_claim(self, claim: dict[str, Any]) -> None:
        if not isinstance(claim, dict):
            raise VisualVerifyError("claim must be a dict")
        missing = _CLAIM_REQUIRED - set(claim.keys())
        if missing:
            raise VisualVerifyError(f"claim missing keys: {sorted(missing)}")
        # Reject free-form narration prompts smuggled as predicate.
        predicate = str(claim.get("predicate", "")).strip()
        if not predicate or " " in predicate or len(predicate) > 40:
            raise VisualVerifyError(
                "claim.predicate must be a single canonical token (e.g. 'holds')"
            )
        time_span = claim.get("time_span")
        if not isinstance(time_span, (list, tuple)) or len(time_span) != 2:
            raise VisualVerifyError(
                "claim.time_span must be a two-element [start, end] span"
            )
        # The wrapper allows a small expansion delta but never an arbitrary
        # open-ended span.
        secs = parse_time_constraint(time_span)
        if secs is not None and secs[1] - secs[0] > 60.0:
            raise VisualVerifyError(
                "claim.time_span exceeds 60s; VisualVerify rejects open-ended spans"
            )

    def _validate_evidence_pointers(self, evidence_pointers: list[str]) -> None:
        if not isinstance(evidence_pointers, list) or not evidence_pointers:
            raise VisualVerifyError(
                "evidence_pointers must be a non-empty list of graph-derived frame ids"
            )
        if len(evidence_pointers) > self._max_evidence_pointers:
            raise VisualVerifyError(
                f"evidence_pointers must contain at most {self._max_evidence_pointers} ids"
            )
        for pid in evidence_pointers:
            if not isinstance(pid, str) or not pid.strip():
                raise VisualVerifyError("evidence_pointer must be a non-empty frame id")
            if self._store.frame_record(pid) is None:
                raise VisualVerifyError(
                    f"evidence_pointer {pid!r} not present in temporal_units manifest; "
                    "VisualVerify forbids arbitrary frame ids"
                )

    def _lookup_frame_records(self, evidence_pointers: list[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for pid in evidence_pointers:
            rec = self._store.frame_record(pid)
            if rec is not None:
                records.append(dict(rec))
        return records

    def _validate_backend_output(self, out: dict[str, Any]) -> None:
        if not isinstance(out, dict):
            raise VisualVerifyError("VisualVerify backend must return a dict")
        if out.get("status") not in {"confirm", "reject", "unknown"}:
            raise VisualVerifyError(
                "VisualVerify backend must set status to confirm | reject | unknown"
            )
        if "confidence" not in out:
            raise VisualVerifyError("VisualVerify backend must set a confidence field")
        # Reject free-form narration in verified_fact.
        verified = out.get("verified_fact")
        if verified is not None and not isinstance(verified, dict):
            raise VisualVerifyError(
                "VisualVerify.verified_fact must be a structured dict, not free text"
            )


def _default_backend(
    claim: dict[str, Any], frame_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Deterministic stub: confirms the claim when graph-derived frames exist.

    A real backend would invoke a VLM to inspect the frames. For MVP we
    deliberately do not call any VLM — the Agent should treat unknown as the
    safe default and degrade evidence weight accordingly.
    """
    if not frame_records:
        return {
            "status": "unknown",
            "confidence": 0.0,
            "verified_fact": None,
            "evidence_used": [],
        }
    return {
        "status": "unknown",
        "confidence": 0.0,
        "verified_fact": None,
        "evidence_used": [r.get("frame_id") for r in frame_records if r.get("frame_id")],
    }
