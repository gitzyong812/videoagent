"""Agent run-report builder (M11).

Produces a strict-mode-labelled report from a list of ``GraphAgent.run``
outputs. The report is the canonical hand-off between the Agent and any
benchmark / paper / replay pipeline:

    - Every run carries a ``run_mode`` label (Strict Graph-only,
      MVP Graph-guided).
    - Every run carries an audit verdict from ``audit_agent_run``.
    - Per-question evidence-chain validity (supported vs unsupported).
    - Aggregate counters for sanity checks.

This module does NOT compare against ground truth; benchmark evaluators
(see ``evaluation/``) layer accuracy on top of the chain returned here.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from sgd.agent.audit import AgentRunAuditError, audit_agent_run


RUN_MODE_LABELS = {
    "strict_graph_only": "Strict Graph-only",
    "mvp_graph_guided": "MVP Graph-guided",
    "builder_assisted": "Builder-assisted",
    "hybrid": "Hybrid",
}


def evidence_chain_validity(answer: dict[str, Any]) -> dict[str, Any]:
    """Score the structural validity of one AnswerWithEvidence chain."""
    chain = answer.get("evidence_chain") or {}
    has_event = bool(chain.get("segment_events"))
    has_fact = bool(chain.get("frame_facts"))
    has_frame = bool(chain.get("support_frames"))
    has_visual = bool(chain.get("visual_evidence"))
    return {
        "has_segment_event": has_event,
        "has_frame_fact": has_fact,
        "has_support_frame": has_frame,
        "has_visual_pointer": has_visual,
        "is_supported": (has_event or has_fact) and has_frame,
        "answer_status": answer.get("status") or "supported",
        "confidence": answer.get("confidence"),
    }


def build_run_report(
    runs: Iterable[dict[str, Any]],
    *,
    run_mode_label: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Build a strict-safe report over one or more agent runs.

    Args:
        runs: iterable of ``GraphAgent.run`` outputs.
        run_mode_label: optional override; otherwise the label is derived
            from each run's ``run_mode``.
        strict: when True, each run is audited and audit failures become
            entries in ``audit_failures``. Any audit failure makes the report
            ``report_status="failed"``.
    """
    entries: list[dict[str, Any]] = []
    audit_failures: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    confidences: list[float] = []

    for run in runs:
        run_mode = run.get("run_mode", "")
        label = run_mode_label or RUN_MODE_LABELS.get(run_mode, run_mode)
        answer = run.get("answer") or {}
        chain_validity = evidence_chain_validity(answer)

        entry = {
            "run_mode": run_mode,
            "run_mode_label": label,
            "question": (run.get("belief_state") or {}).get("question"),
            "answer": answer.get("answer"),
            "confidence": answer.get("confidence"),
            "chain_validity": chain_validity,
            "tool_count": len((run.get("belief_state") or {}).get("tool_history") or []),
            "warnings": (run.get("belief_state") or {}).get("warnings") or [],
        }

        if strict:
            try:
                audit_report = audit_agent_run(run)
                entry["audit"] = audit_report
            except AgentRunAuditError as exc:
                entry["audit"] = {"status": "failed", "error": str(exc)}
                audit_failures.append({"question": entry["question"], "error": str(exc)})
        else:
            entry["audit"] = {"status": "skipped"}

        entries.append(entry)
        counters["total"] += 1
        if chain_validity["is_supported"]:
            counters["supported"] += 1
        else:
            counters["unsupported"] += 1
        if entry["audit"].get("status") == "failed":
            counters["audit_failed"] += 1
        if isinstance(entry["confidence"], (int, float)):
            confidences.append(float(entry["confidence"]))

    report_status = "ok" if not audit_failures else "failed"
    mean_confidence = (
        round(sum(confidences) / len(confidences), 4) if confidences else None
    )

    return {
        "version": "agent_run_report_v1",
        "report_status": report_status,
        "run_mode_label": run_mode_label,
        "summary": {
            "total_runs": counters["total"],
            "supported_runs": counters["supported"],
            "unsupported_runs": counters["unsupported"],
            "audit_failures": counters["audit_failed"],
            "mean_confidence": mean_confidence,
        },
        "entries": entries,
        "audit_failures": audit_failures,
    }
