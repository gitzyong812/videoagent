"""Agent-run audit (M0).

Strict-mode contract check for the output of ``GraphAgent.run``. Complements
the artifact-level audit in ``sgd.scene_graph.pipeline``: artifact audit
guarantees the *read surface* is strict-safe; this audit guarantees the
*execution trace* used a strict-safe action space and produced a structurally
valid AnswerWithEvidence envelope.

Strict graph-only invariants enforced here:

    1. ``run_mode`` is ``strict_graph_only`` or ``mvp_graph_guided``. Hybrid /
       caption-baseline runs cannot pass this audit.
    2. Every tool call in ``tool_history`` uses an Alg-plan §12 semantic name.
       Legacy tool names (``frame_inspect_tool``, ``clip_search_tool``,
       ``global_browse_tool``, ``finish``) are forbidden.
    3. No forbidden field names appear anywhere in the tool history or the
       final answer envelope (caption, caption_context, caption_snippets,
       transcript, narration, script, free_form_summary, summary).
    4. If the answer is not labeled ``status="unsupported"``, the evidence
       chain must contain at least one segment_event or frame_fact AND at
       least one support_frame entry.

Raises ``AgentRunAuditError`` on the first violation. Returns an OK report
otherwise so callers can record the audited tool set.
"""

from __future__ import annotations

from typing import Any, Iterable


ALLOWED_TOOL_NAMES: set[str] = {
    "QuestionDecompose",
    "EntityResolve",
    "RelationQuery",
    "EventSearch",
    "TemporalOrderQuery",
    "EvidenceTrace",
    "VisualVerify",
    "AnswerWithEvidence",
}

FORBIDDEN_TOOL_NAMES: set[str] = {
    "frame_inspect_tool",
    "clip_search_tool",
    "global_browse_tool",
    "caption_search_tool",
    "finish",
    "segment_graph_query_tool",
    "frame_graph_query_tool",
}

FORBIDDEN_FIELDS: set[str] = {
    "caption",
    "caption_context",
    "caption_snippets",
    "transcript",
    "narration",
    "script",
    "free_form_summary",
    "summary",
}

ALLOWED_RUN_MODES: set[str] = {"strict_graph_only", "mvp_graph_guided"}


class AgentRunAuditError(ValueError):
    """Raised when an agent-run output fails the strict-mode contract."""


def audit_agent_run(run_output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(run_output, dict):
        raise AgentRunAuditError("run_output must be a dict")

    run_mode = run_output.get("run_mode")
    if run_mode not in ALLOWED_RUN_MODES:
        raise AgentRunAuditError(
            f"run_mode {run_mode!r} is not strict-mode auditable; "
            f"expected one of {sorted(ALLOWED_RUN_MODES)}"
        )

    belief = run_output.get("belief_state") or {}
    history = belief.get("tool_history") or []

    used_tools: set[str] = set()
    for step in history:
        name = step.get("tool_name", "")
        if name in FORBIDDEN_TOOL_NAMES:
            raise AgentRunAuditError(
                f"forbidden tool {name!r} appeared in tool_history; "
                "strict graph-only Agents may only call Alg-plan §12 tools"
            )
        if name not in ALLOWED_TOOL_NAMES:
            raise AgentRunAuditError(
                f"unknown tool {name!r} in tool_history; not part of Alg-plan §12"
            )
        used_tools.add(name)

    forbidden_hits: list[str] = []
    forbidden_hits.extend(_walk_forbidden_keys(history, path="belief_state.tool_history"))
    forbidden_hits.extend(_walk_forbidden_keys(run_output.get("answer"), path="answer"))
    if forbidden_hits:
        raise AgentRunAuditError(
            f"forbidden fields surfaced in run output: {forbidden_hits[:5]}"
        )

    answer = run_output.get("answer") or {}
    status = answer.get("status")
    chain = answer.get("evidence_chain") or {}

    if status != "unsupported":
        if not (chain.get("segment_events") or chain.get("frame_facts")):
            raise AgentRunAuditError(
                "supported answers must contain at least one segment_event or frame_fact in evidence_chain"
            )
        if not chain.get("support_frames"):
            raise AgentRunAuditError(
                "supported answers must contain at least one support_frame in evidence_chain"
            )

    return {
        "status": "ok",
        "run_mode": run_mode,
        "used_tools": sorted(used_tools),
        "answer_status": status or "supported",
        "answer_confidence": answer.get("confidence"),
    }


def _walk_forbidden_keys(value: Any, *, path: str) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key in FORBIDDEN_FIELDS:
                hits.append(f"{path}.{key}")
            hits.extend(_walk_forbidden_keys(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            hits.extend(_walk_forbidden_keys(child, path=f"{path}[{i}]"))
    return hits
