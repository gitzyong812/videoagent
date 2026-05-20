"""GraphAgent — strict graph-only reasoning loop (M10).

Implements Alg-plan §13 with an MVP-deterministic action selection: the agent
executes the tool plan returned by QuestionDecompose in order, accumulating a
belief state and ending with AnswerWithEvidence. There is no LLM-driven action
selection in MVP; the plan is rule-based and auditable, which keeps the strict
graph-only contract verifiable.

Belief-state schema (per Alg-plan §13.1):

    {
        "question": str,
        "parsed_intent": dict,
        "resolved_entities": {mention: [entity_id, ...]},
        "candidate_events": [event_dict, ...],
        "candidate_facts": [fact_dict, ...],
        "candidate_answers": [str, ...],
        "supporting_subgraph": dict,
        "uncertain_items": [str, ...],
        "conflicts": [str, ...],
        "evidence_chain": dict,
        "tool_history": [envelope, ...],
        "warnings": [str, ...],
    }

Run mode:
    Strict Graph-only by default. GEI-MVP is never injected; legacy
    frame_inspect_tool is reachable only via VisualVerifyAdapter inside
    GraphTools.visual_verify, never as an Agent-facing action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sgd.agent.graph_store import GraphStore
from sgd.agent.tools import GraphTools


RUN_MODE_STRICT = "strict_graph_only"
RUN_MODE_MVP_GUIDED = "mvp_graph_guided"


@dataclass
class BeliefState:
    question: str
    parsed_intent: dict[str, Any] = field(default_factory=dict)
    resolved_entities: dict[str, list[str]] = field(default_factory=dict)
    candidate_events: list[dict[str, Any]] = field(default_factory=list)
    candidate_facts: list[dict[str, Any]] = field(default_factory=list)
    candidate_answers: list[str] = field(default_factory=list)
    supporting_subgraph: dict[str, Any] = field(default_factory=dict)
    uncertain_items: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    evidence_chain: dict[str, Any] = field(default_factory=dict)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "parsed_intent": self.parsed_intent,
            "resolved_entities": self.resolved_entities,
            "candidate_events": self.candidate_events,
            "candidate_facts": self.candidate_facts,
            "candidate_answers": self.candidate_answers,
            "supporting_subgraph": self.supporting_subgraph,
            "uncertain_items": self.uncertain_items,
            "conflicts": self.conflicts,
            "evidence_chain": self.evidence_chain,
            "tool_history": self.tool_history,
            "warnings": self.warnings,
        }


class GraphAgent:
    """Strict graph-only graph-driven reasoning agent."""

    def __init__(
        self,
        tools: GraphTools,
        *,
        run_mode: str = RUN_MODE_STRICT,
        max_relation_facts: int = 50,
        max_events: int = 10,
    ) -> None:
        if run_mode not in {RUN_MODE_STRICT, RUN_MODE_MVP_GUIDED}:
            raise ValueError(f"unsupported run_mode: {run_mode!r}")
        self._tools = tools
        self._run_mode = run_mode
        self._max_relation_facts = max_relation_facts
        self._max_events = max_events

    @property
    def run_mode(self) -> str:
        return self._run_mode

    @classmethod
    def from_inference_dir(
        cls,
        base_dir: str,
        *,
        run_mode: str = RUN_MODE_STRICT,
    ) -> "GraphAgent":
        store = GraphStore.from_dir(base_dir)
        return cls(GraphTools(store), run_mode=run_mode)

    def run(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        belief = BeliefState(question=question)

        decompose_env = self._call("QuestionDecompose", lambda: self._tools.question_decompose(
            question, metadata=metadata
        ), belief)
        belief.parsed_intent = decompose_env["result"]
        belief.warnings.extend(belief.parsed_intent.get("warnings") or [])

        self._resolve_entities(belief)
        self._fetch_candidates(belief)
        answer = self._build_answer(belief)
        belief.evidence_chain = answer["result"]["evidence_chain"]
        return {
            "run_mode": self._run_mode,
            "answer": answer["result"],
            "belief_state": belief.to_dict(),
        }

    # ── Pipeline stages ──────────────────────────────────────────────────

    def _resolve_entities(self, belief: BeliefState) -> None:
        plan = belief.parsed_intent
        for mention in plan.get("entities") or []:
            text = mention.get("mention", "")
            if not text:
                continue
            env = self._call(
                "EntityResolve",
                lambda t=text: self._tools.entity_resolve(t, time_constraint=plan.get("time_constraint")),
                belief,
            )
            candidates = env["result"].get("candidates") or []
            if candidates:
                belief.resolved_entities[text] = [c["entity_id"] for c in candidates]

    def _fetch_candidates(self, belief: BeliefState) -> None:
        plan = belief.parsed_intent
        intent = plan.get("intent")
        time_constraint = plan.get("time_constraint")

        # Decide which target object (if any) to scope facts/events by.
        target_object = self._pick_target_object(belief)

        if intent in {"event_search"}:
            env = self._call(
                "EventSearch",
                lambda: self._tools.event_search(time_constraint=time_constraint, top_k=self._max_events),
                belief,
            )
            belief.candidate_events.extend(env["result"].get("events") or [])
            return

        # For relation / temporal / causal / attribute / counting / entity_resolve
        # intents, refuse to scan the whole graph when no entities are resolved.
        # Returning unscoped facts would let the agent answer with arbitrary
        # evidence and violates strict graph-only support requirements.
        if target_object is None and not belief.resolved_entities:
            belief.uncertain_items.append("entity_resolution_failed")
            belief.warnings.append(
                "EntityResolve found no candidates in the strict-safe alias index; "
                "RelationQuery is skipped to avoid unscoped graph scans."
            )
            return

        rel_env = self._call(
            "RelationQuery",
            lambda: self._tools.relation_query(
                object=target_object,
                time_constraint=time_constraint,
            ),
            belief,
        )
        facts = rel_env["result"].get("facts") or []
        belief.candidate_facts.extend(facts[: self._max_relation_facts])

        operator = plan.get("temporal_operator")
        if operator in {"first", "last", "before", "after"} and target_object is not None:
            order_env = self._call(
                "TemporalOrderQuery",
                lambda op=operator, t=target_object: self._tools.temporal_order_query(
                    operator=op, object=t, time_constraint=time_constraint
                ),
                belief,
            )
            picked = order_env["result"].get("result")
            if picked is not None:
                # Promote the picked candidate as the answer subject (for MVP).
                if picked.get("kind") in {"fact", "global_holder"}:
                    belief.candidate_answers.append(str(picked.get("subject")))
                elif picked.get("kind") == "event":
                    parts = picked.get("participants") or {}
                    if "recipient" in parts:
                        belief.candidate_answers.append(str(parts["recipient"]))
                    elif "agent" in parts:
                        belief.candidate_answers.append(str(parts["agent"]))

    def _pick_target_object(self, belief: BeliefState) -> str | None:
        """Pick a target object entity id from resolved mentions, if any."""
        # Prefer entities whose category is 'object'.
        candidates = []
        for ids in belief.resolved_entities.values():
            for ent_id in ids:
                ent = self._tools._store.entity_by_id(ent_id)  # noqa: SLF001 (internal lookup)
                if ent and ent.get("category") == "object":
                    candidates.append(ent_id)
        if candidates:
            return candidates[0]
        # Fall back to the first resolved entity.
        for ids in belief.resolved_entities.values():
            if ids:
                return ids[0]
        return None

    def _build_answer(self, belief: BeliefState) -> dict[str, Any]:
        events_ids = [e["event_id"] for e in belief.candidate_events if e.get("event_id")]
        facts_ids = [f["fact_id"] for f in belief.candidate_facts if f.get("fact_id")]

        # If nothing matched, return a sentinel envelope with empty chain.
        if not events_ids and not facts_ids:
            belief.uncertain_items.append("no_supporting_evidence_in_graph")
            return _empty_answer_envelope("unknown", belief.warnings)

        # Run EvidenceTrace over the top candidate to materialize support pointers.
        trace_target_id: str | None = None
        trace_target_type: str | None = None
        if events_ids:
            trace_target_id, trace_target_type = events_ids[0], "event"
        elif facts_ids:
            trace_target_id, trace_target_type = facts_ids[0], "fact"

        if trace_target_id and trace_target_type:
            self._call(
                "EvidenceTrace",
                lambda: self._tools.evidence_trace(trace_target_id, trace_target_type),
                belief,
            )

        candidate_answer = belief.candidate_answers[0] if belief.candidate_answers else (
            facts_ids[0] if facts_ids else events_ids[0]
        )

        env = self._call(
            "AnswerWithEvidence",
            lambda: self._tools.answer_with_evidence(
                candidate_answer=candidate_answer,
                supporting_subgraph={
                    "events": events_ids,
                    "facts": facts_ids,
                    "global": [],
                },
            ),
            belief,
        )
        belief.supporting_subgraph = env["query"]["supporting_subgraph"]
        return env

    # ── Tool dispatch with history bookkeeping ───────────────────────────

    def _call(
        self,
        name: str,
        fn,
        belief: BeliefState,
    ) -> dict[str, Any]:
        envelope = fn()
        belief.tool_history.append(
            {
                "tool_name": envelope.get("tool_name", name),
                "query": envelope.get("query"),
                "result_summary": _summarize_result(envelope.get("result", {})),
            }
        )
        return envelope


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Trim the result for trace logging without losing structural shape."""
    summary: dict[str, Any] = {}
    for key in ("missing_slots", "warnings"):
        if key in result:
            summary[key] = result[key]
    for key, value in result.items():
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
        elif key not in summary and not isinstance(value, dict):
            summary[key] = value
    return summary


def _empty_answer_envelope(answer: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "tool_name": "AnswerWithEvidence",
        "query": {"candidate_answer": answer, "supporting_subgraph": {"events": [], "facts": []}},
        "result": {
            "answer": answer,
            "confidence": 0.0,
            "evidence_chain": {
                "global": [],
                "segment_events": [],
                "frame_facts": [],
                "support_frames": [],
                "visual_evidence": [],
                "verification": [],
            },
            "warnings": list(warnings),
            "status": "unsupported",
        },
    }
