"""Rule/template-based QuestionDecompose (M9).

Compiles a natural-language question into a graph-query plan per Alg-plan §12.2.
The output is a JSON-schema-validated dict that the Agent uses to choose its
next semantic tool. No LLM call in MVP; the compiler is intentionally simple
and auditable.

Output shape:
    {
        "tool_name": "QuestionDecompose",
        "query": {"question": str, "metadata": dict | None},
        "result": {
            "intent": "temporal_order_query" | "event_search" | "relation_query"
                       | "attribute_query" | "causal_query" | "counting_query"
                       | "entity_resolve",
            "entities": [{"mention": str, "type": str, "slot": str}],
            "relations": [{"predicate": str, "subject": str, "object": str}],
            "temporal_operator": "first" | "last" | "before" | "after" | None,
            "time_constraint": str | None,
            "missing_slots": [str, ...],
            "requires_evidence": bool,
            "candidate_tool_plan": [str, ...],
            "warnings": [str, ...],
        }
    }

If validation fails, the function raises ``QuestionDecomposeError`` so callers
must not silently consume an invalid plan.
"""

from __future__ import annotations

import re
from typing import Any


class QuestionDecomposeError(ValueError):
    """Raised when the compiled plan fails JSON-schema validation."""


_TEMPORAL_LAST = {"last", "finally", "ultimately", "in the end", "at the end", "eventually"}
_TEMPORAL_FIRST = {"first", "initially", "at the start", "at first", "originally"}
_TEMPORAL_BEFORE = {"before", "prior to", "preceding", "earlier than"}
_TEMPORAL_AFTER = {"after", "following", "subsequent to", "later than"}

_COUNTING_HINTS = {"how many", "count of", "number of"}
_ATTRIBUTE_HINTS = {
    "what color",
    "which color",
    "what is the color",
    "what shape",
    "what kind of",
    "what type of",
}
_CAUSAL_HINTS = {"why", "what causes", "what caused", "because of", "for what reason"}

_INTENT_DEFAULT = "relation_query"

_WORD_SPLIT = re.compile(r"[a-zA-Z][a-zA-Z0-9_'\-]*")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "you",
    "your",
    "they",
    "them",
    "their",
    "it",
    "its",
    "this",
    "that",
    "and",
    "or",
    "but",
    "if",
    "then",
    "what",
    "who",
    "whom",
    "whose",
    "which",
    "where",
    "when",
    "how",
    "many",
}


_REQUIRED_KEYS = {
    "intent",
    "entities",
    "relations",
    "temporal_operator",
    "time_constraint",
    "missing_slots",
    "requires_evidence",
    "candidate_tool_plan",
    "warnings",
}


def _phrase_match(text: str, phrases: set[str]) -> bool:
    return any(p in text for p in phrases)


def _detect_intent_and_operator(
    question_lc: str,
) -> tuple[str, str | None, list[str]]:
    """Return ``(intent, temporal_operator, warnings)``."""
    warnings: list[str] = []

    if _phrase_match(question_lc, _TEMPORAL_LAST):
        return "temporal_order_query", "last", warnings
    if _phrase_match(question_lc, _TEMPORAL_FIRST):
        return "temporal_order_query", "first", warnings
    if _phrase_match(question_lc, _TEMPORAL_BEFORE):
        return "temporal_order_query", "before", warnings
    if _phrase_match(question_lc, _TEMPORAL_AFTER):
        return "temporal_order_query", "after", warnings

    if _phrase_match(question_lc, _CAUSAL_HINTS):
        warnings.append(
            "Causal queries fall back to temporal_order_query predecessors in MVP "
            "(causal graph is not enabled). Answer must be labeled as a "
            "temporal_explanation_hypothesis."
        )
        return "causal_query", "before", warnings

    if _phrase_match(question_lc, _COUNTING_HINTS):
        return "counting_query", None, warnings

    if _phrase_match(question_lc, _ATTRIBUTE_HINTS):
        return "attribute_query", None, warnings

    if question_lc.startswith("who is ") or question_lc.startswith("which person "):
        return "entity_resolve", None, warnings

    if question_lc.startswith("what happens") or question_lc.startswith("what occurs"):
        return "event_search", None, warnings

    return _INTENT_DEFAULT, None, warnings


def _extract_entity_mentions(question: str) -> list[dict[str, str]]:
    """Extract candidate entity mentions as noun-ish tokens.

    The MVP heuristic is intentionally crude: tokenize, drop stopwords, group
    consecutive non-stopword tokens into phrases of length 1–3. Downstream
    EntityResolve does the actual graph lookup.
    """
    mentions: list[dict[str, str]] = []
    tokens = [_canonical_token(t) for t in _WORD_SPLIT.findall(question)]
    buf: list[str] = []
    for tok in tokens + [""]:
        if not tok or tok in _STOPWORDS:
            if buf:
                phrase = " ".join(buf[:3])
                mentions.append({"mention": phrase, "type": "unknown", "slot": "target"})
                buf = []
            continue
        buf.append(tok)
    # Deduplicate keeping order.
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for m in mentions:
        key = m["mention"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    return deduped


def _canonical_token(tok: str) -> str:
    return tok.lower().strip("'-")


def _build_tool_plan(intent: str) -> list[str]:
    if intent == "temporal_order_query":
        return [
            "EntityResolve",
            "RelationQuery",
            "TemporalOrderQuery",
            "EvidenceTrace",
            "AnswerWithEvidence",
        ]
    if intent == "causal_query":
        return [
            "EntityResolve",
            "EventSearch",
            "TemporalOrderQuery",
            "EvidenceTrace",
            "AnswerWithEvidence",
        ]
    if intent == "event_search":
        return [
            "EntityResolve",
            "EventSearch",
            "EvidenceTrace",
            "AnswerWithEvidence",
        ]
    if intent == "attribute_query":
        return [
            "EntityResolve",
            "RelationQuery",
            "EvidenceTrace",
            "VisualVerify",
            "AnswerWithEvidence",
        ]
    if intent == "counting_query":
        return [
            "EntityResolve",
            "RelationQuery",
            "EvidenceTrace",
            "AnswerWithEvidence",
        ]
    if intent == "entity_resolve":
        return ["EntityResolve", "AnswerWithEvidence"]
    return [
        "EntityResolve",
        "RelationQuery",
        "EvidenceTrace",
        "AnswerWithEvidence",
    ]


def _validate(result: dict[str, Any]) -> None:
    missing = _REQUIRED_KEYS - set(result.keys())
    if missing:
        raise QuestionDecomposeError(
            f"QuestionDecompose result missing keys: {sorted(missing)}"
        )
    if not isinstance(result["entities"], list):
        raise QuestionDecomposeError("entities must be a list")
    for ent in result["entities"]:
        if not isinstance(ent, dict) or "mention" not in ent:
            raise QuestionDecomposeError(
                "each entity mention must be a dict with a 'mention' key"
            )
    if not isinstance(result["candidate_tool_plan"], list):
        raise QuestionDecomposeError("candidate_tool_plan must be a list")
    if not result["candidate_tool_plan"]:
        raise QuestionDecomposeError("candidate_tool_plan must not be empty")
    if result["candidate_tool_plan"][-1] != "AnswerWithEvidence":
        raise QuestionDecomposeError(
            "candidate_tool_plan must terminate in AnswerWithEvidence"
        )


def question_decompose(
    question: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a question into an Alg-plan QuestionDecompose result."""
    if not isinstance(question, str) or not question.strip():
        raise QuestionDecomposeError("question must be a non-empty string")

    question_lc = question.lower().strip()
    intent, temporal_operator, warnings = _detect_intent_and_operator(question_lc)
    entities = _extract_entity_mentions(question)
    tool_plan = _build_tool_plan(intent)

    time_constraint: str | None = None
    if metadata and isinstance(metadata.get("time_reference"), str):
        time_constraint = metadata["time_reference"]

    missing_slots: list[str] = []
    if not entities:
        missing_slots.append("target_entity")

    result = {
        "intent": intent,
        "entities": entities,
        "relations": [],
        "temporal_operator": temporal_operator,
        "time_constraint": time_constraint,
        "missing_slots": missing_slots,
        "requires_evidence": True,
        "candidate_tool_plan": tool_plan,
        "warnings": warnings,
    }
    _validate(result)
    return {
        "tool_name": "QuestionDecompose",
        "query": {"question": question, "metadata": metadata or {}},
        "result": result,
    }
