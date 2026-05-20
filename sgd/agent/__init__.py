"""SGD-Agent v2 graph-native agent layer (M9 + M10).

Public surface:
    - GraphStore: loads the strict-safe inference artifact bundle and serves
      lookups for tool wrappers.
    - GraphTools: Alg-plan §12 semantic tools (QuestionDecompose, EntityResolve,
      RelationQuery, EventSearch, TemporalOrderQuery, EvidenceTrace,
      VisualVerify, AnswerWithEvidence).

Module layout:
    graph_store.py        — bundle loader + query helpers.
    question_decompose.py — rule/template-based QuestionDecompose.
    visual_verify.py      — VisualVerify wrapper around frame_inspect_tool.
    tools.py              — the rest of the semantic tools + GraphTools facade.

Agent code (M10) imports GraphTools from this package; legacy low-level
tools (frame_graph_query_tool, segment_graph_query_tool, etc.) are no longer
Agent-facing.
"""

from sgd.agent.audit import AgentRunAuditError, audit_agent_run
from sgd.agent.core import GraphAgent, RUN_MODE_MVP_GUIDED, RUN_MODE_STRICT
from sgd.agent.graph_store import GraphStore
from sgd.agent.question_decompose import question_decompose
from sgd.agent.tools import GraphTools
from sgd.agent.visual_verify import VisualVerifyAdapter

__all__ = [
    "AgentRunAuditError",
    "GraphAgent",
    "GraphStore",
    "GraphTools",
    "RUN_MODE_MVP_GUIDED",
    "RUN_MODE_STRICT",
    "VisualVerifyAdapter",
    "audit_agent_run",
    "question_decompose",
]
