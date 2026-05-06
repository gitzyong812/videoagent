"""Unified entry for the three-layer scene-graph evaluation pipeline.

Subcommands:
- `agent`    → Agent QA accuracy (delegates to eval_agent.main)
- `graph`    → Stage-1 + Stage-2 graph quality + retrieval (delegates to eval_graph.main)
- `stage3`   → Stage-3 global graph metrics (placeholder; see eval_stage3_graph)
- `all`      → run agent + graph in sequence

Each subcommand forwards remaining argv to the underlying script's argparse.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUBCOMMANDS = {"agent", "graph", "stage3", "all"}


def _print_usage() -> None:
    print(
        "Usage: python -m evaluation.run_graph_eval <subcommand> [args...]\n"
        "  subcommands: agent | graph | stage3 | all"
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        _print_usage()
        return 2

    sub = sys.argv[1]
    # Hand the remaining argv to the delegated script
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if sub == "agent":
        from evaluation.eval_agent import main as run
        return run() or 0

    if sub == "graph":
        from evaluation.eval_graph import main as run
        return run() or 0

    if sub == "stage3":
        from evaluation.eval_stage3_graph import main as run
        return run() or 0

    if sub == "all":
        from evaluation.eval_agent import main as run_agent
        from evaluation.eval_graph import main as run_graph
        rc1 = run_agent() or 0
        rc2 = run_graph() or 0
        return rc1 or rc2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
