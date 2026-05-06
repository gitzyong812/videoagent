"""Stage-3 (global scene graph) evaluation — placeholder.

Implementation deferred until §1 Stage-3 builder lands. Planned metrics
(see dev-plan §2.1):

- Event-chain topology accuracy (temporal / causal edges)
- Cross-segment entity binding accuracy
- Global retrieval recall@k

Once `sgd/scene_graph/stage3_builder.py` and `query_global.py` exist, this
module will mirror `eval_graph.py`'s structure: a `_compute_*` block per
metric and a `main()` argparse entry.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "evaluation.eval_stage3_graph is a placeholder. "
        "Stage-3 builder is not implemented yet (dev-plan §1)."
    )


if __name__ == "__main__":
    main()
