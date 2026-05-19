# SGDAgent Planning Documents

Last consolidated: 2026-05-20

## Canonical Documents

- `SGD-Alg-plan.md`: highest-level algorithm specification. It defines the
  representation, graph construction algorithm, caption-free protocol,
  Agent-facing tool semantics, evidence-chain contract, evaluation principles,
  and MVP/full-version boundaries.
- `dev-plan.md`: only canonical development progress document. It tracks
  implementation status by Alg-plan modules M0-M11.

## Archived Historical Plans

- `SG-plan.md`: archived experiment-plan draft. Its useful caption-free and
  evaluation decisions have been consolidated into `SGD-Alg-plan.md`.
- `app-plan.md`: archived demo/UI draft. Non-conflicting demo requirements have
  been consolidated into the M11 section of `dev-plan.md`.

## Operational Context Documents

These files are retained for repository operations and baseline context. They do
not override the canonical algorithm or development plan:

- `handoff-20260513.md`
- `private-public-sync-plan.md`
- `../baselines/new-baseline.md`

## Rule

When documents conflict, follow this order:

1. `SGD-Alg-plan.md`
2. `dev-plan.md`
3. operational context documents
4. archived historical plans
