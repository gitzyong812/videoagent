# SGDAgent Canonical Development Plan

Canonical source:
  `docs/plans/SGD-Alg-plan.md`

Purpose:
  This is the only canonical development progress document. All implementation
  work is organized around the Alg-plan modules M0-M11. Historical plans are
  retained for context only and must not override this file.

Last consolidated:
  2026-05-20

---

## Status Taxonomy

Implemented:
  Conforms to Alg-plan semantics and can be treated as a stable module.

Temporary:
  Usable for MVP, but only as an adapter, navigation aid, or workaround.
  Temporary components must be clearly labeled in experiments and papers.

To Refactor:
  Existing code or fields are useful but conflict with Alg-plan semantics,
  strict graph-only rules, or the target schema.

Planned:
  Not implemented yet.

---

## Current Canonical Decisions

- Alg-plan is the highest instruction for algorithm and system semantics.
- MVP may reuse existing code, but Agent-facing behavior must use Alg-plan tool names.
- Strict graph-only cannot expose captions, caption embeddings, free-form summaries, or free-form visual inspection.
- `frame_inspect_tool` is retained only as an internal `VisualVerify` adapter.
- GEI-MVP is retained only as a navigation aid in MVP Graph-guided mode. It cannot be used as evidence and should not be exposed in Strict Graph-only unless stripped into structured graph signatures.
- After Stage3 lands, GEI must be materialized from `G_global_v1`.
- Free-form `summary` fields are forbidden in strict inference. They must be replaced by structured `graph_signature` or `event_signature` records.
- Final answers must be structured `AnswerWithEvidence` records; benchmark runners may extract the short answer.
- Development progress is tracked only in this file.

---

## Run Mode Taxonomy

Strict Graph-only:
  Agent may access only stripped graph memory, graph-native tool outputs,
  evidence pointers, and structured VisualVerify results.
  Captions, caption embeddings, free-form summaries, transcript snippets,
  GEI prompt timelines, and arbitrary raw-frame inspection are forbidden.

MVP Graph-guided:
  Temporary navigation aids such as GEI-MVP may be used to locate candidate
  intervals, but they cannot be used as answer evidence.

Builder-assisted:
  Captions, transcripts, VLM narrations, and OCR may be used during offline
  construction, but all such text must be stripped from inference artifacts.

Hybrid:
  Caption and graph tools may both be used. This is not a strict graph-only
  setting and must be reported separately.

---

## Strict Inference Boundary

Allowed:
  stripped graph memory; entity records; graph facts; event nodes; temporal
  edges; state carry-over records; evidence pointers; structured verification
  results.

Forbidden:
  captions; caption snippets; transcript snippets; caption embeddings;
  free-form summaries; GEI prompt timelines not materialized from `G_global_v1`;
  arbitrary raw-frame inspection; direct `frame_inspect_tool`;
  `clip_search_tool`; `global_browse_tool`.

Temporary exception:
  `frame_inspect_tool` may be used only internally by `VisualVerify`, and only
  for structured claims grounded by graph-derived evidence pointers.

---

## Progress Overview

| Module | Name | Status | Current Anchor |
|---|---|---|---|
| M0 | Caption-free audit | Temporary | `scripts/audit_caption_free.py`, `tests/test_caption_free_mode.py` |
| M1 | Temporal unit builder | Temporary | `frame_caption.py`, `video_utils.py`, `reproduce/` |
| M2 | Entity registry / lifecycle MVP | To Refactor | Stage1 `entities`, Stage2 `active_entities` |
| M3 | Frame facts | Temporary | `sgd/scene_graph/stage1_builder.py`, `query.py` |
| M4 | Temporal fact consolidation | To Refactor | `sgd/scene_graph/segment_builder.py` |
| M5 | Segment events | To Refactor | Stage2 `temporal_relations`, `summary` |
| M6 | Global graph v1 | Planned | GEI exists as temporary navigation aid |
| M7 | Verify / repair | Planned | policy gates and audit exist; graph verifier absent |
| M8 | Graph store / index | To Refactor | JSON graph files, NanoVectorDB, `additional_data` |
| M9 | Toolset wrapper | Planned | existing low-level tools only |
| M10 | Graph agent | To Refactor | `GRAPH_ONLY_INFERENCE`, `stream_run`, `finish(answer)` |
| M11 | Evaluation / demo | Temporary | `evaluation/`, `scripts/`, `app.py` |

---

## M0. Caption-Free Audit

Status:
  Temporary

Goal:
  Enforce strict graph-only boundaries before, during, and after inference.

Inputs:
  inference graph files, NanoVectorDB metadata, tool outputs, traces,
  benchmark results.

Outputs:
  audit report, stripped inference artifacts, pass/fail gate for strict runs.

Current implementation:
  `scripts/audit_caption_free.py` checks forbidden tools, forbidden payload
  markers, graph-only DB keys, and runtime mode. `tests/test_caption_free_mode.py`
  covers graph-only tool selection, DB metadata stripping, Stage1 caption hiding,
  Stage2 caption/embedding exclusion, and GEI caption exclusion.

Gap to Alg-plan:
  Audit does not yet enforce a raw-vs-stripped graph split. It also does not
  validate structured `AnswerWithEvidence` chains or block free-form `summary`
  fields outside DB payload scans.

Tasks:
  - [x] Add graph-only tool-selection regression tests.
  - [x] Add DB metadata stripping checks.
  - [x] Add Stage1/Stage2 caption leakage regression tests.
  - [x] Add run-level audit script.
  - [ ] Define forbidden/allowed field schema for inference graphs.
  - [ ] Add `construction_graph_raw.json` -> `inference_graph_stripped.json` stripper.
  - [ ] Audit final `AnswerWithEvidence` traces for graph support.
  - [ ] Audit tool inputs/outputs, trace JSONL, benchmark logs, DB metadata, and
    rendered replay/demo logs for leakage.
  - [ ] Add CI/smoke gate for strict graph-only runs.

Acceptance criteria:
  A strict graph-only run fails if captions, caption snippets, caption embeddings,
  transcript snippets, free-form summaries, `clip_search_tool`,
  `global_browse_tool`, direct `frame_inspect_tool`, or GEI prompt timelines
  appear in Agent-visible observations, tool I/O, traces, final answer records,
  graph DB metadata, benchmark logs, or rendered replay/demo logs.

Risks:
  Construction artifacts may still contain captions. The audit must distinguish
  construction-only files from inference-visible files.

---

## M1. Temporal Unit Builder

Status:
  Temporary

Goal:
  Build frame units, micro-clip units, and segment windows that downstream graph
  builders can reference.

Inputs:
  raw video, decoded frames, optional subtitle/ASR for construction only.

Outputs:
  frames, timestamps, segment windows, optional shot/action/event boundary hints.

Current implementation:
  Existing decoding, caption generation, lite mode, and LVBench preparation
  scripts are present. Stage1 sampling uses fixed keyframe intervals; Stage2
  uses 30s windows with overlap.

Gap to Alg-plan:
  No first-class temporal-unit manifest. No content-adaptive sampling, shot
  boundary detection, motion-peak sampling, or event-boundary refinement.

Tasks:
  - [x] Restore `process_video_lite` and `LITE_MODE` for development workflows.
  - [x] Restore `reproduce/` benchmark entrypoints.
  - [ ] Add `temporal_units.json` with frame, micro-clip, and segment entries.
  - [ ] Record exact frame IDs and timestamps used by graph builders.
  - [ ] Add optional shot-boundary / motion-peak hooks.
  - [ ] Ensure temporal units contain no inference-visible caption text.

Acceptance criteria:
  Every fact/event/evidence pointer can refer to stable frame IDs and time spans
  from a temporal-unit manifest.

Risks:
  Existing caption-driven preparation paths are useful for construction but must
  not become inference memory.

---

## M2. Entity Registry / Entity Lifecycle MVP

Status:
  To Refactor

Goal:
  Build `E_video`, a persistent entity registry with aliases, time spans, support
  frames, and enough lifecycle information for EntityResolve and EvidenceTrace.

Inputs:
  Stage1 frame entities, support frames, optional construction-only appearance
  descriptions.

Outputs:
  `entity_registry.json` or equivalent section in stripped graph memory.

Current implementation:
  Stage1 graphs contain entities and per-frame entity mentions. Stage2 segments
  contain active entities. There is no dedicated entity lifecycle registry.

Gap to Alg-plan:
  MVP requires `entity_id`, `category`, `aliases`, `time_spans`,
  `support_frame_ids`, `source`, and `confidence`. Current data is distributed
  across Stage1/Stage2 and is not a stable query target.

Tasks:
  - [ ] Add MVP `EntityRecord` schema.
  - [ ] Build entity time spans from Stage1 frame mentions and Stage2 active entities.
  - [ ] Attach support frame IDs and timestamps.
  - [ ] Preserve aliases only if they are entity-local referring expressions
    derived from graph-safe attributes, categories, or repeated entity labels.
    Allowed examples: `person`, `man in black sweater`, `man with mustache`,
    `red car`.
  - [ ] Strip or mark construction-only any free-form narrative aliases copied
    from captions. Forbidden strict examples: `the man who argues with the woman
    after entering the room`, `the villain who later breaks the mirror`.
  - [ ] Add entity merge/split audit hooks.
  - [ ] Defer bbox, tracklet, mask, crop, and visual embeddings to v2.0 full.

Acceptance criteria:
  EntityResolve can return entity candidates with aliases, time spans, support
  frames, and confidence without exposing captions.

Risks:
  Caption-derived aliases can leak narration. They need provenance and strict
  stripping rules.

---

## M3. Frame Facts

Status:
  Temporary

Goal:
  Extract frame/moment-level graph facts: entity presence, attributes, spatial
  relations, contact/holding relations, and local actions.

Inputs:
  temporal units, entity registry, frame images, construction-only captions if enabled.

Outputs:
  frame facts with fact IDs, time spans, confidence, and support frames.

Current implementation:
  `stage1_builder.py` builds Stage1 frame graphs; `query.py` retrieves frame-level
  graph matches. Caption context can exist in construction graphs and is hidden
  in graph-only query output.

Gap to Alg-plan:
  Current Stage1 schema is frame-centric rather than fact-centric. Facts are not
  first-class `GraphFact` records with stable IDs and evidence pointers.

Tasks:
  - [ ] Add `GraphFact` schema for frame facts.
  - [ ] Convert Stage1 entities/relations into fact records.
  - [ ] Require `source` / `source_provenance` on every fact.
  - [ ] Require `status` on every fact: `unverified`, `verified`, `rejected`,
    or `unsupported`.
  - [ ] Replace free-form `caption_context` in inference artifacts with structured signatures.
  - [ ] Add support frame IDs to every fact.
  - [ ] Add frame-level fact precision evaluation on an annotated subset.

Acceptance criteria:
  RelationQuery can return fact records without captions, and every returned fact
  has `fact_id`, subject, predicate, object, time span, support frames,
  confidence, source/provenance, and status.

Risks:
  Existing caption fallback is useful for graph construction but must remain
  construction-only.

---

## M4. Temporal Fact Consolidation

Status:
  To Refactor

Goal:
  Consolidate repeated frame facts into interval facts.

Inputs:
  frame facts, timestamps, entity registry.

Outputs:
  interval facts with smoothed time spans, confidence aggregation, and conflict flags.

Current implementation:
  Stage2 builder aggregates frame graph content into segment-level structures.
  Interval facts are implicit, not first-class.

Gap to Alg-plan:
  No dedicated interval-fact layer. Temporal smoothing, hysteresis, minimum
  duration filtering, and contradiction suppression are not represented as a
  separate auditable step.

Tasks:
  - [ ] Add interval fact schema.
  - [ ] Group frame facts by subject/predicate/object.
  - [ ] Smooth adjacent observations into time spans.
  - [ ] Track confidence aggregation and conflicts.
  - [ ] Feed interval facts into Segment Event Inducer.

Acceptance criteria:
  Segment event induction consumes interval facts rather than raw frame snippets
  or free-form summaries.

Risks:
  Over-smoothing may hide event boundaries; under-smoothing may fragment events.

---

## M5. Segment Events

Status:
  To Refactor

Goal:
  Induce segment-level event nodes, state transitions, participants, effects,
  support mappings, and structured event signatures.

Inputs:
  interval facts, segment windows, entity registry, neighboring segment context.

Outputs:
  event nodes, state transitions, event signatures, support mappings.

Current implementation:
  Stage2 segment graph contains active entities, temporal relations, summaries,
  optional caption snippets, and optional embeddings.

Gap to Alg-plan:
  Stage2 does not yet expose first-class `EventNode` and `StateTransition`
  records. Free-form `summary` is a strict-mode leakage risk and must become
  `graph_signature` / `event_signature`.

Tasks:
  - [ ] Define `EventNode` and `StateTransition` schema.
  - [ ] Convert `temporal_relations` into event/fact-support structures where possible.
  - [ ] Replace free-form `summary` with structured `event_signature`;
    do not merely rename the field.
  - [ ] Forbid string-valued natural-language `event_signature` in strict mode.
  - [ ] Disable caption snippets in stripped inference graphs.
  - [ ] Add event boundary IoU and participant accuracy evaluation.

Acceptance criteria:
  EventSearch returns event nodes with participants, time spans, confidence, and
  support frame/fact references, not narrative summaries. `event_signature` is
  a structured object containing event type, participants, time span,
  preconditions, effects, and support facts.

Risks:
  Existing Stage2 summaries improve retrieval but are not valid strict evidence.

---

## M6. Global Graph v1

Status:
  Planned

Goal:
  Build the minimal Stage3 global graph needed for long-range temporal reasoning.

Inputs:
  segment events, entity lifecycle records, simple state records.

Outputs:
  `G_global_v1`:
  entity_lifecycle_index, global_event_nodes, temporal_edges, first_last_index,
  simple_state_carry_over.

Current implementation:
  GEI compresses Stage2 segments into a prompt-injected navigation aid. It is not
  a materialized global graph, is not evidence, and is not valid for Strict
  Graph-only unless stripped into structured graph signatures.

Gap to Alg-plan:
  `stage3_builder.py`, `query_global.py`, temporal DAG, first/last index, and
  state carry-over are missing.

Tasks:
  - [ ] Add `GlobalGraphV1` schema.
  - [ ] Implement `sgd/scene_graph/stage3_builder.py`.
  - [ ] Build temporal edges from event time spans.
  - [ ] Build first/last indexes for entity-event and relation queries.
  - [ ] Add simple holder/location/visibility carry-over.
  - [ ] Add `query_global.py` for temporal order queries.
  - [ ] Rebuild GEI from `G_global_v1` when available.
  - [ ] Keep causal edges, episodes, and causal path reasoning out of GlobalGraphV1.

Acceptance criteria:
  TemporalOrderQuery can answer first/last/before/after from `G_global_v1` and
  trace evidence back to segment events and frame facts.

Risks:
  Global graph can reduce accuracy if used broadly. Router/policy must constrain
  it to first/last, before/after, state carry-over, and entity lifecycle questions.

---

## M7. Verify / Repair

Status:
  Planned

Goal:
  Verify graph facts/events/paths and repair or downgrade unsupported structures.

Inputs:
  graph memory, support mappings, evidence pointers, candidate claims.

Outputs:
  verified graph records, repair logs, unsupported/low-confidence markers.

Current implementation:
  Runtime policy gates and controller signals exist. Caption-free audit exists.
  There is no graph verifier or repair loop.

Gap to Alg-plan:
  EntityVerifier, RelationVerifier, StateVerifier, TemporalVerifier,
  EvidenceVerifier, and GraphVerify are absent. MVP uses support completeness
  checks as a lightweight verifier; full `GraphVerify` is deferred to v2.0 full.

Tasks:
  - [ ] Add support completeness checks for facts/events.
  - [ ] Verify `support_frame_ids` exist for evidence-bearing facts/events.
  - [ ] Verify unsupported events are not used as answer evidence.
  - [ ] Verify no forbidden caption-like fields appear in evidence chains.
  - [ ] Add temporal contradiction checks for Stage3 v1.
  - [ ] Mark unsupported events/facts as non-evidence.
  - [ ] Add `GraphVerify` as v2.0 full, after MVP tool wrappers.
  - [ ] Add repair logs for schema-level fixes.

Acceptance criteria:
  AnswerWithEvidence refuses or downgrades answers whose supporting chain lacks
  support frames or contains unsupported events.

Risks:
  Over-verification can create loops; limits and convergence checks are required.

---

## M8. Graph Store / Index

Status:
  To Refactor

Goal:
  Store graph-native memory and indexes separately from caption-based databases.

Inputs:
  verified graph records, stripped inference artifacts, optional graph-derived embeddings.

Outputs:
  GraphStore, EvidenceStore, IndexStore, and stripped inference DB metadata.

Current implementation:
  Stage1/Stage2 graph files and NanoVectorDB metadata are used. Caption-vector DBs
  are skipped in graph-only mode, and caption metadata is stripped from graph-only DBs.

Gap to Alg-plan:
  No explicit GraphStore/EvidenceStore/IndexStore split. No raw/stripped graph
  artifact pair. Graph-derived embeddings need strict provenance rules.

Tasks:
  - [ ] Define graph store layout.
  - [ ] Write stripped inference graph files.
  - [ ] Separate construction raw artifacts from inference artifacts.
  - [ ] Add entity/relation/event/temporal/evidence indexes.
  - [ ] Allow only tuple-label or graph-signature embeddings in strict mode.
  - [ ] Require every embedding entry to record `embedding_source`.
  - [ ] Forbid `embedding_source` values `caption`, `transcript`,
    `free_form_summary`, and `event_sentence_summary` in strict mode.

Acceptance criteria:
  A strict run can be executed from stripped graph memory without reading caption
  files or caption-vector indexes. Every vector index entry records an allowed
  graph-derived `embedding_source`, such as `tuple_label` or
  `structured_graph_signature`.

Risks:
  Existing database metadata may silently point back to caption artifacts; audit
  must catch this.

---

## M9. Toolset Wrapper

Status:
  Planned

Goal:
  Expose Alg-plan semantic tools to the Agent while reusing existing low-level
  implementations internally.

Inputs:
  GraphStore, EvidenceStore, low-level Stage1/Stage2/GEI/frame-inspect functions.

Outputs:
  Agent-facing tools:
  QuestionDecompose, EntityResolve, RelationQuery, EventSearch,
  TemporalOrderQuery, EvidenceTrace, VisualVerify, AnswerWithEvidence.

Current implementation:
  Agent currently sees low-level tools such as `frame_graph_query_tool`,
  `segment_graph_query_tool`, `clip_search_tool` in hybrid mode, and
  `frame_inspect_tool`.

Gap to Alg-plan:
  No wrapper layer. Direct `frame_inspect_tool` exposure must be removed from
  Agent-facing graph-only action space.

Tasks:
  - [ ] Add wrapper module, likely `sgd/graph_tools/` or `sgd/agent/tools.py`.
  - [ ] Implement QuestionDecompose as the source of `parsed_intent`.
  - [ ] Implement EntityResolve over entity registry.
  - [ ] Wrap frame graph query as RelationQuery.
  - [ ] Wrap segment graph query as EventSearch.
  - [ ] Implement TemporalOrderQuery over Stage3 v1 / GEI-MVP navigation.
  - [ ] Implement EvidenceTrace over support mappings.
  - [ ] Wrap frame inspect internally as VisualVerify.
  - [ ] Restrict VisualVerify inputs to structured claims plus graph-derived
  evidence pointers or candidate intervals.
  - [ ] Reject arbitrary timestamps and free-form inspection prompts at the
  VisualVerify wrapper boundary.
  - [ ] Add output sanitizer for every wrapper tool.
  - [ ] Replace `finish(answer)` with AnswerWithEvidence.

Acceptance criteria:
  In graph-only mode, Agent action space contains only Alg-plan semantic tools
  and no direct caption or free-form visual inspection tools. Every wrapper
  sanitizes low-level tool outputs before returning them to the Agent.

Risks:
  Wrapper outputs can accidentally preserve old field names such as `summary` or
  `caption_context`; M0 audit must cover tool outputs.

---

## M10. Graph Agent

Status:
  To Refactor

Goal:
  Run graph-driven observe-reason-act with structured belief state, semantic
  graph tools, evidence tracing, and structured final answers.

Inputs:
  question, graph tools, graph memory, runtime policy.

Outputs:
  AnswerWithEvidence JSON, trace, tool history, confidence/uncertainty metadata.

Current implementation:
  `GRAPH_ONLY_INFERENCE` removes caption search/global browse from graph-only
  tool selection. `stream_run` exists. Policy gates discourage premature answers.

Gap to Alg-plan:
  Agent does not yet use Alg-plan semantic tool names. Belief state is implicit.
  Final `finish(answer)` is not structured evidence JSON. VisualVerify is not a
  constrained Agent-facing action.

Tasks:
  - [ ] Switch graph-only Agent action space to M9 wrapper tools.
  - [ ] Add explicit belief-state fields for parsed intent, resolved entities,
  candidate events/facts, evidence chain, conflicts, and uncertainty.
  - [ ] Require EvidenceTrace before AnswerWithEvidence.
  - [ ] Add structured final-answer schema to traces and benchmark output.
  - [ ] Keep short answer extraction for LVBench evaluator.

Acceptance criteria:
  Every completed graph-only answer must pass AnswerWithEvidence validation:
  at least one supporting graph fact or event; at least one support frame; no
  forbidden caption-like fields; no direct `frame_inspect_tool` call; and if
  GEI-MVP was used, it is recorded as navigation-only and excluded from evidence.

Risks:
  More structured tools can reduce flexibility; MVP adapters should preserve
  enough retrieval recall while keeping the interface clean.

---

## M11. Evaluation / Demo

Status:
  Temporary

Goal:
  Evaluate answer accuracy, evidence correctness, temporal grounding, graph
  quality, caption-free boundaries, and provide replay/demo tools.

Inputs:
  benchmark results, traces, graph artifacts, diagnostic annotations.

Outputs:
  accuracy reports, graph quality reports, caption-free audits, trace replay,
  graph visualization, demo app.

Current implementation:
  `evaluation/` exists with agent/graph/stage3 entrypoints, but Stage3 evaluator
  is a placeholder. Scripts for run summaries, trace rendering, monitoring, and
  smoke runs exist. `app.py` exists and consumes `stream_run`, but it is still a
  text/log style demo and can expose old semantics.

Gap to Alg-plan:
  No evidence-chain evaluation. No graph-native retrieval recall on annotated
  evidence. Demo plan still references captions in graph UI and must be
  sanitized for strict graph-only.

Tasks:
  - [x] Restore trace logger and trace utilities.
  - [x] Restore benchmark/evaluation scripts.
  - [x] Add graph-only smoke runner and caption-free audit runner.
  - [ ] Add run-mode labels to every evaluation report:
    Strict Graph-only, MVP Graph-guided, Builder-assisted, or Hybrid.
  - [ ] Add `run_graph_eval.py all` markdown summary.
  - [ ] Add evidence-chain validity evaluator.
  - [ ] Add diagnostic subset schema and annotation format.
  - [ ] Add graph retrieval recall@k / mIoU / MRR evaluation.
  - [ ] Add Stage3 v1 evaluator after M6 lands.
  - [ ] Sanitize demo/viewer data to show graph facts/evidence, not captions.
  - [ ] In strict mode, demo must render only graph facts, event signatures,
    evidence pointers, and verification results.
  - [ ] Build replay from standardized trace and graph bundle.

Acceptance criteria:
  Main experiments report accuracy, evidence validity, caption-free pass/fail,
  per-module graph quality, and run mode. Demo visualizes graph evidence without
  exposing inference-time captions.

Risks:
  Accuracy-only evaluation can hide caption leakage or unsupported evidence.

---

## Near-Term Work Order

1. Finish M0 strict artifact split and audit extension.
2. Add M1 `temporal_units.json` with stable frame IDs and timestamps.
3. Implement M2 entity registry MVP and M3 fact IDs/support frames.
4. Refactor M5 free-form `summary` into structured `event_signature`.
5. Build M9 wrapper layer so Agent-facing tools match Alg-plan.
6. Upgrade M10 final answer to AnswerWithEvidence.
7. Implement M6 Stage3 v1 temporal graph.
8. Add M11 evidence-chain and graph retrieval evaluation.

---

## Non-Canonical Historical Context

Archived documents:
  `docs/plans/SG-plan.md`
  `docs/plans/app-plan.md`

Operational documents retained as-is:
  `docs/plans/handoff-20260513.md`
  `docs/plans/private-public-sync-plan.md`
  `docs/baselines/new-baseline.md`

These files may provide context but do not define current algorithm semantics or
development order.
