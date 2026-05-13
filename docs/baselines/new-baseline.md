# New VQA Baseline

This file is the development baseline for the next phase. It is built from run
artifacts, configs, logs, scripts, and code only. Paper text is not used as a
source of truth here.

## Baseline Runs

| Scope | Golden run | Questions | Correct | Wrong | Unknown | E2E acc | Answered acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full benchmark | `exp_g1g3_full103_stage12_vqa_20260319_023349_20260319_142130` | 1549 | 1051 | 467 | 31 | 67.85% | 69.24% |
| Bad38 | `graph_vqa_bad38_g5g6_20260316_231130` | 596 | 377 | 209 | 10 | 63.26% | 64.33% |

These are the best VQA runs currently found under
`/Users/toniekule/Dev/dvd_db/logs`.

## Shared Runtime Settings

- Orchestrator: `o4-mini` (`o4-mini-2025-04-16` in trace logs).
- Tool VLM: `gpt-4.1-mini`.
- Caption VLM: `gpt-4.1-mini`.
- Stage1 graph VLM: `gpt-4.1`.
- Embedding model: `text-embedding-3-large`, 3072 dimensions.
- `MAX_ITERATIONS=20`.
- `SINGLE_CHOICE_QA=True`.
- Tool top-k overrides: frame graph `8`, segment graph `6`, graph triples `24`.
- Stage1 graph: enabled, VLM enabled, sample every `5s`, max `120` keyframes, entity threshold `0.78`.
- Stage2 graph: enabled, `30s` windows, `5s` overlap, max `64` temporal relations per segment.
- Runtime policy gates and `controller_signals` were active in both golden VQA runs.
- GEI was not part of these golden runs. The GEI commits landed later, and the later GEI full run regressed badly.

Current `main` differs from the golden runtime: it defaults to `o3` and has
`GEI_ENABLED=True`. Do not compare a current run against this baseline unless
you either replay the historical code state or explicitly set the equivalent
no-GEI, `o4-mini` configuration.

## Full Benchmark Baseline

Run artifacts:

- Run dir: `/Users/toniekule/Dev/dvd_db/logs/exp_g1g3_full103_stage12_vqa_20260319_023349_20260319_142130`
- Results: `benchmark_results.json`
- Summary: `run_summary.json`
- Trace dir: `traces/`
- Assets root: `/Users/toniekule/Dev/dvd_db/LVBench/LVBench_4.1`
- Meta: `/Users/toniekule/Dev/dvd_db/LVBench/LVBench_4.1/video_info.meta.jsonl`
- Caption/graph root: `/Users/toniekule/Dev/dvd_db/eval_artifacts/graph_build_full103_20260319_023349/badcases`

Graph build source:

- Built in 4 workers under `/Users/toniekule/Dev/dvd_db/eval_artifacts/reports/graph_build_full103_20260319_023349/`.
- All worker configs used `--use-vlm`, `--force-rebuild`, `top_k=8`, `top_k_segments=6`, Stage2 `30/5`.
- Stage1/Stage2 graphs exist for all 103 videos.
- The Stage2 graphs include segment embeddings, but do not include `caption_snippets`.

Re-run shape:

```bash
SGD_DATA_ROOT=/Users/toniekule/Dev/dvd_db \
SGD_LVBENCH_ROOT=/Users/toniekule/Dev/dvd_db/LVBench/LVBench_4.1 \
SGD_CAPTION_ROOT=/Users/toniekule/Dev/dvd_db/eval_artifacts/graph_build_full103_20260319_023349/badcases \
SGD_LOGS_ROOT=/Users/toniekule/Dev/dvd_db/logs \
bash scripts/run_baseline.sh \
  --name exp_g1g3_full103_stage12_vqa_20260319_023349 \
  --max-workers 4
```

Before running, make sure the active code/config is equivalent to the golden
runtime: `AOAI_ORCHESTRATOR_LLM_MODEL_NAME="o4-mini"` and no GEI prompt
injection.

Regression gate:

- Golden target: `e2e_acc >= 0.6785`, `unknown <= 31`.
- Investigate if `e2e_acc < 0.6685`, `unknown > 50`, or
  `llm_call_returned_none > 50`.

## Bad38 Baseline

Run artifacts:

- Run dir: `/Users/toniekule/Dev/dvd_db/logs/graph_vqa_bad38_g5g6_20260316_231130`
- Results: `benchmark_results.json`
- Summary: `run_summary.json`
- Trace dir: `traces/`
- Assets root: `/Users/toniekule/Dev/dvd_db/LVBench/LVBench_4.1`
- Meta: `/Users/toniekule/Dev/dvd_db/LVBench/LVBench_4.1/video_info.meta_bad.jsonl`
- Caption/graph root: `/Users/toniekule/Dev/dvd_db/eval_artifacts/graph_build_bad38_20260307_023129/badcases`

Graph build source:

- Report: `/Users/toniekule/Dev/dvd_db/eval_artifacts/reports/graph_build_bad38_20260307_023129/stage12_graph_build_bad38.json`
- 38/38 videos have Stage1 and Stage2 graphs.
- Build used `--use-vlm`, `--force-rebuild`, `top_k=8`, `top_k_segments=6`.
- Stage1 graph config: sample every `5s`, max `120` keyframes.
- Stage2 graph config: `30s` windows, `5s` overlap.
- These Stage2 graphs do not include segment embeddings or `caption_snippets`.

Re-run shape:

```bash
SGD_DATA_ROOT=/Users/toniekule/Dev/dvd_db \
SGD_LVBENCH_ROOT=/Users/toniekule/Dev/dvd_db/LVBench/LVBench_4.1 \
SGD_CAPTION_ROOT=/Users/toniekule/Dev/dvd_db/eval_artifacts/graph_build_bad38_20260307_023129/badcases \
SGD_LOGS_ROOT=/Users/toniekule/Dev/dvd_db/logs \
bash scripts/run_baseline.sh \
  --name graph_vqa_bad38_g5g6 \
  --meta /Users/toniekule/Dev/dvd_db/LVBench/LVBench_4.1/video_info.meta_bad.jsonl \
  --max-workers 4
```

Before running, make sure the active code/config is equivalent to the golden
runtime: `AOAI_ORCHESTRATOR_LLM_MODEL_NAME="o4-mini"` and no GEI prompt
injection.

Regression gate:

- Golden target: `e2e_acc >= 0.6326`, `unknown <= 10`.
- Investigate if `e2e_acc < 0.6226`, `unknown > 15`, or
  `llm_call_returned_none > 15`.

## Known Non-Baselines

- `graph_vqa_full103_boost_v1_gei_20260326_185705`: full run with GEI, 1549
  questions, `785` correct, `465` unknown, `50.68%` E2E. This is a regression
  reference, not a baseline.
- `graph_vqa_bad38_stage12_boost_v1_20260321_082733`: bad38 run, `62.25%`
  E2E. Useful for comparison, but not the best bad38 baseline.

## Script Classification

Keep as baseline/reproduction entrypoints:

- `scripts/run_baseline.sh`: VQA benchmark runner.
- `reproduce/run_benchmark.py`: per-video/per-question benchmark engine.
- `scripts/run_stage12_parallel_build.sh`: full Stage1+Stage2 graph build runner.
- `evaluation/evaluate_stage1_graph.py`: graph build and graph retrieval metrics.
- `scripts/build_run_summary.py`: benchmark summary generator.

Keep as smoke/ops helpers:

- `scripts/run_graph_stage12_smoke.sh`: small graph-enabled smoke pipeline.
- `scripts/resume_benchmark.sh`: resume interrupted benchmark runs.
- `scripts/monitor_progress.sh`: tail and progress monitor for long runs.
- `scripts/smoke_test_embedding.py`: embedding API smoke check.

Keep as diagnostics:

- `scripts/render_trace_report.py`: human-readable trace reports.
- `scripts/trace_index.py`: trace CSV index.
- `scripts/compare_badcase_runs.py`: run-to-run badcase comparison.
- `scripts/analyze_regression_failure_modes.py`: regression bucket analysis.
- `scripts/classify_improvement_causes.py`: improvement cause classification.
- `evaluation/evaluate_results.py`: final result evaluation.

Legacy/manual data prep:

- `reproduce/download_lvbench.sh`
- `reproduce/decode_frames.py`
- `reproduce/transcribe.py`
- `reproduce/prepare_database.py`
- `evaluation/evaluate_results_enhanced.py` is a deprecated compatibility wrapper
  that forwards to `evaluation/evaluate_results.py`.

## Cleanup Policy

- Do not use paper files as baseline evidence.
- Keep tracked paper sources/assets under `docs/paper/` untouched unless the paper
  itself is being edited.
- Remove ignored generated artifacts before committing: `__pycache__/`,
  `.pytest_cache/`, `.DS_Store`, LaTeX build intermediates, and local preview
  outputs.
- Do not delete benchmark logs or graph artifacts under `/Users/toniekule/Dev/dvd_db`
  unless a run has been superseded and archived elsewhere.

