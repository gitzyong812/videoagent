#!/bin/bash
# Run graph-based DVD smoke evaluation with external data/log layout.
# Pipeline:
#   1) (Optional) run graph-enabled QA benchmark to produce per-video logs + traces
#   2) render traces into markdown + CSV index
#   3) run stage1/stage2 graph quality & retrieval evaluation report

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${PYTHON_BIN:-}" || "$PYTHON_BIN" == "python3" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  fi
fi

# -------- defaults (external layout first) --------
DATA_ROOT="${SGD_DATA_ROOT:-/Users/toniekule/Dev/dvd_db}"
EVAL_ARTIFACTS_ROOT="${SGD_EVAL_ARTIFACTS_ROOT:-$DATA_ROOT/eval_artifacts}"
LOGS_BASE_DIR="${SGD_LOGS_ROOT:-$DATA_ROOT/logs}"
ASSETS_ROOT="${SGD_LVBENCH_ROOT:-$DATA_ROOT/LVBench/LVBench_4.1}"

RUN_NAME="stage12_vlm_smoke"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
VIDEOS="Cm73ma6Ibcs,q01CUy_gwdU"

ARTIFACTS_ROOT="${SGD_ARTIFACTS_ROOT:-${SGD_BADCASES_ROOT:-${SGD_CAPTION_ROOT:-}}}"
META_PATH=""
OUTPUT_PATH=""

RUN_BENCHMARK=1
MAX_WORKERS=4

USE_VLM=1
BUILD_IF_MISSING=1
FORCE_REBUILD=1
ENABLE_STAGE2=1

TOP_K=8
TOP_K_SEGMENTS=6
STAGE1_SAMPLE_SECS=0
STAGE1_MAX_KEYFRAMES=20
STAGE2_WINDOW_SECS=30
STAGE2_OVERLAP_SECS=5
TOLERANCE_SEC=5

TRACE_DIR_OVERRIDE=""

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --run-name NAME               Run name prefix (default: $RUN_NAME)
  --videos IDS                  Comma-separated video ids (default: $VIDEOS)
  --artifacts-root DIR          Artifacts root (contains <video_id>/captions and scene_graph files)
  --badcases-root DIR           Legacy alias of --artifacts-root
  --assets-root DIR             Asset root (contains <video_id>/frames and <video_id>/database.json)
  --meta FILE                   Meta jsonl path (default: <assets-root>/video_info.meta.jsonl)
  --output FILE                 Stage12 report output json path
  --trace-dir DIR               Reuse existing trace dir (skip or merge with benchmark traces)
  --skip-benchmark              Skip reproduce.run_benchmark step
  --max-workers N               Benchmark parallel workers (default: $MAX_WORKERS)
  --top-k N                     Stage1 retrieval top-k (default: $TOP_K)
  --top-k-segments N            Stage2 retrieval top-k segments (default: $TOP_K_SEGMENTS)
  --stage1-sample-secs N        Stage1 sample seconds override (default: $STAGE1_SAMPLE_SECS)
  --stage1-max-keyframes N      Stage1 max keyframes (default: $STAGE1_MAX_KEYFRAMES)
  --stage2-window-secs N        Stage2 window seconds (default: $STAGE2_WINDOW_SECS)
  --stage2-overlap-secs N       Stage2 overlap seconds (default: $STAGE2_OVERLAP_SECS)
  --tolerance-sec N             Temporal hit tolerance seconds (default: $TOLERANCE_SEC)
  --use-vlm                     Explicitly enable VLM when building Stage1 graph
  --no-use-vlm                  Disable VLM when building Stage1 graph
  --no-build-if-missing         Disable --build-if-missing
  --no-force-rebuild            Disable --force-rebuild
  --disable-stage2              Disable Stage2 build/eval
  --help, -h                    Show this help

Environment defaults:
  SGD_DATA_ROOT=/Users/toniekule/Dev/dvd_db
  SGD_EVAL_ARTIFACTS_ROOT=\$SGD_DATA_ROOT/eval_artifacts
  SGD_LOGS_ROOT=\$SGD_DATA_ROOT/logs
  SGD_LVBENCH_ROOT=\$SGD_DATA_ROOT/LVBench/LVBench_4.1
  SGD_CAPTION_ROOT=...          # used as fallback for artifacts root
  SGD_ARTIFACTS_ROOT=...
  SGD_BADCASES_ROOT=...         # legacy alias
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --videos) VIDEOS="$2"; shift 2 ;;
    --artifacts-root) ARTIFACTS_ROOT="$2"; shift 2 ;;
    --badcases-root) ARTIFACTS_ROOT="$2"; shift 2 ;;
    --assets-root) ASSETS_ROOT="$2"; shift 2 ;;
    --meta) META_PATH="$2"; shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --trace-dir) TRACE_DIR_OVERRIDE="$2"; shift 2 ;;
    --skip-benchmark) RUN_BENCHMARK=0; shift ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --top-k-segments) TOP_K_SEGMENTS="$2"; shift 2 ;;
    --stage1-sample-secs) STAGE1_SAMPLE_SECS="$2"; shift 2 ;;
    --stage1-max-keyframes) STAGE1_MAX_KEYFRAMES="$2"; shift 2 ;;
    --stage2-window-secs) STAGE2_WINDOW_SECS="$2"; shift 2 ;;
    --stage2-overlap-secs) STAGE2_OVERLAP_SECS="$2"; shift 2 ;;
    --tolerance-sec) TOLERANCE_SEC="$2"; shift 2 ;;
    --use-vlm) USE_VLM=1; shift ;;
    --no-use-vlm) USE_VLM=0; shift ;;
    --no-build-if-missing) BUILD_IF_MISSING=0; shift ;;
    --no-force-rebuild) FORCE_REBUILD=0; shift ;;
    --disable-stage2) ENABLE_STAGE2=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$ARTIFACTS_ROOT" ]]; then
  ARTIFACTS_ROOT="$EVAL_ARTIFACTS_ROOT/${RUN_NAME}_${TIMESTAMP}/badcases"
fi
if [[ -z "$META_PATH" ]]; then
  META_PATH="$ASSETS_ROOT/video_info.meta.jsonl"
fi

RUN_LOG_DIR="$LOGS_BASE_DIR/${RUN_NAME}_${TIMESTAMP}"
PER_VIDEO_LOG_DIR="$RUN_LOG_DIR/per_video"
TRACE_DIR="$RUN_LOG_DIR/traces"
TRACE_REPORT_DIR="$RUN_LOG_DIR/trace_reports"
TRACE_INDEX_CSV="$RUN_LOG_DIR/trace_index.csv"
RUN_SUMMARY_JSON="$RUN_LOG_DIR/run_summary.json"
RUN_LOG_FILE="$RUN_LOG_DIR/run_stage12_pipeline.log"

if [[ -n "$TRACE_DIR_OVERRIDE" ]]; then
  TRACE_DIR="$TRACE_DIR_OVERRIDE"
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$EVAL_ARTIFACTS_ROOT/${RUN_NAME}_${TIMESTAMP}/reports/stage12_eval_report_vlm_smoke.json"
fi

mkdir -p "$RUN_LOG_DIR" "$(dirname "$OUTPUT_PATH")" "$TRACE_REPORT_DIR"
mkdir -p "$ARTIFACTS_ROOT"

# Log all output to main run log.
exec > >(tee -a "$RUN_LOG_FILE") 2>&1

echo "=========================================="
echo "Graph DVD Stage1+Stage2 Smoke Runner"
echo "=========================================="
echo ""

echo -e "${BLUE}Configuration:${NC}"
echo "  Project Root:          $PROJECT_ROOT"
echo "  Data Root:             $DATA_ROOT"
echo "  Assets Root:           $ASSETS_ROOT"
echo "  Eval Artifacts Root:   $EVAL_ARTIFACTS_ROOT"
echo "  Logs Root:             $LOGS_BASE_DIR"
echo "  Run Name:              $RUN_NAME"
echo "  Timestamp:             $TIMESTAMP"
echo "  Artifacts Root:        $ARTIFACTS_ROOT"
echo "  Meta Source:           $META_PATH"
echo "  Videos:                $VIDEOS"
echo "  Run Benchmark:         $RUN_BENCHMARK"
echo "  Per-video Log Dir:     $PER_VIDEO_LOG_DIR"
echo "  Trace Dir:             $TRACE_DIR"
echo "  Trace Report Dir:      $TRACE_REPORT_DIR"
echo "  Stage12 Report Output: $OUTPUT_PATH"
echo ""

if [[ ! -f "$META_PATH" ]]; then
  echo -e "${YELLOW}Error: meta file not found: $META_PATH${NC}"
  exit 1
fi

EVAL_META_PATH="$META_PATH"
if [[ -n "$VIDEOS" ]]; then
  EVAL_META_PATH="$RUN_LOG_DIR/meta_subset.jsonl"
  rm -f "$EVAL_META_PATH"
  IFS=',' read -ra _VIDEO_IDS <<< "$VIDEOS"
  for _vid in "${_VIDEO_IDS[@]}"; do
    _vid_trim="$(echo "$_vid" | xargs)"
    [[ -z "$_vid_trim" ]] && continue
    if command -v rg >/dev/null 2>&1; then
      rg "\"key\"\\s*:\\s*\"${_vid_trim}\"" "$META_PATH" >> "$EVAL_META_PATH" || true
    else
      grep "\"key\": \"${_vid_trim}\"" "$META_PATH" >> "$EVAL_META_PATH" || true
    fi
  done
  if [[ ! -s "$EVAL_META_PATH" ]]; then
    echo -e "${YELLOW}Error: no video rows selected from meta for videos=$VIDEOS${NC}"
    exit 1
  fi
fi

cat > "$RUN_LOG_DIR/run_config.txt" <<EOF
Run Configuration
=================
timestamp: $TIMESTAMP
run_name: $RUN_NAME
project_root: $PROJECT_ROOT
data_root: $DATA_ROOT
eval_artifacts_root: $EVAL_ARTIFACTS_ROOT
logs_root: $LOGS_BASE_DIR
assets_root: $ASSETS_ROOT
artifacts_root: $ARTIFACTS_ROOT
meta_source_path: $META_PATH
meta_eval_path: $EVAL_META_PATH
videos: $VIDEOS
run_benchmark: $RUN_BENCHMARK
max_workers: $MAX_WORKERS
use_vlm: $USE_VLM
build_if_missing: $BUILD_IF_MISSING
force_rebuild: $FORCE_REBUILD
enable_stage2: $ENABLE_STAGE2
top_k: $TOP_K
top_k_segments: $TOP_K_SEGMENTS
stage1_sample_secs: $STAGE1_SAMPLE_SECS
stage1_max_keyframes: $STAGE1_MAX_KEYFRAMES
stage2_window_secs: $STAGE2_WINDOW_SECS
stage2_overlap_secs: $STAGE2_OVERLAP_SECS
tolerance_sec: $TOLERANCE_SEC
trace_dir: $TRACE_DIR
output_path: $OUTPUT_PATH
EOF

if [[ "$RUN_BENCHMARK" -eq 1 ]]; then
  echo "------------------------------------------"
  echo -e "${GREEN}Step 1/5: Run graph-enabled QA benchmark${NC}"
  echo "------------------------------------------"
  mkdir -p "$PER_VIDEO_LOG_DIR" "$TRACE_DIR"

  SGD_TRACE_ENABLED=1 SGD_TRACE_LOG_DIR="$TRACE_DIR" \
  PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/pycache \
  "$PYTHON_BIN" -m reproduce.run_benchmark \
    "$ASSETS_ROOT" \
    "$EVAL_META_PATH" \
    --per-video-log-dir "$PER_VIDEO_LOG_DIR" \
    --max-workers "$MAX_WORKERS" \
    --trace-log-dir "$TRACE_DIR" \
    --caption-root "$ARTIFACTS_ROOT"
else
  echo -e "${YELLOW}Step 1/5 skipped: benchmark is disabled (--skip-benchmark).${NC}"
fi

if [[ -d "$TRACE_DIR" ]]; then
  echo "------------------------------------------"
  echo -e "${GREEN}Step 2/5: Render trace markdown reports${NC}"
  echo "------------------------------------------"
  "$PYTHON_BIN" scripts/render_trace_report.py \
    --trace-dir "$TRACE_DIR" \
    --output-dir "$TRACE_REPORT_DIR"

  echo "------------------------------------------"
  echo -e "${GREEN}Step 3/5: Build trace CSV index${NC}"
  echo "------------------------------------------"
  "$PYTHON_BIN" scripts/trace_index.py \
    "$TRACE_DIR" \
    "$TRACE_INDEX_CSV"
else
  echo -e "${YELLOW}Step 2/5 and 3/5 skipped: trace dir does not exist: $TRACE_DIR${NC}"
fi

echo "------------------------------------------"
echo -e "${GREEN}Step 4/5: Evaluate stage1/stage2 graph metrics${NC}"
echo "------------------------------------------"

EVAL_CMD=(
  "$PYTHON_BIN"
  "$PROJECT_ROOT/evaluation/evaluate_stage1_graph.py"
  --meta "$EVAL_META_PATH"
  --artifacts-root "$ARTIFACTS_ROOT"
  --assets-root "$ASSETS_ROOT"
  --videos "$VIDEOS"
  --top-k "$TOP_K"
  --top-k-segments "$TOP_K_SEGMENTS"
  --stage1-sample-secs "$STAGE1_SAMPLE_SECS"
  --stage1-max-keyframes "$STAGE1_MAX_KEYFRAMES"
  --stage2-window-secs "$STAGE2_WINDOW_SECS"
  --stage2-overlap-secs "$STAGE2_OVERLAP_SECS"
  --tolerance-sec "$TOLERANCE_SEC"
  --trace-dir "$TRACE_DIR"
  --output "$OUTPUT_PATH"
)

if [[ "$BUILD_IF_MISSING" -eq 1 ]]; then
  EVAL_CMD+=(--build-if-missing)
fi
if [[ "$FORCE_REBUILD" -eq 1 ]]; then
  EVAL_CMD+=(--force-rebuild)
fi
if [[ "$USE_VLM" -eq 1 ]]; then
  EVAL_CMD+=(--use-vlm)
else
  EVAL_CMD+=(--no-use-vlm)
fi
if [[ "$ENABLE_STAGE2" -eq 0 ]]; then
  EVAL_CMD+=(--disable-stage2)
fi

PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/pycache "${EVAL_CMD[@]}"

echo "------------------------------------------"
echo -e "${GREEN}Step 5/5: Build run summary JSON${NC}"
echo "------------------------------------------"
if [[ -d "$TRACE_DIR" ]] && find "$TRACE_DIR" -name "*.jsonl" | grep -q .; then
  "$PYTHON_BIN" scripts/build_run_summary.py \
    --run-dir "$RUN_LOG_DIR" \
    --trace-dir "$TRACE_DIR" \
    --trace-index-csv "$TRACE_INDEX_CSV" \
    --benchmark-results "$RUN_LOG_DIR/benchmark_results.json" \
    --run-log "$RUN_LOG_FILE" \
    --stage-report "$OUTPUT_PATH" \
    --output "$RUN_SUMMARY_JSON"
else
  echo -e "${YELLOW}Skip run summary: no trace jsonl files under $TRACE_DIR${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Pipeline finished.${NC}"
echo "=========================================="
echo "Artifacts:"
echo "  Run log:            $RUN_LOG_FILE"
echo "  Run config:         $RUN_LOG_DIR/run_config.txt"
echo "  Per-video logs:     $PER_VIDEO_LOG_DIR"
echo "  Traces:             $TRACE_DIR"
echo "  Trace reports:      $TRACE_REPORT_DIR"
echo "  Trace index CSV:    $TRACE_INDEX_CSV"
echo "  Run summary JSON:   $RUN_SUMMARY_JSON"
echo "  Stage12 report:     $OUTPUT_PATH"
echo ""
