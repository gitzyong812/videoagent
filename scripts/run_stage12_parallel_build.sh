#!/bin/bash
# Build Stage1+Stage2 scene graphs in parallel batches.

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

DATA_ROOT="${SGD_DATA_ROOT:-/Users/toniekule/Dev/dvd_db}"
ASSETS_ROOT="${SGD_LVBENCH_ROOT:-$DATA_ROOT/LVBench/LVBench_4.1}"
META_PATH="$ASSETS_ROOT/video_info.meta.jsonl"
EVAL_ARTIFACTS_ROOT="${SGD_EVAL_ARTIFACTS_ROOT:-$DATA_ROOT/eval_artifacts}"
REPORTS_ROOT="${SGD_REPORTS_ROOT:-$EVAL_ARTIFACTS_ROOT/reports}"

RUN_NAME="graph_build_full103"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG=""
ARTIFACTS_ROOT=""
WORKERS=4
VIDEOS=""

USE_VLM=1
FORCE_REBUILD=1
TOP_K=8
TOP_K_SEGMENTS=6
STAGE2_WINDOW_SECS=30
STAGE2_OVERLAP_SECS=5

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --run-name NAME               Run name prefix (default: $RUN_NAME)
  --artifacts-root DIR          Build artifacts root (<video_id>/captions/...)
  --assets-root DIR             LVBench assets root (default: $ASSETS_ROOT)
  --meta FILE                   Meta jsonl path (default: $META_PATH)
  --videos IDS                  Comma-separated video ids; default is all videos in meta
  --workers N                   Parallel workers (default: $WORKERS)
  --top-k N                     Stage1 retrieval top-k (default: $TOP_K)
  --top-k-segments N            Stage2 retrieval top-k segments (default: $TOP_K_SEGMENTS)
  --stage2-window-secs N        Stage2 window size (default: $STAGE2_WINDOW_SECS)
  --stage2-overlap-secs N       Stage2 overlap size (default: $STAGE2_OVERLAP_SECS)
  --use-vlm                     Enable VLM for Stage1 build
  --no-use-vlm                  Disable VLM for Stage1 build
  --force-rebuild               Force rebuild graphs (default)
  --no-force-rebuild            Reuse existing graphs when present
  --help, -h                    Show this help

Environment defaults:
  SGD_DATA_ROOT=/Users/toniekule/Dev/dvd_db
  SGD_LVBENCH_ROOT=\$SGD_DATA_ROOT/LVBench/LVBench_4.1
  SGD_EVAL_ARTIFACTS_ROOT=\$SGD_DATA_ROOT/eval_artifacts
  SGD_REPORTS_ROOT=\$SGD_EVAL_ARTIFACTS_ROOT/reports
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --artifacts-root) ARTIFACTS_ROOT="$2"; shift 2 ;;
    --assets-root) ASSETS_ROOT="$2"; shift 2 ;;
    --meta) META_PATH="$2"; shift 2 ;;
    --videos) VIDEOS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --top-k-segments) TOP_K_SEGMENTS="$2"; shift 2 ;;
    --stage2-window-secs) STAGE2_WINDOW_SECS="$2"; shift 2 ;;
    --stage2-overlap-secs) STAGE2_OVERLAP_SECS="$2"; shift 2 ;;
    --use-vlm) USE_VLM=1; shift ;;
    --no-use-vlm) USE_VLM=0; shift ;;
    --force-rebuild) FORCE_REBUILD=1; shift ;;
    --no-force-rebuild) FORCE_REBUILD=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$META_PATH" ]]; then
  echo -e "${YELLOW}Error: meta file not found: $META_PATH${NC}"
  exit 1
fi

RUN_TAG="${RUN_NAME}_${TIMESTAMP}"
if [[ -z "$ARTIFACTS_ROOT" ]]; then
  ARTIFACTS_ROOT="$EVAL_ARTIFACTS_ROOT/$RUN_TAG/badcases"
fi
OUTPUT_DIR="$REPORTS_ROOT/$RUN_TAG"
mkdir -p "$ARTIFACTS_ROOT" "$OUTPUT_DIR"

KEYS_FILE="$(mktemp)"
SPLIT_DIR="$(mktemp -d)"
cleanup() {
  rm -f "$KEYS_FILE"
  rm -rf "$SPLIT_DIR"
}
trap cleanup EXIT

export META_PATH VIDEOS WORKERS KEYS_FILE SPLIT_DIR
"$PYTHON_BIN" - <<'PY'
import json
import math
import os

meta_path = os.environ["META_PATH"]
videos = os.environ.get("VIDEOS", "").strip()
workers = max(1, int(os.environ["WORKERS"]))
keys_file = os.environ["KEYS_FILE"]
split_dir = os.environ["SPLIT_DIR"]

if videos:
    keys = [item.strip() for item in videos.split(",") if item.strip()]
else:
    with open(meta_path, "r", encoding="utf-8") as f:
        keys = [json.loads(line)["key"] for line in f if line.strip()]

with open(keys_file, "w", encoding="utf-8") as out:
    for key in keys:
        out.write(key + "\n")

chunk = math.ceil(len(keys) / workers) if keys else 0
for idx in range(workers):
    batch = keys[idx * chunk : (idx + 1) * chunk] if chunk else []
    path = os.path.join(split_dir, f"batch_{idx + 1}.txt")
    with open(path, "w", encoding="utf-8") as out:
        for key in batch:
            out.write(key + "\n")
    print(f"batch_{idx + 1}.txt\t{len(batch)}")
print(f"TOTAL\t{len(keys)}")
PY

TOTAL_VIDEOS="$(wc -l < "$KEYS_FILE" | tr -d ' ')"
if [[ "$TOTAL_VIDEOS" -eq 0 ]]; then
  echo -e "${YELLOW}Error: no videos selected for build${NC}"
  exit 1
fi

echo "=========================================="
echo "Stage1+Stage2 Parallel Graph Build Runner"
echo "=========================================="
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Project Root:      $PROJECT_ROOT"
echo "  Python:            $PYTHON_BIN"
echo "  Assets Root:       $ASSETS_ROOT"
echo "  Meta Path:         $META_PATH"
echo "  Artifacts Root:    $ARTIFACTS_ROOT"
echo "  Output Dir:        $OUTPUT_DIR"
echo "  Workers:           $WORKERS"
echo "  Total Videos:      $TOTAL_VIDEOS"
if [[ -n "$VIDEOS" ]]; then
  echo "  Video Filter:      $VIDEOS"
else
  echo "  Video Filter:      all videos in meta"
fi
echo "  Use VLM:           $USE_VLM"
echo "  Force Rebuild:     $FORCE_REBUILD"
echo ""
echo "Batches:"
for batch_file in "$SPLIT_DIR"/batch_*.txt; do
  echo "  $(basename "$batch_file"): $(wc -l < "$batch_file" | tr -d ' ') videos"
done
echo ""

PIDS=()
WORKER_IDX=0
for batch_file in "$SPLIT_DIR"/batch_*.txt; do
  if [[ ! -s "$batch_file" ]]; then
    continue
  fi
  WORKER_IDX=$((WORKER_IDX + 1))
  VIDEOS_CSV="$(paste -sd, "$batch_file")"
  LOG_FILE="$OUTPUT_DIR/graph_build_worker_${WORKER_IDX}.log"

  CMD=(
    "$PYTHON_BIN" evaluation/evaluate_stage1_graph.py
    --meta "$META_PATH"
    --assets-root "$ASSETS_ROOT"
    --artifacts-root "$ARTIFACTS_ROOT"
    --videos "$VIDEOS_CSV"
    --build-if-missing
    --top-k "$TOP_K"
    --top-k-segments "$TOP_K_SEGMENTS"
    --stage2-window-secs "$STAGE2_WINDOW_SECS"
    --stage2-overlap-secs "$STAGE2_OVERLAP_SECS"
    --output "$OUTPUT_DIR/stage12_build_worker_${WORKER_IDX}.json"
  )

  if [[ "$USE_VLM" -eq 1 ]]; then
    CMD+=(--use-vlm)
  else
    CMD+=(--no-use-vlm)
  fi
  if [[ "$FORCE_REBUILD" -eq 1 ]]; then
    CMD+=(--force-rebuild)
  fi

  echo "[Worker $WORKER_IDX] Starting $(wc -l < "$batch_file" | tr -d ' ') videos -> $LOG_FILE"
  "${CMD[@]}" > "$LOG_FILE" 2>&1 &
  PIDS+=("$!")
done

echo ""
echo "Waiting for ${#PIDS[@]} workers..."
echo "Monitor: tail -f $OUTPUT_DIR/graph_build_worker_*.log"

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=$((FAIL + 1))
  fi
done

if [[ "$FAIL" -gt 0 ]]; then
  echo -e "${YELLOW}[ERROR] $FAIL worker(s) failed. Check logs in $OUTPUT_DIR${NC}"
  exit 1
fi

STAGE1_COUNT="$(find "$ARTIFACTS_ROOT" -type f -name 'scene_graph_stage1.eval.json' | wc -l | tr -d ' ')"
STAGE2_COUNT="$(find "$ARTIFACTS_ROOT" -type f -name 'scene_graph_stage2.eval.json' | wc -l | tr -d ' ')"

echo ""
echo "Build verification:"
echo "  Stage1 graphs: $STAGE1_COUNT"
echo "  Stage2 graphs: $STAGE2_COUNT"

if [[ "$STAGE1_COUNT" != "$TOTAL_VIDEOS" || "$STAGE2_COUNT" != "$TOTAL_VIDEOS" ]]; then
  echo -e "${YELLOW}[ERROR] Graph count mismatch. Expected $TOTAL_VIDEOS videos.${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}All workers finished successfully.${NC}"
echo "Artifacts root: $ARTIFACTS_ROOT"
echo "Reports dir:    $OUTPUT_DIR"
