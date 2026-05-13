#!/bin/bash
# Resume benchmark with a remaining meta file under split-root layout.

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

DATA_ROOT="${SGD_DATA_ROOT:-$PROJECT_ROOT}"
ASSETS_ROOT="${SGD_LVBENCH_ROOT:-$DATA_ROOT/LVBench/LVBench_4.1}"
CAPTION_ROOT="${SGD_CAPTION_ROOT:-$ASSETS_ROOT}"
LOGS_ROOT="${SGD_LOGS_ROOT:-$DATA_ROOT/logs}"
REMAINING_META="${ASSETS_ROOT}/video_info.meta.remaining.jsonl"
RUN_NAME="resume"
MAX_WORKERS=1
KILL_PID=""

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --remaining-meta FILE   Remaining meta jsonl path (default: <assets-root>/video_info.meta.remaining.jsonl)
  --assets-root DIR       Assets root (<video_id>/frames + database.json)
  --caption-root DIR      Caption/artifacts root (<video_id>/captions)
  --logs-root DIR         Logs root (default: \$SGD_LOGS_ROOT or <data-root>/logs)
  --name NAME             Run name prefix (default: $RUN_NAME)
  --max-workers N         Parallel workers (default: $MAX_WORKERS)
  --kill-pid PID          Optional PID to terminate before resume
  --help, -h              Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remaining-meta) REMAINING_META="$2"; shift 2 ;;
    --assets-root) ASSETS_ROOT="$2"; shift 2 ;;
    --caption-root) CAPTION_ROOT="$2"; shift 2 ;;
    --logs-root) LOGS_ROOT="$2"; shift 2 ;;
    --name) RUN_NAME="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --kill-pid) KILL_PID="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$KILL_PID" ]]; then
  if ps -p "$KILL_PID" >/dev/null 2>&1; then
    echo -e "${YELLOW}Stopping existing process: $KILL_PID${NC}"
    kill "$KILL_PID" || true
    sleep 2
    if ps -p "$KILL_PID" >/dev/null 2>&1; then
      kill -9 "$KILL_PID" || true
    fi
  else
    echo -e "${YELLOW}PID not found, skip kill: $KILL_PID${NC}"
  fi
fi

if [[ ! -f "$REMAINING_META" ]]; then
  echo -e "${YELLOW}Error: remaining meta file not found: $REMAINING_META${NC}"
  exit 1
fi

VIDEO_COUNT=$(grep -cve '^\s*$' "$REMAINING_META" || true)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOGS_ROOT}/${RUN_NAME}_${TIMESTAMP}"
PER_VIDEO_LOG_DIR="${RUN_LOG_DIR}/per_video"
TRACE_DIR="${RUN_LOG_DIR}/traces"
RUN_LOG_FILE="${RUN_LOG_DIR}/run_benchmark_all.log"
RUN_SUMMARY_JSON="${RUN_LOG_DIR}/run_summary.json"

mkdir -p "$PER_VIDEO_LOG_DIR" "$TRACE_DIR"

echo "=========================================="
echo "DVD Resume Benchmark Runner"
echo "=========================================="
echo -e "${BLUE}Configuration:${NC}"
echo "  Project Root:    $PROJECT_ROOT"
echo "  Assets Root:     $ASSETS_ROOT"
echo "  Caption Root:    $CAPTION_ROOT"
echo "  Remaining Meta:  $REMAINING_META"
echo "  Videos Pending:  $VIDEO_COUNT"
echo "  Max Workers:     $MAX_WORKERS"
echo "  Run Log Dir:     $RUN_LOG_DIR"
echo ""

cat > "${RUN_LOG_DIR}/run_config.txt" <<EOF
Run Configuration
=================
timestamp: $TIMESTAMP
run_name: $RUN_NAME
assets_root: $ASSETS_ROOT
caption_root: $CAPTION_ROOT
remaining_meta: $REMAINING_META
videos_pending: $VIDEO_COUNT
max_workers: $MAX_WORKERS
trace_dir: $TRACE_DIR
EOF

export SGD_TRACE_ENABLED=1
export SGD_TRACE_LOG_DIR="$TRACE_DIR"

CMD=(
  "$PYTHON_BIN"
  -m
  reproduce.run_benchmark
  "$ASSETS_ROOT"
  "$REMAINING_META"
  --per-video-log-dir "$PER_VIDEO_LOG_DIR"
  --trace-log-dir "$TRACE_DIR"
  --caption-root "$CAPTION_ROOT"
  --max-workers "$MAX_WORKERS"
)

echo "Running:"
printf '  %q' "${CMD[@]}"
echo
echo ""

"${CMD[@]}" 2>&1 | tee "$RUN_LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

if [[ -d "$TRACE_DIR" ]] && find "$TRACE_DIR" -name "*.jsonl" | grep -q .; then
  "$PYTHON_BIN" scripts/build_run_summary.py \
    --run-dir "$RUN_LOG_DIR" \
    --trace-dir "$TRACE_DIR" \
    --benchmark-results "$RUN_LOG_DIR/benchmark_results.json" \
    --run-log "$RUN_LOG_FILE" \
    --output "$RUN_SUMMARY_JSON"
fi

echo ""
if [[ "$EXIT_CODE" -eq 0 ]]; then
  echo -e "${GREEN}Resume benchmark completed.${NC}"
else
  echo -e "${YELLOW}Resume benchmark finished with exit code: $EXIT_CODE${NC}"
fi
echo "  Per-video logs: $PER_VIDEO_LOG_DIR"
echo "  Traces:         $TRACE_DIR"
echo "  Run summary:    $RUN_SUMMARY_JSON"

exit "$EXIT_CODE"
