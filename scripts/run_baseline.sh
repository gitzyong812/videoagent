#!/bin/bash
# Run DVD Baseline Benchmark
# 统一的baseline运行脚本，自动创建带时间戳的日志目录

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "DVD Baseline Benchmark Runner"
echo "=========================================="
echo ""

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${PYTHON_BIN:-}" || "$PYTHON_BIN" == "python3" ]]; then
    if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
    fi
fi

# 配置参数
DATA_ROOT="${SGD_DATA_ROOT:-$PROJECT_ROOT}"
BENCHMARK_DATA_DIR="${SGD_LVBENCH_ROOT:-$DATA_ROOT/LVBench/LVBench_4.1}"
CAPTION_ROOT="${SGD_CAPTION_ROOT:-$BENCHMARK_DATA_DIR}"
LOGS_BASE_DIR="${SGD_LOGS_ROOT:-$DATA_ROOT/logs}"
EVAL_OUTPUT_DIR="${SGD_EVAL_OUTPUT_ROOT:-$DATA_ROOT/evaluation}"
META_FILE_OVERRIDE=""
MAX_WORKERS=1

# 解析命令行参数
SUBSET_FILE=""
RUN_NAME="baseline"

while [[ $# -gt 0 ]]; do
    case $1 in
        --subset)
            SUBSET_FILE="$2"
            RUN_NAME="subset"
            shift 2
            ;;
        --name)
            RUN_NAME="$2"
            shift 2
            ;;
        --assets-root)
            BENCHMARK_DATA_DIR="$2"
            shift 2
            ;;
        --caption-root)
            CAPTION_ROOT="$2"
            shift 2
            ;;
        --meta)
            META_FILE_OVERRIDE="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --subset FILE    Run on a subset file (one video key per line)"
            echo "  --name NAME      Custom name for this run (default: baseline or subset)"
            echo "  --assets-root DIR  Benchmark assets root (<video_id>/frames + database.json)"
            echo "  --caption-root DIR Caption/artifact root (<video_id>/captions)"
            echo "  --meta FILE        Meta jsonl path (default: <assets-root>/video_info.meta.jsonl)"
            echo "  --max-workers N    Parallel workers per video question run (default: $MAX_WORKERS)"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run full benchmark"
            echo "  bash scripts/run_baseline.sh"
            echo ""
            echo "  # Run on test subset"
            echo "  bash scripts/run_baseline.sh --subset /path/to/subset.txt"
            echo ""
            echo "  # Run with custom name"
            echo "  bash scripts/run_baseline.sh --name improved_memory --subset /path/to/subset.txt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 创建带时间戳的日志目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$LOGS_BASE_DIR/${RUN_NAME}_${TIMESTAMP}"
PER_VIDEO_LOG_DIR="$LOG_DIR/per_video"
TRACE_DIR="$LOG_DIR/traces"
mkdir -p "$PER_VIDEO_LOG_DIR"
mkdir -p "$TRACE_DIR"
mkdir -p "$EVAL_OUTPUT_DIR"

# 显示配置信息
echo -e "${BLUE}Configuration:${NC}"
echo "  Project Root:    $PROJECT_ROOT"
echo "  Data Root:       $DATA_ROOT"
echo "  Benchmark Data:  $BENCHMARK_DATA_DIR"
echo "  Caption Root:    $CAPTION_ROOT"
echo "  Log Directory:   $LOG_DIR"
echo "  Trace Directory: $TRACE_DIR"
echo "  Eval Output Dir: $EVAL_OUTPUT_DIR"
echo "  Run Name:        $RUN_NAME"
if [ -n "$SUBSET_FILE" ]; then
    echo "  Subset File:     $SUBSET_FILE"
    VIDEO_COUNT=$(wc -l < "$SUBSET_FILE" | tr -d ' ')
    echo "  Video Count:     $VIDEO_COUNT videos"
else
    echo "  Mode:            Full benchmark (103 videos)"
fi
echo ""

# 检查数据目录
if [ ! -d "$BENCHMARK_DATA_DIR" ]; then
    echo -e "${YELLOW}Warning: Benchmark data directory not found: $BENCHMARK_DATA_DIR${NC}"
    echo "Please update BENCHMARK_DATA_DIR in this script or mount the external drive."
    exit 1
fi

# 检查meta文件
META_FILE="$BENCHMARK_DATA_DIR/video_info.meta.jsonl"
if [ -n "$META_FILE_OVERRIDE" ]; then
    META_FILE="$META_FILE_OVERRIDE"
fi
if [ ! -f "$META_FILE" ]; then
    echo -e "${YELLOW}Error: Meta file not found: $META_FILE${NC}"
    exit 1
fi

# 保存运行配置到日志目录
cat > "$LOG_DIR/run_config.txt" <<EOF
Run Configuration
=================
Timestamp: $TIMESTAMP
Run Name: $RUN_NAME
Benchmark Data: $BENCHMARK_DATA_DIR
Caption Root: $CAPTION_ROOT
Data Root: $DATA_ROOT
Meta File: $META_FILE
Max Workers: $MAX_WORKERS
Subset File: ${SUBSET_FILE:-"None (full benchmark)"}
Log Directory: $LOG_DIR
Eval Output Dir: $EVAL_OUTPUT_DIR
Command: $0 $@
EOF

echo "=========================================="
echo -e "${GREEN}Starting DVD Benchmark...${NC}"
echo "=========================================="
echo ""

# 如果指定了subset文件，创建临时meta文件
ACTUAL_META_FILE="$META_FILE"
if [ -n "$SUBSET_FILE" ]; then
    if [ ! -f "$SUBSET_FILE" ]; then
        echo -e "${YELLOW}Error: Subset file not found: $SUBSET_FILE${NC}"
        exit 1
    fi

    # 创建临时meta文件，只包含subset中的视频
    TEMP_META_FILE="$LOG_DIR/subset_meta.jsonl"
    echo -e "${BLUE}Creating subset meta file...${NC}"

    # 读取subset视频列表并从原meta文件中提取对应行
    while IFS= read -r video_key; do
        # 跳过空行和注释
        [[ -z "$video_key" || "$video_key" =~ ^# ]] && continue
        # 从meta文件中提取该视频的信息（注意key后面有空格）
        grep "\"key\": \"$video_key\"" "$META_FILE" >> "$TEMP_META_FILE" || true
    done < "$SUBSET_FILE"

    ACTUAL_META_FILE="$TEMP_META_FILE"
    echo -e "${GREEN}✓ Subset meta file created: $TEMP_META_FILE${NC}"
    SUBSET_VIDEO_COUNT=$(wc -l < "$TEMP_META_FILE" | tr -d ' ')
    echo "  Videos in subset meta: $SUBSET_VIDEO_COUNT"
    echo ""
fi

# 构建并运行命令
CMD="\"$PYTHON_BIN\" -m reproduce.run_benchmark \"$BENCHMARK_DATA_DIR\" \"$ACTUAL_META_FILE\" --per-video-log-dir \"$PER_VIDEO_LOG_DIR\" --trace-log-dir \"$TRACE_DIR\" --caption-root \"$CAPTION_ROOT\" --max-workers \"$MAX_WORKERS\""

# 运行benchmark并保存日志
export SGD_TRACE_ENABLED=1
export SGD_TRACE_LOG_DIR="$TRACE_DIR"
eval $CMD 2>&1 | tee "$LOG_DIR/run_benchmark_all.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Benchmark completed successfully!${NC}"
else
    echo -e "${YELLOW}✗ Benchmark failed with exit code: $EXIT_CODE${NC}"
fi
echo "=========================================="
echo ""

if [ -d "$TRACE_DIR" ] && find "$TRACE_DIR" -name "*.jsonl" | grep -q .; then
    echo -e "${BLUE}Building run summary JSON...${NC}"
    "$PYTHON_BIN" scripts/build_run_summary.py \
        --run-dir "$LOG_DIR" \
        --trace-dir "$TRACE_DIR" \
        --benchmark-results "$LOG_DIR/benchmark_results.json" \
        --run-log "$LOG_DIR/run_benchmark_all.log" \
        --output "$LOG_DIR/run_summary.json"
else
    echo -e "${YELLOW}Skip run summary: no trace jsonl files under $TRACE_DIR${NC}"
fi

echo "Results saved to:"
echo "  Log directory:    $LOG_DIR"
echo "  Main log:         $LOG_DIR/run_benchmark_all.log"
echo "  Per-video logs:   $PER_VIDEO_LOG_DIR"
echo "  Traces:           $TRACE_DIR"
echo "  Run summary:      $LOG_DIR/run_summary.json"
echo "  Config:           $LOG_DIR/run_config.txt"
echo ""
echo "Next steps:"
echo "  # Evaluate results"
echo "  \"$PYTHON_BIN\" evaluation/evaluate_results.py \\"
echo "    \"$PER_VIDEO_LOG_DIR\" \\"
echo "    --benchmark-results \"$LOG_DIR/benchmark_results.json\" \\"
echo "    --trace-dir \"$TRACE_DIR\" \\"
echo "    --meta \"$ACTUAL_META_FILE\" \\"
echo "    --output \"$EVAL_OUTPUT_DIR/results_${RUN_NAME}_${TIMESTAMP}.json\""
echo ""

exit $EXIT_CODE
