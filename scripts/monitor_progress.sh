#!/bin/bash
# Monitor DVD Benchmark Progress
# 监控最新的benchmark运行进度

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${SGD_DATA_ROOT:-$PROJECT_ROOT}"
LOGS_BASE_DIR="${SGD_LOGS_ROOT:-$DATA_ROOT/logs}"
EVAL_OUTPUT_DIR="${SGD_EVAL_OUTPUT_ROOT:-$DATA_ROOT/evaluation}"

# 获取最新的日志目录
LATEST_LOG=$(ls -t "$LOGS_BASE_DIR" 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log directories found in $LOGS_BASE_DIR"
    exit 1
fi

LOG_DIR="$LOGS_BASE_DIR/$LATEST_LOG"

echo "=========================================="
echo -e "${BLUE}DVD Benchmark Progress Monitor${NC}"
echo "=========================================="
echo "Log Directory: $LOG_DIR"
echo ""

# 读取配置文件
if [ -f "$LOG_DIR/run_config.txt" ]; then
    echo -e "${BLUE}Run Configuration:${NC}"
    grep "Run Name:" "$LOG_DIR/run_config.txt"
    grep "Timestamp:" "$LOG_DIR/run_config.txt"
    echo ""
fi

# 推断总视频数：subset_meta > run_config 的 Meta File > 默认 103
TOTAL_VIDEOS=103
MODE_TEXT="Full Benchmark ($TOTAL_VIDEOS videos)"
META_PATH=""
if [ -f "$LOG_DIR/subset_meta.jsonl" ]; then
    META_PATH="$LOG_DIR/subset_meta.jsonl"
    TOTAL_VIDEOS=$(wc -l < "$LOG_DIR/subset_meta.jsonl" | tr -d ' ')
    MODE_TEXT="Test Subset ($TOTAL_VIDEOS videos)"
elif [ -f "$LOG_DIR/run_config.txt" ]; then
    META_PATH=$(grep -E "^Meta File:" "$LOG_DIR/run_config.txt" | sed 's/^Meta File:[[:space:]]*//')
    if [ -n "$META_PATH" ] && [ -f "$META_PATH" ]; then
        COUNT_FROM_META=$(wc -l < "$META_PATH" | tr -d ' ')
        if [ "$COUNT_FROM_META" -gt 0 ]; then
            TOTAL_VIDEOS="$COUNT_FROM_META"
            MODE_TEXT="Meta Driven ($TOTAL_VIDEOS videos)"
        fi
    fi
fi
echo -e "${BLUE}Mode:${NC} $MODE_TEXT"

# 统计完成的视频数
COMPLETED_VIDEOS=$(ls -1 "$LOG_DIR/per_video/" 2>/dev/null | wc -l | tr -d ' ')
PROGRESS=$((COMPLETED_VIDEOS * 100 / TOTAL_VIDEOS))

echo ""
echo "=========================================="
echo -e "${GREEN}Progress: $COMPLETED_VIDEOS/$TOTAL_VIDEOS videos ($PROGRESS%)${NC}"
echo "=========================================="

# 显示进度条
BAR_LENGTH=50
FILLED=$((PROGRESS * BAR_LENGTH / 100))
printf "["
for ((i=0; i<BAR_LENGTH; i++)); do
    if [ $i -lt $FILLED ]; then
        printf "="
    else
        printf " "
    fi
done
printf "] $PROGRESS%%\n"

echo ""

# 列出已完成的视频
if [ $COMPLETED_VIDEOS -gt 0 ]; then
    echo -e "${BLUE}Completed videos:${NC}"
    ls -1 "$LOG_DIR/per_video/" | sed 's/.log$//' | head -10
    if [ $COMPLETED_VIDEOS -gt 10 ]; then
        echo "... and $((COMPLETED_VIDEOS - 10)) more"
    fi
    echo ""
fi

# 显示最新日志（最后10行）
echo -e "${BLUE}Latest log output:${NC}"
echo "----------------------------------------"
tail -10 "$LOG_DIR/run_benchmark_all.log" | grep -v "WARNING:azure.identity"
echo "----------------------------------------"
echo ""

# 检查进程是否还在运行
if pgrep -f "reproduce.run_benchmark" > /dev/null; then
    echo -e "${GREEN}✓ Benchmark is running${NC}"
    echo ""
    echo "Commands:"
    echo "  # Watch live progress"
    echo "  watch -n 5 bash scripts/monitor_progress.sh"
    echo ""
    echo "  # Follow live log"
    echo "  tail -f $LOG_DIR/run_benchmark_all.log"
else
    echo -e "${YELLOW}✗ Benchmark process not found${NC}"
    echo ""
    if [ $COMPLETED_VIDEOS -eq $TOTAL_VIDEOS ]; then
        echo -e "${GREEN}✓ All videos completed!${NC}"
        echo ""
        echo "Next step:"
        echo "  python evaluation/evaluate_results.py \\"
        echo "    \"$LOG_DIR/per_video\" \\"
        echo "    --benchmark-results \"$LOG_DIR/benchmark_results.json\" \\"
        echo "    --trace-dir \"$LOG_DIR/traces\" \\"
        if [ -n "$META_PATH" ] && [ -f "$META_PATH" ]; then
            echo "    --meta \"$META_PATH\" \\"
        fi
        echo "    --output \"$EVAL_OUTPUT_DIR/results_${LATEST_LOG}.json\""
    else
        echo "Benchmark may have stopped or crashed."
        echo "Check the log file: $LOG_DIR/run_benchmark_all.log"
    fi
fi

echo ""
