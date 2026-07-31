#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$ROOT_DIR/dist}"
STAMP="$(date +%Y%m%d)"
ARCHIVE_PATH="$OUT_DIR/SGDAgent_submission_${STAMP}.tar.gz"

mkdir -p "$OUT_DIR"

COPYFILE_DISABLE=1 tar \
  --format=ustar \
  --no-xattrs \
  --no-mac-metadata \
  --uid 0 \
  --gid 0 \
  --uname root \
  --gname root \
  --exclude='.git' \
  --exclude='dist' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='.claude' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.coverage' \
  --exclude='htmlcov' \
  --exclude='build' \
  --exclude='*.egg-info' \
  --exclude='outputs' \
  --exclude='runs' \
  --exclude='logs' \
  --exclude='reports' \
  --exclude='ckpt' \
  --exclude='video_database' \
  --exclude='data' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='*.pyc' \
  --exclude='*.log' \
  --exclude='*.jsonl' \
  --exclude='*.tmp' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  --exclude='*.tgz' \
  --exclude='*.zip' \
  --exclude='*.7z' \
  --exclude='*.rar' \
  -czf "$ARCHIVE_PATH" \
  -C "$ROOT_DIR" \
  .

echo "Created sanitized archive: $ARCHIVE_PATH"
