#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$ROOT_DIR/SGDAgent_MM2026_code"
OUT_DIR="${1:-$ROOT_DIR/dist}"
STAMP="$(date +%Y%m%d)"
ARCHIVE_PATH="$OUT_DIR/SGDAgent_submission_${STAMP}.tar.gz"

mkdir -p "$OUT_DIR"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Missing supplementary source directory: $SOURCE_DIR" >&2
  exit 1
fi

tar \
  --exclude='.git' \
  --exclude='dist' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='.claude' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.codex_preview' \
  --exclude='video_database' \
  --exclude='data' \
  --exclude='.DS_Store' \
  --exclude='*.pyc' \
  --exclude='*.aux' \
  --exclude='*.bbl' \
  --exclude='*.blg' \
  --exclude='*.fdb_latexmk' \
  --exclude='*.fls' \
  --exclude='*.log' \
  --exclude='*.out' \
  --exclude='*.synctex.gz' \
  --exclude='.env' \
  --exclude='.env.*' \
  -czf "$ARCHIVE_PATH" \
  -C "$SOURCE_DIR" \
  .

echo "Created sanitized archive: $ARCHIVE_PATH"
