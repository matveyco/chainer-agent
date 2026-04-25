#!/usr/bin/env bash
# Truncate large append-only logs in place. Used by the systemd timer
# `chainer-log-maintenance.timer` so we don't fill the disk during long
# unattended runs. systemctl-managed services keep their open file handle
# (no reopen needed).
#
# Strategy: when a log exceeds MAX_BYTES, rotate it once (mv to .1) then
# truncate the original. Keeps one generation of history for forensics.
# A second run will overwrite the .1 file. Cheap and safe.

set -euo pipefail

CHAINER_HOME=${CHAINER_HOME:-/home/m/chainer-agent}
MAX_BYTES=${MAX_BYTES:-104857600}   # 100 MB per log
TRAINING_LOGS_MAX_BYTES=${TRAINING_LOGS_MAX_BYTES:-524288000}  # 500 MB

LOGS=(
  "$CHAINER_HOME/trainer.log"
  "$CHAINER_HOME/bots.log"
  "$CHAINER_HOME/dashboard.log"
  "$CHAINER_HOME/doctor.log"
  "$CHAINER_HOME/daily_report.log"
)

trim() {
  local path="$1"
  local cap="$2"
  [ -f "$path" ] || return 0
  local size
  size=$(stat -c %s "$path")
  if [ "$size" -gt "$cap" ]; then
    cp -f "$path" "$path.1" 2>/dev/null || true
    : > "$path"
    echo "truncated $path (was $size bytes)"
  fi
}

for log in "${LOGS[@]}"; do
  trim "$log" "$MAX_BYTES"
done

# Per-agent training jsonl files: cap each at 50 MB (rolling truncate).
TRAINING_DIR="$CHAINER_HOME/training_logs"
if [ -d "$TRAINING_DIR" ]; then
  TOTAL=$(du -sb "$TRAINING_DIR" 2>/dev/null | awk '{print $1}')
  if [ -n "$TOTAL" ] && [ "$TOTAL" -gt "$TRAINING_LOGS_MAX_BYTES" ]; then
    echo "training_logs total ${TOTAL} bytes > ${TRAINING_LOGS_MAX_BYTES}; trimming individual files"
    for f in "$TRAINING_DIR"/agent_*.jsonl; do
      [ -f "$f" ] || continue
      trim "$f" 52428800
    done
  fi
fi

echo "log maintenance done at $(date -Iseconds)"
