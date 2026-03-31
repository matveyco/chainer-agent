#!/bin/bash
# Deprecated local helper.
# Production should use the dedicated systemd services installed by deploy/setup.sh:
#   chainer-trainer
#   chainer-bots
#   chainer-dashboard

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[$(date)] run-forever.sh is deprecated for production."
echo "[$(date)] Use deploy/setup.sh and the systemd service units instead."
echo "[$(date)] Starting trainer, supervisor, and dashboard once for local/dev only..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

nohup python3 training/trainer.py >> trainer.log 2>&1 &
sleep 2
nohup node src/index.js >> bots.log 2>&1 &
sleep 2
nohup node web/server.js >> dashboard.log 2>&1 &

echo "[$(date)] Started local helper processes."
