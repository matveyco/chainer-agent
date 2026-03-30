#!/bin/bash
# Run chainer-agent bot swarm with auto-restart
# Usage: nohup bash deploy/run-forever.sh > bots.log 2>&1 &

cd "$(dirname "$0")/.."

while true; do
    echo "[$(date)] Starting bot swarm..."
    NO_DASHBOARD=1 node src/index.js
    EXIT_CODE=$?
    echo "[$(date)] Bot swarm exited with code $EXIT_CODE. Restarting in 5s..."
    sleep 5
done
