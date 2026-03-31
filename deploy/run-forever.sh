#!/bin/bash
# Chainer Agent — 24/7 runner for all services
# Manages: Python trainer, Node.js bots, Web dashboard
# Usage: nohup bash deploy/run-forever.sh > runner.log 2>&1 &

cd "$(dirname "$0")/.."
BASEDIR=$(pwd)

echo "[$(date)] === Chainer Agent 24/7 Runner ==="
echo "[$(date)] Base dir: $BASEDIR"

# Ensure venv exists
if [ ! -d "venv" ]; then
    echo "[$(date)] Creating Python venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install torch numpy onnx onnxscript flask
else
    source venv/bin/activate
fi

# Function: start trainer if not running
start_trainer() {
    if ! pgrep -f "python3 training/trainer.py" > /dev/null 2>&1; then
        echo "[$(date)] Starting trainer..."
        cd "$BASEDIR"
        nohup python3 training/trainer.py >> trainer.log 2>&1 &
        sleep 3
        echo "[$(date)] Trainer PID: $(pgrep -f 'python3 training/trainer.py')"
    fi
}

# Function: start dashboard if not running
start_dashboard() {
    if ! pgrep -f "node web/server.js" > /dev/null 2>&1; then
        echo "[$(date)] Starting dashboard..."
        cd "$BASEDIR"
        nohup node web/server.js >> dashboard.log 2>&1 &
        sleep 1
        echo "[$(date)] Dashboard PID: $(pgrep -f 'node web/server.js')"
    fi
}

# Function: check trainer health
check_trainer() {
    curl -s --max-time 5 http://localhost:5555/health > /dev/null 2>&1
    return $?
}

# Start trainer and dashboard first
start_trainer
start_dashboard

# Main loop: run bots, restart on crash
while true; do
    # Ensure trainer is alive
    if ! check_trainer; then
        echo "[$(date)] Trainer down! Restarting..."
        pgrep -f "python3 training/trainer.py" | xargs kill 2>/dev/null || true
        sleep 2
        start_trainer
        sleep 3
    fi

    # Ensure dashboard is alive
    start_dashboard

    echo "[$(date)] Starting bot swarm..."
    cd "$BASEDIR"
    NO_DASHBOARD=1 node src/index.js 2>> bots_errors.log
    EXIT_CODE=$?
    echo "[$(date)] Bots exited (code $EXIT_CODE). Restarting in 5s..."
    sleep 5
done
