#!/bin/bash
# Deploy chainer-agent as 24/7 systemd services on spark.local
# Run this on the target machine: bash deploy/setup.sh

set -e

echo "=== Chainer Agent — Deploy 24/7 Services ==="

# Copy service files
sudo cp deploy/chainer-trainer.service /etc/systemd/system/
sudo cp deploy/chainer-bots.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable chainer-trainer
sudo systemctl enable chainer-bots

# Start services
sudo systemctl start chainer-trainer
sleep 3
sudo systemctl start chainer-bots

echo ""
echo "Services started! Check status:"
echo "  sudo systemctl status chainer-trainer"
echo "  sudo systemctl status chainer-bots"
echo ""
echo "View logs:"
echo "  tail -f ~/chainer-agent/trainer.log"
echo "  tail -f ~/chainer-agent/bots.log"
echo ""
echo "Training stats:"
echo "  curl http://localhost:5555/stats"
