#!/bin/bash
# Deploy chainer-agent as 24/7 systemd services on spark.local
# Run this on the target machine: bash deploy/setup.sh

set -e

echo "=== Chainer Agent — Deploy 24/7 Services ==="

# Copy service files
sudo cp deploy/chainer-trainer.service /etc/systemd/system/
sudo cp deploy/chainer-bots.service /etc/systemd/system/
sudo cp deploy/chainer-dashboard.service /etc/systemd/system/
sudo cp deploy/chainer-doctor.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable chainer-doctor
sudo systemctl enable chainer-trainer
sudo systemctl enable chainer-bots
sudo systemctl enable chainer-dashboard

# Start services
sudo systemctl start chainer-doctor
sleep 1
sudo systemctl start chainer-trainer
sleep 3
sudo systemctl start chainer-bots
sudo systemctl start chainer-dashboard

echo ""
echo "Services started! Check status:"
echo "  sudo systemctl status chainer-doctor"
echo "  sudo systemctl status chainer-trainer"
echo "  sudo systemctl status chainer-bots"
echo "  sudo systemctl status chainer-dashboard"
echo ""
echo "View logs:"
echo "  tail -f ~/chainer-agent/trainer.log"
echo "  tail -f ~/chainer-agent/bots.log"
echo "  tail -f ~/chainer-agent/dashboard.log"
echo ""
echo "Training stats:"
echo "  curl http://localhost:5555/stats"
