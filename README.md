# Chainer Agent

Reliable self-improving bots for the chainer.io arena.

This repo now runs a supervised PPO bot swarm with:

- per-agent ONNX policies served by a Python trainer
- match-boundary LLM strategy coaching
- rostered model alias loading (`latest`, `candidate`, `champion`)
- single-instance runtime locking
- live operator telemetry for rooms, trainer health, joins, inputs, and failures

## What Changed

The old NEAT-oriented control loop has been replaced by a production-oriented runtime:

- `SwarmSupervisor` owns lifecycle, health checks, room scheduling, and telemetry
- `RoomCoordinator` owns queueing, seat assignment tracking, joins, retries, and cleanup
- `training/trainer.py` now exports models through a single-threaded ONNX queue into a filesystem model registry
- the dashboard is now an operator console, not just a trainer stat page

## Architecture

```
┌────────────────────────────┐
│  Web Operator Console      │  port 3000
│  /api/system /api/rooms    │
└──────────────┬─────────────┘
               │
┌──────────────▼─────────────┐
│  Swarm Supervisor          │  port 3101
│  single-instance lock      │
│  room telemetry + events   │
│  one RoomCoordinator/room  │
└──────────────┬─────────────┘
               │
┌──────────────▼─────────────┐
│  Arena Backend             │
│  queue / reservation / ws  │
└──────────────┬─────────────┘
               │
┌──────────────▼─────────────┐
│  PPO Trainer               │  port 5555
│  per-agent registry        │
│  aliases: latest/candidate │
│  champion scaffolding      │
└────────────────────────────┘
```

## Runtime Model

Each bot has two layers:

- Reflex layer: PPO policy exported to ONNX and executed locally with `onnxruntime-node`
- Strategy layer: LLM runs every 3 matches, returns strict JSON, and updates bounded strategy parameters

The LLM is not in the hot path. Live action decisions stay deterministic and low-latency.

## Key Reliability Fixes

- Single-instance lock at `/tmp/chainer-agent.lock`
- Trainer readiness gate before the swarm starts matches
- Per-bot seat assignment tracking instead of assuming one queue result applies to the whole room
- Join failure classification for reserved-seat expiry and locked rooms
- Room-local schema failure containment so bad state patches do not kill the whole swarm
- Telemetry server with `/healthz`, `/readyz`, `/metrics`, `/system`, `/rooms`, `/events`

## Model Registry

Models are stored under:

```text
models/<agent_id>/versions/v000123/
  policy.onnx
  checkpoint.pt
  metadata.json
  eval.json
models/<agent_id>/aliases.json
```

Alias support:

- `latest`: newest exported policy
- `candidate`: newest policy considered for promotion
- `champion`: stable slot for curated/tournament play

Bots can be rostered against aliases instead of raw versions.

## State + Reward

The policy input is now objective-aware. In addition to combat and scoreboard context, the state extractor includes:

- nearest crystal direction and distance
- local crowding
- ability readiness ratio
- recent combat context

Rewards are shaped from:

- score delta
- kills / deaths
- damage dealt / damage taken
- survival
- ability value
- anti-suicide penalty

Episode reward totals are persisted for debugging.

## Project Structure

```text
src/
  bot/
    AgentBrain.js
    SmartBot.js
    StrategicBrain.js
    StateExtractor.js
  runtime/
    SwarmSupervisor.js
    RoomCoordinator.js
    RuntimeState.js
    SingleInstanceLock.js
    Roster.js
  game/
  network/
training/
  trainer.py
  requirements.txt
web/
  server.js
  public/index.html
config/
  default.json
  roster.json
deploy/
  chainer-trainer.service
  chainer-bots.service
  chainer-dashboard.service
```

## Setup

### Prerequisites

- Node.js `20.20.0`
- Python `3.12.3`

Optional local version hints are checked in:

- `.nvmrc`
- `.python-version`

### Install

```bash
npm install

python3 -m venv venv
source venv/bin/activate
pip install -r training/requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Important variables:

```env
GAME_SERVER_URL=https://ai-test-arena.chainers.io
TRAINER_URL=http://localhost:5555
SUPERVISOR_PORT=3101
DASHBOARD_PORT=3000

POPULATION_SIZE=12
NUM_ROOMS=2
MATCH_TIMEOUT=120000
SELECTION_INTERVAL=10
NUM_CULL=5

BOT_ROSTER_PATH=config/roster.json
NODE_TLS_REJECT_UNAUTHORIZED=0
```

LLM coaching:

```env
OLLAMA_CLOUD_API_KEY=...
DEEP_ANALYSIS_MODEL=kimi-k2.5:cloud
OAUTH_API_KEY=...
```

## Roster Configuration

`config/roster.json` defines which agents load into which rooms and which alias each one should use.

Example:

```json
{
  "rooms": [
    [
      { "agentId": "agent_0", "modelAlias": "champion" },
      { "agentId": "agent_1", "modelAlias": "latest" }
    ]
  ]
}
```

This is the hook for future matches, showmatches, and championships.

## Running

### Local

Terminal 1:

```bash
source venv/bin/activate
python training/trainer.py
```

Terminal 2:

```bash
node src/index.js
```

Terminal 3:

```bash
node web/server.js
```

### Production

Install the systemd services on `spark.local`:

```bash
bash deploy/setup.sh
```

Services:

- `chainer-trainer`
- `chainer-bots`
- `chainer-dashboard`

`deploy/run-forever.sh` is kept only as a deprecated local helper and should not be the production control plane.

## Operator APIs

### Supervisor

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /system`
- `GET /rooms`
- `GET /events`

### Trainer

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /stats`
- `POST /experience`
- `POST /episode`
- `GET /model/:agent_id?alias=latest`
- `GET /model/:agent_id/version?alias=latest`
- `GET /model/:agent_id/metadata?alias=latest`
- `GET /model/:agent_id/aliases`
- `POST /model/:agent_id/alias/:alias`
- `POST /select`
- `POST /agent/:id/reset`
- `POST /agent/:target/clone/:source`
- `GET /agent/:id/history`
- `POST /agent/:id/strategy`

### Dashboard

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /api/system`
- `GET /api/rooms`
- `GET /api/events`
- `GET /api/stats`
- `GET /api/profile/:id`

## Testing

Run the unit suite:

```bash
npm test
```

Run the smoke check against live/local services:

```bash
npm run test:smoke
```

Current automated coverage includes:

- single-instance lock behavior
- room assignment grouping
- join error classification
- roster normalization
- shaped reward recording
- structured LLM strategy parsing

## Notes

- `--watch` and NEAT-era resume flows are deprecated in the supervised PPO runtime.
- Existing old checkpoints with incompatible state dimensions are skipped on load.
- The champion/evaluation ladder is scaffolded through model aliases and metadata, ready for stricter promotion workflows.
