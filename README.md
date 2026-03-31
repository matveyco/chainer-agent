# Chainer Agent

Self-learning AI bots for the [chainer.io](https://chainer.io) 3rd-person multiplayer shooter arena. Each agent has a unique personality, its own neural network, and an LLM strategic brain — they fight, learn, and evolve.

## Architecture: Two-Layer Brain

Each agent has **two brain layers** working together:

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT BRAIN                               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LAYER 2: LLM Strategic Brain (kimi-k2.5)            │   │
│  │  Runs: every 3 matches (between matches)             │   │
│  │  Does: analyzes performance, adjusts strategy,        │   │
│  │        formulates plans, reflects on mistakes         │   │
│  │  Output: strategy parameters that OVERRIDE Layer 1    │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                           │ strategy overrides               │
│  ┌────────────────────────▼─────────────────────────────┐   │
│  │  LAYER 1: Neural Network (PPO on GPU)                 │   │
│  │  Runs: 60Hz (every frame)                             │   │
│  │  Does: raw movement, aiming, shooting decisions       │   │
│  │  Trained: continuously via PyTorch on GPU             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Layer 1 (NN/PPO)** handles reflexes — frame-by-frame movement, aim, shoot. Trained on GPU via PyTorch, exported as ONNX models, inference in Node.js.

**Layer 2 (LLM)** handles strategy — it OVERRIDES the NN when needed. A Hunter will force-charge enemies even if the NN says retreat. A Collector will ignore fights to gather crystals. Strategy parameters directly control behavior:

| Parameter | Low (0.0) | High (1.0) |
|-----------|-----------|------------|
| `aggression` | Run away from enemies | Always charge and shoot |
| `accuracy_focus` | Spray bullets randomly | Only perfect aimed shots |
| `crystal_priority` | Ignore crystals | Only collect crystals, avoid fights |
| `ability_usage` | Never use abilities | Spam abilities constantly |
| `retreat_threshold` | Fight to death (Berserker) | Retreat at 70% health (Survivor) |

## Agent Personalities

Each agent gets a deterministic personality archetype that creates **drastically different playstyles**:

| Agent | Archetype | Behavior |
|-------|-----------|----------|
| agent_0, 8, 16 | **Hunter** | Charges enemies, close combat, never retreats |
| agent_1, 9, 17 | **Sniper** | Keeps distance, precise shots only |
| agent_2, 10, 18 | **Collector** | Avoids fights, hoards crystals for score |
| agent_3, 11, 19 | **Survivor** | Retreats early, outlasts everyone |
| agent_4, 12, 20 | **Berserker** | All-in maniac, zero self-preservation |
| agent_5, 13, 21 | **Tactician** | Balanced, adapts to situation |
| agent_6, 14, 22 | **Flanker** | Hit-and-run, attacks from edges |
| agent_7, 15, 23 | **Guardian** | Holds center, punishes intruders |

The LLM reviews performance every 3 matches and adjusts strategy — a Survivor might become more aggressive if it's not scoring enough, or a Hunter might learn to retreat when K/D drops.

## How Training Works

```
                     ┌─── Game Server (Colyseus) ───┐
                     │  Room 1: 12 agents fighting   │
                     │  Room 2: 12 agents fighting   │
                     └───────────┬───────────────────┘
                                 │ experience (state, action, reward)
                     ┌───────────▼───────────────────┐
                     │  PyTorch Trainer (GPU)          │
                     │  24 individual PPO networks     │
                     │  → ONNX export after training   │
                     └───────────┬───────────────────┘
                                 │ ONNX models
                     ┌───────────▼───────────────────┐
                     │  Node.js Bot Swarm              │
                     │  onnxruntime-node inference     │
                     │  + LLM strategic overrides      │
                     └───────────────────────────────┘
                                 │
              every 10 matches:  │  NATURAL SELECTION
              ┌──────────────────▼──────────────────┐
              │  Bottom 5 agents → deleted            │
              │  Top 5 agents → cloned + mutated     │
              │  (2% weight noise on clone)           │
              └──────────────────────────────────────┘
```

1. **24 agents** play in 2 parallel rooms (12 per room)
2. Each agent collects experience (state, action, reward) every frame
3. Experience is sent to **PyTorch trainer on GPU** for PPO updates
4. Updated models exported as ONNX and loaded by bots in real-time
5. Every 3 matches, the **LLM** analyzes performance and adjusts strategy
6. Every 10 matches, **natural selection**: weakest 5 die, strongest 5 cloned with mutations
7. Runs **24/7** with auto-restart

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| NN Training | **PyTorch** (GPU, CUDA) | PPO per-agent policy networks |
| NN Inference | **onnxruntime-node** | Fast ONNX inference in Node.js |
| Strategic Brain | **kimi-k2.5** via Ollama Cloud | LLM strategy analysis + planning |
| Game Client | **colyseus.js** v0.15 | WebSocket multiplayer (same as real players) |
| Protocol | **protobufjs** | Binary movement/shoot messages |
| Dashboard | **Express** + vanilla JS | Real-time web UI at port 3000 |
| Deployment | **systemd** / auto-restart scripts | 24/7 operation |

## Setup

### Prerequisites

- **Node.js 18+**
- **Python 3.10+** with PyTorch
- A running chainer.io game server
- Ollama Cloud API key (for LLM brain)

### Install

```bash
git clone https://github.com/matveyco/chainer-agent.git
cd chainer-agent
npm install

# Python training service
python3 -m venv venv
source venv/bin/activate
pip install torch numpy onnx onnxscript flask
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
GAME_SERVER_URL=https://your-game-server.com
TRAINER_URL=http://localhost:5555
POPULATION_SIZE=12
NUM_ROOMS=2
MATCH_TIMEOUT=120000
SELECTION_INTERVAL=10
NUM_CULL=5
OLLAMA_CLOUD_API_KEY=your-ollama-api-key
DEEP_ANALYSIS_MODEL=kimi-k2.5:cloud
NODE_TLS_REJECT_UNAUTHORIZED=0
```

### Run

```bash
# Option 1: 24/7 mode (manages all services)
nohup bash deploy/run-forever.sh > runner.log 2>&1 &

# Option 2: Manual (3 terminals)
source venv/bin/activate && python training/trainer.py   # Terminal 1: GPU trainer
node src/index.js                                         # Terminal 2: Bot swarm
node web/server.js                                        # Terminal 3: Dashboard
```

### Dashboard

Open `http://your-server:3000` in a browser:
- **Leaderboard** — all agents ranked by score, K/D, model version
- **Agent Detail** — click any agent to see personality, strategy bars, LLM thoughts, score trends
- **Controls** — trigger natural selection, clone agents, reset agents

## Project Structure

```
chainer-agent/
├── src/
│   ├── index.js                  # Entry point
│   ├── bot/
│   │   ├── SmartBot.js           # Bot with dual-brain architecture
│   │   ├── AgentBrain.js         # ONNX inference (NN layer)
│   │   ├── StrategicBrain.js     # LLM strategy (overrides NN)
│   │   └── StateExtractor.js     # Game state → 18-dim input vector
│   ├── evolution/
│   │   ├── Trainer.js            # 24/7 training loop, 2 rooms, selection
│   │   ├── Population.js         # NEAT population (legacy, kept for compat)
│   │   └── Genome.js             # Model persistence
│   ├── network/                  # Colyseus + protobuf + matchmaking
│   ├── game/                     # SpatialGrid, state deserialization
│   └── metrics/                  # Fitness tracking, generation logging
├── training/
│   └── trainer.py                # PyTorch PPO service (GPU)
├── web/
│   ├── server.js                 # Dashboard Express server
│   └── public/index.html         # Dashboard UI
├── deploy/
│   └── run-forever.sh            # 24/7 runner with auto-restart
├── config/default.json           # Default parameters
└── .env.example                  # Environment config template
```

## API Endpoints

### Python Trainer (port 5555)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status + GPU info |
| `/stats` | GET | All agent stats |
| `/model/:id` | GET | Download ONNX model |
| `/experience` | POST | Submit training experience |
| `/select` | POST | Trigger natural selection |
| `/agent/:id/reset` | POST | Reset agent to random |
| `/agent/:id/clone/:src` | POST | Clone source → target |
| `/agent/:id/history` | GET | Training log history |

### Web Dashboard (port 3000)

Proxies to trainer API at `/api/*` and serves the dashboard UI.

## References

- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) — Proximal Policy Optimization
- [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) — Self-improving loops
- [Colyseus](https://colyseus.io) — Multiplayer game framework
- [ONNX Runtime](https://onnxruntime.ai) — Cross-platform ML inference
- [chainer.io](https://chainer.io) — The game

## License

MIT
