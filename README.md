# Chainer Agent

Self-learning AI bots for the [chainer.io](https://chainer.io) 3rd-person multiplayer shooter arena. Bots evolve their neural networks through **NEAT (NeuroEvolution of Augmenting Topologies)** — they fight each other, and the best performers breed the next generation. Over time, each bot develops its own unique strategy to maximize kills, collect crystals, and climb the leaderboard.

Inspired by [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) — clear fitness metric, time-boxed evaluation cycles, autonomous improvement loop.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                      EVOLUTION LOOP                           │
│                                                               │
│   1. Create population of N neural networks (gen 0: random)   │
│   2. Connect N bots to game server via Colyseus WebSocket     │
│   3. Each bot uses its own neural network for 60Hz decisions   │
│   4. Match plays out — bots fight, shoot, collect crystals     │
│   5. Collect fitness: kills, accuracy, survival, damage        │
│   6. Natural selection: best performers survive                │
│   7. Crossover + mutation → new generation                     │
│   8. Save best genomes to disk                                 │
│   9. Repeat forever — each generation gets smarter             │
└──────────────────────────────────────────────────────────────┘
```

Each bot has its **own neural network brain** (14 inputs → 6 outputs). NEAT evolves both the network weights AND topology — starting from simple direct connections and growing hidden layers as needed. Every bot develops a unique strategy through its own evolutionary lineage.

### What Bots Learn

| Generations | Behavior |
|------------|----------|
| 0-5 | Random movement, accidental kills |
| 5-20 | Moving toward enemies, shooting more often |
| 20-50 | Deliberate targeting, improved accuracy, arena awareness |
| 50-100+ | Emergent strategies, ability usage, cooldown management |

### Fitness Function

```
fitness = (kills × 100) + (damage_dealt × 1) - (deaths × 50)
        + (accuracy × 50) + (survival_time × 0.5)
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Neuroevolution | [neataptic](https://github.com/wagenaartje/neataptic) | NEAT — evolves NN weights + topology |
| Game Client | [colyseus.js](https://github.com/colyseus/colyseus.js) v0.15 | WebSocket multiplayer (same protocol as real players) |
| Message Protocol | [protobufjs](https://github.com/protobufjs/protobuf.js) + long | Binary protobuf for movement/shoot messages |
| Spatial Queries | SpatialHashFast | Fast nearest-enemy lookup via spatial hashing |
| Terminal UI | [blessed](https://github.com/chjj/blessed) + [blessed-contrib](https://github.com/yaronn/blessed-contrib) | Real-time training dashboard |

---

## Setup

### Prerequisites

- **Node.js 18+**
- A running chainer.io game server

### Install

```bash
git clone https://github.com/matveyco/chainer-agent.git
cd chainer-agent
npm install
```

### Configure

Copy the example environment file and set your game server URL:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GAME_SERVER_URL=https://your-game-server.com
POPULATION_SIZE=15
MATCH_TIMEOUT=120000
```

See `.env.example` for all available options.

---

## Usage

### Train (Evolution Loop)

```bash
# Start fresh training
npm start

# Override settings via CLI
node src/index.js --population 20
node src/index.js --endpoint https://other-server.com
```

The terminal dashboard shows real-time progress:

```
╔══════════════════════════════════════════════════════════╗
║  GEN: 47    BOTS: 15   TIME: 2h14m                     ║
╠══════════════════════════════════════════════════════════╣
║  Best Fitness:  1646.5     Avg: 535.4                   ║
║  Best K/D:      3.0        Avg K/D: 1.21                ║
║  Accuracy:      42%        Kills: 31                    ║
║  Neurons:       20-21                                   ║
╚══════════════════════════════════════════════════════════╝
```

### Resume Training

```bash
node src/index.js --resume          # Resume from latest snapshot
node src/index.js --resume 50       # Resume from generation 50
```

### Watch Best Bot

```bash
node src/index.js --watch                        # Watch latest best genome
node src/index.js --watch data/best/gen_50.json   # Watch specific genome
```

### Test Connection

```bash
node src/index.js --test-connect
```

---

## Architecture

```
chainer-agent/
├── src/
│   ├── index.js                  # Entry point — CLI, training/watch/test modes
│   ├── bot/
│   │   ├── SmartBot.js           # Neural net-driven bot (60Hz game loop)
│   │   ├── BotBrain.js           # NN inference wrapper (input → action mapping)
│   │   └── StateExtractor.js     # Game state → 14-element normalized input vector
│   ├── evolution/
│   │   ├── Population.js         # NEAT population (wraps neataptic)
│   │   ├── Genome.js             # Save/load neural network genomes as JSON
│   │   └── Trainer.js            # Evolution loop orchestrator
│   ├── network/
│   │   ├── Connection.js         # Colyseus client + room join + handlers
│   │   ├── Protocol.js           # Protobuf encoding (InputMessage, ShootMessage)
│   │   └── Matchmaker.js         # HTTP matchmaking client
│   ├── game/
│   │   ├── SpatialGrid.js        # 2D spatial hash for proximity queries
│   │   ├── GameState.js          # Player positions, health, state tracking
│   │   └── Deserializer.js       # Binary state snapshot deserialization
│   ├── metrics/
│   │   ├── FitnessTracker.js     # Per-bot K/D, damage, accuracy tracking
│   │   ├── GenerationLog.js      # Generation stats aggregation + JSONL logging
│   │   └── Dashboard.js          # Terminal dashboard (blessed)
│   └── utils/
│       ├── math.js               # 3D vector math
│       └── logger.js             # Structured logging
├── config/
│   └── default.json              # Default evolution/fitness/bot parameters
├── protobuf/
│   └── types.json                # Protobuf message definitions
├── data/                         # Created at runtime (gitignored)
│   ├── generations/              # Full population snapshots (every 10 gens)
│   ├── best/                     # Best genome per generation
│   └── logs/                     # Training logs (JSONL)
├── .env.example                  # Environment config template
└── package.json
```

### Neural Network I/O

**14 Inputs (what the bot perceives):**

| # | Input | Description |
|---|-------|-------------|
| 0 | Health | Own health (0-1) |
| 1-2 | Position | Own X, Z in arena |
| 3 | Enemy distance | Distance to nearest enemy |
| 4-5 | Enemy angle | Sin/cos angle to nearest enemy |
| 6 | Enemy health | Nearest enemy health |
| 7 | Enemy count | Nearby enemy count |
| 8 | Center distance | Distance from arena center |
| 9 | Cooldown | Weapon on cooldown? |
| 10-11 | Velocity | Current movement direction |
| 12 | Has target | Enemy in weapon range? |
| 13 | Approach | Moving toward/away from enemy |

**6 Outputs (what the bot does):**

| # | Output | Description |
|---|--------|-------------|
| 0-1 | Move X, Z | Movement direction |
| 2-3 | Aim offset | Aim adjustment around target |
| 4 | Shoot | Fire weapon (threshold > 0.5) |
| 5 | Ability | Use ability (threshold > 0.5) |

---

## Game Protocol

Bots use the exact same Colyseus protocol as real players:

1. **Matchmaking** — `POST /matchmaker/join-queue` → poll `user-queue-position` → room assigned
2. **Join** — `client.joinById(roomId, { userID, weaponType })` via WebSocket
3. **Play** — send `room:player:input` (protobuf) + `room:player:shoot` (protobuf) at 60Hz
4. **Events** — receive `room:player:die`, `room:player:hit`, `room:state:update`

From the game server's perspective, bots are indistinguishable from human players.

---

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GAME_SERVER_URL` | *required* | Game server base URL |
| `ROOM_NAME` | `TimeLimited` | Room type to join |
| `MAP_NAME` | `arena` | Map name |
| `WEAPON_TYPE` | `rocket` | Bot weapon |
| `POPULATION_SIZE` | `15` | Bots per generation |
| `MATCH_TIMEOUT` | `120000` | Max match duration (ms) |
| `NODE_TLS_REJECT_UNAUTHORIZED` | — | Set to `0` for dev/staging servers |
| `NO_DASHBOARD` | — | Set to `1` to disable TUI, use console |
| `DEBUG` | — | Enable debug logging |

### Config File (`config/default.json`)

Evolution parameters, fitness weights, and bot behavior are configured in `config/default.json`. These provide sensible defaults that can be tuned as training progresses.

---

## Persistence

| File | Content | Frequency |
|------|---------|-----------|
| `data/best/gen_N.json` | Best genome (neural network) | Every generation |
| `data/generations/gen_N.json` | Full population snapshot | Every 10 generations |
| `data/logs/training.jsonl` | Generation stats | Every generation |

Genomes are JSON-serialized neataptic Networks. Load one programmatically:

```javascript
const { Network } = require("neataptic");
const genome = require("./data/best/gen_50.json");
const network = Network.fromJSON(genome);
const actions = network.activate(inputVector); // 14 inputs → 6 outputs
```

---

## References

- [NEAT paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) — Stanley & Miikkulainen, 2002
- [neataptic](https://github.com/wagenaartje/neataptic) — NEAT for JavaScript
- [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) — Self-improving experiment loops
- [Colyseus](https://colyseus.io) — Multiplayer game framework
- [chainer.io](https://chainer.io) — The game

## License

MIT
