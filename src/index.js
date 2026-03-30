#!/usr/bin/env node

/**
 * Chainer Agent — Self-Learning Bots for chainer.io Arena
 *
 * NEAT Neuroevolution-powered bots that fight, learn, and improve
 * across generations in the chainer.io 3rd-person shooter.
 *
 * Usage:
 *   node src/index.js                               # Start fresh training
 *   node src/index.js --resume                       # Resume from latest snapshot
 *   node src/index.js --resume 50                    # Resume from generation 50
 *   node src/index.js --watch                        # Watch best genome play
 *   node src/index.js --watch data/best/gen_50.json  # Watch specific genome
 *   node src/index.js --population 20                # Override population size
 *   node src/index.js --endpoint https://...         # Override server endpoint
 *   node src/index.js --test-connect                 # Test single bot connection
 */

require("dotenv").config();

const fs = require("fs");
const path = require("path");
const { Trainer } = require("./evolution/Trainer");
const { Genome } = require("./evolution/Genome");
const { Dashboard } = require("./metrics/Dashboard");
const { SmartBot } = require("./bot/SmartBot");
const { GameState } = require("./game/GameState");
const { generateID } = require("./network/Protocol");
const logger = require("./utils/logger");

// Load config (defaults from file, overridden by env vars)
const config = JSON.parse(
  fs.readFileSync(path.join(__dirname, "../config/default.json"), "utf-8")
);

// Apply environment variable overrides
config.server.endpoint = process.env.GAME_SERVER_URL || null;
config.trainerUrl = process.env.TRAINER_URL || "http://localhost:5555";
if (process.env.ROOM_NAME) config.server.roomName = process.env.ROOM_NAME;
if (process.env.MAP_NAME) config.server.mapName = process.env.MAP_NAME;
if (process.env.WEAPON_TYPE) config.server.weaponType = process.env.WEAPON_TYPE;
if (process.env.POPULATION_SIZE) config.evolution.populationSize = parseInt(process.env.POPULATION_SIZE);
if (process.env.MATCH_TIMEOUT) config.bot.matchTimeout = parseInt(process.env.MATCH_TIMEOUT);
if (process.env.NUM_ROOMS) config.numRooms = parseInt(process.env.NUM_ROOMS);
if (process.env.SELECTION_INTERVAL) config.selectionInterval = parseInt(process.env.SELECTION_INTERVAL);
if (process.env.NUM_CULL) config.numCull = parseInt(process.env.NUM_CULL);

// Parse CLI args (highest priority)
const args = process.argv.slice(2);
const flags = {};
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--resume") {
    flags.resume = args[i + 1] && !args[i + 1].startsWith("--") ? args[++i] : true;
  } else if (args[i] === "--watch") {
    flags.watch = args[i + 1] && !args[i + 1].startsWith("--") ? args[++i] : true;
  } else if (args[i] === "--population") {
    config.evolution.populationSize = parseInt(args[++i]);
  } else if (args[i] === "--endpoint") {
    config.server.endpoint = args[++i];
  } else if (args[i] === "--test-connect") {
    flags.testConnect = true;
  }
}

// Validate required config
if (!config.server.endpoint) {
  console.error("Error: GAME_SERVER_URL is required.");
  console.error("Set it in .env file or pass --endpoint https://your-server.com");
  console.error("See .env.example for reference.");
  process.exit(1);
}

// Ensure data directories exist
fs.mkdirSync(config.persistence.generationsDir, { recursive: true });
fs.mkdirSync(config.persistence.bestDir, { recursive: true });
fs.mkdirSync(config.persistence.logsDir, { recursive: true });

/**
 * Test mode: connect a single bot to verify connectivity.
 */
async function testConnect() {
  logger.info("Testing connection to", config.server.endpoint);

  const { Connection } = require("./network/Connection");
  const conn = new Connection(config.server.endpoint);
  const userID = `test_${generateID(6)}`;

  try {
    const room = await conn.connect(
      userID,
      config.server.roomName,
      config.server.mapName,
      config.server.weaponType,
      true,
      {
        onStateUpdate: (data) => logger.info("Got state update:", data.byteLength, "bytes"),
        onPlayerJoined: (data) => logger.info("Player joined:", data.userID),
        onPlayerDie: (data) => logger.info("Player died:", data.userID),
        onPlayerHit: (data) => logger.info("Player hit:", data.userID, "hp:", data.newHealth),
        onDispose: () => logger.info("Room disposed"),
        onLeave: (code) => logger.info("Left room, code:", code),
      }
    );

    conn.sendLoaded(`TestBot_${userID.slice(-4)}`);
    logger.info(`Connected! Room: ${room.roomId}`);
    logger.info("Sending RTT pings... Press Ctrl+C to stop.");

    setInterval(() => {
      room.send("room:rtt");
    }, 3000);
  } catch (err) {
    logger.error("Connection failed:", err.message);
    process.exit(1);
  }
}

/**
 * Watch mode: load a genome and watch it play.
 */
async function watchMode() {
  const { Network } = require("neataptic");

  let genomePath;
  if (typeof flags.watch === "string") {
    genomePath = flags.watch;
  } else {
    genomePath = Genome.findLatestBest(config.persistence.bestDir);
  }

  if (!genomePath || !fs.existsSync(genomePath)) {
    logger.error("No genome found to watch. Train first or specify path.");
    process.exit(1);
  }

  logger.info("Loading genome from", genomePath);
  const genomeJSON = Genome.load(genomePath);
  const network = Network.fromJSON(genomeJSON);

  const gameState = new GameState();
  const userID = `watch_${generateID(6)}`;
  const bot = new SmartBot(userID, network, config);

  logger.info("Connecting bot to", config.server.endpoint);

  await bot.connect(config.server.endpoint, gameState, true, () => {
    logger.info("Room disposed. Exiting.");
    bot.dispose();
    process.exit(0);
  });

  logger.info(`Bot ${userID} connected. Watching... Press Ctrl+C to stop.`);

  let last = performance.now();
  setInterval(() => {
    const now = performance.now();
    const dt = (now - last) / 1000;
    last = now;
    bot.update(dt);
  }, 1000 / 60);

  setInterval(() => {
    const stats = bot.fitness.toJSON();
    logger.info(`K: ${stats.kills} D: ${stats.deaths} Dmg: ${stats.damageDealt} Acc: ${stats.accuracy}%`);
  }, 10000);
}

/**
 * Training mode: run the evolution loop.
 */
async function trainMode() {
  const dashboard = new Dashboard();
  dashboard.init();

  const trainer = new Trainer(config, (summary) => {
    dashboard.update(summary);
  });

  if (flags.resume) {
    let snapshotPath;
    if (typeof flags.resume === "string") {
      if (fs.existsSync(flags.resume)) {
        snapshotPath = flags.resume;
      } else {
        snapshotPath = path.join(config.persistence.generationsDir, `gen_${flags.resume}.json`);
      }
    } else {
      snapshotPath = Genome.findLatestPopulation(config.persistence.generationsDir);
    }

    if (snapshotPath && fs.existsSync(snapshotPath)) {
      trainer.resumeFrom(snapshotPath);
      dashboard.log(`Resumed from ${path.basename(snapshotPath)}`);
    } else {
      logger.warn("No snapshot found to resume from. Starting fresh.");
    }
  }

  const shutdown = () => {
    logger.info("Shutting down... saving state...");
    trainer.stop();
    try {
      const savePath = trainer.saveState();
      logger.info("Population saved to", savePath);
    } catch (err) {
      logger.error("Failed to save state:", err.message);
    }
    dashboard.destroy();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  dashboard.log(`Training: ${config.evolution.populationSize} bots @ ${config.server.endpoint}`);
  logger.info(`Starting NEAT evolution with ${config.evolution.populationSize} bots`);
  logger.info(`Server: ${config.server.endpoint}`);

  await trainer.run();
}

// Main
(async () => {
  console.log("");
  console.log("  ╔══════════════════════════════════════════════╗");
  console.log("  ║  Chainer Agent — NEAT Neuroevolution Bots    ║");
  console.log("  ║  Self-Learning AI for chainer.io Arena       ║");
  console.log("  ╚══════════════════════════════════════════════╝");
  console.log("");

  if (flags.testConnect) {
    await testConnect();
  } else if (flags.watch) {
    await watchMode();
  } else {
    await trainMode();
  }
})().catch((err) => {
  logger.error("Fatal error:", err);
  process.exit(1);
});
