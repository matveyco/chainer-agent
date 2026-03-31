#!/usr/bin/env node

/**
 * Chainer Agent — Self-Improving Arena Bots
 *
 * Usage:
 *   node src/index.js                      # Start supervised swarm
 *   node src/index.js --test-connect       # Test single bot connection
 *   node src/index.js --population 12      # Override agents per room
 *   node src/index.js --endpoint https://… # Override server endpoint
 */

require("dotenv").config();

const fs = require("fs");
const { Trainer } = require("./evolution/Trainer");
const { Dashboard } = require("./metrics/Dashboard");
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
if (process.env.POPULATION_SIZE) config.rooms.agentsPerRoom = parseInt(process.env.POPULATION_SIZE, 10);
if (process.env.MATCH_TIMEOUT) config.bot.matchTimeout = parseInt(process.env.MATCH_TIMEOUT);
if (process.env.NUM_ROOMS) config.rooms.count = parseInt(process.env.NUM_ROOMS, 10);
if (process.env.SELECTION_INTERVAL) config.training.selectionInterval = parseInt(process.env.SELECTION_INTERVAL, 10);
if (process.env.NUM_CULL) config.training.numCull = parseInt(process.env.NUM_CULL, 10);
if (process.env.SUPERVISOR_PORT) config.runtime.port = parseInt(process.env.SUPERVISOR_PORT, 10);
if (process.env.BOT_ROSTER_PATH) config.persistence.rosterFile = process.env.BOT_ROSTER_PATH;
config.server.authKey = process.env.OAUTH_API_KEY || null;
config.ollamaApiKey = process.env.OLLAMA_CLOUD_API_KEY || null;
config.ollamaModel = process.env.DEEP_ANALYSIS_MODEL || "kimi-k2.5:cloud";

// Parse CLI args (highest priority)
const args = process.argv.slice(2);
const flags = {};
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--resume") {
    flags.resume = args[i + 1] && !args[i + 1].startsWith("--") ? args[++i] : true;
  } else if (args[i] === "--watch") {
    flags.watch = args[i + 1] && !args[i + 1].startsWith("--") ? args[++i] : true;
  } else if (args[i] === "--population") {
    config.rooms.agentsPerRoom = parseInt(args[++i], 10);
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
for (const dir of [
  config.persistence.generationsDir,
  config.persistence.bestDir,
  config.persistence.logsDir,
  "data",
]) {
  fs.mkdirSync(dir, { recursive: true });
}

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
 * Training mode: run the evolution loop.
 */
async function trainMode() {
  const dashboard = new Dashboard();
  dashboard.init();

  const trainer = new Trainer(config);

  if (flags.resume) {
    logger.warn("--resume is deprecated in the supervised PPO runtime. Starting fresh.");
  }

  const shutdown = () => {
    logger.info("Shutting down... saving state...");
    trainer.stop();
    try {
      const savePath = trainer.saveState();
      logger.info("Runtime snapshot saved to", savePath);
    } catch (err) {
      logger.error("Failed to save state:", err.message);
    }
    dashboard.destroy();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  dashboard.log(`Swarm: ${config.rooms.count} rooms × ${config.rooms.agentsPerRoom} agents @ ${config.server.endpoint}`);
  logger.info(
    `Starting supervised swarm with ${config.rooms.count} rooms × ${config.rooms.agentsPerRoom} agents`
  );
  logger.info(`Server: ${config.server.endpoint}`);

  await trainer.run();
}

// Main
(async () => {
  console.log("");
  console.log("  ╔══════════════════════════════════════════════╗");
  console.log("  ║  Chainer Agent — Self-Improving Arena Bots   ║");
  console.log("  ║  PPO Reflexes + Match-Boundary Strategy      ║");
  console.log("  ╚══════════════════════════════════════════════╝");
  console.log("");

  if (flags.testConnect) {
    await testConnect();
  } else if (flags.watch) {
    logger.error("--watch is no longer supported in the PPO supervisor runtime.");
    process.exit(1);
  } else {
    await trainMode();
  }
})().catch((err) => {
  logger.error("Fatal error:", err);
  process.exit(1);
});
