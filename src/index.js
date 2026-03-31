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

require("dotenv").config({ quiet: true });

const fs = require("fs");
const { Trainer } = require("./evolution/Trainer");
const { Dashboard } = require("./metrics/Dashboard");
const { generateID } = require("./network/Protocol");
const {
  applyCliPatches,
  applyEnvOverrides,
  loadConfig,
  parseCliArgs,
  validateEnvContract,
  validateRuntimeVersions,
} = require("./runtime/ConfigContract");
const logger = require("./utils/logger");

let config = applyEnvOverrides(loadConfig(), process.env);

// Parse CLI args (highest priority)
const { flags, patches } = parseCliArgs(process.argv.slice(2));
config = applyCliPatches(config, patches);

// Validate required config
const envValidation = validateEnvContract(config, process.env);
const runtimeValidation = validateRuntimeVersions(config);
const validationErrors = [...envValidation.errors, ...runtimeValidation.errors];
if (validationErrors.length > 0) {
  console.error("Error: production config validation failed.");
  for (const message of validationErrors) {
    console.error(` - ${message}`);
  }
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
