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

// Process-level error handlers. Production was seeing daily silent
// process.exit(1) crashes — systemd restarted within 10s but we lost
// the supervisor counters AND no error trace was captured. These two
// handlers log the cause to stderr (which goes to bots.log via systemd
// StandardError=append) before letting the process die. Without these,
// uncaught Promise rejections in async/await chains kill node silently.
//
// AFTER deploying these handlers we discovered most crashes were
// Colyseus state-schema decode errors ('refId not found') hitting
// uncaughtException via the WebSocket message callback (we can't catch
// them inside Colyseus internals). These errors are isolated to one
// session — patch decoder fails, but the bot can survive if we don't
// exit. Whitelist them for recovery.
const RECOVERABLE_PATTERNS = [
  /refId.*not found/, // colyseus.js schema decode mismatch
];

process.on("uncaughtException", (err, origin) => {
  const msg = (err && (err.message || String(err))) || "";
  const recoverable = RECOVERABLE_PATTERNS.some((re) => re.test(msg));
  if (recoverable) {
    console.error(`[RECOVER] uncaughtException survived (${msg.slice(0, 100)}) at ${origin}`);
    return;
  }
  console.error(`[FATAL] uncaughtException at ${origin}:`, err && (err.stack || err.message || err));
  process.exit(1);
});
process.on("unhandledRejection", (reason, promise) => {
  console.error("[FATAL] unhandledRejection:", reason && (reason.stack || reason.message || reason));
  // Don't exit on unhandled rejection — most are recoverable transient
  // network errors from the LLM coach or trainer fetch. Log and continue.
});

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
  const conn = new Connection(config.server.endpoint, {
    authKey: config.server.authKey,
  });
  const userID = `test_${generateID(6)}`;
  const forceCreateRoom = Boolean(flags.forceCreateRoom || process.env.ALLOW_FORCE_CREATE_ROOM_FOR_DEV === "1");

  try {
    const room = await conn.connectViaActiveQueue(
      userID,
      config.server.weaponType,
      {
        forceCreateRoom,
        queueTimeoutMs: 30000,
        onStateUpdate: (data) => logger.info("Got state update:", data.byteLength, "bytes"),
        onPlayerJoined: (data) => logger.info("Player joined:", data.userID),
        onPlayerDie: (data) => logger.info("Player died:", data.userID),
        onPlayerHit: (data) => logger.info("Player hit:", data.userID, "hp:", data.newHealth),
        onDispose: () => logger.info("Room disposed"),
        onLeave: (code) => logger.info("Left room, code:", code),
      },
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
