const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const CRITICAL_DEPENDENCIES = [
  "colyseus.js",
  "dotenv",
  "express",
  "onnxruntime-node",
  "protobufjs",
];

function loadConfig(configPath = path.resolve("config/default.json")) {
  return JSON.parse(fs.readFileSync(configPath, "utf-8"));
}

function applyEnvOverrides(config, env = process.env) {
  const next = JSON.parse(JSON.stringify(config));
  next.server.endpoint = env.GAME_SERVER_URL || next.server.endpoint || null;
  next.trainerUrl = env.TRAINER_URL || next.trainerUrl || "http://localhost:5555";
  if (env.ROOM_NAME) next.server.roomName = env.ROOM_NAME;
  if (env.MAP_NAME) next.server.mapName = env.MAP_NAME;
  if (env.WEAPON_TYPE) next.server.weaponType = env.WEAPON_TYPE;
  if (env.POPULATION_SIZE) next.rooms.agentsPerRoom = parseInt(env.POPULATION_SIZE, 10);
  if (env.MATCH_TIMEOUT) next.bot.matchTimeout = parseInt(env.MATCH_TIMEOUT, 10);
  if (env.NUM_ROOMS) next.rooms.count = parseInt(env.NUM_ROOMS, 10);
  if (env.SELECTION_INTERVAL) next.training.selectionInterval = parseInt(env.SELECTION_INTERVAL, 10);
  if (env.NUM_CULL) next.training.numCull = parseInt(env.NUM_CULL, 10);
  if (env.DEFAULT_POLICY_FAMILY) next.training.defaultPolicyFamily = env.DEFAULT_POLICY_FAMILY;
  if (env.EVALUATION_SAMPLE_MATCHES) next.evaluation.sampleMatches = parseInt(env.EVALUATION_SAMPLE_MATCHES, 10);
  if (env.EVALUATION_PROMOTION_MARGIN) next.evaluation.promotionMargin = parseFloat(env.EVALUATION_PROMOTION_MARGIN);
  if (env.EVALUATION_MIN_WIN_RATE) next.evaluation.minWinRate = parseFloat(env.EVALUATION_MIN_WIN_RATE);
  if (env.EVALUATION_WINDOW_MINUTES) next.evaluation.windowIntervalMinutes = parseInt(env.EVALUATION_WINDOW_MINUTES, 10);
  if (env.EVALUATION_AUTO_STAGE_MIN_VERSION_DELTA) next.evaluation.autoStageMinVersionDelta = parseInt(env.EVALUATION_AUTO_STAGE_MIN_VERSION_DELTA, 10);
  if (env.EVALUATION_STAGING_RECENT_MATCHES) next.evaluation.stagingRecentMatches = parseInt(env.EVALUATION_STAGING_RECENT_MATCHES, 10);
  if (env.EVALUATION_STAGING_MIN_COMBAT_SIGNAL_RATIO) next.evaluation.stagingMinCombatSignalRatio = parseFloat(env.EVALUATION_STAGING_MIN_COMBAT_SIGNAL_RATIO);
  if (env.EVALUATION_STAGING_MIN_FILL_RATIO) next.evaluation.stagingMinFillRatio = parseFloat(env.EVALUATION_STAGING_MIN_FILL_RATIO);
  if (env.EVALUATION_STAGING_MIN_JOIN_SUCCESS_RATE) next.evaluation.stagingMinJoinSuccessRate = parseFloat(env.EVALUATION_STAGING_MIN_JOIN_SUCCESS_RATE);
  if (env.EVALUATION_STAGING_MIN_SHOT_RATE) next.evaluation.stagingMinShotRate = parseFloat(env.EVALUATION_STAGING_MIN_SHOT_RATE);
  if (env.EVALUATION_STAGING_MIN_POLICY_SHARE) next.evaluation.stagingMinPolicyShare = parseFloat(env.EVALUATION_STAGING_MIN_POLICY_SHARE);
  if (env.EVALUATION_STAGING_MIN_DAMAGE_PER_SHOT) next.evaluation.stagingMinDamagePerShot = parseFloat(env.EVALUATION_STAGING_MIN_DAMAGE_PER_SHOT);
  if (env.SUPERVISOR_PORT) next.runtime.port = parseInt(env.SUPERVISOR_PORT, 10);
  if (env.COMBAT_RECOVERY_WINDOW) next.runtime.combatRecoveryWindow = parseInt(env.COMBAT_RECOVERY_WINDOW, 10);
  if (env.COMBAT_RECOVERY_MIN_SIGNAL_RATIO) next.runtime.combatRecoveryMinSignalRatio = parseFloat(env.COMBAT_RECOVERY_MIN_SIGNAL_RATIO);
  if (env.COMBAT_RECOVERY_COOLDOWN_MS) next.runtime.combatRecoveryCooldownMs = parseInt(env.COMBAT_RECOVERY_COOLDOWN_MS, 10);
  if (env.STRATEGY_COACH_TIMEOUT_MS) next.runtime.strategyCoachTimeoutMs = parseInt(env.STRATEGY_COACH_TIMEOUT_MS, 10);
  if (env.BOT_ROSTER_PATH) next.persistence.rosterFile = env.BOT_ROSTER_PATH;
  if (env.OAUTH_API_KEY) next.server.authKey = env.OAUTH_API_KEY;
  if (env.OLLAMA_CLOUD_API_KEY) next.ollamaApiKey = env.OLLAMA_CLOUD_API_KEY;
  if (env.DEEP_ANALYSIS_MODEL) next.ollamaModel = env.DEEP_ANALYSIS_MODEL;
  return next;
}

function parseCliArgs(args) {
  const flags = {};
  const patches = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--resume") {
      flags.resume = args[i + 1] && !args[i + 1].startsWith("--") ? args[++i] : true;
    } else if (args[i] === "--watch") {
      flags.watch = args[i + 1] && !args[i + 1].startsWith("--") ? args[++i] : true;
    } else if (args[i] === "--population") {
      patches.population = parseInt(args[++i], 10);
    } else if (args[i] === "--endpoint") {
      patches.endpoint = args[++i];
    } else if (args[i] === "--test-connect") {
      flags.testConnect = true;
    } else if (args[i] === "--service") {
      flags.service = args[++i];
    } else if (args[i] === "--live") {
      flags.live = true;
    } else if (args[i] === "--json") {
      flags.json = true;
    }
  }
  return { flags, patches };
}

function applyCliPatches(config, patches = {}) {
  const next = JSON.parse(JSON.stringify(config));
  if (Number.isFinite(patches.population)) next.rooms.agentsPerRoom = patches.population;
  if (patches.endpoint) next.server.endpoint = patches.endpoint;
  return next;
}

function validateEnvContract(config, env = process.env) {
  const errors = [];
  const warnings = [];

  const required = [
    "GAME_SERVER_URL",
    "TRAINER_URL",
    "SUPERVISOR_PORT",
    "DASHBOARD_PORT",
    "ROOM_NAME",
    "MAP_NAME",
    "WEAPON_TYPE",
    "POPULATION_SIZE",
    "MATCH_TIMEOUT",
    "NUM_ROOMS",
    "SELECTION_INTERVAL",
    "NUM_CULL",
    "BOT_ROSTER_PATH",
  ];

  for (const key of required) {
    if (!env[key]) {
      errors.push(`${key} is required`);
    }
  }

  for (const key of ["SUPERVISOR_PORT", "DASHBOARD_PORT", "POPULATION_SIZE", "MATCH_TIMEOUT", "NUM_ROOMS", "SELECTION_INTERVAL", "NUM_CULL"]) {
    if (env[key] && !Number.isFinite(parseInt(env[key], 10))) {
      errors.push(`${key} must be numeric`);
    }
  }

  for (const key of [
    "EVALUATION_SAMPLE_MATCHES",
    "EVALUATION_WINDOW_MINUTES",
    "EVALUATION_AUTO_STAGE_MIN_VERSION_DELTA",
    "EVALUATION_STAGING_RECENT_MATCHES",
    "COMBAT_RECOVERY_WINDOW",
    "COMBAT_RECOVERY_COOLDOWN_MS",
    "STRATEGY_COACH_TIMEOUT_MS",
  ]) {
    if (env[key] && !Number.isFinite(parseInt(env[key], 10))) {
      errors.push(`${key} must be numeric`);
    }
  }

  for (const key of [
    "EVALUATION_PROMOTION_MARGIN",
    "EVALUATION_MIN_WIN_RATE",
    "EVALUATION_STAGING_MIN_COMBAT_SIGNAL_RATIO",
    "EVALUATION_STAGING_MIN_FILL_RATIO",
    "EVALUATION_STAGING_MIN_JOIN_SUCCESS_RATE",
    "EVALUATION_STAGING_MIN_SHOT_RATE",
    "EVALUATION_STAGING_MIN_POLICY_SHARE",
    "EVALUATION_STAGING_MIN_DAMAGE_PER_SHOT",
    "COMBAT_RECOVERY_MIN_SIGNAL_RATIO",
  ]) {
    if (env[key] && !Number.isFinite(parseFloat(env[key]))) {
      errors.push(`${key} must be numeric`);
    }
  }

  if (!config.server?.endpoint) {
    errors.push("GAME_SERVER_URL did not resolve into config.server.endpoint");
  }

  const rosterPath = path.resolve(config.persistence?.rosterFile || env.BOT_ROSTER_PATH || "config/roster.json");
  if (!fs.existsSync(rosterPath)) {
    errors.push(`Roster file not found: ${rosterPath}`);
  }

  if (env.NODE_TLS_REJECT_UNAUTHORIZED === "0" && env.ALLOW_INSECURE_TLS_FOR_DEV !== "1") {
    errors.push("NODE_TLS_REJECT_UNAUTHORIZED=0 is forbidden in production doctor checks");
  }

  if (env.TLS_CA_CERT_PATH && !fs.existsSync(path.resolve(env.TLS_CA_CERT_PATH))) {
    errors.push(`TLS_CA_CERT_PATH not found: ${env.TLS_CA_CERT_PATH}`);
  }

  if (env.DEEP_ANALYSIS_MODEL && !env.OLLAMA_CLOUD_API_KEY) {
    warnings.push("DEEP_ANALYSIS_MODEL is set but OLLAMA_CLOUD_API_KEY is missing; shadow coach will remain disabled");
  }

  if (env.OLLAMA_CLOUD_API_KEY && !env.DEEP_ANALYSIS_MODEL) {
    warnings.push("OLLAMA_CLOUD_API_KEY is set without DEEP_ANALYSIS_MODEL; default model will be used");
  }

  return { errors, warnings, rosterPath };
}

function validateRuntimeVersions(config, packageJsonPath = path.resolve("package.json")) {
  const errors = [];
  const warnings = [];
  const pkg = JSON.parse(fs.readFileSync(packageJsonPath, "utf-8"));

  const nodeMajor = parseInt(process.versions.node.split(".")[0], 10);
  if (!Number.isFinite(nodeMajor) || nodeMajor < 20) {
    errors.push(`Node 20+ required, found ${process.version}`);
  }

  for (const dep of CRITICAL_DEPENDENCIES) {
    const version = pkg.dependencies?.[dep];
    if (!version) {
      errors.push(`Missing critical dependency pin: ${dep}`);
      continue;
    }
    if (/^[~^]/.test(version)) {
      errors.push(`Critical dependency must be pinned exactly: ${dep}@${version}`);
    }
  }

  try {
    const installedColyseus = require("colyseus.js/package.json").version;
    if (config.server?.colyseusVersion && installedColyseus !== config.server.colyseusVersion) {
      errors.push(
        `colyseus.js mismatch: installed ${installedColyseus}, expected ${config.server.colyseusVersion}`
      );
    }
  } catch (err) {
    errors.push(`Unable to inspect colyseus.js: ${err.message}`);
  }

  const python = spawnSync("python3", ["-c", "import sys; print(sys.version.split()[0])"], {
    encoding: "utf-8",
  });
  if (python.status !== 0) {
    errors.push("python3 is required for production trainer checks");
  } else {
    const version = (python.stdout || "").trim();
    const [major, minor] = version.split(".").map((value) => parseInt(value, 10));
    if (!Number.isFinite(major) || !Number.isFinite(minor) || major < 3 || (major === 3 && minor < 12)) {
      errors.push(`Python 3.12+ required, found ${version}`);
    }
  }

  return { errors, warnings };
}

module.exports = {
  applyCliPatches,
  applyEnvOverrides,
  loadConfig,
  parseCliArgs,
  validateEnvContract,
  validateRuntimeVersions,
};
