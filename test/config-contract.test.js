const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const os = require("os");
const path = require("path");

const {
  applyEnvOverrides,
  validateEnvContract,
} = require("../src/runtime/ConfigContract");

function makeConfig() {
  return {
    server: {
      endpoint: "https://arena.example",
      roomName: "TimeLimited",
      mapName: "arena",
      weaponType: "rocket",
      colyseusVersion: "0.15.28",
    },
    trainerUrl: "http://localhost:5555",
    rooms: { count: 2, agentsPerRoom: 12 },
    bot: { matchTimeout: 120000 },
    training: {
      selectionInterval: 10,
      numCull: 5,
      defaultPolicyFamily: "arena-main",
    },
    evaluation: {
      sampleMatches: 3,
      promotionMargin: 0.05,
      minWinRate: 0.55,
      windowIntervalMinutes: 90,
      autoStageMinVersionDelta: 1000,
      stagingRecentMatches: 4,
      stagingMinCombatSignalRatio: 0.75,
      stagingMinFillRatio: 0.95,
      stagingMinJoinSuccessRate: 0.97,
      stagingMinShotRate: 0.03,
      stagingMinPolicyShare: 0.1,
      stagingMinDamagePerShot: 0.25,
    },
    runtime: { port: 3101, strategyCoachTimeoutMs: 3000 },
    persistence: { rosterFile: "config/roster.json" },
  };
}

test("config contract applies evaluation and family env overrides", () => {
  const config = applyEnvOverrides(makeConfig(), {
    GAME_SERVER_URL: "https://arena.override",
    TRAINER_URL: "http://trainer.override:5555",
    DEFAULT_POLICY_FAMILY: "arena-experimental",
    EVALUATION_SAMPLE_MATCHES: "7",
    EVALUATION_PROMOTION_MARGIN: "0.11",
    EVALUATION_MIN_WIN_RATE: "0.66",
    EVALUATION_WINDOW_MINUTES: "120",
    EVALUATION_AUTO_STAGE_MIN_VERSION_DELTA: "1500",
    EVALUATION_STAGING_RECENT_MATCHES: "5",
    EVALUATION_STAGING_MIN_COMBAT_SIGNAL_RATIO: "0.8",
    EVALUATION_STAGING_MIN_FILL_RATIO: "0.98",
    EVALUATION_STAGING_MIN_JOIN_SUCCESS_RATE: "0.99",
    EVALUATION_STAGING_MIN_SHOT_RATE: "0.05",
    EVALUATION_STAGING_MIN_POLICY_SHARE: "0.25",
    EVALUATION_STAGING_MIN_DAMAGE_PER_SHOT: "0.5",
    COMBAT_RECOVERY_WINDOW: "6",
    COMBAT_RECOVERY_MIN_SIGNAL_RATIO: "0.4",
    COMBAT_RECOVERY_COOLDOWN_MS: "120000",
    STRATEGY_COACH_TIMEOUT_MS: "4000",
  });

  assert.equal(config.server.endpoint, "https://arena.override");
  assert.equal(config.trainerUrl, "http://trainer.override:5555");
  assert.equal(config.training.defaultPolicyFamily, "arena-experimental");
  assert.equal(config.evaluation.sampleMatches, 7);
  assert.equal(config.evaluation.promotionMargin, 0.11);
  assert.equal(config.evaluation.minWinRate, 0.66);
  assert.equal(config.evaluation.windowIntervalMinutes, 120);
  assert.equal(config.evaluation.autoStageMinVersionDelta, 1500);
  assert.equal(config.evaluation.stagingRecentMatches, 5);
  assert.equal(config.evaluation.stagingMinCombatSignalRatio, 0.8);
  assert.equal(config.evaluation.stagingMinFillRatio, 0.98);
  assert.equal(config.evaluation.stagingMinJoinSuccessRate, 0.99);
  assert.equal(config.evaluation.stagingMinShotRate, 0.05);
  assert.equal(config.evaluation.stagingMinPolicyShare, 0.25);
  assert.equal(config.evaluation.stagingMinDamagePerShot, 0.5);
  assert.equal(config.runtime.combatRecoveryWindow, 6);
  assert.equal(config.runtime.combatRecoveryMinSignalRatio, 0.4);
  assert.equal(config.runtime.combatRecoveryCooldownMs, 120000);
  assert.equal(config.runtime.strategyCoachTimeoutMs, 4000);
});

test("config contract supports documented compatibility aliases", () => {
  const config = applyEnvOverrides(makeConfig(), {
    ENDPOINT: "https://legacy-endpoint.example",
    ROOM: "LegacyRoom",
    NUM_CLIENTS: "18",
    OAUTH_API_KEY: "secret",
  });

  assert.equal(config.server.endpoint, "https://legacy-endpoint.example");
  assert.equal(config.server.roomName, "LegacyRoom");
  assert.equal(config.rooms.agentsPerRoom, 18);
  assert.equal(config.server.authKey, "secret");
});

test("config contract validates production env completeness and TLS safety", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "chainer-config-"));
  const rosterPath = path.join(tempDir, "roster.json");
  fs.writeFileSync(rosterPath, JSON.stringify({ agents: [{ agentId: "agent_0" }] }));

  const env = {
    GAME_SERVER_URL: "https://arena.example",
    TRAINER_URL: "http://localhost:5555",
    SUPERVISOR_PORT: "3101",
    DASHBOARD_PORT: "3000",
    ROOM_NAME: "TimeLimited",
    MAP_NAME: "arena",
    WEAPON_TYPE: "rocket",
    POPULATION_SIZE: "12",
    MATCH_TIMEOUT: "120000",
    NUM_ROOMS: "2",
    SELECTION_INTERVAL: "10",
    NUM_CULL: "5",
    BOT_ROSTER_PATH: rosterPath,
    OAUTH_API_KEY: "secret",
    NODE_TLS_REJECT_UNAUTHORIZED: "0",
  };

  const config = makeConfig();
  config.persistence.rosterFile = rosterPath;

  const result = validateEnvContract(config, env);
  assert.equal(result.errors.some((item) => item.includes("NODE_TLS_REJECT_UNAUTHORIZED=0")), true);

  const safe = validateEnvContract(config, {
    ...env,
    ALLOW_INSECURE_TLS_FOR_DEV: "1",
    NODE_TLS_REJECT_UNAUTHORIZED: "0",
  });
  assert.equal(safe.errors.length, 0);
});
