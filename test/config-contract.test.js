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
    },
    runtime: { port: 3101 },
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
  });

  assert.equal(config.server.endpoint, "https://arena.override");
  assert.equal(config.trainerUrl, "http://trainer.override:5555");
  assert.equal(config.training.defaultPolicyFamily, "arena-experimental");
  assert.equal(config.evaluation.sampleMatches, 7);
  assert.equal(config.evaluation.promotionMargin, 0.11);
  assert.equal(config.evaluation.minWinRate, 0.66);
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
