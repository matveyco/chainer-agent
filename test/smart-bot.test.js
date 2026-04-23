const test = require("node:test");
const assert = require("node:assert/strict");

const { SmartBot } = require("../src/bot/SmartBot");

function makeBot() {
  return new SmartBot(
    "agent_7_session",
    null,
    {
      server: { mapName: "arena", weaponType: "rocket" },
      training: { defaultModelAlias: "latest" },
      bot: { arenaSafeSize: 55 },
      reward: {},
    },
    "agent_7",
    {}
  );
}

test("smart bot normalizes millisecond weapon cooldowns into seconds", () => {
  const bot = makeBot();
  bot.data = { weaponCoolDown: 1250 };
  assert.equal(bot._getWeaponCooldownSeconds(), 1.25);

  bot.data = { weaponCoolDown: 1.4 };
  assert.equal(bot._getWeaponCooldownSeconds(), 1.4);
});

test("smart bot builds shoot targets from the aimed point instead of random enemy-only fallback", () => {
  const bot = makeBot();
  bot.closestEnemy = { position: { x: 20, y: 0, z: 20 } };
  const target = bot._buildShootTarget({ x: 10, y: 0, z: -5 });

  assert.equal(target.length, 3);
  assert.equal(Math.abs(target[0] - 10) < 1, true);
  assert.equal(Math.abs(target[2] + 5) < 1, true);
});

test("smart bot lead-target aim: predicts enemy position from last track + projectile travel", () => {
  const bot = makeBot();
  bot.positionArray = [0, 0, 0];
  // Enemy has moved from (12, 0) to (15, 0) in 1.0s -> vx = 3 m/s, vz = 0.
  bot._enemyTracks.set("e1", { x: 12, z: 0, t: Date.now() - 1000 });
  bot.strategicBrain = {
    getStrategyVector: () => ({ accuracy_focus: 1.0 }),
  };
  const enemy = {
    id: "e1",
    position: { x: 15, y: 0, z: 0 },
  };
  const aim = bot._computeLeadTarget(enemy);
  // distance=15, projectileSpeed=30 -> travel=0.5s; lead = 3 m/s * 0.5s = 1.5 m
  assert.ok(Math.abs(aim.x - (15 + 1.5)) < 0.1, `expected lead x~16.5, got ${aim.x}`);
  assert.equal(aim.z, 0);
});

test("smart bot lead-target aim: falls back to dead-aim when track is stale", () => {
  const bot = makeBot();
  bot.positionArray = [0, 0, 0];
  bot._enemyTracks.set("e1", { x: 12, z: 0, t: Date.now() - 3000 }); // 3s old > 1s cap
  bot.strategicBrain = { getStrategyVector: () => ({ accuracy_focus: 1.0 }) };
  const aim = bot._computeLeadTarget({ id: "e1", position: { x: 15, y: 0, z: 0 } });
  // Stale track -> no lead; aim equals current position.
  assert.equal(aim.x, 15);
});

test("smart bot cluster-dir: points at centroid of nearby enemies", () => {
  const bot = makeBot();
  bot.positionArray = [0, 0, 0];
  const nearby = [
    { id: "e1", position: { x: 10, z: 10 } },
    { id: "e2", position: { x: 10, z: -10 } },
  ];
  const dir = bot._computeEnemyClusterDir(nearby);
  // Centroid at (10, 0); direction from (0,0) is +X.
  assert.ok(dir.dirX > 0.95, `expected dirX~1, got ${dir.dirX}`);
  assert.ok(Math.abs(dir.dirZ) < 0.1);
});

test("smart bot cluster-dir: null when no enemies", () => {
  const bot = makeBot();
  bot.positionArray = [0, 0, 0];
  assert.equal(bot._computeEnemyClusterDir([]), null);
  assert.equal(bot._computeEnemyClusterDir(null), null);
});

test("smart bot crystal proxy: counts score gains outside combat window", () => {
  const bot = makeBot();
  bot.data = { score: 0 };
  bot.lastCombatAt = Date.now() - 5000; // no recent combat
  bot._sampleCrystalProxy();
  assert.equal(bot.getCrystalPickupsApprox(), 0); // first sample: 0 delta

  bot.data = { score: 100 };
  bot._sampleCrystalProxy();
  assert.equal(bot.getCrystalPickupsApprox(), 1, "score+100 outside combat = 1 pickup");

  bot.data = { score: 200 };
  bot.lastCombatAt = Date.now(); // combat right now
  bot._sampleCrystalProxy();
  assert.equal(bot.getCrystalPickupsApprox(), 1, "score+100 during combat = NOT counted");
});
