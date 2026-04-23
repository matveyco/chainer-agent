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

test("blend: alpha=0 returns pure tactical action (no NN influence)", () => {
  const bot = makeBot();
  const tactical = {
    moveX: 1.0,
    moveZ: 0.0,
    aimOffsetX: 0.5,
    aimOffsetZ: -0.5,
    shouldShoot: true,
    shouldUseAbility: false,
    abilityHint: "rampage",
  };
  const nn = {
    moveX: -1.0,
    moveZ: 0.5,
    aimOffsetX: 2.0,
    aimOffsetZ: 2.0,
    shouldShoot: false,
    shouldUseAbility: true,
  };
  const blended = bot._blendActions(tactical, nn, 0);
  // alpha=0 means pure tactical — NN should be entirely ignored.
  assert.equal(blended.moveX, 1.0);
  assert.equal(blended.moveZ, 0.0);
  assert.equal(blended.aimOffsetX, 0.5);
  assert.equal(blended.shouldShoot, true);
  assert.equal(blended.shouldUseAbility, false);
});

test("blend: continuous outputs are convex combo, discrete outputs OR'd from NN", () => {
  const bot = makeBot();
  const tactical = {
    moveX: 1.0,
    moveZ: 0.0,
    aimOffsetX: 0.0,
    aimOffsetZ: 0.0,
    shouldShoot: false,
    shouldUseAbility: false,
    abilityHint: "jump",
  };
  const nn = {
    moveX: -1.0,
    moveZ: 1.0,
    aimOffsetX: 1.0,
    aimOffsetZ: 1.0,
    shouldShoot: true,
    shouldUseAbility: true,
  };
  const blended = bot._blendActions(tactical, nn, 0.25);
  // 25% NN, 75% tactical for continuous outputs.
  assert.ok(Math.abs(blended.moveX - (0.25 * -1.0 + 0.75 * 1.0)) < 1e-6, `got moveX=${blended.moveX}`);
  assert.ok(Math.abs(blended.moveZ - 0.25) < 1e-6);
  assert.ok(Math.abs(blended.aimOffsetX - 0.25) < 1e-6);
  // NN's shoot/ability go through (OR'd).
  assert.equal(blended.shouldShoot, true);
  assert.equal(blended.shouldUseAbility, true);
  // Ability hint always tactical's choice.
  assert.equal(blended.abilityHint, "jump");
});

test("LOS check: clear arena = always has line of sight", () => {
  const bot = makeBot();
  bot.gameState = { rayDistanceToObstacle: () => 100 }; // nothing blocks ray
  bot.positionArray = [0, 0, 0];
  const enemy = { position: { x: 10, y: 0, z: 0 } };
  assert.equal(bot._hasLineOfSight(enemy), true);
});

test("LOS check: obstacle between us and enemy = blocked", () => {
  const bot = makeBot();
  // Wall between us and enemy: ray reaches only 5m before hitting an obstacle.
  bot.gameState = { rayDistanceToObstacle: () => 5 };
  bot.positionArray = [0, 0, 0];
  const enemy = { position: { x: 15, y: 0, z: 0 } };
  assert.equal(bot._hasLineOfSight(enemy), false);
});

test("LOS check: very close enemy always allowed (avoids edge-case denials)", () => {
  const bot = makeBot();
  bot.gameState = { rayDistanceToObstacle: () => 0 };
  bot.positionArray = [0, 0, 0];
  const enemy = { position: { x: 0.3, y: 0, z: 0 } }; // < 0.5m
  assert.equal(bot._hasLineOfSight(enemy), true);
});

test("stuck escape: no escape when bot is moving normally", () => {
  const bot = makeBot();
  const t0 = Date.now();
  // Simulate clean movement from (0,0,0) to (5,0,5) over 3 seconds.
  for (let i = 0; i < 8; i += 1) {
    bot.positionArray = [i * 0.7, 0, i * 0.7];
    const escape = bot._maybeEscapeStuckCorner();
    assert.equal(escape, null, `tick ${i} should not trigger escape`);
  }
  // After moving 3.5 m, no escape should be active.
  assert.equal(bot._maybeEscapeStuckCorner(), null);
  void t0;
});

test("stuck escape: triggers when bot has barely moved for >3s", () => {
  const bot = makeBot();
  const realNow = Date.now;
  let now = 1_000_000;
  Date.now = () => now;
  try {
    bot._lastObstacleRays = [1, 1, 1, 1, 1, 1, 1, 1];
    bot.positionArray = [10, 0, 10];
    // Six samples 600ms apart = 3.0s span, which is exactly the threshold —
    // capture each return so we can find the tick that fires escape.
    let firedAt = -1;
    let escape = null;
    for (let i = 0; i < 8; i += 1) {
      const result = bot._maybeEscapeStuckCorner();
      if (result && firedAt === -1) {
        firedAt = i;
        escape = result;
      }
      now += 600;
    }
    assert.ok(escape, "escape vector should fire on one of the ticks");
    assert.ok(firedAt >= 4 && firedAt <= 7, `escape should fire mid-window, got tick ${firedAt}`);
    assert.ok(Math.hypot(escape.x, escape.z) > 0.5, "escape vector has magnitude");
  } finally {
    Date.now = realNow;
  }
});

test("stuck escape: picks the most-clear ray as escape direction", () => {
  const bot = makeBot();
  const realNow = Date.now;
  let now = 2_000_000;
  Date.now = () => now;
  try {
    // North (ray 2) is the only fully clear direction; everything else blocked.
    bot._lastObstacleRays = [0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05];
    bot.positionArray = [0, 0, 0];
    let escape = null;
    for (let i = 0; i < 10; i += 1) {
      const result = bot._maybeEscapeStuckCorner();
      if (result && !escape) escape = result;
      now += 600;
    }
    assert.ok(escape, "escape should fire");
    // Ray 2 is +Z (north): unit vector (0, 1).
    assert.ok(Math.abs(escape.z - 1.0) < 0.1, `expected dirZ ~1, got ${escape.z}`);
    assert.ok(Math.abs(escape.x) < 0.1, `expected dirX ~0, got ${escape.x}`);
  } finally {
    Date.now = realNow;
  }
});

test("policy blend alpha: clamps out-of-range values from rewardConfig", () => {
  const bot = makeBot();
  // No brain wired = falls back to default 0.1.
  assert.equal(bot._getPolicyBlendAlpha(), 0.1);

  // Mock brain that returns various values.
  bot.brain = { getPolicyBlendAlpha: () => 0.3 };
  assert.equal(bot._getPolicyBlendAlpha(), 0.3);

  bot.brain = { getPolicyBlendAlpha: () => 1.5 }; // out of range
  assert.equal(bot._getPolicyBlendAlpha(), 1.0);

  bot.brain = { getPolicyBlendAlpha: () => -0.4 };
  assert.equal(bot._getPolicyBlendAlpha(), 0.0);

  bot.brain = { getPolicyBlendAlpha: () => Number.NaN };
  assert.equal(bot._getPolicyBlendAlpha(), 0.1, "NaN falls back to default");
});
