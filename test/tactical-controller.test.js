const test = require("node:test");
const assert = require("node:assert/strict");

const { TacticalController } = require("../src/bot/TacticalController");

test("tactical controller engages and shoots a nearby aggressive enemy", () => {
  const controller = new TacticalController({ archetypeId: "berserker", seed: 4 });
  const result = controller.decide({
    enemy: { distance: 8, dirX: 1, dirZ: 0 },
    strategy: { aggression: 0.9, accuracy_focus: 0.2, ability_usage: 0.9, retreat_threshold: 0.05 },
    healthPercent: 0.9,
    weaponRange: 20,
    cooldownReady: true,
    abilityReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  });

  assert.equal(result.overridden, true);
  assert.equal(result.action.shouldShoot, true);
  assert.equal(Math.hypot(result.action.moveX, result.action.moveZ) > 0.5, true);
  assert.equal(result.reasons.includes("shoot_enemy"), true);
  assert.equal(result.reasons.includes("engage_enemy"), true);
});

test("tactical controller routes toward crystals when combat is absent", () => {
  const controller = new TacticalController({ archetypeId: "collector", seed: 2 });
  const result = controller.decide({
    crystal: { dirX: 0.6, dirZ: -0.8, distance: 6 },
    strategy: { crystal_priority: 0.95 },
    position: { x: 4, z: -3 },
    distanceFromCenter: 5,
  });

  assert.equal(result.overridden, true);
  assert.equal(result.reasons.includes("seek_crystal"), true);
  assert.equal(result.action.moveX > 0, true);
  assert.equal(result.action.moveZ < 0, true);
});

test("tactical controller retreats and refuses to shoot at low health", () => {
  const controller = new TacticalController({ archetypeId: "guardian", seed: 1 });
  const result = controller.decide({
    enemy: { distance: 6, dirX: 1, dirZ: 0 },
    strategy: { aggression: 0.6, retreat_threshold: 0.4, ability_usage: 0.5 },
    healthPercent: 0.2, // well below retreat_threshold
    weaponRange: 20,
    cooldownReady: true,
    abilityReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  });

  assert.equal(result.reasons.includes("retreat_enemy"), true);
  assert.equal(result.action.shouldShoot, false);
  // Retreat = move opposite the enemy direction, so moveX should be negative.
  assert.ok(result.action.moveX < 0, `expected retreat west, got moveX=${result.action.moveX}`);
});

test("tactical controller closes the gap when the enemy is out of range", () => {
  const controller = new TacticalController({ archetypeId: "hunter", seed: 0 });
  const result = controller.decide({
    enemy: { distance: 40, dirX: 1, dirZ: 0 }, // way past engageRange of 23
    strategy: { aggression: 0.6 },
    healthPercent: 1,
    weaponRange: 20,
    cooldownReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  });

  assert.equal(result.reasons.includes("close_gap"), true);
  assert.equal(result.action.shouldShoot, false); // distance > engageRange*1.05
  assert.ok(result.action.moveX > 0.5, "should move east toward enemy");
});

test("tactical controller returns to center when out of bounds", () => {
  const controller = new TacticalController({ archetypeId: "tactician", seed: 3, safeSize: 50 });
  const result = controller.decide({
    position: { x: 60, z: 0 }, // > 50 * 0.92 = 46
    distanceFromCenter: 60,
  });

  assert.equal(result.reasons.includes("return_to_center"), true);
  assert.ok(result.action.moveX < 0, "should head back west toward (0, 0)");
});

test("obstacle avoidance leaves clear directions untouched", () => {
  const controller = new TacticalController({ seed: 0 });
  // All rays clear (1.0).
  const rays = [1, 1, 1, 1, 1, 1, 1, 1];
  const result = controller._avoidObstacles(1, 0, rays);
  assert.equal(result.adjusted, false);
  assert.ok(Math.abs(result.x - 1) < 1e-6);
  assert.ok(Math.abs(result.z) < 1e-6);
});

test("obstacle avoidance reroutes when forward is blocked", () => {
  const controller = new TacticalController({ seed: 0 });
  // Wall straight east (ray 0 = blocked), everything else clear.
  const rays = [0.05, 1, 1, 1, 1, 1, 1, 1];
  const result = controller._avoidObstacles(1, 0, rays);
  assert.equal(result.adjusted, true);
  // Should pick a flanking clear direction (NE or SE), not push into the wall.
  assert.ok(Math.abs(result.x) < 0.99 || result.x < 0, "should not still go due east");
});

test("decide() picks a clear path when its first choice would walk into a wall", () => {
  const controller = new TacticalController({ archetypeId: "hunter", seed: 0 });
  // Enemy is far east → close_gap mode (mostly straight-east desired vector).
  // Ray 0 (E) and ray 1 (NE) blocked; ray 7 (SE) and ray 6 (S) are clear.
  const rays = [0.02, 0.05, 1, 1, 1, 1, 1, 1];
  const result = controller.decide({
    enemy: { distance: 40, dirX: 1, dirZ: 0 },
    strategy: { aggression: 0.7 },
    healthPercent: 1,
    weaponRange: 20,
    cooldownReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
    obstacleRays: rays,
  });

  assert.equal(result.reasons.includes("avoid_obstacle"), true);
  // It should pick SE (ray 7: x≈0.707, z≈-0.707) as the closest clear flank.
  assert.ok(result.action.moveZ < -0.4, `expected reroute south, got moveZ=${result.action.moveZ}`);
});

test("stabilize() is a backwards-compatible alias for decide()", () => {
  const controller = new TacticalController({ archetypeId: "hunter", seed: 0 });
  const ctx = {
    enemy: { distance: 10, dirX: 1, dirZ: 0 },
    strategy: { aggression: 0.7 },
    healthPercent: 1,
    weaponRange: 20,
    cooldownReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  };
  const direct = controller.decide(ctx);
  const wrapped = controller.stabilize({ moveX: 0, moveZ: 0 }, ctx);
  assert.deepEqual(wrapped.action, direct.action);
  assert.deepEqual(wrapped.reasons, direct.reasons);
});

test("Tactician (aggression=0.5) now engages — lower aggression floor 0.55 -> 0.35", () => {
  const controller = new TacticalController({ archetypeId: "tactician", seed: 0 });
  const result = controller.decide({
    enemy: { distance: 12, dirX: 1, dirZ: 0 },
    strategy: { aggression: 0.5, accuracy_focus: 0.6, ability_usage: 0.5, retreat_threshold: 0.35 },
    healthPercent: 1,
    weaponRange: 20,
    cooldownReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  });

  // Pre-fix this archetype was strafe-only at aggression < 0.55. Now it engages.
  assert.equal(result.reasons.includes("engage_enemy"), true,
    `expected engage_enemy, got ${result.reasons.join(",")}`);
  assert.equal(result.action.shouldShoot, true);
});

test("Sniper (aggression=0.2) now shoots beyond closeRange — shoot floor 0.3 -> 0.15", () => {
  const controller = new TacticalController({ archetypeId: "sniper", seed: 0 });
  const result = controller.decide({
    enemy: { distance: 18, dirX: 1, dirZ: 0 }, // > closeRange (10), within engageRange (23)
    strategy: { aggression: 0.2, accuracy_focus: 0.95 },
    healthPercent: 1,
    weaponRange: 20,
    cooldownReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  });

  // Pre-fix Sniper would not shoot at d=18 (aggression=0.2 < 0.3 AND d > 10).
  assert.equal(result.action.shouldShoot, true,
    `Sniper should now shoot at d=18, reasons: ${result.reasons.join(",")}`);
});

test("Sniper (crystal_priority=0.2) now seeks crystals — crystal floor 0.25 -> 0.05", () => {
  const controller = new TacticalController({ archetypeId: "sniper", seed: 0 });
  const result = controller.decide({
    crystal: { dirX: 1, dirZ: 0, distance: 8 },
    strategy: { aggression: 0.2, crystal_priority: 0.2 },
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  });

  // Pre-fix Sniper roamed center; now it heads to the crystal.
  assert.equal(result.reasons.includes("seek_crystal"), true,
    `expected seek_crystal, got ${result.reasons.join(",")}`);
  assert.ok(result.action.moveX > 0.5, "should head east toward crystal");
});

test("seek_cluster fallback heads toward enemy centroid when no enemy/crystal", () => {
  const controller = new TacticalController({ archetypeId: "berserker", seed: 0 });
  // Berserker: crystal_priority=0, so it skips seek_crystal even with a crystal.
  // No enemy in range, no crystal — pre-fix would roam_center.
  const result = controller.decide({
    enemyClusterDir: { dirX: 0.6, dirZ: 0.8 },
    strategy: { aggression: 1.0, crystal_priority: 0.0 },
    position: { x: 10, z: -5 },
    distanceFromCenter: 11,
  });

  assert.equal(result.reasons.includes("seek_cluster"), true,
    `expected seek_cluster, got ${result.reasons.join(",")}`);
  assert.ok(result.action.moveX > 0.4, "should move toward cluster east");
  assert.ok(result.action.moveZ > 0.4, "should move toward cluster north");
});

test("ability picker: jump on out-of-range engagement", () => {
  const controller = new TacticalController({ archetypeId: "hunter", seed: 0 });
  const result = controller.decide({
    enemy: { distance: 35, dirX: 1, dirZ: 0 }, // > engageRange (23)
    strategy: { aggression: 0.9, ability_usage: 0.7 },
    healthPercent: 1,
    weaponRange: 20,
    cooldownReady: true,
    abilityReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  });

  assert.equal(result.action.shouldUseAbility, true);
  assert.equal(result.action.abilityHint, "jump");
});

test("ability picker: rampage on healthy in-range engagement", () => {
  const controller = new TacticalController({ archetypeId: "berserker", seed: 0 });
  const result = controller.decide({
    enemy: { distance: 10, dirX: 1, dirZ: 0 },
    strategy: { aggression: 1.0, ability_usage: 1.0, retreat_threshold: 0.0 },
    healthPercent: 0.9,
    weaponRange: 20,
    cooldownReady: true,
    abilityReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  });

  assert.equal(result.action.shouldUseAbility, true);
  assert.equal(result.action.abilityHint, "rampage");
});

test("ability picker: minePlanting on low-HP retreat with enemy hugging", () => {
  const controller = new TacticalController({ archetypeId: "guardian", seed: 0 });
  const result = controller.decide({
    enemy: { distance: 6, dirX: 1, dirZ: 0 },
    strategy: { aggression: 0.7, ability_usage: 0.8, retreat_threshold: 0.4 },
    healthPercent: 0.2, // low HP
    weaponRange: 20,
    cooldownReady: true,
    abilityReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
  });

  assert.equal(result.action.shouldUseAbility, true);
  assert.equal(result.action.abilityHint, "minePlanting");
});

test("cover-aware retreat: prefers direction with flank obstacles", () => {
  const controller = new TacticalController({ archetypeId: "guardian", seed: 0 });
  // Enemy due east. Direct retreat (W = ray 4) is fully open. North (ray 2)
  // is also open BUT flanked by walls at NE (ray 1) and NW (ray 3).
  // Cover heuristic should pick NORTH because of the flank coverage.
  const rays = [1, 0.05, 1, 0.05, 1, 1, 1, 1];
  const result = controller.decide({
    enemy: { distance: 8, dirX: 1, dirZ: 0 },
    strategy: { aggression: 0.5, retreat_threshold: 0.5 },
    healthPercent: 0.2,
    weaponRange: 20,
    cooldownReady: true,
    position: { x: 0, z: 0 },
    distanceFromCenter: 0,
    obstacleRays: rays,
  });

  assert.equal(result.reasons.includes("retreat_to_cover"), true,
    `expected cover retreat, got ${result.reasons.join(",")}`);
  // Either north (ray 2 = +z) or away-from-enemy direction. Just make sure
  // we're not running due east into the enemy.
  assert.ok(result.action.moveX <= 0.3, `should not run east, got moveX=${result.action.moveX}`);
});
