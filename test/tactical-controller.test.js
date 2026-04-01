const test = require("node:test");
const assert = require("node:assert/strict");

const { TacticalController } = require("../src/bot/TacticalController");

test("tactical controller forces engagement when a nearby enemy meets a passive policy", () => {
  const controller = new TacticalController({ archetypeId: "berserker", seed: 4 });
  const result = controller.stabilize(
    {
      moveX: 0,
      moveZ: 0,
      aimOffsetX: 2,
      aimOffsetZ: -2,
      shouldShoot: false,
      shouldUseAbility: false,
    },
    {
      enemy: { distance: 8, dirX: 1, dirZ: 0 },
      strategy: { aggression: 0.9, accuracy_focus: 0.2, ability_usage: 0.9, retreat_threshold: 0.05 },
      healthPercent: 0.9,
      weaponRange: 20,
      cooldownReady: true,
      abilityReady: true,
      passiveMs: 9000,
      position: { x: 0, z: 0 },
      distanceFromCenter: 0,
    }
  );

  assert.equal(result.overridden, true);
  assert.equal(result.action.shouldShoot, true);
  assert.equal(Math.hypot(result.action.moveX, result.action.moveZ) > 0.5, true);
  assert.equal(result.reasons.includes("shoot_enemy"), true);
});

test("tactical controller routes passive bots toward crystals when combat is absent", () => {
  const controller = new TacticalController({ archetypeId: "collector", seed: 2 });
  const result = controller.stabilize(
    {
      moveX: 0,
      moveZ: 0,
      aimOffsetX: 0,
      aimOffsetZ: 0,
      shouldShoot: false,
      shouldUseAbility: false,
    },
    {
      crystal: { dirX: 0.6, dirZ: -0.8, distance: 6 },
      strategy: { crystal_priority: 0.95 },
      passiveMs: 8000,
      position: { x: 4, z: -3 },
      distanceFromCenter: 5,
    }
  );

  assert.equal(result.overridden, true);
  assert.equal(result.reasons.includes("seek_crystal"), true);
  assert.equal(result.action.moveX > 0, true);
  assert.equal(result.action.moveZ < 0, true);
});
