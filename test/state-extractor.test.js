const test = require("node:test");
const assert = require("node:assert/strict");

const { StateExtractor } = require("../src/bot/StateExtractor");

function makeGameState() {
  return {
    getHealth() { return 75; },
    getPosition() { return { x: 12, y: 0, z: -6 }; },
    getNearbyEnemies() {
      return [{
        id: "enemy_1",
        distance: 10,
        position: { x: 18, y: 0, z: -6 },
      }];
    },
    getHealth(id) {
      return id === "enemy_1" ? 50 : 75;
    },
    getClosestCrystal() {
      return { x: 16, y: 0, z: -2, distance: 8 };
    },
  };
}

test("state extractor emits the full 24-feature vector", () => {
  const extractor = new StateExtractor();
  extractor.setLastMove(3.5, -1.75);
  extractor.setMatchTime(45, 90);
  extractor.setRecentCombat(20, 5);

  const vector = extractor.extract(
    makeGameState(),
    "agent_0",
    20,
    true,
    {
      score: 120,
      kills: 3,
      deaths: 1,
      abilities: new Map([
        ["jump", { ready: true }],
        ["minePlanting", { ready: false }],
      ]),
    }
  );

  assert.equal(vector.length, 24);
  assert.ok(vector[18] >= 0 && vector[18] <= 1);
  assert.ok(Number.isFinite(vector[19]));
  assert.ok(Number.isFinite(vector[20]));
  assert.ok(vector[21] > 0 && vector[21] < 1);
  assert.ok(vector[22] > 0);
  assert.ok(vector[23] > 0);
});
