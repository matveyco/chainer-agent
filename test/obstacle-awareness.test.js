const test = require("node:test");
const assert = require("node:assert/strict");

const { GameState } = require("../src/game/GameState");
const { StateExtractor } = require("../src/bot/StateExtractor");

test("game state ray to obstacle returns max distance when arena is empty", () => {
  const gs = new GameState();
  const distance = gs.rayDistanceToObstacle({ x: 0, z: 0 }, { x: 1, z: 0 }, 12);
  assert.equal(distance, 12);
});

test("game state ray hits an obstacle dead ahead", () => {
  const gs = new GameState();
  gs.setStaticObstacles([{ x: 5, y: 0, z: 0, radius: 1 }]);
  // Ray east from origin hits the obstacle at distance 5 - radius = 4.
  const distance = gs.rayDistanceToObstacle({ x: 0, z: 0 }, { x: 1, z: 0 }, 12);
  assert.ok(Math.abs(distance - 4) < 0.001, `expected ~4, got ${distance}`);
});

test("game state ray ignores obstacle behind origin", () => {
  const gs = new GameState();
  gs.setStaticObstacles([{ x: -5, y: 0, z: 0, radius: 1 }]);
  // Ray east from origin — the obstacle is to the west.
  const distance = gs.rayDistanceToObstacle({ x: 0, z: 0 }, { x: 1, z: 0 }, 12);
  assert.equal(distance, 12);
});

test("game state setStaticObstacles normalises scale into a radius", () => {
  const gs = new GameState();
  gs.setStaticObstacles([{ x: 1, y: 0, z: 1, scale: [4, 2, 6] }]);
  const obs = gs.staticObstacles[0];
  // max(|sx|, |sz|) / 2 = max(4, 6) / 2 = 3
  assert.equal(obs.radius, 3);
});

test("game state removes obstacle by id when breakable destroyed", () => {
  const gs = new GameState();
  gs.setStaticObstacles([
    { x: 1, y: 0, z: 1, radius: 1, id: "crate-a" },
    { x: 5, y: 0, z: 5, radius: 1, id: "crate-b" },
  ]);
  gs.removeStaticObstacle("crate-a");
  assert.equal(gs.staticObstacles.length, 1);
  assert.equal(gs.staticObstacles[0].id, "crate-b");
});

test("state extractor emits 32 features and the last 8 are obstacle raycasts", () => {
  const gs = new GameState();
  gs.addPlayer("me");
  gs.spatialGrid.updateClient("me", { x: 0, y: 0, z: 0 });
  gs.updateHealth("me", 80);
  // Place a wall to the east at distance 4 (after radius=1).
  gs.setStaticObstacles([{ x: 5, y: 0, z: 0, radius: 1 }]);

  const extractor = new StateExtractor();
  const v = extractor.extract(gs, "me", 20, false, { score: 100, kills: 1, deaths: 0 });

  assert.equal(v.length, 32);
  assert.equal(StateExtractor.INPUT_COUNT, 32);
  // Index 24 = east ray. Wall is at distance 4, RAYCAST_RANGE = 12 → 4/12 ≈ 0.333.
  assert.ok(v[24] > 0.3 && v[24] < 0.4, `east ray expected ~0.333, got ${v[24]}`);
  // Index 26 = north ray (positive Z) — no wall there, should be 1.0.
  assert.equal(v[26], 1.0);
});

test("state extractor reports clear arena (all 1.0) when no obstacles loaded", () => {
  const gs = new GameState();
  gs.addPlayer("me");
  gs.spatialGrid.updateClient("me", { x: 0, y: 0, z: 0 });
  gs.updateHealth("me", 100);

  const extractor = new StateExtractor();
  const v = extractor.extract(gs, "me", 20, false, {});

  for (let i = 24; i < 32; i += 1) {
    assert.equal(v[i], 1.0, `ray ${i} should be 1.0 (no obstacles), got ${v[i]}`);
  }
});
