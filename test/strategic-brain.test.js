const test = require("node:test");
const assert = require("node:assert/strict");

const { StrategicBrain } = require("../src/bot/StrategicBrain");

test("strategic brain parses bounded JSON strategy updates", () => {
  const brain = new StrategicBrain("agent_5", "fake-key");
  const parsed = brain._parseStrategy(JSON.stringify({
    analysis: "We are losing too many fights at range.",
    plan: "Take cleaner fights and collect safer crystals.",
    strategy: {
      aggression: 0.55,
      accuracy_focus: 0.8,
      crystal_priority: 0.45,
      ability_usage: 0.6,
      retreat_threshold: 0.3,
    },
  }));

  assert.equal(parsed.plan, "Take cleaner fights and collect safer crystals.");
  assert.equal(parsed.strategy.aggression, 0.55);
  assert.equal(parsed.strategy.retreat_threshold, 0.3);
});

test("strategic brain clamps invalid numeric output", () => {
  const brain = new StrategicBrain("agent_2", "fake-key");
  const parsed = brain._parseStrategy(JSON.stringify({
    analysis: "Clamp values.",
    plan: "Stay bounded.",
    strategy: {
      aggression: 5,
      accuracy_focus: -1,
      crystal_priority: 0.5,
      ability_usage: 0.25,
      retreat_threshold: 0.75,
    },
  }));

  assert.equal(parsed.strategy.aggression, 1);
  assert.equal(parsed.strategy.accuracy_focus, 0);
});

test("strategic brain starts from non-zero archetype defaults", () => {
  const brain = new StrategicBrain("agent_7", "fake-key", undefined, { archetypeId: "collector" });
  const strategy = brain.getStrategyVector();

  assert.equal(strategy.crystal_priority > 0, true);
  assert.equal(strategy.retreat_threshold > 0, true);
});

test("strategic brain times out bounded LLM calls", async (t) => {
  const brain = new StrategicBrain("agent_3", "fake-key", undefined, { timeoutMs: 10 });
  const originalFetch = global.fetch;
  global.fetch = async (_url, { signal }) => new Promise((resolve, reject) => {
    signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
  });
  t.after(() => {
    global.fetch = originalFetch;
  });

  await assert.rejects(() => brain._callLLM("test prompt"), /timeout/i);
});
