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
  const brain = new StrategicBrain("agent_3", "fake-key", undefined, { timeoutMs: 10, fallbackModel: null });
  const originalFetch = global.fetch;
  global.fetch = async (_url, { signal }) => new Promise((resolve, reject) => {
    signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
  });
  t.after(() => {
    global.fetch = originalFetch;
  });

  await assert.rejects(() => brain._callLLM("test prompt"), /timeout/i);
});

test("strategic brain falls back to secondary model on primary timeout", async (t) => {
  const brain = new StrategicBrain("agent_4", "fake-key", "primary-model:cloud", {
    timeoutMs: 50,
    fallbackModel: "fallback-model:cloud",
    reporter: {
      counts: {},
      incrementCounter(name) { this.counts[name] = (this.counts[name] || 0) + 1; },
    },
  });

  let attempts = 0;
  const originalFetch = global.fetch;
  global.fetch = async (_url, opts) => {
    attempts += 1;
    const body = JSON.parse(opts.body);
    if (body.model === "primary-model:cloud") {
      // Primary times out: never resolve, just wait for abort.
      return new Promise((_resolve, reject) => {
        opts.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
      });
    }
    // Fallback succeeds with valid JSON.
    return {
      ok: true,
      json: async () => ({ message: { content: '{"analysis":"from fallback","plan":"x","strategy":{"aggression":0.7,"accuracy_focus":0.5,"crystal_priority":0.3,"ability_usage":0.5,"retreat_threshold":0.2}}' } }),
    };
  };
  t.after(() => { global.fetch = originalFetch; });

  const text = await brain._callLLM("test prompt");
  assert.equal(attempts, 2, "primary attempt + fallback attempt");
  assert.match(text, /from fallback/);
  assert.equal(brain.reporter.counts.llmRetries, 1);
  assert.equal(brain.reporter.counts.llmTimeouts, 1);
});

test("strategic brain does NOT fall back on non-retryable 4xx", async (t) => {
  const brain = new StrategicBrain("agent_8", "fake-key", "primary-model:cloud", {
    timeoutMs: 5000,
    fallbackModel: "fallback-model:cloud",
  });

  let attempts = 0;
  const originalFetch = global.fetch;
  global.fetch = async () => {
    attempts += 1;
    return { ok: false, status: 401, json: async () => ({}) };
  };
  t.after(() => { global.fetch = originalFetch; });

  await assert.rejects(() => brain._callLLM("test prompt"), /401/);
  assert.equal(attempts, 1, "401 is non-retryable; should NOT call fallback");
});

test("strategic brain sends format:'json' to enforce JSON-only output", async (t) => {
  const brain = new StrategicBrain("agent_9", "fake-key", "test-model", { timeoutMs: 5000 });
  let bodySeen = null;
  const originalFetch = global.fetch;
  global.fetch = async (_url, opts) => {
    bodySeen = JSON.parse(opts.body);
    return { ok: true, json: async () => ({ message: { content: "{}" } }) };
  };
  t.after(() => { global.fetch = originalFetch; });

  await brain._callLLMOnce("test-model", "prompt");
  assert.equal(bodySeen.format, "json", "Ollama format:'json' must be set so output is constrained");
  assert.equal(bodySeen.model, "test-model");
});

test("strategic brain default models point to the new fast variants", () => {
  const brain = new StrategicBrain("agent_10", "fake-key");
  assert.equal(brain.model, "kimi-k2.6:cloud");
  assert.equal(brain.fallbackModel, "deepseek-v4-flash:cloud");
});
