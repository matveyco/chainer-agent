const test = require("node:test");
const assert = require("node:assert/strict");

const { AgentBrain, STATE_DIM, ACTION_DIM } = require("../src/bot/AgentBrain");

test("agent brain records shaped reward components", () => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555", {
    rewardConfig: {
      scoreDeltaWeight: 0.02,
      killBonus: 1,
      deathPenalty: -0.75,
      damageWeight: 0.004,
      damageTakenPenalty: -0.002,
      survivalWeight: 0.01,
      abilityValueWeight: 0.08,
      accuracyWeight: 0.2,
      antiSuicidePenalty: -0.3,
    },
  });

  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);

  brain.recordStep({
    currentScore: 50,
    kills: 2,
    deaths: 1,
    gotKill: true,
    damageDealt: 40,
    damageTaken: 20,
    survivalSeconds: 1,
    abilityUsed: true,
    shotAccuracy: 0.5,
    done: false,
  });

  assert.equal(brain.experienceBuffer.length, 1);
  assert.equal(brain.experienceBuffer[0].reward_components.kills, 1);
  assert.ok(brain.experienceBuffer[0].reward > 0);
  assert.ok(brain.episodeRewardTotals.damage > 0);
});

test("agent brain resetMatch clears episode reward totals", () => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555");
  brain.episodeRewardTotals = { scoreDelta: 1.2 };
  brain.resetMatch();
  assert.deepEqual(brain.episodeRewardTotals, {});
});

test("agent brain reloads when trainer reports a different model version", async (t) => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555");
  brain.modelVersion = 100;
  brain.experienceBuffer = [{
    state: Array.from({ length: STATE_DIM }, () => 0),
    action: Array.from({ length: ACTION_DIM }, () => 0),
    reward: 0,
    reward_components: {},
    done: false,
  }];

  let reloads = 0;
  brain._loadModel = async () => { // eslint-disable-line no-underscore-dangle
    reloads += 1;
  };

  const originalFetch = global.fetch;
  global.fetch = async () => ({
    ok: true,
    json: async () => ({ ok: true, accepted: 1, model_version: 42 }),
  });
  t.after(() => {
    global.fetch = originalFetch;
  });

  await brain.flush();
  assert.equal(reloads, 1);
});
