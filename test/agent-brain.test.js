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

test("agent brain emits matchRank reward only on terminal step with rank info", () => {
  const brain = new AgentBrain("agent_rank", "http://localhost:5555", {
    rewardConfig: { matchRankBonus: 1.0 },
  });
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);

  // Mid-match step: no rank info → no matchRank reward.
  brain.recordStep({ currentScore: 100, kills: 1, deaths: 0, done: false });
  assert.equal(brain.experienceBuffer[0].reward_components.matchRank, 0);

  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);

  // Terminal step #1 of 12 → +1.0 bonus.
  brain.recordStep({ currentScore: 200, kills: 1, deaths: 0, done: true, rank: 1, roomSize: 12 });
  const winnerStep = brain.experienceBuffer[brain.experienceBuffer.length - 1];
  assert.equal(winnerStep.reward_components.matchRank, 1.0);

  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);

  // Terminal step last of 12 → -1.0 penalty.
  brain.recordStep({ currentScore: 5, kills: 0, deaths: 5, done: true, rank: 12, roomSize: 12 });
  const loserStep = brain.experienceBuffer[brain.experienceBuffer.length - 1];
  assert.equal(loserStep.reward_components.matchRank, -1.0);
});

test("agent brain matchRank symmetric: top quartile +0.5, bottom quartile -0.5", () => {
  const brain = new AgentBrain("agent_quartile", "http://localhost:5555", {
    rewardConfig: { matchRankBonus: 1.0 },
  });
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);

  // Rank 2 of 12 → top quartile (fraction = 1/11 ≈ 0.09, ≤ 0.25) → +0.5.
  brain.recordStep({ currentScore: 100, kills: 0, deaths: 0, done: true, rank: 2, roomSize: 12 });
  assert.equal(brain.experienceBuffer.at(-1).reward_components.matchRank, 0.5);

  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);

  // Rank 6 of 12 → middle (fraction = 5/11 ≈ 0.45) → 0.
  brain.recordStep({ currentScore: 50, kills: 0, deaths: 0, done: true, rank: 6, roomSize: 12 });
  assert.equal(brain.experienceBuffer.at(-1).reward_components.matchRank, 0);

  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);

  // Rank 11 of 12 → bottom quartile (fraction = 10/11 ≈ 0.91, ≥ 0.75) → -0.5.
  brain.recordStep({ currentScore: 10, kills: 0, deaths: 0, done: true, rank: 11, roomSize: 12 });
  assert.equal(brain.experienceBuffer.at(-1).reward_components.matchRank, -0.5);
});

test("agent brain init fetches per-agent reward weights and merges them in", async (t) => {
  const brain = new AgentBrain("agent_pbt", "http://localhost:5555", {
    rewardConfig: { scoreDeltaWeight: 0.02, killBonus: 1.0 },
  });
  const originalFetch = global.fetch;
  let fetchUrl = null;
  global.fetch = async (url) => {
    fetchUrl = String(url);
    return {
      ok: true,
      json: async () => ({
        reward_weights: { killBonus: 3.5, matchRankBonus: 50 },
      }),
    };
  };
  t.after(() => {
    global.fetch = originalFetch;
  });

  await brain._fetchRewardWeights();
  assert.equal(fetchUrl.endsWith("/agent/agent_pbt/reward-weights"), true);
  // Merge: untouched defaults stay, fetched values override.
  assert.equal(brain.rewardConfig.scoreDeltaWeight, 0.02);
  assert.equal(brain.rewardConfig.killBonus, 3.5);
  assert.equal(brain.rewardConfig.matchRankBonus, 50);
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
