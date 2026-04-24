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

test("agent brain crystalPickup reward fires per crystal delta", () => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555", {
    rewardConfig: { crystalPickupBonus: 0.5 },
  });
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);
  brain.recordStep({ done: false, crystalDelta: 3, survivalSeconds: 0.05 });
  assert.equal(brain.experienceBuffer[0].reward_components.crystalPickup, 1.5);
});

test("agent brain firstBlood fires once per match", () => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555", {
    rewardConfig: { firstBloodBonus: 3.0, killBonus: 1.0, streakBonus: 0.3 },
  });
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);
  brain.recordStep({ done: false, gotKill: true, gotFirstBlood: true, currentStreak: 1 });
  assert.equal(brain.experienceBuffer[0].reward_components.firstBlood, 3.0);
  assert.equal(brain.experienceBuffer[0].reward_components.streak, 0.3);
});

test("agent brain streak reward scales with current streak length", () => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555", {
    rewardConfig: { streakBonus: 0.3 },
  });
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);
  brain.recordStep({ done: false, gotKill: true, currentStreak: 5 });
  assert.equal(brain.experienceBuffer[0].reward_components.streak, 1.5); // 5 * 0.3
});

test("agent brain outnumberedSurvival fires when alive with 3+ enemies in close range", () => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555", {
    rewardConfig: { outnumberedSurvivalBonus: 0.05, survivalWeight: 0 },
  });
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);
  brain.recordStep({ done: false, survivalSeconds: 2, nearbyEnemyCount: 4, died: false });
  assert.ok(Math.abs(brain.experienceBuffer[0].reward_components.outnumberedSurvival - 0.1) < 1e-6);

  // Now with only 2 nearby enemies — bonus should NOT fire.
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);
  brain.recordStep({ done: false, survivalSeconds: 2, nearbyEnemyCount: 2, died: false });
  assert.equal(brain.experienceBuffer[1].reward_components.outnumberedSurvival, 0);
});

test("agent brain wallShot penalty fires per LOS-vetoed shot", () => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555", {
    rewardConfig: { wallShotPenalty: -0.1 },
  });
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);
  brain.recordStep({ done: false, wallShotsRecent: 4 });
  assert.ok(Math.abs(brain.experienceBuffer[0].reward_components.wallShot + 0.4) < 1e-6);
});

test("agent brain win bonus fires only on rank=1 terminal step", () => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555", {
    rewardConfig: { winBonus: 50.0, matchRankBonus: 25.0, lastPlacePenalty: -20.0 },
  });
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);

  // Winner: gets winBonus.
  brain.recordStep({ done: true, rank: 1, roomSize: 12 });
  assert.equal(brain.experienceBuffer[0].reward_components.win, 50.0);
  assert.equal(brain.experienceBuffer[0].reward_components.lastPlace, 0);

  // Mid-pack: no winBonus, no lastPlacePenalty.
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);
  brain.recordStep({ done: true, rank: 6, roomSize: 12 });
  assert.equal(brain.experienceBuffer[1].reward_components.win, 0);
  assert.equal(brain.experienceBuffer[1].reward_components.lastPlace, 0);

  // Last place: no winBonus, gets lastPlacePenalty.
  brain.lastState = Array.from({ length: STATE_DIM }, () => 0);
  brain.lastAction = Array.from({ length: ACTION_DIM }, () => 0);
  brain.recordStep({ done: true, rank: 12, roomSize: 12 });
  assert.equal(brain.experienceBuffer[2].reward_components.win, 0);
  assert.equal(brain.experienceBuffer[2].reward_components.lastPlace, -20.0);
});

test("agent brain getEpisodeRewardTotals returns rounded shallow snapshot", () => {
  const brain = new AgentBrain("agent_test", "http://localhost:5555");
  brain.episodeRewardTotals = { kills: 1.123456, scoreDelta: 0.0001, win: 50 };
  const snap = brain.getEpisodeRewardTotals();
  assert.deepEqual(snap, { kills: 1.123, scoreDelta: 0, win: 50 });
  // Mutating the snapshot must not affect the brain's state.
  snap.kills = 999;
  assert.equal(brain.episodeRewardTotals.kills, 1.123456);
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
