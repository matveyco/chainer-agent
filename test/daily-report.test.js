const test = require("node:test");
const assert = require("node:assert/strict");

const {
  aggregatePerAgent,
  aggregatePerArchetype,
  summarizeLLM,
  renderMarkdown,
  lastMatchRewardBreakdown,
} = require("../scripts/daily_report");

test("aggregatePerAgent sums score, kills, deaths and computes win rate / K-D", () => {
  const matches = [
    {
      agentResults: [
        { agentId: "agent_0", displayName: "chnr_1", archetypeId: "hunter", score: 4000, kills: 8, deaths: 4, shotsFired: 200, crystalsCollectedApprox: 5, rank: 1 },
        { agentId: "agent_1", displayName: "chnr_2", archetypeId: "sniper", score: 1500, kills: 2, deaths: 6, shotsFired: 80, crystalsCollectedApprox: 2, rank: 5 },
      ],
    },
    {
      agentResults: [
        { agentId: "agent_0", displayName: "chnr_1", archetypeId: "hunter", score: 6000, kills: 12, deaths: 6, shotsFired: 300, crystalsCollectedApprox: 7, rank: 1 },
        { agentId: "agent_1", displayName: "chnr_2", archetypeId: "sniper", score: 2500, kills: 4, deaths: 5, shotsFired: 120, crystalsCollectedApprox: 4, rank: 4 },
      ],
    },
  ];
  const rows = aggregatePerAgent(matches);
  assert.equal(rows[0].agentId, "agent_0");
  assert.equal(rows[0].matches, 2);
  assert.equal(rows[0].avgScore, 5000);
  assert.equal(rows[0].topScore, 6000);
  assert.equal(rows[0].wins, 2);
  assert.equal(rows[0].winRate, 1.0);
  assert.equal(rows[0].avgKD, +(20 / 10).toFixed(2));
  assert.equal(rows[0].crystalsPerMatch, 6.0);

  assert.equal(rows[1].agentId, "agent_1");
  assert.equal(rows[1].wins, 0);
});

test("aggregatePerArchetype averages across agents per archetype", () => {
  const perAgent = [
    { agentId: "a", archetypeId: "hunter", avgScore: 5000, winRate: 1.0 },
    { agentId: "b", archetypeId: "hunter", avgScore: 3000, winRate: 0.5 },
    { agentId: "c", archetypeId: "sniper", avgScore: 2000, winRate: 0.0 },
  ];
  const rows = aggregatePerArchetype(perAgent);
  const hunter = rows.find((r) => r.archetype === "hunter");
  assert.equal(hunter.bots, 2);
  assert.equal(hunter.avgScore, 4000);
  assert.equal(hunter.winRate, 0.75);
  const sniper = rows.find((r) => r.archetype === "sniper");
  assert.equal(sniper.avgScore, 2000);
});

test("summarizeLLM detects strategy drift vs archetype defaults", () => {
  const profiles = {
    agent_0: {
      personality: { archetype: "hunter" },
      strategy: { aggression: 0.95, accuracy_focus: 0.4, crystal_priority: 0.1, ability_usage: 0.7, retreat_threshold: 0.1 },
      lifetime: { matches: 10 },
      thought_log: [{ plan: "Hunt aggressively", timestamp: "2026-04-23T12:00:00Z" }],
    },
    agent_1: {
      personality: { archetype: "sniper" },
      strategy: { aggression: 0.2, accuracy_focus: 0.95, crystal_priority: 0.2, ability_usage: 0.3, retreat_threshold: 0.4 },
      lifetime: { matches: 5 },
      thought_log: [],
    },
  };
  const rows = summarizeLLM(profiles);
  // Hunter's aggression mutated 0.9 -> 0.95 (drift 0.05/5 = 0.01 avg). Sniper unchanged.
  const hunter = rows.find((r) => r.agentId === "agent_0");
  const sniper = rows.find((r) => r.agentId === "agent_1");
  assert.ok(hunter.avgStrategyDrift > 0, "hunter shows drift");
  assert.equal(sniper.avgStrategyDrift, 0, "sniper at archetype defaults shows no drift");
  assert.equal(hunter.lastPlan, "Hunt aggressively");
  assert.equal(sniper.lastPlan, null);
});

test("lastMatchRewardBreakdown picks the last match's per-agent reward signals", () => {
  const matches = [
    {
      finishedAt: "2026-04-24T10:00:00Z",
      agentResults: [
        {
          agentId: "agent_0",
          displayName: "chnr_1",
          archetypeId: "hunter",
          rank: 1,
          score: 5000,
          rewardTotals: { kills: 12, win: 50, scoreDelta: 2.5, deaths: -3.0 },
        },
        {
          agentId: "agent_1",
          displayName: "chnr_2",
          archetypeId: "collector",
          rank: 4,
          score: 2200,
          rewardTotals: { crystalPickup: 8.5, scoreDelta: 1.2, kills: 1.0 },
        },
        {
          agentId: "agent_2",
          displayName: "chnr_3",
          archetypeId: "berserker",
          rank: 12,
          score: 100,
          rewardTotals: { lastPlace: -20, deaths: -7.5, kills: 4 },
        },
      ],
    },
  ];
  const breakdown = lastMatchRewardBreakdown(matches);
  assert.ok(breakdown);
  assert.equal(breakdown.finishedAt, "2026-04-24T10:00:00Z");
  // Sorted by rank.
  assert.equal(breakdown.rows[0].rank, 1);
  assert.equal(breakdown.rows[2].rank, 12);
  // Top signals — sorted by absolute contribution.
  assert.match(breakdown.rows[0].topSignals, /win=50/, "winner has win=50 in top signals");
  assert.match(breakdown.rows[1].topSignals, /crystalPickup=8\.5/, "collector's signal is crystalPickup");
  assert.match(breakdown.rows[2].topSignals, /lastPlace=-20/, "last-place stings");
});

test("lastMatchRewardBreakdown returns null when no matches have rewardTotals", () => {
  assert.equal(lastMatchRewardBreakdown([]), null);
  assert.equal(
    lastMatchRewardBreakdown([{ finishedAt: "x", agentResults: [{ agentId: "a", score: 100 }] }]),
    null
  );
});

test("renderMarkdown produces a non-empty markdown report", () => {
  const report = {
    generatedAt: "2026-04-23T15:00:00Z",
    hours: 24,
    matchCount: 3,
    perAgent: [
      { agentId: "a", displayName: "chnr_1", archetypeId: "hunter", matches: 3, avgScore: 4500, topScore: 7000, winRate: 0.66, avgKD: 1.8, crystalsPerMatch: 5.2 },
    ],
    perArchetype: [{ archetype: "hunter", bots: 1, avgScore: 4500, winRate: 0.66 }],
    alphaDrift: [
      { agentId: "a", displayName: "chnr_1", archetype: "hunter", alpha: 0.12, delta: 0.02, generation: 1 },
    ],
    llmRows: [
      { agentId: "a", archetype: "hunter", lifetimeMatches: 3, thoughts: 1, avgStrategyDrift: 0.05, lastPlan: "Push center" },
    ],
    pbtCycles: [
      { timestamp: "2026-04-23T14:00:00Z", cohortSize: 3, bestFitness: 1.2, bestAgent: "a", pairs: ["b<-a(g1)"] },
    ],
    counters: { chainer_shotsFired: 400, chainer_crystalPickupsApprox: 100 },
    trainerState: { family: "arena-main", aliases: { latest: 5400, candidate: 1, champion: 1 }, trainSteps: 5400, boundAgents: 12 },
  };
  const md = renderMarkdown(report);
  assert.ok(md.includes("Chainer Bots — Daily Report"));
  assert.ok(md.includes("chnr_1"));
  assert.ok(md.includes("hunter"));
  assert.ok(md.includes("policyBlendAlpha"));
  assert.ok(md.includes("Push center"));
  assert.ok(md.includes("latest=5400"));
});
