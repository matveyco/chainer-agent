const test = require("node:test");
const assert = require("node:assert/strict");

const {
  resolveStableModelAlias,
  shouldStageChallenger,
} = require("../src/runtime/Automation");

test("stable track falls back to champion when candidate is stale", () => {
  assert.equal(resolveStableModelAlias({
    aliases: { candidate: 12, champion: 7 },
    last_eval_report: { passed: false, candidate_version: 12 },
  }, {
    training: { stableFallbackAlias: "champion" },
  }), "champion");

  assert.equal(resolveStableModelAlias({
    aliases: { candidate: 12, champion: 7 },
    last_eval_report: { passed: true, candidate_version: 12 },
  }, {
    training: { stableFallbackAlias: "champion" },
  }), "candidate");
});

test("challenger staging requires healthy recent live metrics", () => {
  const verdict = shouldStageChallenger({
    familyStatus: {
      aliases: { latest: 2500, challenger: 1000, candidate: 900, champion: 800 },
      last_eval_report: { candidate_version: 1000, challenger_version: 1000, passed: false },
    },
    recentMatches: [
      {
        mode: "training",
        track: "training",
        hasCombatSignal: true,
        fillRatio: 1,
        totalShotsFired: 30,
        totalDecisionsMade: 300,
        totalTacticalOverrides: 100,
        totalDamageDealt: 120,
      },
      {
        mode: "training",
        track: "training",
        hasCombatSignal: true,
        fillRatio: 1,
        totalShotsFired: 25,
        totalDecisionsMade: 260,
        totalTacticalOverrides: 80,
        totalDamageDealt: 100,
      },
      {
        mode: "training",
        track: "training",
        hasCombatSignal: true,
        fillRatio: 1,
        totalShotsFired: 28,
        totalDecisionsMade: 250,
        totalTacticalOverrides: 75,
        totalDamageDealt: 110,
      },
      {
        mode: "training",
        track: "training",
        hasCombatSignal: true,
        fillRatio: 1,
        totalShotsFired: 32,
        totalDecisionsMade: 280,
        totalTacticalOverrides: 90,
        totalDamageDealt: 130,
      },
    ],
    counters: { joinAttempts: 100, joinSuccesses: 99 },
    config: {
      evaluation: {
        autoStageMinVersionDelta: 1000,
        stagingRecentMatches: 4,
        stagingMinCombatSignalRatio: 0.75,
        stagingMinFillRatio: 0.95,
        stagingMinJoinSuccessRate: 0.97,
        stagingMinShotRate: 0.03,
        stagingMinPolicyShare: 0.1,
        stagingMinDamagePerShot: 0.25,
      },
    },
  });

  assert.equal(verdict.ok, true);
  assert.equal(verdict.metrics.latestVersion, 2500);
});

test("staging gate: configured threshold of 0 must NOT fall back to default (the || → ?? fix)", () => {
  // Reproduces the production bug: hybrid bots reported policyShare=0.
  // With config stagingMinPolicyShare=0.0 the gate USED TO compute
  // 0 || 0.1 → 0.1 in JS and rejected as policy_share_low. Fixed to ??.
  const verdict = shouldStageChallenger({
    familyStatus: {
      aliases: { latest: 2500, challenger: 1000, candidate: 900, champion: 800 },
      last_eval_report: { candidate_version: 1000, challenger_version: 1000, passed: false },
    },
    recentMatches: [
      {
        mode: "training", track: "training", hasCombatSignal: true, fillRatio: 1,
        totalShotsFired: 30, totalDecisionsMade: 300, totalTacticalOverrides: 100,
        totalDamageDealt: 120,
        // CRITICAL: policyShare = (decisions - overrides) / decisions = 0
        // (because tactical drives every decision in the hybrid).
        policyShare: 0,
      },
    ],
    counters: { joinAttempts: 100, joinSuccesses: 99 },
    config: {
      evaluation: {
        autoStageMinVersionDelta: 1000,
        stagingRecentMatches: 1,
        stagingMinCombatSignalRatio: 0.75,
        stagingMinFillRatio: 0.95,
        stagingMinJoinSuccessRate: 0.97,
        stagingMinShotRate: 0.01,
        stagingMinPolicyShare: 0.0, // <- the magic value that was being silently overridden
        stagingMinDamagePerShot: 0.25,
      },
    },
  });
  assert.notEqual(verdict.reason, "policy_share_low",
    `staging at 0.0 must accept policyShare=0; got reason=${verdict.reason}`);
  assert.equal(verdict.ok, true);
});
