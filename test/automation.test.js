const test = require("node:test");
const assert = require("node:assert/strict");

const {
  computeCombatSignalRatio,
  selectSafeRecoveryVersion,
  shouldQueueAutomaticEvaluation,
} = require("../src/runtime/Automation");

test("automation computes combat signal ratio from recent matches", () => {
  const ratio = computeCombatSignalRatio([
    { hasCombatSignal: true },
    { totalDamageDealt: 12 },
    { totalKills: 0, totalDeaths: 0, totalScore: 0 },
    { totalScore: 50 },
  ]);

  assert.equal(ratio, 0.75);
});

test("automation prefers latest passed evaluation version for recovery", () => {
  const version = selectSafeRecoveryVersion({
    aliases: { champion: 194 },
    evaluation_history: [
      { candidate_version: 1622, passed: true },
      { candidate_version: 22930, passed: false },
      { candidate_version: 13589, passed: true },
    ],
  });

  assert.equal(version, 13589);
});

test("automation falls back to champion version when no passed evaluation exists", () => {
  const version = selectSafeRecoveryVersion({
    aliases: { champion: 194 },
    evaluation_history: [{ candidate_version: 2000, passed: false }],
  });

  assert.equal(version, 194);
});

test("automation queues evaluation only for a new latest snapshot at the interval", () => {
  assert.equal(shouldQueueAutomaticEvaluation({
    totalMatches: 10,
    selectionInterval: 10,
    lastQueuedMatchCount: 8,
    latestVersion: 150,
    candidateVersion: 100,
    hasCurrentJob: false,
    queuedJobs: 0,
  }), true);

  assert.equal(shouldQueueAutomaticEvaluation({
    totalMatches: 10,
    selectionInterval: 10,
    lastQueuedMatchCount: 10,
    latestVersion: 150,
    candidateVersion: 100,
    hasCurrentJob: false,
    queuedJobs: 0,
  }), false);

  assert.equal(shouldQueueAutomaticEvaluation({
    totalMatches: 10,
    selectionInterval: 10,
    lastQueuedMatchCount: 0,
    latestVersion: 150,
    candidateVersion: 150,
    hasCurrentJob: false,
    queuedJobs: 0,
  }), false);
});
