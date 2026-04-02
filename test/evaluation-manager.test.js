const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const os = require("os");
const path = require("path");

const { EvaluationManager } = require("../src/runtime/EvaluationManager");

function makeRuntimeState() {
  return {
    incrementCounter() {},
    recordEvent() {},
  };
}

function makeConfig(tempDir) {
  return {
    training: {
      defaultPolicyFamily: "arena-main",
    },
    evaluation: {
      sampleMatches: 1,
      promotionMargin: 0.05,
      minWinRate: 0.55,
      historyLimit: 10,
    },
    persistence: {
      evaluationStateFile: path.join(tempDir, "evaluation_state.json"),
      evaluationHistoryFile: path.join(tempDir, "evaluation_history.jsonl"),
    },
  };
}

test("evaluation manager aggregates candidate ladder wins into a passing report", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "chainer-eval-"));
  const manager = new EvaluationManager({
    config: makeConfig(tempDir),
    runtimeState: makeRuntimeState(),
    roster: [[
      { agentId: "agent_0", modelAlias: "candidate", policyFamily: "arena-main", archetypeId: "tactician" },
      { agentId: "agent_1", modelAlias: "champion", policyFamily: "arena-main", archetypeId: "collector" },
    ]],
    trainerUrl: "http://localhost:5555",
  });

  manager.queueRun({ requestedBy: "test", candidateVersion: 5, championVersion: 4 });

  const completed = await manager.runNext({
    fetchFamilyStatus: async () => ({
      aliases: { challenger: 5, candidate: 3, champion: 4 },
      champion_history: [],
    }),
    runRoomBatch: async () => ([
      {
        summary: {
          roomId: "room-a",
          connectedAgents: 2,
          expectedAgents: 2,
          hasCombatSignal: true,
          agentResults: [
            { evaluationSide: "challenger", score: 600, kills: 4, deaths: 1, damageDealt: 900, survivalTime: 90 },
            { evaluationSide: "champion", score: 400, kills: 2, deaths: 2, damageDealt: 600, survivalTime: 80 },
          ],
        },
      },
    ]),
    submitReport: async () => ({ ok: true }),
    promoteCandidate: async () => ({ ok: true }),
  });

  assert.equal(completed.status, "completed");
  assert.equal(completed.report.passed, true);
  assert.equal(completed.report.challenger_version, 5);
  assert.equal(completed.report.candidate.avg_score > completed.report.champion.avg_score, true);
  assert.equal(completed.promotedCandidateVersion, 5);
  assert.equal(manager.getHistory(1)[0].id, completed.id);
});

test("evaluation manager fails closed on partial or duplicate evaluation rooms", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "chainer-eval-"));
  const manager = new EvaluationManager({
    config: makeConfig(tempDir),
    runtimeState: makeRuntimeState(),
    roster: [[
      { agentId: "agent_0", modelAlias: "candidate", policyFamily: "arena-main", archetypeId: "tactician" },
      { agentId: "agent_1", modelAlias: "champion", policyFamily: "arena-main", archetypeId: "collector" },
    ], [
      { agentId: "agent_2", modelAlias: "candidate", policyFamily: "arena-main", archetypeId: "hunter" },
      { agentId: "agent_3", modelAlias: "champion", policyFamily: "arena-main", archetypeId: "guardian" },
    ]],
    trainerUrl: "http://localhost:5555",
  });

  manager.queueRun({ requestedBy: "test", candidateVersion: 5, championVersion: 4 });

  const completed = await manager.runNext({
    fetchFamilyStatus: async () => ({
      aliases: { challenger: 5, candidate: 3, champion: 4 },
      champion_history: [],
    }),
    runRoomBatch: async () => ([
      {
        summary: {
          roomId: "dup-room",
          connectedAgents: 2,
          agentResults: [
            { evaluationSide: "challenger", score: 0 },
            { evaluationSide: "champion", score: 0 },
          ],
          hasCombatSignal: false,
        },
      },
      {
        summary: {
          roomId: "dup-room",
          connectedAgents: 1,
          agentResults: [
            { evaluationSide: "challenger", score: 100 },
          ],
          hasCombatSignal: true,
        },
      },
    ]),
    submitReport: async () => ({ ok: true }),
    promoteCandidate: async () => ({ ok: true }),
  });

  assert.equal(completed.status, "failed");
  assert.match(completed.error, /incomplete|duplicate room id|no combat or score signal/);
  assert.equal(manager.getHistory(1)[0].status, "failed");
});

test("evaluation manager fails stale in-flight jobs on restart", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "chainer-eval-"));
  const stateFile = path.join(tempDir, "evaluation_state.json");
  fs.writeFileSync(stateFile, JSON.stringify({
    current: {
      id: "stale-job",
      familyId: "arena-main",
      status: "running",
      requestedAt: new Date().toISOString(),
    },
    queue: [],
  }));

  const manager = new EvaluationManager({
    config: makeConfig(tempDir),
    runtimeState: makeRuntimeState(),
    roster: [[
      { agentId: "agent_0", modelAlias: "candidate", policyFamily: "arena-main", archetypeId: "tactician" },
      { agentId: "agent_1", modelAlias: "champion", policyFamily: "arena-main", archetypeId: "collector" },
    ]],
    trainerUrl: "http://localhost:5555",
  });

  assert.equal(manager.getStatus().current, null);
  assert.equal(manager.getHistory(1)[0].id, "stale-job");
  assert.equal(manager.getHistory(1)[0].status, "failed");
  assert.match(manager.getHistory(1)[0].error, /restart/);
});

test("evaluation manager exposes schedule state and keeps candidate unchanged on failed job", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "chainer-eval-"));
  const manager = new EvaluationManager({
    config: makeConfig(tempDir),
    runtimeState: makeRuntimeState(),
    roster: [[
      { agentId: "agent_0", modelAlias: "challenger", policyFamily: "arena-main", archetypeId: "tactician" },
      { agentId: "agent_1", modelAlias: "champion", policyFamily: "arena-main", archetypeId: "collector" },
    ]],
    trainerUrl: "http://localhost:5555",
  });

  manager.setScheduleState({
    nextWindowAt: "2026-04-02T12:00:00.000Z",
    stagedChallengerVersion: 12,
  });
  manager.queueRun({ requestedBy: "test", challengerVersion: 12, championVersion: 9 });

  let promoted = false;
  const completed = await manager.runNext({
    fetchFamilyStatus: async () => ({
      aliases: { challenger: 12, candidate: 7, champion: 9 },
      champion_history: [],
    }),
    runRoomBatch: async () => ([
      {
        summary: {
          roomId: "room-a",
          connectedAgents: 2,
          expectedAgents: 2,
          hasCombatSignal: true,
          agentResults: [
            { evaluationSide: "challenger", score: 300, kills: 1, deaths: 3, damageDealt: 200, survivalTime: 40 },
            { evaluationSide: "champion", score: 500, kills: 3, deaths: 1, damageDealt: 450, survivalTime: 80 },
          ],
        },
      },
    ]),
    submitReport: async () => ({ ok: true }),
    promoteCandidate: async () => {
      promoted = true;
      return { ok: true };
    },
  });

  assert.equal(completed.report.passed, false);
  assert.equal(promoted, false);
  assert.equal(manager.getStatus().schedule.activeWindow, false);
  assert.equal(manager.getStatus().schedule.stagedChallengerVersion, 12);
});
