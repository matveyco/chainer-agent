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
      aliases: { candidate: 5, champion: 4 },
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
            { evaluationSide: "candidate", score: 600, kills: 4, deaths: 1, damageDealt: 900, survivalTime: 90 },
            { evaluationSide: "champion", score: 400, kills: 2, deaths: 2, damageDealt: 600, survivalTime: 80 },
          ],
        },
      },
    ]),
    submitReport: async () => ({ ok: true }),
  });

  assert.equal(completed.status, "completed");
  assert.equal(completed.report.passed, true);
  assert.equal(completed.report.candidate.avg_score > completed.report.champion.avg_score, true);
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
      aliases: { candidate: 5, champion: 4 },
      champion_history: [],
    }),
    runRoomBatch: async () => ([
      {
        summary: {
          roomId: "dup-room",
          connectedAgents: 2,
          agentResults: [
            { evaluationSide: "candidate", score: 0 },
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
            { evaluationSide: "candidate", score: 100 },
          ],
          hasCombatSignal: true,
        },
      },
    ]),
    submitReport: async () => ({ ok: true }),
  });

  assert.equal(completed.status, "failed");
  assert.match(completed.error, /incomplete|duplicate room id|no combat or score signal/);
  assert.equal(manager.getHistory(1)[0].status, "failed");
});
