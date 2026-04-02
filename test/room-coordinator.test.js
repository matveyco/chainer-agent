const test = require("node:test");
const assert = require("node:assert/strict");

const { RoomCoordinator } = require("../src/runtime/RoomCoordinator");

function makeRuntimeState() {
  const counters = {};
  return {
    ensureRoom() { return {}; },
    incrementCounter(name, amount = 1) { counters[name] = (counters[name] || 0) + amount; },
    updateRoom() {},
    observe() {},
    noteRoomError() {},
    get counters() { return counters; },
  };
}

test("room coordinator chooses the room with the strongest assignment group", () => {
  const runtimeState = makeRuntimeState();
  const coordinator = new RoomCoordinator({
    roomIndex: 0,
    roster: [{ agentId: "agent_0" }, { agentId: "agent_1" }, { agentId: "agent_2" }, { agentId: "agent_3" }],
    config: { rooms: { minReadyRatio: 0.5 }, bot: {}, server: {} },
    runtimeState,
    strategicBrains: new Map(),
  });

  const selection = coordinator._selectBestAssignmentGroup([
    { userID: "u0", assignment: { room: { roomId: "room-a", publicAddress: "a" } } },
    { userID: "u1", assignment: { room: { roomId: "room-a", publicAddress: "a" } } },
    { userID: "u2", assignment: { room: { roomId: "room-b", publicAddress: "b" } } },
    { userID: "u3", assignment: { room: { roomId: "room-a", publicAddress: "a" } } },
  ]);

  assert.equal(selection.roomId, "room-a");
  assert.equal(selection.sessions.length, 3);
  assert.equal(selection.sessionIds.has("u3"), true);
});

test("room coordinator skips backend rooms already claimed by another track", () => {
  const runtimeState = makeRuntimeState();
  const coordinator = new RoomCoordinator({
    roomIndex: 1,
    roster: [{ agentId: "agent_0" }, { agentId: "agent_1" }, { agentId: "agent_2" }, { agentId: "agent_3" }],
    config: { rooms: { minReadyRatio: 0.5 }, bot: {}, server: {} },
    runtimeState,
    strategicBrains: new Map(),
  });

  const selection = coordinator._selectBestAssignmentGroup([
    { userID: "u0", assignment: { room: { roomId: "room-a", publicAddress: "a" } } },
    { userID: "u1", assignment: { room: { roomId: "room-a", publicAddress: "a" } } },
    { userID: "u2", assignment: { room: { roomId: "room-b", publicAddress: "b" } } },
    { userID: "u3", assignment: { room: { roomId: "room-b", publicAddress: "b" } } },
  ], new Set(["room-a"]));

  assert.equal(selection.roomId, "room-b");
  assert.equal(selection.sessions.length, 2);
});

test("room coordinator classifies seat expiry and room lock join failures", () => {
  const runtimeState = makeRuntimeState();
  const coordinator = new RoomCoordinator({
    roomIndex: 0,
    roster: [{ agentId: "agent_0" }],
    config: { rooms: { minReadyRatio: 0.5 }, bot: {}, server: {} },
    runtimeState,
    strategicBrains: new Map(),
  });

  coordinator._classifyJoinError(new Error("User doesn't have a reserved seat.!"));
  coordinator._classifyJoinError(new Error('room "abc" is locked'));

  assert.equal(runtimeState.counters.seatExpired, 1);
  assert.equal(runtimeState.counters.lockedRooms, 1);
});

test("room coordinator summary includes activity validity signals", () => {
  const runtimeState = makeRuntimeState();
  const coordinator = new RoomCoordinator({
    roomIndex: 0,
    roster: [{ agentId: "agent_0" }, { agentId: "agent_1" }],
    config: { rooms: { minReadyRatio: 0.5 }, bot: {}, server: {} },
    runtimeState,
    strategicBrains: new Map(),
  });

  const selection = {
    roomId: "room-a",
    publicAddress: "https://arena.example",
    sessions: [{ userID: "u0" }, { userID: "u1" }],
  };
  const connectedSessions = [{ userID: "u0" }, { userID: "u1" }];
  const agentResults = [
    { score: 250, kills: 2, deaths: 1, damageDealt: 300, inputsSent: 120, stateUpdates: 40, decisionsMade: 60, shotsFired: 9, tacticalOverrides: 12 },
    { score: 100, kills: 1, deaths: 2, damageDealt: 150, inputsSent: 118, stateUpdates: 0, decisionsMade: 58, shotsFired: 7, tacticalOverrides: 10 },
  ];

  const summary = coordinator._buildMatchSummary(selection, connectedSessions, agentResults);

  assert.equal(summary.expectedAgents, 2);
  assert.equal(summary.connectedAgents, 2);
  assert.equal(summary.totalScore, 350);
  assert.equal(summary.totalDamageDealt, 450);
  assert.equal(summary.totalKills, 3);
  assert.equal(summary.totalDeaths, 3);
  assert.equal(summary.totalInputsSent, 238);
  assert.equal(summary.totalStateUpdates, 40);
  assert.equal(summary.totalDecisionsMade, 118);
  assert.equal(summary.totalShotsFired, 16);
  assert.equal(summary.totalTacticalOverrides, 22);
  assert.equal(summary.hasCombatSignal, true);
});
