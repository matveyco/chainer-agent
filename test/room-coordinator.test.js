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
