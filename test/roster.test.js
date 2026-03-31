const test = require("node:test");
const assert = require("node:assert/strict");

const { normalizeRosterDocument } = require("../src/runtime/Roster");

test("roster normalization chunks a flat agent list by room size", () => {
  const rooms = normalizeRosterDocument({
    agents: [
      { agentId: "agent_0", modelAlias: "champion" },
      { agentId: "agent_1" },
      { agentId: "agent_2" },
    ],
  }, {
    rooms: { agentsPerRoom: 2 },
    training: { defaultModelAlias: "latest" },
  });

  assert.equal(rooms.length, 2);
  assert.equal(rooms[0][0].modelAlias, "champion");
  assert.equal(rooms[1][0].agentId, "agent_2");
});
