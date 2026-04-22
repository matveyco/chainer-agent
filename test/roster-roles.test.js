const test = require("node:test");
const assert = require("node:assert/strict");

const { buildDefaultRoster, flattenRosterAgents, LEAGUE_ROLES, defaultLeagueRole } = require("../src/runtime/Roster");

test("default roster assigns league roles using the documented pattern", () => {
  const config = {
    rooms: { count: 2, agentsPerRoom: 12 },
    training: { defaultModelAlias: "latest" },
  };
  const roster = buildDefaultRoster(config);
  const agents = flattenRosterAgents(roster);
  assert.equal(agents.length, 24);

  // Pattern: main, main_exploiter, league_exploiter, league_exploiter, repeat.
  // Across 24 agents that's 6 mains, 6 main exploiters, 12 league exploiters.
  const counts = { main: 0, main_exploiter: 0, league_exploiter: 0 };
  for (const agent of agents) {
    assert.ok(LEAGUE_ROLES.includes(agent.role), `unexpected role ${agent.role}`);
    counts[agent.role] += 1;
  }
  assert.equal(counts.main, 6);
  assert.equal(counts.main_exploiter, 6);
  assert.equal(counts.league_exploiter, 12);
});

test("default league role rotates through the pattern", () => {
  assert.equal(defaultLeagueRole(0), "main");
  assert.equal(defaultLeagueRole(1), "main_exploiter");
  assert.equal(defaultLeagueRole(2), "league_exploiter");
  assert.equal(defaultLeagueRole(3), "league_exploiter");
  assert.equal(defaultLeagueRole(4), "main");
});

test("default displayName uses ai_chainer prefix to avoid agent_* collisions", () => {
  const config = { rooms: { count: 1, agentsPerRoom: 4 } };
  const roster = buildDefaultRoster(config);
  const agents = flattenRosterAgents(roster);
  for (const agent of agents) {
    assert.match(agent.agentId, /^agent_\d+$/, "internal id stays agent_*");
    assert.match(agent.displayName, /^ai_chainer_\d+$/, `displayName should be ai_chainer_*, got ${agent.displayName}`);
  }
});

test("explicit role on roster entry is preserved", () => {
  const roster = buildDefaultRoster({ rooms: { count: 1, agentsPerRoom: 2 } });
  // Manually inject a role and re-normalize via buildDefaultRoster's downstream path:
  // here we simulate a roster.json that explicitly tags an agent.
  const { normalizeRosterDocument } = require("../src/runtime/Roster");
  const doc = {
    rooms: [
      [
        { agentId: "agent_x", role: "main_exploiter" },
        { agentId: "agent_y", role: "main" },
      ],
    ],
  };
  const normalized = normalizeRosterDocument(doc, { rooms: { count: 1, agentsPerRoom: 2 } });
  const agents = flattenRosterAgents(normalized);
  assert.equal(agents.find((a) => a.agentId === "agent_x").role, "main_exploiter");
  assert.equal(agents.find((a) => a.agentId === "agent_y").role, "main");
});
