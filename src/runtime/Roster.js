const fs = require("fs");
const path = require("path");

function normalizeAgent(agent, index, defaultAlias) {
  return {
    agentId: agent.agentId || agent.id || `agent_${index}`,
    displayName: agent.displayName || agent.userName || agent.agentId || `agent_${index}`,
    modelAlias: agent.modelAlias || agent.alias || defaultAlias,
  };
}

function buildDefaultRoster(config) {
  const roomCount = config.rooms?.count || 2;
  const agentsPerRoom = config.rooms?.agentsPerRoom || 12;
  const defaultAlias = config.training?.defaultModelAlias || "latest";
  const rooms = [];
  let index = 0;

  for (let roomIndex = 0; roomIndex < roomCount; roomIndex++) {
    const roomAgents = [];
    for (let i = 0; i < agentsPerRoom; i++) {
      roomAgents.push(normalizeAgent({ agentId: `agent_${index}` }, index, defaultAlias));
      index += 1;
    }
    rooms.push(roomAgents);
  }

  return rooms;
}

function normalizeRosterDocument(doc, config) {
  const defaultAlias = config.training?.defaultModelAlias || "latest";
  if (Array.isArray(doc?.rooms)) {
    return doc.rooms.map((room, roomIndex) =>
      (room || []).map((agent, agentIndex) =>
        normalizeAgent(agent, roomIndex * 1000 + agentIndex, defaultAlias)
      )
    );
  }

  const agents = Array.isArray(doc?.agents) ? doc.agents : Array.isArray(doc) ? doc : null;
  if (!agents) {
    return buildDefaultRoster(config);
  }

  const normalized = agents.map((agent, index) => normalizeAgent(agent, index, defaultAlias));
  const roomSize = config.rooms?.agentsPerRoom || 12;
  const rooms = [];
  for (let i = 0; i < normalized.length; i += roomSize) {
    rooms.push(normalized.slice(i, i + roomSize));
  }
  return rooms;
}

function loadRoster(config) {
  const rosterPath = path.resolve(config.persistence?.rosterFile || "config/roster.json");
  if (!fs.existsSync(rosterPath)) {
    return buildDefaultRoster(config);
  }

  try {
    const doc = JSON.parse(fs.readFileSync(rosterPath, "utf-8"));
    return normalizeRosterDocument(doc, config);
  } catch {
    return buildDefaultRoster(config);
  }
}

module.exports = {
  buildDefaultRoster,
  loadRoster,
  normalizeRosterDocument,
};
