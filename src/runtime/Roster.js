const fs = require("fs");
const path = require("path");
const { DEFAULT_ARCHETYPE_ID, getArchetypeByIndex } = require("../bot/archetypes");

function getDefaultTrack(roomIndex, config) {
  const configured = Array.isArray(config.rooms?.tracks) ? config.rooms.tracks[roomIndex] : null;
  if (configured) return configured;
  if (roomIndex === 0) return "stable";
  return "training";
}

function normalizeAgent(agent, index, defaultAlias) {
  const archetype = getArchetypeByIndex(index);
  return {
    agentId: agent.agentId || agent.id || `agent_${index}`,
    displayName: agent.displayName || agent.userName || agent.agentId || `agent_${index}`,
    modelAlias: agent.modelAlias || agent.alias || defaultAlias,
    policyFamily: agent.policyFamily || agent.family || "arena-main",
    archetypeId: agent.archetypeId || archetype.id || DEFAULT_ARCHETYPE_ID,
    modelVersion: Number.isFinite(agent.modelVersion) ? agent.modelVersion : null,
  };
}

function normalizeRoom(room, roomIndex, config, defaultAlias) {
  const sourceAgents = Array.isArray(room) ? room : Array.isArray(room?.agents) ? room.agents : [];
  const track = room?.track || getDefaultTrack(roomIndex, config);
  return {
    roomIndex,
    track,
    label: room?.label || `${track}-${roomIndex}`,
    agents: sourceAgents.map((agent, agentIndex) =>
      normalizeAgent(agent, roomIndex * 1000 + agentIndex, defaultAlias)
    ),
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
    rooms.push({
      roomIndex,
      track: getDefaultTrack(roomIndex, config),
      label: `${getDefaultTrack(roomIndex, config)}-${roomIndex}`,
      agents: roomAgents,
    });
  }

  return rooms;
}

function normalizeRosterDocument(doc, config) {
  const defaultAlias = config.training?.defaultModelAlias || "latest";
  if (Array.isArray(doc?.rooms)) {
    return doc.rooms.map((room, roomIndex) => normalizeRoom(room, roomIndex, config, defaultAlias));
  }

  const agents = Array.isArray(doc?.agents) ? doc.agents : Array.isArray(doc) ? doc : null;
  if (!agents) {
    return buildDefaultRoster(config);
  }

  const normalized = agents.map((agent, index) => normalizeAgent(agent, index, defaultAlias));
  const roomSize = config.rooms?.agentsPerRoom || 12;
  const rooms = [];
  for (let i = 0; i < normalized.length; i += roomSize) {
    const roomIndex = rooms.length;
    rooms.push({
      roomIndex,
      track: getDefaultTrack(roomIndex, config),
      label: `${getDefaultTrack(roomIndex, config)}-${roomIndex}`,
      agents: normalized.slice(i, i + roomSize),
    });
  }
  return rooms;
}

function flattenRosterAgents(rooms = []) {
  return rooms.flatMap((room) => room?.agents || []);
}

function countRosterAgents(rooms = []) {
  return flattenRosterAgents(rooms).length;
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
  countRosterAgents,
  flattenRosterAgents,
  getDefaultTrack,
  loadRoster,
  normalizeRosterDocument,
};
