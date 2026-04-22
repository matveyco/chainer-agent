const fs = require("fs");
const path = require("path");
const { DEFAULT_ARCHETYPE_ID, getArchetypeByIndex } = require("../bot/archetypes");

function getDefaultTrack(roomIndex, config) {
  const configured = Array.isArray(config.rooms?.tracks) ? config.rooms.tracks[roomIndex] : null;
  if (configured) return configured;
  if (roomIndex === 0) return "stable";
  return "training";
}

/**
 * League roles used to bias PBT and (later) opponent sampling:
 * - main:             never overwritten by PBT exploit; preserves diversity.
 * - main_exploiter:   targets a main agent's weaknesses; PBT-replaceable.
 * - league_exploiter: free agent, fully PBT-replaceable; default for new agents.
 *
 * Default assignment when nothing else is specified: every 4th agent is a main,
 * the next is a main_exploiter, the rest are league_exploiters.
 * For 24 agents that gives 6 main / 6 main_exploiter / 12 league_exploiter — a
 * scaled-down approximation of the FTW Quake III recipe.
 */
const LEAGUE_ROLES = ["main", "main_exploiter", "league_exploiter"];
const LEAGUE_DEFAULT_PATTERN = ["main", "main_exploiter", "league_exploiter", "league_exploiter"];

function defaultLeagueRole(index) {
  return LEAGUE_DEFAULT_PATTERN[index % LEAGUE_DEFAULT_PATTERN.length];
}

// Visible name prefix sent in room:player:loaded.profile.userName.
// Kept distinct from the internal agentId (which keys the model registry)
// so we don't collide with other bot services using "agent_*" naming.
const DISPLAY_NAME_PREFIX = process.env.BOT_DISPLAY_NAME_PREFIX || "ai_chainer";

function defaultDisplayName(agentId, index) {
  // If the agentId looks like our internal "agent_<n>" key, swap the prefix
  // so the player-facing name is e.g. "ai_chainer_3" instead of "agent_3".
  const match = String(agentId || "").match(/^agent_(\d+)$/);
  if (match) return `${DISPLAY_NAME_PREFIX}_${match[1]}`;
  return agentId || `${DISPLAY_NAME_PREFIX}_${index}`;
}

function normalizeAgent(agent, index, defaultAlias) {
  const archetype = getArchetypeByIndex(index);
  const role = LEAGUE_ROLES.includes(agent.role) ? agent.role : defaultLeagueRole(index);
  const agentId = agent.agentId || agent.id || `agent_${index}`;
  return {
    agentId,
    displayName: agent.displayName || agent.userName || defaultDisplayName(agentId, index),
    modelAlias: agent.modelAlias || agent.alias || defaultAlias,
    policyFamily: agent.policyFamily || agent.family || "arena-main",
    archetypeId: agent.archetypeId || archetype.id || DEFAULT_ARCHETYPE_ID,
    modelVersion: Number.isFinite(agent.modelVersion) ? agent.modelVersion : null,
    role,
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
  LEAGUE_ROLES,
  defaultLeagueRole,
};
