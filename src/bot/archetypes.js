const ARCHETYPE_LIBRARY = [
  {
    id: "hunter",
    name: "Hunter",
    traits: "Aggressive predator. Charges enemies, stays close, never retreats.",
    defaults: { aggression: 0.9, accuracy_focus: 0.4, crystal_priority: 0.1, ability_usage: 0.7, retreat_threshold: 0.1 },
  },
  {
    id: "sniper",
    name: "Sniper",
    traits: "Patient marksman. Keeps maximum distance, only shoots with clear aim.",
    defaults: { aggression: 0.2, accuracy_focus: 0.95, crystal_priority: 0.2, ability_usage: 0.3, retreat_threshold: 0.4 },
  },
  {
    id: "collector",
    name: "Collector",
    traits: "Crystal hoarder. Avoids fights, prioritizes score through crystals and survival.",
    defaults: { aggression: 0.1, accuracy_focus: 0.3, crystal_priority: 0.95, ability_usage: 0.2, retreat_threshold: 0.6 },
  },
  {
    id: "survivor",
    name: "Survivor",
    traits: "Cockroach. Retreats early, heals, outlasts everyone.",
    defaults: { aggression: 0.3, accuracy_focus: 0.5, crystal_priority: 0.4, ability_usage: 0.5, retreat_threshold: 0.7 },
  },
  {
    id: "berserker",
    name: "Berserker",
    traits: "All-in maniac. Constant shooting, constant abilities, zero self-preservation.",
    defaults: { aggression: 1.0, accuracy_focus: 0.2, crystal_priority: 0.0, ability_usage: 1.0, retreat_threshold: 0.0 },
  },
  {
    id: "tactician",
    name: "Tactician",
    traits: "Balanced fighter. Adapts to situation and trades off score, risk, and objectives.",
    defaults: { aggression: 0.5, accuracy_focus: 0.6, crystal_priority: 0.3, ability_usage: 0.5, retreat_threshold: 0.35 },
  },
  {
    id: "flanker",
    name: "Flanker",
    traits: "Edge player. Circles the arena, attacks from behind, hit-and-run specialist.",
    defaults: { aggression: 0.6, accuracy_focus: 0.7, crystal_priority: 0.2, ability_usage: 0.6, retreat_threshold: 0.3 },
  },
  {
    id: "guardian",
    name: "Guardian",
    traits: "Area controller. Holds center, punishes intruders, values objective denial.",
    defaults: { aggression: 0.7, accuracy_focus: 0.5, crystal_priority: 0.4, ability_usage: 0.8, retreat_threshold: 0.25 },
  },
];

const DEFAULT_ARCHETYPE_ID = "tactician";

function getArchetypeById(archetypeId) {
  return ARCHETYPE_LIBRARY.find((entry) => entry.id === archetypeId) || null;
}

function getArchetypeByIndex(index) {
  return ARCHETYPE_LIBRARY[index % ARCHETYPE_LIBRARY.length];
}

function getDefaultArchetype(archetypeId, index = 0) {
  return getArchetypeById(archetypeId) || getArchetypeByIndex(index);
}

module.exports = {
  ARCHETYPE_LIBRARY,
  DEFAULT_ARCHETYPE_ID,
  getArchetypeById,
  getArchetypeByIndex,
  getDefaultArchetype,
};
