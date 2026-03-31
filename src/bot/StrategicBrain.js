/**
 * LLM-powered Strategic Brain for each agent.
 *
 * Architecture:
 *   NN (PPO) = reflexes: raw movement, aim, shoot decisions at 60Hz
 *   LLM (this) = strategy: high-level behavior that OVERRIDES NN when needed
 *
 * The LLM doesn't suggest — it COMMANDS. Strategy parameters directly control:
 *   - Whether the bot charges enemies or keeps distance
 *   - Whether it shoots on sight or waits for good aim
 *   - Whether it hunts for crystals instead of fighting
 *   - When to retreat vs fight to the death
 *   - How to use abilities
 *
 * Each agent has a personality archetype that sets initial behavior,
 * then the LLM adjusts strategy every 3 matches based on performance.
 */

const logger = require("../utils/logger");

const OLLAMA_API = "https://api.ollama.com/api/chat";
const DEFAULT_MODEL = "kimi-k2.5:cloud";

// Personality archetypes — each creates VERY different playstyles
const ARCHETYPES = [
  {
    name: "Hunter",
    traits: "Aggressive predator. Charges enemies, stays close, never retreats.",
    defaults: { aggression: 0.9, accuracy_focus: 0.4, crystal_priority: 0.1, ability_usage: 0.7, retreat_threshold: 0.1 },
  },
  {
    name: "Sniper",
    traits: "Patient marksman. Keeps maximum distance, only shoots with clear aim.",
    defaults: { aggression: 0.2, accuracy_focus: 0.95, crystal_priority: 0.2, ability_usage: 0.3, retreat_threshold: 0.4 },
  },
  {
    name: "Collector",
    traits: "Crystal hoarder. Avoids fights, prioritizes score through crystals and survival.",
    defaults: { aggression: 0.1, accuracy_focus: 0.3, crystal_priority: 0.95, ability_usage: 0.2, retreat_threshold: 0.6 },
  },
  {
    name: "Survivor",
    traits: "Cockroach. Impossible to kill. Retreats early, heals, outlasts everyone.",
    defaults: { aggression: 0.3, accuracy_focus: 0.5, crystal_priority: 0.4, ability_usage: 0.5, retreat_threshold: 0.7 },
  },
  {
    name: "Berserker",
    traits: "All-in maniac. Constant shooting, constant abilities, zero self-preservation.",
    defaults: { aggression: 1.0, accuracy_focus: 0.2, crystal_priority: 0.0, ability_usage: 1.0, retreat_threshold: 0.0 },
  },
  {
    name: "Tactician",
    traits: "Balanced fighter. Adapts to situation. Engages when advantageous, retreats when not.",
    defaults: { aggression: 0.5, accuracy_focus: 0.6, crystal_priority: 0.3, ability_usage: 0.5, retreat_threshold: 0.35 },
  },
  {
    name: "Flanker",
    traits: "Edge player. Circles the arena, attacks from behind, hit-and-run specialist.",
    defaults: { aggression: 0.6, accuracy_focus: 0.7, crystal_priority: 0.2, ability_usage: 0.6, retreat_threshold: 0.3 },
  },
  {
    name: "Guardian",
    traits: "Area controller. Holds center, punishes anyone who comes close, uses all abilities.",
    defaults: { aggression: 0.7, accuracy_focus: 0.5, crystal_priority: 0.4, ability_usage: 0.8, retreat_threshold: 0.25 },
  },
];

class StrategicBrain {
  constructor(agentId, apiKey, model = DEFAULT_MODEL) {
    this.agentId = agentId;
    this.apiKey = apiKey;
    this.model = model;

    // Deterministic personality from agent ID
    const idx = parseInt(agentId.replace(/\D/g, "")) || 0;
    const archetype = ARCHETYPES[idx % ARCHETYPES.length];

    this.personality = {
      archetype: archetype.name,
      traits: archetype.traits,
      seed: idx,
    };

    // Strategy — initialized from archetype defaults (NOT random!)
    this.strategy = { ...archetype.defaults };

    // State
    this.matchHistory = [];
    this.currentPlan = `Playing as ${archetype.name}: ${archetype.traits}`;
    this.lastAnalysis = "";
    this.thoughtLog = [];
    this.totalKills = 0;
    this.totalDeaths = 0;
    this.totalScore = 0;
    this.matchesPlayed = 0;
  }

  /**
   * HARD behavior override based on strategy parameters.
   * This doesn't "nudge" the NN — it OVERRIDES it when strategy demands it.
   *
   * Called every decision tick (~20Hz) by SmartBot._applyDecision()
   */
  modifyAction(action, ctx) {
    const s = this.strategy;

    // ═══ AGGRESSION: controls approach vs avoidance ═══
    if (ctx.hasEnemy) {
      if (s.aggression > 0.7) {
        // HIGH AGGRESSION: force charge toward enemy regardless of NN
        // NN might say "strafe left" but Hunter says "CHARGE"
        action.shouldShoot = true; // Always shoot when aggressive
        action.shouldUseAbility = s.ability_usage > 0.5;
      } else if (s.aggression < 0.3) {
        // LOW AGGRESSION: override movement to move AWAY from enemy
        action.moveX = -action.moveX;
        action.moveZ = -action.moveZ;
        action.shouldShoot = ctx.enemyDistance < 10; // Only shoot if very close
      }
    } else {
      // No enemy visible
      if (s.crystal_priority > 0.7) {
        // Crystal-focused: move toward center (where crystals spawn)
        const cx = -ctx.posX; // Direction toward center
        const cz = -ctx.posZ;
        const len = Math.sqrt(cx * cx + cz * cz) || 1;
        action.moveX = cx / len;
        action.moveZ = cz / len;
      }
    }

    // ═══ ACCURACY: controls aim precision ═══
    if (s.accuracy_focus > 0.8) {
      // SNIPER MODE: nearly zero aim offset, only shoot if very accurate
      action.aimOffsetX *= 0.1;
      action.aimOffsetZ *= 0.1;
      // Don't shoot unless enemy is close enough for a good shot
      if (ctx.enemyDistance > 15) {
        action.shouldShoot = false;
      }
    } else if (s.accuracy_focus < 0.3) {
      // SPRAY MODE: wider aim offset, shoot everything
      action.aimOffsetX *= 2.0;
      action.aimOffsetZ *= 2.0;
      if (ctx.hasEnemy) action.shouldShoot = true;
    }

    // ═══ RETREAT: survival instinct ═══
    if (ctx.healthPercent < s.retreat_threshold) {
      if (s.retreat_threshold > 0.5) {
        // STRONG RETREAT (Survivor/Collector): run away, stop shooting, use defensive ability
        action.moveX = -action.moveX || 0.5;
        action.moveZ = -action.moveZ || 0.5;
        action.shouldShoot = false;
        action.shouldUseAbility = true; // Use ability to survive (jump, shield, etc)
      } else if (s.retreat_threshold > 0.2) {
        // MODERATE RETREAT: backpedal but keep shooting
        action.moveX *= -0.7;
        action.moveZ *= -0.7;
      }
      // retreat_threshold <= 0.2: Berserker mode, never retreat, fight to death
    }

    // ═══ ABILITY USAGE ═══
    if (s.ability_usage > 0.8) {
      // ABILITY SPAM: always try to use abilities
      action.shouldUseAbility = true;
    } else if (s.ability_usage < 0.2) {
      // ABILITY HOARDER: never use abilities
      action.shouldUseAbility = false;
    }

    // ═══ CRYSTAL PRIORITY: override combat for resource gathering ═══
    if (s.crystal_priority > 0.8 && ctx.hasEnemy && ctx.enemyDistance > 12) {
      // COLLECTOR: ignore distant enemies, keep gathering
      action.shouldShoot = false;
    }

    return action;
  }

  /**
   * Post-match LLM analysis. Runs every 3 matches.
   * The LLM reviews performance and adjusts strategy parameters.
   */
  async analyzeMatch(matchResult) {
    this.matchesPlayed++;
    this.totalKills += matchResult.kills || 0;
    this.totalDeaths += matchResult.deaths || 0;
    this.totalScore += matchResult.score || 0;

    this.matchHistory.push({
      match: this.matchesPlayed,
      score: matchResult.score || 0,
      kills: matchResult.kills || 0,
      deaths: matchResult.deaths || 0,
      damage_dealt: matchResult.damageDealt || 0,
      survival_time: matchResult.survivalTime || 0,
    });
    if (this.matchHistory.length > 20) this.matchHistory.shift();

    // Only call LLM every 3 matches
    if (this.matchesPlayed % 3 !== 0 || !this.apiKey) return;

    try {
      const analysis = await this._callLLM(this._buildPrompt(matchResult));
      if (analysis) {
        this.lastAnalysis = analysis;
        this._parseStrategy(analysis);
        this.thoughtLog.push({
          match: this.matchesPlayed,
          timestamp: new Date().toISOString(),
          thought: analysis.substring(0, 500),
        });
        if (this.thoughtLog.length > 30) this.thoughtLog.shift();
        logger.info(`[${this.agentId}/${this.personality.archetype}] LLM: ${this.currentPlan.substring(0, 80)}`);
      }
    } catch (err) {
      logger.debug(`[${this.agentId}] LLM failed: ${err.message}`);
    }
  }

  _buildPrompt(lastMatch) {
    const avgScore = this.totalScore / Math.max(this.matchesPlayed, 1);
    const avgKD = this.totalKills / Math.max(this.totalDeaths, 1);
    const recent = this.matchHistory.slice(-5);

    return `You are "${this.agentId}", a combat AI in a 3D multiplayer shooter arena.

YOUR PERSONALITY: ${this.personality.archetype}
${this.personality.traits}

CURRENT STRATEGY PARAMETERS:
aggression=${this.strategy.aggression.toFixed(2)} (0=run away, 1=always charge)
accuracy_focus=${this.strategy.accuracy_focus.toFixed(2)} (0=spray randomly, 1=only perfect shots)
crystal_priority=${this.strategy.crystal_priority.toFixed(2)} (0=ignore crystals, 1=only collect crystals)
ability_usage=${this.strategy.ability_usage.toFixed(2)} (0=save abilities, 1=spam abilities)
retreat_threshold=${this.strategy.retreat_threshold.toFixed(2)} (health% to start retreating, 0=never retreat)

HOW THESE CONTROL YOU:
- aggression>0.7: you CHARGE enemies and always shoot. <0.3: you RUN AWAY from enemies.
- accuracy>0.8: you only shoot at close range with perfect aim. <0.3: you spray bullets everywhere.
- crystal_priority>0.8: you ignore enemies and collect crystals for score. <0.2: you ignore crystals.
- ability_usage>0.8: you spam abilities constantly. <0.2: you never use abilities.
- retreat_threshold>0.5: you retreat at 50% health. =0: you fight to death.

LIFETIME: ${this.matchesPlayed} matches, ${this.totalKills} kills, ${this.totalDeaths} deaths, K/D=${avgKD.toFixed(2)}, avg_score=${avgScore.toFixed(0)}

LAST 5 MATCHES:
${recent.map(m => `  match ${m.match}: score=${m.score} kills=${m.kills} deaths=${m.deaths}`).join("\n")}

PREVIOUS PLAN: ${this.currentPlan}

Analyze your performance and adjust strategy. Your goal: MAXIMIZE SCORE and get #1 on leaderboard.

Respond in EXACTLY this format:
ANALYSIS: [one sentence about what's working/not working]
PLAN: [one sentence about what to do differently]
STRATEGY: aggression=X accuracy_focus=X crystal_priority=X ability_usage=X retreat_threshold=X

Values must be 0.0 to 1.0. Stay in character as ${this.personality.archetype}.`;
  }

  _parseStrategy(text) {
    const stratMatch = text.match(/STRATEGY:\s*(.+)/i);
    if (stratMatch) {
      const line = stratMatch[1];
      for (const param of ["aggression", "accuracy_focus", "crystal_priority", "ability_usage", "retreat_threshold"]) {
        const m = line.match(new RegExp(`${param}\\s*=\\s*([0-9.]+)`, "i"));
        if (m) {
          const val = parseFloat(m[1]);
          if (Number.isFinite(val) && val >= 0 && val <= 1) {
            this.strategy[param] = val;
          }
        }
      }
    }

    const planMatch = text.match(/PLAN:\s*(.+)/i);
    if (planMatch) {
      this.currentPlan = planMatch[1].trim();
    }
  }

  getProfile() {
    return {
      agent_id: this.agentId,
      personality: this.personality,
      strategy: { ...this.strategy },
      current_plan: this.currentPlan,
      last_analysis: this.lastAnalysis,
      match_history: this.matchHistory,
      thought_log: this.thoughtLog,
      lifetime: {
        matches: this.matchesPlayed,
        total_kills: this.totalKills,
        total_deaths: this.totalDeaths,
        total_score: this.totalScore,
        avg_kd: +(this.totalKills / Math.max(this.totalDeaths, 1)).toFixed(2),
      },
    };
  }

  async _callLLM(prompt) {
    const res = await fetch(OLLAMA_API, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages: [{ role: "user", content: prompt }],
        stream: false,
      }),
    });
    if (!res.ok) throw new Error(`LLM API ${res.status}`);
    const data = await res.json();
    return data.message?.content || "";
  }
}

module.exports = { StrategicBrain };
