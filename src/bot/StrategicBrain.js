/**
 * LLM-powered Strategic Brain for each agent.
 *
 * Sits on TOP of the neural network (PPO) layer. The NN handles frame-by-frame
 * reflexes (movement, aiming, shooting). The LLM handles:
 *   - Strategy formulation ("should I be aggressive or defensive?")
 *   - Behavior analysis ("why am I dying so much?")
 *   - Tactical adjustments ("focus on crystals, avoid open areas")
 *   - Self-reflection after each match
 *
 * The LLM outputs strategic parameters that modify the NN's behavior:
 *   - aggression (0-1): how close to get to enemies
 *   - accuracy_focus (0-1): tight aim vs spray
 *   - crystal_priority (0-1): prioritize crystals over kills
 *   - ability_usage (0-1): how often to use abilities
 *   - retreat_threshold (0-1): health % to retreat at
 *
 * These parameters are injected into the SmartBot decision-making loop,
 * modifying the raw NN outputs before they become game actions.
 */

const logger = require("../utils/logger");

const OLLAMA_API = "https://api.ollama.com/api/chat";
const DEFAULT_MODEL = "kimi-k2.5:cloud";

class StrategicBrain {
  constructor(agentId, apiKey, model = DEFAULT_MODEL) {
    this.agentId = agentId;
    this.apiKey = apiKey;
    this.model = model;

    // Agent identity / DNA
    this.personality = this._generatePersonality();

    // Current strategy (modified by LLM after each match)
    this.strategy = {
      aggression: 0.5 + (Math.random() - 0.5) * 0.4,      // 0.3 - 0.7
      accuracy_focus: 0.5 + (Math.random() - 0.5) * 0.4,
      crystal_priority: Math.random() * 0.5,                 // 0 - 0.5
      ability_usage: 0.3 + Math.random() * 0.4,              // 0.3 - 0.7
      retreat_threshold: 0.2 + Math.random() * 0.3,          // 0.2 - 0.5
    };

    // Behavioral log (what the agent "remembers")
    this.matchHistory = [];  // Last 10 match summaries
    this.currentPlan = "Explore arena, find enemies, learn to fight.";
    this.lastAnalysis = "";
    this.thoughtLog = [];    // LLM reasoning history

    // Stats for LLM context
    this.totalKills = 0;
    this.totalDeaths = 0;
    this.totalScore = 0;
    this.matchesPlayed = 0;
  }

  _generatePersonality() {
    const archetypes = [
      { name: "Hunter", traits: "aggressive, pursues enemies relentlessly, prefers close combat" },
      { name: "Sniper", traits: "patient, keeps distance, waits for perfect shots, high accuracy" },
      { name: "Collector", traits: "prioritizes crystals and score, avoids unnecessary fights" },
      { name: "Survivor", traits: "cautious, retreats when hurt, maximizes survival time" },
      { name: "Berserker", traits: "reckless, always attacking, uses abilities constantly" },
      { name: "Tactician", traits: "balanced, adapts strategy based on situation, smart positioning" },
      { name: "Flanker", traits: "stays at edges, attacks from unexpected angles, hit-and-run" },
      { name: "Guardian", traits: "controls center area, holds position, punishes approaching enemies" },
    ];

    // Each agent gets a deterministic personality based on their ID number
    const idx = parseInt(this.agentId.replace(/\D/g, "")) || 0;
    const archetype = archetypes[idx % archetypes.length];

    return {
      archetype: archetype.name,
      traits: archetype.traits,
      seed: idx,
    };
  }

  /**
   * After a match ends, ask the LLM to analyze performance and adjust strategy.
   * This is the core "thinking" step — runs between matches (not during).
   */
  async analyzeMatch(matchResult) {
    this.matchesPlayed++;
    this.totalKills += matchResult.kills || 0;
    this.totalDeaths += matchResult.deaths || 0;
    this.totalScore += matchResult.score || 0;

    // Store match summary
    this.matchHistory.push({
      match: this.matchesPlayed,
      score: matchResult.score || 0,
      kills: matchResult.kills || 0,
      deaths: matchResult.deaths || 0,
      damage_dealt: matchResult.damageDealt || 0,
      damage_taken: matchResult.damageTaken || 0,
      survival_time: matchResult.survivalTime || 0,
    });
    if (this.matchHistory.length > 10) this.matchHistory.shift();

    // Only call LLM every 3 matches to save API calls
    if (this.matchesPlayed % 3 !== 0) return;

    try {
      const analysis = await this._callLLM(this._buildAnalysisPrompt(matchResult));
      if (analysis) {
        this.lastAnalysis = analysis;
        this._parseStrategyFromAnalysis(analysis);
        this.thoughtLog.push({
          match: this.matchesPlayed,
          timestamp: new Date().toISOString(),
          thought: analysis.substring(0, 500),
        });
        if (this.thoughtLog.length > 20) this.thoughtLog.shift();

        logger.info(`[${this.agentId}] LLM strategy update: ${this.currentPlan.substring(0, 80)}`);
      }
    } catch (err) {
      logger.debug(`[${this.agentId}] LLM analysis failed: ${err.message}`);
    }
  }

  _buildAnalysisPrompt(lastMatch) {
    const avgScore = this.totalScore / Math.max(this.matchesPlayed, 1);
    const avgKD = this.totalKills / Math.max(this.totalDeaths, 1);
    const recentTrend = this.matchHistory.slice(-5);
    const scoresTrend = recentTrend.map(m => m.score).join(", ");
    const killsTrend = recentTrend.map(m => m.kills).join(", ");

    return `You are the strategic brain of an AI bot called "${this.agentId}" in a 3D multiplayer shooter arena game.

PERSONALITY: ${this.personality.archetype} — ${this.personality.traits}

CURRENT STRATEGY:
- aggression: ${this.strategy.aggression.toFixed(2)} (0=passive, 1=very aggressive)
- accuracy_focus: ${this.strategy.accuracy_focus.toFixed(2)} (0=spray, 1=precise)
- crystal_priority: ${this.strategy.crystal_priority.toFixed(2)} (0=ignore crystals, 1=focus crystals)
- ability_usage: ${this.strategy.ability_usage.toFixed(2)} (0=never, 1=always)
- retreat_threshold: ${this.strategy.retreat_threshold.toFixed(2)} (health % to retreat)

LIFETIME STATS: ${this.matchesPlayed} matches, ${this.totalKills} kills, ${this.totalDeaths} deaths, avg score ${avgScore.toFixed(0)}, K/D ${avgKD.toFixed(2)}

RECENT 5 MATCHES (score trend): ${scoresTrend}
RECENT 5 MATCHES (kills trend): ${killsTrend}

LAST MATCH: score=${lastMatch.score}, kills=${lastMatch.kills}, deaths=${lastMatch.deaths}, damage_dealt=${lastMatch.damageDealt || 0}

PREVIOUS PLAN: ${this.currentPlan}

Based on your performance trend:
1. What's working and what's not? (1 sentence)
2. What should you change? (1 sentence)
3. New plan for next matches (1 sentence)
4. Adjust strategy parameters — output EXACTLY in this format:
STRATEGY: aggression=X accuracy_focus=X crystal_priority=X ability_usage=X retreat_threshold=X

Keep values between 0.0 and 1.0. Be specific and actionable. Stay in character as a ${this.personality.archetype}.`;
  }

  _parseStrategyFromAnalysis(text) {
    // Extract strategy line
    const match = text.match(/STRATEGY:\s*(.+)/i);
    if (!match) return;

    const line = match[1];
    const params = ["aggression", "accuracy_focus", "crystal_priority", "ability_usage", "retreat_threshold"];

    for (const param of params) {
      const regex = new RegExp(`${param}\\s*=\\s*([0-9.]+)`, "i");
      const m = line.match(regex);
      if (m) {
        const val = parseFloat(m[1]);
        if (Number.isFinite(val) && val >= 0 && val <= 1) {
          this.strategy[param] = val;
        }
      }
    }

    // Extract plan (line before STRATEGY)
    const lines = text.split("\n").filter(l => l.trim());
    for (const l of lines) {
      if (l.includes("plan") || l.includes("Plan") || l.startsWith("3")) {
        this.currentPlan = l.replace(/^\d+[\.\)]\s*/, "").trim();
        break;
      }
    }
  }

  /**
   * Modify raw NN action outputs based on current strategy.
   * Called every decision tick by SmartBot.
   */
  modifyAction(action, gameContext) {
    const s = this.strategy;

    // Aggression modifies movement toward/away from enemies
    if (gameContext.hasEnemy) {
      // High aggression = move toward enemy, low = strafe/retreat
      action.moveX = action.moveX * (0.5 + s.aggression * 0.5);
      action.moveZ = action.moveZ * (0.5 + s.aggression * 0.5);
    }

    // Accuracy focus tightens aim offset
    action.aimOffsetX *= (1 - s.accuracy_focus * 0.7);
    action.aimOffsetZ *= (1 - s.accuracy_focus * 0.7);

    // Shoot threshold modified by aggression
    if (s.aggression > 0.7) {
      action.shouldShoot = action.shouldShoot || gameContext.hasEnemy;
    }

    // Retreat when health below threshold
    if (gameContext.healthPercent < s.retreat_threshold && gameContext.hasEnemy) {
      action.moveX *= -0.5;  // Reverse direction
      action.moveZ *= -0.5;
      action.shouldShoot = s.aggression > 0.6; // Only shoot while retreating if aggressive
    }

    // Ability usage
    if (s.ability_usage < 0.3) {
      action.shouldUseAbility = false;
    }

    return action;
  }

  /**
   * Get full agent profile for the dashboard.
   */
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
        avg_kd: this.totalKills / Math.max(this.totalDeaths, 1),
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

    if (!res.ok) {
      throw new Error(`LLM API ${res.status}`);
    }

    const data = await res.json();
    return data.message?.content || "";
  }
}

module.exports = { StrategicBrain };
