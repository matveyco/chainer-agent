/**
 * Match-boundary strategic coach.
 *
 * The LLM only runs between matches and produces bounded strategy values.
 * Hot-path action selection remains deterministic and low-latency.
 */

const logger = require("../utils/logger");
const { getDefaultArchetype } = require("./archetypes");

const OLLAMA_API = "https://api.ollama.com/api/chat";
// Primary: kimi-k2.6 (NO :cloud suffix). Direct probe vs Ollama Cloud
// catalog showed:
//   kimi-k2.6:cloud         -> ~60s for trivial prompt (slow path)
//   kimi-k2.6 (no suffix)   -> ~4s for the same prompt (fast path)
//   deepseek-v4-flash       -> ~1.2s
// The :cloud suffix triggers some slower routing on Ollama's side; without
// it we hit the latency-optimised endpoint. ~595B params, half k2.5's size.
const DEFAULT_MODEL = "kimi-k2.6";
// Fallback: deepseek-v4-flash is the latency-optimised variant (~140B);
// used as a one-shot retry when the primary times out or 5xx's. ~1s avg.
const DEFAULT_FALLBACK_MODEL = "deepseek-v4-flash";

class StrategicBrain {
  constructor(agentId, apiKey, model = DEFAULT_MODEL, options = {}) {
    this.agentId = agentId;
    this.apiKey = apiKey;
    this.model = model;
    this.fallbackModel = options.fallbackModel || DEFAULT_FALLBACK_MODEL;
    this.trainerUrl = options.trainerUrl || null;
    this.reporter = options.reporter || null;
    this.timeoutMs = Math.max(500, Number(options.timeoutMs || 3000));

    const idx = parseInt(agentId.replace(/\D/g, ""), 10) || 0;
    const archetype = getDefaultArchetype(options.archetypeId, idx);

    this.personality = {
      archetype_id: archetype.id,
      archetype: archetype.name,
      traits: archetype.traits,
      seed: idx,
    };

    this.strategy = { ...archetype.defaults, ...(options.initialStrategy || {}) };
    this.matchHistory = [];
    this.currentPlan = `Playing as ${archetype.name}: ${archetype.traits}`;
    this.lastAnalysis = "";
    this.thoughtLog = [];
    this.totalKills = 0;
    this.totalDeaths = 0;
    this.totalScore = 0;
    this.matchesPlayed = 0;
  }

  modifyAction(action, ctx) {
    const s = this.strategy;

    if (ctx.hasEnemy) {
      if (s.aggression > 0.7) {
        action.shouldShoot = true;
        action.shouldUseAbility = action.shouldUseAbility || s.ability_usage > 0.55;
      } else if (s.aggression < 0.3) {
        action.moveX = -action.moveX;
        action.moveZ = -action.moveZ;
        action.shouldShoot = ctx.enemyDistance < 10;
      }
    } else if (s.crystal_priority > 0.7 && ctx.closestCrystal) {
      action.moveX = ctx.closestCrystal.dirX;
      action.moveZ = ctx.closestCrystal.dirZ;
    }

    if (s.accuracy_focus > 0.8) {
      action.aimOffsetX *= 0.1;
      action.aimOffsetZ *= 0.1;
      if (ctx.enemyDistance > 15) action.shouldShoot = false;
    } else if (s.accuracy_focus < 0.3) {
      action.aimOffsetX *= 2.0;
      action.aimOffsetZ *= 2.0;
      if (ctx.hasEnemy) action.shouldShoot = true;
    }

    if (ctx.healthPercent < s.retreat_threshold) {
      if (s.retreat_threshold > 0.5) {
        action.moveX = -action.moveX || 0.5;
        action.moveZ = -action.moveZ || 0.5;
        action.shouldShoot = false;
        action.shouldUseAbility = true;
      } else if (s.retreat_threshold > 0.2) {
        action.moveX *= -0.7;
        action.moveZ *= -0.7;
      }
    }

    if (s.ability_usage > 0.8) {
      action.shouldUseAbility = true;
    } else if (s.ability_usage < 0.2) {
      action.shouldUseAbility = false;
    }

    if (s.crystal_priority > 0.8 && ctx.hasEnemy && ctx.enemyDistance > 12) {
      action.shouldShoot = false;
    }

    return action;
  }

  async analyzeMatch(matchResult) {
    this.matchesPlayed += 1;
    this.totalKills += matchResult.kills || 0;
    this.totalDeaths += matchResult.deaths || 0;
    this.totalScore += matchResult.score || 0;

    this.matchHistory.push({
      match: this.matchesPlayed,
      score: matchResult.score || 0,
      kills: matchResult.kills || 0,
      deaths: matchResult.deaths || 0,
      crystals: matchResult.crystals || 0,
      damage_dealt: matchResult.damageDealt || 0,
      damage_taken: matchResult.damageTaken || 0,
      survival_time: matchResult.survivalTime || 0,
    });
    if (this.matchHistory.length > 20) this.matchHistory.shift();

    if (this.matchesPlayed % 3 !== 0 || !this.apiKey) return;

    try {
      const previous = { ...this.strategy };
      const analysis = await this._callLLM(this._buildPrompt(matchResult));
      const parsed = this._parseStrategy(analysis);
      if (!parsed) return;

      this.lastAnalysis = parsed.analysis;
      this.currentPlan = parsed.plan;
      this.strategy = parsed.strategy;

      const diff = {};
      for (const [key, value] of Object.entries(parsed.strategy)) {
        if (previous[key] !== value) {
          diff[key] = {
            previous: +previous[key].toFixed(3),
            next: +value.toFixed(3),
          };
        }
      }

      this.thoughtLog.push({
        match: this.matchesPlayed,
        timestamp: new Date().toISOString(),
        thought: parsed.analysis,
        plan: parsed.plan,
        diff,
      });
      if (this.thoughtLog.length > 30) this.thoughtLog.shift();

      this.reporter?.incrementCounter("llmAnalyses");
      logger.info(`[${this.agentId}/${this.personality.archetype}] plan=${this.currentPlan.substring(0, 80)}`);
      await this._persistStrategyDiff(parsed, diff);
    } catch (err) {
      this.reporter?.incrementCounter("llmFailures");
      logger.warn(`[${this.agentId}/${this.personality.archetype}] LLM coach failed: ${err.message}`);
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

  getStrategyVector() {
    return {
      aggression: this.strategy.aggression ?? 0,
      accuracy_focus: this.strategy.accuracy_focus ?? 0,
      crystal_priority: this.strategy.crystal_priority ?? 0,
      ability_usage: this.strategy.ability_usage ?? 0,
      retreat_threshold: this.strategy.retreat_threshold ?? 0,
    };
  }

  _buildPrompt(lastMatch) {
    const avgScore = this.totalScore / Math.max(this.matchesPlayed, 1);
    const avgKD = this.totalKills / Math.max(this.totalDeaths, 1);
    const recent = this.matchHistory.slice(-5);

    // Tight, structured prompt. Output cap is ~200 chars per text field so
    // fast inference completes within the timeout. The "format: json" arg
    // we pass to Ollama enforces JSON-only output server-side too.
    return `You coach "${this.agentId}" (${this.personality.archetype}) in an arena shooter.

Output JSON ONLY, no prose, this schema:
{"analysis":"<= 200 chars, what just happened","plan":"<= 200 chars, next match plan","strategy":{"aggression":0..1,"accuracy_focus":0..1,"crystal_priority":0..1,"ability_usage":0..1,"retreat_threshold":0..1}}

Stay in character: ${this.personality.traits}
Optimize for: winning the round, surviving fights, collecting crystals.

Current: ${JSON.stringify(this.strategy)}
Lifetime: matches=${this.matchesPlayed} k=${this.totalKills} d=${this.totalDeaths} kd=${avgKD.toFixed(2)} avg_score=${avgScore.toFixed(0)}
Recent matches: ${recent.map((m) => JSON.stringify(m)).join("|")}
Last: ${JSON.stringify(lastMatch)}`;
  }

  _parseStrategy(text) {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start === -1 || end === -1) return null;

    const payload = JSON.parse(text.slice(start, end + 1));
    const strategy = {};
    for (const key of ["aggression", "accuracy_focus", "crystal_priority", "ability_usage", "retreat_threshold"]) {
      const raw = Number(payload.strategy?.[key]);
      strategy[key] = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : this.strategy[key];
    }

    return {
      analysis: String(payload.analysis || "").trim() || "No analysis returned.",
      plan: String(payload.plan || "").trim() || this.currentPlan,
      strategy,
    };
  }

  async _persistStrategyDiff(parsed, diff) {
    if (!this.trainerUrl) return;
    try {
      await fetch(`${this.trainerUrl}/agent/${this.agentId}/strategy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis: parsed.analysis,
          plan: parsed.plan,
          strategy: parsed.strategy,
          diff,
          personality: this.personality,
        }),
      });
    } catch {}
  }

  async _callLLM(prompt) {
    // Try primary, then optional fallback once, then give up. Each attempt
    // gets its own AbortController so a slow primary doesn't eat the
    // fallback's timeout budget.
    try {
      return await this._callLLMOnce(this.model, prompt);
    } catch (primaryErr) {
      const isRetryable = primaryErr.retryable === true;
      const haveFallback = !!this.fallbackModel && this.fallbackModel !== this.model;
      if (!isRetryable || !haveFallback) throw primaryErr;
      this.reporter?.incrementCounter("llmRetries");
      try {
        return await this._callLLMOnce(this.fallbackModel, prompt);
      } catch (fallbackErr) {
        // Surface the original (primary) error message + the fact that fallback
        // also failed so the failure log is actionable.
        const composite = new Error(`${primaryErr.message} (fallback ${this.fallbackModel} also failed: ${fallbackErr.message})`);
        composite.retryable = false;
        throw composite;
      }
    }
  }

  /**
   * Single attempt at one specific model. Marks errors as `retryable` when
   * they're transient (timeout / 5xx) so the caller can decide whether to
   * try the fallback. format:"json" tells Ollama to constrain output to
   * valid JSON server-side, which trims trailing prose and finishes faster.
   */
  async _callLLMOnce(model, prompt) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(OLLAMA_API, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          stream: false,
          format: "json",
        }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const err = new Error(`LLM API ${res.status} (${model})`);
        err.retryable = res.status >= 500 || res.status === 429;
        if (res.status >= 500) this.reporter?.incrementCounter("llmServerErrors");
        throw err;
      }
      const data = await res.json();
      return data.message?.content || "";
    } catch (err) {
      if (err.name === "AbortError") {
        const tErr = new Error(`LLM timeout after ${this.timeoutMs}ms (${model})`);
        tErr.retryable = true;
        this.reporter?.incrementCounter("llmTimeouts");
        throw tErr;
      }
      // Network errors (fetch failed, DNS, etc.) — treat as retryable.
      if (err.retryable === undefined) err.retryable = true;
      throw err;
    } finally {
      clearTimeout(timeout);
    }
  }
}

module.exports = { StrategicBrain };
