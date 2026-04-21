/**
 * ONNX-based agent brain.
 * Loads a per-agent ONNX model from the training service and runs inference.
 * Collects experience tuples and sends them back for PPO training.
 */

const logger = require("../utils/logger");

// State dimensions must match Python trainer.
const STATE_DIM = 24;
const ACTION_DIM = 6;

class AgentBrain {
  /**
   * @param {string} agentId
   * @param {string} trainerUrl
   * @param {Object} options
   */
  constructor(agentId, trainerUrl, options = {}) {
    this.agentId = agentId;
    this.trainerUrl = trainerUrl;
    this.modelAlias = options.modelAlias || "latest";
    this.modelVersionHint = options.modelVersion ?? null;
    this.policyFamily = options.policyFamily || "arena-main";
    this.archetypeId = options.archetypeId || "tactician";
    this.role = options.role || "league_exploiter";
    this.rewardConfig = options.rewardConfig || {};
    this.reporter = options.reporter || null;
    this.strategyProvider = options.strategyProvider || null;
    this.session = null;
    this.ort = null;
    this.modelVersion = 0;
    this.metadata = null;
    this.experienceBuffer = [];
    this.episodeRewardTotals = {};
    this.lastState = null;
    this.lastAction = null;
    this.lastScore = 0;
    this.lastKills = 0;
    this.lastDeaths = 0;
    this.ready = false;
    this.loadingPromise = null;
    this.nextLoadAttemptAt = 0;
  }

  async _fetchRewardWeights() {
    try {
      const res = await fetch(`${this.trainerUrl}/agent/${this.agentId}/reward-weights`);
      if (!res.ok) return;
      const body = await res.json();
      const weights = body?.reward_weights;
      if (weights && typeof weights === "object") {
        // Merge: trainer-supplied per-agent weights override the bot's defaults,
        // but anything missing falls back to the local config so we never get
        // unknown undefined fields after a partial update.
        this.rewardConfig = { ...this.rewardConfig, ...weights };
      }
    } catch (err) {
      logger.debug(`Agent ${this.agentId}: reward-weights fetch failed: ${err.message}`);
    }
  }

  async init() {
    // Fetch agent-specific reward weights first so any subsequent recordStep
    // calls use the per-agent genome instead of the global defaults.
    await this._fetchRewardWeights();
    try {
      this.ort = require("onnxruntime-node");
    } catch {
      logger.warn(`Agent ${this.agentId}: onnxruntime-node not available, using random actions`);
      this.ready = false;
      return;
    }

    await this._loadModel(true);
  }

  async decide(stateVector) {
    const stateArr = Array.from(stateVector);
    for (let i = 0; i < stateArr.length; i++) {
      if (!Number.isFinite(stateArr[i])) stateArr[i] = 0;
    }
    this.lastState = stateArr;

    if (!this.ready && Date.now() >= this.nextLoadAttemptAt) {
      this._loadModel().catch(() => {});
    }

    let actionValues;
    if (this.ready && this.session) {
      try {
        const tensor = new this.ort.Tensor("float32", stateVector, [1, STATE_DIM]);
        const results = await this.session.run({ state: tensor });
        actionValues = Array.from(results.action.data);
      } catch (err) {
        this.ready = false;
        this._reportModelFailure(err);
        actionValues = this._randomAction();
      }
    } else {
      actionValues = this._randomAction();
    }

    this.lastAction = actionValues;

    return {
      moveX: actionValues[0],
      moveZ: actionValues[1],
      aimOffsetX: actionValues[2] * 3,
      aimOffsetZ: actionValues[3] * 3,
      shouldShoot: actionValues[4] > 0,
      shouldUseAbility: actionValues[5] > 0,
    };
  }

  recordStep(...args) {
    const transition = this._normalizeTransitionArgs(args);
    if (!this.lastState || !this.lastAction) return;

    const scoreDelta = transition.currentScore - this.lastScore;
    const killDelta = Math.max(0, transition.kills - this.lastKills);
    const deathDelta = Math.max(0, transition.deaths - this.lastDeaths);

    const rewardComponents = {
      scoreDelta: scoreDelta * (this.rewardConfig.scoreDeltaWeight ?? 0.02),
      kills: (transition.gotKill ? 1 : killDelta) * (this.rewardConfig.killBonus ?? 1.0),
      deaths: (transition.died ? 1 : deathDelta) * (this.rewardConfig.deathPenalty ?? -0.75),
      damage: transition.damageDealt * (this.rewardConfig.damageWeight ?? 0.004),
      damageTaken: transition.damageTaken * (this.rewardConfig.damageTakenPenalty ?? -0.002),
      survival: transition.survivalSeconds * (this.rewardConfig.survivalWeight ?? 0.01),
      ability: transition.abilityUsed ? (this.rewardConfig.abilityValueWeight ?? 0.08) : 0,
      accuracy: transition.shotAccuracy * (this.rewardConfig.accuracyWeight ?? 0.2),
      antiSuicide: transition.died && !transition.gotKill ? (this.rewardConfig.antiSuicidePenalty ?? -0.3) : 0,
      // Terminal match-rank bonus (only on done=true, when rank info is present).
      // Rewards #1 finish, penalises last finish; tunable via reward.matchRankBonus.
      matchRank:
        transition.done && transition.rank > 0 && transition.roomSize > 0
          ? this._computeRankReward(transition.rank, transition.roomSize) *
            (this.rewardConfig.matchRankBonus ?? 1.0)
          : 0,
    };

    const reward = Object.values(rewardComponents).reduce((sum, value) => sum + value, 0);

    this.lastScore = transition.currentScore;
    this.lastKills = transition.kills;
    this.lastDeaths = transition.deaths;

    for (const [key, value] of Object.entries(rewardComponents)) {
      this.episodeRewardTotals[key] = (this.episodeRewardTotals[key] || 0) + value;
    }

    this.experienceBuffer.push({
      state: this.lastState,
      action: this.lastAction,
      reward,
      reward_components: rewardComponents,
      done: transition.done,
      score: transition.currentScore,
      kills: transition.kills,
      deaths: transition.deaths,
    });
  }

  async flush() {
    if (this.experienceBuffer.length === 0) return;

    this.experienceBuffer = this.experienceBuffer.filter((entry) => {
      if (!entry.state || !entry.action) return false;
      return entry.state.every(Number.isFinite) && entry.action.every(Number.isFinite);
    });
    if (this.experienceBuffer.length === 0) return;

    try {
      const res = await fetch(`${this.trainerUrl}/experience`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_id: this.agentId,
          model_alias: this.modelAlias,
          model_version: this.modelVersionHint,
          policy_family: this.policyFamily,
          archetype_id: this.archetypeId,
          role: this.role,
          strategy_vector: this._getStrategyVector(),
          transitions: this.experienceBuffer,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        this.reporter?.incrementCounter("experienceFlushes");
        this.reporter?.observe("experienceFlushSizeAvg", this.experienceBuffer.length);
        if (data.model_version && data.model_version !== this.modelVersion) {
          await this._loadModel(true);
        }
      }
    } catch (err) {
      logger.debug(`Agent ${this.agentId}: flush failed: ${err.message}`);
    }

    this.experienceBuffer = [];
  }

  async reportEpisode(summary) {
    const payload = typeof summary === "object"
      ? summary
      : { score: summary, kills: arguments[1], deaths: arguments[2] };

    try {
      await fetch(`${this.trainerUrl}/episode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_id: this.agentId,
          model_alias: this.modelAlias,
          model_version: this.modelVersionHint,
          policy_family: this.policyFamily,
          archetype_id: this.archetypeId,
          role: this.role,
          strategy_vector: this._getStrategyVector(),
          reward_totals: this.episodeRewardTotals,
          ...payload,
        }),
      });
    } catch {}
  }

  async reportStrategy(strategyPayload) {
    try {
      await fetch(`${this.trainerUrl}/agent/${this.agentId}/strategy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...strategyPayload,
          policy_family: this.policyFamily,
          archetype_id: this.archetypeId,
        }),
      });
    } catch {}
  }

  resetMatch() {
    this.lastScore = 0;
    this.lastKills = 0;
    this.lastDeaths = 0;
    this.lastState = null;
    this.lastAction = null;
    this.episodeRewardTotals = {};
  }

  async _loadModel(force = false) {
    if (!force && Date.now() < this.nextLoadAttemptAt) return;
    if (this.loadingPromise) return this.loadingPromise;

    this.loadingPromise = (async () => {
      try {
        const url = new URL(`${this.trainerUrl}/model/${this.agentId}`);
        url.searchParams.set("alias", this.modelAlias);
        url.searchParams.set("policy_family", this.policyFamily);
        url.searchParams.set("archetype_id", this.archetypeId);
        if (this.modelVersionHint !== null && this.modelVersionHint !== undefined) {
          url.searchParams.set("version", String(this.modelVersionHint));
        }

        const startedAt = Date.now();
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const buffer = await res.arrayBuffer();
        const modelData = new Uint8Array(buffer);
        this.session = await this.ort.InferenceSession.create(Buffer.from(modelData));
        this.ready = true;

        const metadataUrl = new URL(`${this.trainerUrl}/model/${this.agentId}/metadata`);
        metadataUrl.searchParams.set("alias", this.modelAlias);
        metadataUrl.searchParams.set("policy_family", this.policyFamily);
        metadataUrl.searchParams.set("archetype_id", this.archetypeId);
        if (this.modelVersionHint !== null && this.modelVersionHint !== undefined) {
          metadataUrl.searchParams.set("version", String(this.modelVersionHint));
        }
        const metadataRes = await fetch(metadataUrl).catch(() => null);
        if (metadataRes?.ok) {
          this.metadata = await metadataRes.json();
          this.modelVersion = this.metadata.model_version || 0;
        } else {
          const versionUrl = new URL(`${this.trainerUrl}/model/${this.agentId}/version`);
          versionUrl.searchParams.set("alias", this.modelAlias);
          versionUrl.searchParams.set("policy_family", this.policyFamily);
          versionUrl.searchParams.set("archetype_id", this.archetypeId);
          if (this.modelVersionHint !== null && this.modelVersionHint !== undefined) {
            versionUrl.searchParams.set("version", String(this.modelVersionHint));
          }
          const versionRes = await fetch(versionUrl);
          const versionData = await versionRes.json();
          this.modelVersion = versionData.version || 0;
        }

        const latency = Date.now() - startedAt;
        this.reporter?.incrementCounter("modelFetches");
        this.reporter?.observe("modelFetchLatencyMsAvg", latency);
        logger.debug(`Agent ${this.agentId}: loaded ${this.modelAlias} v${this.modelVersion}`);
      } catch (err) {
        this.ready = false;
        this._reportModelFailure(err);
      } finally {
        this.loadingPromise = null;
      }
    })();

    return this.loadingPromise;
  }

  _normalizeTransitionArgs(args) {
    if (args.length === 1 && typeof args[0] === "object") {
      return {
        currentScore: args[0].currentScore || 0,
        kills: args[0].kills || 0,
        deaths: args[0].deaths || 0,
        died: !!args[0].died,
        gotKill: !!args[0].gotKill,
        damageDealt: args[0].damageDealt || 0,
        damageTaken: args[0].damageTaken || 0,
        survivalSeconds: args[0].survivalSeconds || 0,
        abilityUsed: !!args[0].abilityUsed,
        shotAccuracy: args[0].shotAccuracy || 0,
        done: !!args[0].done,
        rank: args[0].rank || 0,
        roomSize: args[0].roomSize || 0,
      };
    }

    return {
      currentScore: args[0] || 0,
      kills: args[1] || 0,
      deaths: args[2] || 0,
      died: !!args[3],
      gotKill: !!args[4],
      damageDealt: args[5] || 0,
      damageTaken: 0,
      survivalSeconds: 0,
      abilityUsed: false,
      shotAccuracy: 0,
      done: !!args[6],
      rank: 0,
      roomSize: 0,
    };
  }

  /**
   * Map (rank, roomSize) → terminal reward signal in roughly [-1, +1].
   *  rank=1            → +1.0
   *  top 25% (incl 1)  → +0.5
   *  middle half       → 0
   *  bottom 25%        → -0.5
   *  last              → -1.0
   * Symmetric around the median so the trainer learns to target the top half.
   */
  _computeRankReward(rank, roomSize) {
    if (roomSize <= 1) return 0;
    if (rank === 1) return 1.0;
    if (rank === roomSize) return -1.0;
    const fraction = (rank - 1) / (roomSize - 1); // 0 best ... 1 worst
    if (fraction <= 0.25) return 0.5;
    if (fraction >= 0.75) return -0.5;
    return 0;
  }

  _randomAction() {
    return Array.from({ length: ACTION_DIM }, () => Math.random() * 2 - 1);
  }

  _reportModelFailure(err) {
    this.nextLoadAttemptAt = Date.now() + 10000;
    this.reporter?.incrementCounter("modelFetchFailures");
    logger.warn(`Agent ${this.agentId}: model load failed: ${err.message}`);
  }

  _getStrategyVector() {
    const raw = typeof this.strategyProvider === "function" ? this.strategyProvider() : null;
    return {
      aggression: raw?.aggression ?? 0,
      accuracy_focus: raw?.accuracy_focus ?? 0,
      crystal_priority: raw?.crystal_priority ?? 0,
      ability_usage: raw?.ability_usage ?? 0,
      retreat_threshold: raw?.retreat_threshold ?? 0,
    };
  }
}

module.exports = { AgentBrain, STATE_DIM, ACTION_DIM };
