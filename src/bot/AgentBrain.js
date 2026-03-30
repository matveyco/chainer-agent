/**
 * ONNX-based agent brain.
 * Loads a per-agent ONNX model from the training service and runs inference.
 * Collects experience tuples and sends them back for PPO training.
 */

const fs = require("fs");
const path = require("path");
const logger = require("../utils/logger");

// State dimensions must match Python trainer
const STATE_DIM = 18;
const ACTION_DIM = 6;

class AgentBrain {
  /**
   * @param {string} agentId — persistent agent identity
   * @param {string} trainerUrl — URL of Python training service
   */
  constructor(agentId, trainerUrl) {
    this.agentId = agentId;
    this.trainerUrl = trainerUrl;
    this.session = null;
    this.ort = null;
    this.modelVersion = 0;
    this.experienceBuffer = [];
    this.lastState = null;
    this.lastAction = null;
    this.lastScore = 0;
    this.ready = false;
  }

  /**
   * Load ONNX model from training service.
   */
  async init() {
    try {
      this.ort = require("onnxruntime-node");
    } catch {
      logger.warn(`Agent ${this.agentId}: onnxruntime-node not available, using random actions`);
      this.ready = false;
      return;
    }

    await this._loadModel();
  }

  async _loadModel() {
    try {
      // Fetch ONNX model from trainer
      const res = await fetch(`${this.trainerUrl}/model/${this.agentId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const buffer = await res.arrayBuffer();
      const modelData = new Uint8Array(buffer);

      // Load into ONNX Runtime
      this.session = await this.ort.InferenceSession.create(Buffer.from(modelData));
      this.ready = true;

      // Check version
      const verRes = await fetch(`${this.trainerUrl}/model/${this.agentId}/version`);
      const verData = await verRes.json();
      this.modelVersion = verData.version || 0;

      logger.debug(`Agent ${this.agentId}: loaded model v${this.modelVersion}`);
    } catch (err) {
      logger.warn(`Agent ${this.agentId}: model load failed: ${err.message}`);
      this.ready = false;
    }
  }

  /**
   * Run inference: state vector → action vector.
   * @param {Float32Array} stateVector — 18-element normalized state
   * @returns {Object} action decisions
   */
  async decide(stateVector) {
    // Store state for experience collection
    this.lastState = Array.from(stateVector);

    let actionValues;

    if (this.ready && this.session) {
      try {
        const tensor = new this.ort.Tensor("float32", stateVector, [1, STATE_DIM]);
        const results = await this.session.run({ state: tensor });
        actionValues = Array.from(results.action.data);
      } catch {
        actionValues = this._randomAction();
      }
    } else {
      actionValues = this._randomAction();
    }

    this.lastAction = actionValues;

    // Map action vector to game decisions
    // Actions are tanh-bounded [-1, 1] from the network
    return {
      moveX: actionValues[0],
      moveZ: actionValues[1],
      aimOffsetX: actionValues[2] * 3, // Scale to ±3 units
      aimOffsetZ: actionValues[3] * 3,
      shouldShoot: actionValues[4] > 0, // Threshold at 0
      shouldUseAbility: actionValues[5] > 0,
    };
  }

  /**
   * Record a reward step. Called after each game tick where we have new info.
   * @param {number} currentScore — game's actual score
   * @param {number} kills — total kills
   * @param {number} deaths — total deaths
   * @param {boolean} died — did we just die this tick?
   * @param {boolean} gotKill — did we get a kill this tick?
   * @param {number} damageDealt — damage dealt this tick
   * @param {boolean} done — match ended?
   */
  recordStep(currentScore, kills, deaths, died, gotKill, damageDealt, done) {
    if (!this.lastState || !this.lastAction) return;

    // Reward = score delta + kill bonus - death penalty
    const scoreDelta = currentScore - this.lastScore;
    let reward = scoreDelta * 0.01; // Normalize score to reasonable range

    if (gotKill) reward += 1.0;
    if (died) reward -= 0.5;
    if (damageDealt > 0) reward += damageDealt * 0.005;

    this.lastScore = currentScore;

    this.experienceBuffer.push({
      state: this.lastState,
      action: this.lastAction,
      reward,
      done,
      score: currentScore,
      kills,
      deaths,
    });
  }

  /**
   * Send collected experience to training service.
   * Call this periodically or at end of match.
   */
  async flush() {
    if (this.experienceBuffer.length === 0) return;

    try {
      const res = await fetch(`${this.trainerUrl}/experience`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_id: this.agentId,
          transitions: this.experienceBuffer,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        // Reload model if new version available
        if (data.model_version > this.modelVersion) {
          await this._loadModel();
        }
      }
    } catch (err) {
      logger.debug(`Agent ${this.agentId}: flush failed: ${err.message}`);
    }

    this.experienceBuffer = [];
  }

  /**
   * Report end-of-match stats.
   */
  async reportEpisode(score, kills, deaths) {
    try {
      await fetch(`${this.trainerUrl}/episode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_id: this.agentId,
          score,
          kills,
          deaths,
        }),
      });
    } catch {}
  }

  _randomAction() {
    return Array.from({ length: ACTION_DIM }, () => Math.random() * 2 - 1);
  }

  /**
   * Reset score tracking for new match.
   */
  resetMatch() {
    this.lastScore = 0;
    this.lastState = null;
    this.lastAction = null;
  }
}

module.exports = { AgentBrain, STATE_DIM, ACTION_DIM };
