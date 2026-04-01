/**
 * Smart bot with per-agent ONNX inference brain.
 */

const { AgentBrain } = require("./AgentBrain");
const { StrategicBrain } = require("./StrategicBrain");
const { StateExtractor } = require("./StateExtractor");
const { TacticalController } = require("./TacticalController");
const { FitnessTracker } = require("../metrics/FitnessTracker");
const { encodeInput, encodeShoot } = require("../network/Protocol");
const { rand, getArrayLength, getDirectionArray } = require("../utils/math");

const INPUT_BUFFER_SIZE = 16;
const ABILITIES = Object.freeze(["rampage", "jump", "minePlanting"]);

class SmartBot {
  constructor(userID, _unused, config, agentId, options = {}) {
    this.userID = userID;
    this.agentId = agentId || userID;
    this.displayName = options.displayName || this.agentId;
    this.modelAlias = options.modelAlias || config.training?.defaultModelAlias || "latest";
    this.modelVersion = options.modelVersion ?? null;
    this.policyFamily = options.policyFamily || "arena-main";
    this.archetypeId = options.archetypeId || "tactician";
    this.mode = options.mode || "training";
    this.reporter = options.reporter || null;
    this.config = config;
    this.mapName = config.server.mapName;
    this.weaponType = config.server.weaponType;
    this.agentSeed = parseInt(String(this.agentId).replace(/\D/g, ""), 10) || 0;

    this.brain = null;
    this.strategicBrain = null;
    if (config.ollamaApiKey) {
      this.strategicBrain = new StrategicBrain(
        this.agentId,
        config.ollamaApiKey,
        config.ollamaModel || "kimi-k2.5:cloud",
        {
          archetypeId: this.archetypeId,
          trainerUrl: config.trainerUrl,
          reporter: options.reporter || null,
        }
      );
    }

    this.stateExtractor = new StateExtractor();
    this.tacticalController = new TacticalController({
      archetypeId: this.archetypeId,
      seed: this.agentSeed,
      safeSize: config.bot?.arenaSafeSize,
    });
    this.fitness = new FitnessTracker();

    this.room = null;
    this.gameState = null;
    this.data = null;
    this.connected = false;
    this.alive = true;
    this.matchEnded = false;

    this.inputBuffer = new Array(INPUT_BUFFER_SIZE);
    for (let i = 0; i < INPUT_BUFFER_SIZE; i++) {
      this.inputBuffer[i] = {
        inputMove: new Float32Array(3),
        target: new Float32Array(3),
        animation: 0,
        speed: 0,
      };
    }
    this.inputIndex = 0;
    this.lastInputIndex = 0;

    this.coolDownTimer = 0;
    this.abilityTimer = rand(5, 15);
    this.searchClosestTimer = 0;
    this.rttTimer = 0;

    this.closestEnemy = null;
    this.closestCrystal = null;
    this.lastMoveX = 0;
    this.lastMoveZ = 0;
    this.positionArray = [0, 0, 0];
    this.positionVector = { x: 0, y: 0, z: 0 };
    this.decisionInFlight = false;
    this.lastAbilityUsed = false;
    this.lastTargetPoint = { x: 0, y: 0, z: 0 };
    this.lastCombatAt = Date.now();
    this.lastShotAt = 0;
    this.decisionsMade = 0;
    this.shotsFired = 0;
    this.tacticalOverrides = 0;
    this.tacticalReasons = [];

    this._lastGotKill = false;
    this._lastDied = false;
    this._lastDamageDealt = 0;
    this._lastDamageTaken = 0;
    this._tickCounter = 0;
    this._stateUpdatesSeen = 0;
    this._inputsSent = 0;
  }

  async initBrain(trainerUrl) {
    this.brain = new AgentBrain(this.agentId, trainerUrl, {
      modelAlias: this.modelAlias,
      modelVersion: this.modelVersion,
      policyFamily: this.policyFamily,
      archetypeId: this.archetypeId,
      rewardConfig: this.config.reward,
      reporter: this.reporter,
      strategyProvider: () => this.strategicBrain?.getStrategyVector?.() || null,
    });
    await this.brain.init();
    this.brain.resetMatch();
  }

  update(dt) {
    if (!this.connected || !this.data || this.matchEnded) return;
    if (!this.alive || this.data.health === 0) {
      this.fitness.updateSurvival(0);
      return;
    }

    this.fitness.updateSurvival(dt);
    this._tickCounter += 1;

    const myPos = this.gameState.getPosition(this.userID);
    if (myPos) {
      this.positionArray[0] = this.positionVector.x = myPos.x;
      this.positionArray[1] = this.positionVector.y = myPos.y;
      this.positionArray[2] = this.positionVector.z = myPos.z;
    }

    const input = this.inputBuffer[this.inputIndex];
    if (!input) return;

    if (this.coolDownTimer > 0) this.coolDownTimer -= dt;
    if (this.abilityTimer > 0) this.abilityTimer -= dt;
    if (this.searchClosestTimer > 0) this.searchClosestTimer -= dt;
    if (this.rttTimer > 0) this.rttTimer -= dt;

    const weaponRange = this.data.weaponTargetDistance || 20;

    if (this.searchClosestTimer <= 0) {
      this.closestEnemy = this.gameState.getClosestEnemy(this.userID, weaponRange);
      this.closestCrystal = this.gameState.getClosestCrystal(this.positionVector);
      this.searchClosestTimer = 0.5;
      if (this.rttTimer <= 0) {
        this.room.send("room:rtt");
        this.rttTimer = 3;
      }
    }

    if (
      this._tickCounter % (this.config.bot.decisionIntervalTicks || 3) === 0 &&
      this.brain &&
      !this.decisionInFlight
    ) {
      this.stateExtractor.setLastMove(this.lastMoveX, this.lastMoveZ);
      this.stateExtractor.setRecentCombat(this._lastDamageDealt, this._lastDamageTaken);
      const stateVector = this.stateExtractor.extract(
        this.gameState,
        this.userID,
        weaponRange,
        this.coolDownTimer > 0,
        this.data
      );

      this.decisionInFlight = true;
      this.brain.decide(stateVector)
        .then((decision) => this._applyDecision(decision))
        .catch(() => {})
        .finally(() => {
          this.decisionInFlight = false;
        });

      this.brain.recordStep({
        currentScore: this.data.score || 0,
        kills: this.data.kills || 0,
        deaths: this.data.deaths || 0,
        died: this._lastDied,
        gotKill: this._lastGotKill,
        damageDealt: this._lastDamageDealt,
        damageTaken: this._lastDamageTaken,
        survivalSeconds: dt,
        abilityUsed: this.lastAbilityUsed,
        shotAccuracy: this.fitness.getAccuracy(),
        done: false,
      });

      if (this._lastDamageDealt > 0 || this._lastDamageTaken > 0 || this._lastGotKill || this._lastDied) {
        this.lastCombatAt = Date.now();
      }

      this._lastGotKill = false;
      this._lastDied = false;
      this._lastDamageDealt = 0;
      this._lastDamageTaken = 0;
      this.lastAbilityUsed = false;
    }

    const distFromCenter = getArrayLength(this.positionArray);
    if (this.mapName === "arena" && distFromCenter > this.config.bot.arenaSafeSize) {
      const dirToCenter = getDirectionArray(this.positionArray, [0, 0, 0]);
      input.inputMove.set(new Float32Array(dirToCenter));
      input.target.set(new Float32Array([0, 0, 0]));
      input.animation = 2;
      input.speed = 7;
      this._sendInputsBatch([input]);
      this.inputIndex = (this.inputIndex + 1) % INPUT_BUFFER_SIZE;
      return;
    }

    this._sendPosition();
    this.inputIndex = (this.inputIndex + 1) % INPUT_BUFFER_SIZE;
  }

  _applyDecision(decision) {
    if (!decision || this.matchEnded) return;
    this.decisionsMade += 1;

    if (this.strategicBrain) {
      const crystalCtx = this.closestCrystal
        ? (() => {
            const dx = this.closestCrystal.x - this.positionArray[0];
            const dz = this.closestCrystal.z - this.positionArray[2];
            const len = Math.sqrt(dx * dx + dz * dz) || 1;
            return { dirX: dx / len, dirZ: dz / len };
          })()
        : null;

      decision = this.strategicBrain.modifyAction(decision, {
        hasEnemy: !!this.closestEnemy,
        enemyDistance: this.closestEnemy?.distance || 999,
        healthPercent: (this.data?.health || 100) / 100,
        score: this.data?.score || 0,
        kills: this.data?.kills || 0,
        deaths: this.data?.deaths || 0,
        posX: this.positionArray[0] / 60,
        posZ: this.positionArray[2] / 60,
        closestCrystal: crystalCtx,
      });
    }

    const tactical = this.tacticalController.stabilize(decision, {
      enemy: this.closestEnemy ? this._getEnemyContext() : null,
      crystal: this.closestCrystal ? this._getCrystalContext() : null,
      strategy: this.strategicBrain?.getStrategyVector?.() || null,
      healthPercent: (this.data?.health || 100) / 100,
      weaponRange: this.data?.weaponTargetDistance || 20,
      cooldownReady: this.coolDownTimer <= 0,
      abilityReady: this._hasReadyAbility(),
      passiveMs: Date.now() - Math.max(this.lastCombatAt || 0, this.lastShotAt || 0),
      position: this.positionVector,
      distanceFromCenter: getArrayLength(this.positionArray),
    });
    decision = tactical.action;
    if (tactical.overridden) {
      this.tacticalOverrides += 1;
      this.tacticalReasons = tactical.reasons;
      this.reporter?.incrementCounter("tacticalOverrides");
    }

    const input = this.inputBuffer[this.inputIndex];
    if (!input) return;

    let moveX = decision.moveX;
    let moveZ = decision.moveZ;
    const moveLen = Math.sqrt(moveX * moveX + moveZ * moveZ);
    if (moveLen > 0) {
      moveX /= moveLen;
      moveZ /= moveLen;
    }

    this.lastMoveX = moveX * 7;
    this.lastMoveZ = moveZ * 7;

    input.inputMove[0] = moveX;
    input.inputMove[1] = 0;
    input.inputMove[2] = moveZ;
    input.speed = 7;
    input.animation = moveLen > 0.1 ? 2 : 0;

    if (this.closestEnemy) {
      input.target[0] = this.closestEnemy.position.x + decision.aimOffsetX;
      input.target[1] = this.closestEnemy.position.y || 0;
      input.target[2] = this.closestEnemy.position.z + decision.aimOffsetZ;
    } else if (this.closestCrystal) {
      input.target[0] = this.closestCrystal.x;
      input.target[1] = this.closestCrystal.y || 0;
      input.target[2] = this.closestCrystal.z;
    } else {
      input.target[0] = this.positionArray[0] + moveX * 10;
      input.target[1] = 0;
      input.target[2] = this.positionArray[2] + moveZ * 10;
    }

    this.lastTargetPoint = {
      x: input.target[0],
      y: input.target[1],
      z: input.target[2],
    };

    if (decision.shouldShoot && this.closestEnemy && this.coolDownTimer <= 0 && this.data?.health > 0) {
      this._shoot(this.lastTargetPoint);
    }

    if (decision.shouldUseAbility && this.abilityTimer <= 0) {
      this._useAbility();
      this.abilityTimer = rand(5, 15);
    }
  }

  _shoot(targetPoint = null) {
    if (!this.closestEnemy || !this.data) return;
    const target = this._buildShootTarget(targetPoint);
    const buffer = encodeShoot(this.positionArray, target, this.data.weaponType || this.weaponType);
    this.room.send("room:player:shoot", buffer);
    this.fitness.recordShot();
    this.shotsFired += 1;
    this.lastShotAt = Date.now();
    this.reporter?.incrementCounter("shotsFired");
    this.coolDownTimer = this._getWeaponCooldownSeconds() + rand(0.05, 0.25);
  }

  _useAbility() {
    if (!this.data?.abilities) return;
    const ability = ABILITIES[Math.floor(Math.random() * ABILITIES.length)];
    const abilityData = this.data.abilities.get?.(ability);
    if (abilityData?.ready) {
      this.room.send("room:player:ability:use", { ability });
      this.fitness.recordAbilityUse();
      this.lastAbilityUsed = true;
    }
  }

  _sendPosition() {
    if (this.inputIndex === this.lastInputIndex) return;
    const inputs = [];
    let index = this.lastInputIndex;
    while (index !== this.inputIndex) {
      inputs.push(this.inputBuffer[index]);
      index = (index + 1) % INPUT_BUFFER_SIZE;
    }
    this.lastInputIndex = this.inputIndex;
    this._sendInputsBatch(inputs);
  }

  _sendInputsBatch(inputs) {
    const buffer = encodeInput(this.userID, inputs);
    this.room.send("room:player:input", buffer);
    this._inputsSent += inputs.length;
    this.reporter?.incrementCounter("inputsSent", inputs.length);
  }

  markStateUpdate() {
    this._stateUpdatesSeen += 1;
    this.reporter?.incrementCounter("stateUpdates");
  }

  getRuntimeStats() {
    return {
      inputsSent: this._inputsSent,
      stateUpdates: this._stateUpdatesSeen,
      decisionsMade: this.decisionsMade,
      shotsFired: this.shotsFired,
      tacticalOverrides: this.tacticalOverrides,
      modelAlias: this.modelAlias,
      modelVersion: this.brain?.modelVersion || 0,
      policyFamily: this.policyFamily,
      archetypeId: this.archetypeId,
    };
  }

  getFitness(weights) {
    return this.fitness.computeFitness(weights);
  }

  dispose() {
    this.matchEnded = true;
    this.connected = false;
    this.room = null;
    this.data = null;
    this.gameState = null;
    this.closestEnemy = null;
    this.closestCrystal = null;
  }

  _getEnemyContext() {
    if (!this.closestEnemy) return null;
    const dx = this.closestEnemy.position.x - this.positionArray[0];
    const dz = this.closestEnemy.position.z - this.positionArray[2];
    const length = Math.hypot(dx, dz) || 1;
    return {
      distance: this.closestEnemy.distance || length,
      dirX: dx / length,
      dirZ: dz / length,
    };
  }

  _getCrystalContext() {
    if (!this.closestCrystal) return null;
    const dx = this.closestCrystal.x - this.positionArray[0];
    const dz = this.closestCrystal.z - this.positionArray[2];
    const length = Math.hypot(dx, dz) || 1;
    return {
      dirX: dx / length,
      dirZ: dz / length,
      distance: this.closestCrystal.distance || length,
    };
  }

  _hasReadyAbility() {
    if (!this.data?.abilities?.values) return false;
    for (const ability of this.data.abilities.values()) {
      if (ability?.ready) return true;
    }
    return false;
  }

  _getWeaponCooldownSeconds() {
    const raw = Number(this.data?.weaponCoolDown);
    if (!Number.isFinite(raw) || raw <= 0) return 1;
    return raw > 10 ? raw / 1000 : raw;
  }

  _buildShootTarget(targetPoint = null) {
    const strategy = this.strategicBrain?.getStrategyVector?.() || {};
    const accuracy = Math.max(0, Math.min(1, Number(strategy.accuracy_focus ?? 0.5) || 0.5));
    const jitter = 0.1 + (1 - accuracy) * 0.6;
    const fallback = this.closestEnemy?.position || { x: 0, y: 0, z: 0 };
    const base = targetPoint || this.lastTargetPoint || fallback;
    return [
      Number(base.x ?? fallback.x) + rand(-jitter, jitter),
      Number(base.y ?? fallback.y) || 0,
      Number(base.z ?? fallback.z) + rand(-jitter, jitter),
    ];
  }
}

module.exports = { SmartBot };
