/**
 * Smart bot with per-agent ONNX inference brain.
 *
 * Each bot has a persistent agent identity. The neural network is loaded
 * from the Python training service and runs via onnxruntime-node.
 * Experience (state, action, reward) is sent back for PPO training.
 */

const { AgentBrain } = require("./AgentBrain");
const { StateExtractor } = require("./StateExtractor");
const { FitnessTracker } = require("../metrics/FitnessTracker");
const { encodeInput, encodeShoot, generateID } = require("../network/Protocol");
const { rand, getArrayLength, getDirectionArray } = require("../utils/math");

const INPUT_BUFFER_SIZE = 16;
const ABILITIES = Object.freeze(["rampage", "jump", "minePlanting"]);

class SmartBot {
  constructor(userID, _unused, config, agentId) {
    this.userID = userID;
    this.agentId = agentId || userID;
    this.config = config;
    this.mapName = config.server.mapName;
    this.weaponType = config.server.weaponType;

    // Brain (initialized later via initBrain)
    this.brain = null;
    this.stateExtractor = new StateExtractor();
    this.fitness = new FitnessTracker();

    // Connection state (set by Trainer)
    this.room = null;
    this.gameState = null;
    this.data = null;
    this.connected = false;
    this.alive = true;
    this.matchEnded = false;

    // Input buffer
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

    // Timers
    this.coolDownTimer = 0;
    this.abilityTimer = rand(5, 15);
    this.searchClosestTimer = 0;
    this.rttTimer = 0;
    this.experienceTimer = 0; // Send experience every N seconds

    // State
    this.closestEnemy = null;
    this.lastMoveX = 0;
    this.lastMoveZ = 0;
    this.positionArray = [0, 0, 0];
    this.positionVector = { x: 0, y: 0, z: 0 };

    // Per-tick reward tracking (reset each tick)
    this._lastGotKill = false;
    this._lastDied = false;
    this._lastDamageDealt = 0;
    this._tickCounter = 0;
  }

  async initBrain(trainerUrl) {
    this.brain = new AgentBrain(this.agentId, trainerUrl);
    await this.brain.init();
    this.brain.resetMatch();
  }

  /**
   * Main update loop — called at 60Hz by Trainer.
   */
  update(dt) {
    if (!this.connected || !this.data || this.matchEnded) return;
    if (!this.alive || this.data.health === 0) {
      this.fitness.updateSurvival(0);
      return;
    }

    this.fitness.updateSurvival(dt);
    this._tickCounter++;

    // Update position
    const myPos = this.gameState.getPosition(this.userID);
    if (myPos) {
      this.positionArray[0] = this.positionVector.x = myPos.x;
      this.positionArray[1] = this.positionVector.y = myPos.y;
      this.positionArray[2] = this.positionVector.z = myPos.z;
    }

    const input = this.inputBuffer[this.inputIndex];
    if (!input) return;

    // Timers
    if (this.coolDownTimer > 0) this.coolDownTimer -= dt;
    if (this.abilityTimer > 0) this.abilityTimer -= dt;
    if (this.searchClosestTimer > 0) this.searchClosestTimer -= dt;
    if (this.rttTimer > 0) this.rttTimer -= dt;

    const weaponRange = this.data.weaponTargetDistance || 20;

    // Search for enemies
    if (this.searchClosestTimer <= 0) {
      this.closestEnemy = this.gameState.getClosestEnemy(this.userID, weaponRange);
      this.searchClosestTimer = 0.5;
      if (this.rttTimer <= 0) {
        this.room.send("room:rtt");
        this.rttTimer = 3;
      }
    }

    // Get neural network decision (every 3rd tick for performance)
    if (this._tickCounter % 3 === 0 && this.brain) {
      this.stateExtractor.setLastMove(this.lastMoveX, this.lastMoveZ);
      const stateVector = this.stateExtractor.extract(
        this.gameState, this.userID, weaponRange,
        this.coolDownTimer > 0, this.data
      );

      // Synchronous-style: decide returns a promise but we handle it
      this.brain.decide(stateVector).then((decision) => {
        this._applyDecision(decision);
      }).catch(() => {});

      // Record experience step (every 3rd tick)
      const score = this.data.score || 0;
      const kills = this.data.kills || 0;
      const deaths = this.data.deaths || 0;
      this.brain.recordStep(
        score, kills, deaths,
        this._lastDied, this._lastGotKill, this._lastDamageDealt,
        false
      );

      // Reset per-tick flags
      this._lastGotKill = false;
      this._lastDied = false;
      this._lastDamageDealt = 0;
    }

    // Arena boundary override
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

    // Send position
    this._sendPosition();
    this.inputIndex = (this.inputIndex + 1) % INPUT_BUFFER_SIZE;
  }

  _applyDecision(decision) {
    if (!decision || this.matchEnded) return;

    const input = this.inputBuffer[this.inputIndex];
    if (!input) return;

    // Movement
    let moveX = decision.moveX;
    let moveZ = decision.moveZ;
    const moveLen = Math.sqrt(moveX * moveX + moveZ * moveZ);
    if (moveLen > 0) { moveX /= moveLen; moveZ /= moveLen; }

    this.lastMoveX = moveX * 7;
    this.lastMoveZ = moveZ * 7;

    input.inputMove[0] = moveX;
    input.inputMove[1] = 0;
    input.inputMove[2] = moveZ;
    input.speed = 7;
    input.animation = moveLen > 0.1 ? 2 : 0;

    // Aim
    if (this.closestEnemy) {
      input.target[0] = this.closestEnemy.position.x + decision.aimOffsetX;
      input.target[1] = this.closestEnemy.position.y || 0;
      input.target[2] = this.closestEnemy.position.z + decision.aimOffsetZ;
    } else {
      input.target[0] = this.positionArray[0] + moveX * 10;
      input.target[1] = 0;
      input.target[2] = this.positionArray[2] + moveZ * 10;
    }

    // Shoot
    if (decision.shouldShoot && this.closestEnemy && this.coolDownTimer <= 0 && this.data?.health > 0) {
      this._shoot();
    }

    // Ability
    if (decision.shouldUseAbility && this.abilityTimer <= 0) {
      this._useAbility();
      this.abilityTimer = rand(5, 15);
    }
  }

  _shoot() {
    if (!this.closestEnemy || !this.data) return;
    const target = [
      this.closestEnemy.position.x + rand(-1, 1),
      (this.closestEnemy.position.y || 0),
      this.closestEnemy.position.z + rand(-1, 1),
    ];
    const buffer = encodeShoot(this.positionArray, target, this.data.weaponType || this.weaponType);
    this.room.send("room:player:shoot", buffer);
    this.fitness.recordShot();
    this.coolDownTimer = (this.data.weaponCoolDown || 1) + rand(0.3, 1.5);
  }

  _useAbility() {
    if (!this.data?.abilities) return;
    const ability = ABILITIES[Math.floor(Math.random() * ABILITIES.length)];
    const abilityData = this.data.abilities.get?.(ability);
    if (abilityData?.ready) {
      this.room.send("room:player:ability:use", { ability });
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
  }
}

module.exports = { SmartBot };
