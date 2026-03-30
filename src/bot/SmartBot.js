/**
 * Neural network-driven bot.
 * Replaces the random decisions of player.js with neural network inference.
 * Same protocol: input buffer, protobuf encoding, shoot mechanics, respawn, arena bounds.
 */

const { BotBrain } = require("./BotBrain");
const { StateExtractor } = require("./StateExtractor");
const { FitnessTracker } = require("../metrics/FitnessTracker");
const { Connection } = require("../network/Connection");
const { encodeInput, encodeShoot, generateID } = require("../network/Protocol");
const { rand, normalizeArray, getArrayLength, getDirectionArray, clamp } = require("../utils/math");
const logger = require("../utils/logger");

const INPUT_BUFFER_SIZE = 16;
const ABILITIES = Object.freeze(["rampage", "jump", "minePlanting"]);

// Bot profile names
const NAMES = [
  "Neo", "Trinity", "Morpheus", "Oracle", "Cipher", "Tank", "Dozer", "Switch",
  "Apoc", "Mouse", "Niobe", "Ghost", "Sparks", "Jax", "Hawk", "Viper",
  "Phoenix", "Storm", "Blaze", "Shadow", "Ace", "Nova", "Fury", "Echo",
];

class SmartBot {
  /**
   * @param {string} userID - Unique bot identifier
   * @param {Object} network - Neataptic Network (genome)
   * @param {Object} config - Bot configuration
   */
  constructor(userID, network, config) {
    this.userID = userID;
    this.config = config;
    this.mapName = config.server.mapName;
    this.weaponType = config.server.weaponType;

    // Neural network
    this.brain = new BotBrain(network);
    this.stateExtractor = new StateExtractor();

    // Fitness tracking
    this.fitness = new FitnessTracker();

    // Connection
    this.connection = null;
    this.room = null;
    this.gameState = null; // Set externally (shared across all bots)

    // Player state
    this.data = null; // From room.state.players
    this.positionArray = [0, 0, 0];
    this.positionVector = { x: 0, y: 0, z: 0 };
    this.alive = true;

    // Input buffer (same as loadtest player.js)
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

    // State
    this.closestEnemy = null;
    this.lastMoveX = 0;
    this.lastMoveZ = 0;

    // Match state
    this.connected = false;
    this.matchEnded = false;
  }

  /**
   * Connect to the game server.
   * @param {string} endpoint
   * @param {Object} gameState - Shared GameState instance
   * @param {boolean} isFirstBot - Whether this is the first bot (handles state updates)
   * @param {Function} onDispose - Called when room disposes
   */
  async connect(endpoint, gameState, isFirstBot, onDispose) {
    this.gameState = gameState;
    this.connection = new Connection(endpoint);

    const callbacks = {
      onPlayerJoined: (data) => {
        gameState.addPlayer(data.userID, { x: 0, y: 0, z: 0 });
        if (data.userID === this.userID) {
          this.data = this.room?.state?.players?.get(this.userID);
          if (this.data) {
            gameState.setPlayerData(this.userID, this.data);
          }
        }
      },
      onPlayerLeft: (data) => {
        gameState.removePlayer(data.userID);
      },
      onPlayerDie: (data) => {
        gameState.handleDeath(data.userID);

        // Track kills
        if (data.killerID === this.userID && data.userID !== this.userID) {
          this.fitness.recordKill();
        }

        // Track own death
        if (data.userID === this.userID) {
          this.fitness.recordDeath();
          this.alive = false;
          // Auto-respawn
          setTimeout(() => {
            if (this.room && !this.matchEnded) {
              this.room.send("room:player:respawn");
            }
          }, this.config.bot.respawnDelay * 1000);
        }
      },
      onPlayerHit: (data) => {
        gameState.updateHealth(data.userID, data.newHealth);

        // Track damage dealt by this bot
        if (data.ownerID === this.userID && data.userID !== this.userID) {
          this.fitness.recordDamageDealt(data.damage || 0);
        }

        // Track damage taken
        if (data.userID === this.userID) {
          this.fitness.recordDamageTaken(data.damage || 0);
        }
      },
      onPlayerRespawn: (data) => {
        gameState.handleRespawn(data.userID);
        if (data.userID === this.userID) {
          this.alive = true;
          this.fitness.recordRespawn();
        }
      },
      onDispose: () => {
        this.matchEnded = true;
        if (onDispose) onDispose();
      },
      onLeave: () => {
        this.matchEnded = true;
      },
      onTime: () => {},
      onRtt: () => {},
    };

    // Only the first bot processes state updates (optimization from loadtest)
    if (isFirstBot) {
      callbacks.onStateUpdate = (data) => {
        gameState.processStateUpdate(data);
      };
    }

    this.room = await this.connection.connect(
      this.userID,
      this.config.server.roomName,
      this.config.server.mapName,
      this.weaponType,
      false,
      callbacks
    );

    // Send loaded profile
    const name = NAMES[Math.floor(Math.random() * NAMES.length)];
    this.connection.sendLoaded(`AI_${name}_${this.userID.slice(-4)}`);

    this.connected = true;
    return this.room;
  }

  /**
   * Main update loop — called at 60Hz.
   * Extract state → brain.decide() → apply actions → send protobuf.
   */
  update(dt) {
    if (!this.connected || !this.data || this.matchEnded) return;
    if (!this.alive || this.data.health === 0) {
      this.fitness.updateSurvival(0); // Not alive, no survival time
      return;
    }

    // Track survival time
    this.fitness.updateSurvival(dt);

    // Update position from game state
    const myPos = this.gameState.getPosition(this.userID);
    if (myPos) {
      this.positionArray[0] = this.positionVector.x = myPos.x;
      this.positionArray[1] = this.positionVector.y = myPos.y;
      this.positionArray[2] = this.positionVector.z = myPos.z;
    }

    const input = this.inputBuffer[this.inputIndex];
    if (!input) return;

    // Update timers
    if (this.coolDownTimer > 0) this.coolDownTimer -= dt;
    if (this.abilityTimer > 0) this.abilityTimer -= dt;
    if (this.searchClosestTimer > 0) this.searchClosestTimer -= dt;
    if (this.rttTimer > 0) this.rttTimer -= dt;

    // Get weapon range from player data
    const weaponRange = this.data.weaponTargetDistance || 50;

    // Search for closest enemy periodically
    if (this.searchClosestTimer <= 0) {
      this.closestEnemy = this.gameState.getClosestEnemy(this.userID, weaponRange);
      this.searchClosestTimer = 0.5; // Search every 0.5s

      // RTT ping
      if (this.rttTimer <= 0) {
        this.room.send("room:rtt");
        this.rttTimer = 3;
      }
    }

    // Extract state and get neural network decision
    this.stateExtractor.setLastMove(this.lastMoveX, this.lastMoveZ);
    const inputVector = this.stateExtractor.extract(
      this.gameState,
      this.userID,
      weaponRange,
      this.coolDownTimer > 0
    );

    const decision = this.brain.decide(inputVector);

    // Apply movement
    let moveX = decision.moveX;
    let moveZ = decision.moveZ;
    const moveLen = Math.sqrt(moveX * moveX + moveZ * moveZ);
    if (moveLen > 0) {
      moveX /= moveLen;
      moveZ /= moveLen;
    }

    this.lastMoveX = moveX * 7; // speed 7
    this.lastMoveZ = moveZ * 7;

    // Arena boundary override (safety check — same as loadtest)
    const distFromCenter = getArrayLength(this.positionArray);
    if (this.mapName === "arena" && distFromCenter > this.config.bot.arenaSafeSize) {
      const dirToCenter = getDirectionArray(this.positionArray, [0, 0, 0]);
      input.inputMove.set(new Float32Array(dirToCenter));
      input.target.set(new Float32Array([0, 0, 0]));
      input.animation = 2;
      input.speed = 7;

      // Force send and return early
      this.sendInputs([input]);
      this.inputIndex = (this.inputIndex + 1) % INPUT_BUFFER_SIZE;
      return;
    }

    // Set input buffer
    input.inputMove[0] = moveX;
    input.inputMove[1] = 0;
    input.inputMove[2] = moveZ;
    input.speed = 7;
    input.animation = moveLen > 0.1 ? 2 : 0; // 2 = running, 0 = idle

    // Set aim target
    if (this.closestEnemy) {
      input.target[0] = this.closestEnemy.position.x + decision.aimOffsetX;
      input.target[1] = this.closestEnemy.position.y || 0;
      input.target[2] = this.closestEnemy.position.z + decision.aimOffsetZ;
    } else {
      // Aim in movement direction
      input.target[0] = this.positionArray[0] + moveX * 10;
      input.target[1] = 0;
      input.target[2] = this.positionArray[2] + moveZ * 10;
    }

    // Shoot decision
    if (
      decision.shouldShoot &&
      this.closestEnemy &&
      this.coolDownTimer <= 0 &&
      this.data.health > 0
    ) {
      this._shoot();
    }

    // Ability decision
    if (decision.shouldUseAbility && this.abilityTimer <= 0) {
      this._useAbility();
      this.abilityTimer = rand(5, 15);
    }

    // Send position
    this._sendPosition();
    this.inputIndex = (this.inputIndex + 1) % INPUT_BUFFER_SIZE;
  }

  _shoot() {
    if (!this.closestEnemy || !this.data) return;

    const origin = this.positionArray;
    const target = [
      this.closestEnemy.position.x + rand(-2, 2),
      (this.closestEnemy.position.y || 0) + rand(-0.5, 0.5),
      this.closestEnemy.position.z + rand(-2, 2),
    ];

    const buffer = encodeShoot(origin, target, this.data.weaponType || this.weaponType);
    this.room.send("room:player:shoot", buffer);

    this.fitness.recordShot();
    this.coolDownTimer = (this.data.weaponCoolDown || 1) + rand(0.5, 2);
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

    const buffer = encodeInput(this.userID, inputs);
    this.room.send("room:player:input", buffer);
  }

  sendInputs(inputs) {
    const buffer = encodeInput(this.userID, inputs);
    this.room.send("room:player:input", buffer);
  }

  /**
   * Get this bot's fitness score.
   */
  getFitness(weights) {
    return this.fitness.computeFitness(weights);
  }

  dispose() {
    this.matchEnded = true;
    this.connected = false;
    if (this.connection) {
      this.connection.dispose();
      this.connection = null;
    }
    this.room = null;
    this.data = null;
    this.gameState = null;
    this.closestEnemy = null;
  }
}

module.exports = { SmartBot };
