/**
 * Evolution loop controller.
 * Orchestrates: create bots → connect → play match → collect fitness → evolve → repeat.
 *
 * Key insight: matchmaking requires ALL bots to join the queue before a room is created.
 * So we use a two-phase approach:
 *   Phase 1: All bots POST to join-queue (fast, no waiting)
 *   Phase 2: Poll until room assigned, then all bots joinById via Colyseus
 */

const { Client } = require("colyseus.js");
const { Population } = require("./Population");
const { Genome } = require("./Genome");
const { SmartBot } = require("../bot/SmartBot");
const { GameState } = require("../game/GameState");
const { GenerationLog } = require("../metrics/GenerationLog");
const { generateID } = require("../network/Protocol");
const logger = require("../utils/logger");

const jsonHeaders = { "Content-Type": "application/json" };

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: { ...jsonHeaders, ...options.headers },
  });
  return res.json().catch(() => ({}));
}

class Trainer {
  /**
   * @param {Object} config - Full config object
   * @param {Function} onGenerationComplete - Callback with generation summary
   */
  constructor(config, onGenerationComplete = null) {
    this.config = config;
    this.population = new Population(config.evolution);
    this.onGenerationComplete = onGenerationComplete;
    this.running = false;
    this.bots = [];
    this.gameLoop = null;
  }

  /**
   * Load a saved population to resume training.
   */
  resumeFrom(filePath) {
    const data = Genome.loadPopulation(filePath);
    this.population.loadFromJSON(data.genomes);
    this.population.generation = data.generation;
    logger.info(`Resumed from generation ${data.generation}`);
  }

  /**
   * Run the evolution loop indefinitely.
   */
  async run() {
    this.running = true;

    while (this.running) {
      try {
        await this.runGeneration();
      } catch (err) {
        logger.error("Generation failed:", err.message);
        await this._sleep(5000);
      }

      // Brief pause between generations
      await this._sleep(3000);
    }
  }

  /**
   * Run a single generation.
   *
   * Two-phase connection:
   *   1. Queue all bots via POST /matchmaker/join-queue
   *   2. Poll until room assigned, then Colyseus joinById for each
   */
  async runGeneration() {
    const generation = this.population.getGeneration();
    logger.setGeneration(generation);
    logger.info(`--- Generation ${generation} ---`);

    const endpoint = this.config.server.endpoint;
    const genomes = this.population.getGenomes();
    const gameState = new GameState();
    const genLog = new GenerationLog();
    this.bots = [];

    // Create SmartBot for each genome
    const userIDs = [];
    for (let i = 0; i < genomes.length; i++) {
      const userID = `neat_${generation}_${i}_${generateID(4)}`;
      userIDs.push(userID);
      const bot = new SmartBot(userID, genomes[i], this.config);
      this.bots.push(bot);
    }

    // ========== PHASE 1: Queue all bots ==========
    logger.info(`Queueing ${this.bots.length} bots...`);

    for (let i = 0; i < this.bots.length; i++) {
      try {
        await fetchJSON(`${endpoint}/matchmaker/join-queue`, {
          method: "POST",
          body: JSON.stringify({
            userID: userIDs[i],
            roomName: this.config.server.roomName,
            mapName: this.config.server.mapName,
            forceCreateRoom: false,
          }),
        });
      } catch (err) {
        logger.warn(`Bot ${i} queue join failed: ${err.message}`);
      }
      await this._sleep(50); // Small stagger
    }

    logger.info("All bots queued. Waiting for room assignment...");

    // ========== PHASE 2: Poll for room assignment ==========
    let roomData = null;
    for (let attempt = 0; attempt < 60; attempt++) {
      if (!this.running) return;
      await this._sleep(2000);

      try {
        const posRes = await fetchJSON(
          `${endpoint}/matchmaker/user-queue-position/${userIDs[0]}`
        );
        if (posRes.data?.room) {
          roomData = posRes.data.room;
          logger.info(`Room assigned: ${roomData.roomId} @ ${roomData.publicAddress}`);
          break;
        }
        logger.debug(`Waiting... pos: ${posRes.data?.position} queue: ${posRes.data?.queueLength}`);
      } catch (err) {
        logger.debug(`Poll error: ${err.message}`);
      }
    }

    if (!roomData) {
      logger.error("No room assigned after 120s. Cleaning up queue...");
      await this._cleanupQueue(endpoint, userIDs);
      throw new Error("Room assignment timeout");
    }

    // ========== PHASE 3: Colyseus joinById for each bot ==========
    const host = roomData.publicAddress.replace(/^https?:\/\//, "");
    const clientUrl = `https://${host}`;

    let matchEnded = false;
    const matchEndPromise = new Promise((resolve) => {
      const onDispose = () => {
        if (!matchEnded) {
          matchEnded = true;
          logger.info("Room disposed");
          resolve();
        }
      };

      // Connect each bot via Colyseus
      (async () => {
        for (let i = 0; i < this.bots.length; i++) {
          if (matchEnded || !this.running) break;
          const bot = this.bots[i];

          try {
            const client = new Client(clientUrl);
            const room = await client.joinById(roomData.roomId, {
              userID: userIDs[i],
              weaponType: this.config.server.weaponType,
            });

            // Wire up the bot
            bot.room = room;
            bot.gameState = gameState;

            // Set up message handlers
            this._setupBotHandlers(bot, room, gameState, i === 0, onDispose);

            // Send loaded profile
            const names = ["Neo","Trinity","Morpheus","Oracle","Cipher","Tank","Storm","Blaze","Shadow","Ace","Nova","Fury","Echo","Viper","Phoenix","Hawk"];
            room.send("room:player:loaded", {
              profile: {
                userName: `AI_${names[i % names.length]}_${userIDs[i].slice(-4)}`,
                wallet: "0x0",
                models: [],
                textures: [],
              },
            });

            // Try to get player data immediately (fallback for if joined event already fired)
            setTimeout(() => {
              if (!bot.data) {
                bot.data = room.state?.players?.get?.(userIDs[i]);
              }
              bot.connected = true;
            }, 500);

            logger.debug(`Bot ${i + 1}/${this.bots.length} connected to room`);
          } catch (err) {
            logger.warn(`Bot ${i} Colyseus join failed: ${err.message}`);
          }

          await this._sleep(this.config.bot.clientStaggerMs);
        }

        const connectedCount = this.bots.filter((b) => b.connected).length;
        logger.info(`${connectedCount}/${this.bots.length} bots connected. Match playing...`);
      })();

      // Match timeout
      setTimeout(() => {
        if (!matchEnded) {
          matchEnded = true;
          logger.info("Match timeout reached");
          resolve();
        }
      }, this.config.bot.matchTimeout);
    });

    // Start 60Hz game loop
    let lastTime = performance.now();
    this.gameLoop = setInterval(() => {
      const now = performance.now();
      const dt = (now - lastTime) / 1000;
      lastTime = now;

      for (const bot of this.bots) {
        try {
          bot.update(dt);
        } catch {}
      }
    }, 1000 / 60);

    // Wait for match end
    await matchEndPromise;

    // Stop game loop
    clearInterval(this.gameLoop);
    this.gameLoop = null;

    // Collect fitness
    const connectedBots = this.bots.filter((b) => b.connected);
    logger.info(`Match ended. Collecting fitness from ${connectedBots.length} bots.`);

    for (let i = 0; i < this.bots.length; i++) {
      const fitness = this.bots[i].getFitness(this.config.fitness);
      this.population.setFitness(i, fitness);
      genLog.addResult(this.bots[i].userID, this.bots[i].fitness, fitness);
    }

    // Log generation stats
    const summary = genLog.appendToLog(this.config.persistence.logsDir, generation);
    logger.info(
      `Gen ${generation} results: best=${summary.bestFitness} avg=${summary.avgFitness} ` +
      `K/D=${summary.bestKD}/${summary.avgKD} kills=${summary.totalKills}`
    );

    // Save best genome
    const bestGenome = this.population.getBestGenome();
    Genome.saveBest(bestGenome, generation, this.config.persistence.bestDir);

    // Save full population periodically
    if (generation % this.config.persistence.savePopulationEvery === 0) {
      Genome.savePopulation(this.population, generation, this.config.persistence.generationsDir);
      logger.info(`Population snapshot saved`);
    }

    // Notify callback
    if (this.onGenerationComplete) {
      this.onGenerationComplete({
        ...summary,
        neuronRange: this.population.getNeuronRange(),
        connectionRange: this.population.getConnectionRange(),
      });
    }

    // Clean up
    for (const bot of this.bots) {
      try {
        if (bot.room) {
          bot.room.leave();
          bot.room.removeAllListeners();
        }
      } catch {}
      bot.dispose();
    }
    this.bots = [];
    gameState.clear();

    // Evolve
    await this.population.evolve();
    logger.info(`Evolved to generation ${this.population.getGeneration()}`);
  }

  /**
   * Set up message handlers for a bot's room connection.
   */
  _setupBotHandlers(bot, room, gameState, isFirst, onDispose) {
    const noop = () => {};

    // Player joined
    room.onMessage("room:player:joined", (data) => {
      gameState.addPlayer(data.userID, { x: 0, y: 0, z: 0 });
      if (data.userID === bot.userID) {
        bot.data = room.state?.players?.get(bot.userID);
        if (bot.data) {
          gameState.setPlayerData(bot.userID, bot.data);
        }
      }
    });

    // Combat events — all bots track their own kills/hits
    room.onMessage("room:player:die", (data) => {
      gameState.handleDeath(data.userID);
      if (data.killerID === bot.userID && data.userID !== bot.userID) {
        bot.fitness.recordKill();
      }
      if (data.userID === bot.userID) {
        bot.fitness.recordDeath();
        bot.alive = false;
        setTimeout(() => {
          if (bot.room && !bot.matchEnded) {
            bot.room.send("room:player:respawn");
          }
        }, this.config.bot.respawnDelay * 1000);
      }
    });

    room.onMessage("room:player:hit", (data) => {
      gameState.updateHealth(data.userID, data.newHealth);
      if (data.ownerID === bot.userID && data.userID !== bot.userID) {
        bot.fitness.recordDamageDealt(data.damage || 0);
      }
      if (data.userID === bot.userID) {
        bot.fitness.recordDamageTaken(data.damage || 0);
      }
    });

    room.onMessage("room:player:respawn", (data) => {
      gameState.handleRespawn(data.userID);
      if (data.userID === bot.userID) {
        bot.alive = true;
        bot.fitness.recordRespawn();
      }
    });

    room.onMessage("room:player:left", (data) => {
      gameState.removePlayer(data.userID);
    });

    // Only first bot handles state updates and dispose (optimization)
    if (isFirst) {
      room.onMessage("room:state:update", (data) => {
        gameState.processStateUpdate(data);
      });
      room.onMessage("room:dispose", onDispose);
      room.onLeave(() => onDispose());
    } else {
      room.onMessage("room:state:update", noop);
      room.onMessage("room:dispose", noop);
    }

    // Register all other handlers to prevent warnings
    room.onMessage("room:rtt", noop);
    room.onMessage("room:time", noop);
    room.onMessage("room:player:heal", noop);
    room.onMessage("room:player:shield", noop);
    room.onMessage("room:player:loaded", noop);
    room.onMessage("room:player:rejoined", noop);
    room.onMessage("room:leaderboard:update", noop);
    room.onMessage("room:breakable:destroy", noop);
    room.onMessage("room:invite:userID", noop);
    room.onMessage("player:consecutive-kills", noop);
    room.onMessage("player:first-kill", noop);
    room.onMessage("player:one-shot-kill", noop);
    room.onMessage("player:several-kills-at-once", noop);
    room.onMessage("player:shield-deflected", noop);
    room.onMessage("room:session:replaced", noop);
    room.onMessage("__playground_message_types", noop);
  }

  async _cleanupQueue(endpoint, userIDs) {
    for (const id of userIDs) {
      try {
        await fetch(`${endpoint}/matchmaker/leave-queue/${id}`, { method: "DELETE" });
      } catch {}
    }
  }

  stop() {
    this.running = false;
    if (this.gameLoop) {
      clearInterval(this.gameLoop);
      this.gameLoop = null;
    }
  }

  saveState() {
    return Genome.savePopulation(
      this.population,
      this.population.getGeneration(),
      this.config.persistence.generationsDir
    );
  }

  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

module.exports = { Trainer };
