/**
 * Training loop controller.
 *
 * Orchestrates: create bots → connect → play match → collect experience → send to trainer → repeat.
 *
 * Each bot has a PERSISTENT identity (agent_0, agent_1, ...) that survives across matches.
 * Experience is sent to the Python PPO training service which trains per-agent policy networks.
 */

const { Client } = require("colyseus.js");
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
  constructor(config, onMatchComplete = null) {
    this.config = config;
    this.onMatchComplete = onMatchComplete;
    this.running = false;
    this.bots = [];
    this.gameLoop = null;
    this.matchCount = 0;

    // Persistent agent IDs
    this.agentIds = [];
    for (let i = 0; i < config.evolution.populationSize; i++) {
      this.agentIds.push(`agent_${i}`);
    }
  }

  async run() {
    this.running = true;
    logger.info(`Starting training with ${this.agentIds.length} persistent agents`);
    logger.info(`Trainer service: ${this.config.trainerUrl}`);

    // Check trainer health
    try {
      const health = await fetchJSON(`${this.config.trainerUrl}/health`);
      logger.info(`Trainer status: ${health.status}, device: ${health.device}`);
    } catch (err) {
      logger.error(`Cannot reach trainer at ${this.config.trainerUrl}: ${err.message}`);
      logger.error("Start the Python training service first: python training/trainer.py");
      return;
    }

    while (this.running) {
      try {
        await this.runMatch();
        this.matchCount++;
      } catch (err) {
        logger.error("Match failed:", err.message);
        await this._sleep(5000);
      }
      await this._sleep(3000);
    }
  }

  async runMatch() {
    logger.info(`--- Match ${this.matchCount} ---`);

    const endpoint = this.config.server.endpoint;
    const gameState = new GameState();
    this.bots = [];

    // Create SmartBot for each persistent agent
    const userIDs = [];
    for (let i = 0; i < this.agentIds.length; i++) {
      const userID = `${this.agentIds[i]}_${generateID(4)}`;
      userIDs.push(userID);
      const bot = new SmartBot(userID, null, this.config, this.agentIds[i]);
      this.bots.push(bot);
    }

    // Phase 1: Queue all bots
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
      } catch {}
      await this._sleep(50);
    }

    // Phase 2: Poll for room
    let roomData = null;
    for (let attempt = 0; attempt < 60; attempt++) {
      if (!this.running) return;
      await this._sleep(2000);
      try {
        const posRes = await fetchJSON(`${endpoint}/matchmaker/user-queue-position/${userIDs[0]}`);
        if (posRes.data?.room) {
          roomData = posRes.data.room;
          logger.info(`Room: ${roomData.roomId}`);
          break;
        }
      } catch {}
    }

    if (!roomData) {
      logger.error("No room assigned");
      for (const id of userIDs) {
        fetch(`${endpoint}/matchmaker/leave-queue/${id}`, { method: "DELETE" }).catch(() => {});
      }
      throw new Error("Room assignment timeout");
    }

    // Phase 3: Connect all bots
    const host = roomData.publicAddress.replace(/^https?:\/\//, "");
    const clientUrl = `https://${host}`;

    let matchEnded = false;
    const matchEndPromise = new Promise((resolve) => {
      const onDispose = () => {
        if (!matchEnded) {
          matchEnded = true;
          resolve();
        }
      };

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

            bot.room = room;
            bot.gameState = gameState;
            this._setupBotHandlers(bot, room, gameState, i === 0, onDispose);

            // Send loaded profile
            room.send("room:player:loaded", {
              profile: {
                userName: `${this.agentIds[i]}`,
                wallet: "0x0",
                models: [],
                textures: [],
              },
            });

            // Initialize brain and set connected after delay
            await bot.initBrain(this.config.trainerUrl);
            setTimeout(() => {
              if (!bot.data) bot.data = room.state?.players?.get?.(userIDs[i]);
              bot.connected = true;
            }, 500);
          } catch (err) {
            logger.warn(`Bot ${i} join failed: ${err.message}`);
          }

          await this._sleep(this.config.bot.clientStaggerMs);
        }

        const connectedCount = this.bots.filter((b) => b.connected).length;
        logger.info(`${connectedCount}/${this.bots.length} bots connected`);
      })();

      setTimeout(() => {
        if (!matchEnded) {
          matchEnded = true;
          resolve();
        }
      }, this.config.bot.matchTimeout);
    });

    // 60Hz game loop
    let lastTime = performance.now();
    this.gameLoop = setInterval(() => {
      const now = performance.now();
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      for (const bot of this.bots) {
        try { bot.update(dt); } catch {}
      }
    }, 1000 / 60);

    // Flush experience periodically
    const flushInterval = setInterval(async () => {
      for (const bot of this.bots) {
        if (bot.brain) await bot.brain.flush();
      }
    }, 10000);

    await matchEndPromise;

    clearInterval(this.gameLoop);
    clearInterval(flushInterval);
    this.gameLoop = null;

    // End of match: flush remaining experience and report episode
    logger.info("Match ended. Flushing experience...");
    for (const bot of this.bots) {
      if (bot.brain) {
        // Record final done step
        const score = bot.data?.score || 0;
        const kills = bot.data?.kills || 0;
        const deaths = bot.data?.deaths || 0;
        bot.brain.recordStep(score, kills, deaths, false, false, 0, true);
        await bot.brain.flush();
        await bot.brain.reportEpisode(score, kills, deaths);
        logger.info(
          `  ${bot.agentId}: score=${score} kills=${kills} deaths=${deaths}`
        );
      }
    }

    // Get stats from trainer
    try {
      const stats = await fetchJSON(`${this.config.trainerUrl}/stats`);
      if (stats.agents) {
        logger.info(`Trainer: ${stats.num_agents} agents on ${stats.device}`);
        for (const a of stats.agents.slice(0, 5)) {
          logger.info(
            `  ${a.agent_id}: v${a.model_version} score=${a.avg_score_50} kd=${a.avg_kd_50}`
          );
        }
      }
    } catch {}

    // Notify callback
    if (this.onMatchComplete) {
      const genLog = new GenerationLog();
      for (const bot of this.bots) {
        genLog.addResult(bot.agentId, bot.fitness, bot.getFitness(this.config.fitness));
      }
      this.onMatchComplete(genLog.getSummary(this.matchCount));
    }

    // Cleanup
    for (const bot of this.bots) {
      try {
        if (bot.room) { bot.room.leave(); bot.room.removeAllListeners(); }
      } catch {}
      bot.dispose();
    }
    this.bots = [];
    gameState.clear();
  }

  _setupBotHandlers(bot, room, gameState, isFirst, onDispose) {
    const noop = () => {};

    room.onMessage("room:player:joined", (data) => {
      gameState.addPlayer(data.userID, { x: 0, y: 0, z: 0 });
      if (data.userID === bot.userID) {
        bot.data = room.state?.players?.get?.(bot.userID);
      }
    });

    room.onMessage("room:player:die", (data) => {
      gameState.handleDeath(data.userID);
      if (data.killerID === bot.userID && data.userID !== bot.userID) {
        bot.fitness.recordKill();
        bot._lastGotKill = true;
      }
      if (data.userID === bot.userID) {
        bot.fitness.recordDeath();
        bot.alive = false;
        bot._lastDied = true;
        setTimeout(() => {
          if (bot.room && !bot.matchEnded) bot.room.send("room:player:respawn");
        }, this.config.bot.respawnDelay * 1000);
      }
    });

    room.onMessage("room:player:hit", (data) => {
      gameState.updateHealth(data.userID, data.newHealth);
      if (data.ownerID === bot.userID && data.userID !== bot.userID) {
        bot.fitness.recordDamageDealt(data.damage || 0);
        bot._lastDamageDealt += (data.damage || 0);
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

    room.onMessage("room:player:left", (data) => gameState.removePlayer(data.userID));

    room.onMessage("room:time", (data) => {
      if (data && data.left !== undefined && data.spent !== undefined) {
        bot.stateExtractor?.setMatchTime(data.left, data.left + data.spent);
      }
    });

    if (isFirst) {
      room.onMessage("room:state:update", (data) => gameState.processStateUpdate(data));
      room.onMessage("room:dispose", onDispose);
      room.onLeave(() => onDispose());
    } else {
      room.onMessage("room:state:update", noop);
      room.onMessage("room:dispose", noop);
    }

    room.onMessage("room:rtt", noop);
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

  stop() {
    this.running = false;
    if (this.gameLoop) { clearInterval(this.gameLoop); this.gameLoop = null; }
  }

  _sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }
}

module.exports = { Trainer };
