/**
 * Training loop controller — 24/7 continuous operation.
 *
 * Runs multiple rooms in parallel. Each room has a subset of agents.
 * After every SELECTION_INTERVAL matches, triggers natural selection:
 *   - Cull 5 weakest agents
 *   - Clone 5 strongest with weight mutations
 *
 * Designed for non-stop operation with auto-recovery.
 */

const { Client } = require("colyseus.js");
const { SmartBot } = require("../bot/SmartBot");
const { GameState } = require("../game/GameState");
const { generateID } = require("../network/Protocol");
const logger = require("../utils/logger");

const jsonHeaders = { "Content-Type": "application/json" };

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, { ...options, headers: { ...jsonHeaders, ...options.headers } });
  return res.json().catch(() => ({}));
}

class Trainer {
  constructor(config, onMatchComplete = null) {
    this.config = config;
    this.onMatchComplete = onMatchComplete;
    this.running = false;
    this.totalMatches = 0;
    this.startTime = Date.now();

    // Number of parallel rooms
    this.numRooms = config.numRooms || 2;
    // Agents per room
    this.agentsPerRoom = config.evolution.populationSize;
    // Total agents across all rooms
    this.totalAgents = this.numRooms * this.agentsPerRoom;

    // Natural selection interval (every N total matches across all rooms)
    this.selectionInterval = config.selectionInterval || 10;
    this.numCull = config.numCull || 5;

    // Build persistent agent IDs
    this.allAgentIds = [];
    for (let i = 0; i < this.totalAgents; i++) {
      this.allAgentIds.push(`agent_${i}`);
    }

    logger.info(`Configured: ${this.numRooms} rooms × ${this.agentsPerRoom} agents = ${this.totalAgents} total`);
    logger.info(`Selection every ${this.selectionInterval} matches, culling ${this.numCull} weakest`);
  }

  async run() {
    this.running = true;

    // Check trainer service
    try {
      const health = await fetchJSON(`${this.config.trainerUrl}/health`);
      logger.info(`Trainer: ${health.status} on ${health.device}`);
    } catch (err) {
      logger.error(`Cannot reach trainer at ${this.config.trainerUrl}: ${err.message}`);
      logger.error("Start the Python training service first: python training/trainer.py");
      return;
    }

    logger.info("Starting 24/7 training loop...");

    // Run rooms in parallel, forever
    while (this.running) {
      const roomPromises = [];

      // Split agents across rooms
      for (let r = 0; r < this.numRooms; r++) {
        const start = r * this.agentsPerRoom;
        const end = start + this.agentsPerRoom;
        const roomAgents = this.allAgentIds.slice(start, end);

        roomPromises.push(
          this._runRoom(r, roomAgents).catch((err) => {
            logger.error(`Room ${r} failed: ${err.message}`);
          })
        );

        // Stagger room starts by 5s
        if (r < this.numRooms - 1) {
          await this._sleep(5000);
        }
      }

      // Wait for all rooms to finish their match
      await Promise.all(roomPromises);

      // Natural selection check
      if (this.totalMatches > 0 && this.totalMatches % this.selectionInterval === 0) {
        await this._triggerSelection();
      }

      // Log uptime
      const uptime = this._formatUptime(Date.now() - this.startTime);
      logger.info(`Uptime: ${uptime} | Total matches: ${this.totalMatches}`);

      // Brief pause between rounds
      await this._sleep(2000);
    }
  }

  async _runRoom(roomIndex, agentIds) {
    const endpoint = this.config.server.endpoint;
    const gameState = new GameState();
    const bots = [];
    const userIDs = [];

    for (let i = 0; i < agentIds.length; i++) {
      const userID = `${agentIds[i]}_${generateID(4)}`;
      userIDs.push(userID);
      bots.push(new SmartBot(userID, null, this.config, agentIds[i]));
    }

    // Phase 1: Queue
    for (let i = 0; i < bots.length; i++) {
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

    // Phase 2: Wait for room
    let roomData = null;
    for (let attempt = 0; attempt < 60; attempt++) {
      if (!this.running) return;
      await this._sleep(2000);
      try {
        const posRes = await fetchJSON(`${endpoint}/matchmaker/user-queue-position/${userIDs[0]}`);
        if (posRes.data?.room) {
          roomData = posRes.data.room;
          break;
        }
      } catch {}
    }

    if (!roomData) {
      for (const id of userIDs) {
        fetch(`${endpoint}/matchmaker/leave-queue/${id}`, { method: "DELETE" }).catch(() => {});
      }
      throw new Error(`Room ${roomIndex}: assignment timeout`);
    }

    logger.info(`Room ${roomIndex}: ${roomData.roomId} (${agentIds.length} agents)`);

    // Phase 3: Connect
    const host = roomData.publicAddress.replace(/^https?:\/\//, "");
    const clientUrl = `https://${host}`;

    let matchEnded = false;
    const matchEndPromise = new Promise((resolve) => {
      const onDispose = () => { if (!matchEnded) { matchEnded = true; resolve(); } };

      (async () => {
        for (let i = 0; i < bots.length; i++) {
          if (matchEnded || !this.running) break;
          const bot = bots[i];

          try {
            const client = new Client(clientUrl);
            const room = await client.joinById(roomData.roomId, {
              userID: userIDs[i],
              weaponType: this.config.server.weaponType,
            });

            bot.room = room;
            bot.gameState = gameState;
            this._setupBotHandlers(bot, room, gameState, i === 0, onDispose);

            room.send("room:player:loaded", {
              profile: { userName: agentIds[i], wallet: "0x0", models: [], textures: [] },
            });

            await bot.initBrain(this.config.trainerUrl);
            setTimeout(() => {
              if (!bot.data) bot.data = room.state?.players?.get?.(userIDs[i]);
              bot.connected = true;
            }, 500);
          } catch (err) {
            logger.warn(`Room ${roomIndex} bot ${i} join failed: ${err.message}`);
          }

          await this._sleep(this.config.bot.clientStaggerMs);
        }
      })();

      setTimeout(() => { if (!matchEnded) { matchEnded = true; resolve(); } }, this.config.bot.matchTimeout);
    });

    // 60Hz loop
    let lastTime = performance.now();
    const gameLoop = setInterval(() => {
      const now = performance.now();
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      for (const bot of bots) { try { bot.update(dt); } catch {} }
    }, 1000 / 60);

    // Periodic flush
    const flushInterval = setInterval(async () => {
      for (const bot of bots) { if (bot.brain) await bot.brain.flush(); }
    }, 10000);

    await matchEndPromise;

    clearInterval(gameLoop);
    clearInterval(flushInterval);

    // End of match: flush + report
    for (const bot of bots) {
      if (bot.brain) {
        const score = bot.data?.score || 0;
        const kills = bot.data?.kills || 0;
        const deaths = bot.data?.deaths || 0;
        bot.brain.recordStep(score, kills, deaths, false, false, 0, true);
        await bot.brain.flush();
        await bot.brain.reportEpisode(score, kills, deaths);
      }
    }

    this.totalMatches++;

    // Log match results
    const results = bots
      .filter((b) => b.data)
      .map((b) => ({ id: b.agentId, s: b.data?.score || 0, k: b.data?.kills || 0, d: b.data?.deaths || 0 }))
      .sort((a, b) => b.s - a.s);

    if (results.length > 0) {
      const top = results[0];
      logger.info(
        `Room ${roomIndex} done: #1 ${top.id} (score=${top.s} k=${top.k} d=${top.d}) | ` +
        `total_kills=${results.reduce((s, r) => s + r.k, 0)}`
      );
    }

    // Cleanup
    for (const bot of bots) {
      try { if (bot.room) { bot.room.leave(); bot.room.removeAllListeners(); } } catch {}
      bot.dispose();
    }
    gameState.clear();
  }

  async _triggerSelection() {
    logger.info("=== TRIGGERING NATURAL SELECTION ===");
    try {
      const res = await fetchJSON(`${this.config.trainerUrl}/select`, {
        method: "POST",
        body: JSON.stringify({ num_cull: this.numCull }),
      });
      if (res.ok) {
        logger.info(`Selection complete. ${res.agents?.length} agents active.`);
      }
    } catch (err) {
      logger.error(`Selection failed: ${err.message}`);
    }
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
      if (data.userID === bot.userID) bot.fitness.recordDamageTaken(data.damage || 0);
    });

    room.onMessage("room:player:respawn", (data) => {
      gameState.handleRespawn(data.userID);
      if (data.userID === bot.userID) { bot.alive = true; bot.fitness.recordRespawn(); }
    });

    room.onMessage("room:player:left", (data) => gameState.removePlayer(data.userID));

    room.onMessage("room:time", (data) => {
      if (data?.left !== undefined && data?.spent !== undefined) {
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
  }

  _formatUptime(ms) {
    const h = Math.floor(ms / 3600000);
    const m = Math.floor((ms % 3600000) / 60000);
    return `${h}h${m}m`;
  }

  _sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }
}

module.exports = { Trainer };
