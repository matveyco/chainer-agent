const { Client } = require("colyseus.js");
const { SmartBot } = require("../bot/SmartBot");
const { GameState } = require("../game/GameState");
const { generateID } = require("../network/Protocol");
const logger = require("../utils/logger");

const JSON_HEADERS = { "Content-Type": "application/json" };

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: { ...JSON_HEADERS, ...options.headers },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 404) {
    throw new Error(`${options.method || "GET"} ${url} -> ${res.status} ${JSON.stringify(data)}`);
  }
  return data;
}

function cleanPublicAddress(value) {
  return String(value || "")
    .replace(/["']/g, "")
    .replace(/\s/g, "")
    .replace(/^https?:\/\//, "");
}

class RoomCoordinator {
  constructor({ roomIndex, roster, config, runtimeState, strategicBrains, mode = "training", jobId = null, templateId = null }) {
    this.roomIndex = roomIndex;
    this.roster = roster;
    this.config = config;
    this.runtimeState = runtimeState;
    this.strategicBrains = strategicBrains;
    this.mode = mode;
    this.jobId = jobId;
    this.templateId = templateId;
    this.roomState = this.runtimeState.ensureRoom(roomIndex);
    this.requiredAgents = Math.max(
      1,
      Math.ceil(this.roster.length * (this.config.rooms?.minReadyRatio || 0.5))
    );
  }

  async runMatch() {
    const gameState = new GameState();
    const sessions = this._createSessions(gameState);

    this.runtimeState.updateRoom(this.roomIndex, {
      status: "queueing",
      phase: "queueing",
      targetAgents: this.roster.length,
      assignedAgents: 0,
      connectedAgents: 0,
      livePlayers: 0,
      stateUpdates: 0,
      inputsSent: 0,
      roomId: null,
      publicAddress: null,
      lastError: null,
      mode: this.mode,
      jobId: this.jobId,
      templateId: this.templateId,
    });

    try {
      await this._queueBots(sessions);
      const selection = await this._collectAssignments(sessions);
      if (!selection || selection.sessions.length < this.requiredAgents) {
        this.runtimeState.incrementCounter("roomAbandons");
        this.runtimeState.updateRoom(this.roomIndex, {
          status: "abandoned",
          phase: "assignment_failed",
          assignedAgents: selection?.sessions.length || 0,
          lastError: selection ? "not enough assigned seats" : "assignment timeout",
          mode: this.mode,
          jobId: this.jobId,
          templateId: this.templateId,
        });
        return { connectedCount: 0, roomId: null };
      }

      await this._leaveQueue(sessions.filter((session) => !selection.sessionIds.has(session.userID)));
      const connectedSessions = await this._connectAssignedSessions(selection, gameState);

      if (connectedSessions.length < this.requiredAgents) {
        this.runtimeState.incrementCounter("roomAbandons");
        this.runtimeState.updateRoom(this.roomIndex, {
          status: "abandoned",
          phase: "connect_failed",
          connectedAgents: connectedSessions.length,
          lastError: `too few bots (${connectedSessions.length})`,
          mode: this.mode,
          jobId: this.jobId,
          templateId: this.templateId,
        });
        await this._cleanupSessions(sessions);
        return { connectedCount: connectedSessions.length, roomId: selection.roomId };
      }

      await this._runMatchLoop(selection, connectedSessions);
      const agentResults = await this._finalizeMatch(connectedSessions);
      const summary = this._buildMatchSummary(selection, connectedSessions, agentResults);
      this.runtimeState.recordMatchSummary(summary);

      const top = summary.agentResults[0];

      if (top) {
        logger.info(
          `Room ${this.roomIndex} done: #1 ${top.id} (score=${top.score} k=${top.kills} d=${top.deaths})`
        );
      }

      this.runtimeState.updateRoom(this.roomIndex, {
        status: "completed",
        phase: "completed",
        lastMatchEndedAt: new Date().toISOString(),
        connectedAgents: connectedSessions.length,
        mode: this.mode,
        jobId: this.jobId,
        templateId: this.templateId,
      });

      return {
        connectedCount: connectedSessions.length,
        roomId: selection.roomId,
        summary,
      };
    } catch (err) {
      this.runtimeState.noteRoomError(this.roomIndex, err);
      return {
        connectedCount: 0,
        roomId: null,
        error: err.message,
      };
    } finally {
      await this._cleanupSessions(sessions);
      gameState.clear();
    }
  }

  _createSessions(gameState) {
    return this.roster.map((agent, index) => {
      const userID = `${agent.agentId}_${generateID(4)}`;
      const bot = new SmartBot(userID, null, this.config, agent.agentId, {
        displayName: agent.displayName,
        modelAlias: agent.modelAlias,
        modelVersion: agent.modelVersion,
        policyFamily: agent.policyFamily,
        archetypeId: agent.archetypeId,
        reporter: this.runtimeState,
      });

      if (this.strategicBrains.has(agent.agentId)) {
        bot.strategicBrain = this.strategicBrains.get(agent.agentId);
      } else if (bot.strategicBrain) {
        this.strategicBrains.set(agent.agentId, bot.strategicBrain);
      }

      bot.gameState = gameState;

      return {
        index,
        userID,
        agent,
        bot,
        queued: false,
        queueLeft: false,
        assignment: null,
        room: null,
        client: null,
        connected: false,
      };
    });
  }

  async _queueBots(sessions) {
    const endpoint = this.config.server.endpoint;
    for (const session of sessions) {
      await fetchJSON(`${endpoint}/matchmaker/join-queue`, {
        method: "POST",
        body: JSON.stringify({
          userID: session.userID,
          roomName: this.config.server.roomName,
          mapName: this.config.server.mapName,
          forceCreateRoom: false,
        }),
      });
      session.queued = true;
      session.queuedAt = Date.now();
      this.runtimeState.incrementCounter("queueJoinAttempts");
      await this._sleep(this.config.bot.clientStaggerMs || 100);
    }
  }

  async _collectAssignments(sessions) {
    const endpoint = this.config.server.endpoint;
    const deadline = Date.now() + (this.config.rooms?.assignmentTimeoutMs || 90000);
    const pollMs = this.config.rooms?.assignmentPollMs || 1500;
    let firstAssignmentAt = null;
    let bestSelection = null;

    while (Date.now() < deadline) {
      const unsettled = sessions.filter((session) => !session.assignment);
      if (unsettled.length === 0) break;

      const responses = await Promise.all(unsettled.map(async (session) => {
        try {
          const data = await fetchJSON(
            `${endpoint}/matchmaker/user-queue-position/${session.userID}`
          );
          return { session, data };
        } catch (err) {
          return { session, error: err };
        }
      }));

      for (const response of responses) {
        if (response.data?.data?.room) {
          response.session.assignment = {
            room: response.data.data.room,
            assignedAt: Date.now(),
          };
          this.runtimeState.incrementCounter("queueAssignments");
          if (!firstAssignmentAt) firstAssignmentAt = response.session.assignment.assignedAt;
        }
      }

      bestSelection = this._selectBestAssignmentGroup(sessions);
      if (
        bestSelection &&
        bestSelection.sessions.length >= this.requiredAgents &&
        firstAssignmentAt &&
        Date.now() - firstAssignmentAt >= pollMs
      ) {
        break;
      }

      await this._sleep(pollMs);
    }

    if (!bestSelection) {
      this.runtimeState.incrementCounter("queueTimeouts");
      await this._leaveQueue(sessions);
      return null;
    }

    const queueWaitMs = bestSelection.sessions.reduce(
      (sum, session) => sum + ((session.assignment?.assignedAt || Date.now()) - session.queuedAt),
      0
    ) / Math.max(bestSelection.sessions.length, 1);

    this.runtimeState.observe("queueWaitMsAvg", queueWaitMs);
    this.runtimeState.updateRoom(this.roomIndex, {
      phase: "assigned",
      status: "assigned",
      assignedAgents: bestSelection.sessions.length,
      queueWaitMs: Math.round(queueWaitMs),
      roomId: bestSelection.roomId,
      publicAddress: bestSelection.publicAddress,
      mode: this.mode,
      jobId: this.jobId,
      templateId: this.templateId,
    });

    return bestSelection;
  }

  _selectBestAssignmentGroup(sessions) {
    const grouped = new Map();
    for (const session of sessions) {
      const roomId = session.assignment?.room?.roomId;
      if (!roomId) continue;
      if (!grouped.has(roomId)) {
        grouped.set(roomId, {
          roomId,
          publicAddress: session.assignment.room.publicAddress,
          sessions: [],
          sessionIds: new Set(),
        });
      }
      const group = grouped.get(roomId);
      group.sessions.push(session);
      group.sessionIds.add(session.userID);
    }

    return [...grouped.values()].sort((a, b) => b.sessions.length - a.sessions.length)[0] || null;
  }

  async _connectAssignedSessions(selection) {
    const joinOptions = { weaponType: this.config.server.weaponType };
    if (this.config.server.authKey) joinOptions.OAuthAPIKey = this.config.server.authKey;

    const host = cleanPublicAddress(selection.publicAddress);
    const clientUrl = `https://${host}`;

    this.runtimeState.updateRoom(this.roomIndex, {
      status: "connecting",
      phase: "connecting",
      roomId: selection.roomId,
      publicAddress: host,
      lastMatchStartedAt: new Date().toISOString(),
      mode: this.mode,
      jobId: this.jobId,
      templateId: this.templateId,
    });

    const connectedSessions = [];
    const onDispose = (() => {
      let settled = false;
      return () => {
        if (settled) return false;
        settled = true;
        return true;
      };
    })();

    const results = await Promise.all(selection.sessions.map(async (session, index) => {
      this.runtimeState.incrementCounter("joinAttempts");
      try {
        session.client = new Client(clientUrl);
        session.room = await session.client.joinById(selection.roomId, {
          ...joinOptions,
          userID: session.userID,
        });

        session.bot.room = session.room;
        session.bot.connected = true;
        session.connected = true;
        this._setupBotHandlers(session, index === 0, onDispose);

        session.room.send("room:player:loaded", {
          profile: {
            userName: session.agent.displayName,
            wallet: "0x0",
            models: [{ alias: session.agent.modelAlias }],
            textures: [],
          },
        });

        await session.bot.initBrain(this.config.trainerUrl);
        session.bot.data = session.room.state?.players?.get?.(session.userID);
        session.agent.loadedModelVersion = session.bot.brain?.modelVersion || 0;
        connectedSessions.push(session);
        this.runtimeState.incrementCounter("joinSuccesses");
        return true;
      } catch (err) {
        this.runtimeState.incrementCounter("joinFailures");
        this._classifyJoinError(err);
        logger.warn(`Room ${this.roomIndex} bot ${session.index} failed: ${err.message}`);
        return false;
      }
    }));

    const connected = results.filter(Boolean).length;
    this.runtimeState.updateRoom(this.roomIndex, {
      status: connected >= this.requiredAgents ? "running" : "abandoned",
      phase: connected >= this.requiredAgents ? "running" : "connect_failed",
      connectedAgents: connected,
      fillRatio: +(connected / this.roster.length).toFixed(3),
      mode: this.mode,
      jobId: this.jobId,
      templateId: this.templateId,
    });

    await this._sleep(1000);
    for (const session of connectedSessions) {
      if (!session.bot.data && session.room) {
        session.bot.data = session.room.state?.players?.get?.(session.userID);
      }
    }

    return connectedSessions;
  }

  _setupBotHandlers(session, isPrimary, onDispose) {
    const { bot, room } = session;
    const safe = (fn, context) => (payload) => {
      try {
        fn(payload);
      } catch (err) {
        if (context === "schema") {
          this.runtimeState.incrementCounter("schemaCrashes");
        }
        this.runtimeState.noteRoomError(this.roomIndex, err, { context });
        if (onDispose()) {
          try { room.leave(); } catch {}
        }
      }
    };

    const noop = () => {};

    room.onMessage("room:player:joined", safe((data) => {
      bot.gameState.addPlayer(data.userID, { x: 0, y: 0, z: 0 });
      this.runtimeState.updateRoom(this.roomIndex, {
        livePlayers: bot.gameState.getLivePlayerCount(),
      });
      if (data.userID === bot.userID) {
        bot.data = room.state?.players?.get?.(bot.userID);
      }
    }, "player_joined"));

    room.onMessage("room:player:die", safe((data) => {
      bot.gameState.handleDeath(data.userID);
      this.runtimeState.updateRoom(this.roomIndex, {
        livePlayers: bot.gameState.getLivePlayerCount(),
      });
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
    }, "player_die"));

    room.onMessage("room:player:hit", safe((data) => {
      bot.gameState.updateHealth(data.userID, data.newHealth);
      if (data.ownerID === bot.userID && data.userID !== bot.userID) {
        bot.fitness.recordDamageDealt(data.damage || 0);
        bot._lastDamageDealt += data.damage || 0;
      }
      if (data.userID === bot.userID) {
        bot.fitness.recordDamageTaken(data.damage || 0);
        bot._lastDamageTaken += data.damage || 0;
      }
    }, "player_hit"));

    room.onMessage("room:player:respawn", safe((data) => {
      bot.gameState.handleRespawn(data.userID);
      if (data.userID === bot.userID) {
        bot.alive = true;
        bot.fitness.recordRespawn();
      }
      this.runtimeState.updateRoom(this.roomIndex, {
        livePlayers: bot.gameState.getLivePlayerCount(),
      });
    }, "player_respawn"));

    room.onMessage("room:player:left", safe((data) => {
      bot.gameState.removePlayer(data.userID);
      this.runtimeState.updateRoom(this.roomIndex, {
        livePlayers: bot.gameState.getLivePlayerCount(),
      });
    }, "player_left"));

    room.onMessage("room:time", safe((data) => {
      if (data?.left !== undefined && data?.spent !== undefined) {
        bot.stateExtractor.setMatchTime(data.left, data.left + data.spent);
      }
    }, "time"));

    if (isPrimary) {
      room.onMessage("room:state:update", safe((data) => {
        bot.gameState.processStateUpdate(data);
        bot.markStateUpdate();
        this.runtimeState.updateRoom(this.roomIndex, {
          stateUpdates: bot.getRuntimeStats().stateUpdates,
          livePlayers: bot.gameState.getLivePlayerCount(),
        });
      }, "schema"));
      room.onMessage("room:dispose", () => {
        onDispose();
      });
      room.onLeave(() => {
        onDispose();
      });
    } else {
      room.onMessage("room:state:update", noop);
      room.onMessage("room:dispose", noop);
      room.onLeave(noop);
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

    if (typeof room.onError === "function") {
      room.onError((code, message) => {
        const err = new Error(message || String(code));
        this._classifyJoinError(err);
        this.runtimeState.noteRoomError(this.roomIndex, err, { code });
      });
    }
  }

  async _runMatchLoop(selection, connectedSessions) {
    let matchResolved = false;
    const matchEndPromise = new Promise((resolve) => {
      const primary = connectedSessions[0];
      const finish = () => {
        if (matchResolved) return;
        matchResolved = true;
        resolve();
      };

      if (primary?.room) {
        primary.room.onMessage("room:dispose", finish);
        primary.room.onLeave(finish);
      }

      setTimeout(finish, this.config.bot.matchTimeout);
    });

    let lastTime = performance.now();
    const gameLoop = setInterval(() => {
      const now = performance.now();
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      for (const session of connectedSessions) {
        try {
          session.bot.update(dt);
          const stats = session.bot.getRuntimeStats();
          this.runtimeState.updateRoom(this.roomIndex, {
            inputsSent: stats.inputsSent,
          });
        } catch (err) {
          this.runtimeState.noteRoomError(this.roomIndex, err, { context: "bot_update" });
        }
      }
    }, 1000 / 60);

    const flushInterval = setInterval(async () => {
      for (const session of connectedSessions) {
        if (session.bot.brain) await session.bot.brain.flush();
      }
    }, 10000);

    await matchEndPromise;
    clearInterval(gameLoop);
    clearInterval(flushInterval);

    this.runtimeState.updateRoom(this.roomIndex, {
      phase: "finalizing",
      status: "finalizing",
      roomId: selection.roomId,
      mode: this.mode,
      jobId: this.jobId,
      templateId: this.templateId,
    });
  }

  async _finalizeMatch(connectedSessions) {
    const agentResults = [];
    for (const session of connectedSessions) {
      const summary = {
        score: session.bot.data?.score || 0,
        kills: session.bot.data?.kills || 0,
        deaths: session.bot.data?.deaths || 0,
        damageDealt: session.bot.fitness.damageDealt || 0,
        damageTaken: session.bot.fitness.damageTaken || 0,
        survivalTime: session.bot.fitness.survivalTime || 0,
        abilitiesUsed: session.bot.fitness.abilitiesUsed || 0,
      };

      if (session.bot.brain) {
        session.bot.brain.recordStep({
          currentScore: summary.score,
          kills: summary.kills,
          deaths: summary.deaths,
          damageDealt: 0,
          damageTaken: 0,
          survivalSeconds: 0,
          done: true,
        });
        await session.bot.brain.flush();
        await session.bot.brain.reportEpisode(summary);
      }

      if (session.bot.strategicBrain) {
        this.strategicBrains.set(session.bot.agentId, session.bot.strategicBrain);
        await session.bot.strategicBrain.analyzeMatch(summary).catch(() => {});
      }

      agentResults.push(this._buildAgentResult(session, summary));
    }

    this._persistProfiles();
    agentResults.sort((left, right) => right.score - left.score);
    return agentResults;
  }

  async _cleanupSessions(sessions) {
    for (const session of sessions) {
      if (session.room) {
        try {
          session.room.leave();
          session.room.removeAllListeners();
        } catch {}
      }
      session.bot.dispose();
    }
    await this._leaveQueue(sessions);
  }

  async _leaveQueue(sessions) {
    if (!sessions.length) return;
    const endpoint = this.config.server.endpoint;
    await Promise.all(sessions.map(async (session) => {
      if (!session.queued || session.queueLeft) return;
      session.queueLeft = true;
      try {
        await fetch(`${endpoint}/matchmaker/leave-queue/${session.userID}`, { method: "DELETE" });
        this.runtimeState.incrementCounter("queueLeaves");
      } catch {}
    }));
  }

  _classifyJoinError(err) {
    const message = err?.message || "";
    if (/reserved seat/i.test(message)) {
      this.runtimeState.incrementCounter("seatExpired");
    }
    if (/locked/i.test(message)) {
      this.runtimeState.incrementCounter("lockedRooms");
    }
  }

  _persistProfiles() {
    try {
      const profiles = {};
      for (const [agentId, brain] of this.strategicBrains.entries()) {
        profiles[agentId] = brain.getProfile();
      }
      const file = this.config.persistence?.profilesFile || "data/agent_profiles.json";
      const resolved = require("path").resolve(file);
      require("fs").mkdirSync(require("path").dirname(resolved), { recursive: true });
      require("fs").writeFileSync(resolved, JSON.stringify(profiles, null, 2));
    } catch {}
  }

  _buildAgentResult(session, summary) {
    const runtime = session.bot.getRuntimeStats();
    return {
      id: session.bot.agentId,
      agentId: session.bot.agentId,
      displayName: session.agent.displayName,
      policyFamily: session.agent.policyFamily || session.bot.policyFamily,
      archetypeId: session.agent.archetypeId || session.bot.archetypeId,
      modelAlias: session.agent.modelAlias || session.bot.modelAlias,
      modelVersion: runtime.modelVersion || session.agent.loadedModelVersion || session.agent.modelVersion || 0,
      evaluationSide: session.agent.evaluationSide || null,
      score: summary.score,
      kills: summary.kills,
      deaths: summary.deaths,
      damageDealt: summary.damageDealt,
      damageTaken: summary.damageTaken,
      survivalTime: summary.survivalTime,
      abilitiesUsed: summary.abilitiesUsed,
      inputsSent: runtime.inputsSent,
      stateUpdates: runtime.stateUpdates,
    };
  }

  _buildMatchSummary(selection, connectedSessions, agentResults) {
    const startedAt = this.roomState.lastMatchStartedAt || new Date().toISOString();
    const finishedAt = new Date().toISOString();
    return {
      roomIndex: this.roomIndex,
      mode: this.mode,
      jobId: this.jobId,
      templateId: this.templateId,
      roomId: selection.roomId,
      publicAddress: cleanPublicAddress(selection.publicAddress),
      assignedAgents: selection.sessions.length,
      connectedAgents: connectedSessions.length,
      fillRatio: +(connectedSessions.length / Math.max(this.roster.length, 1)).toFixed(3),
      startedAt,
      finishedAt,
      durationMs: Math.max(0, new Date(finishedAt).getTime() - new Date(startedAt).getTime()),
      winner: agentResults[0] || null,
      agentResults,
    };
  }

  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

module.exports = { RoomCoordinator };
