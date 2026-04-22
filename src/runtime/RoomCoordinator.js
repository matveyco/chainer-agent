const { Client } = require("colyseus.js");
const { SmartBot } = require("../bot/SmartBot");
const { GameState } = require("../game/GameState");
const { generateID } = require("../network/Protocol");
const { Matchmaker } = require("../network/Matchmaker");
const logger = require("../utils/logger");

function cleanPublicAddress(value) {
  return String(value || "")
    .replace(/["']/g, "")
    .replace(/\s/g, "")
    .replace(/^https?:\/\//, "");
}

function numberOrZero(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

class RoomCoordinator {
  constructor({
    roomIndex,
    roster,
    config,
    runtimeState,
    strategicBrains,
    mode = "training",
    jobId = null,
    templateId = null,
    track = "training",
    resolvedAlias = null,
    resolvedVersion = null,
    queuePlan = null,
    claimBackendRoom = null,
    releaseBackendRoom = null,
    getClaimedBackendRoomIds = null,
  }) {
    this.roomIndex = roomIndex;
    this.roster = roster;
    this.config = config;
    this.runtimeState = runtimeState;
    this.strategicBrains = strategicBrains;
    this.mode = mode;
    this.jobId = jobId;
    this.templateId = templateId;
    this.track = track;
    this.resolvedAlias = resolvedAlias;
    this.resolvedVersion = Number(resolvedVersion || 0);
    this.queuePlan = queuePlan || null;
    this.claimBackendRoom = claimBackendRoom;
    this.releaseBackendRoom = releaseBackendRoom;
    this.getClaimedBackendRoomIds = getClaimedBackendRoomIds;
    this.matchmaker = new Matchmaker(this.config.server.endpoint, {
      authKey: this.config.server.authKey,
      pollMs: this.config.rooms?.assignmentPollMs || 1500,
    });
    this.claimedRoomId = null;
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
      track: this.track,
      resolvedAlias: this.resolvedAlias,
      resolvedVersion: this.resolvedVersion,
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
          track: this.track,
          resolvedAlias: this.resolvedAlias,
          resolvedVersion: this.resolvedVersion,
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
          track: this.track,
          resolvedAlias: this.resolvedAlias,
          resolvedVersion: this.resolvedVersion,
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
          track: this.track,
          resolvedAlias: this.resolvedAlias,
          resolvedVersion: this.resolvedVersion,
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
      this._releaseClaimedBackendRoom();
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
        role: agent.role,
        mode: this.mode,
        track: this.track,
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
    const roomName = this.queuePlan?.roomName || this.config.server.roomName;
    const mapName = this.queuePlan?.mapName || this.config.server.mapName;
    for (const session of sessions) {
      const joined = await this.matchmaker.joinQueue(session.userID, roomName, mapName, false);
      session.queued = true;
      session.queuedAt = Date.now();
      if (joined?.room) {
        session.assignment = {
          room: joined.room,
          assignedAt: Date.now(),
        };
        this.runtimeState.incrementCounter("queueAssignments");
      }
      this.runtimeState.incrementCounter("queueJoinAttempts");
      await this._sleep(this.config.bot.clientStaggerMs || 100);
    }
  }

  async _collectAssignments(sessions) {
    const deadline = Date.now() + (this.config.rooms?.assignmentTimeoutMs || 90000);
    const pollMs = this.config.rooms?.assignmentPollMs || 1500;
    let firstAssignmentAt = null;
    let bestSelection = null;
    let selected = null;

    while (Date.now() < deadline) {
      const unsettled = sessions.filter((session) => !session.assignment);
      if (unsettled.length > 0) {
        const responses = await Promise.all(unsettled.map(async (session) => {
          try {
            const data = await this.matchmaker.getQueuePosition(session.userID);
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
      }

      const claimedRoomIds = this.getClaimedBackendRoomIds?.(this.roomIndex) || new Set();
      await this._requeueClaimedAssignments(sessions, claimedRoomIds);

      bestSelection = this._selectBestAssignmentGroup(sessions, claimedRoomIds);
      if (
        bestSelection &&
        bestSelection.sessions.length >= this.requiredAgents &&
        firstAssignmentAt &&
        Date.now() - firstAssignmentAt >= pollMs &&
        this._claimSelection(bestSelection)
      ) {
        selected = bestSelection;
        break;
      }

      await this._sleep(pollMs);
    }

    if (!selected) {
      this.runtimeState.incrementCounter("queueTimeouts");
      await this._leaveQueue(sessions);
      return null;
    }

    const queueWaitMs = selected.sessions.reduce(
      (sum, session) => sum + ((session.assignment?.assignedAt || Date.now()) - session.queuedAt),
      0
    ) / Math.max(selected.sessions.length, 1);

    this.runtimeState.observe("queueWaitMsAvg", queueWaitMs);
    this.runtimeState.updateRoom(this.roomIndex, {
      phase: "assigned",
      status: "assigned",
      assignedAgents: selected.sessions.length,
      queueWaitMs: Math.round(queueWaitMs),
      roomId: selected.roomId,
      publicAddress: selected.publicAddress,
      mode: this.mode,
      jobId: this.jobId,
      templateId: this.templateId,
      track: this.track,
      resolvedAlias: this.resolvedAlias,
      resolvedVersion: this.resolvedVersion,
    });

    return selected;
  }

  _selectBestAssignmentGroup(sessions, excludedRoomIds = new Set()) {
    const grouped = new Map();
    for (const session of sessions) {
      const roomId = session.assignment?.room?.roomId;
      if (!roomId || excludedRoomIds.has(roomId)) continue;
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

  _claimSelection(selection) {
    if (!selection?.roomId || !this.claimBackendRoom) {
      this.claimedRoomId = selection?.roomId || null;
      return true;
    }
    const claimed = this.claimBackendRoom(selection.roomId, {
      roomIndex: this.roomIndex,
      track: this.track,
      mode: this.mode,
    });
    if (claimed) {
      this.claimedRoomId = selection.roomId;
    }
    return claimed;
  }

  _releaseClaimedBackendRoom() {
    if (!this.claimedRoomId || !this.releaseBackendRoom) return;
    this.releaseBackendRoom(this.claimedRoomId, { roomIndex: this.roomIndex });
    this.claimedRoomId = null;
  }

  async _requeueClaimedAssignments(sessions, claimedRoomIds) {
    if (!claimedRoomIds?.size) return;
    const conflicted = sessions.filter((session) => {
      const roomId = session.assignment?.room?.roomId;
      if (!roomId || !claimedRoomIds.has(roomId)) return false;
      const now = Date.now();
      if (session.lastConflictRequeueAt && now - session.lastConflictRequeueAt < (this.config.rooms?.requeueBackoffMs || 3000)) {
        return false;
      }
      session.lastConflictRequeueAt = now;
      return true;
    });

    if (!conflicted.length) return;

    this.runtimeState.recordEvent?.("warn", "requeueing conflicted room assignments", {
      roomIndex: this.roomIndex,
      conflictedCount: conflicted.length,
      track: this.track,
    });
    await this._leaveQueue(conflicted);
    for (const session of conflicted) {
      session.assignment = null;
      session.queued = false;
      session.queueLeft = false;
      session.queuedAt = null;
    }
    await this._sleep(250 + Math.floor(Math.random() * 500));
    await this._queueBots(conflicted);
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
      track: this.track,
      resolvedAlias: this.resolvedAlias,
      resolvedVersion: this.resolvedVersion,
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

    // Snapshot static obstacles from the room's Colyseus state once after
    // join so bots can perceive walls / crates / barrels via raycasts.
    // The shared GameState is reused across all sessions in this room.
    const primary = connectedSessions[0];
    if (primary?.room?.state) {
      try {
        const obstacles = this._collectObstaclesFromRoomState(primary.room.state);
        primary.bot.gameState.setStaticObstacles(obstacles);
        this.runtimeState.recordEvent("info", "obstacles loaded", {
          roomIndex: this.roomIndex,
          count: obstacles.length,
        });
      } catch (err) {
        this.runtimeState.recordEvent("warn", "obstacles snapshot failed", {
          roomIndex: this.roomIndex,
          error: err.message,
        });
      }
    }

    return connectedSessions;
  }

  /**
   * Pull static obstacle geometry out of the Colyseus room state. The
   * server exposes dynamicColliders, acidBarrels, and (sometimes)
   * breakables — we read whatever is present and normalise to
   * {x, y, z, radius, kind, id}. Defensive: every field access is guarded
   * because the underlying Colyseus schema can be undefined or empty.
   */
  _collectObstaclesFromRoomState(state) {
    const out = [];
    const harvest = (collection, kind) => {
      if (!collection) return;
      const iter = typeof collection.forEach === "function" ? collection : null;
      if (iter) {
        iter.forEach((entry, key) => out.push(this._normalizeObstacle(entry, kind, key)));
      } else if (typeof collection[Symbol.iterator] === "function") {
        for (const entry of collection) out.push(this._normalizeObstacle(entry, kind, null));
      }
    };
    harvest(state.dynamicColliders, "dynamicCollider");
    harvest(state.acidBarrels, "acidBarrel");
    harvest(state.breakables, "breakable");
    return out.filter(Boolean);
  }

  _normalizeObstacle(entry, kind, key) {
    if (!entry) return null;
    // Position can live as x/y/z fields or position[3] depending on the
    // Colyseus schema generation.
    const x = Number(entry.x ?? entry.position?.[0] ?? entry.sx);
    const y = Number(entry.y ?? entry.position?.[1] ?? entry.sy ?? 0);
    const z = Number(entry.z ?? entry.position?.[2] ?? entry.sz);
    if (!Number.isFinite(x) || !Number.isFinite(z)) return null;
    let radius = Number(entry.radius);
    if (!Number.isFinite(radius)) {
      const sx = Number(entry.scaleX ?? entry.scale?.[0]);
      const sz = Number(entry.scaleZ ?? entry.scale?.[2]);
      if (Number.isFinite(sx) && Number.isFinite(sz)) {
        radius = Math.max(Math.abs(sx), Math.abs(sz)) / 2;
      } else {
        radius = kind === "acidBarrel" ? 1.0 : 1.5; // crates ~3m wide, barrels ~2m
      }
    }
    return {
      x,
      y,
      z,
      radius: Math.max(0.3, radius),
      kind,
      id: entry.id || entry.dynamicColliderID || entry.acidBarrelID || entry.breakableID || key || null,
    };
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
    // Remove the destroyed obstacle from the bot's static map so the
    // raycast features stop reporting it as blocking. Only the primary
    // session owns the shared GameState.
    room.onMessage("room:breakable:destroy", isPrimary
      ? safe((data) => {
          const id = data?.breakableID || data?.id;
          if (id) bot.gameState.removeStaticObstacle(id);
        }, "breakable_destroy")
      : noop);
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

    // Track server-reported time-left so we don't quit a match the server is
    // still running. The chainers room (TimeLimited) typically runs for
    // ~10 minutes; the local matchTimeout is purely a safety net for the case
    // where the server stops sending room:time updates entirely.
    let lastTimeLeftMs = null;
    let lastTimeUpdateAt = null;
    const trackTime = (data) => {
      if (data && Number.isFinite(data.left)) {
        lastTimeLeftMs = Number(data.left);
        lastTimeUpdateAt = Date.now();
      }
    };
    for (const session of connectedSessions) {
      session.room?.onMessage("room:time", trackTime);
    }

    const matchEndPromise = new Promise((resolve) => {
      const primary = connectedSessions[0];
      const finish = (reason) => {
        if (matchResolved) return;
        matchResolved = true;
        if (reason) {
          this.runtimeState.recordEvent("info", "match end", {
            roomIndex: this.roomIndex,
            reason,
            lastTimeLeftMs,
          });
        }
        resolve();
      };

      if (primary?.room) {
        primary.room.onMessage("room:dispose", () => finish("dispose"));
        primary.room.onLeave(() => finish("leave"));
      }

      // Safety-net poll: only force-end if (a) the server has gone silent on
      // room:time for >60s AND we've waited at least matchTimeout, OR (b) the
      // server explicitly reported left<=0. Otherwise trust the server clock.
      const safetyMs = Math.max(this.config.bot.matchTimeout, 60000);
      const watchdog = setInterval(() => {
        if (matchResolved) {
          clearInterval(watchdog);
          return;
        }
        if (lastTimeLeftMs !== null && lastTimeLeftMs <= 0) {
          clearInterval(watchdog);
          finish("server-time-zero");
          return;
        }
        const silentMs = lastTimeUpdateAt ? Date.now() - lastTimeUpdateAt : safetyMs;
        if (silentMs >= safetyMs) {
          clearInterval(watchdog);
          finish("server-silent");
        }
      }, 5000);
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
      track: this.track,
      resolvedAlias: this.resolvedAlias,
      resolvedVersion: this.resolvedVersion,
    });
  }

  async _finalizeMatch(connectedSessions) {
    // First pass: collect summaries so we can compute final rank before
    // pushing the terminal experience step into each agent's brain.
    const sessionData = connectedSessions.map((session) => {
      const playerState = this._refreshPlayerData(session);
      const summary = {
        score: numberOrZero(playerState?.score),
        kills: numberOrZero(playerState?.kills),
        deaths: numberOrZero(playerState?.deaths),
        damageDealt: numberOrZero(session.bot.fitness.damageDealt),
        damageTaken: numberOrZero(session.bot.fitness.damageTaken),
        survivalTime: numberOrZero(session.bot.fitness.survivalTime),
        abilitiesUsed: numberOrZero(session.bot.fitness.abilitiesUsed),
      };
      return { session, summary };
    });

    // Rank by score (ties: stable order is fine — equal scores get adjacent ranks).
    const ranked = [...sessionData].sort((left, right) => right.summary.score - left.summary.score);
    const rankByAgentId = new Map();
    ranked.forEach((entry, index) => {
      rankByAgentId.set(entry.session.bot.agentId, index + 1);
    });
    const roomSize = sessionData.length;

    const agentResults = [];
    for (const { session, summary } of sessionData) {
      const rank = rankByAgentId.get(session.bot.agentId) || 0;

      if (session.bot.brain) {
        session.bot.brain.recordStep({
          currentScore: summary.score,
          kills: summary.kills,
          deaths: summary.deaths,
          damageDealt: 0,
          damageTaken: 0,
          survivalSeconds: 0,
          done: true,
          rank,
          roomSize,
        });
        await session.bot.brain.flush();
        await session.bot.brain.reportEpisode({ ...summary, rank, roomSize });
      }

      if (session.bot.strategicBrain && this.track === "stable" && this.mode === "training") {
        this.strategicBrains.set(session.bot.agentId, session.bot.strategicBrain);
        await session.bot.strategicBrain.analyzeMatch({ ...summary, rank, roomSize }).catch(() => {});
      }

      agentResults.push({ ...this._buildAgentResult(session, summary), rank, roomSize });
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
    await Promise.all(sessions.map(async (session) => {
      if (!session.queued || session.queueLeft) return;
      session.queueLeft = true;
      try {
        await this.matchmaker.leaveQueue(session.userID);
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
      track: this.track,
      score: summary.score,
      kills: summary.kills,
      deaths: summary.deaths,
      damageDealt: summary.damageDealt,
      damageTaken: summary.damageTaken,
      survivalTime: summary.survivalTime,
      abilitiesUsed: summary.abilitiesUsed,
      decisionsMade: runtime.decisionsMade || 0,
      policyLedDecisions: runtime.policyLedDecisions || 0,
      shotsFired: runtime.shotsFired || 0,
      shotRate: runtime.shotRate || 0,
      tacticalOverrides: runtime.tacticalOverrides || 0,
      tacticalOverrideRatio: runtime.tacticalOverrideRatio || 0,
      combatInactivityMs: runtime.combatInactivityMs || 0,
      inputsSent: runtime.inputsSent,
      stateUpdates: runtime.stateUpdates,
    };
  }

  _buildMatchSummary(selection, connectedSessions, agentResults) {
    const startedAt = this.roomState.lastMatchStartedAt || new Date().toISOString();
    const finishedAt = new Date().toISOString();
    const totalScore = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.score), 0);
    const totalDamageDealt = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.damageDealt), 0);
    const totalKills = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.kills), 0);
    const totalDeaths = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.deaths), 0);
    const totalInputsSent = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.inputsSent), 0);
    const totalStateUpdates = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.stateUpdates), 0);
    const totalDecisionsMade = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.decisionsMade), 0);
    const totalPolicyLedDecisions = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.policyLedDecisions), 0);
    const totalShotsFired = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.shotsFired), 0);
    const totalTacticalOverrides = agentResults.reduce((sum, agent) => sum + numberOrZero(agent.tacticalOverrides), 0);
    const avgTacticalOverrideRatio = agentResults.length
      ? agentResults.reduce((sum, agent) => sum + numberOrZero(agent.tacticalOverrideRatio), 0) / agentResults.length
      : 0;
    const avgCombatInactivityMs = agentResults.length
      ? agentResults.reduce((sum, agent) => sum + numberOrZero(agent.combatInactivityMs), 0) / agentResults.length
      : 0;
    const shotRate = totalDecisionsMade > 0 ? totalShotsFired / totalDecisionsMade : 0;
    const policyShare = totalDecisionsMade > 0 ? totalPolicyLedDecisions / totalDecisionsMade : 0;
    const damagePerShot = totalShotsFired > 0 ? totalDamageDealt / totalShotsFired : 0;
    const hasCombatSignal = totalScore > 0 || totalDamageDealt > 0 || totalKills > 0 || totalDeaths > 0;

    return {
      roomIndex: this.roomIndex,
      mode: this.mode,
      track: this.track,
      jobId: this.jobId,
      templateId: this.templateId,
      resolvedAlias: this.resolvedAlias,
      resolvedVersion: this.resolvedVersion,
      roomId: selection.roomId,
      publicAddress: cleanPublicAddress(selection.publicAddress),
      expectedAgents: this.roster.length,
      assignedAgents: selection.sessions.length,
      connectedAgents: connectedSessions.length,
      fillRatio: +(connectedSessions.length / Math.max(this.roster.length, 1)).toFixed(3),
      startedAt,
      finishedAt,
      durationMs: Math.max(0, new Date(finishedAt).getTime() - new Date(startedAt).getTime()),
      totalScore,
      totalDamageDealt,
      totalKills,
      totalDeaths,
      totalInputsSent,
      totalStateUpdates,
      totalDecisionsMade,
      totalPolicyLedDecisions,
      totalShotsFired,
      totalTacticalOverrides,
      shotRate: +shotRate.toFixed(4),
      policyShare: +policyShare.toFixed(4),
      damagePerShot: +damagePerShot.toFixed(4),
      avgTacticalOverrideRatio: +avgTacticalOverrideRatio.toFixed(4),
      avgCombatInactivityMs: +avgCombatInactivityMs.toFixed(1),
      hasCombatSignal,
      winner: agentResults[0] || null,
      agentResults,
    };
  }

  _refreshPlayerData(session) {
    const playerState = session.room?.state?.players?.get?.(session.userID);
    if (playerState) {
      session.bot.data = playerState;
    }
    return session.bot.data || playerState || null;
  }

  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

module.exports = { RoomCoordinator };
