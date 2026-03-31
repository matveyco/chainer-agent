const fs = require("fs");
const path = require("path");

class RuntimeState {
  constructor(config) {
    this.config = config;
    this.eventsLimit = config.runtime?.eventsLimit || 200;
    this.stateFile = config.persistence?.runtimeStateFile || "data/runtime_state.json";
    this.eventsFile = config.persistence?.runtimeEventsFile || "data/runtime_events.jsonl";
    this.matchesFile = config.persistence?.matchSummariesFile || "data/match_summaries.jsonl";
    this.state = {
      runId: `${Date.now()}-${process.pid}`,
      pid: process.pid,
      startedAt: new Date().toISOString(),
      status: "starting",
      config: {
        endpoint: config.server?.endpoint || null,
        trainerUrl: config.trainerUrl || null,
        rooms: config.rooms?.count || 0,
        agentsPerRoom: config.rooms?.agentsPerRoom || 0,
      },
      supervisor: {
        activeRunnerCount: 1,
        lockFile: config.runtime?.lockFile || null,
        colyseusVersion: null,
        totalMatches: 0,
        selectionRuns: 0,
        lastSelectionAt: null,
        lastError: null,
        lastLoopAt: null,
        currentMode: "training",
      },
      trainer: {
        reachable: false,
        ready: false,
        latencyMs: null,
        lastOkAt: null,
        lastError: null,
      },
      counters: {
        queueJoinAttempts: 0,
        queueAssignments: 0,
        queueTimeouts: 0,
        queueLeaves: 0,
        joinAttempts: 0,
        joinSuccesses: 0,
        joinFailures: 0,
        seatExpired: 0,
        lockedRooms: 0,
        schemaCrashes: 0,
        roomAbandons: 0,
        modelFetchFailures: 0,
        modelFetches: 0,
        experienceFlushes: 0,
        llmAnalyses: 0,
        llmFailures: 0,
        inputsSent: 0,
        stateUpdates: 0,
        evaluationRuns: 0,
        evaluationFailures: 0,
      },
      observations: {
        queueWaitMsAvg: 0,
        roomFillRatioAvg: 0,
        modelFetchLatencyMsAvg: 0,
        experienceFlushSizeAvg: 0,
      },
      rooms: [],
      events: [],
      matches: [],
      evaluation: {
        current: null,
        queue: [],
        history: [],
      },
    };
    this.observationSamples = new Map();
    this.flushTimer = null;
  }

  start() {
    const flushMs = this.config.runtime?.telemetryFlushMs || 1000;
    this._flush();
    this.flushTimer = setInterval(() => this._flush(), flushMs);
    this.flushTimer.unref?.();
  }

  stop() {
    if (this.flushTimer) clearInterval(this.flushTimer);
    this.state.status = "stopped";
    this._flush();
  }

  setStatus(status) {
    this.state.status = status;
    this.state.supervisor.lastLoopAt = new Date().toISOString();
  }

  setColyseusVersion(version) {
    this.state.supervisor.colyseusVersion = version;
  }

  setActiveRunnerCount(count) {
    this.state.supervisor.activeRunnerCount = count;
  }

  setMode(mode) {
    this.state.supervisor.currentMode = mode || "training";
    this.state.supervisor.lastLoopAt = new Date().toISOString();
  }

  setTrainerStatus(patch) {
    Object.assign(this.state.trainer, patch);
  }

  ensureRoom(roomIndex) {
    if (!this.state.rooms[roomIndex]) {
      this.state.rooms[roomIndex] = {
        roomIndex,
        status: "idle",
        phase: "idle",
        targetAgents: 0,
        assignedAgents: 0,
        connectedAgents: 0,
        livePlayers: 0,
        stateUpdates: 0,
        inputsSent: 0,
        queueWaitMs: null,
        fillRatio: 0,
        roomId: null,
        publicAddress: null,
        lastMatchStartedAt: null,
        lastMatchEndedAt: null,
        lastError: null,
        mode: "training",
        jobId: null,
        templateId: null,
      };
    }
    return this.state.rooms[roomIndex];
  }

  updateRoom(roomIndex, patch) {
    const room = this.ensureRoom(roomIndex);
    Object.assign(room, patch);
    if (typeof room.targetAgents === "number" && room.targetAgents > 0) {
      room.fillRatio = +(room.connectedAgents / room.targetAgents).toFixed(3);
      this.observe("roomFillRatioAvg", room.fillRatio);
    }
  }

  incrementCounter(name, amount = 1) {
    if (!(name in this.state.counters)) this.state.counters[name] = 0;
    this.state.counters[name] += amount;
  }

  observe(name, value) {
    if (!Number.isFinite(value)) return;
    const sample = this.observationSamples.get(name) || { count: 0, total: 0 };
    sample.count += 1;
    sample.total += value;
    this.observationSamples.set(name, sample);
    this.state.observations[name] = +(sample.total / sample.count).toFixed(2);
  }

  recordEvent(level, message, context = {}) {
    const event = {
      ts: new Date().toISOString(),
      level,
      message,
      context,
    };
    this.state.events.unshift(event);
    if (this.state.events.length > this.eventsLimit) {
      this.state.events.length = this.eventsLimit;
    }
    this._appendEvent(event);
  }

  noteRoomError(roomIndex, err, context = {}) {
    const message = err?.message || String(err);
    this.updateRoom(roomIndex, { lastError: message, status: "error" });
    this.state.supervisor.lastError = message;
    this.recordEvent("error", `room_${roomIndex}: ${message}`, context);
  }

  markSelectionRun() {
    this.state.supervisor.selectionRuns += 1;
    this.state.supervisor.lastSelectionAt = new Date().toISOString();
  }

  markMatchComplete() {
    this.state.supervisor.totalMatches += 1;
  }

  recordMatchSummary(summary) {
    this.state.matches.unshift(summary);
    if (this.state.matches.length > this.eventsLimit) {
      this.state.matches.length = this.eventsLimit;
    }
    const resolved = path.resolve(this.matchesFile);
    fs.mkdirSync(path.dirname(resolved), { recursive: true });
    fs.appendFile(resolved, JSON.stringify(summary) + "\n", () => {});
  }

  setEvaluationSnapshot(snapshot) {
    this.state.evaluation = JSON.parse(JSON.stringify(snapshot || {
      current: null,
      queue: [],
      history: [],
    }));
  }

  getSystemSnapshot() {
    return JSON.parse(JSON.stringify({
      runId: this.state.runId,
      pid: this.state.pid,
      startedAt: this.state.startedAt,
      status: this.state.status,
      config: this.state.config,
      supervisor: this.state.supervisor,
      trainer: this.state.trainer,
      counters: this.state.counters,
      observations: this.state.observations,
    }));
  }

  getRoomsSnapshot() {
    return JSON.parse(JSON.stringify(this.state.rooms.filter(Boolean)));
  }

  getEventsSnapshot(limit = this.eventsLimit) {
    return JSON.parse(JSON.stringify(this.state.events.slice(0, limit)));
  }

  getMatchSummariesSnapshot(limit = this.eventsLimit) {
    return JSON.parse(JSON.stringify(this.state.matches.slice(0, limit)));
  }

  getEvaluationSnapshot() {
    return JSON.parse(JSON.stringify(this.state.evaluation));
  }

  toMetrics() {
    const lines = [
      "# TYPE chainer_supervisor_up gauge",
      `chainer_supervisor_up 1`,
      "# TYPE chainer_supervisor_total_matches counter",
      `chainer_supervisor_total_matches ${this.state.supervisor.totalMatches}`,
      "# TYPE chainer_trainer_reachable gauge",
      `chainer_trainer_reachable ${this.state.trainer.reachable ? 1 : 0}`,
      "# TYPE chainer_trainer_ready gauge",
      `chainer_trainer_ready ${this.state.trainer.ready ? 1 : 0}`,
    ];

    for (const [name, value] of Object.entries(this.state.counters)) {
      lines.push(`# TYPE chainer_${name} counter`);
      lines.push(`chainer_${name} ${value}`);
    }

    for (const [name, value] of Object.entries(this.state.observations)) {
      lines.push(`# TYPE chainer_${name} gauge`);
      lines.push(`chainer_${name} ${Number.isFinite(value) ? value : 0}`);
    }

    for (const room of this.state.rooms.filter(Boolean)) {
      lines.push(`chainer_room_fill_ratio{room="${room.roomIndex}"} ${room.fillRatio || 0}`);
      lines.push(`chainer_room_connected_agents{room="${room.roomIndex}"} ${room.connectedAgents || 0}`);
      lines.push(`chainer_room_state_updates{room="${room.roomIndex}"} ${room.stateUpdates || 0}`);
      lines.push(`chainer_room_inputs_sent{room="${room.roomIndex}"} ${room.inputsSent || 0}`);
    }

    return `${lines.join("\n")}\n`;
  }

  _appendEvent(event) {
    const resolved = path.resolve(this.eventsFile);
    fs.mkdirSync(path.dirname(resolved), { recursive: true });
    fs.appendFile(resolved, JSON.stringify(event) + "\n", () => {});
  }

  _flush() {
    const resolved = path.resolve(this.stateFile);
    fs.mkdirSync(path.dirname(resolved), { recursive: true });
    fs.writeFileSync(resolved, JSON.stringify(this.state, null, 2));
  }
}

module.exports = { RuntimeState };
