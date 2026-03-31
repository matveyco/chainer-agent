const fs = require("fs");
const path = require("path");
const express = require("express");
const { spawnSync } = require("child_process");
const { RoomCoordinator } = require("./RoomCoordinator");
const { RuntimeState } = require("./RuntimeState");
const { EvaluationManager } = require("./EvaluationManager");
const { SingleInstanceLock } = require("./SingleInstanceLock");
const { loadRoster } = require("./Roster");
const { StrategicBrain } = require("../bot/StrategicBrain");
const logger = require("../utils/logger");

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

class SwarmSupervisor {
  constructor(config, onUpdate = null) {
    this.config = config;
    this.onUpdate = onUpdate;
    this.runtimeState = new RuntimeState(config);
    this.lock = new SingleInstanceLock(config.runtime?.lockFile || "/tmp/chainer-agent.lock");
    this.running = false;
    this.telemetryServer = null;
    this.strategicBrains = new Map();
    this.roster = loadRoster(config);
    this.evaluationManager = new EvaluationManager({
      config,
      runtimeState: this.runtimeState,
      roster: this.roster,
      trainerUrl: config.trainerUrl,
    });
    this.totalMatches = 0;
    this.selectionInterval = config.training?.selectionInterval || 10;
    this.numCull = config.training?.numCull || 5;
  }

  async run() {
    this._assertColyseusCompatibility();
    this._acquireLock();
    this.runtimeState.start();
    this.runtimeState.setStatus("starting");
    this.runtimeState.recordEvent("info", "supervisor starting", {
      rosterRooms: this.roster.length,
      totalAgents: this.roster.flat().length,
    });
    this._registerProcessHooks();
    this._initStrategicBrains();
    this._persistProfiles();
    this._startTelemetryServer();

    this.running = true;
    this.runtimeState.setStatus("running");
    this._refreshEvaluationSnapshot();

    while (this.running) {
      this.runtimeState.setActiveRunnerCount(this._countActiveRunners());
      const trainerStatus = await this._probeTrainer();
      this.runtimeState.setTrainerStatus(trainerStatus);
      this.onUpdate?.(this.runtimeState.getSystemSnapshot());

      if (!trainerStatus.reachable) {
        this.runtimeState.recordEvent("warn", "trainer unreachable, waiting", {
          trainerUrl: this.config.trainerUrl,
        });
        await this._sleep(5000);
        continue;
      }

      await this._maybeQueueAutomaticEvaluation();

      let results = null;
      if (this.evaluationManager.currentJob || this.evaluationManager.queue.length > 0) {
        this.runtimeState.setMode("evaluation");
        results = await this.evaluationManager.runNext({
          runRoomBatch: (rosters, options) => this._runRoomBatch(rosters, options),
          fetchFamilyStatus: (familyId) => this._fetchFamilyStatus(familyId),
          submitReport: (report) => this._submitEvaluationReport(report),
        });
      } else {
        this.runtimeState.setMode("training");
        results = await this._runRoomBatch(this.roster, { mode: "training" });
      }
      this._refreshEvaluationSnapshot();

      if (Array.isArray(results)) {
        for (const result of results) {
          if (result.connectedCount > 0) {
            this.totalMatches += 1;
            this.runtimeState.markMatchComplete();
          }
        }
      }

      await this._sleep(this.config.rooms?.requeueBackoffMs || 3000);
    }
  }

  stop() {
    this.running = false;
    this.runtimeState.recordEvent("info", "supervisor stopping");
    this.runtimeState.stop();
    this.lock.release();
    if (this.telemetryServer) {
      this.telemetryServer.close();
      this.telemetryServer = null;
    }
  }

  saveState() {
    const file = path.resolve(
      this.config.persistence?.generationsDir || "data/generations",
      `runtime_${Date.now()}.json`
    );
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify({
      system: this.runtimeState.getSystemSnapshot(),
      rooms: this.runtimeState.getRoomsSnapshot(),
      matches: this.runtimeState.getMatchSummariesSnapshot(),
      evaluation: this.runtimeState.getEvaluationSnapshot(),
      profiles: this.getAllProfiles(),
    }, null, 2));
    return file;
  }

  resumeFrom() {
    logger.warn("Runtime resume snapshots are not supported in the PPO supervisor. Starting fresh.");
  }

  getAgentProfile(agentId) {
    const brain = this.strategicBrains.get(agentId);
    return brain ? brain.getProfile() : null;
  }

  getAllProfiles() {
    const profiles = {};
    for (const [agentId, brain] of this.strategicBrains.entries()) {
      profiles[agentId] = brain.getProfile();
    }
    return profiles;
  }

  _acquireLock() {
    const payload = this.lock.acquire({
      runId: this.runtimeState.state.runId,
      command: process.argv.join(" "),
    });
    this.runtimeState.recordEvent("info", "single-instance lock acquired", payload);
  }

  _assertColyseusCompatibility() {
    const version = require("colyseus.js/package.json").version;
    this.runtimeState.setColyseusVersion(version);
    const expected = this.config.server?.colyseusVersion;
    if (expected && version !== expected) {
      throw new Error(`Unsupported colyseus.js version ${version}; expected ${expected}`);
    }
  }

  _initStrategicBrains() {
    if (!this.config.ollamaApiKey) return;
    for (const agent of this.roster.flat()) {
      this.strategicBrains.set(
        agent.agentId,
        new StrategicBrain(agent.agentId, this.config.ollamaApiKey, this.config.ollamaModel, {
          archetypeId: agent.archetypeId,
          trainerUrl: this.config.trainerUrl,
          reporter: this.runtimeState,
        })
      );
    }
  }

  _persistProfiles() {
    const file = path.resolve(this.config.persistence?.profilesFile || "data/agent_profiles.json");
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(this.getAllProfiles(), null, 2));
  }

  _registerProcessHooks() {
    const shutdown = () => {
      try {
        this.stop();
      } finally {
        process.exit(0);
      }
    };

    process.once("SIGINT", shutdown);
    process.once("SIGTERM", shutdown);
  }

  _startTelemetryServer() {
    const app = express();
    app.use(express.json());

    app.get("/healthz", (req, res) => {
      res.json({
        ok: true,
        status: this.runtimeState.state.status,
        run_id: this.runtimeState.state.runId,
      });
    });

    app.get("/readyz", (req, res) => {
      const ready = this.runtimeState.state.trainer.ready;
      res.status(ready ? 200 : 503).json({
        ok: ready,
        trainer: this.runtimeState.state.trainer,
      });
    });

    app.get("/metrics", (req, res) => {
      res.type("text/plain").send(this.runtimeState.toMetrics());
    });

    app.get("/system", (req, res) => {
      res.json(this.runtimeState.getSystemSnapshot());
    });

    app.get("/rooms", (req, res) => {
      res.json(this.runtimeState.getRoomsSnapshot());
    });

    app.get("/events", (req, res) => {
      const limit = Math.max(1, Math.min(500, parseInt(req.query.limit || "100", 10)));
      res.json(this.runtimeState.getEventsSnapshot(limit));
    });

    app.get("/matches", (req, res) => {
      const limit = Math.max(1, Math.min(500, parseInt(req.query.limit || "50", 10)));
      res.json(this.runtimeState.getMatchSummariesSnapshot(limit));
    });

    app.get("/eval/status", (req, res) => {
      res.json(this.evaluationManager.getStatus());
    });

    app.get("/eval/history", (req, res) => {
      const limit = Math.max(1, Math.min(200, parseInt(req.query.limit || "25", 10)));
      res.json(this.evaluationManager.getHistory(limit));
    });

    app.post("/eval/run", async (req, res) => {
      try {
        const job = this.evaluationManager.queueRun(req.body || {});
        this._refreshEvaluationSnapshot();
        res.status(202).json(job);
      } catch (err) {
        res.status(500).json({ error: err.message });
      }
    });

    const port = this.config.runtime?.port || 3101;
    this.telemetryServer = app.listen(port, "0.0.0.0", () => {
      logger.info(`Supervisor telemetry listening on http://0.0.0.0:${port}`);
    });
  }

  async _probeTrainer() {
    const startedAt = Date.now();
    try {
      const [health, ready] = await Promise.all([
        fetchJSON(`${this.config.trainerUrl}/healthz`),
        fetchJSON(`${this.config.trainerUrl}/readyz`),
      ]);

      return {
        reachable: health.ok,
        ready: ready.ok,
        latencyMs: Date.now() - startedAt,
        lastOkAt: new Date().toISOString(),
        lastError: health.ok ? null : `health status ${health.status}`,
      };
    } catch (err) {
      return {
        reachable: false,
        ready: false,
        latencyMs: Date.now() - startedAt,
        lastOkAt: this.runtimeState.state.trainer.lastOkAt,
        lastError: err.message,
      };
    }
  }

  async _triggerSelection() {
    try {
      const response = await fetchJSON(`${this.config.trainerUrl}/select`, {
        method: "POST",
        body: JSON.stringify({ num_cull: this.numCull }),
      });
      if (response.ok) {
        this.runtimeState.markSelectionRun();
        this.runtimeState.recordEvent("info", "selection complete", {
          activeAgents: response.data?.agents?.length || 0,
        });
      } else {
        throw new Error(response.data?.error || `HTTP ${response.status}`);
      }
    } catch (err) {
      this.runtimeState.recordEvent("error", "selection failed", { error: err.message });
    }
  }

  _countActiveRunners() {
    try {
      const result = spawnSync("ps", ["-Ao", "pid=,comm=,args="], { encoding: "utf-8" });
      if (result.status === 0) {
        return result.stdout
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean)
          .filter((line) => /\bnode\b/.test(line) && line.includes("src/index.js"))
          .length || 1;
      }
    } catch {}
    return 1;
  }

  async _runRoomBatch(roomRosters, options = {}) {
    if ((options.mode || "training") === "evaluation") {
      const results = [];
      for (let roomIndex = 0; roomIndex < roomRosters.length; roomIndex++) {
        const coordinator = new RoomCoordinator({
          roomIndex,
          roster: roomRosters[roomIndex],
          config: this.config,
          runtimeState: this.runtimeState,
          strategicBrains: this.strategicBrains,
          mode: options.mode || "training",
          jobId: options.jobId || null,
          templateId: options.templateId || null,
        });

        results.push(
          await coordinator.runMatch().catch((err) => ({
            connectedCount: 0,
            error: err.message,
            summary: null,
          }))
        );

        if (roomIndex < roomRosters.length - 1) {
          await this._sleep(this.config.rooms?.staggerMs || 5000);
        }
      }

      return results;
    }

    const roomPromises = [];
    for (let roomIndex = 0; roomIndex < roomRosters.length; roomIndex++) {
      const coordinator = new RoomCoordinator({
        roomIndex,
        roster: roomRosters[roomIndex],
        config: this.config,
        runtimeState: this.runtimeState,
        strategicBrains: this.strategicBrains,
        mode: options.mode || "training",
        jobId: options.jobId || null,
        templateId: options.templateId || null,
      });

      roomPromises.push(
        coordinator.runMatch().catch((err) => ({
          connectedCount: 0,
          error: err.message,
          summary: null,
        }))
      );

      if (roomIndex < roomRosters.length - 1) {
        await this._sleep(this.config.rooms?.staggerMs || 5000);
      }
    }

    return Promise.all(roomPromises);
  }

  async _fetchFamilyStatus(familyId) {
    const response = await fetchJSON(`${this.config.trainerUrl}/family/${familyId}/status`);
    if (!response.ok) {
      throw new Error(response.data?.error || `family status HTTP ${response.status}`);
    }
    return response.data;
  }

  async _submitEvaluationReport(report) {
    if (!report) return null;
    const response = await fetchJSON(`${this.config.trainerUrl}/eval/report`, {
      method: "POST",
      body: JSON.stringify(report),
    });
    if (!response.ok) {
      throw new Error(response.data?.error || `eval report HTTP ${response.status}`);
    }
    return response.data;
  }

  async _maybeQueueAutomaticEvaluation() {
    if (this.totalMatches <= 0 || this.totalMatches % this.selectionInterval !== 0) {
      return null;
    }
    try {
      const job = await this.evaluationManager.maybeQueueAutomatic((familyId) =>
        this._fetchFamilyStatus(familyId)
      );
      if (job) {
        this.runtimeState.recordEvent("info", "automatic evaluation queued", {
          familyId: job.familyId,
          candidateVersion: job.candidateVersion,
        });
      }
      return job;
    } catch (err) {
      this.runtimeState.recordEvent("warn", "automatic evaluation queue failed", {
        error: err.message,
      });
      return null;
    }
  }

  _refreshEvaluationSnapshot() {
    this.runtimeState.setEvaluationSnapshot({
      ...this.evaluationManager.getStatus(),
      history: this.evaluationManager.getHistory(),
    });
  }

  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

module.exports = { SwarmSupervisor };
