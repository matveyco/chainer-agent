const fs = require("fs");
const path = require("path");

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

class EvaluationManager {
  constructor({ config, runtimeState, roster, trainerUrl }) {
    this.config = config;
    this.runtimeState = runtimeState;
    this.baseRoster = roster;
    this.trainerUrl = trainerUrl;
    this.historyLimit = config.evaluation?.historyLimit || 50;
    this.stateFile = path.resolve(config.persistence?.evaluationStateFile || "data/evaluation_state.json");
    this.historyFile = path.resolve(config.persistence?.evaluationHistoryFile || "data/evaluation_history.jsonl");
    this.queue = [];
    this.currentJob = null;
    this.history = [];
    this._load();
  }

  queueRun(options = {}) {
    const job = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      familyId: options.familyId || this.config.training?.defaultPolicyFamily || "arena-main",
      requestedAt: new Date().toISOString(),
      requestedBy: options.requestedBy || "api",
      matchesPerTemplate: options.matchesPerTemplate || this.config.evaluation?.sampleMatches || 3,
      status: "queued",
      reason: options.reason || "manual",
      results: [],
      candidateVersion: options.candidateVersion || null,
      championVersion: options.championVersion || null,
      templateIds: [],
      progress: { completed: 0, total: 0 },
      report: null,
    };
    this.queue.push(job);
    this.runtimeState.recordEvent("info", "evaluation queued", {
      jobId: job.id,
      familyId: job.familyId,
      requestedBy: job.requestedBy,
    });
    this._persist();
    return clone(job);
  }

  async maybeQueueAutomatic(fetchFamilyStatus) {
    const familyId = this.config.training?.defaultPolicyFamily || "arena-main";
    const familyStatus = await fetchFamilyStatus(familyId);
    if (!familyStatus?.aliases) return null;

    const candidateVersion = familyStatus.aliases.candidate || 0;
    if (candidateVersion <= 0) return null;

    const alreadyQueued =
      this.currentJob?.familyId === familyId && this.currentJob?.candidateVersion === candidateVersion
      || this.queue.some((job) => job.familyId === familyId && job.candidateVersion === candidateVersion)
      || this.history.some((job) => job.familyId === familyId && job.candidateVersion === candidateVersion);
    if (alreadyQueued) return null;

    const job = this.queueRun({
      familyId,
      requestedBy: "auto",
      reason: "interval",
      candidateVersion,
      championVersion: familyStatus.aliases.champion || 0,
    });
    return job;
  }

  async runNext({ runRoomBatch, fetchFamilyStatus, submitReport }) {
    if (this.currentJob || this.queue.length === 0) return null;

    const job = this.queue.shift();
    const familyStatus = await fetchFamilyStatus(job.familyId);
    const templates = this._buildTemplates(job, familyStatus);

    job.status = "running";
    job.startedAt = new Date().toISOString();
    job.candidateVersion = familyStatus?.aliases?.candidate || job.candidateVersion || 0;
    job.championVersion = familyStatus?.aliases?.champion || job.championVersion || 0;
    job.templateIds = templates.map((template) => template.id);
    job.progress.total = templates.length * job.matchesPerTemplate;
    this.currentJob = job;
    this._persist();
    this.runtimeState.recordEvent("info", "evaluation started", {
      jobId: job.id,
      familyId: job.familyId,
      candidateVersion: job.candidateVersion,
      championVersion: job.championVersion,
    });

    try {
      for (const template of templates) {
        for (let iteration = 0; iteration < job.matchesPerTemplate; iteration++) {
          const results = await runRoomBatch(template.rooms, {
            mode: "evaluation",
            jobId: job.id,
            templateId: template.id,
          });
          this._assertValidRunResult(job, template, iteration, results);
          job.results.push({
            templateId: template.id,
            iteration,
            rooms: results.map((result) => result.summary).filter(Boolean),
          });
          job.progress.completed += 1;
          this._persist();
        }
      }

      job.finishedAt = new Date().toISOString();
      job.status = "completed";
      job.report = this._aggregateJob(job);
      await submitReport(job.report);
      this.runtimeState.incrementCounter("evaluationRuns");
      this.runtimeState.recordEvent("info", "evaluation completed", {
        jobId: job.id,
        familyId: job.familyId,
        passed: job.report.passed,
      });
    } catch (err) {
      job.finishedAt = new Date().toISOString();
      job.status = "failed";
      job.error = err.message;
      this.runtimeState.incrementCounter("evaluationFailures");
      this.runtimeState.recordEvent("error", "evaluation failed", {
        jobId: job.id,
        familyId: job.familyId,
        error: err.message,
      });
    } finally {
      this.history.unshift(clone(job));
      if (this.history.length > this.historyLimit) this.history.length = this.historyLimit;
      this.currentJob = null;
      this._persist();
    }

    return clone(job);
  }

  getStatus() {
    return {
      current: this.currentJob ? clone(this.currentJob) : null,
      queue: this.queue.map(clone),
    };
  }

  getHistory(limit = this.historyLimit) {
    return this.history.slice(0, limit).map(clone);
  }

  _buildTemplates(job, familyStatus) {
    const historical = (familyStatus?.champion_history || [])
      .map((entry) => entry.candidate_version || entry.version || 0)
      .filter((value) => value && value !== job.candidateVersion && value !== job.championVersion)
      .slice(-1);

    const baseTemplates = [
      {
        id: "candidate-champion-split-a",
        resolver: (index) => (index % 2 === 0 ? { alias: "candidate", side: "candidate" } : { alias: "champion", side: "champion" }),
      },
      {
        id: "candidate-champion-split-b",
        resolver: (index) => (index % 2 === 1 ? { alias: "candidate", side: "candidate" } : { alias: "champion", side: "champion" }),
      },
    ];

    if (historical[0]) {
      baseTemplates.push({
        id: "candidate-history-split",
        resolver: (index) => (
          index % 2 === 0
            ? { alias: "candidate", side: "candidate" }
            : { alias: "champion", side: "historical", version: historical[0] }
        ),
      });
    }

    return baseTemplates.map((template) => ({
      id: template.id,
      rooms: this.baseRoster.map((room) =>
        room.map((agent, index) => {
          const side = template.resolver(index);
          return {
            ...agent,
            modelAlias: side.alias,
            modelVersion: side.version ?? null,
            evaluationSide: side.side,
          };
        })
      ),
    }));
  }

  _aggregateJob(job) {
    const perSide = {
      candidate: { score: 0, kills: 0, deaths: 0, damage: 0, survival: 0, count: 0 },
      champion: { score: 0, kills: 0, deaths: 0, damage: 0, survival: 0, count: 0 },
      historical: { score: 0, kills: 0, deaths: 0, damage: 0, survival: 0, count: 0 },
    };
    let roomComparisons = 0;
    let candidateRoomWins = 0;

    for (const run of job.results) {
      for (const room of run.rooms) {
        const roomBuckets = {
          candidate: { score: 0, count: 0 },
          champion: { score: 0, count: 0 },
          historical: { score: 0, count: 0 },
        };

        for (const agent of room.agentResults || []) {
          const side = agent.evaluationSide || "candidate";
          if (!perSide[side]) continue;
          perSide[side].score += agent.score || 0;
          perSide[side].kills += agent.kills || 0;
          perSide[side].deaths += agent.deaths || 0;
          perSide[side].damage += agent.damageDealt || 0;
          perSide[side].survival += agent.survivalTime || 0;
          perSide[side].count += 1;
          roomBuckets[side].score += agent.score || 0;
          roomBuckets[side].count += 1;
        }

        const opponent = roomBuckets.champion.count > 0 ? "champion" : roomBuckets.historical.count > 0 ? "historical" : null;
        if (roomBuckets.candidate.count > 0 && opponent) {
          roomComparisons += 1;
          const candidateAvg = roomBuckets.candidate.score / roomBuckets.candidate.count;
          const opponentAvg = roomBuckets[opponent].score / roomBuckets[opponent].count;
          if (candidateAvg > opponentAvg) candidateRoomWins += 1;
        }
      }
    }

    const candidateAvgScore = perSide.candidate.count ? perSide.candidate.score / perSide.candidate.count : 0;
    const opponentBucket = perSide.champion.count ? perSide.champion : perSide.historical;
    const opponentAvgScore = opponentBucket.count ? opponentBucket.score / opponentBucket.count : 0;
    const scoreMargin = opponentAvgScore > 0 ? (candidateAvgScore - opponentAvgScore) / opponentAvgScore : (candidateAvgScore > 0 ? 1 : 0);
    const winRate = roomComparisons > 0 ? candidateRoomWins / roomComparisons : 0;
    const survivalRatio =
      opponentBucket.count > 0
        ? (perSide.candidate.survival / Math.max(perSide.candidate.count, 1)) / (opponentBucket.survival / opponentBucket.count || 1)
        : 1;
    const passed =
      scoreMargin >= (this.config.evaluation?.promotionMargin || 0.05) &&
      winRate >= (this.config.evaluation?.minWinRate || 0.55) &&
      survivalRatio >= 0.95;

    return {
      family_id: job.familyId,
      candidate_version: job.candidateVersion,
      champion_version: job.championVersion,
      job_id: job.id,
      templates: job.templateIds,
      runs: job.results.length,
      room_comparisons: roomComparisons,
      candidate: {
        avg_score: +candidateAvgScore.toFixed(2),
        avg_kd: +(perSide.candidate.kills / Math.max(perSide.candidate.deaths, 1)).toFixed(2),
        avg_damage: +(perSide.candidate.damage / Math.max(perSide.candidate.count, 1)).toFixed(2),
      },
      champion: {
        avg_score: +opponentAvgScore.toFixed(2),
        avg_kd: +(opponentBucket.kills / Math.max(opponentBucket.deaths, 1)).toFixed(2),
        avg_damage: +(opponentBucket.damage / Math.max(opponentBucket.count, 1)).toFixed(2),
      },
      score_margin: +scoreMargin.toFixed(4),
      win_rate: +winRate.toFixed(4),
      survival_ratio: +survivalRatio.toFixed(4),
      passed,
      recorded_at: new Date().toISOString(),
    };
  }

  _assertValidRunResult(job, template, iteration, results) {
    if (!Array.isArray(results) || results.length !== template.rooms.length) {
      throw new Error(
        `evaluation ${job.id} ${template.id} iteration ${iteration}: expected ${template.rooms.length} room results, received ${Array.isArray(results) ? results.length : 0}`
      );
    }

    const roomIds = new Set();

    results.forEach((result, roomIndex) => {
      const summary = result?.summary;
      const expectedAgents = template.rooms[roomIndex]?.length || 0;
      if (!summary) {
        throw new Error(
          `evaluation ${job.id} ${template.id} iteration ${iteration}: room ${roomIndex} did not return a summary`
        );
      }

      if (summary.connectedAgents !== expectedAgents || summary.agentResults?.length !== expectedAgents) {
        throw new Error(
          `evaluation ${job.id} ${template.id} iteration ${iteration}: room ${roomIndex} incomplete (${summary.connectedAgents}/${expectedAgents})`
        );
      }

      if (!summary.roomId) {
        throw new Error(
          `evaluation ${job.id} ${template.id} iteration ${iteration}: room ${roomIndex} missing room id`
        );
      }

      if (roomIds.has(summary.roomId)) {
        throw new Error(
          `evaluation ${job.id} ${template.id} iteration ${iteration}: duplicate room id ${summary.roomId}`
        );
      }
      roomIds.add(summary.roomId);

      const hasOpponent = summary.agentResults.some((agent) => ["champion", "historical"].includes(agent.evaluationSide));
      const hasCandidate = summary.agentResults.some((agent) => agent.evaluationSide === "candidate");
      if (!hasCandidate || !hasOpponent) {
        throw new Error(
          `evaluation ${job.id} ${template.id} iteration ${iteration}: room ${roomIndex} missing candidate/opponent sides`
        );
      }

      if (!summary.hasCombatSignal) {
        throw new Error(
          `evaluation ${job.id} ${template.id} iteration ${iteration}: room ${roomIndex} produced no combat or score signal`
        );
      }
    });
  }

  _load() {
    try {
      if (fs.existsSync(this.stateFile)) {
        const state = JSON.parse(fs.readFileSync(this.stateFile, "utf-8"));
        this.queue = state.queue || [];
        this.currentJob = state.current || null;
      }
      if (fs.existsSync(this.historyFile)) {
        const lines = fs.readFileSync(this.historyFile, "utf-8").split("\n").filter(Boolean);
        this.history = lines.map((line) => JSON.parse(line)).slice(-this.historyLimit).reverse();
      }
    } catch {}
  }

  _persist() {
    fs.mkdirSync(path.dirname(this.stateFile), { recursive: true });
    fs.writeFileSync(
      this.stateFile,
      JSON.stringify({ current: this.currentJob, queue: this.queue }, null, 2)
    );
    if (this.history.length > 0) {
      fs.mkdirSync(path.dirname(this.historyFile), { recursive: true });
      fs.writeFileSync(
        this.historyFile,
        this.history
          .slice()
          .reverse()
          .map((entry) => JSON.stringify(entry))
          .join("\n") + "\n"
      );
    }
  }
}

module.exports = { EvaluationManager };
