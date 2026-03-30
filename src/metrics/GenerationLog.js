/**
 * Aggregates fitness results for an entire generation.
 * Logs summary stats and appends to training log file.
 */

const fs = require("fs");
const path = require("path");

class GenerationLog {
  constructor() {
    this.results = [];
  }

  addResult(botId, fitnessTracker, fitness) {
    this.results.push({
      botId,
      fitness,
      ...fitnessTracker.toJSON(),
    });
  }

  getSummary(generation) {
    if (this.results.length === 0) {
      return { generation, botCount: 0 };
    }

    const fitnesses = this.results.map((r) => r.fitness);
    const kds = this.results.map((r) => r.kdRatio);
    const accuracies = this.results.map((r) => r.accuracy);
    const kills = this.results.map((r) => r.kills);

    return {
      generation,
      botCount: this.results.length,
      bestFitness: Math.round(Math.max(...fitnesses) * 10) / 10,
      avgFitness: Math.round((fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length) * 10) / 10,
      worstFitness: Math.round(Math.min(...fitnesses) * 10) / 10,
      bestKD: Math.max(...kds),
      avgKD: Math.round((kds.reduce((a, b) => a + b, 0) / kds.length) * 100) / 100,
      avgAccuracy: Math.round(accuracies.reduce((a, b) => a + b, 0) / accuracies.length),
      totalKills: kills.reduce((a, b) => a + b, 0),
      timestamp: new Date().toISOString(),
    };
  }

  appendToLog(logsDir, generation) {
    const summary = this.getSummary(generation);
    const logFile = path.join(logsDir, "training.jsonl");
    fs.mkdirSync(path.dirname(logFile), { recursive: true });
    fs.appendFileSync(logFile, JSON.stringify(summary) + "\n");
    return summary;
  }

  clear() {
    this.results = [];
  }
}

module.exports = { GenerationLog };
