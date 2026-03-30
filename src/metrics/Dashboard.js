/**
 * Terminal dashboard for monitoring training progress.
 * Uses blessed + blessed-contrib for real-time charts.
 * Falls back to console output if TTY is not available.
 */

class Dashboard {
  constructor() {
    this.history = [];
    this.startTime = Date.now();
    this.useTUI = false;
    this.screen = null;
    this.widgets = {};
  }

  init() {
    // Try to use blessed TUI
    if (process.stdout.isTTY && !process.env.NO_DASHBOARD) {
      try {
        this._initBlessed();
        this.useTUI = true;
        return;
      } catch {
        // Fall back to console
      }
    }

    // Console fallback
    this.useTUI = false;
  }

  _initBlessed() {
    const blessed = require("blessed");
    const contrib = require("blessed-contrib");

    this.screen = blessed.screen({
      smartCSR: true,
      title: "AI Bot Training Dashboard",
    });

    const grid = new contrib.grid({ rows: 12, cols: 12, screen: this.screen });

    // Fitness chart (top left)
    this.widgets.fitnessChart = grid.set(0, 0, 6, 8, contrib.line, {
      label: " Fitness Over Generations ",
      style: { line: "yellow", text: "green", baseline: "black" },
      showLegend: true,
      legend: { width: 20 },
    });

    // Stats box (top right)
    this.widgets.stats = grid.set(0, 8, 6, 4, blessed.box, {
      label: " Current Stats ",
      content: "Waiting for first generation...",
      padding: { left: 1, top: 1 },
      style: { fg: "white", border: { fg: "cyan" } },
      border: { type: "line" },
    });

    // Generation table (bottom left)
    this.widgets.table = grid.set(6, 0, 6, 8, contrib.table, {
      label: " Generation History ",
      columnSpacing: 2,
      columnWidth: [6, 10, 10, 8, 8, 8],
      keys: true,
      fg: "white",
      selectedFg: "white",
      selectedBg: "blue",
      interactive: false,
    });

    // Status box (bottom right)
    this.widgets.status = grid.set(6, 8, 6, 4, contrib.log, {
      label: " Status Log ",
      fg: "green",
      selectedFg: "green",
      padding: { left: 1 },
    });

    // Exit handler
    this.screen.key(["escape", "q", "C-c"], () => {
      process.emit("SIGINT");
    });

    this.screen.render();
  }

  /**
   * Update dashboard with new generation data.
   */
  update(summary) {
    this.history.push(summary);

    if (this.useTUI) {
      this._updateBlessed(summary);
    } else {
      this._updateConsole(summary);
    }
  }

  _updateBlessed(summary) {
    const elapsed = this._formatTime(Date.now() - this.startTime);

    // Update fitness chart
    const bestData = this.history.map((h) => h.bestFitness || 0);
    const avgData = this.history.map((h) => h.avgFitness || 0);
    const xLabels = this.history.map((_, i) => String(i));

    this.widgets.fitnessChart.setData([
      { title: "Best", x: xLabels, y: bestData, style: { line: "yellow" } },
      { title: "Avg", x: xLabels, y: avgData, style: { line: "cyan" } },
    ]);

    // Update stats
    const neurons = summary.neuronRange || { min: "?", max: "?" };
    const conns = summary.connectionRange || { min: "?", max: "?" };
    const improvement = this.history.length > 1
      ? ((summary.bestFitness - this.history[0].bestFitness) / Math.max(Math.abs(this.history[0].bestFitness), 1) * 100).toFixed(1)
      : "0.0";

    this.widgets.stats.setContent(
      `{bold}GENERATION:{/bold} ${summary.generation}\n` +
      `{bold}BOTS:{/bold} ${summary.botCount}  {bold}TIME:{/bold} ${elapsed}\n` +
      `\n` +
      `{yellow-fg}Best Fitness:{/yellow-fg}  ${summary.bestFitness}\n` +
      `{cyan-fg}Avg Fitness:{/cyan-fg}   ${summary.avgFitness}\n` +
      `Worst Fitness: ${summary.worstFitness}\n` +
      `\n` +
      `{green-fg}Best K/D:{/green-fg}  ${summary.bestKD}\n` +
      `Avg K/D:   ${summary.avgKD}\n` +
      `Accuracy:  ${summary.avgAccuracy}%\n` +
      `Kills:     ${summary.totalKills}\n` +
      `\n` +
      `Neurons:   ${neurons.min}-${neurons.max}\n` +
      `Conns:     ${conns.min}-${conns.max}\n` +
      `Improve:   ${improvement}%`
    );
    this.widgets.stats.options.tags = true;

    // Update table
    const rows = this.history.slice(-15).reverse().map((h) => [
      String(h.generation),
      String(h.bestFitness),
      String(h.avgFitness),
      String(h.bestKD),
      String(h.avgKD),
      `${h.avgAccuracy}%`,
    ]);

    this.widgets.table.setData({
      headers: ["Gen", "Best Fit", "Avg Fit", "Best K/D", "Avg K/D", "Acc"],
      data: rows,
    });

    // Status log
    this.widgets.status.log(
      `Gen ${summary.generation}: F=${summary.bestFitness} K/D=${summary.bestKD} K=${summary.totalKills}`
    );

    this.screen.render();
  }

  _updateConsole(summary) {
    const elapsed = this._formatTime(Date.now() - this.startTime);
    const neurons = summary.neuronRange || { min: "?", max: "?" };
    const bar = this._progressBar(summary.avgFitness, 1000, 30);

    console.log("");
    console.log("╔══════════════════════════════════════════════════════════╗");
    console.log(`║  GEN: ${String(summary.generation).padEnd(5)} BOTS: ${String(summary.botCount).padEnd(4)} TIME: ${elapsed.padEnd(10)} ║`);
    console.log("╠══════════════════════════════════════════════════════════╣");
    console.log(`║  Best Fitness:  ${String(summary.bestFitness).padEnd(10)} Avg: ${String(summary.avgFitness).padEnd(15)} ║`);
    console.log(`║  Best K/D:      ${String(summary.bestKD).padEnd(10)} Avg: ${String(summary.avgKD).padEnd(15)} ║`);
    console.log(`║  Accuracy:      ${String(summary.avgAccuracy + "%").padEnd(10)} Kills: ${String(summary.totalKills).padEnd(13)} ║`);
    console.log(`║  Neurons:       ${String(neurons.min + "-" + neurons.max).padEnd(38)} ║`);
    console.log(`║  ${bar} ║`);
    console.log("╚══════════════════════════════════════════════════════════╝");
  }

  _progressBar(value, max, width) {
    const ratio = Math.min(Math.max(value / max, 0), 1);
    const filled = Math.round(ratio * width);
    const empty = width - filled;
    return "█".repeat(filled) + "░".repeat(empty) + ` ${Math.round(ratio * 100)}%`;
  }

  _formatTime(ms) {
    const s = Math.floor(ms / 1000);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}h${m}m`;
    if (m > 0) return `${m}m${sec}s`;
    return `${sec}s`;
  }

  /**
   * Log a status message.
   */
  log(message) {
    if (this.useTUI && this.widgets.status) {
      this.widgets.status.log(message);
      this.screen.render();
    } else {
      console.log(`[Dashboard] ${message}`);
    }
  }

  destroy() {
    if (this.screen) {
      this.screen.destroy();
    }
  }
}

module.exports = { Dashboard };
