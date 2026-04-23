#!/usr/bin/env node
/**
 * Chainer Bots — Daily Report
 *
 * Aggregates the last N hours of match summaries, agent profiles, PBT events,
 * and trainer state into a human-readable Markdown report.
 *
 * Usage:
 *   node scripts/daily_report.js                          # print to stdout
 *   node scripts/daily_report.js --hours 24               # custom window
 *   node scripts/daily_report.js --write                  # also save to data/reports/YYYY-MM-DD.md
 *   node scripts/daily_report.js --json                   # emit JSON instead of Markdown
 *
 * Reads:
 *   - data/match_summaries.jsonl
 *   - data/agent_profiles.json   (LLM-mutated strategy vectors)
 *   - training_logs/pbt.jsonl    (genetic algorithm exploit/explore events)
 *   - http://localhost:5555/agent/<id>/reward-weights  (current alpha + reward genome)
 *   - http://localhost:5555/families                   (model versions, aliases)
 *   - http://localhost:3101/metrics                    (live counters)
 *
 * Read-only — never modifies state.
 */

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const MATCH_SUMMARIES = path.join(ROOT, "data/match_summaries.jsonl");
const AGENT_PROFILES = path.join(ROOT, "data/agent_profiles.json");
const PBT_LOG = path.join(ROOT, "training_logs/pbt.jsonl");
const REPORTS_DIR = path.join(ROOT, "data/reports");
const TRAINER_URL = process.env.TRAINER_URL || "http://localhost:5555";
const SUPERVISOR_URL = process.env.SUPERVISOR_URL || "http://localhost:3101";

const ARCHETYPE_DEFAULTS = {
  hunter: { aggression: 0.9, accuracy_focus: 0.4, crystal_priority: 0.1, ability_usage: 0.7, retreat_threshold: 0.1 },
  sniper: { aggression: 0.2, accuracy_focus: 0.95, crystal_priority: 0.2, ability_usage: 0.3, retreat_threshold: 0.4 },
  collector: { aggression: 0.1, accuracy_focus: 0.3, crystal_priority: 0.95, ability_usage: 0.2, retreat_threshold: 0.6 },
  survivor: { aggression: 0.3, accuracy_focus: 0.5, crystal_priority: 0.4, ability_usage: 0.5, retreat_threshold: 0.7 },
  berserker: { aggression: 1.0, accuracy_focus: 0.2, crystal_priority: 0.0, ability_usage: 1.0, retreat_threshold: 0.0 },
  tactician: { aggression: 0.5, accuracy_focus: 0.6, crystal_priority: 0.3, ability_usage: 0.5, retreat_threshold: 0.35 },
  flanker: { aggression: 0.6, accuracy_focus: 0.7, crystal_priority: 0.2, ability_usage: 0.6, retreat_threshold: 0.3 },
  guardian: { aggression: 0.7, accuracy_focus: 0.5, crystal_priority: 0.4, ability_usage: 0.8, retreat_threshold: 0.25 },
};

function parseArgs(argv) {
  const args = { hours: 24, write: false, json: false };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--hours" && argv[i + 1]) args.hours = Number(argv[++i]) || 24;
    else if (a === "--write") args.write = true;
    else if (a === "--json") args.json = true;
  }
  return args;
}

function readJsonlLast(filePath, hoursBack) {
  if (!fs.existsSync(filePath)) return [];
  const cutoff = Date.now() - hoursBack * 3600 * 1000;
  const lines = fs.readFileSync(filePath, "utf-8").split("\n").filter(Boolean);
  const recent = [];
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    try {
      const entry = JSON.parse(lines[i]);
      const ts = entry.finishedAt
        ? Date.parse(entry.finishedAt)
        : entry.timestamp
        ? entry.timestamp * 1000
        : null;
      if (ts === null) {
        recent.push(entry);
        continue;
      }
      if (ts < cutoff) break;
      recent.push(entry);
    } catch {
      // skip malformed lines
    }
  }
  return recent.reverse();
}

function readJson(filePath, fallback) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return fallback;
  }
}

async function fetchJsonSafe(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function fetchTextSafe(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return "";
    return await res.text();
  } catch {
    return "";
  }
}

function aggregatePerAgent(matches) {
  const byAgent = new Map();
  for (const match of matches) {
    const results = match.agentResults || [];
    for (const a of results) {
      const key = a.agentId || a.id;
      if (!byAgent.has(key)) {
        byAgent.set(key, {
          agentId: key,
          displayName: a.displayName,
          archetypeId: a.archetypeId,
          matches: 0,
          totalScore: 0,
          topScore: 0,
          totalKills: 0,
          totalDeaths: 0,
          totalShots: 0,
          totalCrystals: 0,
          wins: 0,
          top3: 0,
        });
      }
      const e = byAgent.get(key);
      e.matches += 1;
      e.totalScore += Number(a.score) || 0;
      e.topScore = Math.max(e.topScore, Number(a.score) || 0);
      e.totalKills += Number(a.kills) || 0;
      e.totalDeaths += Number(a.deaths) || 0;
      e.totalShots += Number(a.shotsFired) || 0;
      e.totalCrystals += Number(a.crystalsCollectedApprox) || 0;
      if (Number(a.rank) === 1) e.wins += 1;
      if (Number(a.rank) > 0 && Number(a.rank) <= 3) e.top3 += 1;
    }
  }
  return [...byAgent.values()]
    .map((e) => ({
      ...e,
      avgScore: e.matches ? Math.round(e.totalScore / e.matches) : 0,
      avgKD: e.totalDeaths > 0 ? +(e.totalKills / e.totalDeaths).toFixed(2) : e.totalKills,
      winRate: e.matches ? +(e.wins / e.matches).toFixed(2) : 0,
      crystalsPerMatch: e.matches ? +(e.totalCrystals / e.matches).toFixed(1) : 0,
    }))
    .sort((a, b) => b.avgScore - a.avgScore);
}

function aggregatePerArchetype(perAgent) {
  const byArch = new Map();
  for (const e of perAgent) {
    const key = e.archetypeId || "unknown";
    if (!byArch.has(key)) {
      byArch.set(key, { archetype: key, bots: 0, totalAvgScore: 0, totalWinRate: 0 });
    }
    const r = byArch.get(key);
    r.bots += 1;
    r.totalAvgScore += e.avgScore;
    r.totalWinRate += e.winRate;
  }
  return [...byArch.values()]
    .map((r) => ({
      archetype: r.archetype,
      bots: r.bots,
      avgScore: r.bots ? Math.round(r.totalAvgScore / r.bots) : 0,
      winRate: r.bots ? +(r.totalWinRate / r.bots).toFixed(2) : 0,
    }))
    .sort((a, b) => b.avgScore - a.avgScore);
}

async function collectAlphaDrift(perAgent) {
  const rows = [];
  for (const a of perAgent) {
    const data = await fetchJsonSafe(`${TRAINER_URL}/agent/${a.agentId}/reward-weights`);
    const alpha = data?.reward_weights?.policyBlendAlpha;
    const generation = data?.pbt_generation ?? 0;
    if (alpha === undefined) continue;
    rows.push({
      agentId: a.agentId,
      displayName: a.displayName,
      archetype: a.archetypeId,
      alpha: +alpha.toFixed(3),
      generation,
      delta: +(alpha - 0.1).toFixed(3),
    });
  }
  return rows.sort((a, b) => b.alpha - a.alpha);
}

function summarizeLLM(profiles) {
  const rows = [];
  for (const [agentId, profile] of Object.entries(profiles || {})) {
    const archetype = profile?.personality?.archetype || "unknown";
    const baseline = ARCHETYPE_DEFAULTS[archetype] || {};
    const strategy = profile?.strategy || {};
    let drift = 0;
    let driftCount = 0;
    for (const key of Object.keys(baseline)) {
      const before = Number(baseline[key]);
      const after = Number(strategy[key]);
      if (Number.isFinite(before) && Number.isFinite(after)) {
        drift += Math.abs(after - before);
        driftCount += 1;
      }
    }
    const avgDrift = driftCount ? +(drift / driftCount).toFixed(3) : 0;
    const log = Array.isArray(profile?.thought_log) ? profile.thought_log : [];
    const lastPlan = log.length ? log[log.length - 1] : null;
    rows.push({
      agentId,
      archetype,
      lifetimeMatches: profile?.lifetime?.matches || 0,
      avgStrategyDrift: avgDrift,
      thoughts: log.length,
      lastPlan: lastPlan?.plan?.slice(0, 100) || null,
      lastPlanAt: lastPlan?.timestamp || null,
    });
  }
  return rows.sort((a, b) => b.avgStrategyDrift - a.avgStrategyDrift);
}

function summarizePbt(events, hoursBack) {
  const cutoff = Date.now() / 1000 - hoursBack * 3600;
  const recent = events.filter((e) => (e.timestamp || 0) >= cutoff);
  return recent.map((cycle) => ({
    timestamp: new Date((cycle.timestamp || 0) * 1000).toISOString(),
    cohortSize: cycle.cohort_size || cycle.events?.length || 0,
    bestFitness: cycle.ranked?.[0]?.fitness ?? null,
    bestAgent: cycle.ranked?.[0]?.agent_id || null,
    pairs: (cycle.events || []).map((e) => `${e.child}<-${e.parent}(g${e.generation})`),
  }));
}

function parsePromMetrics(text) {
  const out = {};
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const m = line.match(/^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([\d.eE+\-]+)/);
    if (m) out[m[1]] = Number(m[2]);
  }
  return out;
}

function fmtTable(rows, columns) {
  if (!rows.length) return "_(none)_\n";
  const header = `| ${columns.map((c) => c.label).join(" | ")} |`;
  const sep = `| ${columns.map(() => "---").join(" | ")} |`;
  const body = rows
    .map((row) => `| ${columns.map((c) => formatCell(row[c.key], c)).join(" | ")} |`)
    .join("\n");
  return `${header}\n${sep}\n${body}\n`;
}

function formatCell(value, col) {
  if (value === null || value === undefined) return "—";
  if (col.format === "pct") return `${Math.round(value * 100)}%`;
  if (col.format === "num") return Number(value).toLocaleString();
  if (col.format === "fixed1") return Number(value).toFixed(1);
  if (col.format === "fixed3") return Number(value).toFixed(3);
  return String(value);
}

function renderMarkdown(report) {
  const lines = [];
  const fmtDate = (d) => new Date(d).toISOString().replace("T", " ").slice(0, 19) + "Z";
  lines.push(`# Chainer Bots — Daily Report`);
  lines.push(``);
  lines.push(`Generated: \`${fmtDate(Date.now())}\``);
  lines.push(`Window: last **${report.hours}h** (${report.matchCount} matches)`);
  lines.push(``);

  lines.push(`## Top Performers`);
  lines.push(
    fmtTable(report.perAgent.slice(0, 12), [
      { key: "displayName", label: "Bot" },
      { key: "archetypeId", label: "Archetype" },
      { key: "matches", label: "Matches" },
      { key: "avgScore", label: "Avg Score", format: "num" },
      { key: "topScore", label: "Top Score", format: "num" },
      { key: "winRate", label: "Win Rate", format: "pct" },
      { key: "avgKD", label: "K/D" },
      { key: "crystalsPerMatch", label: "Crystals/Match", format: "fixed1" },
    ])
  );

  lines.push(`## Per-Archetype Performance`);
  lines.push(
    fmtTable(report.perArchetype, [
      { key: "archetype", label: "Archetype" },
      { key: "bots", label: "Bots" },
      { key: "avgScore", label: "Avg Score", format: "num" },
      { key: "winRate", label: "Win Rate", format: "pct" },
    ])
  );

  lines.push(`## Hybrid Blend Evolution (policyBlendAlpha)`);
  lines.push(`Default \`α=0.10\` (10% NN, 90% tactical). PBT mutates ±0.02 per cycle for the worst 25%.`);
  lines.push(``);
  lines.push(
    fmtTable(report.alphaDrift.slice(0, 12), [
      { key: "displayName", label: "Bot" },
      { key: "archetype", label: "Archetype" },
      { key: "alpha", label: "α", format: "fixed3" },
      { key: "delta", label: "Δ vs default", format: "fixed3" },
      { key: "generation", label: "PBT Gen" },
    ])
  );

  lines.push(`## LLM Strategic Updates`);
  const llmRows = report.llmRows.filter((r) => r.thoughts > 0);
  lines.push(`${llmRows.length} of ${report.llmRows.length} agents have at least one LLM-coached strategy update.`);
  lines.push(``);
  lines.push(
    fmtTable(llmRows.slice(0, 8), [
      { key: "agentId", label: "Agent" },
      { key: "archetype", label: "Archetype" },
      { key: "lifetimeMatches", label: "Matches" },
      { key: "thoughts", label: "Plans" },
      { key: "avgStrategyDrift", label: "Avg Drift", format: "fixed3" },
      { key: "lastPlan", label: "Last Plan" },
    ])
  );

  lines.push(`## PBT Cycles (genetic algorithm)`);
  if (!report.pbtCycles.length) {
    lines.push(`_No PBT cycles in the last ${report.hours}h._`);
  } else {
    for (const c of report.pbtCycles) {
      lines.push(
        `- ${c.timestamp} — best ${c.bestAgent} fitness=${(c.bestFitness ?? 0).toFixed(2)}; replaced ${c.cohortSize}: ${c.pairs.join(", ")}`
      );
    }
  }
  lines.push(``);

  lines.push(`## Live Counters`);
  const interesting = [
    "chainer_supervisor_total_matches",
    "chainer_shotsFired",
    "chainer_crystalPickupsApprox",
    "chainer_losVetoes",
    "chainer_stuckEscapes",
    "chainer_blendedDecisions",
    "chainer_tacticalOverrides",
    "chainer_llmAnalyses",
    "chainer_llmFailures",
    "chainer_modelFetchFailures",
    "chainer_experienceFlushes",
  ];
  for (const key of interesting) {
    if (report.counters[key] !== undefined) {
      lines.push(`- \`${key.replace(/^chainer_/, "")}\`: **${Math.round(report.counters[key]).toLocaleString()}**`);
    }
  }
  lines.push(``);

  lines.push(`## Trainer State`);
  if (report.trainerState) {
    const s = report.trainerState;
    lines.push(`- Family: \`${s.family || "?"}\``);
    lines.push(`- Aliases: \`latest=${s.aliases?.latest}, candidate=${s.aliases?.candidate}, champion=${s.aliases?.champion}\``);
    lines.push(`- Train steps: **${s.trainSteps?.toLocaleString() || "?"}**`);
    lines.push(`- Bound agents: ${s.boundAgents || "?"}`);
  } else {
    lines.push(`_Trainer unreachable._`);
  }
  lines.push(``);

  return lines.join("\n");
}

async function buildReport({ hours }) {
  const matches = readJsonlLast(MATCH_SUMMARIES, hours);
  const profiles = readJson(AGENT_PROFILES, {});
  const pbtEvents = readJsonlLast(PBT_LOG, hours);
  const families = await fetchJsonSafe(`${TRAINER_URL}/families`);
  const familyStatus = await fetchJsonSafe(`${TRAINER_URL}/family/arena-main/status`);
  const metricsText = await fetchTextSafe(`${SUPERVISOR_URL}/metrics`);

  const perAgent = aggregatePerAgent(matches);
  const perArchetype = aggregatePerArchetype(perAgent);
  const alphaDrift = await collectAlphaDrift(perAgent);
  const llmRows = summarizeLLM(profiles);
  const pbtCycles = summarizePbt(pbtEvents, hours);
  const counters = parsePromMetrics(metricsText);

  let trainerState = null;
  if (familyStatus) {
    trainerState = {
      family: "arena-main",
      aliases: familyStatus.aliases,
      trainSteps: familyStatus.train_steps,
      boundAgents: Array.isArray(familyStatus.bound_agents) ? familyStatus.bound_agents.length : null,
    };
  } else if (Array.isArray(families) && families[0]) {
    trainerState = {
      family: families[0].family_id,
      aliases: families[0].aliases,
      trainSteps: families[0].train_steps,
      boundAgents: null,
    };
  }

  return {
    generatedAt: new Date().toISOString(),
    hours,
    matchCount: matches.length,
    perAgent,
    perArchetype,
    alphaDrift,
    llmRows,
    pbtCycles,
    counters,
    trainerState,
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const report = await buildReport({ hours: args.hours });

  if (args.json) {
    process.stdout.write(JSON.stringify(report, null, 2) + "\n");
    return;
  }

  const md = renderMarkdown(report);
  if (args.write) {
    fs.mkdirSync(REPORTS_DIR, { recursive: true });
    const day = new Date().toISOString().slice(0, 10);
    const out = path.join(REPORTS_DIR, `${day}.md`);
    fs.writeFileSync(out, md);
    fs.writeFileSync(path.join(REPORTS_DIR, "latest.md"), md);
    console.log(`wrote ${out}`);
  } else {
    process.stdout.write(md);
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error("daily_report failed:", err);
    process.exit(1);
  });
}

module.exports = { buildReport, renderMarkdown, aggregatePerAgent, aggregatePerArchetype, summarizeLLM };
