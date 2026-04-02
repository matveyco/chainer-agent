#!/usr/bin/env node

require("dotenv").config({ quiet: true });

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");
const {
  applyCliPatches,
  applyEnvOverrides,
  loadConfig,
  parseCliArgs,
  validateEnvContract,
  validateRuntimeVersions,
} = require("../src/runtime/ConfigContract");
const { countRosterAgents, normalizeRosterDocument } = require("../src/runtime/Roster");

async function checkUrl(url) {
  const res = await fetch(url);
  return { ok: res.ok, status: res.status };
}

function inspectPythonEnvironment() {
  const probe = spawnSync("python3", ["-c", "import sys, json; print(json.dumps({'python': sys.version.split()[0]}))"], {
    encoding: "utf-8",
  });
  if (probe.status !== 0) {
    return { ok: false, error: probe.stderr.trim() || "python3 unavailable" };
  }

  try {
    const parsed = JSON.parse(probe.stdout.trim());
    return { ok: true, ...parsed };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

(async () => {
  const { flags, patches } = parseCliArgs(process.argv.slice(2));
  let config = loadConfig();
  config = applyEnvOverrides(config, process.env);
  config = applyCliPatches(config, patches);

  const envCheck = validateEnvContract(config, process.env);
  const versionCheck = validateRuntimeVersions(config);
  const errors = [...envCheck.errors, ...versionCheck.errors];
  const warnings = [...envCheck.warnings, ...versionCheck.warnings];

  const report = {
    ok: errors.length === 0,
    service: flags.service || "all",
    errors,
    warnings,
    config: {
      endpoint: config.server?.endpoint || null,
      trainerUrl: config.trainerUrl || null,
      supervisorPort: config.runtime?.port || null,
      rooms: config.rooms?.count || 0,
      agentsPerRoom: config.rooms?.agentsPerRoom || 0,
      rosterFile: envCheck.rosterPath,
    },
    roster: {
      ok: false,
      rooms: 0,
      agents: 0,
    },
    python: inspectPythonEnvironment(),
    live: null,
  };

  if (!report.python.ok) {
    report.errors.push(`Python environment check failed: ${report.python.error}`);
    report.ok = false;
  }

  try {
    const rosterDoc = JSON.parse(fs.readFileSync(path.resolve(envCheck.rosterPath), "utf-8"));
    const roster = normalizeRosterDocument(rosterDoc, config);
    report.roster = {
      ok: true,
      rooms: roster.length,
      agents: countRosterAgents(roster),
    };
  } catch (err) {
    report.errors.push(`Invalid roster: ${err.message}`);
    report.ok = false;
  }

  if (flags.live) {
    report.live = {};
    const checks = [
      ["trainerHealth", `${process.env.TRAINER_URL || "http://localhost:5555"}/healthz`],
      ["trainerReady", `${process.env.TRAINER_URL || "http://localhost:5555"}/readyz`],
      ["supervisorHealth", `${process.env.SUPERVISOR_URL || `http://localhost:${process.env.SUPERVISOR_PORT || 3101}`}/healthz`],
      ["supervisorReady", `${process.env.SUPERVISOR_URL || `http://localhost:${process.env.SUPERVISOR_PORT || 3101}`}/readyz`],
      ["dashboardHealth", `http://localhost:${process.env.DASHBOARD_PORT || 3000}/healthz`],
      ["arenaReachable", process.env.GAME_SERVER_URL],
    ];

    for (const [name, url] of checks) {
      if (!url) continue;
      try {
        report.live[name] = await checkUrl(url);
        if (!report.live[name].ok) {
          report.errors.push(`${name} returned HTTP ${report.live[name].status}`);
        }
      } catch (err) {
        report.live[name] = { ok: false, error: err.message };
        report.errors.push(`${name} failed: ${err.message}`);
      }
    }
    report.ok = report.errors.length === 0;
  }

  if (flags.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(`[doctor] service=${report.service} ok=${report.ok}`);
    for (const error of report.errors) console.log(`ERROR ${error}`);
    for (const warning of report.warnings) console.log(`WARN ${warning}`);
    console.log(`Roster: ${report.roster.agents} agents across ${report.roster.rooms} rooms`);
    if (report.live) {
      for (const [name, result] of Object.entries(report.live)) {
        console.log(`${name}: ${result.ok ? "ok" : "fail"} ${result.status || result.error || ""}`.trim());
      }
    }
  }

  process.exit(report.ok ? 0 : 1);
})().catch((err) => {
  console.error(`[doctor] fatal: ${err.message}`);
  process.exit(1);
});
