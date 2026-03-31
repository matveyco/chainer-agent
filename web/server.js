/**
 * Chainer Agent — Operator Dashboard Server
 */

require("dotenv").config({ path: require("path").join(__dirname, "../.env"), quiet: true });

const express = require("express");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const TRAINER_URL = process.env.TRAINER_URL || "http://localhost:5555";
const SUPERVISOR_URL = process.env.SUPERVISOR_URL || `http://localhost:${process.env.SUPERVISOR_PORT || 3101}`;
const PORT = parseInt(process.env.DASHBOARD_PORT || "3000", 10);
const RUNTIME_STATE_FILE = path.join(__dirname, "../data/runtime_state.json");

async function fetchJSON(baseUrl, endpoint, options = {}) {
  const res = await fetch(`${baseUrl}${endpoint}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `${baseUrl}${endpoint} -> ${res.status}`);
    err.status = res.status;
    err.payload = data;
    throw err;
  }
  return data;
}

async function fetchText(baseUrl, endpoint) {
  const res = await fetch(`${baseUrl}${endpoint}`);
  if (!res.ok) {
    throw new Error(`${baseUrl}${endpoint} -> ${res.status}`);
  }
  return res.text();
}

async function runDoctor({ live = false } = {}) {
  return new Promise((resolve, reject) => {
    const args = [path.join(__dirname, "../scripts/doctor.js"), "--service", "dashboard", "--json"];
    if (live) args.push("--live");

    const child = spawn(process.execPath, args, {
      cwd: path.join(__dirname, ".."),
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      try {
        const payload = stdout.trim() ? JSON.parse(stdout) : { ok: code === 0 };
        payload.exitCode = code;
        if (stderr.trim()) payload.stderr = stderr.trim();
        resolve(payload);
      } catch (err) {
        reject(new Error(stderr.trim() || err.message));
      }
    });
  });
}

function readRuntimeSnapshot() {
  try {
    if (fs.existsSync(RUNTIME_STATE_FILE)) {
      return JSON.parse(fs.readFileSync(RUNTIME_STATE_FILE, "utf-8"));
    }
  } catch {}
  return null;
}

async function supervisorOrSnapshot(endpoint, fallbackSelector = null) {
  try {
    return await fetchJSON(SUPERVISOR_URL, endpoint);
  } catch (err) {
    const snapshot = readRuntimeSnapshot();
    if (snapshot && fallbackSelector) {
      return fallbackSelector(snapshot);
    }
    throw err;
  }
}

app.get("/healthz", async (req, res) => {
  try {
    const [trainer, supervisor] = await Promise.all([
      fetchJSON(TRAINER_URL, "/healthz"),
      supervisorOrSnapshot("/healthz", (snapshot) => ({
        ok: snapshot.status !== "stopped",
        status: snapshot.status,
        run_id: snapshot.runId,
      })),
    ]);
    res.json({ ok: true, trainer, supervisor });
  } catch (err) {
    res.status(503).json({ ok: false, error: err.message });
  }
});

app.get("/readyz", async (req, res) => {
  const results = await Promise.allSettled([
    fetchJSON(TRAINER_URL, "/readyz"),
    supervisorOrSnapshot("/readyz", (snapshot) => ({
      ok: snapshot?.trainer?.ready ?? false,
      trainer: snapshot?.trainer || {},
    })),
  ]);

  const ready = results.every((result) => result.status === "fulfilled" && result.value.ok !== false);
  res.status(ready ? 200 : 503).json({
    ok: ready,
    trainer: results[0].status === "fulfilled" ? results[0].value : { error: results[0].reason.message },
    supervisor: results[1].status === "fulfilled" ? results[1].value : { error: results[1].reason.message },
  });
});

app.get("/metrics", async (req, res) => {
  const parts = [];
  try {
    parts.push(await fetchText(TRAINER_URL, "/metrics"));
  } catch {
    parts.push("# trainer metrics unavailable\n");
  }
  try {
    parts.push(await fetchText(SUPERVISOR_URL, "/metrics"));
  } catch {
    parts.push("# supervisor metrics unavailable\n");
  }
  res.type("text/plain").send(parts.join("\n"));
});

app.get("/api/stats", async (req, res) => {
  try {
    const data = await fetchJSON(TRAINER_URL, "/stats");
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: "Trainer unreachable", message: err.message });
  }
});

app.get("/api/health", async (req, res) => {
  try {
    const [trainer, supervisor] = await Promise.all([
      fetchJSON(TRAINER_URL, "/healthz"),
      supervisorOrSnapshot("/system", (snapshot) => snapshot),
    ]);
    res.json({ trainer, supervisor });
  } catch (err) {
    res.status(503).json({ error: "Services unreachable", message: err.message });
  }
});

app.get("/api/system", async (req, res) => {
  try {
    const [stats, runtime] = await Promise.all([
      fetchJSON(TRAINER_URL, "/stats"),
      supervisorOrSnapshot("/system", (snapshot) => ({
        runId: snapshot.runId,
        pid: snapshot.pid,
        startedAt: snapshot.startedAt,
        status: snapshot.status,
        config: snapshot.config,
        supervisor: snapshot.supervisor,
        trainer: snapshot.trainer,
        counters: snapshot.counters,
        observations: snapshot.observations,
      })),
    ]);
    res.json({ runtime, trainer: stats });
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/rooms", async (req, res) => {
  try {
    const rooms = await supervisorOrSnapshot("/rooms", (snapshot) => snapshot.rooms || []);
    res.json(rooms);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/events", async (req, res) => {
  try {
    const limit = Math.max(1, Math.min(500, parseInt(req.query.limit || "100", 10)));
    const events = await supervisorOrSnapshot(`/events?limit=${limit}`, (snapshot) => snapshot.events || []);
    res.json(events);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/matches", async (req, res) => {
  try {
    const limit = Math.max(1, Math.min(500, parseInt(req.query.limit || "50", 10)));
    const matches = await supervisorOrSnapshot(`/matches?limit=${limit}`, (snapshot) => snapshot.matches || []);
    res.json(matches);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/eval/status", async (req, res) => {
  try {
    const evaluation = await supervisorOrSnapshot("/eval/status", (snapshot) => snapshot.evaluation || { current: null, queue: [] });
    res.json(evaluation);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/eval/history", async (req, res) => {
  try {
    const limit = Math.max(1, Math.min(200, parseInt(req.query.limit || "25", 10)));
    const history = await supervisorOrSnapshot(`/eval/history?limit=${limit}`, (snapshot) => snapshot.evaluation?.history || []);
    res.json(history);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.post("/api/eval/run", async (req, res) => {
  try {
    const data = await fetchJSON(SUPERVISOR_URL, "/eval/run", {
      method: "POST",
      body: JSON.stringify(req.body || {}),
    });
    res.status(202).json(data);
  } catch (err) {
    res.status(err.status || 503).json({ error: err.message, payload: err.payload });
  }
});

app.get("/api/families", async (req, res) => {
  try {
    const data = await fetchJSON(TRAINER_URL, "/families");
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/eval/report-history", async (req, res) => {
  try {
    const query = new URLSearchParams(req.query).toString();
    const suffix = query ? `?${query}` : "";
    const data = await fetchJSON(TRAINER_URL, `/eval/history${suffix}`);
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.post("/api/promotion/candidate/:family", async (req, res) => {
  try {
    const data = await fetchJSON(TRAINER_URL, `/promotion/candidate/${req.params.family}`, {
      method: "POST",
      body: JSON.stringify(req.body || {}),
    });
    res.json(data);
  } catch (err) {
    res.status(err.status || 503).json({ error: err.message, payload: err.payload });
  }
});

app.post("/api/promotion/champion/:family/approve", async (req, res) => {
  try {
    const data = await fetchJSON(TRAINER_URL, `/promotion/champion/${req.params.family}/approve`, {
      method: "POST",
      body: JSON.stringify(req.body || {}),
    });
    res.json(data);
  } catch (err) {
    res.status(err.status || 503).json({ error: err.message, payload: err.payload });
  }
});

app.post("/api/promotion/champion/:family/reject", async (req, res) => {
  try {
    const data = await fetchJSON(TRAINER_URL, `/promotion/champion/${req.params.family}/reject`, {
      method: "POST",
      body: JSON.stringify(req.body || {}),
    });
    res.json(data);
  } catch (err) {
    res.status(err.status || 503).json({ error: err.message, payload: err.payload });
  }
});

app.get("/api/diagnostics", async (req, res) => {
  try {
    const report = await runDoctor({ live: req.query.live === "1" });
    res.status(report.ok ? 200 : 503).json(report);
  } catch (err) {
    res.status(503).json({ ok: false, error: err.message });
  }
});

app.get("/api/model/:id/aliases", async (req, res) => {
  try {
    const aliases = await fetchJSON(TRAINER_URL, `/model/${req.params.id}/aliases`);
    res.json(aliases);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/model/:id/metadata", async (req, res) => {
  try {
    const query = new URLSearchParams(req.query).toString();
    const suffix = query ? `?${query}` : "";
    const metadata = await fetchJSON(TRAINER_URL, `/model/${req.params.id}/metadata${suffix}`);
    res.json(metadata);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/agent/:id/history", async (req, res) => {
  try {
    const data = await fetchJSON(TRAINER_URL, `/agent/${req.params.id}/history`);
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.post("/api/select", async (req, res) => {
  try {
    const data = await fetchJSON(TRAINER_URL, "/select", {
      method: "POST",
      body: JSON.stringify(req.body || {}),
    });
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.post("/api/agent/:id/reset", async (req, res) => {
  try {
    const data = await fetchJSON(TRAINER_URL, `/agent/${req.params.id}/reset`, { method: "POST" });
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.post("/api/agent/:target/clone/:source", async (req, res) => {
  try {
    const data = await fetchJSON(TRAINER_URL, `/agent/${req.params.target}/clone/${req.params.source}`, {
      method: "POST",
      body: JSON.stringify(req.body || {}),
    });
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/profiles", (req, res) => {
  try {
    const profilePath = path.join(__dirname, "../data/agent_profiles.json");
    if (fs.existsSync(profilePath)) {
      return res.json(JSON.parse(fs.readFileSync(profilePath, "utf-8")));
    }
  } catch {}
  return res.json({});
});

app.get("/api/profile/:id", (req, res) => {
  try {
    const profilePath = path.join(__dirname, "../data/agent_profiles.json");
    if (fs.existsSync(profilePath)) {
      const profiles = JSON.parse(fs.readFileSync(profilePath, "utf-8"));
      return res.json(profiles[req.params.id] || { error: "Not found" });
    }
  } catch (err) {
    return res.json({ error: err.message });
  }
  return res.json({ error: "No profiles yet" });
});

app.get("/{*path}", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[Dashboard] http://0.0.0.0:${PORT}`);
  console.log(`[Dashboard] Trainer: ${TRAINER_URL}`);
  console.log(`[Dashboard] Supervisor: ${SUPERVISOR_URL}`);
});
