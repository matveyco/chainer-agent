/**
 * Chainer Agent — Web Dashboard Server
 *
 * Express server that:
 * - Serves the dashboard UI (static files)
 * - Proxies API calls to the Python trainer service
 * - Provides system-level endpoints
 */

require("dotenv").config({ path: require("path").join(__dirname, "../.env") });

const express = require("express");
const path = require("path");
const fs = require("fs");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const TRAINER_URL = process.env.TRAINER_URL || "http://localhost:5555";
const PORT = parseInt(process.env.DASHBOARD_PORT || "3000");

// Proxy helper
async function trainerFetch(endpoint, options = {}) {
  const res = await fetch(`${TRAINER_URL}${endpoint}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  return res.json();
}

// ── API Routes ──

app.get("/api/stats", async (req, res) => {
  try {
    const data = await trainerFetch("/stats");
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: "Trainer unreachable", message: err.message });
  }
});

app.get("/api/health", async (req, res) => {
  try {
    const data = await trainerFetch("/health");
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: "Trainer unreachable" });
  }
});

app.get("/api/agent/:id/history", async (req, res) => {
  try {
    const data = await trainerFetch(`/agent/${req.params.id}/history`);
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.post("/api/select", async (req, res) => {
  try {
    const data = await trainerFetch("/select", {
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
    const data = await trainerFetch(`/agent/${req.params.id}/reset`, { method: "POST" });
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.post("/api/agent/:target/clone/:source", async (req, res) => {
  try {
    const data = await trainerFetch(`/agent/${req.params.target}/clone/${req.params.source}`, {
      method: "POST",
      body: JSON.stringify(req.body || {}),
    });
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

// Fallback to index.html for SPA
app.get("/{*path}", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[Dashboard] http://0.0.0.0:${PORT}`);
  console.log(`[Dashboard] Trainer: ${TRAINER_URL}`);
});
