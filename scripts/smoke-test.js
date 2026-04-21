#!/usr/bin/env node

require("dotenv").config();

const { runBotServiceProbe } = require("../src/network/BotServiceProbe");

const TRAINER_URL = process.env.TRAINER_URL || "http://localhost:5555";
const SUPERVISOR_URL = process.env.SUPERVISOR_URL || `http://localhost:${process.env.SUPERVISOR_PORT || 3101}`;
const GAME_SERVER_URL = process.env.GAME_SERVER_URL || process.env.ENDPOINT;

async function checkJSON(url) {
  const res = await fetch(url);
  const text = await res.text();
  return {
    ok: res.ok,
    status: res.status,
    body: text.slice(0, 400),
  };
}

(async () => {
  const checks = [
    ["trainer health", `${TRAINER_URL}/healthz`],
    ["trainer ready", `${TRAINER_URL}/readyz`],
    ["supervisor health", `${SUPERVISOR_URL}/healthz`],
    ["supervisor ready", `${SUPERVISOR_URL}/readyz`],
  ];

  if (GAME_SERVER_URL) {
    checks.push(["game server", `${GAME_SERVER_URL}`]);
  }

  let failed = false;
  for (const [name, url] of checks) {
    try {
      const result = await checkJSON(url);
      console.log(`${name}: ${result.status} ${result.ok ? "ok" : "fail"}`);
      console.log(result.body);
      if (!result.ok) failed = true;
    } catch (err) {
      failed = true;
      console.log(`${name}: error ${err.message}`);
    }
    console.log("---");
  }

  if (GAME_SERVER_URL && process.env.OAUTH_API_KEY) {
    const protocol = await runBotServiceProbe({
      endpoint: GAME_SERVER_URL,
      authKey: process.env.OAUTH_API_KEY,
      roomName: process.env.ROOM_NAME || process.env.ROOM,
      mapName: process.env.MAP_NAME,
      weaponType: process.env.WEAPON_TYPE || "rocket",
      queueTimeoutMs: 8000,
      assignmentTimeoutMs: 12000,
      rttTimeoutMs: 4000,
      pollMs: 1000,
    });

    console.log(`protocol: ${protocol.ok ? "ok" : `fail (${protocol.failedStage || "unknown"})`}`);
    console.log(JSON.stringify(protocol, null, 2));
    console.log("---");
    if (!protocol.ok) failed = true;
  } else {
    failed = true;
    console.log("protocol: error GAME_SERVER_URL/ENDPOINT or OAUTH_API_KEY missing");
    console.log("---");
  }

  process.exit(failed ? 1 : 0);
})();
