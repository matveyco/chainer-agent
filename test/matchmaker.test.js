const test = require("node:test");
const assert = require("node:assert/strict");

const { Matchmaker, buildMatchmakerHeaders } = require("../src/network/Matchmaker");

test("matchmaker adds bearer auth to HTTP requests", async (t) => {
  const calls = [];
  const originalFetch = global.fetch;
  global.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    return {
      ok: true,
      status: 200,
      json: async () => ({ success: true, data: { roomName: "TimeLimited", mapName: "arena" } }),
    };
  };
  t.after(() => {
    global.fetch = originalFetch;
  });

  const matchmaker = new Matchmaker("https://arena.example", { authKey: "secret-token" });
  const result = await matchmaker.getQueueToJoin();

  assert.equal(result.active, true);
  assert.equal(calls[0].url, "https://arena.example/matchmaker/queue-to-join");
  assert.equal(calls[0].options.headers.Authorization, "Bearer secret-token");
  assert.equal(buildMatchmakerHeaders("secret-token").Authorization, "Bearer secret-token");
});

test("matchmaker waits for queue activation and then returns the active room", async (t) => {
  const responses = [
    { ok: true, status: 200, body: { success: false, data: null } },
    { ok: true, status: 200, body: { success: true, data: { roomName: "TimeLimited", mapName: "arena" } } },
  ];
  const originalFetch = global.fetch;
  global.fetch = async () => {
    const next = responses.shift();
    return {
      ok: next.ok,
      status: next.status,
      json: async () => next.body,
    };
  };
  t.after(() => {
    global.fetch = originalFetch;
  });

  const matchmaker = new Matchmaker("https://arena.example", { authKey: "secret-token", pollMs: 1 });
  const result = await matchmaker.waitForActiveQueue({ timeoutMs: 20, pollMs: 1 });

  assert.equal(result.active, true);
  assert.equal(result.data.roomName, "TimeLimited");
});

test("matchmaker uses delete leave-queue and room join options include OAuthAPIKey", async (t) => {
  const calls = [];
  const originalFetch = global.fetch;
  global.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    return {
      ok: true,
      status: 200,
      json: async () => ({ success: true, data: { message: "ok" } }),
    };
  };
  t.after(() => {
    global.fetch = originalFetch;
  });

  const matchmaker = new Matchmaker("https://arena.example", { authKey: "secret-token" });
  await matchmaker.leaveQueue("bot_123");
  const joinOptions = matchmaker.buildRoomJoinOptions({
    userID: "bot_123",
    weaponType: "rocket",
  });

  assert.equal(calls[0].url, "https://arena.example/matchmaker/leave-queue/bot_123");
  assert.equal(calls[0].options.method, "DELETE");
  assert.equal(joinOptions.OAuthAPIKey, "secret-token");
  assert.equal(joinOptions.weaponType, "rocket");
});
