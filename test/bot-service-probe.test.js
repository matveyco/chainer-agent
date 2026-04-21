const test = require("node:test");
const assert = require("node:assert/strict");

const { runBotServiceProbe } = require("../src/network/BotServiceProbe");

class FakeRoom {
  constructor(roomId) {
    this.roomId = roomId;
    this.handlers = new Map();
    this.sent = [];
    this.left = false;
  }

  onMessage(type, handler) {
    this.handlers.set(type, handler);
  }

  send(type, payload) {
    this.sent.push({ type, payload });
    if (type === "room:rtt") {
      const handler = this.handlers.get("room:rtt");
      setTimeout(() => handler?.(Date.now()), 5);
    }
  }

  async leave() {
    this.left = true;
  }

  removeAllListeners() {
    this.handlers.clear();
  }
}

class FakeClient {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.joinCalls = [];
    FakeClient.instances.push(this);
  }

  async joinById(roomId, options) {
    this.joinCalls.push({ roomId, options });
    this.room = new FakeRoom(roomId);
    return this.room;
  }
}

FakeClient.instances = [];

test("bot service probe exercises the documented authenticated lifecycle", async (t) => {
  const calls = [];
  const originalFetch = global.fetch;
  global.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (url.endsWith("/matchmaker/queue-status")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true, data: { queueLength: 1, minUsers: 2 } }),
      };
    }
    if (url.endsWith("/matchmaker/queue-to-join")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true, data: { roomName: "TimeLimited", mapName: "arena" } }),
      };
    }
    if (url.endsWith("/matchmaker/join-queue")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true, data: { position: 1 } }),
      };
    }
    if (url.includes("/matchmaker/user-queue-position/")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          success: true,
          data: {
            room: {
              roomId: "room-1",
              publicAddress: "https://arena-host.example",
            },
          },
        }),
      };
    }
    if (url.includes("/matchmaker/leave-queue/")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true, data: { message: "left" } }),
      };
    }
    throw new Error(`unexpected url ${url}`);
  };
  t.after(() => {
    global.fetch = originalFetch;
    FakeClient.instances.length = 0;
  });

  const probe = await runBotServiceProbe({
    endpoint: "https://arena.example",
    authKey: "secret-token",
    weaponType: "rocket",
    pollMs: 1,
    queueTimeoutMs: 10,
    assignmentTimeoutMs: 10,
    rttTimeoutMs: 100,
    ClientClass: FakeClient,
  });

  assert.equal(probe.ok, true);
  assert.equal(probe.failedStage, null);
  assert.equal(probe.stages.queueToJoin.ok, true);
  assert.equal(probe.stages.joinById.ok, true);
  assert.equal(probe.stages.rtt.ok, true);
  assert.equal(probe.stages.leaveQueue.ok, true);
  assert.equal(calls[0].options.headers.Authorization, "Bearer secret-token");
  assert.equal(FakeClient.instances[0].endpoint, "https://arena-host.example");
  assert.equal(FakeClient.instances[0].joinCalls[0].options.OAuthAPIKey, "secret-token");
});
