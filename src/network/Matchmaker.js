/**
 * Authenticated matchmaker client for the external bot-service contract.
 */

const logger = require("../utils/logger");

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeEndpoint(endpoint) {
  return String(endpoint || "").replace(/\/+$/, "");
}

function buildMatchmakerHeaders(authKey, extraHeaders = {}) {
  const headers = { "Content-Type": "application/json", ...extraHeaders };
  if (authKey) {
    headers.Authorization = `Bearer ${authKey}`;
  }
  return headers;
}

class Matchmaker {
  constructor(endpoint, options = {}) {
    this.endpoint = normalizeEndpoint(endpoint);
    this.authKey = options.authKey || null;
    this.defaultPollMs = Math.max(1, Number(options.pollMs || 2000));
  }

  buildRoomJoinOptions({ userID, weaponType, isSpectator = false } = {}) {
    const joinOptions = {
      userID,
      weaponType,
    };

    if (this.authKey) {
      joinOptions.OAuthAPIKey = this.authKey;
    }

    if (isSpectator) {
      joinOptions.isSpectator = true;
    }

    return joinOptions;
  }

  async request(path, options = {}) {
    const { allowStatuses = [], headers = {}, ...rest } = options;
    const url = `${this.endpoint}${path}`;
    const res = await fetch(url, {
      ...rest,
      headers: buildMatchmakerHeaders(this.authKey, headers),
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok && !allowStatuses.includes(res.status)) {
      throw new Error(`${rest.method || "GET"} ${url} -> ${res.status} ${JSON.stringify(data)}`);
    }

    return {
      ok: res.ok,
      status: res.status,
      data,
      url,
    };
  }

  async getQueueToJoin() {
    const response = await this.request("/matchmaker/queue-to-join", {
      method: "GET",
      allowStatuses: [401, 403, 404],
    });
    const payload = response.data?.data || null;

    return {
      ok: response.ok,
      status: response.status,
      authorized: ![401, 403].includes(response.status),
      active: Boolean(response.ok && response.data?.success && payload?.roomName),
      data: payload,
      raw: response.data,
    };
  }

  async waitForActiveQueue(options = {}) {
    const timeoutMs = Math.max(0, Number(options.timeoutMs || 0));
    const pollMs = Math.max(1, Number(options.pollMs || this.defaultPollMs));
    const deadline = timeoutMs > 0 ? Date.now() + timeoutMs : 0;
    let last = null;

    do {
      last = await this.getQueueToJoin();
      if (last.active) {
        return last;
      }
      if (!last.authorized) {
        throw new Error(`queue-to-join unauthorized (${last.status})`);
      }
      if (timeoutMs === 0 || Date.now() >= deadline) {
        break;
      }
      await sleep(pollMs);
    } while (Date.now() < deadline);

    return last;
  }

  async getQueueStatus() {
    const response = await this.request("/matchmaker/queue-status", {
      method: "GET",
      allowStatuses: [401, 403, 404],
    });

    return {
      ok: response.ok,
      status: response.status,
      authorized: ![401, 403].includes(response.status),
      data: response.data?.data || response.data || null,
      raw: response.data,
    };
  }

  async joinQueue(userID, roomName, mapName, forceCreateRoom = false) {
    const response = await this.request("/matchmaker/join-queue", {
      method: "POST",
      body: JSON.stringify({
        userID,
        roomName,
        mapName,
        forceCreateRoom,
      }),
    });

    if (response.data?.error) {
      throw new Error(response.data.error);
    }

    return response.data?.data || {};
  }

  async getQueuePosition(userID) {
    const response = await this.request(`/matchmaker/user-queue-position/${userID}`, {
      method: "GET",
      allowStatuses: [404],
    });
    return {
      ok: response.ok,
      status: response.status,
      data: response.data?.data || null,
      raw: response.data,
    };
  }

  async waitForAssignment(userID, options = {}) {
    const timeoutMs = Math.max(1000, Number(options.timeoutMs || 120000));
    const pollMs = Math.max(1, Number(options.pollMs || this.defaultPollMs));
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
      const position = await this.getQueuePosition(userID);
      options.onPoll?.(position);
      if (position.data?.room) {
        return position.data.room;
      }
      await sleep(pollMs);
    }

    throw new Error(`Matchmaking timeout for ${userID}`);
  }

  async reserveSeat(userID, roomName, mapName, options = {}) {
    const queueResult = await this.joinQueue(
      userID,
      roomName,
      mapName,
      Boolean(options.forceCreateRoom)
    );

    if (queueResult?.room) {
      return queueResult.room;
    }

    return this.waitForAssignment(userID, options);
  }

  async leaveQueue(userID) {
    try {
      return await this.request(`/matchmaker/leave-queue/${userID}`, {
        method: "DELETE",
        allowStatuses: [404],
      });
    } catch (err) {
      logger.debug(`leave-queue failed for ${userID}: ${err.message}`);
      throw err;
    }
  }

  async checkQueueToJoin() {
    const result = await this.getQueueToJoin();
    return result.active ? result.data : null;
  }

  dispose() {}
}

module.exports = {
  Matchmaker,
  buildMatchmakerHeaders,
};
