/**
 * HTTP matchmaking client.
 * Adapted from server's matchmaking.js — no config dependency, takes endpoint as param.
 * No auth headers for staging server.
 */

const logger = require("../utils/logger");

const jsonHeaders = { "Content-Type": "application/json" };

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: { ...jsonHeaders, ...options.headers },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 404) {
    throw new Error(`${options.method || "GET"} ${url} -> ${res.status} ${JSON.stringify(data)}`);
  }
  return data;
}

class Matchmaker {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.intervalId = null;
  }

  /**
   * Join matchmaking queue and wait for room assignment.
   * @param {string} userID
   * @param {string} roomName
   * @param {string} mapName
   * @param {boolean} forceCreateRoom
   * @returns {Promise<{roomId: string, publicAddress: string}>}
   */
  async join(userID, roomName, mapName, forceCreateRoom = false) {
    return new Promise(async (resolve, reject) => {
      try {
        const res = await fetchJSON(`${this.endpoint}/matchmaker/join-queue`, {
          method: "POST",
          body: JSON.stringify({
            userID,
            roomName,
            mapName,
            forceCreateRoom,
          }),
        });

        if (res.error) {
          return reject(new Error(res.error));
        }

        // Immediate room assignment
        if (res.data?.room) {
          return resolve(res.data.room);
        }

        // Poll for room assignment
        this.intervalId = setInterval(async () => {
          try {
            const pos = await fetchJSON(
              `${this.endpoint}/matchmaker/user-queue-position/${userID}`
            );

            if (pos.data?.room) {
              clearInterval(this.intervalId);
              this.intervalId = null;
              return resolve(pos.data.room);
            }

            logger.debug(`${userID} queue position: ${pos.data?.position ?? "?"}`);
          } catch (err) {
            clearInterval(this.intervalId);
            this.intervalId = null;
            return reject(err);
          }
        }, 2000);

        // Timeout after 120s
        setTimeout(() => {
          if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
            reject(new Error(`Matchmaking timeout for ${userID}`));
          }
        }, 120000);
      } catch (err) {
        reject(err);
      }
    });
  }

  /**
   * Check if there's an active queue/room to join
   */
  async checkQueueToJoin() {
    try {
      const res = await fetchJSON(`${this.endpoint}/matchmaker/queue-to-join`);
      if (res.success && res.data?.roomName) {
        return res.data;
      }
      return null;
    } catch {
      return null;
    }
  }

  dispose() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
}

module.exports = { Matchmaker };
