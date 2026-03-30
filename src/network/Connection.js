/**
 * Colyseus connection wrapper.
 * Handles room joining and message handler setup.
 */

const { Client } = require("colyseus.js");
const { Matchmaker } = require("./Matchmaker");
const logger = require("../utils/logger");

class Connection {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.matchmaker = new Matchmaker(endpoint);
    this.client = null;
    this.room = null;
  }

  /**
   * Connect a bot to the game server through matchmaking.
   * @param {string} userID
   * @param {string} roomName
   * @param {string} mapName
   * @param {string} weaponType
   * @param {boolean} forceCreateRoom
   * @param {Object} callbacks - Event callbacks
   * @returns {Object} room
   */
  async connect(userID, roomName, mapName, weaponType, forceCreateRoom = false, callbacks = {}) {
    // Join matchmaking queue
    const roomData = await this.matchmaker.join(userID, roomName, mapName, forceCreateRoom);

    if (!roomData?.roomId || !roomData?.publicAddress) {
      throw new Error(`Invalid room data for ${userID}: ${JSON.stringify(roomData)}`);
    }

    // Connect via Colyseus
    const host = roomData.publicAddress.replace(/^https?:\/\//, "");
    const clientEndpoint = `https://${host}`;

    this.client = new Client(clientEndpoint);
    this.room = await this.client.joinById(roomData.roomId, {
      userID,
      weaponType,
    });

    logger.debug(`${userID} joined room ${this.room.roomId}`);

    // Setup message handlers
    this._setupHandlers(callbacks);

    return this.room;
  }

  _setupHandlers(callbacks) {
    const noop = () => {};
    const room = this.room;

    // State updates
    room.onMessage("room:state:update", callbacks.onStateUpdate || noop);
    room.onMessage("room:player:joined", callbacks.onPlayerJoined || noop);
    room.onMessage("room:player:left", callbacks.onPlayerLeft || noop);

    // Combat
    room.onMessage("room:player:die", callbacks.onPlayerDie || noop);
    room.onMessage("room:player:hit", callbacks.onPlayerHit || noop);
    room.onMessage("room:player:respawn", callbacks.onPlayerRespawn || noop);

    // Room lifecycle
    room.onMessage("room:dispose", callbacks.onDispose || noop);
    room.onMessage("room:time", callbacks.onTime || noop);
    room.onMessage("room:rtt", callbacks.onRtt || noop);

    // Other events (no-ops to prevent warnings)
    room.onMessage("room:player:heal", noop);
    room.onMessage("room:player:shield", noop);
    room.onMessage("room:player:loaded", noop);
    room.onMessage("room:player:rejoined", noop);
    room.onMessage("room:leaderboard:update", noop);
    room.onMessage("room:breakable:destroy", noop);
    room.onMessage("room:invite:userID", noop);
    room.onMessage("player:consecutive-kills", noop);
    room.onMessage("player:first-kill", noop);
    room.onMessage("player:one-shot-kill", noop);
    room.onMessage("player:several-kills-at-once", noop);
    room.onMessage("player:shield-deflected", noop);
    room.onMessage("room:session:replaced", noop);
    room.onMessage("__playground_message_types", noop);

    // Room leave handler
    room.onLeave(callbacks.onLeave || noop);
  }

  /**
   * Send loaded profile to server
   */
  sendLoaded(userName) {
    if (!this.room) return;
    this.room.send("room:player:loaded", {
      profile: {
        userName,
        wallet: "0x0",
        models: [],
        textures: [],
      },
    });
  }

  dispose() {
    this.matchmaker.dispose();
    if (this.room) {
      try {
        this.room.leave();
        this.room.removeAllListeners();
      } catch {}
      this.room = null;
    }
    this.client = null;
  }
}

module.exports = { Connection };
