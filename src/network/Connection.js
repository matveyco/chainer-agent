/**
 * Colyseus connection wrapper.
 * Handles room joining and message handler setup.
 */

const { Client } = require("colyseus.js");
const { Matchmaker } = require("./Matchmaker");
const logger = require("../utils/logger");

class Connection {
  constructor(endpoint, options = {}) {
    this.endpoint = endpoint;
    this.matchmaker = new Matchmaker(endpoint, options);
    this.client = null;
    this.room = null;
    this.userID = null;
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
    this.userID = userID;
    const roomData = await this.matchmaker.reserveSeat(userID, roomName, mapName, {
      forceCreateRoom,
    });

    if (!roomData?.roomId || !roomData?.publicAddress) {
      throw new Error(`Invalid room data for ${userID}: ${JSON.stringify(roomData)}`);
    }

    // Connect via Colyseus
    const host = roomData.publicAddress.replace(/^https?:\/\//, "");
    const clientEndpoint = `https://${host}`;

    this.client = new Client(clientEndpoint);
    this.room = await this.client.joinById(
      roomData.roomId,
      this.matchmaker.buildRoomJoinOptions({ userID, weaponType })
    );

    logger.debug(`${userID} joined room ${this.room.roomId}`);

    // Setup message handlers
    this._setupHandlers(callbacks);

    return this.room;
  }

  async connectViaActiveQueue(userID, weaponType, options = {}) {
    const {
      queueTimeoutMs = 30000,
      pollMs = 1000,
      forceCreateRoom = false,
      callbacks,
      ...callbackOptions
    } = options;
    const queueInfo = await this.matchmaker.waitForActiveQueue({
      timeoutMs: queueTimeoutMs,
      pollMs,
    });

    if (!queueInfo?.active || !queueInfo?.data?.roomName) {
      throw new Error("No active queue or open room was exposed by queue-to-join");
    }

    return this.connect(
      userID,
      queueInfo.data.roomName,
      queueInfo.data.mapName,
      weaponType,
      Boolean(forceCreateRoom),
      callbacks || callbackOptions
    );
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
    if (this.userID) {
      this.matchmaker.leaveQueue(this.userID).catch(() => {});
    }
    this.client = null;
    this.userID = null;
  }
}

module.exports = { Connection };
