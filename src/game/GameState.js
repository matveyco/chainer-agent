/**
 * Tracks all player positions, health, and game state.
 * Updated from room messages and state snapshots.
 */

const { SpatialGrid } = require("./SpatialGrid");
const { GameStateDeserializer } = require("./Deserializer");

class GameState {
  constructor() {
    this.spatialGrid = new SpatialGrid();
    this.deserializer = new GameStateDeserializer();
    this.playerHealth = new Map(); // userID -> health
    this.playerData = new Map(); // userID -> full player state from room.state
  }

  /**
   * Process binary room:state:update message
   */
  processStateUpdate(buffer) {
    const snapshot = this.deserializer.deserializeSnapshot(buffer);
    for (const state of snapshot.state.players) {
      this.spatialGrid.updateClient(state.id, {
        x: state.x,
        y: state.y,
        z: state.z,
      });
    }
    return snapshot;
  }

  /**
   * Handle room:player:joined
   */
  addPlayer(userID, position = { x: 0, y: 0, z: 0 }) {
    this.spatialGrid.addClient(userID, position);
    this.playerHealth.set(userID, 100);
  }

  /**
   * Handle room:player:left
   */
  removePlayer(userID) {
    this.spatialGrid.removeClient(userID);
    this.playerHealth.delete(userID);
    this.playerData.delete(userID);
  }

  /**
   * Update player data from room.state.players.get(userID)
   */
  setPlayerData(userID, data) {
    this.playerData.set(userID, data);
  }

  /**
   * Update health from room:player:hit
   */
  updateHealth(userID, newHealth) {
    this.playerHealth.set(userID, newHealth);
  }

  /**
   * Handle room:player:die
   */
  handleDeath(userID) {
    this.playerHealth.set(userID, 0);
  }

  /**
   * Handle room:player:respawn
   */
  handleRespawn(userID) {
    this.playerHealth.set(userID, 100);
  }

  /**
   * Get own position from spatial grid
   */
  getPosition(userID) {
    const client = this.spatialGrid.getClient(userID);
    if (!client) return null;
    return client.position;
  }

  /**
   * Get closest living enemy within range
   */
  getClosestEnemy(myId, range) {
    const myPos = this.getPosition(myId);
    if (!myPos) return null;

    const nearby = this.spatialGrid.findNear(myPos, range, myId);
    for (const candidate of nearby) {
      const health = this.playerHealth.get(candidate.id);
      if (health !== undefined && health > 0) {
        return candidate;
      }
    }
    return null;
  }

  /**
   * Get all nearby enemies within range
   */
  getNearbyEnemies(myId, range) {
    const myPos = this.getPosition(myId);
    if (!myPos) return [];

    return this.spatialGrid.findNear(myPos, range, myId).filter((c) => {
      const health = this.playerHealth.get(c.id);
      return health !== undefined && health > 0;
    });
  }

  /**
   * Get player health
   */
  getHealth(userID) {
    return this.playerHealth.get(userID) ?? 0;
  }

  clear() {
    this.spatialGrid.clear();
    this.playerHealth.clear();
    this.playerData.clear();
  }
}

module.exports = { GameState };
