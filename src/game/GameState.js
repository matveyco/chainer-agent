/**
 * Tracks all player positions, health, and game state.
 * Updated from room messages and state snapshots.
 */

const { SpatialGrid } = require("./SpatialGrid");
const { GameStateDeserializer } = require("./Deserializer");

/**
 * Snapshot of static obstacle geometry the bot uses for collision-aware
 * navigation. Each entry is approximated as a sphere on the XZ plane:
 *   { x, y, z, radius }
 * Sourced from room.state.dynamicColliders + acidBarrels + breakables and
 * trimmed when room:breakable:destroy fires.
 */
class GameState {
  constructor() {
    this.spatialGrid = new SpatialGrid();
    this.deserializer = new GameStateDeserializer();
    this.playerHealth = new Map(); // userID -> health
    this.playerData = new Map(); // userID -> full player state from room.state
    this.battleCrystals = [];
    this.staticObstacles = []; // [{x, y, z, radius, kind, id}]
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
    this.battleCrystals = snapshot.state.battleCrystals || [];
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

  getClosestCrystal(position) {
    if (!position || this.battleCrystals.length === 0) return null;
    let closest = null;
    let bestDistance = Infinity;

    for (const crystal of this.battleCrystals) {
      const dx = crystal.x - position.x;
      const dy = (crystal.y || 0) - (position.y || 0);
      const dz = crystal.z - position.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (distance < bestDistance) {
        bestDistance = distance;
        closest = { ...crystal, distance };
      }
    }

    return closest;
  }

  getNearbyCrystalCount(position, range) {
    if (!position || this.battleCrystals.length === 0) return 0;
    let count = 0;
    for (const crystal of this.battleCrystals) {
      const dx = crystal.x - position.x;
      const dy = (crystal.y || 0) - (position.y || 0);
      const dz = crystal.z - position.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (distance <= range) count += 1;
    }
    return count;
  }

  getLivePlayerCount() {
    let alive = 0;
    for (const health of this.playerHealth.values()) {
      if (health > 0) alive += 1;
    }
    return alive;
  }

  /**
   * Replace the static obstacle list with a fresh snapshot. Each entry
   * may carry { x, y, z, radius, kind, id }; we accept either `radius`
   * directly or {scale: [sx, sy, sz]} from the protobuf shape.
   */
  setStaticObstacles(list) {
    if (!Array.isArray(list)) {
      this.staticObstacles = [];
      return;
    }
    this.staticObstacles = list
      .map((raw) => {
        const x = Number(raw?.x ?? raw?.position?.[0]);
        const y = Number(raw?.y ?? raw?.position?.[1]);
        const z = Number(raw?.z ?? raw?.position?.[2]);
        if (!Number.isFinite(x) || !Number.isFinite(z)) return null;
        let radius = Number(raw?.radius);
        if (!Number.isFinite(radius)) {
          const scale = raw?.scale;
          if (Array.isArray(scale) && scale.length >= 3) {
            // scale is full extents; radius ~= max(half-extent on XZ).
            radius = Math.max(Math.abs(Number(scale[0]) || 0), Math.abs(Number(scale[2]) || 0)) / 2;
          } else {
            radius = 1.0; // safe default for an unknown collider
          }
        }
        return {
          x,
          y: Number.isFinite(y) ? y : 0,
          z,
          radius: Math.max(0.2, radius),
          kind: raw?.kind || raw?.type || "obstacle",
          id: raw?.id || null,
        };
      })
      .filter(Boolean);
  }

  removeStaticObstacle(idOrPredicate) {
    if (!this.staticObstacles.length) return;
    if (typeof idOrPredicate === "function") {
      this.staticObstacles = this.staticObstacles.filter((o) => !idOrPredicate(o));
    } else {
      this.staticObstacles = this.staticObstacles.filter((o) => o.id !== idOrPredicate);
    }
  }

  getStaticObstacleCount() {
    return this.staticObstacles.length;
  }

  /**
   * Cast a ray on the XZ plane from `origin` in `direction` (auto-normalised)
   * and return distance to the nearest static obstacle within `maxDistance`,
   * or `maxDistance` if nothing is hit.
   *
   * Treats each obstacle as a vertical cylinder of radius `obstacle.radius`,
   * which is a reasonable approximation for crates / barrels / fences in a
   * top-down arena.
   */
  rayDistanceToObstacle(origin, direction, maxDistance) {
    if (!origin || !direction || !this.staticObstacles.length) return maxDistance;
    const dx = Number(direction.x ?? direction[0]) || 0;
    const dz = Number(direction.z ?? direction[2]) || 0;
    const length = Math.sqrt(dx * dx + dz * dz);
    if (length === 0) return maxDistance;
    const ux = dx / length;
    const uz = dz / length;

    let best = maxDistance;
    for (const obs of this.staticObstacles) {
      // Vector from origin to obstacle center on XZ plane.
      const ox = obs.x - origin.x;
      const oz = obs.z - origin.z;
      // Project onto ray direction; if behind origin, skip.
      const t = ox * ux + oz * uz;
      if (t < 0 || t > best) continue;
      // Perpendicular distance from obstacle center to the ray.
      const perpX = ox - t * ux;
      const perpZ = oz - t * uz;
      const perpDist = Math.sqrt(perpX * perpX + perpZ * perpZ);
      if (perpDist <= obs.radius) {
        // Distance from origin to where ray enters the cylinder.
        const inside = Math.sqrt(Math.max(0, obs.radius * obs.radius - perpDist * perpDist));
        const hit = Math.max(0, t - inside);
        if (hit < best) best = hit;
      }
    }
    return best;
  }

  clear() {
    this.spatialGrid.clear();
    this.playerHealth.clear();
    this.playerData.clear();
    this.battleCrystals = [];
    this.staticObstacles = [];
  }
}

module.exports = { GameState };
