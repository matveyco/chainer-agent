/**
 * Transforms raw game state into a normalized input vector for the neural network.
 *
 * 14 inputs:
 *  0: own health (0-1)
 *  1: own X position (clamped ±1, normalized by arena size)
 *  2: own Z position (clamped ±1)
 *  3: distance to nearest enemy (0-1)
 *  4: angle to nearest enemy (sin)
 *  5: angle to nearest enemy (cos)
 *  6: nearest enemy health (0-1)
 *  7: nearby enemy count (0-1, /10)
 *  8: distance from arena center (0-1)
 *  9: cooldown active (0 or 1)
 * 10: own velocity X (-1 to 1)
 * 11: own velocity Z (-1 to 1)
 * 12: has target in range (0 or 1)
 * 13: moving toward/away from enemy (dot product, -1 to 1)
 */

const { clamp, getArrayLength } = require("../utils/math");

const INPUT_COUNT = 14;
const ARENA_SIZE = 60;
const MAX_HEALTH = 100;
const MAX_SPEED = 7;

class StateExtractor {
  constructor() {
    this.inputVector = new Float64Array(INPUT_COUNT);
    this.lastMoveX = 0;
    this.lastMoveZ = 0;
  }

  static get INPUT_COUNT() {
    return INPUT_COUNT;
  }

  /**
   * Update last known velocity for input vector
   */
  setLastMove(x, z) {
    this.lastMoveX = x;
    this.lastMoveZ = z;
  }

  /**
   * Extract normalized input vector from game state.
   * @param {Object} gameState - GameState instance
   * @param {string} myId - This bot's userID
   * @param {number} weaponRange - Weapon target distance
   * @param {boolean} cooldownActive - Whether weapon is on cooldown
   * @returns {Float64Array} 14-element normalized input vector
   */
  extract(gameState, myId, weaponRange, cooldownActive) {
    const v = this.inputVector;
    v.fill(0);

    // Own health
    const health = gameState.getHealth(myId);
    v[0] = health / MAX_HEALTH;

    // Own position
    const myPos = gameState.getPosition(myId);
    if (!myPos) return v;

    v[1] = clamp(myPos.x / ARENA_SIZE, -1, 1);
    v[2] = clamp(myPos.z / ARENA_SIZE, -1, 1);

    // Distance from center
    const distFromCenter = getArrayLength([myPos.x, myPos.y || 0, myPos.z]);
    v[8] = clamp(distFromCenter / ARENA_SIZE, 0, 1);

    // Cooldown
    v[9] = cooldownActive ? 1 : 0;

    // Velocity
    v[10] = clamp(this.lastMoveX / MAX_SPEED, -1, 1);
    v[11] = clamp(this.lastMoveZ / MAX_SPEED, -1, 1);

    // Enemy info
    const nearbyEnemies = gameState.getNearbyEnemies(myId, weaponRange);
    const closestEnemy = nearbyEnemies.length > 0 ? nearbyEnemies[0] : null;

    v[7] = clamp(nearbyEnemies.length / 10, 0, 1);

    if (closestEnemy) {
      v[3] = clamp(closestEnemy.distance / ARENA_SIZE, 0, 1);

      // Angle to enemy
      const dx = closestEnemy.position.x - myPos.x;
      const dz = closestEnemy.position.z - myPos.z;
      const angle = Math.atan2(dz, dx);
      v[4] = Math.sin(angle);
      v[5] = Math.cos(angle);

      // Enemy health
      const enemyHealth = gameState.getHealth(closestEnemy.id);
      v[6] = enemyHealth / MAX_HEALTH;

      // Has target
      v[12] = 1;

      // Moving toward/away (dot product of velocity and direction to enemy)
      const dirLen = Math.sqrt(dx * dx + dz * dz);
      if (dirLen > 0) {
        const ndx = dx / dirLen;
        const ndz = dz / dirLen;
        const velLen = Math.sqrt(this.lastMoveX * this.lastMoveX + this.lastMoveZ * this.lastMoveZ);
        if (velLen > 0) {
          v[13] = clamp(
            (this.lastMoveX / velLen) * ndx + (this.lastMoveZ / velLen) * ndz,
            -1,
            1
          );
        }
      }
    } else {
      v[3] = 1.0; // No enemy = max distance
      v[12] = 0;
    }

    return v;
  }
}

module.exports = { StateExtractor };
