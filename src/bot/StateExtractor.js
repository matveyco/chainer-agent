/**
 * Transforms raw game state into a normalized input vector for the neural network.
 *
 * 24 inputs (must match Python trainer STATE_DIM):
 *   0: own health (0-1)
 *   1: own X position (clamped ±1)
 *   2: own Z position (clamped ±1)
 *   3: distance to nearest enemy (0-1)
 *   4: angle to nearest enemy (sin)
 *   5: angle to nearest enemy (cos)
 *   6: nearest enemy health (0-1)
 *   7: nearby enemy count (0-1)
 *   8: distance from arena center (0-1)
 *   9: cooldown active (0 or 1)
 *  10: own velocity X (-1 to 1)
 *  11: own velocity Z (-1 to 1)
 *  12: has target in range (0 or 1)
 *  13: moving toward/away from enemy (dot product)
 *  14: own score (normalized, /1000)
 *  15: own kills (normalized, /20)
 *  16: own deaths (normalized, /20)
 *  17: match time remaining (0-1)
 *  18: distance to nearest crystal (0-1)
 *  19: angle to nearest crystal (sin)
 *  20: angle to nearest crystal (cos)
 *  21: ability readiness ratio (0-1)
 *  22: recent damage dealt (0-1)
 *  23: recent damage taken (0-1)
 */

const { clamp, getArrayLength } = require("../utils/math");

const INPUT_COUNT = 24;
const ARENA_SIZE = 60;
const MAX_HEALTH = 100;
const MAX_SPEED = 7;
const MAX_DAMAGE_WINDOW = 100;

class StateExtractor {
  constructor() {
    this.inputVector = new Float32Array(INPUT_COUNT);
    this.lastMoveX = 0;
    this.lastMoveZ = 0;
    this.matchTimeRemaining = 1.0;
    this.recentDamageDealt = 0;
    this.recentDamageTaken = 0;
  }

  static get INPUT_COUNT() {
    return INPUT_COUNT;
  }

  setLastMove(x, z) {
    this.lastMoveX = x;
    this.lastMoveZ = z;
  }

  setMatchTime(remaining, total) {
    this.matchTimeRemaining = total > 0 ? clamp(remaining / total, 0, 1) : 1;
  }

  setRecentCombat(damageDealt, damageTaken) {
    this.recentDamageDealt = damageDealt || 0;
    this.recentDamageTaken = damageTaken || 0;
  }

  /**
   * Extract normalized input vector from game state.
   * @param {Object} gameState - GameState instance
   * @param {string} myId - This bot's userID
   * @param {number} weaponRange - Weapon target distance
   * @param {boolean} cooldownActive - Whether weapon is on cooldown
   * @param {Object} playerData - room.state.players data (score, kills, deaths)
   * @returns {Float32Array} 24-element normalized input vector
   */
  extract(gameState, myId, weaponRange, cooldownActive, playerData) {
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

      const dx = closestEnemy.position.x - myPos.x;
      const dz = closestEnemy.position.z - myPos.z;
      const angle = Math.atan2(dz, dx);
      v[4] = Math.sin(angle);
      v[5] = Math.cos(angle);

      const enemyHealth = gameState.getHealth(closestEnemy.id);
      v[6] = enemyHealth / MAX_HEALTH;

      v[12] = 1;

      // Dot product: moving toward/away
      const dirLen = Math.sqrt(dx * dx + dz * dz);
      if (dirLen > 0) {
        const ndx = dx / dirLen;
        const ndz = dz / dirLen;
        const velLen = Math.sqrt(this.lastMoveX * this.lastMoveX + this.lastMoveZ * this.lastMoveZ);
        if (velLen > 0) {
          v[13] = clamp(
            (this.lastMoveX / velLen) * ndx + (this.lastMoveZ / velLen) * ndz,
            -1, 1
          );
        }
      }
    } else {
      v[3] = 1.0;
    }

    // Score, kills, deaths from player data
    if (playerData) {
      v[14] = clamp((playerData.score || 0) / 1000, 0, 1);
      v[15] = clamp((playerData.kills || 0) / 20, 0, 1);
      v[16] = clamp((playerData.deaths || 0) / 20, 0, 1);
    }

    // Match time remaining
    v[17] = this.matchTimeRemaining;

    // Crystal/objective awareness
    const closestCrystal = gameState.getClosestCrystal(myPos);
    if (closestCrystal) {
      v[18] = clamp(closestCrystal.distance / ARENA_SIZE, 0, 1);
      const cdx = closestCrystal.x - myPos.x;
      const cdz = closestCrystal.z - myPos.z;
      const cAngle = Math.atan2(cdz, cdx);
      v[19] = Math.sin(cAngle);
      v[20] = Math.cos(cAngle);
    } else {
      v[18] = 1.0;
    }

    // Ability readiness ratio
    const abilities = playerData?.abilities;
    if (abilities?.size) {
      let ready = 0;
      let total = 0;
      for (const ability of abilities.values()) {
        total += 1;
        if (ability?.ready) ready += 1;
      }
      v[21] = total > 0 ? clamp(ready / total, 0, 1) : 0;
    }

    // Recent combat context
    v[22] = clamp(this.recentDamageDealt / MAX_DAMAGE_WINDOW, 0, 1);
    v[23] = clamp(this.recentDamageTaken / MAX_DAMAGE_WINDOW, 0, 1);

    return v;
  }
}

module.exports = { StateExtractor };
