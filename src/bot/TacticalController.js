const { getDefaultArchetype } = require("./archetypes");

const RAYCAST_COUNT = 8;
const RAYCAST_RANGE = 12; // metres — must match StateExtractor
// Below this normalised clearance we consider a direction blocked and try to
// route around. 0.35 ≈ 4.2 m of headroom — about one body length past a crate.
const CLEAR_THRESHOLD = 0.35;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalize2D(x, z) {
  const length = Math.hypot(x, z);
  if (length <= 1e-6) {
    return { x: 0, z: 0, length: 0 };
  }
  return { x: x / length, z: z / length, length };
}

class TacticalController {
  constructor(options = {}) {
    const seed = options.seed ?? 0;
    const archetype = getDefaultArchetype(options.archetypeId, seed);

    this.defaults = { ...archetype.defaults };
    this.seed = seed;
    this.safeSize = options.safeSize || 55;
    this.passiveWindowMs = options.passiveWindowMs || 7000;
    this.enemyRangeRatio = options.enemyRangeRatio || 1.15;
    this.closeRange = options.closeRange || 10;
    this.strafeSign = seed % 2 === 0 ? 1 : -1;
  }

  /**
   * Primary decision maker. Produces a complete action from scratch using the
   * agent's strategy, sensor context, and obstacle raycasts. The neural net's
   * output is no longer fed in — it stays advisory and is only used for
   * gradient updates downstream.
   *
   * @param {Object} ctx
   * @param {Object|null} ctx.enemy   { distance, dirX, dirZ }
   * @param {Object|null} ctx.crystal { dirX, dirZ, distance }
   * @param {Object} ctx.position     { x, z }
   * @param {Object} ctx.strategy     per-agent strategy params
   * @param {number} ctx.healthPercent 0..1
   * @param {number} ctx.weaponRange
   * @param {boolean} ctx.cooldownReady
   * @param {boolean} ctx.abilityReady
   * @param {number} ctx.distanceFromCenter
   * @param {number[]|undefined} ctx.obstacleRays  8 normalised raycast clearances
   *                                               in compass order E,NE,N,NW,W,SW,S,SE
   * @returns {{action: Object, overridden: boolean, reasons: string[]}}
   */
  decide(ctx = {}) {
    const reasons = [];
    const strategy = this._resolveStrategy(ctx.strategy);
    const enemy = ctx.enemy || null;
    const distance = enemy ? enemy.distance ?? Infinity : Infinity;
    const weaponRange = Math.max(6, ctx.weaponRange || 20);
    const engageRange = weaponRange * this.enemyRangeRatio;
    const healthPercent = ctx.healthPercent ?? 1;
    const lowHealth = healthPercent <= strategy.retreat_threshold;
    const distanceFromCenter = ctx.distanceFromCenter || 0;
    const outOfBounds = distanceFromCenter > this.safeSize * 0.92;

    let desiredX = 0;
    let desiredZ = 0;

    if (outOfBounds) {
      const back = normalize2D(-(ctx.position?.x || 0), -(ctx.position?.z || 0));
      desiredX = back.x;
      desiredZ = back.z;
      reasons.push("return_to_center");
    } else if (enemy && lowHealth) {
      const strafeX = -enemy.dirZ * this.strafeSign;
      const strafeZ = enemy.dirX * this.strafeSign;
      desiredX = -enemy.dirX * 0.85 + strafeX * 0.3;
      desiredZ = -enemy.dirZ * 0.85 + strafeZ * 0.3;
      reasons.push("retreat_enemy");
    } else if (enemy) {
      const strafeX = -enemy.dirZ * this.strafeSign;
      const strafeZ = enemy.dirX * this.strafeSign;
      if (distance > engageRange) {
        desiredX = enemy.dirX * 0.9 + strafeX * 0.2;
        desiredZ = enemy.dirZ * 0.9 + strafeZ * 0.2;
        reasons.push("close_gap");
      } else if (strategy.aggression >= 0.55) {
        desiredX = enemy.dirX * 0.6 + strafeX * 0.55;
        desiredZ = enemy.dirZ * 0.6 + strafeZ * 0.55;
        reasons.push("engage_enemy");
      } else {
        desiredX = strafeX * 0.95 + enemy.dirX * 0.15;
        desiredZ = strafeZ * 0.95 + enemy.dirZ * 0.15;
        reasons.push("strafe_enemy");
      }
    } else if (ctx.crystal && strategy.crystal_priority >= 0.25) {
      desiredX = ctx.crystal.dirX;
      desiredZ = ctx.crystal.dirZ;
      reasons.push("seek_crystal");
    } else {
      const cx = -(ctx.position?.x || 0);
      const cz = -(ctx.position?.z || 0);
      const roam = normalize2D(cx + 0.45 * this.strafeSign, cz - 0.45 * this.strafeSign);
      desiredX = roam.x;
      desiredZ = roam.z;
      reasons.push("roam_center");
    }

    const avoided = this._avoidObstacles(desiredX, desiredZ, ctx.obstacleRays);
    if (avoided.adjusted) reasons.push("avoid_obstacle");

    const move = normalize2D(avoided.x, avoided.z);

    const next = {
      moveX: move.x,
      moveZ: move.z,
      aimOffsetX: 0,
      aimOffsetZ: 0,
      shouldShoot: false,
      shouldUseAbility: false,
    };

    if (
      enemy &&
      ctx.cooldownReady &&
      distance <= engageRange * 1.05 &&
      !lowHealth &&
      (strategy.aggression >= 0.3 || distance <= this.closeRange)
    ) {
      next.shouldShoot = true;
      reasons.push("shoot_enemy");
    }

    if (
      ctx.abilityReady &&
      enemy &&
      distance <= Math.max(8, weaponRange * 0.85) &&
      (strategy.ability_usage >= 0.4 || lowHealth)
    ) {
      next.shouldUseAbility = true;
      reasons.push("use_ability");
    }

    return {
      action: next,
      overridden: true,
      reasons,
    };
  }

  /**
   * Backward-compatible wrapper. The NN action is ignored — `decide()` is now
   * authoritative — but the return shape and reason vocabulary are preserved
   * so existing callers and tests continue to work.
   */
  stabilize(_action, ctx = {}) {
    return this.decide(ctx);
  }

  /**
   * Steer around obstacles using the 8-direction raycast clearances.
   * Returns the (possibly rotated) move direction plus an `adjusted` flag.
   */
  _avoidObstacles(desiredX, desiredZ, rays) {
    if (!Array.isArray(rays) || rays.length !== RAYCAST_COUNT) {
      return { x: desiredX, z: desiredZ, adjusted: false };
    }
    const len = Math.hypot(desiredX, desiredZ);
    if (len < 1e-6) return { x: 0, z: 0, adjusted: false };

    const ndx = desiredX / len;
    const ndz = desiredZ / len;

    // Sample clearance in the desired direction by interpolating the two
    // flanking rays. Ray i sits at angle 2π·i/8.
    const angle = Math.atan2(ndz, ndx);
    const norm = ((angle / (Math.PI * 2)) + 1) % 1;
    const idxF = norm * RAYCAST_COUNT;
    const i0 = Math.floor(idxF) % RAYCAST_COUNT;
    const i1 = (i0 + 1) % RAYCAST_COUNT;
    const t = idxF - Math.floor(idxF);
    const desiredClearance = rays[i0] * (1 - t) + rays[i1] * t;

    if (desiredClearance >= CLEAR_THRESHOLD) {
      return { x: ndx, z: ndz, adjusted: false };
    }

    // Forward is blocked. Pick the ray that maximises (alignment + bias) ×
    // clearance², heavily penalising directions that are themselves blocked.
    let bestScore = -Infinity;
    let bestX = ndx;
    let bestZ = ndz;
    for (let i = 0; i < RAYCAST_COUNT; i += 1) {
      const a = (Math.PI * 2 * i) / RAYCAST_COUNT;
      const rx = Math.cos(a);
      const rz = Math.sin(a);
      const align = ndx * rx + ndz * rz; // [-1, 1]
      const clearance = rays[i];
      if (clearance < CLEAR_THRESHOLD * 0.5) continue;
      const score = (align + 1.2) * (clearance * clearance);
      if (score > bestScore) {
        bestScore = score;
        bestX = rx;
        bestZ = rz;
      }
    }
    return { x: bestX, z: bestZ, adjusted: true };
  }

  _resolveStrategy(strategy) {
    return {
      aggression: clamp(Number(strategy?.aggression ?? this.defaults.aggression) || 0, 0, 1),
      accuracy_focus: clamp(Number(strategy?.accuracy_focus ?? this.defaults.accuracy_focus) || 0, 0, 1),
      crystal_priority: clamp(Number(strategy?.crystal_priority ?? this.defaults.crystal_priority) || 0, 0, 1),
      ability_usage: clamp(Number(strategy?.ability_usage ?? this.defaults.ability_usage) || 0, 0, 1),
      retreat_threshold: clamp(Number(strategy?.retreat_threshold ?? this.defaults.retreat_threshold) || 0, 0, 1),
    };
  }
}

TacticalController.RAYCAST_COUNT = RAYCAST_COUNT;
TacticalController.RAYCAST_RANGE = RAYCAST_RANGE;
TacticalController.CLEAR_THRESHOLD = CLEAR_THRESHOLD;

module.exports = { TacticalController };
