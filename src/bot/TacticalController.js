const { getDefaultArchetype } = require("./archetypes");

const RAYCAST_COUNT = 8;
const RAYCAST_RANGE = 12; // metres — must match StateExtractor
// Below this normalised clearance we consider a direction blocked and try to
// route around. 0.35 ≈ 4.2 m of headroom — about one body length past a crate.
const CLEAR_THRESHOLD = 0.35;

// Behaviour thresholds. Raised once before, then lowered after observing that
// half the archetypes (Sniper / Collector / Survivor / Tactician) were stuck
// in pure strafe mode and never shooting beyond ~10 m. The new floors let
// every archetype actually fight; Sniper still fights less because of its
// other strategy params (low aggression -> only 0.15+ which keeps it
// shooting from cover, low ability_usage -> still rare abilities).
const ENGAGE_AGGRESSION_FLOOR = 0.35;
const SHOOT_AGGRESSION_FLOOR = 0.15;
const CRYSTAL_PRIORITY_FLOOR = 0.05;
const ABILITY_USAGE_FLOOR = 0.2;

// Universal "I'm about to die" floor — even Berserker (retreat_threshold=0)
// pulls back below 15% HP. This is "survivor tactics for everyone" without
// erasing per-archetype identity (Survivor still retreats much earlier).
const PANIC_HEALTH_FLOOR = 0.15;
// Outnumbered if this many enemies are within close range simultaneously.
const OUTNUMBERED_THRESHOLD = 3;
// Probability per decision tick of picking a random ready ability instead
// of the contextual choice. Lets the population discover unusual tactical
// uses (e.g. mine in the middle of a fight, jump as a feint) that we
// wouldn't have hand-coded.
const ABILITY_EXPLORE_PROB = 0.08;

// Ability identifiers must match SmartBot.ABILITIES exactly.
const ABILITY_JUMP = "jump";
const ABILITY_MINE = "minePlanting";
const ABILITY_RAMPAGE = "rampage";
const ALL_ABILITIES = [ABILITY_JUMP, ABILITY_MINE, ABILITY_RAMPAGE];

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
   * @param {Object|null} ctx.enemy            { distance, dirX, dirZ }
   * @param {Object|null} ctx.crystal          { dirX, dirZ, distance }
   * @param {Object|null} ctx.enemyClusterDir  { dirX, dirZ } pointing toward
   *                                           the centroid of nearby enemies,
   *                                           used as the no-enemy-no-crystal
   *                                           fallback so bots seek the fight.
   * @param {Object} ctx.position              { x, z }
   * @param {Object} ctx.strategy              per-agent strategy params
   * @param {number} ctx.healthPercent         0..1
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
    // Survivor tactics for every archetype: respect the per-archetype retreat
    // threshold (Survivor at 0.7, Berserker at 0.0) but enforce a hard 15%
    // floor — a near-dead Berserker still pulls back briefly, otherwise it
    // dies on every encounter. Visible to the user as fewer "bot stands and
    // dies" scenes.
    const effectiveRetreat = Math.max(strategy.retreat_threshold, PANIC_HEALTH_FLOOR);
    const lowHealth = healthPercent <= effectiveRetreat;
    const nearbyEnemyCount = Number(ctx.nearbyEnemyCount) || (enemy ? 1 : 0);
    // Outnumbered = retreat-style behaviour even if HP is fine. Berserker
    // (aggression > 0.9) is exempt because that's literally its thing.
    const outnumbered = nearbyEnemyCount >= OUTNUMBERED_THRESHOLD && strategy.aggression < 0.9;
    const distanceFromCenter = ctx.distanceFromCenter || 0;
    const outOfBounds = distanceFromCenter > this.safeSize * 0.92;

    let desiredX = 0;
    let desiredZ = 0;

    if (outOfBounds) {
      const back = normalize2D(-(ctx.position?.x || 0), -(ctx.position?.z || 0));
      desiredX = back.x;
      desiredZ = back.z;
      reasons.push("return_to_center");
    } else if (enemy && (lowHealth || outnumbered)) {
      // Cover-aware retreat: if the obstacle rays show somewhere we can put a
      // wall between us and the enemy, head there instead of running in a
      // straight line away. Now also triggers when outnumbered (3+ enemies in
      // close range) so even healthy aggressive bots disengage when focused.
      const reasonTag = outnumbered && !lowHealth ? "retreat_outnumbered" : "retreat_enemy";
      const coverTag = outnumbered && !lowHealth ? "retreat_outnumbered" : "retreat_to_cover";
      const cover = this._findCover(enemy, ctx.obstacleRays);
      if (cover) {
        desiredX = cover.x;
        desiredZ = cover.z;
        reasons.push(coverTag);
      } else {
        const strafeX = -enemy.dirZ * this.strafeSign;
        const strafeZ = enemy.dirX * this.strafeSign;
        desiredX = -enemy.dirX * 0.85 + strafeX * 0.3;
        desiredZ = -enemy.dirZ * 0.85 + strafeZ * 0.3;
        reasons.push(reasonTag);
      }
    } else if (enemy) {
      const strafeX = -enemy.dirZ * this.strafeSign;
      const strafeZ = enemy.dirX * this.strafeSign;
      if (distance > engageRange) {
        desiredX = enemy.dirX * 0.9 + strafeX * 0.2;
        desiredZ = enemy.dirZ * 0.9 + strafeZ * 0.2;
        reasons.push("close_gap");
      } else if (strategy.aggression >= ENGAGE_AGGRESSION_FLOOR) {
        desiredX = enemy.dirX * 0.6 + strafeX * 0.55;
        desiredZ = enemy.dirZ * 0.6 + strafeZ * 0.55;
        reasons.push("engage_enemy");
      } else {
        desiredX = strafeX * 0.95 + enemy.dirX * 0.15;
        desiredZ = strafeZ * 0.95 + enemy.dirZ * 0.15;
        reasons.push("strafe_enemy");
      }
    } else if (ctx.crystal && strategy.crystal_priority >= CRYSTAL_PRIORITY_FLOOR) {
      desiredX = ctx.crystal.dirX;
      desiredZ = ctx.crystal.dirZ;
      reasons.push("seek_crystal");
    } else if (ctx.enemyClusterDir) {
      // No nearby enemy and no crystal — head to where the action is rather
      // than wander to (0, 0). This is the single biggest fix for the
      // "bots stand around between fights" complaint.
      desiredX = ctx.enemyClusterDir.dirX;
      desiredZ = ctx.enemyClusterDir.dirZ;
      reasons.push("seek_cluster");
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
      abilityHint: null,
    };

    if (
      enemy &&
      ctx.cooldownReady &&
      distance <= engageRange * 1.05 &&
      !lowHealth &&
      (strategy.aggression >= SHOOT_AGGRESSION_FLOOR || distance <= this.closeRange)
    ) {
      next.shouldShoot = true;
      reasons.push("shoot_enemy");
    }

    if (ctx.abilityReady) {
      const ability = this._pickAbility({
        enemy,
        distance,
        engageRange,
        weaponRange,
        lowHealth,
        outnumbered,
        nearbyEnemyCount,
        strategy,
        recentDamageTaken: Number(ctx.recentDamageTaken) || 0,
        stationaryMs: Number(ctx.stationaryMs) || 0,
        reasons,
      });
      if (ability) {
        next.shouldUseAbility = true;
        next.abilityHint = ability;
        reasons.push(`ability:${ability}`);
      }
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
   * Pick a context-appropriate ability id, or null if nothing fits. Caller
   * is responsible for confirming the ability is actually ready (cooldown).
   * Three abilities (must match SmartBot.ABILITIES):
   *   - jump:        gap-close, low-HP escape, OR panic-jump on recent damage
   *   - minePlanting: area denial — behind us during retreat OR when stationary
   *                   for a while in a busy zone, OR random experimentation
   *   - rampage:     burst commit when engaged healthy OR last-stand outnumbered
   *
   * Also rolls an 8% "experimentation" die: pick a random ability from the
   * pool instead of the contextual choice. Lets the population discover
   * unconventional tactical uses (e.g. mines mid-fight, jump as a feint)
   * that PBT/PPO can then reinforce if they pay off.
   */
  _pickAbility({
    enemy,
    distance,
    engageRange,
    weaponRange,
    lowHealth,
    outnumbered,
    nearbyEnemyCount,
    strategy,
    recentDamageTaken,
    stationaryMs,
  }) {
    // Experimentation: small probability of picking a non-contextual ability.
    // Only fires if the agent has any base willingness to use abilities at
    // all; that way Collector (ability_usage=0.2) experiments rarely while
    // Berserker (1.0) experiments often.
    if (
      strategy.ability_usage >= ABILITY_USAGE_FLOOR &&
      Math.random() < ABILITY_EXPLORE_PROB * strategy.ability_usage
    ) {
      const choice = ALL_ABILITIES[Math.floor(Math.random() * ALL_ABILITIES.length)];
      return choice;
    }

    // Last-stand rampage when outnumbered with HP still up — better to take
    // some down with you than die strafing.
    if (
      outnumbered &&
      !lowHealth &&
      strategy.ability_usage >= ABILITY_USAGE_FLOOR
    ) {
      return ABILITY_RAMPAGE;
    }

    // No enemy in range — area denial. If we've been planted in one spot for
    // a few seconds, drop a mine; it might catch a wanderer.
    if (!enemy) {
      if (
        stationaryMs >= 4000 &&
        strategy.ability_usage >= ABILITY_USAGE_FLOOR
      ) {
        return ABILITY_MINE;
      }
      return null;
    }

    // Panic jump: just took meaningful damage, get out of the danger arc.
    // Even passive archetypes use this — it's a survival reflex.
    if (recentDamageTaken >= 25 && distance <= weaponRange) {
      return ABILITY_JUMP;
    }

    if (lowHealth && distance < weaponRange) {
      // Retreating — mine if enemy is hugging (drop it behind), otherwise
      // jump to clear distance.
      if (distance <= 8) return ABILITY_MINE;
      return ABILITY_JUMP;
    }

    if (distance > engageRange && strategy.ability_usage >= ABILITY_USAGE_FLOOR) {
      // Out of range — jump to gap-close.
      return ABILITY_JUMP;
    }

    if (
      distance <= engageRange &&
      strategy.aggression >= ENGAGE_AGGRESSION_FLOOR &&
      strategy.ability_usage >= ABILITY_USAGE_FLOOR
    ) {
      return ABILITY_RAMPAGE;
    }

    return null;
  }

  /**
   * Look for a clearance pattern that puts an obstacle between us and the
   * enemy, i.e. low clearance in the direction we choose AND that direction
   * has at least some component AWAY from the enemy. Returns null if no good
   * cover candidate exists (caller falls back to plain retreat).
   */
  _findCover(enemy, rays) {
    if (!Array.isArray(rays) || rays.length !== RAYCAST_COUNT) return null;
    const enemyDirX = enemy.dirX || 0;
    const enemyDirZ = enemy.dirZ || 0;

    let bestScore = -Infinity;
    let bestX = 0;
    let bestZ = 0;
    for (let i = 0; i < RAYCAST_COUNT; i += 1) {
      const a = (Math.PI * 2 * i) / RAYCAST_COUNT;
      const rx = Math.cos(a);
      const rz = Math.sin(a);
      const awayFromEnemy = -(rx * enemyDirX + rz * enemyDirZ); // [-1, +1]; +1 = directly away
      if (awayFromEnemy < -0.25) continue; // never run TOWARD the enemy
      const clearance = rays[i];
      if (clearance < CLEAR_THRESHOLD * 0.4) continue; // direction is impassable
      // Reward: walls in the direction we move (low clearance flank rays at i±2),
      // some away-from-enemy component, and a passable forward path.
      const flankBlocked = (1 - rays[(i + 2) % RAYCAST_COUNT]) + (1 - rays[(i + RAYCAST_COUNT - 2) % RAYCAST_COUNT]);
      const score = awayFromEnemy * 1.5 + flankBlocked * 0.8 + clearance * 0.4;
      if (score > bestScore) {
        bestScore = score;
        bestX = rx;
        bestZ = rz;
      }
    }
    if (bestScore <= 0) return null;
    return { x: bestX, z: bestZ };
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
TacticalController.ENGAGE_AGGRESSION_FLOOR = ENGAGE_AGGRESSION_FLOOR;
TacticalController.SHOOT_AGGRESSION_FLOOR = SHOOT_AGGRESSION_FLOOR;
TacticalController.CRYSTAL_PRIORITY_FLOOR = CRYSTAL_PRIORITY_FLOOR;
TacticalController.ABILITY_USAGE_FLOOR = ABILITY_USAGE_FLOOR;

module.exports = { TacticalController };
