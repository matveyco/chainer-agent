const { getDefaultArchetype } = require("./archetypes");

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalize2D(x, z) {
  const length = Math.hypot(x, z);
  if (length <= 1e-6) {
    return { x: 0, z: 0, length: 0 };
  }
  return {
    x: x / length,
    z: z / length,
    length,
  };
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

  stabilize(action, ctx = {}) {
    const next = {
      moveX: Number(action?.moveX) || 0,
      moveZ: Number(action?.moveZ) || 0,
      aimOffsetX: clamp(Number(action?.aimOffsetX) || 0, -3, 3),
      aimOffsetZ: clamp(Number(action?.aimOffsetZ) || 0, -3, 3),
      shouldShoot: !!action?.shouldShoot,
      shouldUseAbility: !!action?.shouldUseAbility,
    };

    const reasons = [];
    const strategy = this._resolveStrategy(ctx.strategy);
    const move = normalize2D(next.moveX, next.moveZ);
    const passive = (ctx.passiveMs || 0) >= this.passiveWindowMs;

    if (ctx.enemy) {
      this._stabilizeCombat(next, ctx, strategy, move, passive, reasons);
    } else if (ctx.crystal && (move.length < 0.2 || passive)) {
      next.moveX = ctx.crystal.dirX;
      next.moveZ = ctx.crystal.dirZ;
      reasons.push("seek_crystal");
    } else if (move.length < 0.12) {
      const centerX = -(ctx.position?.x || 0);
      const centerZ = -(ctx.position?.z || 0);
      const roam = normalize2D(centerX + 0.35 * this.strafeSign, centerZ - 0.35 * this.strafeSign);
      next.moveX = roam.x;
      next.moveZ = roam.z;
      reasons.push("roam_center");
    }

    if ((ctx.distanceFromCenter || 0) > this.safeSize * 0.92) {
      const retreat = normalize2D(-(ctx.position?.x || 0), -(ctx.position?.z || 0));
      next.moveX = retreat.x;
      next.moveZ = retreat.z;
      reasons.push("return_to_center");
    }

    const normalizedMove = normalize2D(next.moveX, next.moveZ);
    next.moveX = normalizedMove.x;
    next.moveZ = normalizedMove.z;
    next.aimOffsetX = clamp(next.aimOffsetX, -3, 3);
    next.aimOffsetZ = clamp(next.aimOffsetZ, -3, 3);

    return {
      action: next,
      overridden: reasons.length > 0,
      reasons,
    };
  }

  _stabilizeCombat(action, ctx, strategy, move, passive, reasons) {
    const enemy = ctx.enemy;
    const distance = enemy.distance || Infinity;
    const weaponRange = Math.max(6, ctx.weaponRange || 20);
    const engageRange = weaponRange * this.enemyRangeRatio;
    const shouldForceMovement = move.length < 0.22 || passive || distance < this.closeRange;
    const lowHealth = (ctx.healthPercent || 1) <= strategy.retreat_threshold;

    if (shouldForceMovement) {
      const strafeX = -enemy.dirZ * this.strafeSign;
      const strafeZ = enemy.dirX * this.strafeSign;

      let desiredX;
      let desiredZ;
      if (lowHealth) {
        desiredX = -enemy.dirX * 0.75 + strafeX * 0.35;
        desiredZ = -enemy.dirZ * 0.75 + strafeZ * 0.35;
        action.shouldShoot = false;
        action.shouldUseAbility = true;
        reasons.push("retreat_enemy");
      } else if (distance <= engageRange && (strategy.aggression >= 0.45 || passive)) {
        desiredX = enemy.dirX * 0.7 + strafeX * 0.45;
        desiredZ = enemy.dirZ * 0.7 + strafeZ * 0.45;
        reasons.push("engage_enemy");
      } else {
        desiredX = strafeX * 0.9 + enemy.dirX * 0.25;
        desiredZ = strafeZ * 0.9 + enemy.dirZ * 0.25;
        reasons.push("strafe_enemy");
      }

      const desired = normalize2D(desiredX, desiredZ);
      action.moveX = desired.x;
      action.moveZ = desired.z;
    }

    const aimScale = strategy.accuracy_focus >= 0.75 ? 0.1 : 0.25;
    action.aimOffsetX *= aimScale;
    action.aimOffsetZ *= aimScale;

    if (
      ctx.cooldownReady &&
      distance <= Math.max(8, engageRange) &&
      !lowHealth &&
      (strategy.aggression >= 0.35 || passive || distance <= this.closeRange)
    ) {
      action.shouldShoot = true;
      reasons.push("shoot_enemy");
    }

    if (
      ctx.abilityReady &&
      distance <= Math.max(8, weaponRange * 0.75) &&
      !lowHealth &&
      (strategy.ability_usage >= 0.45 || passive)
    ) {
      action.shouldUseAbility = true;
      reasons.push("use_ability");
    }
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

module.exports = { TacticalController };
