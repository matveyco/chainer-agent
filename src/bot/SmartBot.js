/**
 * Smart bot with per-agent ONNX inference brain.
 */

const { AgentBrain } = require("./AgentBrain");
const { StrategicBrain } = require("./StrategicBrain");
const { StateExtractor } = require("./StateExtractor");
const { TacticalController } = require("./TacticalController");
const { FitnessTracker } = require("../metrics/FitnessTracker");
const { encodeInput, encodeShoot } = require("../network/Protocol");
const { rand, getArrayLength, getDirectionArray } = require("../utils/math");

const INPUT_BUFFER_SIZE = 16;
const ABILITIES = Object.freeze(["rampage", "jump", "minePlanting"]);

class SmartBot {
  constructor(userID, _unused, config, agentId, options = {}) {
    this.userID = userID;
    this.agentId = agentId || userID;
    this.displayName = options.displayName || this.agentId;
    this.modelAlias = options.modelAlias || config.training?.defaultModelAlias || "latest";
    this.modelVersion = options.modelVersion ?? null;
    this.policyFamily = options.policyFamily || "arena-main";
    this.archetypeId = options.archetypeId || "tactician";
    this.role = options.role || "league_exploiter"; // PBT league role
    this.mode = options.mode || "training";
    this.track = options.track || "training";
    this.reporter = options.reporter || null;
    this.config = config;
    this.mapName = config.server.mapName;
    this.weaponType = config.server.weaponType;
    this.agentSeed = parseInt(String(this.agentId).replace(/\D/g, ""), 10) || 0;

    this.brain = null;
    this.strategicBrain = null;
    if (config.ollamaApiKey) {
      this.strategicBrain = new StrategicBrain(
        this.agentId,
        config.ollamaApiKey,
        config.ollamaModel || undefined,
        {
          archetypeId: this.archetypeId,
          trainerUrl: config.trainerUrl,
          reporter: options.reporter || null,
          timeoutMs: config.runtime?.strategyCoachTimeoutMs || 90000,
          fallbackModel: config.ollamaFallbackModel || undefined,
        }
      );
    }

    this.stateExtractor = new StateExtractor();
    this.tacticalController = new TacticalController({
      archetypeId: this.archetypeId,
      seed: this.agentSeed,
      safeSize: config.bot?.arenaSafeSize,
    });
    this.fitness = new FitnessTracker();

    this.room = null;
    this.gameState = null;
    this.data = null;
    this.connected = false;
    this.alive = true;
    this.matchEnded = false;

    this.inputBuffer = new Array(INPUT_BUFFER_SIZE);
    for (let i = 0; i < INPUT_BUFFER_SIZE; i++) {
      this.inputBuffer[i] = {
        inputMove: new Float32Array(3),
        target: new Float32Array(3),
        animation: 0,
        speed: 0,
      };
    }
    this.inputIndex = 0;
    this.lastInputIndex = 0;

    this.coolDownTimer = 0;
    this.abilityTimer = rand(5, 15);
    this.searchClosestTimer = 0;
    this.rttTimer = 0;

    this.closestEnemy = null;
    this.closestCrystal = null;
    this.lastMoveX = 0;
    this.lastMoveZ = 0;
    this.positionArray = [0, 0, 0];
    this.positionVector = { x: 0, y: 0, z: 0 };
    this.decisionInFlight = false;
    this.lastAbilityUsed = false;
    this.lastTargetPoint = { x: 0, y: 0, z: 0 };
    this.lastCombatAt = Date.now();
    this.lastShotAt = 0;
    this.decisionsMade = 0;
    this.shotsFired = 0;
    this.tacticalOverrides = 0;
    this.tacticalReasons = [];

    this._lastGotKill = false;
    this._lastDied = false;
    this._lastDamageDealt = 0;
    this._lastDamageTaken = 0;
    this._tickCounter = 0;
    this._stateUpdatesSeen = 0;
    this._inputsSent = 0;
    this._lastObstacleRays = null;
    // Per-enemy {x, z, t} for velocity estimation used by lead-target aim.
    // Keyed by enemy id; trimmed opportunistically in _updateEnemyTrack.
    this._enemyTracks = new Map();
    // Crystal-pickup proxy: when score jumps without a recent kill or damage
    // deal, that's most likely a crystal pickup. Kept per match.
    this._lastScoreSeen = 0;
    this._crystalPickupsApprox = 0;
    // Per-step crystal delta for the reward signal (consumed once per
    // recordStep, then reset).
    this._crystalDeltaSinceLastStep = 0;
    // Cached cluster direction (refreshed on each decision tick).
    this._lastEnemyClusterDir = null;
    // Streak / first-blood tracking — used by the new shaping rewards.
    this._currentKillStreak = 0;
    this._matchHasFirstBlood = false; // true after this bot's first kill of the match
    // Wall-shot counter (LOS-vetoed shoot intents). Bumped by _applyDecision
    // on every veto, drained per recordStep.
    this._wallShotsSinceLastStep = 0;
  }

  async initBrain(trainerUrl) {
    this.brain = new AgentBrain(this.agentId, trainerUrl, {
      modelAlias: this.modelAlias,
      modelVersion: this.modelVersion,
      policyFamily: this.policyFamily,
      archetypeId: this.archetypeId,
      role: this.role,
      rewardConfig: this.config.reward,
      reporter: this.reporter,
      strategyProvider: () => this.strategicBrain?.getStrategyVector?.() || null,
    });
    await this.brain.init();
    this.brain.resetMatch();
  }

  update(dt) {
    if (!this.connected || !this.data || this.matchEnded) return;
    if (!this.alive || this.data.health === 0) {
      this.fitness.updateSurvival(0);
      return;
    }

    this.fitness.updateSurvival(dt);
    this._tickCounter += 1;

    const myPos = this.gameState.getPosition(this.userID);
    if (myPos) {
      this.positionArray[0] = this.positionVector.x = myPos.x;
      this.positionArray[1] = this.positionVector.y = myPos.y;
      this.positionArray[2] = this.positionVector.z = myPos.z;
    }

    const input = this.inputBuffer[this.inputIndex];
    if (!input) return;

    if (this.coolDownTimer > 0) this.coolDownTimer -= dt;
    if (this.abilityTimer > 0) this.abilityTimer -= dt;
    if (this.searchClosestTimer > 0) this.searchClosestTimer -= dt;
    if (this.rttTimer > 0) this.rttTimer -= dt;

    const weaponRange = this.data.weaponTargetDistance || 20;

    if (this.searchClosestTimer <= 0) {
      this.closestEnemy = this.gameState.getClosestEnemy(this.userID, weaponRange);
      this.closestCrystal = this.gameState.getClosestCrystal(this.positionVector);
      // Track every nearby enemy for velocity estimation (lead-target aim).
      // Range is 1.5× weaponRange so we have a velocity estimate ready by the
      // time an enemy enters firing range.
      const nearby = this.gameState.getNearbyEnemies(this.userID, weaponRange * 1.5);
      this._updateEnemyTracks(nearby);
      this._lastEnemyClusterDir = this._computeEnemyClusterDir(nearby);
      // Outnumbered detection: count enemies within close-quarters range so the
      // tactical controller can trigger retreat-style behaviour even at full HP.
      const closeNearby = this.gameState.getNearbyEnemies(this.userID, weaponRange);
      this._lastNearbyEnemyCount = closeNearby.length;
      // Crystal-pickup proxy: any score gain not explained by recent damage
      // dealt or kills is most likely a crystal pickup. Sample on the same
      // cadence as the enemy search so we don't miss tight pickup windows.
      this._sampleCrystalProxy();
      this.searchClosestTimer = 0.5;
      if (this.rttTimer <= 0) {
        this.room.send("room:rtt");
        this.rttTimer = 3;
      }
    }

    if (
      this._tickCounter % (this.config.bot.decisionIntervalTicks || 3) === 0 &&
      this.brain &&
      !this.decisionInFlight
    ) {
      this.stateExtractor.setLastMove(this.lastMoveX, this.lastMoveZ);
      this.stateExtractor.setRecentCombat(this._lastDamageDealt, this._lastDamageTaken);
      const stateVector = this.stateExtractor.extract(
        this.gameState,
        this.userID,
        weaponRange,
        this.coolDownTimer > 0,
        this.data
      );

      // Snapshot the 8 obstacle raycasts (state vector indices 24..31) so the
      // tactical controller can route around walls. Slice copies the values —
      // the underlying buffer gets reused on the next extract().
      this._lastObstacleRays = Array.from(stateVector.slice(24, 32));

      this.decisionInFlight = true;
      this.brain.decide(stateVector)
        .then((nnDecision) => this._applyDecision(nnDecision))
        .catch(() => {})
        .finally(() => {
          this.decisionInFlight = false;
        });

      // Maintain kill streak + first-blood signals for the new reward shaping.
      // Streak resets on death; first-blood fires once per match per agent.
      let gotFirstBloodNow = false;
      if (this._lastGotKill) {
        this._currentKillStreak += 1;
        if (!this._matchHasFirstBlood) {
          gotFirstBloodNow = true;
          this._matchHasFirstBlood = true;
        }
      }
      if (this._lastDied) this._currentKillStreak = 0;

      // Drain crystal-delta and wall-shot counters into this step's reward.
      const crystalDelta = this._crystalDeltaSinceLastStep;
      const wallShotsRecent = this._wallShotsSinceLastStep;
      this._crystalDeltaSinceLastStep = 0;
      this._wallShotsSinceLastStep = 0;

      this.brain.recordStep({
        currentScore: this.data.score || 0,
        kills: this.data.kills || 0,
        deaths: this.data.deaths || 0,
        died: this._lastDied,
        gotKill: this._lastGotKill,
        damageDealt: this._lastDamageDealt,
        damageTaken: this._lastDamageTaken,
        survivalSeconds: dt,
        abilityUsed: this.lastAbilityUsed,
        shotAccuracy: this.fitness.getAccuracy(),
        done: false,
        // New per-step shaping signals:
        crystalDelta,
        currentStreak: this._currentKillStreak,
        gotFirstBlood: gotFirstBloodNow,
        nearbyEnemyCount: this._lastNearbyEnemyCount || 0,
        wallShotsRecent,
      });

      if (this._lastDamageDealt > 0 || this._lastDamageTaken > 0 || this._lastGotKill || this._lastDied) {
        this.lastCombatAt = Date.now();
      }

      // Carry damage-taken into the *next* decision so the tactical ability
      // picker can react to it (panic-jump). Clearing _lastDamageTaken here
      // would make the signal vanish before decide() consumes it.
      this._recentDamageTakenWindow = this._lastDamageTaken;

      this._lastGotKill = false;
      this._lastDied = false;
      this._lastDamageDealt = 0;
      this._lastDamageTaken = 0;
      this.lastAbilityUsed = false;
    }

    const distFromCenter = getArrayLength(this.positionArray);
    if (this.mapName === "arena" && distFromCenter > this.config.bot.arenaSafeSize) {
      const dirToCenter = getDirectionArray(this.positionArray, [0, 0, 0]);
      input.inputMove.set(new Float32Array(dirToCenter));
      input.target.set(new Float32Array([0, 0, 0]));
      input.animation = 2;
      input.speed = 7;
      this._sendInputsBatch([input]);
      this.inputIndex = (this.inputIndex + 1) % INPUT_BUFFER_SIZE;
      return;
    }

    this._sendPosition();
    this.inputIndex = (this.inputIndex + 1) % INPUT_BUFFER_SIZE;
  }

  _applyDecision(nnDecision = null) {
    if (this.matchEnded) return;
    this.decisionsMade += 1;

    // Tactical controller is the high-level decision maker; the neural net is
    // a learned advisor that gets blended in by `policy_blend_alpha` (PBT-
    // mutated per-agent reward weight). At α=0 we get pure tactical (today's
    // behaviour). At α=1 we get pure NN. Discrete actions (shoot/ability) get
    // OR'd from the NN side rather than blended — tactical's shouldShoot is
    // also LOS-gated so it stops shooting walls, while NN's shouldShoot can
    // fire freely (intentional: lets the NN discover unusual tactics).
    const tactical = this.tacticalController.decide({
      enemy: this.closestEnemy ? this._getEnemyContext() : null,
      crystal: this.closestCrystal ? this._getCrystalContext() : null,
      enemyClusterDir: this._lastEnemyClusterDir,
      nearbyEnemyCount: this._lastNearbyEnemyCount || 0,
      recentDamageTaken: this._recentDamageTakenWindow || 0,
      stationaryMs: this._computeStationaryMs(),
      strategy: this.strategicBrain?.getStrategyVector?.() || null,
      healthPercent: (this.data?.health || 100) / 100,
      weaponRange: this.data?.weaponTargetDistance || 20,
      cooldownReady: this.coolDownTimer <= 0,
      abilityReady: this._hasReadyAbility(),
      position: this.positionVector,
      distanceFromCenter: getArrayLength(this.positionArray),
      obstacleRays: this._lastObstacleRays,
    });

    const alpha = this._getPolicyBlendAlpha();
    const decision = this._blendActions(tactical.action, nnDecision, alpha);

    // LOS gate (TACTICAL side only): if the tactical controller wanted to
    // shoot but our line of sight to the enemy is blocked by an obstacle,
    // hold fire. This kills the wall-shooting bug. NN's shoot signal (mixed
    // in via `decision.shouldShoot` above only when α>0 and NN agrees) is
    // intentionally NOT gated — the NN can choose to learn LOS itself.
    if (
      tactical.action.shouldShoot &&
      !(alpha > 0 && nnDecision?.shouldShoot) &&
      this.closestEnemy &&
      !this._hasLineOfSight(this.closestEnemy)
    ) {
      decision.shouldShoot = false;
      this.tacticalReasons = [...tactical.reasons, "los_blocked"];
      this.reporter?.incrementCounter("losVetoes");
      // Wall-shot signal for the reward function: every veto is a tactical
      // controller mistake we want to discourage. Drained per recordStep.
      this._wallShotsSinceLastStep += 1;
    } else {
      this.tacticalReasons = tactical.reasons;
    }

    this.tacticalOverrides += 1;
    this.reporter?.incrementCounter("tacticalOverrides");
    if (alpha > 0 && nnDecision) this.reporter?.incrementCounter("blendedDecisions");

    const input = this.inputBuffer[this.inputIndex];
    if (!input) return;

    // Stuck-in-corner escape. If we've barely moved over the last 3 s we're
    // probably wedged against geometry. Override the move vector with a
    // randomised escape (back away + lateral) for a brief window.
    const escape = this._maybeEscapeStuckCorner();

    let moveX = escape ? escape.x : decision.moveX;
    let moveZ = escape ? escape.z : decision.moveZ;
    const moveLen = Math.sqrt(moveX * moveX + moveZ * moveZ);
    if (moveLen > 0) {
      moveX /= moveLen;
      moveZ /= moveLen;
    }

    this.lastMoveX = moveX * 7;
    this.lastMoveZ = moveZ * 7;

    input.inputMove[0] = moveX;
    input.inputMove[1] = 0;
    input.inputMove[2] = moveZ;
    input.speed = 7;
    input.animation = moveLen > 0.1 ? 2 : 0;

    if (this.closestEnemy) {
      const aim = this._computeLeadTarget(this.closestEnemy);
      input.target[0] = aim.x + decision.aimOffsetX;
      input.target[1] = aim.y;
      input.target[2] = aim.z + decision.aimOffsetZ;
    } else if (this.closestCrystal) {
      input.target[0] = this.closestCrystal.x;
      input.target[1] = this.closestCrystal.y || 0;
      input.target[2] = this.closestCrystal.z;
    } else {
      input.target[0] = this.positionArray[0] + moveX * 10;
      input.target[1] = 0;
      input.target[2] = this.positionArray[2] + moveZ * 10;
    }

    this.lastTargetPoint = {
      x: input.target[0],
      y: input.target[1],
      z: input.target[2],
    };

    if (decision.shouldShoot && this.closestEnemy && this.coolDownTimer <= 0 && this.data?.health > 0) {
      this._shoot(this.lastTargetPoint);
    }

    if (decision.shouldUseAbility && this.abilityTimer <= 0) {
      this._useAbility(decision.abilityHint);
      this.abilityTimer = rand(5, 15);
    }
  }

  /**
   * Blend tactical and NN actions. Continuous outputs (move, aim) get a
   * convex combination; discrete outputs (shoot, ability) keep tactical's
   * decision as the floor and let the NN add to it when α > 0. Ability
   * choice still comes from tactical (it knows what's contextually right).
   */
  _blendActions(tacticalAction, nnDecision, alpha) {
    if (!nnDecision || alpha <= 0) {
      return { ...tacticalAction };
    }
    const a = Math.max(0, Math.min(1, alpha));
    return {
      moveX: a * (Number(nnDecision.moveX) || 0) + (1 - a) * tacticalAction.moveX,
      moveZ: a * (Number(nnDecision.moveZ) || 0) + (1 - a) * tacticalAction.moveZ,
      aimOffsetX: a * (Number(nnDecision.aimOffsetX) || 0) + (1 - a) * tacticalAction.aimOffsetX,
      aimOffsetZ: a * (Number(nnDecision.aimOffsetZ) || 0) + (1 - a) * tacticalAction.aimOffsetZ,
      shouldShoot: tacticalAction.shouldShoot || !!nnDecision.shouldShoot,
      shouldUseAbility: tacticalAction.shouldUseAbility || !!nnDecision.shouldUseAbility,
      abilityHint: tacticalAction.abilityHint,
    };
  }

  _getPolicyBlendAlpha() {
    const fromBrain = this.brain?.getPolicyBlendAlpha?.();
    if (Number.isFinite(fromBrain)) return Math.max(0, Math.min(1, fromBrain));
    return 0.1;
  }

  /**
   * Cast a ray from our position toward the enemy and check whether an
   * obstacle blocks it before the enemy distance. Returns true iff we have
   * a clear shot.
   */
  _hasLineOfSight(enemy) {
    if (!this.gameState || typeof this.gameState.rayDistanceToObstacle !== "function") return true;
    if (!enemy?.position) return true;
    const dx = enemy.position.x - this.positionArray[0];
    const dz = enemy.position.z - this.positionArray[2];
    const distance = Math.hypot(dx, dz);
    if (distance < 0.5) return true;
    const dir = { x: dx / distance, z: dz / distance };
    const origin = { x: this.positionArray[0], z: this.positionArray[2] };
    // Slightly less than the actual enemy distance so the ray reports a hit
    // only when an obstacle is genuinely between us — touching the enemy
    // hitbox itself doesn't count.
    const reachable = this.gameState.rayDistanceToObstacle(origin, dir, distance - 0.5);
    return reachable >= distance - 0.5;
  }

  /**
   * Detect when we've barely moved for ~3 s (probably wedged against a wall
   * the avoidance heuristic can't escape) and produce a one-shot escape
   * vector. We sample position every decision tick into a small ring buffer
   * and pick a random direction biased AWAY from the geometry mass.
   *
   * Returns a {x, z} unit-ish vector when an escape is active, else null.
   */
  /**
   * Time spent inside a small radius (~2.5 m) using the same position ring
   * the stuck-escape uses. Returns ms; 0 if we don't have enough history or
   * if the bot has been moving freely.
   */
  _computeStationaryMs() {
    if (!this._stuckSamples?.length) return 0;
    const now = Date.now();
    let cx = 0;
    let cz = 0;
    for (const s of this._stuckSamples) {
      cx += s.x;
      cz += s.z;
    }
    cx /= this._stuckSamples.length;
    cz /= this._stuckSamples.length;
    const px = this.positionArray[0];
    const pz = this.positionArray[2];
    const radius = Math.hypot(px - cx, pz - cz);
    if (radius > 2.5) return 0;
    return now - this._stuckSamples[0].t;
  }

  _maybeEscapeStuckCorner() {
    const now = Date.now();
    if (!this._stuckSamples) {
      this._stuckSamples = [];
      this._stuckEscapeUntil = 0;
      this._stuckEscapeDir = null;
    }
    // Honor an active escape window before sampling — we don't want to keep
    // re-detecting "stuck" while we're literally executing the escape.
    if (now < this._stuckEscapeUntil && this._stuckEscapeDir) {
      return this._stuckEscapeDir;
    }
    this._stuckSamples.push({
      x: this.positionArray[0],
      z: this.positionArray[2],
      t: now,
    });
    // Keep the last ~5 s of position history. Samples come in at the
    // decision rate (~20 Hz) so a 3 s span is ~60 entries.
    while (this._stuckSamples.length && now - this._stuckSamples[0].t > 5000) {
      this._stuckSamples.shift();
    }
    // Require at least ~3 s of actual elapsed history before judging "stuck"
    // (length alone isn't enough — at 20 Hz, 6 samples is only 300 ms).
    if (now - this._stuckSamples[0].t < 3000) return null;
    // Centroid-radius check: total displacement isn't enough — a bot
    // oscillating along a wall can travel >1 m without making progress.
    // Compute the centroid of the last 3 s of positions and check the
    // bounding radius. If every sample sits within ~2 m of the centroid the
    // bot is wedged regardless of how far it "moved".
    let cx = 0;
    let cz = 0;
    for (const s of this._stuckSamples) {
      cx += s.x;
      cz += s.z;
    }
    cx /= this._stuckSamples.length;
    cz /= this._stuckSamples.length;
    let radius = 0;
    for (const s of this._stuckSamples) {
      const d = Math.hypot(s.x - cx, s.z - cz);
      if (d > radius) radius = d;
    }
    if (radius >= 2.0) return null; // moving freely within a 4m+ region

    // Pick the most-clear ray as escape direction so we don't push back into
    // the wall we were stuck on. If we have no rays, just pick a random
    // direction so something changes.
    const rays = this._lastObstacleRays;
    let dir = null;
    if (Array.isArray(rays) && rays.length === 8) {
      let bestIdx = 0;
      let bestClearance = -1;
      for (let i = 0; i < 8; i += 1) {
        if (rays[i] > bestClearance) {
          bestClearance = rays[i];
          bestIdx = i;
        }
      }
      const angle = (Math.PI * 2 * bestIdx) / 8;
      dir = { x: Math.cos(angle), z: Math.sin(angle) };
    } else {
      const angle = Math.random() * Math.PI * 2;
      dir = { x: Math.cos(angle), z: Math.sin(angle) };
    }
    this._stuckEscapeDir = dir;
    this._stuckEscapeUntil = now + 1000; // commit to escape for 1 s
    this._stuckSamples = []; // reset so the next stuck check waits a fresh 3 s
    this.reporter?.incrementCounter("stuckEscapes");
    return dir;
  }

  _shoot(targetPoint = null) {
    if (!this.closestEnemy || !this.data) return;
    const target = this._buildShootTarget(targetPoint);
    const buffer = encodeShoot(this.positionArray, target, this.data.weaponType || this.weaponType);
    this.room.send("room:player:shoot", buffer);
    this.fitness.recordShot();
    this.shotsFired += 1;
    this.lastShotAt = Date.now();
    this.reporter?.incrementCounter("shotsFired");
    this.coolDownTimer = this._getWeaponCooldownSeconds() + rand(0.05, 0.25);
  }

  _useAbility(hint = null) {
    if (!this.data?.abilities) return;
    // Honor the tactical controller's preferred ability if it's actually
    // ready; otherwise fall through to any other ready ability so we don't
    // sit on a 5–15s cooldown waiting for the perfect one.
    const order = [];
    if (hint && ABILITIES.includes(hint)) order.push(hint);
    for (const id of ABILITIES) {
      if (id !== hint) order.push(id);
    }
    for (const ability of order) {
      const abilityData = this.data.abilities.get?.(ability);
      if (abilityData?.ready) {
        this.room.send("room:player:ability:use", { ability });
        this.fitness.recordAbilityUse();
        this.lastAbilityUsed = true;
        return;
      }
    }
  }

  _sendPosition() {
    if (this.inputIndex === this.lastInputIndex) return;
    const inputs = [];
    let index = this.lastInputIndex;
    while (index !== this.inputIndex) {
      inputs.push(this.inputBuffer[index]);
      index = (index + 1) % INPUT_BUFFER_SIZE;
    }
    this.lastInputIndex = this.inputIndex;
    this._sendInputsBatch(inputs);
  }

  _sendInputsBatch(inputs) {
    const buffer = encodeInput(this.userID, inputs);
    this.room.send("room:player:input", buffer);
    this._inputsSent += inputs.length;
    this.reporter?.incrementCounter("inputsSent", inputs.length);
  }

  markStateUpdate() {
    this._stateUpdatesSeen += 1;
    this.reporter?.incrementCounter("stateUpdates");
  }

  getRuntimeStats() {
    const policyLedDecisions = Math.max(0, this.decisionsMade - this.tacticalOverrides);
    const shotRate = this.decisionsMade > 0 ? this.shotsFired / this.decisionsMade : 0;
    return {
      inputsSent: this._inputsSent,
      stateUpdates: this._stateUpdatesSeen,
      decisionsMade: this.decisionsMade,
      policyLedDecisions,
      shotsFired: this.shotsFired,
      shotRate: +shotRate.toFixed(4),
      tacticalOverrides: this.tacticalOverrides,
      tacticalOverrideRatio: this.decisionsMade > 0 ? +(this.tacticalOverrides / this.decisionsMade).toFixed(4) : 0,
      combatInactivityMs: Math.max(0, Date.now() - Math.max(this.lastCombatAt || 0, this.lastShotAt || 0)),
      modelAlias: this.modelAlias,
      modelVersion: this.brain?.modelVersion || 0,
      policyFamily: this.policyFamily,
      archetypeId: this.archetypeId,
      track: this.track,
    };
  }

  getFitness(weights) {
    return this.fitness.computeFitness(weights);
  }

  dispose() {
    this.matchEnded = true;
    this.connected = false;
    this.room = null;
    this.data = null;
    this.gameState = null;
    this.closestEnemy = null;
    this.closestCrystal = null;
  }

  _getEnemyContext() {
    if (!this.closestEnemy) return null;
    const dx = this.closestEnemy.position.x - this.positionArray[0];
    const dz = this.closestEnemy.position.z - this.positionArray[2];
    const length = Math.hypot(dx, dz) || 1;
    return {
      distance: this.closestEnemy.distance || length,
      dirX: dx / length,
      dirZ: dz / length,
    };
  }

  _getCrystalContext() {
    if (!this.closestCrystal) return null;
    const dx = this.closestCrystal.x - this.positionArray[0];
    const dz = this.closestCrystal.z - this.positionArray[2];
    const length = Math.hypot(dx, dz) || 1;
    return {
      dirX: dx / length,
      dirZ: dz / length,
      distance: this.closestCrystal.distance || length,
    };
  }

  _hasReadyAbility() {
    if (!this.data?.abilities?.values) return false;
    for (const ability of this.data.abilities.values()) {
      if (ability?.ready) return true;
    }
    return false;
  }

  _getWeaponCooldownSeconds() {
    const raw = Number(this.data?.weaponCoolDown);
    if (!Number.isFinite(raw) || raw <= 0) return 1;
    return raw > 10 ? raw / 1000 : raw;
  }

  _buildShootTarget(targetPoint = null) {
    const strategy = this.strategicBrain?.getStrategyVector?.() || {};
    const accuracy = Math.max(0, Math.min(1, Number(strategy.accuracy_focus ?? 0.5) || 0.5));
    const jitter = 0.1 + (1 - accuracy) * 0.6;
    const fallback = this.closestEnemy?.position || { x: 0, y: 0, z: 0 };
    const base = targetPoint || this.lastTargetPoint || fallback;
    return [
      Number(base.x ?? fallback.x) + rand(-jitter, jitter),
      Number(base.y ?? fallback.y) || 0,
      Number(base.z ?? fallback.z) + rand(-jitter, jitter),
    ];
  }

  /**
   * Predict where the enemy will be when our projectile arrives. Linear
   * extrapolation from the last two observed positions; aim point sits at
   * `enemy.position + velocity * travelTime`. accuracy_focus modulates how
   * much we lead — high focus = full lead, low focus = mostly dead-on so we
   * keep the existing "spray and pray" behaviour for sloppy archetypes.
   */
  _computeLeadTarget(enemy) {
    const fallback = {
      x: enemy.position.x,
      y: enemy.position.y || 0,
      z: enemy.position.z,
    };
    if (!enemy?.id) return fallback;

    const track = this._enemyTracks.get(enemy.id);
    if (!track) return fallback;

    const dt = (Date.now() - track.t) / 1000;
    if (dt <= 0 || dt > 1.0) return fallback; // stale; don't trust velocity

    const vx = (enemy.position.x - track.x) / dt;
    const vz = (enemy.position.z - track.z) / dt;
    // Cap velocity to game-plausible movement speed (~7-12 m/s).
    const speed = Math.hypot(vx, vz);
    if (speed > 15) return fallback;

    const dx = enemy.position.x - this.positionArray[0];
    const dz = enemy.position.z - this.positionArray[2];
    const distance = Math.hypot(dx, dz);
    const projectileSpeed = this._getProjectileSpeed();
    const travelTime = distance / projectileSpeed;

    const strategy = this.strategicBrain?.getStrategyVector?.() || {};
    const focus = Math.max(0, Math.min(1, Number(strategy.accuracy_focus ?? 0.5) || 0.5));
    const leadFactor = focus; // 0 = no lead, 1 = full lead

    return {
      x: enemy.position.x + vx * travelTime * leadFactor,
      y: enemy.position.y || 0,
      z: enemy.position.z + vz * travelTime * leadFactor,
    };
  }

  _getProjectileSpeed() {
    // Rocket projectile in this arena travels ~30 m/s. The exact constant
    // isn't exposed in any contract we have, but lead-target with this value
    // is much closer than dead-aim so it's still a clear win.
    return 30;
  }

  /**
   * Update per-enemy {x, z, t} tracks for velocity estimation. Called on the
   * same 0.5s cadence as enemy search; entries that haven't been seen for
   * >2s get dropped so the map can't grow unbounded across matches.
   */
  _updateEnemyTracks(nearbyEnemies) {
    const now = Date.now();
    const seen = new Set();
    for (const enemy of nearbyEnemies) {
      if (!enemy?.id || !enemy?.position) continue;
      seen.add(enemy.id);
      this._enemyTracks.set(enemy.id, {
        x: enemy.position.x,
        z: enemy.position.z,
        t: now,
      });
    }
    for (const [id, track] of this._enemyTracks) {
      if (!seen.has(id) && now - track.t > 2000) {
        this._enemyTracks.delete(id);
      }
    }
  }

  /**
   * Direction to the centroid of nearby enemies. Powers the
   * "no enemy in range, no crystal — go where the fight is" fallback in
   * TacticalController.decide. Returns null if there's nothing to head toward.
   */
  _computeEnemyClusterDir(nearbyEnemies) {
    if (!nearbyEnemies?.length) return null;
    let sumX = 0;
    let sumZ = 0;
    let count = 0;
    for (const enemy of nearbyEnemies) {
      if (!enemy?.position) continue;
      sumX += enemy.position.x;
      sumZ += enemy.position.z;
      count += 1;
    }
    if (!count) return null;
    const cx = sumX / count - this.positionArray[0];
    const cz = sumZ / count - this.positionArray[2];
    const len = Math.hypot(cx, cz);
    if (len < 1e-3) return null;
    return { dirX: cx / len, dirZ: cz / len };
  }

  /**
   * Approximate crystal pickups by counting score increases unaccompanied by
   * recent damage-dealt or kills. This is a proxy because the server doesn't
   * expose a "you picked up a crystal" event to bots, but it gives us a
   * signal that's at least directionally correct in match summaries.
   */
  _sampleCrystalProxy() {
    const score = Number(this.data?.score) || 0;
    const delta = score - this._lastScoreSeen;
    this._lastScoreSeen = score;
    if (delta <= 0) return;
    const recentlyHit = Date.now() - this.lastCombatAt < 1500;
    if (!recentlyHit && delta > 0) {
      this._crystalPickupsApprox += 1;
      this._crystalDeltaSinceLastStep += 1;
      this.reporter?.incrementCounter("crystalPickupsApprox");
    }
  }

  resetMatchStats() {
    this._enemyTracks.clear();
    this._lastEnemyClusterDir = null;
    this._currentKillStreak = 0;
    this._matchHasFirstBlood = false;
    this._crystalDeltaSinceLastStep = 0;
    this._wallShotsSinceLastStep = 0;
    this._lastScoreSeen = 0;
    this._crystalPickupsApprox = 0;
  }

  getCrystalPickupsApprox() {
    return this._crystalPickupsApprox;
  }
}

module.exports = { SmartBot };
