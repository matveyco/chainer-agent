/**
 * Per-bot fitness tracker.
 * Collects kills, deaths, damage, accuracy, survival time during a match.
 */

class FitnessTracker {
  constructor() {
    this.kills = 0;
    this.deaths = 0;
    this.damageDealt = 0;
    this.damageTaken = 0;
    this.shotsFired = 0;
    this.shotsHit = 0;
    this.survivalTime = 0;
    this.alive = true;
    this._startTime = Date.now();
  }

  recordKill() {
    this.kills++;
  }

  recordDeath() {
    this.deaths++;
    this.alive = false;
  }

  recordRespawn() {
    this.alive = true;
  }

  recordDamageDealt(amount) {
    this.damageDealt += amount;
    this.shotsHit++;
  }

  recordDamageTaken(amount) {
    this.damageTaken += amount;
  }

  recordShot() {
    this.shotsFired++;
  }

  updateSurvival(dt) {
    if (this.alive) {
      this.survivalTime += dt;
    }
  }

  getAccuracy() {
    if (this.shotsFired === 0) return 0;
    return this.shotsHit / this.shotsFired;
  }

  getKDRatio() {
    return this.kills / Math.max(this.deaths, 1);
  }

  /**
   * Compute composite fitness score.
   * @param {Object} weights - Fitness weights from config
   * @returns {number}
   */
  computeFitness(weights) {
    const accuracy = this.getAccuracy();
    return (
      this.kills * weights.killWeight +
      this.deaths * weights.deathPenalty +
      this.damageDealt * weights.damageDealtWeight +
      accuracy * weights.accuracyWeight +
      this.survivalTime * weights.survivalTimeWeight
    );
  }

  toJSON() {
    return {
      kills: this.kills,
      deaths: this.deaths,
      damageDealt: Math.round(this.damageDealt),
      damageTaken: Math.round(this.damageTaken),
      shotsFired: this.shotsFired,
      shotsHit: this.shotsHit,
      accuracy: Math.round(this.getAccuracy() * 100),
      kdRatio: Math.round(this.getKDRatio() * 100) / 100,
      survivalTime: Math.round(this.survivalTime),
    };
  }
}

module.exports = { FitnessTracker };
