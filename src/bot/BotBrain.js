/**
 * Neural network brain wrapper.
 * Wraps a neataptic Network for inference and output processing.
 */

class BotBrain {
  /**
   * @param {Object} network - A neataptic Network instance
   */
  constructor(network) {
    this.network = network;
  }

  /**
   * Run neural network inference and map outputs to game actions.
   * @param {Float64Array} inputVector - 14-element normalized input vector
   * @returns {Object} Action decisions
   */
  decide(inputVector) {
    const outputs = this.network.activate(Array.from(inputVector));

    return {
      // Movement direction (tanh-like: outputs are already 0-1 from sigmoid, remap to -1..1)
      moveX: outputs[0] * 2 - 1,
      moveZ: outputs[1] * 2 - 1,

      // Aim offset (scaled to ±3 units for jitter around enemy position)
      aimOffsetX: (outputs[2] * 2 - 1) * 3,
      aimOffsetZ: (outputs[3] * 2 - 1) * 3,

      // Binary decisions (threshold at 0.5)
      shouldShoot: outputs[4] > 0.5,
      shouldUseAbility: outputs[5] > 0.5,
    };
  }
}

module.exports = { BotBrain };
