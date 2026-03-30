/**
 * NEAT population management.
 * Wraps neataptic's Neat class with project-specific configuration.
 */

const { Neat, methods, architect } = require("neataptic");
const { StateExtractor } = require("../bot/StateExtractor");

const OUTPUT_COUNT = 6; // moveX, moveZ, aimOffsetX, aimOffsetZ, shoot, ability

class Population {
  /**
   * @param {Object} evolutionConfig - Evolution parameters from config
   */
  constructor(evolutionConfig) {
    this.config = evolutionConfig;
    this.generation = 0;

    this.neat = new Neat(StateExtractor.INPUT_COUNT, OUTPUT_COUNT, null, {
      popsize: evolutionConfig.populationSize,
      elitism: evolutionConfig.elitism,
      mutationRate: evolutionConfig.mutationRate,
      mutationAmount: evolutionConfig.mutationAmount,
      selection: methods.selection.POWER,
      mutation: [
        methods.mutation.ADD_NODE,
        methods.mutation.SUB_NODE,
        methods.mutation.ADD_CONN,
        methods.mutation.SUB_CONN,
        methods.mutation.MOD_WEIGHT,
        methods.mutation.MOD_BIAS,
        methods.mutation.MOD_ACTIVATION,
        methods.mutation.ADD_GATE,
        methods.mutation.SUB_GATE,
        methods.mutation.ADD_SELF_CONN,
        methods.mutation.SUB_SELF_CONN,
        methods.mutation.ADD_BACK_CONN,
        methods.mutation.SUB_BACK_CONN,
      ],
    });
  }

  /**
   * Get all genomes (networks) for the current generation.
   * @returns {Array<Network>}
   */
  getGenomes() {
    return this.neat.population;
  }

  /**
   * Set fitness score on a genome by index.
   */
  setFitness(index, score) {
    if (index < this.neat.population.length) {
      this.neat.population[index].score = score;
    }
  }

  /**
   * Evolve the population to the next generation.
   * @returns {Array<Network>} New generation
   */
  async evolve() {
    await this.neat.evolve();
    this.generation++;
    return this.neat.population;
  }

  /**
   * Get the best genome (highest fitness).
   */
  getBestGenome() {
    let best = this.neat.population[0];
    for (const genome of this.neat.population) {
      if ((genome.score || 0) > (best.score || 0)) {
        best = genome;
      }
    }
    return best;
  }

  /**
   * Get current generation number.
   */
  getGeneration() {
    return this.generation;
  }

  /**
   * Get population size.
   */
  getPopulationSize() {
    return this.neat.population.length;
  }

  /**
   * Get neuron count range across population.
   */
  getNeuronRange() {
    let min = Infinity;
    let max = 0;
    for (const genome of this.neat.population) {
      const count = genome.nodes.length;
      if (count < min) min = count;
      if (count > max) max = count;
    }
    return { min, max };
  }

  /**
   * Get connection count range across population.
   */
  getConnectionRange() {
    let min = Infinity;
    let max = 0;
    for (const genome of this.neat.population) {
      const count = genome.connections.length;
      if (count < min) min = count;
      if (count > max) max = count;
    }
    return { min, max };
  }

  /**
   * Load population from saved genomes.
   * @param {Array<Object>} genomeJSONs - Array of serialized genome JSONs
   */
  loadFromJSON(genomeJSONs) {
    const { Network } = require("neataptic");
    this.neat.population = genomeJSONs.map((json) => Network.fromJSON(json));
  }

  /**
   * Export all genomes as JSON.
   */
  toJSON() {
    return this.neat.population.map((genome) => genome.toJSON());
  }
}

module.exports = { Population };
