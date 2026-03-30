/**
 * Genome persistence — save/load neural network genomes.
 */

const fs = require("fs");
const path = require("path");

class Genome {
  /**
   * Save a single genome to a JSON file.
   */
  static save(network, filePath) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify(network.toJSON(), null, 2));
  }

  /**
   * Load a genome from a JSON file.
   * @returns {Object} Network JSON (use Network.fromJSON() to instantiate)
   */
  static load(filePath) {
    const data = fs.readFileSync(filePath, "utf-8");
    return JSON.parse(data);
  }

  /**
   * Save the best genome for a generation.
   */
  static saveBest(network, generation, bestDir) {
    const filePath = path.join(bestDir, `gen_${generation}.json`);
    Genome.save(network, filePath);
    return filePath;
  }

  /**
   * Save the full population snapshot.
   */
  static savePopulation(population, generation, generationsDir) {
    const filePath = path.join(generationsDir, `gen_${generation}.json`);
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    const data = {
      generation,
      timestamp: new Date().toISOString(),
      genomes: population.toJSON(),
    };
    fs.writeFileSync(filePath, JSON.stringify(data));
    return filePath;
  }

  /**
   * Load a population snapshot.
   * @returns {{ generation: number, genomes: Array<Object> }}
   */
  static loadPopulation(filePath) {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    return data;
  }

  /**
   * Find the latest population snapshot in a directory.
   */
  static findLatestPopulation(generationsDir) {
    if (!fs.existsSync(generationsDir)) return null;

    const files = fs
      .readdirSync(generationsDir)
      .filter((f) => f.startsWith("gen_") && f.endsWith(".json"))
      .sort((a, b) => {
        const numA = parseInt(a.match(/gen_(\d+)/)?.[1] || 0);
        const numB = parseInt(b.match(/gen_(\d+)/)?.[1] || 0);
        return numB - numA;
      });

    if (files.length === 0) return null;
    return path.join(generationsDir, files[0]);
  }

  /**
   * Find the latest best genome.
   */
  static findLatestBest(bestDir) {
    if (!fs.existsSync(bestDir)) return null;

    const files = fs
      .readdirSync(bestDir)
      .filter((f) => f.startsWith("gen_") && f.endsWith(".json"))
      .sort((a, b) => {
        const numA = parseInt(a.match(/gen_(\d+)/)?.[1] || 0);
        const numB = parseInt(b.match(/gen_(\d+)/)?.[1] || 0);
        return numB - numA;
      });

    if (files.length === 0) return null;
    return path.join(bestDir, files[0]);
  }
}

module.exports = { Genome };
