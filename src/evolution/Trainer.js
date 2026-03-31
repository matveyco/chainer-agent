/**
 * Compatibility wrapper around the runtime supervisor.
 */

const { SwarmSupervisor } = require("../runtime/SwarmSupervisor");

class Trainer {
  constructor(config, onUpdate = null) {
    this.supervisor = new SwarmSupervisor(config, onUpdate);
  }

  run() {
    return this.supervisor.run();
  }

  stop() {
    return this.supervisor.stop();
  }

  saveState() {
    return this.supervisor.saveState();
  }

  resumeFrom(snapshotPath) {
    return this.supervisor.resumeFrom(snapshotPath);
  }

  getAgentProfile(agentId) {
    return this.supervisor.getAgentProfile(agentId);
  }

  getAllProfiles() {
    return this.supervisor.getAllProfiles();
  }
}

module.exports = { Trainer };
