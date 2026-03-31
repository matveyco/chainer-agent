const fs = require("fs");
const path = require("path");

class SingleInstanceLock {
  constructor(lockFile) {
    this.lockFile = path.resolve(lockFile);
    this.acquired = false;
  }

  acquire(metadata = {}) {
    fs.mkdirSync(path.dirname(this.lockFile), { recursive: true });
    const existing = this._readLock();

    if (existing?.pid && this._pidExists(existing.pid)) {
      const err = new Error(`Another swarm instance is already running with pid ${existing.pid}`);
      err.code = "INSTANCE_LOCKED";
      err.lock = existing;
      throw err;
    }

    if (existing) {
      try {
        fs.unlinkSync(this.lockFile);
      } catch {}
    }

    const payload = {
      pid: process.pid,
      startedAt: new Date().toISOString(),
      ...metadata,
    };
    fs.writeFileSync(this.lockFile, JSON.stringify(payload, null, 2));
    this.acquired = true;
    return payload;
  }

  release() {
    if (!this.acquired) return;
    try {
      const existing = this._readLock();
      if (existing?.pid === process.pid) {
        fs.unlinkSync(this.lockFile);
      }
    } catch {}
    this.acquired = false;
  }

  _readLock() {
    try {
      return JSON.parse(fs.readFileSync(this.lockFile, "utf-8"));
    } catch {
      return null;
    }
  }

  _pidExists(pid) {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  }
}

module.exports = { SingleInstanceLock };
