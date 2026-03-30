/**
 * Simple structured logger with generation-aware prefixes.
 */

class Logger {
  constructor() {
    this.generation = null;
  }

  setGeneration(gen) {
    this.generation = gen;
  }

  _prefix() {
    const ts = new Date().toISOString().slice(11, 19);
    const gen = this.generation !== null ? ` [Gen ${this.generation}]` : "";
    return `[${ts}]${gen}`;
  }

  info(...args) {
    console.log(this._prefix(), ...args);
  }

  warn(...args) {
    console.warn(this._prefix(), "WARN", ...args);
  }

  error(...args) {
    console.error(this._prefix(), "ERROR", ...args);
  }

  debug(...args) {
    if (process.env.DEBUG) {
      console.log(this._prefix(), "DEBUG", ...args);
    }
  }
}

module.exports = new Logger();
