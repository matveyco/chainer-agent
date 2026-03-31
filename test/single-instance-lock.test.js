const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const os = require("os");
const path = require("path");

const { SingleInstanceLock } = require("../src/runtime/SingleInstanceLock");

test("single instance lock acquires and releases", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "chainer-lock-"));
  const lockPath = path.join(tempDir, "swarm.lock");
  const lock = new SingleInstanceLock(lockPath);

  const payload = lock.acquire({ runId: "test-run" });
  assert.equal(payload.pid, process.pid);
  assert.equal(fs.existsSync(lockPath), true);

  lock.release();
  assert.equal(fs.existsSync(lockPath), false);
});

test("single instance lock clears stale pid", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "chainer-lock-"));
  const lockPath = path.join(tempDir, "swarm.lock");
  fs.writeFileSync(lockPath, JSON.stringify({ pid: 99999999, runId: "stale" }));

  const lock = new SingleInstanceLock(lockPath);
  const payload = lock.acquire({ runId: "fresh" });
  assert.equal(payload.pid, process.pid);

  lock.release();
});
