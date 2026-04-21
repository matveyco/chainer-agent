const { Client } = require("colyseus.js");
const { Matchmaker } = require("./Matchmaker");

function makeProfile(userID) {
  return {
    profile: {
      userName: `Probe_${String(userID || "").slice(-6)}`,
      wallet: "0x0",
      models: [],
      textures: [],
    },
  };
}

function cleanPublicAddress(value) {
  return String(value || "")
    .replace(/["']/g, "")
    .replace(/\s/g, "")
    .replace(/^https?:\/\//, "");
}

function createProbeResult(userID) {
  return {
    ok: false,
    userID,
    queue: null,
    room: null,
    failedStage: null,
    stages: {
      queueToJoin: null,
      queueStatus: null,
      joinQueue: null,
      assignment: null,
      joinById: null,
      loaded: null,
      rtt: null,
      leaveQueue: null,
    },
  };
}

function markStage(result, stage, payload) {
  result.stages[stage] = payload;
  if (payload?.ok === false && !result.failedStage) {
    result.failedStage = stage;
  }
}

function probeError(message, extra = {}) {
  return { ok: false, error: message, ...extra };
}

async function runBotServiceProbe(options = {}) {
  const endpoint = options.endpoint;
  const authKey = options.authKey || null;
  const weaponType = options.weaponType || "rocket";
  const userID = options.userID || `probe_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
  const queueTimeoutMs = Math.max(1000, Number(options.queueTimeoutMs || 15000));
  const assignmentTimeoutMs = Math.max(1000, Number(options.assignmentTimeoutMs || 30000));
  const rttTimeoutMs = Math.max(1000, Number(options.rttTimeoutMs || 5000));
  const forceCreateRoom = Boolean(options.forceCreateRoom);
  const ClientClass = options.ClientClass || Client;
  const matchmaker = new Matchmaker(endpoint, {
    authKey,
    pollMs: options.pollMs || 1000,
  });
  const result = createProbeResult(userID);

  let client = null;
  let room = null;
  let joinedQueue = false;

  try {
    const queueStatus = await matchmaker.getQueueStatus().catch((err) => ({
      ok: false,
      status: null,
      error: err.message,
    }));
    result.stages.queueStatus = queueStatus.ok === false
      ? probeError(queueStatus.error || "queue-status failed", { status: queueStatus.status })
      : {
        ok: true,
        status: queueStatus.status,
        authorized: queueStatus.authorized,
        data: queueStatus.data,
      };

    const queueInfo = await matchmaker.waitForActiveQueue({
      timeoutMs: queueTimeoutMs,
      pollMs: options.pollMs || 1000,
    });

    result.queue = queueInfo?.data || null;
    if (!queueInfo?.authorized) {
      markStage(result, "queueToJoin", probeError("queue-to-join unauthorized", { status: queueInfo?.status || null }));
      return result;
    }
    if (!queueInfo?.active || !queueInfo?.data?.roomName) {
      markStage(result, "queueToJoin", probeError("no active queue or open room exposed by queue-to-join", {
        status: queueInfo?.status || null,
      }));
      return result;
    }

    markStage(result, "queueToJoin", {
      ok: true,
      status: queueInfo.status,
      roomName: queueInfo.data.roomName,
      mapName: queueInfo.data.mapName || null,
    });

    const joined = await matchmaker.joinQueue(
      userID,
      queueInfo.data.roomName,
      queueInfo.data.mapName,
      forceCreateRoom
    );
    joinedQueue = true;
    markStage(result, "joinQueue", {
      ok: true,
      roomAssignedImmediately: Boolean(joined?.room),
      position: joined?.position ?? null,
    });

    const assignedRoom = joined?.room
      || await matchmaker.waitForAssignment(userID, {
        timeoutMs: assignmentTimeoutMs,
        pollMs: options.pollMs || 1000,
      });
    result.room = assignedRoom || null;
    if (!assignedRoom?.roomId || !assignedRoom?.publicAddress) {
      markStage(result, "assignment", probeError("assignment did not include roomId/publicAddress"));
      return result;
    }

    markStage(result, "assignment", {
      ok: true,
      roomId: assignedRoom.roomId,
      publicAddress: assignedRoom.publicAddress,
    });

    const host = cleanPublicAddress(assignedRoom.publicAddress);
    client = new ClientClass(`https://${host}`);
    room = await client.joinById(
      assignedRoom.roomId,
      matchmaker.buildRoomJoinOptions({ userID, weaponType })
    );

    markStage(result, "joinById", {
      ok: true,
      roomId: room.roomId,
      publicAddress: host,
    });

    room.send("room:player:loaded", makeProfile(userID));
    markStage(result, "loaded", { ok: true });

    const rttInfo = await new Promise((resolve, reject) => {
      let timer = null;
      const cleanup = () => {
        if (timer) clearTimeout(timer);
      };

      room.onMessage("room:rtt", (value) => {
        cleanup();
        resolve({ ok: true, value });
      });

      timer = setTimeout(() => {
        reject(new Error(`room:rtt timed out after ${rttTimeoutMs}ms`));
      }, rttTimeoutMs);

      room.send("room:rtt");
    });

    markStage(result, "rtt", rttInfo);
    result.ok = true;
    return result;
  } catch (err) {
    if (!result.failedStage) {
      if (!result.stages.queueToJoin) {
        markStage(result, "queueToJoin", probeError(err.message));
      } else if (!result.stages.joinQueue) {
        markStage(result, "joinQueue", probeError(err.message));
      } else if (!result.stages.assignment) {
        markStage(result, "assignment", probeError(err.message));
      } else if (!result.stages.joinById) {
        markStage(result, "joinById", probeError(err.message));
      } else if (!result.stages.loaded) {
        markStage(result, "loaded", probeError(err.message));
      } else {
        markStage(result, "rtt", probeError(err.message));
      }
    }
    return result;
  } finally {
    try {
      if (room) {
        await room.leave?.();
        room.removeAllListeners?.();
      }
    } catch {}

    try {
      if (joinedQueue) {
        await matchmaker.leaveQueue(userID);
        markStage(result, "leaveQueue", { ok: true });
      } else {
        markStage(result, "leaveQueue", { ok: true, skipped: true });
      }
    } catch (err) {
      markStage(result, "leaveQueue", probeError(err.message));
    }
  }
}

module.exports = {
  runBotServiceProbe,
};
