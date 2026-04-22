# Bot service — what we run, where, and how to find us in your logs

This document is for the chainer game-server team to inspect what our bots do
end-to-end. If you cannot see our bots in the matchmaker queue, this should
tell you exactly where to look.

## Bot service code

Public repository: <https://github.com/matveyco/chainer-agent>

Key files:

- [`src/network/Matchmaker.js`](../src/network/Matchmaker.js) — every HTTP call we make against your matchmaker
- [`src/network/Connection.js`](../src/network/Connection.js) — Colyseus join logic
- [`src/runtime/RoomCoordinator.js`](../src/runtime/RoomCoordinator.js) — per-room queue → assignment → join → match-loop lifecycle
- [`src/runtime/SwarmSupervisor.js`](../src/runtime/SwarmSupervisor.js) — top-level loop that decides when to queue agents

We also have a single-bot dry-run probe that walks the whole protocol step-by-step:
[`src/network/BotServiceProbe.js`](../src/network/BotServiceProbe.js).

## Where our bots live and what URL they hit

Production host: `spark.local` (single Linux box, ARM64, NVIDIA GB10 GPU).

Environment from the running `.env` (secrets redacted):

```env
GAME_SERVER_URL=https://ai-test-arena.chainers.io
ROOM_NAME=TimeLimited
MAP_NAME=arena
WEAPON_TYPE=rocket
OAUTH_API_KEY=<redacted>
POPULATION_SIZE=12
NUM_ROOMS=2
```

We confirmed `https://ai-test-arena.chainers.io` and
`https://stageserver.chainers.io` are pointing at the same backend — joining
the queue on one is visible on the other (same `playersInQueue` count from
both `queue-status` endpoints, same `position` in `user-queue-position`). So
your dashboard, regardless of which DNS name it queries, sees the queue we
fill. Either name works for us.

## Exact HTTP requests we send

All matchmaker requests carry `Authorization: Bearer <OAUTH_API_KEY>` and
`Content-Type: application/json` (see `buildMatchmakerHeaders` in
`src/network/Matchmaker.js`).

### 1. Probe whether you have an active queue

```http
GET /matchmaker/queue-to-join HTTP/1.1
Authorization: Bearer <OAUTH_API_KEY>
```

We poll this every supervisor tick. **If it returns 404 we fall back to
`queue-status`** and, if your matchmaker has any `${ROOM_NAME}-*` key
(e.g. `TimeLimited-arena`), we treat that as an active queue and use our
configured `(roomName, mapName)` to seed it ourselves. That's the only way
the queue ever fills on a quiet test environment, so this fallback is what
makes our bots show up at all.

### 2. Queue 24 bots (12 per room × 2 rooms) in a burst

For each agent, in parallel:

```http
POST /matchmaker/join-queue HTTP/1.1
Authorization: Bearer <OAUTH_API_KEY>
Content-Type: application/json

{"userID":"agent_3_a1b2","roomName":"TimeLimited","mapName":"arena","forceCreateRoom":false}
```

`userID` format is `agent_<index>_<random4>`. Index `0..23`. The random suffix
changes every cycle so a bot that just left can re-queue without conflict.

### 3. Poll for room assignment

```http
GET /matchmaker/user-queue-position/agent_3_a1b2 HTTP/1.1
Authorization: Bearer <OAUTH_API_KEY>
```

Polled every `assignmentPollMs` (default 1500ms). When the response contains
`data.room`, we take `roomId` and `publicAddress` from there.

### 4. Connect via Colyseus WebSocket

```javascript
const client = new colyseus.Client(`https://${publicAddress}`);
const room = await client.joinById(roomId, {
  userID: "agent_3_a1b2",
  weaponType: "rocket",
  OAuthAPIKey: process.env.OAUTH_API_KEY,
});
```

We pin `colyseus.js@0.15.28` to match your server.

### 5. Send `room:player:loaded` immediately after join

```javascript
room.send("room:player:loaded", {
  profile: {
    userName: "agent_3",
    wallet: "0x0",
    models: [],
    textures: [],
  },
});
```

### 6. In-match traffic

- `room:player:input` — protobuf `InputMessage` ring buffer, 16 slots, sent
  every decision tick (about 20Hz)
- `room:player:shoot` — protobuf `ShootMessage`, throttled to weapon cooldown
- `room:player:ability:use` — `{ ability: "rampage" | "jump" | "minePlanting" }`
- `room:player:respawn` — sent ~3s after we receive our own `room:player:die`
- `room:rtt` — keep-alive, every 3s

We listen to `room:state:update`, `room:player:joined`, `room:player:die`,
`room:player:hit`, `room:player:respawn`, `room:player:left`, `room:time`,
`room:dispose`.

### 7. Leave on match end

When we receive `room:dispose` (or the room reports `time.left <= 0`, or our
local 15-min safety net fires) we:

```http
DELETE /matchmaker/leave-queue/agent_3_a1b2 HTTP/1.1
Authorization: Bearer <OAUTH_API_KEY>
```

…for each bot, then dispose of the Colyseus client.

## Identifying our traffic in your logs

- Every userID matches the regex `agent_\d+_[a-z0-9]{4}` (24 distinct base IDs
  per cycle, fresh suffix each cycle).
- All matchmaker calls carry the same `Authorization: Bearer
  ftrRJaLoTvAkYClzV6anwyya` token (the staging token from your config).
- All Colyseus connections from us include `OAuthAPIKey` in the `joinById`
  options block.
- Our user agent on `fetch` is whatever Node 20 emits by default
  (`node-fetch/undici`), which should distinguish us from real PlayCanvas
  clients.

## What to grep on your server side to find us

If you have nginx access logs:

```bash
grep -E "agent_[0-9]+_[a-z0-9]{4}" access.log | head -20
```

Or, if our `OAUTH_API_KEY` is unique to bots, just grep on that token
prefix.

## Operator endpoints we expose (so you can see the same view we have)

If you SSH onto `spark.local`:

```bash
curl http://localhost:5555/health     # PPO trainer status
curl http://localhost:3101/system     # Supervisor status (matches, queue, counters)
curl http://localhost:3101/rooms      # Per-room state in real time
curl http://localhost:3101/events     # Last 200 supervisor events
curl http://localhost:3000/api/system # Same data via the operator dashboard
```

The operator dashboard is at <http://spark.local:3000> when reachable.

## Recent client builds we've been running against

- `https://playcanv.as/b/800b2402?overlay=false` (older build)
- `https://playcanv.as/b/4b35ca00?overlay=false` (current — death effect +
  map-seam fixes)

Our bots don't load the PlayCanvas client at all — they speak directly to the
matchmaker + Colyseus room — so the client build only matters for what real
players see.

## Reproduce a single-bot lifecycle in 30 seconds

If you want a deterministic isolated trace, run our probe:

```bash
git clone https://github.com/matveyco/chainer-agent
cd chainer-agent
npm install
GAME_SERVER_URL=https://ai-test-arena.chainers.io \
OAUTH_API_KEY=<paste here> \
node -e "require('./src/network/BotServiceProbe').runBotServiceProbe({
  endpoint: process.env.GAME_SERVER_URL,
  authKey: process.env.OAUTH_API_KEY,
}).then(r => console.log(JSON.stringify(r, null, 2)))"
```

That walks `queue-to-join` → `queue-status` → `join-queue` → `user-queue-position`
→ `joinById` → `room:player:loaded` → `room:rtt` → `leave-queue` and prints
each stage's HTTP status + payload, so you can match it line-by-line against
your matchmaker logs.

If the probe succeeds end-to-end and you still don't see the bot in your
backend dashboard, the issue is on the dashboard side, not on ours.
