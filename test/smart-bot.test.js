const test = require("node:test");
const assert = require("node:assert/strict");

const { SmartBot } = require("../src/bot/SmartBot");

function makeBot() {
  return new SmartBot(
    "agent_7_session",
    null,
    {
      server: { mapName: "arena", weaponType: "rocket" },
      training: { defaultModelAlias: "latest" },
      bot: { arenaSafeSize: 55 },
      reward: {},
    },
    "agent_7",
    {}
  );
}

test("smart bot normalizes millisecond weapon cooldowns into seconds", () => {
  const bot = makeBot();
  bot.data = { weaponCoolDown: 1250 };
  assert.equal(bot._getWeaponCooldownSeconds(), 1.25);

  bot.data = { weaponCoolDown: 1.4 };
  assert.equal(bot._getWeaponCooldownSeconds(), 1.4);
});

test("smart bot builds shoot targets from the aimed point instead of random enemy-only fallback", () => {
  const bot = makeBot();
  bot.closestEnemy = { position: { x: 20, y: 0, z: 20 } };
  const target = bot._buildShootTarget({ x: 10, y: 0, z: -5 });

  assert.equal(target.length, 3);
  assert.equal(Math.abs(target[0] - 10) < 1, true);
  assert.equal(Math.abs(target[2] + 5) < 1, true);
});
