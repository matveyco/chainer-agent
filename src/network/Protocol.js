/**
 * Protobuf encoding for game messages.
 * Handles InputMessage and ShootMessage binary encoding.
 */

const protobuf = require("protobufjs");
const Long = require("long");
const types = require("../../protobuf/types.json");

const root = protobuf.Root.fromJSON(types);
const InputMessage = root.lookupType("InputMessage");
const ShootMessage = root.lookupType("ShootMessage");

// Shared buffers to avoid allocations (same pattern as loadtest player.js)
const sharedBuffers = {};
const sharedViews = {};

function toArrayBuffer(buffer, name) {
  if (!sharedBuffers[name]) {
    sharedBuffers[name] = new ArrayBuffer(buffer.length);
    sharedViews[name] = new Uint8Array(sharedBuffers[name]);
  }

  // Resize if needed
  if (buffer.length > sharedBuffers[name].byteLength) {
    sharedBuffers[name] = new ArrayBuffer(buffer.length);
    sharedViews[name] = new Uint8Array(sharedBuffers[name]);
  }

  sharedViews[name].set(buffer);
  return sharedBuffers[name].slice(0, buffer.length);
}

/**
 * Generate a random alphanumeric ID
 */
function generateID(length = 8) {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

/**
 * Encode an InputMessage for room:player:input
 * @param {string} userID
 * @param {Array<{inputMove: Float32Array|number[], target: Float32Array|number[], animation: number, speed: number}>} inputs
 * @returns {ArrayBuffer}
 */
function encodeInput(userID, inputs) {
  const message = InputMessage.create({ userID, inputs });
  const buffer = InputMessage.encode(message).finish();
  return toArrayBuffer(buffer, "input");
}

/**
 * Encode a ShootMessage for room:player:shoot
 * @param {number[]} origin - [x, y, z]
 * @param {number[]} target - [x, y, z]
 * @param {string} weaponType
 * @returns {ArrayBuffer}
 */
function encodeShoot(origin, target, weaponType) {
  const shootData = {
    origin,
    target,
    weaponType,
    time: Long.fromValue(Date.now()),
    shotID: generateID(),
  };
  const buffer = ShootMessage.encode(shootData).finish();
  return toArrayBuffer(buffer, "shoot");
}

module.exports = { encodeInput, encodeShoot, generateID };
