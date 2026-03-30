/**
 * Math utilities for 3D vector operations.
 * Adapted from the game server's math.js
 */

function rand(min = 0, max = 1) {
  return Math.random() * (max - min) + min;
}

function clamp(x, a, b) {
  return Math.min(Math.max(x, a), b);
}

function sat(x) {
  return Math.min(Math.max(x, 0.0), 1.0);
}

function getDistance(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

function getDirection(from, to) {
  const x = to.x - from.x;
  const y = to.y - from.y;
  const z = to.z - from.z;
  const l = Math.sqrt(x * x + y * y + z * z);
  if (l === 0 || isNaN(l)) return { x: 0, y: 0, z: 0 };
  return { x: x / l, y: y / l, z: z / l };
}

function getDirectionArray(from, to) {
  const x = to[0] - from[0];
  const y = to[1] - from[1];
  const z = to[2] - from[2];
  const l = Math.sqrt(x * x + y * y + z * z);
  if (l === 0 || isNaN(l)) return [0, 0, 0];
  return [x / l, y / l, z / l];
}

function normalizeArray(v) {
  const l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (l === 0) return [0, 0, 0];
  return [v[0] / l, v[1] / l, v[2] / l];
}

function getArrayLength(vec) {
  return Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

function normalize01(value, min, max) {
  return clamp((value - min) / (max - min), 0, 1);
}

module.exports = {
  rand,
  clamp,
  sat,
  getDistance,
  getDirection,
  getDirectionArray,
  normalizeArray,
  getArrayLength,
  normalize01,
};
