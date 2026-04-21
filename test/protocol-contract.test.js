const test = require("node:test");
const assert = require("node:assert/strict");

const types = require("../protobuf/types.json");

test("protobuf contract exposes Input, InputMessage, and ShootMessage shapes from the developer doc", () => {
  const input = types.nested.Input.fields;
  const inputMessage = types.nested.InputMessage.fields;
  const shoot = types.nested.ShootMessage.fields;

  assert.equal(input.inputMove.type, "float");
  assert.equal(input.inputMove.rule, "repeated");
  assert.equal(input.target.type, "float");
  assert.equal(input.animation.type, "int32");
  assert.equal(input.speed.type, "uint32");

  assert.equal(inputMessage.userID.type, "string");
  assert.equal(inputMessage.inputs.type, "Input");
  assert.equal(inputMessage.inputs.rule, "repeated");

  assert.equal(shoot.origin.type, "float");
  assert.equal(shoot.origin.rule, "repeated");
  assert.equal(shoot.target.type, "float");
  assert.equal(shoot.weaponType.type, "string");
  assert.equal(shoot.time.type, "uint64");
  assert.equal(shoot.shotID.type, "string");
});
