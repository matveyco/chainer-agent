/**
 * Deserializes binary room:state:update messages from the game server.
 * Direct copy from server's GameStateDeserializer — no dependencies.
 */

class GameStateDeserializer {
  deserializeSnapshot(buffer) {
    const view = new DataView(buffer);
    let offset = 0;

    // Read game state ID
    const idLength = view.getUint8(offset);
    offset += 1;
    const decoder = new TextDecoder();
    const id = decoder.decode(new Uint8Array(buffer, offset, idLength));
    offset += idLength;

    // Read timestamp
    const time = Number(view.getBigInt64(offset, true));
    offset += 8;

    // Read players
    const playerCount = view.getUint32(offset, true);
    offset += 4;
    const players = [];

    for (let i = 0; i < playerCount; i++) {
      const playerIdLength = view.getUint8(offset);
      offset += 1;
      const playerId = decoder.decode(new Uint8Array(buffer, offset, playerIdLength));
      offset += playerIdLength;

      const x = view.getFloat32(offset, true);
      offset += 4;
      const y = view.getFloat32(offset, true);
      offset += 4;
      const z = view.getFloat32(offset, true);
      offset += 4;
      const targetX = view.getFloat32(offset, true);
      offset += 4;
      const targetY = view.getFloat32(offset, true);
      offset += 4;
      const targetZ = view.getFloat32(offset, true);
      offset += 4;

      const animation = view.getInt8(offset);
      offset += 1;

      players.push({ id: playerId, x, y, z, targetX, targetY, targetZ, animation });
    }

    // Read battle crystals
    const crystalCount = view.getUint32(offset, true);
    offset += 4;
    const battleCrystals = [];

    for (let i = 0; i < crystalCount; i++) {
      const crystalIdLength = view.getUint8(offset);
      offset += 1;
      const crystalId = decoder.decode(new Uint8Array(buffer, offset, crystalIdLength));
      offset += crystalIdLength;

      const x = view.getFloat32(offset, true);
      offset += 4;
      const y = view.getFloat32(offset, true);
      offset += 4;
      const z = view.getFloat32(offset, true);
      offset += 4;

      battleCrystals.push({ id: crystalId, x, y, z });
    }

    return { id, time, state: { players, battleCrystals } };
  }
}

module.exports = { GameStateDeserializer };
