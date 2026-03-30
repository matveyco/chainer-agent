/**
 * Spatial hash grid for fast nearest-enemy queries.
 * Adapted from server's SpatialGrid — uses local math utils.
 */

const math = require("../utils/math");

class SpatialHashFast {
  constructor(bounds = { min: -90, max: 90 }, dimensions = [10, 10]) {
    const [x, y] = dimensions;
    this._cells = [...Array(x)].map(() => [...Array(y)].map(() => null));
    this._dimensions = dimensions;
    this._bounds = bounds;
    this._queryIds = 0;
    this._clientMap = new Map();
  }

  _getCellIndex(position) {
    const x = math.sat((position.x - this._bounds.min) / (this._bounds.max - this._bounds.min));
    const z = math.sat((position.z - this._bounds.min) / (this._bounds.max - this._bounds.min));
    const xIndex = Math.floor(x * (this._dimensions[0] - 1));
    const zIndex = Math.floor(z * (this._dimensions[1] - 1));
    return [xIndex, zIndex];
  }

  addClient(id, position = { x: 0, y: 0, z: 0 }, size = 1) {
    if (this._clientMap.has(id)) return;

    const client = {
      id,
      position,
      size,
      _cells: { min: null, max: null, nodes: null },
      _queryId: -1,
    };

    this._clientMap.set(id, client);
    this._insert(client);
    return client;
  }

  removeClient(id) {
    const client = this._clientMap.get(id);
    if (!client) return false;
    this._remove(client);
    this._clientMap.delete(id);
    return true;
  }

  updateClient(id, newPosition) {
    let client = this._clientMap.get(id);
    if (!client) {
      client = this.addClient(id, newPosition);
      return;
    }

    client.position = newPosition;
    const [x, z] = [newPosition.x, newPosition.z];
    const halfSize = client.size / 2;

    const i1 = this._getCellIndex({ x: x - halfSize, z: z - halfSize });
    const i2 = this._getCellIndex({ x: x + halfSize, z: z + halfSize });

    if (
      client._cells.min &&
      client._cells.min[0] === i1[0] &&
      client._cells.min[1] === i1[1] &&
      client._cells.max[0] === i2[0] &&
      client._cells.max[1] === i2[1]
    ) {
      return;
    }

    this._remove(client);
    this._insert(client);
  }

  findNear(position, range = 15, excludeID = null) {
    const [x, z] = [position.x, position.z];
    const i1 = this._getCellIndex({ x: x - range, z: z - range });
    const i2 = this._getCellIndex({ x: x + range, z: z + range });

    const clients = [];
    const queryId = this._queryIds++;

    for (let x = i1[0], xn = i2[0]; x <= xn; ++x) {
      for (let y = i1[1], yn = i2[1]; y <= yn; ++y) {
        let head = this._cells[x][y];
        while (head) {
          const v = head.client;
          head = head.next;
          if (v._queryId !== queryId && v.id !== excludeID) {
            v._queryId = queryId;
            const distance = math.getDistance(v.position, position);
            if (distance <= range) {
              clients.push({ id: v.id, position: v.position, distance });
            }
          }
        }
      }
    }

    clients.sort((a, b) => a.distance - b.distance);
    return clients;
  }

  _insert(client) {
    const [x, z] = [client.position.x, client.position.z];
    const halfSize = client.size / 2;
    const i1 = this._getCellIndex({ x: x - halfSize, z: z - halfSize });
    const i2 = this._getCellIndex({ x: x + halfSize, z: z + halfSize });
    const nodes = [];

    for (let x = i1[0], xn = i2[0]; x <= xn; ++x) {
      nodes.push([]);
      for (let y = i1[1], yn = i2[1]; y <= yn; ++y) {
        const xi = x - i1[0];
        const head = { next: null, prev: null, client };
        nodes[xi].push(head);
        head.next = this._cells[x][y];
        if (this._cells[x][y]) {
          this._cells[x][y].prev = head;
        }
        this._cells[x][y] = head;
      }
    }

    client._cells.min = i1;
    client._cells.max = i2;
    client._cells.nodes = nodes;
  }

  _remove(client) {
    if (!client._cells.min || !client._cells.max || !client._cells.nodes) return;
    const i1 = client._cells.min;
    const i2 = client._cells.max;

    for (let x = i1[0], xn = i2[0]; x <= xn; ++x) {
      for (let y = i1[1], yn = i2[1]; y <= yn; ++y) {
        const xi = x - i1[0];
        const yi = y - i1[1];
        const node = client._cells.nodes[xi][yi];
        if (node.next) node.next.prev = node.prev;
        if (node.prev) node.prev.next = node.next;
        if (!node.prev) this._cells[x][y] = node.next;
      }
    }

    client._cells.min = null;
    client._cells.max = null;
    client._cells.nodes = null;
  }

  getClient(id) {
    return this._clientMap.get(id);
  }

  clear() {
    for (let x = 0; x < this._dimensions[0]; x++) {
      for (let y = 0; y < this._dimensions[1]; y++) {
        this._cells[x][y] = null;
      }
    }
    this._clientMap.clear();
  }
}

module.exports = { SpatialGrid: SpatialHashFast };
