import * as THREE from 'three/webgpu';
import { attribute, cross, float, Fn, normalLocal, normalize, storage, vec3, vec4, cameraViewMatrix, cameraProjectionMatrix, screenSize } from 'three/tsl';
import type { ShipPhysics } from './physics.ts';
import { BODY_FLOATS } from '../avbd3d/gpu/layout.ts';

/** Original textured meshes, deformed directly from the AVBD storage buffer. */
export class ShipGpuSkin {
  readonly attribute: THREE.StorageBufferAttribute;
  readonly buffer: GPUBuffer;
  private readonly materials: THREE.Material[] = [];

  constructor(renderer: THREE.WebGPURenderer, count: number) {
    this.attribute = new THREE.StorageBufferAttribute(new Float32Array(count * BODY_FLOATS), 4);
    const backend = renderer.backend as unknown as {
      createStorageAttribute(attribute: THREE.BufferAttribute): void;
      get(attribute: THREE.BufferAttribute): { buffer: GPUBuffer };
    };
    backend.createStorageAttribute(this.attribute);
    this.buffer = backend.get(this.attribute).buffer;
  }

  bind(physics: ShipPhysics, meshes: THREE.Mesh[], sources: THREE.MeshStandardMaterial[]): void {
    const bodies = storage(this.attribute, 'vec4', physics.reference.bodies.length * BODY_FLOATS / 4);
    type Vec3 = THREE.Node<'vec3'>;
    type Vec4 = THREE.Node<'vec4'>;
    const rotate = (q: Vec4, v: Vec3): Vec3 => {
      const t = cross(q.xyz, v).mul(2);
      return v.add(t.mul(q.w)).add(cross(q.xyz, t));
    };
    physics.skins.forEach((skin, meshIndex) => {
      const count = skin.weights.length, indices = new Float32Array(count * 4), weights = new Float32Array(count * 4);
      const positions = Array.from({ length: 4 }, () => new Float32Array(count * 3));
      const normals = Array.from({ length: 4 }, () => new Float32Array(count * 3));
      skin.weights.forEach((influences, vertex) => {
        influences.forEach((influence, j) => {
          indices[vertex * 4 + j] = influence.body;
          weights[vertex * 4 + j] = influence.weight;
          positions[j].set(influence.local.toArray(), vertex * 3);
          const normal = new THREE.Vector3().fromArray(skin.bindings!,skin.offsets![vertex]+j*8+5);
          normals[j].set(normal.toArray(), vertex * 3);
        });
      });
      // One interleaved vertex buffer stays within WebGPU's portable eight-buffer limit.
      const data = new Float32Array(count * 32);
      for (let i = 0; i < count; i++) {
        data.set(indices.subarray(i * 4, i * 4 + 4), i * 32);
        data.set(weights.subarray(i * 4, i * 4 + 4), i * 32 + 4);
        for (let j = 0; j < 4; j++) {
          data.set(positions[j].subarray(i * 3, i * 3 + 3), i * 32 + 8 + j * 6);
          data.set(normals[j].subarray(i * 3, i * 3 + 3), i * 32 + 11 + j * 6);
        }
      }
      const interleaved = new THREE.InterleavedBuffer(data, 32);
      skin.geometry.setAttribute('shipIndices', new THREE.InterleavedBufferAttribute(interleaved, 4, 0));
      skin.geometry.setAttribute('shipWeights', new THREE.InterleavedBufferAttribute(interleaved, 4, 4));
      let position: Vec3 = vec3(0), normal: Vec3 = vec3(0);
      const ids = attribute('shipIndices', 'vec4'), ws = attribute('shipWeights', 'vec4');
      const components = ['x', 'y', 'z', 'w'] as const;
      for (let j = 0; j < 4; j++) {
        skin.geometry.setAttribute(`shipLocal${j}`, new THREE.InterleavedBufferAttribute(interleaved, 3, 8 + j * 6));
        skin.geometry.setAttribute(`shipNormal${j}`, new THREE.InterleavedBufferAttribute(interleaved, 3, 11 + j * 6));
        const offset = ids[components[j]].mul(float(BODY_FLOATS / 4)).toUint();
        const q = bodies.element(offset.add(1));
        position = position.add(rotate(q, attribute(`shipLocal${j}`, 'vec3')).add(bodies.element(offset).xyz).mul(ws[components[j]]));
        normal = normal.add(rotate(q, attribute(`shipNormal${j}`, 'vec3')).mul(ws[components[j]]));
      }
      const material = new THREE.MeshStandardNodeMaterial().copy(sources[meshIndex]);
      material.positionNode = Fn(() => {
        // Deform the base normal before the original material applies its normal map.
        normalLocal.assign(normalize(normal));
        if(meshIndex===1) {
          // Round ropes can still disappear when their projected width falls
          // below one pixel. Slightly thicken all cordage and keep distant
          // cross-sections resolved, leaving hardware/cloth untouched.
          const radius=attribute('ropeRadius','float');
          const depth=cameraViewMatrix.mul(vec4(position,1)).z.abs();
          const minimum=depth.mul(3).div(screenSize.y.mul(cameraProjectionMatrix.mul(vec4(0,1,0,0)).y.abs()).max(1)).min(0.12);
          const extra=radius.greaterThan(0).select(minimum.sub(radius).max(radius.mul(0.55)),float(0));
          return position.add(normalize(normal).mul(extra));
        }
        return position;
      })();
      meshes[meshIndex].material = material;
      this.materials.push(material);
    });
  }

  dispose(): void {
    this.materials.forEach(material => material.dispose());
    this.buffer.destroy();
  }
}
