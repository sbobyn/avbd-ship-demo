// Shared teardown for standalone WebGPU storage buffers (STE-1819). Three.js only frees a bare
// StorageBufferAttribute's GPUBuffer when a BufferGeometry that owns it is disposed (see
// node_modules/three/src/renderers/common/Geometries.js's onDispose, which walks a render object's geometry
// attributes into `this.attributes.delete(...)`). A compute-only buffer never belongs to such a geometry, so every
// lab that runs its own compute pass has to reach into the renderer's attribute store directly to release it.
//
// Checked three@0.185.1 (Renderer.js, Attributes.js, Geometries.js, Bindings.js, BufferAttribute.dispose) for a
// public way to do this instead: there isn't one. `renderer.dispose()` tears down the whole renderer, and
// `BufferAttribute.dispose()` only dispatches a 'dispose' event that nothing observes unless the attribute sits in
// a geometry the renderer has already rendered. The only thing that actually frees a standalone attribute's GPU
// buffer is `Attributes.delete()`, reached through the renderer's private `_attributes` field. So this helper still
// uses `_attributes`, but centralizes the reach-in behind one warning instead of five silent optional chains.
import * as THREE from 'three/webgpu';

/** The slice of three's private `Attributes` (renderers/common/Attributes.js) this helper needs. */
interface AttributeStore {
  delete(attribute: THREE.BufferAttribute): unknown;
}

let warnedMissingStore = false;

/** Frees standalone storage buffers (attributes not attached to any rendered geometry) through the renderer's
 *  internal attribute store. `renderer` may be null/undefined: that's a no-op, for teardown that runs after the
 *  renderer itself was already disposed. If three.js renames or removes the internal store or its `delete` method,
 *  this warns once (naming the installed three.js version) instead of leaking every buffer silently. */
export function releaseStorage(renderer: THREE.WebGPURenderer | null | undefined, attributes: readonly THREE.BufferAttribute[]): void {
  if (!renderer) return;
  const store = (renderer as unknown as { _attributes?: AttributeStore })._attributes;
  if (!store || typeof store.delete !== 'function') {
    if (!warnedMissingStore) {
      warnedMissingStore = true;
      console.warn(`releaseStorage: three.js r${THREE.REVISION}'s WebGPURenderer has no usable _attributes.delete(); storage buffers will leak until src/demos/gpu-resources.ts is updated for this three.js version.`);
    }
    return;
  }
  for (const attribute of attributes) store.delete(attribute);
}
