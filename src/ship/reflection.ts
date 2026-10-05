import { Matrix4 } from 'three';

/** Mirror the off-axis lens shift along with ReflectorNode's horizontal view axis. */
export function mirrorWaterProjection(source: Matrix4, target: Matrix4): Matrix4 {
  target.copy(source);
  target.elements[8]=-source.elements[8];
  return target;
}
