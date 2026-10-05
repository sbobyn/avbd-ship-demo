import { Matrix4 } from 'three/webgpu';

// Camera-heading compass: preserve direction and speed without pitch foreshortening.
// SVG y points down; world z is up in the ship demo.
export function windViewBasis(view:Matrix4): [number,number,number,number] {
  const m=view.elements;
  const length=Math.hypot(m[0],m[4]);
  const x=length>1e-8?m[0]/length:1, y=length>1e-8?m[4]/length:0;
  return [x,y,y,-x];
}
export function projectWind(basis:readonly number[],x:number,y:number): [number,number] {
  return [basis[0]*x+basis[2]*y,basis[1]*x+basis[3]*y];
}
export function unprojectWind(basis:readonly number[],x:number,y:number): [number,number] {
  const determinant=basis[0]*basis[3]-basis[1]*basis[2];
  // OrbitControls stays above the horizon. Guard only the exact edge-on case.
  if(Math.abs(determinant)<0.0001) return [basis[0]*x,basis[2]*x];
  return [(basis[3]*x-basis[2]*y)/determinant,(-basis[1]*x+basis[0]*y)/determinant];
}
