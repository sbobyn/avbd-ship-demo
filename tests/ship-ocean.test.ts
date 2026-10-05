import assert from 'node:assert/strict';
import { test } from 'node:test';
import { CASCADES } from '../src/ship/ocean/model.ts';
import { oceanSurface } from '../src/ship/ocean/reference.ts';

test('ship spectral ocean FFT agrees with direct synthesis for all fields', () => {
  for (const base of CASCADES) {
    const cascade={...base,n:8};
    const fft=oceanSurface(cascade,13.7,'fft'), dft=oceanSurface(cascade,13.7,'dft');
    for(const field of ['height','slopeX','slopeZ','dispX','dispZ'] as const) {
      for(let i=0;i<fft[field].length;i++) {
        assert.ok(Number.isFinite(fft[field][i]));
        assert.ok(Math.abs(fft[field][i]-dft[field][i])<1e-9, `${base.name} ${field} mismatch`);
      }
    }
    assert.ok(fft.maxImaginary<1e-9);
  }
});

import { bandWeight } from '../src/ship/ocean/bands.ts';
test('spectral bands preserve energy across both cascade transitions', () => {
  for(let i=0;i<1000;i++) {
    const k=Math.exp(-7+i*0.015);
    const sum=CASCADES.reduce((value,c)=>value+bandWeight(k,c.minWavenumber,c.maxWavenumber),0);
    assert.ok(Math.abs(sum-1)<1e-12, `spectral gap or overlap at ${k}`);
  }
});

test('render-resolution cascades retain finite nonzero wave energy', () => {
  for (const base of CASCADES) {
    const surface=oceanSurface({...base,n:128},5,'fft');
    let energy=0;
    for(const height of surface.height) { assert.ok(Number.isFinite(height)); energy+=height*height; }
    assert.ok(energy>1e-8, `${base.name} lost its spectrum`);
  }
});

import { BoxGeometry, BufferGeometry, Float32BufferAttribute } from 'three';
import { contactTriangles, waterlineEnvelope } from '../src/ship/ocean/hull.ts';
test('wake sources use triangle intersections with the hull waterline', () => {
  const hull=new BoxGeometry(18,5,4).translate(2,0.4,0);
  const aboveDeck=new BoxGeometry(50,20,1).translate(0,0,8);
  const line=waterlineEnvelope([hull,aboveDeck]);
  assert.ok(Math.abs(line.stern+7)<1e-6);
  assert.ok(Math.abs(line.bow-11)<1e-6);
  assert.ok(Math.abs(line.beam-2.5)<1e-6);
  assert.ok(Math.abs(line.centerY-0.4)<1e-6);
  assert.equal(line.bowContacts.length,8);
  for(const [i,point] of line.bowContacts.entries()) {
    assert.ok(Math.abs(point.y-(i%2?-2.1:2.9))<1e-6,'stations intersect the real flat sides, not an ellipse');
    assert.ok(Math.abs(Math.hypot(point.nx,point.ny)-1)<1e-6);
    assert.ok(point.ny*(i%2?-1:1)>0,'normals point out of the hull');
    assert.ok(point.x<line.bow && point.x>line.stern);
  }
  assert.throws(()=>waterlineEnvelope([aboveDeck]), /does not intersect/);
});


test('hull contact inputs preserve every indexed triangle in model coordinates',()=>{
  const hull=new BoxGeometry(18,5,4).translate(2,0.4,0);
  const triangles=contactTriangles([hull]), positions=hull.getAttribute('position'), indices=hull.getIndex()!;
  assert.equal(triangles.length,indices.count*4);
  for(let i=0;i<indices.count;i++) {
    const index=indices.getX(i);
    assert.deepEqual(Array.from(triangles.subarray(i*4,i*4+4)),[positions.getX(index),positions.getY(index),positions.getZ(index),1]);
  }
  hull.dispose();
});


test('contact refinement preserves hull faces and bounds waterline edge lengths',()=>{
  const hull=new BoxGeometry(18,5,4).translate(2,0.4,0), triangles=contactTriangles([hull],0.5);
  assert.ok(triangles.length>hull.getIndex()!.count*4);
  assert.throws(()=>contactTriangles([hull],0),/positive/);
  for(let i=0;i<triangles.length;i+=12) {
    const points=Array.from({length:3},(_,v)=>Array.from(triangles.subarray(i+v*4,i+v*4+3)));
    for(const p of points) assert.ok(Math.min(Math.abs(p[0]+7),Math.abs(p[0]-11),Math.abs(p[1]+2.1),Math.abs(p[1]-2.9),Math.abs(p[2]+2),Math.abs(p[2]-2))<0.00001);
    if(Math.min(...points.map(p=>p[2]))<2.5 && Math.max(...points.map(p=>p[2]))>-2.5) {
      for(let e=0;e<3;e++) assert.ok(Math.hypot(...points[e].map((v,k)=>v-points[(e+1)%3][k]))<=0.500001);
    }
  }
  hull.dispose();
});


test('refined contact normals retain the model interpolation without facet resets',()=>{
  const geometry=new BufferGeometry();
  geometry.setAttribute('position',new Float32BufferAttribute([0,0,0,2,0,0,0,2,0],3));
  geometry.setAttribute('normal',new Float32BufferAttribute([0,0,1,0.6,0,0.8,0,0.6,0.8],3));
  const normals:number[]=[], triangles=contactTriangles([geometry],0.25,normals);
  assert.equal(normals.length,triangles.length);
  for(let i=0;i<triangles.length;i+=4) {
    const x=triangles[i], y=triangles[i+1];
    assert.ok(Math.abs(normals[i]-x*0.3)<1e-7);
    assert.ok(Math.abs(normals[i+1]-y*0.3)<1e-7);
    assert.ok(Math.abs(normals[i+2]-(1-(x+y)*0.1))<1e-7,'subdivision preserves the original barycentric normal field');
    assert.equal(normals[i+3],0);
  }
  geometry.deleteAttribute('normal'); const missing:number[]=[];
  contactTriangles([geometry],0.25,missing);
  assert.ok(missing.every(v=>v===0),'missing normals explicitly select the GPU face-normal fallback');
  geometry.dispose();
});
