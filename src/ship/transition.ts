/** Two-stage exponential response: retargeting preserves both value and velocity. */
export class SmoothTransition<K extends string> {
  readonly values: Record<K, number>;
  private intermediate: Record<K, number>;
  private target: Record<K, number>;
  private keys: K[];
  private rate: number;
  constructor(initial: Record<K, number>, rate = 2.5) {
    this.rate=rate;
    this.values={...initial}; this.intermediate={...initial}; this.target={...initial};
    this.keys=Object.keys(initial) as K[];
  }
  setTarget(target: Record<K, number>): void { this.target={...target}; }
  advance(dt: number): boolean {
    if(dt<=0) return false;
    const k=this.rate*dt, decay=Math.exp(-k);
    let changed=false;
    for(const key of this.keys) {
      const goal=this.target[key], old=this.values[key], mid=this.intermediate[key];
      const next=goal+(old-goal+k*(mid-goal))*decay;
      this.intermediate[key]=goal+(mid-goal)*decay;
      this.values[key]=next;
      if(Math.abs(next-goal)+Math.abs(this.intermediate[key]-goal)<1e-5*(1+Math.abs(goal))) {
        this.values[key]=goal; this.intermediate[key]=goal;
      }
      changed ||= old!==this.values[key];
    }
    return changed;
  }
}

export function nearestAngle(current: number, target: number): number {
  return current+Math.atan2(Math.sin(target-current),Math.cos(target-current));
}

/** Continue the dusk-to-night azimuth direction through the next sunrise. */
export function nextDayAngle(current: number, target: number): number {
  const turn=2*Math.PI;
  return current-((current-target)%turn+turn)%turn;
}
