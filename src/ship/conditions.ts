/** Two-stage response keeps both values and their rates continuous at preset changes. */
export class Conditions {
  private intermediate = [0, 0, 0, 0, 0, 7];
  private values = [0, 0, 0, 0, 0, 7];
  phase = 0;
  advance(target: { wind: number; direction: number; gust: number; heave: number; roll: number; period: number }, dt: number) {
    const angle = target.direction * Math.PI / 180;
    const targets = [Math.cos(angle)*target.wind, Math.sin(angle)*target.wind, target.gust, target.heave, target.roll, target.period];
    const blend = 1 - Math.exp(-dt * 2);
    for (let i=0;i<targets.length;i++) {
      this.intermediate[i] += (targets[i]-this.intermediate[i])*blend;
      this.values[i] += (this.intermediate[i]-this.values[i])*blend;
    }
    this.phase += dt * 2 * Math.PI / this.values[5];
    const wind = Math.hypot(this.values[0],this.values[1]);
    return { wind, direction: Math.atan2(this.values[1],this.values[0])*180/Math.PI, gust:this.values[2], heave:this.values[3], roll:this.values[4], period:this.values[5] };
  }
}
