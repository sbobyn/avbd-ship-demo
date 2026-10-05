/** Damped sailing controls; a prescribed hull controller, not a hydrodynamic solver. */
export interface PilotControls { rudder:number; power:number; autoTrim:boolean; trim:number }
export class Navigation {
  x=0; y=0; distance=0; heading=0; speed=0; turnRate=0; rudder=0;
  advance(wind:number, direction:number, pilot:PilotControls, dt:number): void {
    const response=1-Math.exp(-dt*1.5);
    this.rudder+=(Math.max(-1,Math.min(1,pilot.rudder))-this.rudder)*response;
    // Wind points downwind. Square sails lose drive when sailing into it.
    const relative=direction*Math.PI/180-this.heading;
    const windAlong=Math.cos(relative);
    const reach=Math.max(0,Math.min(1,(windAlong+0.85)/1.1));
    const drive=reach*reach*(3-2*reach)*(0.8+0.2*Math.abs(Math.sin(relative)));
    const trim=pilot.autoTrim?1:0.45+0.55*Math.cos(pilot.trim*Math.PI/180-relative)**2;
    const target=Math.min(20/1.94384,wind*0.86*drive*trim)*Math.max(0,Math.min(1,pilot.power));
    const previousSpeed=this.speed, previousHeading=this.heading;
    this.speed+=(target-this.speed)*(1-Math.exp(-dt/6));
    const turn=-this.rudder*Math.min(this.speed/1.5,1)*0.09;
    this.turnRate+=(turn-this.turnRate)*(1-Math.exp(-dt/2));
    this.heading+=this.turnRate*dt;
    const travel=(previousSpeed+this.speed)*0.5*dt, heading=(previousHeading+this.heading)*0.5;
    this.distance+=travel;
    this.x+=Math.cos(heading)*travel;
    this.y+=Math.sin(heading)*travel;
  }
}
