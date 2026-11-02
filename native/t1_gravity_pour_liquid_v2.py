"""Uncalibrated reduced-order hydrostatic / Torricelli / ballistic liquid.

Not CFD/SPH; fixed rigid-body mass is deliberately unchanged. The hydrostatic
surface is normal to WORLD gravity, clipped by the actual round-bottom cavity.
No pose writes, no target-volume valve, no phase-based suppression of spilling.
"""
from dataclasses import dataclass
import json
import numpy as np
from t1_liquid_palette import LIQUID_RGBA

@dataclass
class PourPhysics:
    gravity_m_s2: float = 9.81
    discharge_coefficient: float = .62  # engineering assumption, not calibrated
    effective_outlet_width_m: float = .007
    parcel_interval_s: float = .004
    hydro_interval_s: float = .01
    axial_quadrature: int = 96
    residual_stop_m3: float = .05e-6
    target_delivered_m3: float | None = None  # control advice ONLY, never a valve

class GravityPourLedger:
    def __init__(self, c=None, physics=None):
        if c is None:
            from t1_bimanual_glass_preparation import BimanualGlassConfig
            c=BimanualGlassConfig()
        self.config=c;self.physics=physics or PourPhysics()
        self.tube_volume_m3=float(c.tube_initial_volume_m3);self.receiver_volume_m3=float(c.receiver_initial_volume_m3)
        self.spilled_m3=0.;self.delivered_m3=0.;self.events=[];self.parcels=[];self.last_time=None
        self.mixing_proxy_angular_travel_rad=0.;self._previous_axis=None;self._previous_mouth=None
        self._hydro_at=-np.inf;self._record_at=-np.inf;self.records=[];self.surface_z=0.;self.flow_rate=0.
        R=c.tube_outer_radius_m;ri=R-c.tube_wall_thickness_m;L=c.tube_length_m
        self.ri=ri;self.R=R;self.L=L
        dz=(L-(R-ri))/self.physics.axial_quadrature
        self.z=R-ri+(np.arange(self.physics.axial_quadrature)+.5)*dz;self.dz=dz
        self.r=np.where(self.z<R,np.sqrt(np.maximum(0,ri**2-(self.z-R)**2)),ri)

    def volume_below(self, level_local, rotation):
        """Integrate circular disk segments below a world-horizontal plane."""
        n=rotation[2];s=float(np.linalg.norm(n[:2]));v=level_local-n[2]*self.z
        if s<1e-8:area=np.pi*self.r**2*(v>=0)
        else:
            h=np.clip(v/s,-self.r,self.r);a=np.clip(h/self.r,-1,1)
            area=self.r**2*(np.arcsin(a)+np.pi/2)+h*np.sqrt(np.maximum(0,self.r**2-h*h))
        return float(np.sum(area)*self.dz)

    def surface(self, origin, rotation):
        lo=-self.L-self.ri;hi=self.L+self.ri
        for _ in range(28):
            mid=(lo+hi)/2
            if self.volume_below(mid,rotation)<self.tube_volume_m3:lo=mid
            else:hi=mid
        return float(origin[2]+(lo+hi)/2)

    def advance_pose(self, time, mouth, rotation, *, allow_mix=False):
        time=float(time);mouth=np.asarray(mouth,float);rotation=np.asarray(rotation,float).reshape(3,3)
        dt=0. if self.last_time is None else time-self.last_time
        if dt<0:raise ValueError('Clock reversed')
        axis=rotation[:,2];origin=mouth-axis*self.L
        if allow_mix and self._previous_axis is not None:self.mixing_proxy_angular_travel_rad+=float(np.arccos(np.clip(axis@self._previous_axis,-1,1)))
        mouth_velocity=np.zeros(3) if not dt or self._previous_mouth is None else (mouth-self._previous_mouth)/dt
        orientation_changed=self._previous_axis is not None and np.linalg.norm(axis-self._previous_axis)>.002
        self._previous_axis=axis.copy();self._previous_mouth=mouth.copy();self.last_time=time
        if time-self._hydro_at>=self.physics.hydro_interval_s-1e-9 or dt==0 or orientation_changed:
            self.surface_z=self.surface(origin,rotation);self._hydro_at=time
        transverse=np.array([0.,0.,-1.])+axis*axis[2];tn=np.linalg.norm(transverse)
        lip=mouth+(transverse/tn*self.ri if tn>1e-9 else np.zeros(3))
        head=max(0.,self.surface_z-lip[2]);g=self.physics.gravity_m_s2
        # Broad-crested effective opening: wetted depth grows continuously from zero.
        area=self.physics.effective_outlet_width_m*min(head,2*self.ri)
        self.flow_rate=float(self.physics.discharge_coefficient*area*np.sqrt(2*g*head)) if self.tube_volume_m3>1e-14 else 0.
        emitted=float(min(self.tube_volume_m3,self.flow_rate*dt));self.tube_volume_m3-=emitted
        if emitted>0:
            speed=np.sqrt(2*g*head)*self.physics.discharge_coefficient
            # Axis is the outward outlet direction; falling velocity is integrated below.
            velocity=mouth_velocity+speed*axis
            self.parcels.append(dict(position=lip.copy(),velocity=velocity,volume=emitted,born=time))
        self._advance_parcels(dt, time)
        if time-self._record_at>=.02-1e-9:
            self.records.append(self.state(time,origin,rotation));self._record_at=time
        tilt=float(np.degrees(np.arccos(np.clip(axis[2],-1,1))))
        return dict(time_s=time,actual_tilt_deg=tilt,mouth_world_m=mouth.tolist(),over_receiver=bool(np.linalg.norm(lip[:2]-self.config.receiver_xy_m)<self.config.receiver_inner_radius_m),flow_m3=emitted,flow_rate_m3_s=self.flow_rate,free_surface_world_z_m=self.surface_z,airborne_m3=self.airborne_m3,stop_requested=self.stop_requested())

    def _advance_parcels(self,dt,time):
        c=self.config;g=self.physics.gravity_m_s2;alive=[]
        rim=c.receiver_bottom_z_m+getattr(c,'beaker_height_m',.056)-getattr(c,'beaker_bottom_assumption_m',.002)
        for p in self.parcels:
            # Newly emitted parcel starts at the interval endpoint.
            if p['born']==time:alive.append(p);continue
            old=p['position'].copy();v=p['velocity'].copy();new=old+v*dt;new[2]-=.5*g*dt*dt;p['velocity'][2]-=g*dt
            if old[2]>=rim and new[2]<rim:
                u=(old[2]-rim)/(old[2]-new[2]);cross=old+u*(new-old)
                inside=np.linalg.norm(cross[:2]-c.receiver_xy_m)<c.receiver_inner_radius_m
                if inside:
                    capacity=np.pi*c.receiver_inner_radius_m**2*(rim-c.receiver_bottom_z_m)
                    accepted=float(min(p['volume'],max(0.,capacity-self.receiver_volume_m3)))
                    self.receiver_volume_m3+=accepted;self.delivered_m3+=accepted;self.spilled_m3+=p['volume']-accepted
                    continue
            if new[2]<c.receiver_bottom_z_m-.003:self.spilled_m3+=p['volume'];continue
            p['position']=new;alive.append(p)
        self.parcels=alive

    @property
    def airborne_m3(self):return float(sum(p['volume'] for p in self.parcels))
    def stop_requested(self):
        target=self.physics.target_delivered_m3
        return bool(self.tube_volume_m3<=self.physics.residual_stop_m3 if target is None else self.delivered_m3+self.airborne_m3>=target)
    def state(self,time,origin,rotation):
        return dict(time_s=float(time),tube_volume_m3=self.tube_volume_m3,receiver_volume_m3=self.receiver_volume_m3,spilled_m3=self.spilled_m3,delivered_m3=self.delivered_m3,airborne_m3=self.airborne_m3,free_surface_plane_world=[0.,0.,1.,-self.surface_z],tube_origin_world_m=np.asarray(origin).tolist(),tube_rotation_world=np.asarray(rotation).tolist(),flow_rate_m3_s=self.flow_rate,parcels=[dict(position_m=p['position'].tolist(),velocity_m_s=p['velocity'].tolist(),volume_m3=p['volume']) for p in self.parcels])
    def export_record(self,path):
        from pathlib import Path
        Path(path).write_text(json.dumps(dict(model='uncalibrated hydrostatic/Torricelli/ballistic v2',liquid_rgba=LIQUID_RGBA,states=self.records,audit=self.audit()),separators=(',',':')))
    def observe(self,m,d,handles,*,allow_pour=False,allow_mix=False):
        # allow_pour deliberately is not a physical valve: an accidentally tilted open tube spills.
        b=m.body(handles['tube_body']).id
        result=self.advance_pose(d.time,d.site_xpos[m.site(handles['tube_mouth_site']).id],d.xmat[b].reshape(3,3),allow_mix=allow_mix)
        self.update_visual(m,d,handles);return result
    def update_visual(self,m,d,h,*args):
        # Hide rigidly rotating old tube column and decorative old stream.
        for name in [h['tube_visual']]+[self.config.prefix+'tube_bottom_liquid_'+str(i) for i in range(12)]+[self.config.prefix+'pour_visual_'+str(i) for i in range(12)]:
            try:m.geom_rgba[m.geom(name).id,3]=0
            except KeyError:pass
        c=self.config;gid=m.geom(h['receiver_visual']).id;height=max(.00001,self.receiver_volume_m3/(np.pi*c.receiver_inner_radius_m**2))
        m.geom_pos[gid,2]=c.receiver_bottom_z_m+height/2;m.geom_size[gid,1]=height/2;m.geom_rgba[gid]=[*LIQUID_RGBA[:3],LIQUID_RGBA[3] if self.receiver_volume_m3>0 else 0]
    def audit(self):
        c=self.config;error=float(c.tube_initial_volume_m3+c.receiver_initial_volume_m3-self.tube_volume_m3-self.receiver_volume_m3-self.spilled_m3-self.airborne_m3)
        return dict(model='uncalibrated reduced-order hydrostatic/Torricelli/ballistic; NOT CFD/SPH',tube_remaining_m3=float(self.tube_volume_m3),receiver_volume_m3=float(self.receiver_volume_m3),delivered_m3=float(self.delivered_m3),spilled_m3=float(self.spilled_m3),airborne_m3=self.airborne_m3,balance_error_m3=error,receiver_surface_z_m=float(c.receiver_bottom_z_m+self.receiver_volume_m3/(np.pi*c.receiver_inner_radius_m**2)),mixing_proxy_angular_travel_rad=float(self.mixing_proxy_angular_travel_rad),material_homogeneity_verified=False,fixed_mass_surrogate_kg=float(c.tube_mass_kg))
    def source_geometry(self):
        from t1_aspiration_stage import SourceGeometry
        c=self.config;return SourceGeometry(tuple(float(x) for x in c.receiver_xy_m),float(c.receiver_inner_radius_m),float(c.receiver_bottom_z_m),float(self.receiver_volume_m3),float(self.audit()['receiver_surface_z_m']))
