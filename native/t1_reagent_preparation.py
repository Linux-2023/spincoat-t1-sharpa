"""Independent prep assets and conservative visible-liquid proxy, not mixing CFD.

add_reagent_preparation(spec, config) mutates only the supplied uncompiled spec.
No robot/camera/baseline files changed. Tube remains a free dynamic body, never welded.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,mujoco
from t1_spincoat_scene import ORIGINAL

@dataclass
class PreparationConfig:
 tube_xy_m:tuple=(.41,.16)
 tabletop_z_m:float=.8000005
 tube_support_height_m:float=.06
 tube_grasp_height_m:float=.05
 tube_initial_volume_m3:float=25e-6
 target_pour_volume_m3:float=20e-6
 tube_capacity_m3:float=50e-6
 tube_mass_kg:float=.037
 receiver_xy_m:tuple=(.4,.05)
 receiver_inner_radius_m:float=.0195
 receiver_bottom_z_m:float=.84
 receiver_initial_volume_m3:float=0.
 proxy_flow_rate_m3_s:float=5e-6
 proxy_min_pour_tilt_deg:float=70.
 pour_mouth_height_m:float=1.04
 prepour_mouth_height_m:float=1.10
 proxy_max_pour_height_above_bottom_m:float=.24
 prefix:str='prep/'

def add_reagent_preparation(spec,config=None):
 """Return handles dict. Caller compiles and copies existing state by joint name."""
 c=config or PreparationConfig();p=c.prefix
 spec.memory=max(spec.memory,128*1024*1024)
 if not 0<=c.tube_initial_volume_m3<=c.tube_capacity_m3:raise ValueError('Invalid tube volume')
 xml=ET.parse(ORIGINAL/'assets/mjcf/object/centrifuge_50ml.xml').getroot();xml.find('compiler').set('meshdir',str(ORIGINAL/'assets/meshes'));body=xml.find('worldbody/body');body.set('name','reagent_tube');body.find('freejoint').set('name','reagent_tube_free')
 for cap in list(body.findall('body')):body.remove(cap)
 inert=ET.SubElement(body,'inertial',pos='0 0 0.055',mass=str(c.tube_mass_kg),diaginertia=' '.join(str(v*c.tube_mass_kg/.037) for v in [.000041,.000041,.0000038]))
 child=mujoco.MjSpec.from_string(ET.tostring(xml,encoding='unicode'));tb=child.body('reagent_tube')
 for g in child.geoms:
  if g.contype or g.conaffinity:g.friction[:]=[1.0,.005,.0005];g.condim=4
  elif 'visual' in g.name:g.material='';g.rgba[:]=[.86,.94,1,.28]
 tb.add_site(name='tube_mouth',pos=[0,0,.11465],size=[.002],rgba=[1,.3,.1,1])
 tb.add_site(name='tube_grasp',pos=[0,0,c.tube_grasp_height_m],size=[.002],rgba=[1,.3,.1,0])
 tb.add_geom(name='tube_liquid_proxy_visual',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,.025],size=[.0123,.02,0],contype=0,conaffinity=0,density=0,rgba=[.83,.25,.09,.78],group=2)
 spec.worldbody.add_frame(pos=[*c.tube_xy_m,c.tabletop_z_m+c.tube_support_height_m+.008]).attach_body(tb,prefix=p)
 holder=spec.worldbody.add_body(name=p+'tube_holder',pos=[*c.tube_xy_m,c.tabletop_z_m+c.tube_support_height_m])
 if c.tube_support_height_m>0:holder.add_geom(name=p+'holder_plinth',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,-c.tube_support_height_m/2],size=[.020,.020,c.tube_support_height_m/2],rgba=[.80,.84,.81,1])
 holder.add_geom(name=p+'holder_base',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,.004],size=[.017,.004,0],rgba=[.85,.88,.83,1])
 for i in range(24):
  angle=2*np.pi*i/24
  # Open U nest: the west approach side has no raised collar, while base and east cradle stay physical.
  if np.cos(angle)<.2:continue
  holder.add_geom(name=p+'holder_ring_'+str(i),type=mujoco.mjtGeom.mjGEOM_BOX,pos=[.018*np.cos(angle),.018*np.sin(angle),.012],quat=[np.cos(angle/2),0,0,np.sin(angle/2)],size=[.0022,.0026,.003],rgba=[.82,.87,.84,1])
 for x in [-.042,.042]:
  holder.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[x,0,-c.tube_support_height_m+.004],size=[.006,.032,.004],rgba=[.82,.87,.84,1])
 riser=spec.worldbody.add_body(name=p+'receiver_nest',pos=[*c.receiver_xy_m,c.tabletop_z_m])
 riser.add_geom(name=p+'receiver_nest_solid',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,(c.receiver_bottom_z_m-.006-c.tabletop_z_m)/2],size=[.027,(c.receiver_bottom_z_m-.006-c.tabletop_z_m)/2,0],rgba=[.48,.52,.53,1])
 receiver=spec.body('aspiration_source')
 try:spec.geom('aspiration_source_liquid_visual').rgba[3]=0.
 except KeyError:pass
 receiver.add_geom(name=p+'receiver_liquid_proxy_visual',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,c.receiver_bottom_z_m+.00005],size=[c.receiver_inner_radius_m,.00005,0],contype=0,conaffinity=0,density=0,group=2,rgba=[.83,.25,.09,0])
 for i in range(12):spec.worldbody.add_geom(name=p+'pour_visual_'+str(i),type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.0014,0,0],contype=0,conaffinity=0,density=0,group=2,rgba=[.83,.25,.09,0])
 return dict(config=asdict(c),child_spec=child,tube_body=p+'reagent_tube',tube_joint=p+'reagent_tube_free',tube_mouth_site=p+'tube_mouth',tube_grasp_site=p+'tube_grasp',holder_body=p+'tube_holder',tube_visual=p+'tube_liquid_proxy_visual',receiver_visual=p+'receiver_liquid_proxy_visual',scope='Upstream50mL open tube visual/hollow collision wall reused; cap intentionally absent. Low custom holder. Fixed-mass surrogate; variable-mass fluid mechanics not simulated.')

class PreparationLedger:
 """Proxy liquid volume conservation driven only by actual mouth pose and tilt."""
 def __init__(self,c=None):
  self.config=c or PreparationConfig();self.tube_volume_m3=self.config.tube_initial_volume_m3;self.receiver_volume_m3=self.config.receiver_initial_volume_m3;self.spilled_m3=0.;self.delivered_m3=0.;self.events=[];self.last_time=None;self.mixing_proxy_angular_travel_rad=0.;self._previous_axis=None
 def observe(self,m,d,handles,*,allow_pour=False,allow_mix=False):
  c=self.config;time=float(d.time);dt=0. if self.last_time is None else time-self.last_time
  if dt<0:raise ValueError('Clock reversed')
  body=m.body(handles['tube_body']).id;mouth=d.site_xpos[m.site(handles['tube_mouth_site']).id].copy();axis=d.xmat[body].reshape(3,3)[:,2];tilt=float(np.degrees(np.arccos(np.clip(axis[2],-1,1))))
  if allow_mix and self._previous_axis is not None:self.mixing_proxy_angular_travel_rad+=float(np.arccos(np.clip(axis@self._previous_axis,-1,1)))
  self._previous_axis=axis.copy();self.last_time=time;flow=0.;inside=np.linalg.norm(mouth[:2]-c.receiver_xy_m)<c.receiver_inner_radius_m-.003;above=c.receiver_bottom_z_m+.006<mouth[2]<c.receiver_bottom_z_m+c.proxy_max_pour_height_above_bottom_m
  if allow_pour and dt>0 and tilt>=c.proxy_min_pour_tilt_deg and self.delivered_m3<c.target_pour_volume_m3:
   flow=min(self.tube_volume_m3,c.proxy_flow_rate_m3_s*dt,c.target_pour_volume_m3-self.delivered_m3);self.tube_volume_m3-=flow
   if inside and above:self.receiver_volume_m3+=flow;self.delivered_m3+=flow
   else:self.spilled_m3+=flow
   if flow:self.events.append(dict(time_s=time,delivered_m3=flow if inside and above else 0.,spilled_m3=flow if not(inside and above) else 0.))
  self.update_visual(m,d,handles,mouth,flow>0,inside and above)
  return dict(time_s=time,actual_tilt_deg=tilt,mouth_world_m=mouth.tolist(),over_receiver=bool(inside and above),flow_m3=flow)
 def update_visual(self,m,d,h,mouth=None,flow=False,inside=False):
  c=self.config
  for name,vol,radius,bottom in [(h['tube_visual'],self.tube_volume_m3,.0123,.002),(h['receiver_visual'],self.receiver_volume_m3,c.receiver_inner_radius_m,c.receiver_bottom_z_m)]:
   g=m.geom(name).id;height=max(.0001,vol/(np.pi*radius**2));m.geom_size[g,1]=height/2;m.geom_pos[g,2]=bottom+height/2;m.geom_rgba[g,3]=.78 if vol>1e-12 else 0.
  for i in range(12):
   g=m.geom(c.prefix+'pour_visual_'+str(i)).id;m.geom_rgba[g,3]=.8 if flow else 0.
   if flow:
    u=(i+1)/13;end=np.r_[np.asarray(c.receiver_xy_m),c.receiver_bottom_z_m+self.receiver_volume_m3/(np.pi*c.receiver_inner_radius_m**2)] if inside else mouth-[0,0,.10];m.geom_pos[g]=mouth*(1-u)+end*u
 def audit(self):
  c=self.config;error=c.tube_initial_volume_m3+c.receiver_initial_volume_m3-self.tube_volume_m3-self.receiver_volume_m3-self.spilled_m3
  return dict(model='pose-gated conservative proxy volume, not resolved liquid/mixing CFD',tube_remaining_m3=self.tube_volume_m3,receiver_volume_m3=self.receiver_volume_m3,delivered_m3=self.delivered_m3,spilled_m3=self.spilled_m3,balance_error_m3=error,receiver_surface_z_m=c.receiver_bottom_z_m+self.receiver_volume_m3/(np.pi*c.receiver_inner_radius_m**2),mixing_proxy_angular_travel_rad=self.mixing_proxy_angular_travel_rad,material_homogeneity_verified=False,fixed_mass_surrogate_kg=c.tube_mass_kg)
 def source_geometry(self):
  from t1_aspiration_stage import SourceGeometry
  c=self.config;return SourceGeometry(c.receiver_xy_m,c.receiver_inner_radius_m,c.receiver_bottom_z_m,self.receiver_volume_m3,c.receiver_bottom_z_m+self.receiver_volume_m3/(np.pi*c.receiver_inner_radius_m**2))






