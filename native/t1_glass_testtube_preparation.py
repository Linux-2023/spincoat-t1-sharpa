"""Catalog-dimension traditional glass tube + ordinary perforated 2x3 rack.
Independent module: does not modify recipe02 or the conical-tube assets.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np,mujoco
from t1_spincoat_scene import ROOT
from t1_reagent_preparation import PreparationConfig,PreparationLedger

CATALOG='https://www.fishersci.com/shop/products/eisco-rimmed-borosilicate-glass-test-tubes/s41472'
@dataclass
class GlassPreparationConfig(PreparationConfig):
 tube_support_height_m:float=.04
 tube_grasp_height_m:float=.115
 tube_capacity_m3:float=55e-6
 tube_mass_kg:float=.055
 tube_outer_radius_m:float=.0125
 tube_wall_thickness_m:float=.0011
 tube_length_m:float=.150
 rack_lower_height_m:float=.040
 rack_upper_height_m:float=.085
 rack_hole_radius_m:float=.01375
 rack_pitch_m:float=.045
 rack_rows:int=2
 rack_columns:int=3


def _lathe_obj(path,profile,n=64):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);verts=[];faces=[]
 for radius,z in profile:
  for k in range(n):a=2*np.pi*k/n;verts.append((radius*np.cos(a),radius*np.sin(a),z))
 for j in range(len(profile)-1):
  for k in range(n):a=j*n+k+1;b=j*n+(k+1)%n+1;c=(j+1)*n+(k+1)%n+1;d=(j+1)*n+k+1;faces.extend([(a,b,c),(a,c,d)])
 path.write_text(''.join('v %.10f %.10f %.10f\n'%v for v in verts)+''.join('f %d %d %d\n'%f for f in faces));return path


def _add_glass_tube(spec,name,xyz,c,free=False):
 R=c.tube_outer_radius_m;wall=c.tube_wall_thickness_m;ri=R-wall;L=c.tube_length_m
 # Complete outer hemisphere, cylindrical body, rolled rim, and inner cavity.
 profile=[(max(1e-7,R*np.sin(t)),R-R*np.cos(t)) for t in np.linspace(0,np.pi/2,13)]
 profile.extend([(R,L-.001),(R+.00045,L-.00055),(R+.00025,L),(ri-.0001,L),(ri,L-.001)])
 profile.extend([(max(1e-7,ri*np.sin(t)),R-ri*np.cos(t)) for t in np.linspace(np.pi/2,0,13)])
 profile.append(profile[0]);meshpath=_lathe_obj(ROOT/'assets/reagent_preparation/catalog_glass_25x150.obj',profile)
 if not any(mesh.name=='traditional_glass25x150' for mesh in spec.meshes):spec.add_mesh(name='traditional_glass25x150',file=str(meshpath))
 b=spec.worldbody.add_body(name=name,pos=xyz)
 if free:
  b.add_freejoint(name=c.prefix+'reagent_tube_free');mass=c.tube_mass_kg;b.mass=mass;b.ipos=[0,0,L*.48];b.inertia=[mass*(L*L/12+R*R/4)]*2+[mass*R*R*.7];b.explicitinertial=True
 b.add_geom(name=name+'_glass_visual',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='traditional_glass25x150',contype=0,conaffinity=0,density=0,rgba=[.83,.94,1,.26],group=2)
 if True:
  # Hollow cylindrical walls; no filled cylinder closes the mouth.
  mid=R-wall/2
  for k in range(48):
   a=2*np.pi*k/48;b.add_geom(name=name+'_wall_'+str(k),type=mujoco.mjtGeom.mjGEOM_BOX,pos=[mid*np.cos(a),mid*np.sin(a),(R+L)/2],quat=[np.cos(a/2),0,0,np.sin(a/2)],size=[wall/2,mid*np.tan(np.pi/48)*1.015,(L-R)/2],friction=[.8,.004,.0003],condim=4,group=3)
  # Thin spherical-bottom shell, approximated by meridian capsule patches.
  for k in range(32):
   a=2*np.pi*k/32
   for j in range(8):
    t0=np.pi/2*j/8;t1=np.pi/2*(j+1)/8;p0=[mid*np.sin(t0)*np.cos(a),mid*np.sin(t0)*np.sin(a),R-mid*np.cos(t0)];p1=[mid*np.sin(t1)*np.cos(a),mid*np.sin(t1)*np.sin(a),R-mid*np.cos(t1)];b.add_geom(name=name+'_bottom_'+str(k)+'_'+str(j),type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=p0+p1,size=[wall/2,0,0],friction=[.8,.004,.0003],condim=4,group=3)
 return b


def add_reagent_preparation(spec,config=None):
 c=config or GlassPreparationConfig();p=c.prefix;spec.memory=max(spec.memory,192*1024*1024);z=c.tabletop_z_m+c.rack_lower_height_m+.004
 if not 0<=c.tube_initial_volume_m3<=c.tube_capacity_m3:raise ValueError('Invalid glass-tube initial volume')
 tube=_add_glass_tube(spec,p+'reagent_tube',[*c.tube_xy_m,z+.0002],c,free=True)
 tube.add_site(name=p+'tube_mouth',pos=[0,0,c.tube_length_m],size=[.002],rgba=[1,.3,.1,1]);tube.add_site(name=p+'tube_grasp',pos=[0,0,c.tube_grasp_height_m],size=[.002],rgba=[1,.3,.1,0]);tube.add_geom(name=p+'tube_liquid_proxy_visual',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,.04],size=[c.tube_outer_radius_m-c.tube_wall_thickness_m,.02,0],contype=0,conaffinity=0,density=0,rgba=[.83,.25,.09,.78],group=2)
 for i in range(12):tube.add_geom(name=p+'tube_bottom_liquid_'+str(i),type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,.005],size=[.005,.0005,0],contype=0,conaffinity=0,density=0,rgba=[.83,.25,.09,.78],group=2)
 rack=spec.worldbody.add_body(name=p+'tube_holder',pos=[*c.tube_xy_m,c.tabletop_z_m]);pitch=c.rack_pitch_m;xr=[-.024,(c.rack_columns-1)*pitch+.024];yr=[-.024,(c.rack_rows-1)*pitch+.024];color=[.74,.60,.38,1]
 rack.add_geom(name=p+'rack_lower_tray',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[sum(xr)/2,sum(yr)/2,c.rack_lower_height_m],size=[(xr[1]-xr[0])/2,(yr[1]-yr[0])/2,.004],rgba=color)
 # Ordinary flat upper plate with real circular holes, constructed as physical strips.
 # Each strip excludes the circle conservatively; no collider spans a hole.
 for row in range(c.rack_rows):
  for col in range(c.rack_columns):
   cx,cy=col*pitch,row*pitch;left=(-.024 if col==0 else -pitch/2);right=(.024 if col==c.rack_columns-1 else pitch/2);bottom=(-.024 if row==0 else -pitch/2);top=(.024 if row==c.rack_rows-1 else pitch/2);edges=np.linspace(bottom,top,49)
   for k,(yl,yh) in enumerate(zip(edges,edges[1:])):
    closest=0. if yl<=0<=yh else min(abs(yl),abs(yh));gap=np.sqrt(max(0,c.rack_hole_radius_m**2-closest**2))
    spans=[(left,right)] if closest>=c.rack_hole_radius_m else [(left,-gap),(gap,right)]
    for side,(xl,xh) in enumerate(spans):
     if xh-xl<1e-6:continue
     rack.add_geom(name=p+'rack_top_strip_%d_%d_%d_%d'%(row,col,k,side),type=mujoco.mjtGeom.mjGEOM_BOX,pos=[cx+(xl+xh)/2,cy+(yl+yh)/2,c.rack_upper_height_m],size=[(xh-xl)/2,(yh-yl)/2,.003],rgba=color)
 for x in xr:
  for y in yr:rack.add_geom(name=p+'rack_leg_'+str(x)+'_'+str(y),type=mujoco.mjtGeom.mjGEOM_BOX,pos=[x,y,c.rack_upper_height_m/2],size=[.005,.005,c.rack_upper_height_m/2+.003],rgba=color)
 for j,(row,col) in enumerate([(0,2),(1,1),(1,2)]):_add_glass_tube(spec,p+'empty_glass_tube_'+str(j),[c.tube_xy_m[0]+col*pitch,c.tube_xy_m[1]+row*pitch,z],c,free=False)
 nest=spec.worldbody.add_body(name=p+'receiver_nest',pos=[*c.receiver_xy_m,c.tabletop_z_m]);hh=(c.receiver_bottom_z_m-.006-c.tabletop_z_m)/2;nest.add_geom(name=p+'receiver_nest_solid',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,hh],size=[.027,hh,0],rgba=[.48,.52,.53,1])
 try:spec.geom('aspiration_source_liquid_visual').rgba[3]=0.
 except KeyError:pass
 spec.body('aspiration_source').add_geom(name=p+'receiver_liquid_proxy_visual',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,c.receiver_bottom_z_m+.00005],size=[c.receiver_inner_radius_m,.00005,0],contype=0,conaffinity=0,density=0,rgba=[.83,.25,.09,0],group=2)
 for i in range(12):spec.worldbody.add_geom(name=p+'pour_visual_'+str(i),type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.0014,0,0],contype=0,conaffinity=0,density=0,rgba=[.83,.25,.09,0],group=2)
 rack.add_site(name=p+'rack_display_center',pos=[sum(xr)/2,sum(yr)/2,c.rack_upper_height_m],size=[.002],rgba=[0,0,0,0])
 return dict(preparation_asset_type='traditional_glass_testtube',ledger_module='t1_glass_testtube_preparation',ledger_class='GlassPreparationLedger',config_class='GlassPreparationConfig',rack_body=p+'tube_holder',rack_display_center_site=p+'rack_display_center',rack_top_world_z_m=c.tabletop_z_m+c.rack_upper_height_m+.003,config=asdict(c),child_spec=mujoco.MjSpec(),tube_body=p+'reagent_tube',tube_joint=p+'reagent_tube_free',tube_mouth_site=p+'tube_mouth',tube_grasp_site=p+'tube_grasp',holder_body=p+'tube_holder',tube_visual=p+'tube_liquid_proxy_visual',receiver_visual=p+'receiver_liquid_proxy_visual',catalog_url=CATALOG,scope='25x150mm round-bottom borosilicate nominal55mL catalog-dimension prototype, not manufacturer CAD; ordinary two-layer2x3 perforated rack; only target tube is dynamic; background empty tubes are fixed physical shell obstacles. Fixed mass and conservative visible fluid proxy, no glass fracture/CFD.')


class GlassPreparationLedger(PreparationLedger):
 def update_visual(self,m,d,h,mouth=None,flow=False,inside=False):
  # Parent updates cup and flow; replace tube column with round-bottom volume geometry.
  super().update_visual(m,d,h,mouth,flow,inside);c=self.config;R=c.tube_outer_radius_m;ri=R-c.tube_wall_thickness_m;cap=2*np.pi*ri**3/3;volume=self.tube_volume_m3
  if volume>=cap:level=R+(volume-cap)/(np.pi*ri**2)
  else:
   lo,hi=0.,ri
   for _ in range(45):
    u=(lo+hi)/2;v=np.pi*u*u*(ri-u/3)
    if v<volume:lo=u
    else:hi=u
   level=R-ri+(lo+hi)/2
  g=m.geom(h['tube_visual']).id;height=max(.0001,level-R);m.geom_pos[g,2]=R+height/2;m.geom_size[g,0]=ri;m.geom_size[g,1]=height/2;m.geom_rgba[g,3]=.78 if level>R else 0
  for i in range(12):
   low=R-ri+ri*i/12;high=min(level,R-ri+ri*(i+1)/12);g=m.geom(c.prefix+'tube_bottom_liquid_'+str(i)).id;mid=(low+max(low,high))/2;radius=np.sqrt(max(1e-10,ri**2-(R-mid)**2));m.geom_size[g,0]=radius;m.geom_size[g,1]=max(.00001,(high-low)/2);m.geom_pos[g,2]=mid;m.geom_rgba[g,3]=.78 if high>low else 0

# Matching names make the root builder select this independent module explicitly.
PreparationConfig=GlassPreparationConfig
PreparationLedger=GlassPreparationLedger

