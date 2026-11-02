"""Independent left-tube/shared professional Griffin beaker layout.
Catalog-dimension prototypes; no changes to acceptedrecipe02.
"""
from dataclasses import dataclass,asdict,replace
import numpy as np,mujoco
from t1_spincoat_scene import ROOT
from t1_glass_testtube_preparation import GlassPreparationConfig,GlassPreparationLedger,add_reagent_preparation as add_glass,_lathe_obj
BEAKER_CATALOG='https://ecatalog.corning.com/life-sciences/b2b/CA/en/General-Labware/Beakers/Beakers%2C-Glass/PYREX%C2%AE-Beakers%2C-Low-Form%2C-Double-Scale%2C-Graduated/p/1000-50'
@dataclass
class BimanualGlassConfig(GlassPreparationConfig):
 tube_xy_m:tuple=(.43,.70)
 receiver_xy_m:tuple=(.45,.40)
 receiver_bottom_z_m:float=.8020005
 receiver_inner_radius_m:float=.0197
 beaker_outer_radius_m:float=.021
 beaker_height_m:float=.056
 beaker_wall_assumption_m:float=.0013
 beaker_bottom_assumption_m:float=.002
 preparation_side:str='left'
 rack_yaw_deg:float=-90.
 transport_yaw_deg:float=-45.


def _digits(body,text,y,z,prefix):
 segments={'0':'abcdef','1':'bc','2':'abged','3':'abgcd','4':'fgbc','5':'afgcd','6':'afgecd','7':'abc','8':'abcdefg','9':'abfgcd'};coords={'a':(0,.0022,True),'g':(0,0,True),'d':(0,-.0022,True),'f':(-.0011,.0011,False),'b':(.0011,.0011,False),'e':(-.0011,-.0011,False),'c':(.0011,-.0011,False)}
 for i,digit in enumerate(text):
  for seg in segments[digit]:
   yy,zz,horizontal=coords[seg];body.add_geom(name=prefix+'digit_'+str(i)+seg,type=mujoco.mjtGeom.mjGEOM_BOX,pos=[.02115,y+i*.0032+yy,z+zz],size=[.0001,.00105 if horizontal else .00017,.00017 if horizontal else .00105],contype=0,conaffinity=0,density=0,rgba=[.94,.97,1,1],group=2)


def add_reagent_preparation(spec,config=None):
 c=config or BimanualGlassConfig();temporary=replace(c,receiver_bottom_z_m=c.tabletop_z_m+.0061);h=add_glass(spec,temporary);p=c.prefix;receiver=spec.body('aspiration_source');receiver.pos[:]=[*c.receiver_xy_m,0.];liquidname=p+'receiver_liquid_proxy_visual'
 # RESET-only replacement: old generic cup/riser geometries are not retained as active obstacles.
 for g in receiver.geoms:
  if g.name!=liquidname:g.contype=0;g.conaffinity=0;g.rgba[3]=0
 for g in next(body for body in spec.bodies if body.name==p+'receiver_nest').geoms:g.contype=0;g.conaffinity=0;g.rgba[3]=0
 next(g for g in spec.geoms if g.name==liquidname).pos[:]=[0,0,c.receiver_bottom_z_m+.00005]
 R=c.beaker_outer_radius_m;ri=c.receiver_inner_radius_m;H=c.beaker_height_m;bottom=c.beaker_bottom_assumption_m
 profile=[(1e-7,0),(R,0),(R,H-.001),(R+.0005,H),(ri,H),(ri,bottom),(1e-7,bottom),(1e-7,0)];meshfile=_lathe_obj(ROOT/'assets/reagent_preparation/catalog_griffin50.obj',profile)
 if not any(x.name=='griffin50_catalog_mesh' for x in spec.meshes):spec.add_mesh(name='griffin50_catalog_mesh',file=str(meshfile))
 b=receiver.add_body(name=p+'shared_griffin_beaker',pos=[0,0,c.tabletop_z_m]);b.add_geom(name=p+'griffin_glass_visual',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='griffin50_catalog_mesh',contype=0,conaffinity=0,density=0,rgba=[.83,.94,1,.25],group=2);b.add_geom(name=p+'griffin_floor',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,bottom/2],size=[R,bottom/2,0],rgba=[.83,.94,1,.25],group=3)
 for k in range(64):
  angle=2*np.pi*k/64;wall=R-ri;mid=(R+ri)/2;height=H-bottom-(.004 if k in [15,16,17] else 0);b.add_geom(name=p+'griffin_wall_'+str(k),type=mujoco.mjtGeom.mjGEOM_BOX,pos=[mid*np.cos(angle),mid*np.sin(angle),bottom+height/2],quat=[np.cos(angle/2),0,0,np.sin(angle/2)],size=[wall/2,mid*np.tan(np.pi/64)*1.02,height/2],rgba=[.83,.94,1,.25],group=3)
 # Small outward pouring lip, on the+Y side away from the main graduation face.
 for side in [-1,1]:b.add_geom(name=p+'griffin_spout_'+str(side),type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=[side*.005,.020,H-.001,side*.0025,.026,H+.0005],size=[.00075,0,0],rgba=[.82,.94,1,.65],group=2)
 for ml in [10,20,30,40]:
  zz=bottom+ml*1e-6/(np.pi*ri**2);b.add_geom(name=p+'graduation_'+str(ml),type=mujoco.mjtGeom.mjGEOM_BOX,pos=[R+.0002,0,zz],size=[.00012,.008,.00025],contype=0,conaffinity=0,density=0,rgba=[.96,.98,1,1],group=2);_digits(b,str(ml),-.007,zz+.0035,p+'mark'+str(ml))
 _digits(b,'50',-.004,H-.006,p+'nominal50');b.add_site(name=p+'beaker_display_center',pos=[0,0,H/2],size=[.002],rgba=[0,0,0,0])
 # RESET-only rack orientation: target free tube stays fixed, ordinary holes/background move together.
 yaw=np.deg2rad(c.rack_yaw_deg);rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]);pivot=np.array(c.tube_xy_m);qz=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
 for rb in spec.bodies:
  if rb.name==p+'tube_holder' or rb.name.startswith(p+'empty_glass_tube_'):
   rb.pos[:2]=pivot+rot@(np.array(rb.pos[:2])-pivot);rb.quat=qz
 h.update(config=asdict(c),ledger_module='t1_bimanual_glass_preparation',ledger_class='BimanualGlassLedger',config_class='BimanualGlassConfig',preparation_asset_type='bimanual_traditional_glass_testtube',preparation_side='left',manipulation_side='left',side='left',receiver_body=p+'shared_griffin_beaker',receiver_display_center_site=p+'beaker_display_center',beaker_catalog_url=BEAKER_CATALOG,beaker_dimensions_scope='Corning1000-50 nominal50mL/OD42mm/H56mm/spout/graduations10-40mL; wall1.3mm/bottom2mm engineering assumptions, not manufacturer measurements; marks visual labels not calibrated volumetry.',scope=h['scope']+' Left rack and shared professional Griffin50mL receiver, fixed directly on tabletop; original spincoater collection cavity unchanged.')
 return h
class BimanualGlassLedger(GlassPreparationLedger):pass
PreparationConfig=BimanualGlassConfig
PreparationLedger=BimanualGlassLedger
