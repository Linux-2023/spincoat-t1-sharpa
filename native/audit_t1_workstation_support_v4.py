import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle,Circle


def clip(poly,axis,bound,greater):
 out=[]
 for a,b in zip(poly,np.roll(poly,-1,axis=0)):
  ina=(a[axis]>=bound) if greater else (a[axis]<=bound)
  inb=(b[axis]>=bound) if greater else (b[axis]<=bound)
  if ina:out.append(a)
  if ina!=inb:out.append(a+(b-a)*(bound-a[axis])/(b[axis]-a[axis]))
 return np.array(out)
def area(p):
 return float(abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2) if len(p)>2 else 0.
def footprint(m,d,geom):
 g=m.geom(geom).id;s=m.geom_size[g];xy=np.array([[-s[0],-s[1],0],[s[0],-s[1],0],[s[0],s[1],0],[-s[0],s[1],0]])
 return (d.geom_xpos[g]+xy@d.geom_xmat[g].reshape(3,3).T)[:,:2]
def support(poly,bounds):
 clipped=poly.copy()
 for axis,lo,hi in [(0,bounds[0],bounds[1]),(1,bounds[2],bounds[3])]:
  clipped=clip(clipped,axis,lo,True);clipped=clip(clipped,axis,hi,False)
 margins=[float(poly[:,0].min()-bounds[0]),float(bounds[1]-poly[:,0].max()),float(poly[:,1].min()-bounds[2]),float(bounds[3]-poly[:,1].max())]
 return dict(polygon_world_xy_m=poly.tolist(),support_fraction=area(clipped)/area(poly),edge_margins_m=dict(front_x_min=margins[0],rear_x_max=margins[1],side_y_min=margins[2],side_y_max=margins[3]),whole_footprint_supported=bool(min(margins)>=0))

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 m=mujoco.MjModel.from_binary_path(str(a.source/'model.mjb'));d=mujoco.MjData(m)
 with np.load(a.source/'handoff_initial.npz',allow_pickle=False) as z:
  print('initial keys',z.files);d.qpos[:]=z['qpos'][0] if z['qpos'].ndim>1 else z['qpos']
 mujoco.mj_forward(m,d);g=m.geom('bench').id;c=d.geom_xpos[g];s=m.geom_size[g];bounds=[float(c[0]-s[0]),float(c[0]+s[0]),float(c[1]-s[1]),float(c[1]+s[1])]
 oldpoly=footprint(m,d,'stand/stand_base');center=oldpoly.mean(axis=0);rows=[]
 candidates=[('v3 unsupported',center),('original station',center+[.075,0]),('inside side station',np.array([.47,-.08])),('rear side station',np.array([.48,-.12])),('front supported station',np.array([.405,-.08]))]
 fig,ax=plt.subplots(figsize=(10,8));ax.add_patch(Rectangle((bounds[0],bounds[2]),bounds[1]-bounds[0],bounds[3]-bounds[2],facecolor='#eeeeee',edgecolor='black'));ax.axvline(bounds[0],color='black',lw=2)
 colors=['red','#888888','#2b83ba','#1a9850','#984ea3']
 for (name,cc),color in zip(candidates,colors):
  poly=oldpoly+(cc-center);r=support(poly,bounds);r.update(name=name,center_world_xy_m=cc.tolist(),delta_from_original_station_world_m=(cc-(center+[.075,0])).tolist());rows.append(r);ax.add_patch(Polygon(poly,fill=False,ec=color,lw=2,label=name));ax.text(cc[0]+.01,cc[1]+.015,name,color=color,fontsize=8)
 for xy,label in [((.4,.23),'v3 cup'),((.36,.35),'LEFT cup candidate'),((.43,-.34),'coater')]:
  ax.add_patch(Circle(xy,.021 if label!='coater' else .072,fill=False,ec='orange' if label!='coater' else 'blue'));ax.text(xy[0]+.024,xy[1],label,fontsize=8)
 ax.plot([.4,.43],[.23,-.34],'--',color='orange',label='cup to coater center line');ax.set(xlim=(.22,.75),ylim=(-.51,.52),xlabel='world X (m)',ylabel='world Y (m)',title='Physical bench support and complete rotated stand footprint\nMeasured v3 model; candidate layout only, no new physics rollout');ax.set_aspect('equal');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(a.out/'support_footprints.png',dpi=160);plt.close(fig)
 torso=[m.joint('astribot_torso_joint_'+str(i)) for i in range(1,5)];qadr=[int(j.qposadr[0]) for j in torso];orig=d.qpos.copy();shoulder=m.body('astribot_right_arm_link_1').id if False else m.body('astribot_arm_right_link_1').id if False else None
 names=[m.body(i).name for i in range(m.nbody) if 'torso_link_4' in m.body(i).name or ('right' in m.body(i).name and 'link_1' in m.body(i).name)];print('bodies',names)
 top=m.body('astribot_torso_link_4').id;initial=d.xpos[top].copy();R0=d.xmat[top].reshape(3,3).copy();squats=[]
 def contacts(data):
  pairs={}
  for co in data.contact:
   if co.dist < -.001:
    pair=tuple(sorted(m.geom(int(g)).name or str(g) for g in co.geom));pairs['|'.join(pair)]=float(-co.dist)
  return pairs
 baseline=contacts(d)
 for aa in [.3,.4,.45,.5,.55,.6]:
  d.qpos[:]=orig;d.qpos[qadr]=[aa,-2*aa,aa,0];mujoco.mj_forward(m,d);cs=contacts(d);new={k:v for k,v in cs.items() if k not in baseline}
  tag='squat_'+str(round(aa*100));np.savez_compressed(a.out/(tag+'_STATIC.npz'),qpos=d.qpos.copy(),qvel=np.zeros(m.nv),torso_qpos=d.qpos[qadr],scope=np.array('STATIC FK/contact screening; NOT actual squat'))
  squats.append(dict(a_rad=aa,torso_q_rad=d.qpos[qadr].tolist(),torso4_delta_m=(d.xpos[top]-initial).tolist(),upperbody_rotation_error_matrix_max=float(np.max(abs(d.xmat[top].reshape(3,3)-R0))),new_native_penetrations_over_1mm=new,all_native_penetrations_over_1mm=cs))
 mujoco.mj_saveModel(m,str(a.out/'STATIC_model.mjb'),None)
 report=dict(scope='STATIC support geometry and coupled torso FK/native contact screen, not actual trajectory or balance proof',source=str(a.source),source_model_sha256=hashlib.sha256((a.source/'model.mjb').read_bytes()).hexdigest(),table_support_xy_bounds_m=bounds,table_top_z_m=float(c[2]+s[2]),stand_candidates=rows,torso_joint_names=[j.name for j in torso],torso_joint_range_rad=[j.range.tolist() for j in torso],squat_fk=squats)
 (a.out/'support_and_squat_STATIC.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
