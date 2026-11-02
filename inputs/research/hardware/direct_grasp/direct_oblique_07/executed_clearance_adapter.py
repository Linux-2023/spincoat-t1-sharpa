"""Independent A approach candidate: retreat reset + turn away from the tool."""
import os,json,argparse,types
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from t1_spincoat_scene import ROOT,arm_indices,hand_indices
from t1_mobile_stage import MobileDockController
from run_t1_front_workflow import adapted_runner,run_child

def prepare_clearance05(m,d,out,coarse_stop=False,runner_factory=None,absolute_mcp_overlay=False):
 a=types.SimpleNamespace(base_x=-.08,base_y=.4,base_approach=True)
 bqa=int(m.joint('chassis_free').qposadr[0]);d.qpos[bqa:bqa+3]=[-.08,.4,.09117];mujoco.mj_forward(m,d)
 config=json.loads((ROOT/'research/hardware/task_ready_natural_01_pinky_010mm.json').read_text());profile=config['motion_profile'];profile=json.loads(json.dumps(profile));profile['duration_s']=5.8
 ids,qa,ad=arm_indices(m);hj,hq,hd=hand_indices(m);palm=m.body('sharpa_right/right_hand_C_MC').id
 def base_target(t):
  if not a.base_approach:return a.base_x
  u=np.clip((t-.7)/2.,0,1);return a.base_x*(1-u*u*u*(10+u*(-15+6*u)))
 lo=m.jnt_range[ids,0]+.04;hi=m.jnt_range[ids,1]-.04;work=mujoco.MjData(m);work.qpos[:]=d.qpos;previous=np.clip(d.qpos[qa],lo+1e-6,hi-1e-6)
 targets=[('retracted_task_ready',0.,[.20,.02,1.10],[.5,.5,.5,.5]),('ready_hold',.7,[.20,.02,1.10],[.5,.5,.5,.5]),('turn_away_from_tool',1.9,[.22,.02,1.16],[2**-.5,2**-.5,0,0]),('high_pregrasp',2.9,[.31,.01,1.13],[2**-.5,2**-.5,0,0]),('diagonal_pregrasp',4.1,[0.3, 0.01, 1.025],[2**-.5,2**-.5,0,0]),('direct_oblique_entry',5.3,[.350,.01,1.002],[2**-.5,2**-.5,0,0]),('grasp_settle',5.8,[.350,.01,1.002],[2**-.5,2**-.5,0,0])]
 frames=[];ik_audit=[]
 for name,time,point,quat in targets:
  point=np.asarray(point);quat=np.asarray(quat)
  work.qpos[bqa]=base_target(time)
  def residual(q):
   work.qpos[qa]=q;mujoco.mj_kinematics(m,work);er=np.zeros(3);mujoco.mju_subQuat(er,quat,work.xquat[palm]);return np.r_[100*(work.xpos[palm]-point),10*er,.0001*(q-previous)]
  fits=[least_squares(residual,seed,bounds=(lo,hi),max_nfev=300,ftol=1e-11,xtol=1e-11,gtol=1e-11) for seed in [previous,np.clip(config['motion_profile']['waypoints'][min(len(frames),5)]['arm_q_rad'],lo+1e-6,hi-1e-6)]]
  best=min(fits,key=lambda f:np.linalg.norm(f.fun));rr=residual(best.x);previous=best.x.copy();frames.append({'name':name,'time_s':time,'arm_q_rad':previous.tolist()});ik_audit.append({'name':name,'target_wrist_m':point.tolist(),'position_error_mm':float(np.linalg.norm(rr[:3])*10),'orientation_error_deg':float(np.rad2deg(np.linalg.norm(rr[3:6])/10))})
 profile['waypoints']=frames;profile['hand_profile']['start_time_s']=1.9;profile['hand_profile']['end_time_s']=5.1;profile['provenance']='Independent retract/side-clearance model IK candidate; official S1 PI/teleop qualitative reference only, no joint-angle retargeting from video';profile['ik']=ik_audit;profile['base_reset_xy_m']=[a.base_x,a.base_y]
 path=out/'motion_profile.json';path.write_text(json.dumps(profile,indent=2))
 if max(r['position_error_mm'] for r in ik_audit)>.5 or max(r['orientation_error_deg'] for r in ik_audit)>.5:
  raise RuntimeError('clearance05 IK not feasible: '+json.dumps(ik_audit))
 d.qpos[qa]=frames[0]['arm_q_rad'];mujoco.mj_forward(m,d);dock=MobileDockController(m,[a.base_x,a.base_y]);
 if coarse_stop:
  from t1_coarse_stop import CoarseStopController
  dock=CoarseStopController(m,[0.,.4])
 prefix_fk=mujoco.MjData(m);armact=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in ids])
 module=(runner_factory or adapted_runner)('grasp');dummy=types.SimpleNamespace(to_xml=lambda:'<!-- Independent binary-model candidate; model.mjb is authoritative -->');audit=[];contacts=[];initial_pipe=d.xpos[m.body('pipette/pipette').id].copy();settled=None
 original_ik=module.ik
 def close_entry_ik(model,data,position,quat,**kw):
  if a.base_approach and data.time==0. and kw.get('iterations')==400:
   # Historical child validates a future grasp before running its approach.
   # Evaluate that validation at the future dock without changing live state.
   future=mujoco.MjData(model);future.qpos[:]=data.qpos;future.qpos[bqa]=0.;kw['seed']=np.array(frames[-1]['arm_q_rad'])
   return original_ik(model,future,position,quat,**kw)
  point=np.asarray(position).copy();u=np.clip((data.time-5.8)/1.5,0,1)
  if 5.8<=data.time<7.3:point[0]-=.025*(1-u*u*u*(10+u*(-15+6*u)))
  return original_ik(model,data,point,quat,**kw)
 module.ik=close_entry_ik
 module.workflow_left_gate=lambda:bool(dock.stopped)
 mcp_nominal={};mcp_trace=[]
 def callback(model,data,after):
  nonlocal settled
  if not after:
   dock.step(data,[base_target(data.time),a.base_y],enable_gate=coarse_stop or data.time>=2.7)
   if coarse_stop and data.time<5.8 and round(data.time/.001)%20==0:
    # Preserve the planned WORLD wrist path while the coarse base stops short.
    # Compensate through arm IK only; never adjust root/object qpos at runtime.
    prefix_fk.qpos[:]=data.qpos;prefix_fk.qpos[bqa]=base_target(data.time);prefix_fk.qpos[qa]=data.ctrl[armact]
    mujoco.mj_kinematics(model,prefix_fk)
    corrected,err=original_ik(model,data,prefix_fk.xpos[palm].copy(),prefix_fk.xquat[palm].copy(),seed=data.ctrl[armact].copy(),iterations=45)
    data.ctrl[armact]=corrected

   t=float(data.time);u=np.clip((t-1.9)/1.1,0,1);u=u*u*u*(10+u*(-15+6*u));v=np.clip((t-5.8)/.8,0,1);v=v*v*v*(10+v*(-15+6*v));offset=.1*u*(1-v)
   for j in hj:
    if 'thumb' not in model.joint(j).name and 'MCP_FE' in model.joint(j).name:
     act=int(np.flatnonzero(model.actuator_trnid[:,0]==j)[0])
     if round(data.time/.001)%20==0 or act not in mcp_nominal:mcp_nominal[act]=float(data.ctrl[act])
     if absolute_mcp_overlay:data.ctrl[act]=mcp_nominal[act]-offset
     else:data.ctrl[act]-=offset
   if absolute_mcp_overlay:mcp_trace.append([float(data.time),float(offset),*[mcp_nominal[k] for k in sorted(mcp_nominal)],*[float(data.ctrl[k]) for k in sorted(mcp_nominal)]])
   return
  t=float(data.time)
  if settled is None and t>=.7:settled=data.xpos[model.body('pipette/pipette').id].copy()
  for ci,c in enumerate(data.contact):
   names=[model.body(model.geom_bodyid[int(g)]).name for g in [c.geom1,c.geom2]]
   if any(n.startswith(('sharpa_right/','sharpa_left/')) for n in names) and any(n.startswith('pipette/') for n in names):
    f=np.zeros(6);mujoco.mj_contactForce(model,data,ci,f)
    contacts.append({'time_s':t,'bodies':names,'force_N':float(np.linalg.norm(f[:3])),'penetration_m':max(0,-float(c.dist))})
  if round(t/.001)%20==0:audit.append([t,*data.xpos[model.body('pipette/pipette').id],data.qpos[model.joint('pipette/pipette_button').qposadr[0]]])
 def audit_summary():
  if absolute_mcp_overlay:np.savez_compressed(out/'mcp_overlay_trace.npz',samples=np.asarray(mcp_trace),mode=np.array('absolute cached20ms nominal minus current offset'))
  early=[c for c in contacts if c['time_s']<5.8]
  return {'mcp_overlay_mode':'absolute_cached_nominal' if absolute_mcp_overlay else 'legacy_accumulating','formal_close_time_s':5.8,'prefix_hand_tool_contacts_1khz':early,'prefix_clear':not early,'all_hand_tool_contacts_1khz':contacts,'ik':ik_audit,'base_reset_xy_m':[-.08,.4],'initial_palm_world_m':[.2,.02,1.10]}
 audit_summary.dock=dock
 return module,path,callback,audit_summary

