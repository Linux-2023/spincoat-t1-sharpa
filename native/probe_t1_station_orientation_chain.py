"""Independent B: fresh reset-reoriented rack/tool, frozen support04 grip, live aspiration.

Only this script and versioned outputs are written. Existing scripts are read and
compiled into isolated modules; no process-external monkey patches or asset edits.
"""
import os,json,types,inspect,argparse,hashlib
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from t1_spincoat_scene import ROOT,arm_indices
import t1_support_grasp

def qfrom(R):return Rotation.from_matrix(R).as_quat()[[3,0,1,2]]

def actual_lift_gate(m,d,records,timing,out):
 """Read-only stage acceptance; failed pick never enters aspiration."""
 z=np.load(out/'handoff_initial.npz');check=mujoco.MjData(m);check.qpos[:]=z['qpos'];mujoco.mj_forward(m,check)
 tip=m.site('pipette/tip_site').id;initial_z=float(check.site_xpos[tip,2]);lift=float(d.site_xpos[tip,2]-initial_z)
 held=[r for r in records if r[0]>=timing['hold_start_s']]
 counts=np.array([r[2][1:] for r in held]);positive=np.array([r[3][1:] for r in held])
 gate=dict(actual_tip_lift_m=lift,required_tip_lift_m=.06,hold_observer_samples_1ms=len(held),four_geometry_support_all_samples=bool(len(held) and (counts>0).all()),four_positive_wrench_support_all_samples=bool(len(held) and (positive>0).all()),min_geometry_contacts_per_finger=counts.min(axis=0).tolist() if len(held) else [],min_positive_contacts_per_finger=positive.min(axis=0).tolist() if len(held) else [],scope='Read-only privileged simulator stage acceptance; no force-driven pose/torque adjustment. No held seed.')
 gate['pass']=bool(lift>=.06 and gate['four_geometry_support_all_samples'] and gate['four_positive_wrench_support_all_samples'])
 (out/'actual_held_lift_gate.json').write_text(json.dumps(gate,indent=2));return gate

def reset_station(m,d,angle,out):
 assert d.time==0.,'Station can only change at reset'
 mujoco.mj_forward(m,d);pipe=m.body('pipette/pipette').id;center=d.xpos[pipe].copy();turn=Rotation.from_euler('z',angle,degrees=True).as_matrix();qbefore=d.qpos.copy();bpos=m.body_pos.copy();bquat=m.body_quat.copy();stand=[b for b in range(m.nbody) if m.body(b).name.startswith('stand/')];roots=[b for b in stand if int(m.body_parentid[b]) not in stand]
 assert roots,'No stand root'
 changes=[]
 for b in roots:
  par=int(m.body_parentid[b]);P=d.xpos[par].copy();PR=d.xmat[par].reshape(3,3).copy();target=center+turn@(d.xpos[b]-center);R=turn@d.xmat[b].reshape(3,3)
  m.body_pos[b]=PR.T@(target-P);m.body_quat[b]=qfrom(PR.T@R);changes.append(dict(body=m.body(b).name,original_pos=bpos[b].tolist(),new_pos=m.body_pos[b].tolist(),original_quat=bquat[b].tolist(),new_quat=m.body_quat[b].tolist()))
 pj=int(m.body_jntadr[pipe]);assert m.jnt_type[pj]==mujoco.mjtJoint.mjJNT_FREE;pq=int(m.jnt_qposadr[pj]);Rpipe=d.xmat[pipe].reshape(3,3).copy();d.qpos[pq:pq+3]=center;d.qpos[pq+3:pq+7]=qfrom(turn@Rpipe);mujoco.mj_forward(m,d)
 # Every robot joint stays identical at this reset operation.
 robotmask=np.ones(m.nq,bool);robotmask[pq:pq+7]=False
 report=dict(station_yaw_deg=angle,pivot_world_m=center.tolist(),phase='RESET only, before first mj_step; not a held snapshot',source_qpos='integrated_support04_01/grasp.npz first sample; qvel reset to zero as original support probe',model_body_changes=changes,pipette_free_qpos=d.qpos[pq:pq+7].tolist(),robot_qpos_unchanged_by_station_reset=bool(np.array_equal(qbefore[robotmask],d.qpos[robotmask])),original_gravity_settle_0p7s_retained=True,adapter_unchanged=True)
 np.savez_compressed(out/'station_reset_initial_state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
 (out/'station_reset.json').write_text(json.dumps(report,indent=2))
 return center,turn

def station_prepare(m,d,out,plan,*,time_scale=1.25):
 center,turn=reset_station(m,d,station_prepare.angle,out)
 ids,qa,_=arm_indices(m);palm=m.body('sharpa_right/right_hand_C_MC').id;elbow=m.body('astribot_arm_right_link_5').id;wrist=m.body('astribot_arm_right_link_6').id;mid=m.body('sharpa_right/right_middle_MCP_VL').id
 native_seed=json.loads((ROOT/'research/hardware/neutral_wrist/station_orientation_B01/yaw60/report.json').read_text())['best']['arm_q_rad'];seed=np.asarray(native_seed);closing_scale=station_prepare.closing_scale;extra_close=2*(closing_scale-1)*time_scale;verticalize_start=5.8+8.5*time_scale+extra_close
 def task_clock(seconds):
  clock=seconds/time_scale
  if clock<=1.5:return clock
  if clock<1.5+2*closing_scale:return 1.5+(clock-1.5)/closing_scale
  return clock-2*(closing_scale-1)
 def transform_point(point):return center+turn@(np.asarray(point)-center)
 def transform_quat(quat):return qfrom(turn@Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix())
 solver_audit=[]
 def solve(work,point,quat,previous,lo,hi,residual):
  RG=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
  def measure(q):
   work.qpos[qa]=q;mujoco.mj_kinematics(m,work);f=work.xpos[wrist]-work.xpos[elbow];v=work.xpos[mid]-work.xpos[palm]
   return f,v,np.r_[work.xpos[palm]-point,Rotation.from_matrix(work.xmat[palm].reshape(3,3)@RG.T).as_rotvec()]
  def objective(q):
   f,v,_=measure(q);return 1-f@v/(np.linalg.norm(f)*np.linalg.norm(v))+.0002*np.sum((q-previous)**2)
  fits=[minimize(objective,np.clip(start,lo,hi),method='SLSQP',bounds=list(zip(lo,hi)),constraints={'type':'eq','fun':lambda q:measure(q)[2]},options={'ftol':1e-10,'maxiter':150}) for start in [previous,seed]]
  valid=[r for r in fits if np.linalg.norm(measure(r.x)[2][:3])<.0002 and np.linalg.norm(measure(r.x)[2][3:])<np.deg2rad(.1)]
  best=min(valid,key=lambda r:objective(r.x)) if valid else min(fits,key=lambda r:np.linalg.norm(measure(r.x)[2]))
  f,v,e=measure(best.x);solver_audit.append(dict(point_world_m=np.asarray(point).tolist(),palm_error_mm=float(np.linalg.norm(e[:3])*1000),orientation_error_deg=float(np.degrees(np.linalg.norm(e[3:]))),forearm_palm_angle_deg=float(np.degrees(np.arccos(np.clip(f@v/(np.linalg.norm(f)*np.linalg.norm(v)),-1,1)))),success=bool(valid)))
  return types.SimpleNamespace(x=best.x,fun=residual(best.x))
 env=dict(station_point=transform_point,station_quat=transform_quat,station_pose_solve=solve,station_verticalize_start=verticalize_start)
 src=(ROOT/'research/hardware/direct_grasp/direct_oblique_07/executed_clearance_adapter.py').read_text(encoding='utf-8-sig')
 needle='  point=np.asarray(point);quat=np.asarray(quat)'
 assert src.count(needle)==1
 src=src.replace(needle,needle+'\n  if time>=1.9:point=station_point(point);quat=station_quat(quat)')
 needle='  best=min(fits,key=lambda f:np.linalg.norm(f.fun));rr=residual(best.x);previous=best.x.copy();'
 assert src.count(needle)==1
 src=src.replace(needle,'  best=station_pose_solve(work,point,quat,previous,lo,hi,residual) if time>=1.9 else min(fits,key=lambda f:np.linalg.norm(f.fun));rr=residual(best.x);previous=best.x.copy();')
 # Transform only original world-coordinate close/lift requests. Later
 # verticalization uses live measured tool/hand coordinates and is NOT rotated twice.
 src=src.replace('return original_ik(model,future,position,quat,**kw)','return original_ik(model,future,station_point(position),station_quat(quat),**kw)')
 needle='  return original_ik(model,data,point,quat,**kw)';assert src.count(needle)==1
 src=src.replace(needle,'  if data.time<station_verticalize_start:point=station_point(point);quat=station_quat(quat)\n'+needle)
 (out/'station_clearance_adapter_source.py').write_text(src)
 prepare=inspect.getsource(t1_support_grasp.prepare_support_grasp)
 needle="('task_t=t-prefix','task_t=(t-prefix)/'+str(time_scale))";assert prepare.count(needle)==1
 prepare=prepare.replace(needle,"('task_t=t-prefix','task_t=station_task_clock(t-prefix)')")
 needle='module.support_fit=fit;module.support_open=openq;return module';assert prepare.count(needle)==1
 prepare=prepare.replace(needle,'module.support_fit=fit;module.support_open=openq;module.station_task_clock=station_task_clock;return module')
 needle="src=(ROOT/'research/hardware/direct_grasp/direct_oblique_07/executed_clearance_adapter.py').read_text()";assert prepare.count(needle)==1
 prepare=prepare.replace(needle,'src=station_clearance_source')
 needle="module=types.ModuleType('support_clearance');exec(";assert prepare.count(needle)==1
 prepare=prepare.replace(needle,"module=types.ModuleType('support_clearance');module.__dict__.update(station_env);exec(")
 glob=dict(t1_support_grasp.prepare_support_grasp.__globals__);glob.update(station_clearance_source=src,station_env=env,station_task_clock=task_clock)
 exec(compile(prepare,str(out/'station_prepare_support_source.py'),'exec'),glob);(out/'station_prepare_support_source.py').write_text(prepare)
 result=glob['prepare_support_grasp'](m,d,out,plan,time_scale=time_scale)
 timing=result[4]
 for key in ['hold_start_s','lift_start_s','postlift_hold_start_s','verticalize_start_s','stage_end_s']:timing[key]+=extra_close
 timing.update(closing_scale_only=closing_scale,closing_added_seconds=extra_close)
 profile=json.loads(result[1].read_text());profile['staged_grasp_timing']=timing;result[1].write_text(json.dumps(profile,indent=2))
 (out/'station_prefix_ik_audit.json').write_text(json.dumps(solver_audit,indent=2))
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--angle-deg',type=float,default=60.);p.add_argument('--time-scale',type=float,default=1.25);p.add_argument('--closing-scale',type=float,default=1.);a=p.parse_args();station_prepare.angle=a.angle_deg;station_prepare.closing_scale=a.closing_scale
 source=(ROOT/'scripts/probe_t1_support_grasp.py').read_text(encoding='utf-8-sig')
 source=source.replace("out=ROOT/'research/hardware/grasp_azimuth'/a.tag","out=ROOT/'research/hardware/neutral_wrist'/a.tag")
 source=source.replace("plan=make_support_plan(out/'plan')","plan=json.loads((ROOT/'research/hardware/grasp_azimuth/support_staged_04/contact_plan.json').read_text())")
 source=source.replace("ref=ROOT/'logs/spincoat/t1/integrated_direct07_01'","ref=ROOT/'logs/spincoat/t1/integrated_support04_01'")
 source=source.replace('str(11.5*a.time_scale)','str(11.5*a.time_scale+station_close_extra)')
 needle="  if ga['prefix_clear'] and not gr['failure_reason'] and not gr['robot_station_contact_violation']:";assert source.count(needle)==1
 source=source.replace(needle,"  held=station_actual_lift_gate(m,d,records,timing,out);result['actual_held_lift_gate']=held\n"+needle[:-1]+" and held['pass']:")
 # Add read-only full-rate relative 6D and longitudinal wrist-angle diagnostics.
 needle="records.append((float(data.time),float(data.qpos[model.joint('pipette/pipette_button').qposadr[0]]),counts,positive,normal,magnitude,segments,rack_count,rack_force))"
 assert source.count(needle)==1
 extra="pb=model.body('pipette/pipette').id;hb=model.body('sharpa_right/right_hand_C_MC').id;R=data.xmat[hb].reshape(3,3);relative_position=R.T@(data.xpos[pb]-data.xpos[hb]);relative_rotation=R.T@data.xmat[pb].reshape(3,3);f=data.xpos[model.body('astribot_arm_right_link_6').id]-data.xpos[model.body('astribot_arm_right_link_5').id];v=data.xpos[model.body('sharpa_right/right_middle_MCP_VL').id]-data.xpos[hb];wrist_angle=float(np.degrees(np.arccos(np.clip(f@v/(np.linalg.norm(f)*np.linalg.norm(v)),-1,1))))\n  "
 source=source.replace(needle,extra+needle.replace('rack_count,rack_force))','rack_count,rack_force,relative_position.copy(),relative_rotation.copy(),wrist_angle))'))
 source=source.replace("'tool_rack_normal_force_N'];arrays=","'tool_rack_normal_force_N','relative_tool_position_in_palm_m','relative_tool_rotation_in_palm','forearm_palm_angle_deg'];arrays=")
 # Isolated module will use our preparation function in place of imported helper.
 engine=types.ModuleType('station_orientation_B_chain');engine.__file__=str(ROOT/'scripts/probe_t1_support_grasp.py');exec(compile(source,engine.__file__,'exec'),engine.__dict__);engine.prepare_support_grasp=station_prepare;engine.station_actual_lift_gate=actual_lift_gate;engine.station_close_extra=2*(a.closing_scale-1)*a.time_scale
 import sys
 old=sys.argv;sys.argv=['station_B','--tag',a.tag,'--time-scale',str(a.time_scale)]
 out=ROOT/'research/hardware/neutral_wrist'/a.tag
 try:engine.main()
 finally:
  sys.argv=old
  if out.exists():
   snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True);files=[Path(__file__),ROOT/'scripts/probe_t1_support_grasp.py',ROOT/'scripts/t1_support_grasp.py',ROOT/'scripts/run_t1_integrated_workflow.py',ROOT/'scripts/run_t1_spincoat_episode.py',ROOT/'scripts/t1_coarse_stop.py'];hashes={}
   for path in files:
    target=snapshot/path.name;target.write_bytes(path.read_bytes());hashes[path.name]=hashlib.sha256(target.read_bytes()).hexdigest()
   (out/'executed_station_chain_engine.py').write_text(source)
   (out/'station_chain_provenance.json').write_text(json.dumps(dict(phase='Fresh initial → gravity settle/approach → seat/close/lift → same-live aspiration',station_yaw_deg=a.angle_deg,initial_qpos_source='support04 grasp[0], no held seed',hand_targets='Frozen support_staged_04 contact_plan; unchanged',world_station_pose_changed=True,adapter_unchanged=True,runtime_root_or_object_pose_writes=False,source_sha256=hashes,simulation_video_rendered=False),indent=2))

if __name__=='__main__':main()
