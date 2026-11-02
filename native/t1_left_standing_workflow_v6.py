"""Opt-in canonical v6 LEFT: standing lean, fixed prelift integral, own-live grasp retarget.
No held restore, object pose write, force/gain/friction/acceptance change.
"""
from pathlib import Path
import copy,json
import numpy as np,mujoco
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices

def _work_pose(m,w,qa,palm,q,P,R):
 w.qpos[qa]=q;mujoco.mj_kinematics(m,w)
 return np.r_[1000*(w.xpos[palm]-P),100*Rotation.from_matrix(w.xmat[palm].reshape(3,3)@R.T).as_rotvec()]

def _native(m,w,palm,tube,tq,rel,E):
 H=w.xmat[palm].reshape(3,3);w.qpos[tq:tq+3]=w.xpos[palm]+H@rel;quat=np.empty(4);mujoco.mju_mat2Quat(quat,(H@E).ravel());w.qpos[tq+3:tq+7]=quat;mujoco.mj_kinematics(m,w);mujoco.mj_comPos(m,w);mujoco.mj_collision(m,w);bad=[]
 for c in w.contact:
  if c.dist>=0:continue
  names=[m.body(int(m.geom_bodyid[int(g)])).name for g in c.geom];gs=[m.geom(int(g)).name or '' for g in c.geom];left=any(n.startswith(('sharpa_left/','astribot_arm_left_')) for n in names);held=tube in names;env=any(n.startswith(('prep/','lab_bench','stand/','spincoater/','aspiration_source','pipette/')) and n!=tube for n in names);digits=[next((v for v in ['thumb','index','middle','ring','pinky'] if '/left_'+v+'_' in n),None) for n in names]
  if ((left or held) and env) or (all(v is not None for v in digits) and digits[0]!=digits[1]):bad.append(dict(geoms=gs,depth_mm=-float(c.dist)*1000))
  if held and left and (not any('elastomer' in n for n in gs) or -float(c.dist)>.0011):bad.append(dict(geoms=gs,kind='nonpad_or_overwall',depth_mm=-float(c.dist)*1000))
 return bad

def retarget_from_live(m,d,profile):
 """One explicit GT-geometry planning reference after own mixing/fold; live state untouched."""
 p=copy.deepcopy(profile);ids,qa,_=arm_indices(m,'left');w=mujoco.MjData(m);w.qpos[:]=d.qpos;mujoco.mj_kinematics(m,w);palm=m.body('sharpa_left/left_hand_C_MC').id;tube=p.get('tube_body','prep/reagent_tube');tb=m.body(tube).id;tq=int(m.joint('prep/reagent_tube_free').qposadr[0]);mouth=m.site(p.get('tube_mouth_site','prep/tube_mouth')).id;H=w.xmat[palm].reshape(3,3).copy();E=H.T@w.xmat[tb].reshape(3,3);ml=H.T@(w.site_xpos[mouth]-w.xpos[palm]);rel=H.T@(w.xpos[tb]-w.xpos[palm]);oldE=np.array(p.get('measured_hand_to_tube_R',E));oldml=np.array(p.get('measured_mouth_in_hand_m',ml));old_reference_available=('measured_hand_to_tube_R' in p and 'measured_mouth_in_hand_m' in p);az=float(p['tilt_axis_azimuth_deg']);axis=np.r_[np.cos(np.radians(az)),np.sin(np.radians(az)),0.];xyz=np.array(p['mouth_xyz_waypoints_m']);curve=PchipInterpolator(xyz[:,0],xyz[:,1:],axis=0);nodes=np.array(p['joint_reference_keyframes_rad']);qref=PchipInterpolator(nodes[:,0],nodes[:,1:],axis=0);lo=m.jnt_range[ids,0]+.0501;hi=m.jnt_range[ids,1]-.0501;maximum=float(p['max_tilt_deg']);q=np.asarray(qref(maximum));rows=[]
 el=m.body('astribot_arm_left_link_5').id;wr=m.body('astribot_arm_left_link_6').id;mc=m.body('sharpa_left/left_middle_MCP_VL').id
 def wrist():
  a=w.xpos[wr]-w.xpos[el];b=w.xpos[mc]-w.xpos[palm];return float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
 for angle in np.linspace(maximum,0,int(round(maximum))+1):
  u=float(np.clip(angle/50.,0,1));v=float(np.clip((angle-50.)/max(maximum-50.,1e-9),0,1));gamma=20*u**3*(10+u*(-15+6*u))+(float(p.get('v6_axial_yaw_end_deg',20.))-20)*v**3*(10+v*(-15+6*v));T=Rotation.from_rotvec(np.radians(angle)*axis).as_matrix()@Rotation.from_euler('z',gamma,degrees=True).as_matrix();R=T@E.T;P=curve(angle)-R@ml;prior=q.copy();lower=np.maximum(lo,q-.15);upper=np.minimum(hi,q+.15)
  def pose(v):return _work_pose(m,w,qa,palm,v,P,R)
  def objective(v):pose(v);return float(1-np.cos(np.radians(wrist()))+.01*np.sum((v-prior)**2))
  fit=minimize(objective,q,method='SLSQP',bounds=list(zip(lower,upper)),constraints=dict(type='eq',fun=pose),options=dict(maxiter=180,ftol=1e-10));q=fit.x;e=pose(q);wrdeg=wrist();bad=_native(m,w,palm,tube,tq,rel,E)
  row=dict(angle_deg=float(angle),arm_q_rad=q.tolist(),position_error_mm=float(np.linalg.norm(e[:3])),orientation_error_deg=float(np.degrees(np.linalg.norm(e[3:])/100)),wrist_deg=wrdeg,bad=bad);rows.append(row)
  if row['position_error_mm']>.1 or row['orientation_error_deg']>.01 or wrdeg>50 or bad:raise RuntimeError('Own-live v6 pour retarget rejected '+json.dumps(row))
 p['joint_reference_keyframes_rad']=[[r['angle_deg'],*r['arm_q_rad']] for r in rows[::-1]];p['measured_hand_to_tube_R']=E.tolist();p['measured_mouth_in_hand_m']=ml.tolist();p['source_state_time_s']=float(d.time);p['source_state_path']='own same-live actual postmix and idlefold, not archived checkpoint';p['dynamic_status']='PLANNED_FROM_OWN_LIVE_STATE_NOT_YET_EXECUTED'
 report=dict(scope=retarget_from_live.__doc__,actual_reference_time_s=float(d.time),reference_is_privileged_simulation_geometry=True,no_live_pose_or_qpos_writes=True,static_reference_comparison_available=old_reference_available,grasp_rotation_change_from_static_deg=float(np.degrees(Rotation.from_matrix(E@oldE.T).magnitude())),mouth_in_hand_change_from_static_mm=float(np.linalg.norm(ml-oldml)*1000),node_count=len(rows),wrist_max_deg=max(r['wrist_deg'] for r in rows),nodes=rows,geometry_scope='Enabled native at knots on fresh compiled scene, no full-visual or between-knot certificate; actual raw requires audit')
 return p,report

def make_standing_transport(profile,original):
 """Own-state C2 joint entry; root return route retains original actuator transport."""
 def advance(m,d,arm_command,palm_waypoints_world_m,target_palm_R_world,step_callback,*,phase_name,config=None,duration_s=None,ik_solver=None):
  if phase_name!='coarse_carry_to_receiver':return original(m,d,arm_command,palm_waypoints_world_m,target_palm_R_world,step_callback,phase_name=phase_name,config=config,duration_s=duration_s,ik_solver=ik_solver)
  ids,qa,_=arm_indices(m,'left');acts=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in ids]);start=np.asarray(arm_command).copy();end=np.array(profile['joint_reference_keyframes_rad'][0][1:]);duration=4. if duration_s is None else float(duration_s);ts=np.arange(round(duration/m.opt.timestep)+1)*m.opt.timestep;u=ts/duration;s=u**3*(10+u*(-15+6*u));commands=start[None,:]+s[:,None]*(end-start)[None,:];margin=float(np.minimum(commands-m.jnt_range[ids,0],m.jnt_range[ids,1]-commands).min());speed=float(1.875*np.max(abs(end-start))/duration);cmargin=float(np.minimum(commands-m.actuator_ctrlrange[acts,0],m.actuator_ctrlrange[acts,1]-commands).min())
  if margin<.049 or speed>2 or cmargin<0:raise RuntimeError('V6 own-state coarse entry original speed/margin/range guard failed '+str((margin,speed,cmargin)))
  w=mujoco.MjData(m);w.qpos[:]=d.qpos;mujoco.mj_kinematics(m,w);pb=m.body('sharpa_left/left_hand_C_MC').id;tube='prep/reagent_tube';tb=m.body(tube).id;tq=int(m.joint('prep/reagent_tube_free').qposadr[0]);H=w.xmat[pb].reshape(3,3).copy();E=H.T@w.xmat[tb].reshape(3,3);rel=H.T@(w.xpos[tb]-w.xpos[pb]);sample_indices=np.unique(np.r_[np.arange(0,len(commands),10),len(commands)-1]);bad=[];wrist_values=[];tilt_values=[];el=m.body('astribot_arm_left_link_5').id;wr=m.body('astribot_arm_left_link_6').id;mc=m.body('sharpa_left/left_middle_MCP_VL').id
  for k in sample_indices:
   w.qpos[qa]=commands[k];mujoco.mj_kinematics(m,w);items=_native(m,w,pb,tube,tq,rel,E)
   a=w.xpos[wr]-w.xpos[el];b=w.xpos[mc]-w.xpos[pb];wrist=float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))));tilt=float(np.degrees(np.arccos(np.clip(w.xmat[tb].reshape(3,3)[2,2],-1,1))));wrist_values.append(wrist);tilt_values.append(tilt)
   if wrist>50 or tilt>60:items.append(dict(kind='entry_wrist_or_upright_guard',wrist_deg=wrist,tube_tilt_deg=tilt))
   if items:bad.append(dict(t=float(ts[k]),pairs=items))
  if bad:raise RuntimeError('V6 own-state entry native hard geometry rejected '+json.dumps(bad[:3]))
  t0=float(d.time)
  for q in commands[1:]:
   before=float(d.time);step_callback(q,phase_name)
   if abs(float(d.time)-before-m.opt.timestep)>1e-9:raise RuntimeError('V6 caller did not advance exactly one native step')
  return end.copy(),dict(scope=make_standing_transport.__doc__,execution_start_s=t0,execution_end_s=float(d.time),duration_s=duration,maximum_command_speed_rad_s=speed,command_joint_margin_rad=margin,actuator_control_margin_rad=cmargin,planning_native_samples=len(sample_indices),planned_wrist_max_deg=max(wrist_values),planned_tube_tilt_max_deg=max(tilt_values),entry_tube_upright_limit_deg=60.,original_world_waypoints_used=False,reference='Own-live current actuator target to own-live retargeted firstq; no old03 pose/certificate guard',geometry_scope='10ms enabled native conditional rigid-grip screen, not CCD/fullvisual or actual grip success'),dict(time=ts,arm_commands=commands,geometric_arm_q=commands,reference_semantics=np.array('Own-live C2 joint reference'))
 return advance

def adapt_generated_stage_source(source):
 from t1_left_standing_body_v6 import adapt_v6_stage_source
 source,_=adapt_v6_stage_source(source)
 source=source.replace('V6_STANDING_BODY_CONFIG','V4_BODY_CONFIG')
 def once(old,new):
  nonlocal source
  if source.count(old)!=1:raise RuntimeError('V6 LEFT generated anchor mismatch '+repr(old[:110])+' '+str(source.count(old)))
  source=source.replace(old,new)
 once("phase('lift_tube',", "events.append(dict(time_s=float(d.time),phase='fixed_prelift_encoder_freeze_v6',offset_rad=FREEZE_LEFT_ENCODER().tolist(),scope='Fixed phase boundary, no tactile input; retain current I, original nominal/P/gain/cap unchanged'));phase('lift_tube',")
 # Keep a local profile so actual retarget does not mutate the frozen input JSON.
 source=source.replace('HUMAN_POUR_PROFILE','V6_RUNTIME_PROFILE')
 once(' out=Path(out);out.mkdir(parents=True,exist_ok=True)'," out=Path(out);out.mkdir(parents=True,exist_ok=True)\n import copy\n V6_RUNTIME_PROFILE=copy.deepcopy(HUMAN_POUR_PROFILE)\n if getattr(dock,'mode',None)!='approach' or len(getattr(dock,'control_records',[])):raise RuntimeError('V6 prep requires an unused coarse dock before first wheel step')\n dock.station_xy=np.array([0.,.30]);dock.target=dock.station_xy.copy()")
 marker='  coarse_receiver_mouth=prepour.copy()'
 inject="""  from t1_left_standing_workflow_v6 import retarget_from_live,make_standing_transport
  np.savez_compressed(out/'actual_mixed_folded_state_v6.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),time=float(d.time))
  V6_RUNTIME_PROFILE,v6_profile_review=retarget_from_live(m,d,V6_RUNTIME_PROFILE)
  (out/'live_standing_profile_v6.json').write_text(json.dumps(V6_RUNTIME_PROFILE,indent=2));(out/'live_standing_retarget_review_v6.json').write_text(json.dumps(v6_profile_review,indent=2))
  V6_ADVANCE_CARRY=make_standing_transport(V6_RUNTIME_PROFILE,V3_ADVANCE_COARSE)
  Rh=d.xmat[palm].reshape(3,3);mouth_local=Rh.T@(d.site_xpos[mouth]-d.xpos[palm]);bodyRrelative=Rh.T@d.xmat[body].reshape(3,3);upright_hand=v4_initial_pour_target(0.,V6_RUNTIME_PROFILE)[1]@bodyRrelative.T
"""
 once(marker,inject+marker)
 once('armcmd,coarse_report,coarse_trace=V3_ADVANCE_COARSE(m,d,armcmd,carry_points','armcmd,coarse_report,coarse_trace=V6_ADVANCE_CARRY(m,d,armcmd,carry_points')
 return source

def adapt_launch_source(source,profile=None):
 """Outer adapter for v4.transformed_source() plus root canonical scene adapter.
Preserves optional explicit profile argv; no baseline source files are edited.
"""
 if profile is not None:
  p=json.loads(Path(profile).read_text()) if not isinstance(profile,dict) else profile
  if not np.allclose(p['receiver_xy_m'],[.60,.30]) or not np.allclose(p['expected_torso_q_rad'],[0,0,np.radians(12),0]):raise ValueError('V6 standing profile/layout mismatch')
 api='from t1_human_pour_profile_v4 import advance_human_pour_v4 as advance_human_pour_v2'
 if source.count(api)==1:
  from t1_deferred_idle_restore_v5 import adapt_launch_source as defer
  source=defer(source)
 old='from t1_loaded_crouch_v4 import LoadedCrouchControlHook as CrouchControlHook';new='from t1_left_standing_body_v6 import StandingLeanControlHook as CrouchControlHook'
 if source.count(old)!=2:raise RuntimeError('V6 body hook import anchors require exactly2')
 source=source.replace(old,new)
 anchor="path=out/'executed_human_glass_stage_v4.py';path.write_text(source)"
 if source.count(anchor)!=1:raise RuntimeError('V6 final generated compile anchor requires exactly1')
 source=source.replace(anchor,"from t1_left_standing_workflow_v6 import adapt_generated_stage_source\\n        source=adapt_generated_stage_source(source)\\n        "+anchor)
 return source
