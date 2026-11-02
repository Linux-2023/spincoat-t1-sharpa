"""Actual right-hand tube pick/shake/pour/return prototype. No hand weld/pose writes.

Only actuator targets and robot-joint gravity compensation drive mechanics.
Visible fluid is a pose-gated conservative proxy. Actual geometric held gate
prevents an empty hand from proceeding to mixing/pouring.
"""
from pathlib import Path
import json,numpy as np,mujoco
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import ROOT,arm_indices,hand_indices
from probe_t1_reagent_reach import solve_arm
from t1_reagent_preparation import PreparationConfig,PreparationLedger

def smooth(u):
 u=float(np.clip(u,0,1));return u*u*u*(10+u*(-15+6*u))

def advance_reagent_preparation(m,d,dock,handles,out,*,plan=None,return_to_ready_joint_target=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if plan is None:raise ValueError('Explicit tube grasp plan required; legacy plan has no opposing thumb and is not accepted')
 c=PreparationConfig(**handles['config']);ledger=PreparationLedger(c);ids,qa,ad=arm_indices(m);hj,hq,hd=hand_indices(m);armacts=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in ids]);hacts=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in hj]);wheel=np.array([a for a in range(m.nu) if m.actuator(a).name.startswith('astribot_chassis_motor_')]);robotacts=np.array([a for a in range(m.nu) if a not in wheel and m.joint(int(m.actuator_trnid[a,0])).name.startswith(('astribot_','sharpa_'))]);robotdofs=np.unique(m.jnt_dofadr[m.actuator_trnid[robotacts,0]]);base=int(m.joint('chassis_free').dofadr[0]);baseqa=int(m.joint('chassis_free').qposadr[0]);body=m.body(handles['tube_body']).id;palm=m.body('sharpa_right/right_hand_C_MC').id;mouth=m.site(handles['tube_mouth_site']).id;mujoco.mj_forward(m,d);initial_ctrl=d.ctrl.copy();initialq=d.qpos.copy()
 for a in robotacts:initial_ctrl[a]=initialq[int(m.jnt_qposadr[int(m.actuator_trnid[a,0])])]
 d.ctrl[robotacts]=initial_ctrl[robotacts]
 ready_target=np.asarray(return_to_ready_joint_target) if return_to_ready_joint_target is not None else initialq.copy();tube_initial=d.xpos[body].copy();start=float(d.time);rec=[(float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),'prep_initial_unheld',ledger.tube_volume_m3,ledger.receiver_volume_m3,0.,d.site_xpos[mouth].copy())];full=[];events=[];failure=None;ik_audit=[];closed=np.array(plan['close_command_hand_q_rad']);openq=closed.copy()
 for i,j in enumerate(hj):
  if 'thumb' not in m.joint(j).name:openq[i]*=.40
  else:openq[i]=initialq[hq[i]]
 # Open thumb uses initial ready pose; closed thumb is supplied by explicit tube plan.
 (out/'scene_handles.json').write_text(json.dumps({k:v for k,v in handles.items() if k!='child_spec'},indent=2))
 (out/'preparation_config.json').write_text(json.dumps(handles['config'],indent=2))
 np.savez_compressed(out/'return_to_ready_joint_target.npz',qpos=ready_target,ctrl_targets=initial_ctrl)
 armcmd=d.ctrl[armacts].copy();handcmd=d.ctrl[hacts].copy();held_reference=[None];environment=[];held_gate=None;fine_started=[None];pour_entry_gate=None
 def sample(phase,pour=False,mix=False):
  nonlocal failure
  d.ctrl[robotacts]=initial_ctrl[robotacts];d.ctrl[armacts]=armcmd;d.ctrl[hacts]=handcmd;dock.step(d);d.qfrc_applied[:]=0;d.qfrc_applied[robotdofs]=d.qfrc_bias[robotdofs];mujoco.mj_step(m,d);obs=ledger.observe(m,d,handles,allow_pour=pour,allow_mix=mix);counts=np.zeros(5,int);forces=np.zeros(5);digit_names=['thumb','index','middle','ring','pinky']
  for ci,contact in enumerate(d.contact):
   names=[m.body(int(m.geom_bodyid[int(g)])).name for g in contact.geom]
   if handles['tube_body'] in names:
    f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
    for i,name in enumerate(digit_names):
     if any(n.startswith('sharpa_right/right_'+name+'_') for n in names):counts[i]+=1;forces[i]+=np.linalg.norm(f[:3])
   if any(n.startswith('sharpa_right/') for n in names) and any(n.startswith(('prep/tube_holder','prep/receiver_nest','aspiration_source','stand/','spincoater/','lab_bench','pipette/')) for n in names):
    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench);force=float(np.linalg.norm(wrench[:3]))
    if force>.1:environment.append(dict(time_s=float(d.time),phase=phase,bodies=names,geoms=[m.geom(int(g)).name for g in contact.geom],depth_mm=float(-contact.dist*1000),contact_force_N=force,normal_force_N=float(wrench[0])))
  R=d.xmat[palm].reshape(3,3);rel=R.T@(d.xpos[body]-d.xpos[palm]);rot=R.T@d.xmat[body].reshape(3,3);speed=float(np.linalg.norm(d.qvel[base:base+2]));yaw=float(abs(d.qvel[base+5]));workspace=float(np.linalg.norm(d.qpos[baseqa:baseqa+2]-dock.station_xy));full.append((float(d.time),phase,counts.copy(),forces.copy(),rel.copy(),rot.copy(),speed,yaw,workspace,d.ctrl[wheel].copy(),d.xpos[body].copy(),d.site_xpos[mouth].copy(),obs['actual_tilt_deg'],ledger.tube_volume_m3,ledger.receiver_volume_m3,obs['flow_m3']))
  if len(full)%20==0:rec.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),phase,ledger.tube_volume_m3,ledger.receiver_volume_m3,obs['flow_m3'],d.site_xpos[mouth].copy()))
  return counts,forces
 def phase(name,duration,goalarm=None,goalhand=None,pour=False,mix=False):
  nonlocal armcmd,handcmd
  events.append(dict(time_s=float(d.time),phase=name));q0=armcmd.copy();h0=handcmd.copy();feedback_anchor=armcmd.copy();feedback_goal=armcmd.copy()
  for k in range(int(round(duration/m.opt.timestep))):
   u=smooth((k+1)*m.opt.timestep/duration)
   if goalarm is not None:armcmd=(1-u)*q0+u*np.asarray(goalarm)
   if goalhand is not None:handcmd=(1-u)*h0+u*np.asarray(goalhand)
   if name=='pour_hold':
    if k%200==0:
     Rh=d.xmat[palm].reshape(3,3);local_mouth=Rh.T@(d.site_xpos[mouth]-d.xpos[palm]);local_body=Rh.T@d.xmat[body].reshape(3,3);desiredR=Rotation.from_euler('x',100,degrees=True).as_matrix()@local_body.T;feedback_anchor=armcmd.copy();feedback_goal=goal(pourmouth-desiredR@local_mouth,desiredR)
    blend=smooth((k%200+1)/200);armcmd=(1-blend)*feedback_anchor+blend*feedback_goal
   sample(name,pour,mix)
 def goal(point,R,seed=None):
  q,err=solve_arm(m,d,np.asarray(point),np.asarray(R),side='right',seed=armcmd if seed is None else seed);ik_audit.append(dict(time_s=float(d.time),point_world_m=np.asarray(point).tolist(),**err))
  if err['position_error_mm']>1. or err['orientation_error_deg']>1.:raise RuntimeError('prep arm IK failed '+str(err))
  return q
 try:
  events.append(dict(time_s=float(d.time),phase='coarse_dock_before_prep'))
  for k in range(12000):
   sample('coarse_dock_before_prep')
   if dock.stopped:break
  if not dock.stopped:raise RuntimeError('Base did not stop before prep')
  fine_started[0]=float(d.time);phase('open_hand',.6,goalhand=openq);reach=plan['reach'];R=np.asarray(reach['pickup_palm_R_world']);pick=tube_initial+np.array([0,0,c.tube_grasp_height_m])-R@np.asarray(reach['mirrored_contact_offset_palm_m']);phase('approach_above_tube',3.,goal(pick+[0,0,.15],R));phase('lower_to_tube',2.5,goal(pick,R));phase('close_on_tube',3.,goalhand=closed);phase('hold_before_tube_lift',1.);
  if environment:raise RuntimeError('Unexpected force-bearing hand/environment contact before tube lift')
  beforelift=d.xpos[body].copy();phase('lift_tube',3.,goal(pick+[0,0,.10],R));window=full[-500:];allfour=all((r[2][1:]>0).all() and (r[3][1:]>.01).all() for r in window);opposed=all(r[2][0]>0 and r[3][0]>.01 and int(((r[2][1:]>0)&(r[3][1:]>.01)).sum())>=2 for r in window);positions=np.array([r[4] for r in window]);rotations=np.array([r[5] for r in window]);trans=float(np.linalg.norm(positions-positions[0],axis=1).max());angle=float(np.degrees(Rotation.from_matrix(np.einsum('ij,njk->nik',rotations[0].T,rotations)).magnitude()).max());stable=trans<.002 and angle<3.;lift=float(d.xpos[body,2]-tube_initial[2]);held_gate=dict(tube_lift_mm=lift*1000,required_lift_mm=60.,last_halfsecond_samples=len(window),all_four_contact_force_last_halfsecond=bool(allfour),opposing_thumb_plus_two_non_thumb_last_halfsecond=bool(opposed),relative_translation_last_halfsecond_mm=trans*1000,relative_rotation_last_halfsecond_deg=angle,criteria='Actual lift>=60mm; thumb+>=2 nonthumb contact and force>.01N continuously0.5s; relative position change<2mm/full rotation<3deg over same window. Prototype task thresholds, not pipette4finger acceptance.',pass_=bool(lift>=.06 and opposed and stable));(out/'actual_tube_held_gate.json').write_text(json.dumps(held_gate,indent=2))
  if not held_gate['pass_']:raise RuntimeError('actual tube held/lift gate failed')
  # Gentle actual orientation oscillation, not an asserted chemical mixing result.
  center=d.xpos[palm].copy();Rbase=d.xmat[palm].reshape(3,3).copy();plus=goal(center,Rbase@Rotation.from_euler('x',5,degrees=True).as_matrix());minus=goal(center,Rbase@Rotation.from_euler('x',-5,degrees=True).as_matrix());events.append(dict(time_s=float(d.time),phase='gentle_shake'))
  for k in range(2400):
   u=(k+1)*.001/2.4;amplitude=np.sin(np.pi*u)**2;wave=amplitude*np.sin(2*np.pi*2*u*2.4);armcmd=(plus+minus)/2+wave*(plus-minus)/2;sample('gentle_shake',mix=True)
  # Capture actual grasp transform only as an IK target, never force the object.
  Rh=d.xmat[palm].reshape(3,3);mouth_local=Rh.T@(d.site_xpos[mouth]-d.xpos[palm]);bodyRrelative=Rh.T@d.xmat[body].reshape(3,3);upright_hand=bodyRrelative.T;pourmouth=np.array([*c.receiver_xy_m,c.pour_mouth_height_m]);prepour=np.array([*c.receiver_xy_m,c.prepour_mouth_height_m]);phase('carry_above_receiver',3.,goal(prepour-upright_hand@mouth_local,upright_hand))
  events.append(dict(time_s=float(d.time),phase='settle_before_precise_pour'))
  for k in range(4000):
   sample('settle_before_precise_pour')
   if dock.stopped:break
  pour_entry_gate=dict(time_s=float(d.time),passed=bool(dock.stopped),actual_speed_m_s=float(np.linalg.norm(d.qvel[base:base+2])),actual_yaw_rate_rad_s=float(abs(d.qvel[base+5])))
  if not dock.stopped:raise RuntimeError('Actual base did not settle before precision pour')
  for angle in [25,50,75,100]:
   turn=Rotation.from_euler('x',angle,degrees=True).as_matrix();Rp=turn@upright_hand;mp=prepour+(pourmouth-prepour)*angle/100;phase('tilt_tube_'+str(angle),1.,goal(mp-Rp@mouth_local,Rp),pour=True)
  phase('pour_hold',5.,pour=True)
  if ledger.delivered_m3<c.target_pour_volume_m3*.99:raise RuntimeError('actual pose-gated pour did not deliver target')
  # Lift the tilted tube clear of the rim before rotating back upright.
  phase('raise_tube_after_pour',1.2,goal(prepour-Rotation.from_euler('x',100,degrees=True).as_matrix()@upright_hand@mouth_local,Rotation.from_euler('x',100,degrees=True).as_matrix()@upright_hand))
  for angle in [75,50,25,0]:
   Rp=Rotation.from_euler('x',angle,degrees=True).as_matrix()@upright_hand;phase('untilt_tube_'+str(angle),.6,goal(prepour-Rp@mouth_local,Rp))
  # Place tube on the original holder using measured current grip transform.
  Rh=d.xmat[palm].reshape(3,3);root_local=Rh.T@(d.xpos[body]-d.xpos[palm]);bodyRrelative=Rh.T@d.xmat[body].reshape(3,3);returnR=bodyRrelative.T;phase('return_above_holder',3.,goal(tube_initial+[0,0,.10]-returnR@root_local,returnR));phase('lower_into_holder',2.5,goal(tube_initial+[0,0,.001]-returnR@root_local,returnR));phase('release_tube',2.,goalhand=openq);phase('withdraw_empty_hand',2.,goal(tube_initial+[0,0,.15]-returnR@root_local,returnR));phase('return_to_original_ready_joints',3.,goalarm=ready_target[qa],goalhand=ready_target[hq]);phase('settle_original_ready',.7)
 except RuntimeError as exc:failure=str(exc)
 rec.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),'prep_end',ledger.tube_volume_m3,ledger.receiver_volume_m3,0.,d.site_xpos[mouth].copy()));arrays={name:np.array([r[i] for r in rec]) for i,name in enumerate(['time','qpos','qvel','ctrl','phase','tube_volume_m3','receiver_volume_m3','flow_volume_m3','tube_mouth_world_m'])};arrays['flow_volume_m3']=np.asarray(arrays['flow_volume_m3']);fullarrays={name:np.array([r[i] for r in full]) for i,name in enumerate(['time','phase','finger_contacts','finger_force_magnitude_N','tube_relative_position_in_palm_m','tube_relative_rotation_in_palm','base_speed_m_s','base_yaw_rate_rad_s','workspace_error_m','wheel_command','tube_position_world_m','tube_mouth_world_m','tube_tilt_deg','tube_volume_m3','receiver_volume_m3','flow_volume_m3'])};np.savez_compressed(out/'fullrate_preparation_observer.npz',**fullarrays);fine=fullarrays['time']>=fine_started[0] if fine_started[0] is not None else np.zeros(len(full),bool);bad=fine&((fullarrays['base_speed_m_s']>=.001)|(fullarrays['base_yaw_rate_rad_s']>=.002)|(fullarrays['workspace_error_m']>.01));wheel_zero_after_stop=bool(fine.any() and np.max(abs(fullarrays['wheel_command'][fine]))<1e-12 and dock.mode=='zero_speed_hold');tube_return_error=float(np.linalg.norm(d.xpos[body]-tube_initial));report=dict(success=bool(failure is None and wheel_zero_after_stop and not environment and ledger.delivered_m3>=c.target_pour_volume_m3*.99 and tube_return_error<.005),functional_success=bool(failure is None),failure=failure,entry_time_s=start,end_time_s=float(d.time),duration_s=float(d.time-start),held_gate=held_gate,pour_entry_gate=pour_entry_gate,zero_wheel_commands_after_initial_stop=wheel_zero_after_stop,base_acceptance_scope='Preparation stage: zero wheel commands/no world PI after initial stop; passive manipulation reaction fully reported. Original physical stop required again at precision pour entry. Existing pipette gates unchanged.',base_fine_violations_1ms=int(bad.sum()),base_max_speed_mm_s=float(fullarrays['base_speed_m_s'][fine].max()*1000) if fine.any() else None,events=events,ik_audit=ik_audit,environment_events=environment,ledger=ledger.audit(),tube_return_position_error_mm=tube_return_error*1000,return_to_ready_max_arm_joint_error_rad=float(abs(d.qpos[qa]-ready_target[qa]).max()),runtime_body_or_qpos_assignments=False,no_hand_weld=True,source_geometry_available=ledger.receiver_volume_m3>0,scope='Physical actuator-driven free-tube manipulation; conservative pose-gated fluid visual proxy. No chemical homogeneity, variable mass, or resolved liquid CFD.');(out/'reagent_preparation_report.json').write_text(json.dumps(report,indent=2));return arrays,report,ledger






