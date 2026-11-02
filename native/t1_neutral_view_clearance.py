"""Short same-live arm-only clearance for the unchanged onboard workspace camera.

Targets are a relative +Y arm retreat from actual mobile-end palm pose. Grasp
preload, camera, wheel zero-speed hold, liquid accounting, and physics unchanged.
"""
from pathlib import Path
import json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices,hand_indices

def advance_view_clearance(m,d,dock,ledger,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if getattr(dock,'mode',None)!='zero_speed_hold':raise RuntimeError('View clearance requires coarse-latched zero-speed hold; cannot start new position correction')
 ids,qa,ad=arm_indices(m);palm=m.body('sharpa_right/right_hand_C_MC').id;pipe=m.body('pipette/pipette').id;tip=m.site('pipette/tip_site').id;button=int(m.joint('pipette/pipette_button').qposadr[0]);baseqa=int(m.joint('chassis_free').qposadr[0]);basedv=int(m.joint('chassis_free').dofadr[0]);mujoco.mj_forward(m,d);startp=d.xpos[palm].copy();R0=d.xmat[palm].reshape(3,3).copy();startq=d.qpos[qa].copy();ctrl0=d.ctrl.copy();goal=startp+np.array([0.,.10,0.]);quat=Rotation.from_matrix(R0).as_quat()[[3,0,1,2]]
 from t1_spincoat_scene import ik
 endq,err=ik(m,d,goal,quat,seed=startq,iterations=180)
 if err>.0005:raise RuntimeError('View-clearance arm IK failed: '+str(err))
 wheel=np.array([a for a in range(m.nu) if m.actuator(a).name.startswith('astribot_chassis_motor_')]);nonwheel=np.array([a for a in range(m.nu) if a not in wheel]);armact=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in ids]);nonbase=np.unique([int(m.jnt_dofadr[int(m.actuator_trnid[a,0])]) for a in nonwheel if m.actuator_trnid[a,0]>=0 and m.joint(int(m.actuator_trnid[a,0])).name.startswith(('astribot_','sharpa_'))]);relative0=R0.T@(d.xpos[pipe]-startp);relativeR0=R0.T@d.xmat[pipe].reshape(3,3);start=float(d.time);move_s=1.2;settle_s=.3;max_settle_s=.8;total=move_s+max_settle_s;rows=[];full=[];digits=['index','middle','ring','pinky'];env=[];last_relative=relative0.copy();last_rot=relativeR0.copy()
 for k in range(int(round(total/m.opt.timestep))):
  elapsed=float(d.time-start);u=np.clip(elapsed/move_s,0,1);blend=u*u*u*(10+u*(-15+6*u));d.ctrl[nonwheel]=ctrl0[nonwheel];d.ctrl[armact]=startq+blend*(endq-startq);dock.step(d)
  if getattr(dock,'mode',None)!='zero_speed_hold' or np.max(abs(d.ctrl[wheel]))>1e-12:raise RuntimeError('View clearance re-enabled base motion; refused')
  d.qfrc_applied[:]=0.;d.qfrc_applied[nonbase]=d.qfrc_bias[nonbase];mujoco.mj_step(m,d)
  if ledger is not None and d.time>ledger._last_time:ledger.update(d.time,float(d.qpos[button]),d.site_xpos[tip],allow_dispense=False)
  Rh=d.xmat[palm].reshape(3,3);relative=Rh.T@(d.xpos[pipe]-d.xpos[palm]);relativeR=Rh.T@d.xmat[pipe].reshape(3,3);last_relative=relative;last_rot=relativeR;forces=np.zeros(4);counts=np.zeros(4,int)
  for ci,c in enumerate(d.contact):
   ns=[m.body(m.geom_bodyid[int(g)]).name for g in [c.geom1,c.geom2]]
   if any(n.startswith('pipette/') for n in ns):
    f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
    for i,name in enumerate(digits):
     if any(n.startswith('sharpa_right/right_'+name+'_') for n in ns):counts[i]+=1;forces[i]+=np.linalg.norm(f[:3])
   if c.dist<0 and any(n.startswith(('sharpa_right/','astribot_arm_','astribot_head_','astribot_torso_')) for n in ns) and any(n.startswith(('stand/','spincoater/','lab_bench','aspiration_source')) for n in ns):env.append({'time_s':float(d.time),'bodies':ns,'depth_mm':float(-c.dist*1000)})
  speed=float(np.linalg.norm(d.qvel[basedv:basedv+2]));yawrate=float(abs(d.qvel[basedv+5]));drift=float(np.linalg.norm(relative-relative0));rotchange=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(relativeR@relativeR0.T).as_rotvec())));full.append([float(d.time),speed,yawrate,drift,rotchange,float(d.qpos[button]),*counts,*forces,*d.ctrl[wheel]])
  finished=elapsed>=move_s+settle_s and dock.stopped and np.linalg.norm(d.xpos[palm]-goal)<.002
  if k%20==0 or k==int(round(total/m.opt.timestep))-1 or finished:rows.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),'view_clearance_move' if elapsed<move_s else 'view_clearance_settle',d.site_xpos[tip].copy()))
  if finished:break
 full=np.asarray(full);actual={'time':np.array([r[0] for r in rows]),'qpos':np.asarray([r[1] for r in rows]),'qvel':np.asarray([r[2] for r in rows]),'ctrl':np.asarray([r[3] for r in rows]),'phase':np.array([r[4] for r in rows]),'tip':np.asarray([r[5] for r in rows])};bad=(full[:,1]>=.001)|(full[:,2]>=.002);np.savez_compressed(out/'fullrate_clearance_observer.npz',samples=full,columns=np.array(['time','base_speed_m_s','base_yaw_rate_rad_s','palm_pipe_translation_change_m','palm_pipe_rotation_change_deg','button_qpos_m',*[n+'_contacts' for n in digits],*[n+'_force_N' for n in digits],*[f'wheel_cmd_{i}' for i in range(len(wheel))]]))
 report={'scope':'Actual live arm-only camera-clearance transition; no camera/object/base pose reset; no full-chain claim from helper alone.','start_time_s':start,'end_time_s':float(d.time),'move_duration_s':move_s,'settle_minimum_s':settle_s,'settle_maximum_s':max_settle_s,'actual_settle_duration_s':float(d.time-start-move_s),'start_palm_world_m':startp.tolist(),'goal_palm_world_m':goal.tolist(),'start_arm_q_rad':startq.tolist(),'goal_arm_q_rad':endq.tolist(),'ik_error_mm':err*1000,'actual_palm_endpoint_error_mm':float(np.linalg.norm(d.xpos[palm]-goal)*1000),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ledger_observed':ledger is not None,'ledger_final':ledger.audit() if ledger is not None else None,'max_palm_pipe_translation_change_mm':float(full[:,3].max()*1000),'max_palm_pipe_rotation_change_deg':float(full[:,4].max()),'four_finger_positive_force_samples':int(np.all(full[:,10:14]>.01,axis=1).sum()),'fullrate_samples':len(full),'base_fine_violations':int(bad.sum()),'base_speed_max_mm_s':float(full[:,1].max()*1000),'base_yaw_rate_max_rad_s':float(full[:,2].max()),'final_stopped_gate':bool(dock.stopped),'wheel_cmd_max_abs':float(np.max(abs(full[:,14:]))),'enabled_robot_environment_contacts':env,'no_object_or_base_runtime_pose_writes':True,'gravity_compensation_dofs':nonbase.tolist(),'gravity_compensation_scope':'Actuated nonwheel robot joints only; no wheel, pipette freejoint, rotor, or other object force compensation','success':bool(np.linalg.norm(d.xpos[palm]-goal)<.002 and dock.stopped and not env and np.all(full[-1,10:14]>.01) and full[-1,3]<.002),'success_scope':'Transition endpoint, actual held support, original stopped gate and enabled-environment contact checks only; RGB and complete workflow checked by caller'};(out/'view_clearance_report.json').write_text(json.dumps(report,indent=2));return actual,report


