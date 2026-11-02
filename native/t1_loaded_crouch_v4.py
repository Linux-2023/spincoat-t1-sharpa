"""Coordinated torso height change with actuator-only LEFT world-palm hold.

Standing pickup/mixing are preserved. Body motion happens in the free corridor
before/after pouring. The IK solver modifies its own planning MjData only.
"""
from pathlib import Path
import json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from t1_torso_crouch_v4 import TorsoCrouchOverlay,CrouchConfig
from t1_spincoat_scene import arm_indices
from t1_left_loaded_continuous_neutral_v3 import solve_arm

BODY_CONFIG=CrouchConfig(target_joint_rad=(.5,-1.,.7617993877991494,0.),down_phase='crouch_before_pour',up_phase='stand_after_pour')
BODY_PHASES=(BODY_CONFIG.down_phase,'settle_crouched',BODY_CONFIG.up_phase,'settle_standing')

class LoadedCrouchControlHook:
 def __init__(self,original=None,config=None):
  self.original=original;self.config=config or BODY_CONFIG;self.overlay=None;self.anchor=None;self.q=None;self.last_update=-1.;self.active=False;self.rows=[]
 def __call__(self,m,d,phase):
  if self.original is not None:self.original(m,d,phase)
  if self.overlay is None:
   self.overlay=TorsoCrouchOverlay(m,self.config);self.palm=m.body('sharpa_left/left_hand_C_MC').id;js,self.qa,_=arm_indices(m,'left');self.acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in js])
  self.overlay(m,d,phase)
  active=phase in BODY_PHASES
  if active and not self.active:
   self.anchor=(d.xpos[self.palm].copy(),d.xmat[self.palm].reshape(3,3).copy());self.q=d.qpos[self.qa].copy();self.last_update=-1.
  if active:
   if float(d.time)-self.last_update>=.005-1e-9:
    q,report=solve_arm(m,d,self.anchor[0],self.anchor[1],side='left',seed=self.q)
    if np.max(abs(q-self.q))>.01:raise RuntimeError('Loaded body compensation IK joint increment exceeds .01rad per5ms update')
    self.q=q;self.last_update=float(d.time)
   d.ctrl[self.acts]=self.q
   pe=float(np.linalg.norm(d.xpos[self.palm]-self.anchor[0]));re=float(Rotation.from_matrix(d.xmat[self.palm].reshape(3,3)@self.anchor[1].T).magnitude())
   self.rows.append((float(d.time),str(phase),pe,re,self.q.copy(),d.qpos[self.qa].copy(),self.anchor[0].copy()))
   if pe>.005 or re>np.deg2rad(2.):raise RuntimeError('Loaded body world-palm tracking departed5mm/2deg diagnostic limit')
  self.active=active
 def save(self,out):
  out=Path(out);r=self.overlay.save(out) if self.overlay else dict(samples=0)
  names=['time','phase','palm_position_error_m','palm_rotation_error_rad','arm_target_rad','arm_actual_rad','palm_anchor_world_m'];np.savez_compressed(out/'loaded_body_compensation.npz',**{k:np.array([row[i] for row in self.rows]) for i,k in enumerate(names)})
  r.update(world_palm_compensation_samples=len(self.rows),max_palm_position_error_mm=max((r[2]*1000 for r in self.rows),default=None),max_palm_orientation_error_deg=max((float(np.degrees(r[3])) for r in self.rows),default=None));return r

def adapt_loaded_body_stage_source(source):
 def once(old,new):
  nonlocal source
  if source.count(old)!=1:raise RuntimeError('Loaded body stage source anchor mismatch '+repr(old[:100]))
  source=source.replace(old,new)
 held="  if not held_gate['pass_']:raise RuntimeError('actual tube held/lift gate failed')"
 once(held,held+"\n  if plan.get('save_actual_held_checkpoint',False) or plan.get('diagnostic_stop_after_held',False):\n   np.savez_compressed(out/'actual_held_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),time=float(d.time))\n   (out/'actual_held_ledger.json').write_text(json.dumps(ledger.audit(),indent=2))\n  if plan.get('diagnostic_stop_after_held',False):raise RuntimeError('DIAGNOSTIC_STOP_AFTER_ACCEPTED_HELD_GATE; mix/pour/return not executed; not full task success')")
 routine=""" def coordinated_body_phase(name,settle_name,target):
  nonlocal armcmd
  phase(name,V4_BODY_CONFIG.duration_s)
  armcmd=d.ctrl[armacts].copy();body_qa=np.array([m.joint('astribot_torso_joint_'+str(k)).qposadr[0] for k in range(1,5)]);body_va=np.array([m.joint('astribot_torso_joint_'+str(k)).dofadr[0] for k in range(1,5)])
  stable=0.;gate_start=float(d.time)
  for body_tick in range(round(5./m.opt.timestep)):
   sample(settle_name)
   qualified=bool(dock.stopped and np.linalg.norm(d.qvel[base:base+2])<.001 and abs(d.qvel[base+5])<.002 and np.max(np.abs(d.qpos[body_qa]-target))<.005 and np.max(np.abs(d.qvel[body_va]))<.005)
   stable=stable+m.opt.timestep if qualified else 0.
   if stable>=.5:break
  else:raise RuntimeError('Coordinated loaded body failed original base and5mrad torso gate stable0.5s')
  armcmd=d.ctrl[armacts].copy();events.append(dict(time_s=float(d.time),phase=settle_name+'_gate_passed',elapsed_s=float(d.time-gate_start),scope='Actuator body plus world-palm IK compensation; original base stop thresholds unchanged'))
"""
 once(" def phase(name,duration,goalarm=None,goalhand=None,pour=False,mix=False):",routine+" def phase(name,duration,goalarm=None,goalhand=None,pour=False,mix=False):")
 marker="  # Capture actual grasp transform only as an IK target, never force the object."
 once(marker,"  coordinated_body_phase('crouch_before_pour','settle_crouched',np.asarray(V4_BODY_CONFIG.target_joint_rad))\n  from t1_human_pour_profile_v4 import target as v4_initial_pour_target\n"+marker)
 once("upright_hand=Rotation.from_euler('z',pour_profile['transport_yaw_deg'],degrees=True).as_matrix()@bodyRrelative.T", "upright_hand=v4_initial_pour_target(0.,HUMAN_POUR_PROFILE)[1]@bodyRrelative.T")
 marker="  clear_lift=rack_clearance_lift_m(plan,float(d.xpos[body,2]))"
 # Only the post-pour occurrence remains in v3 after pre-mix replacement.
 once(marker,"  body_qa=np.array([m.joint('astribot_torso_joint_'+str(k)).qposadr[0] for k in range(1,5)])\n  coordinated_body_phase('stand_after_pour','settle_standing',initialq[body_qa])\n"+marker)
 marker="fine=fullarrays['time']>=fine_started[0] if fine_started[0] is not None else np.zeros(len(full),bool)"
 once(marker,marker+";fine &= ~np.isin(fullarrays['phase'],['crouch_before_pour','settle_crouched','stand_after_pour','settle_standing'])")
 return source,dict(V4_BODY_CONFIG=BODY_CONFIG)

def adapt_loaded_coarse_entry_source(source):
 """Apply only after the v3 coarse-carry adapter has generated this block."""
 marker="   coarse_height=np.asarray(HUMAN_POUR_PROFILE['mouth_height_waypoints_m']);coarse_receiver_mouth[2]=np.interp(0.,coarse_height[:,0],coarse_height[:,1])"
 if source.count(marker)!=1:raise RuntimeError('V4 generated coarse entry anchor mismatch')
 return source.replace(marker,marker+"\n  from t1_human_pour_profile_v4 import mouth_position as v4_pour_mouth_position\n  coarse_receiver_mouth=v4_pour_mouth_position(0.,HUMAN_POUR_PROFILE)")
