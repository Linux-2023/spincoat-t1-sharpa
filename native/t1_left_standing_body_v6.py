"""Opt-in v6: unheld straight-leg forward lean, then standing LEFT manipulation.
Source adapter requires the fully generated v5 LEFT stage. Shared defaults unchanged.
"""
import numpy as np
from t1_torso_crouch_v4 import TorsoCrouchOverlay, CrouchConfig
CONFIG=CrouchConfig(target_joint_rad=(0.,0.,float(np.radians(12.)),0.),down_phase='standing_lean_before_pickup',up_phase='straighten_after_preparation',duration_s=3.)
def standing_lean_config(lean_deg=12.,duration_s=3.):
 if not 0.<=float(lean_deg)<=20.:raise ValueError('Standing forward lean outside declared0..20degree planning range')
 return CrouchConfig(target_joint_rad=(0.,0.,float(np.radians(lean_deg)),0.),down_phase=CONFIG.down_phase,up_phase=CONFIG.up_phase,duration_s=float(duration_s))

class StandingLeanControlHook:
 def __init__(self,original=None,config=CONFIG):self.original=original;self.config=config;self.overlay=None
 def __call__(self,m,d,phase):
  if self.original is not None:self.original(m,d,phase)
  if self.overlay is None:self.overlay=TorsoCrouchOverlay(m,self.config)
  self.overlay(m,d,phase)
 def save(self,out):
  r=self.overlay.save(out) if self.overlay is not None else dict(samples=0)
  r.update(body_policy='standing straight legs, forward lean only; no loaded body transition',new_physics_or_pose_writes=False)
  return r

def adapt_v6_stage_source(source,*,restore_after_preparation=False,config=CONFIG,has_loaded_body_adapter=True):
 def once(old,new):
  nonlocal source
  if source.count(old)!=1:raise RuntimeError('V6 standing source anchor count '+str(source.count(old))+' for '+repr(old[:100]))
  source=source.replace(old,new)
 # Remove the two old held-body motions, not their observations or any gates.
 if has_loaded_body_adapter:
  once("  coordinated_body_phase('crouch_before_pour','settle_crouched',np.asarray(V4_BODY_CONFIG.target_joint_rad))", "  # V6: torso already in standing forward lean before pickup; no loaded crouch.")
 if has_loaded_body_adapter:
  once("  coordinated_body_phase('stand_after_pour','settle_standing',initialq[body_qa])", "  # V6: remain in standing lean through return and empty-hand withdrawal.")
 marker="  fine_started[0]=float(d.time);phase('open_hand',.6,goalhand=openq);"
 routine="""  body_qa=np.array([m.joint('astribot_torso_joint_'+str(k)).qposadr[0] for k in range(1,5)]);body_va=np.array([m.joint('astribot_torso_joint_'+str(k)).dofadr[0] for k in range(1,5)])
  phase('standing_lean_before_pickup',V6_STANDING_BODY_CONFIG.duration_s)
  body_stable=0.
  for body_tick in range(round(5./m.opt.timestep)):
   sample('settle_standing_lean')
   qualified=bool(dock.stopped and np.linalg.norm(d.qvel[base:base+2])<.001 and abs(d.qvel[base+5])<.002 and np.max(np.abs(d.qpos[body_qa]-np.asarray(V6_STANDING_BODY_CONFIG.target_joint_rad)))<.005 and np.max(np.abs(d.qvel[body_va]))<.005)
   body_stable=body_stable+m.opt.timestep if qualified else 0.
   if body_stable>=.5:break
  else:raise RuntimeError('V6 standing lean failed original stopped base and torso5mrad stability gate')
  events.append(dict(time_s=float(d.time),phase='standing_lean_gate_passed',body_target_rad=list(V6_STANDING_BODY_CONFIG.target_joint_rad),scope='Unheld straight-leg standing reach; no squatting or object pose change'))
"""
 once(marker,routine+marker)
 marker="phase('settle_original_ready',.7)"
 tail="""phase('settle_original_ready',.7)
  phase('straighten_after_preparation',V6_STANDING_BODY_CONFIG.duration_s)
  body_stable=0.
  for body_tick in range(round(5./m.opt.timestep)):
   sample('settle_straightened')
   qualified=bool(dock.stopped and np.linalg.norm(d.qvel[base:base+2])<.001 and abs(d.qvel[base+5])<.002 and np.max(np.abs(d.qpos[body_qa]-initialq[body_qa]))<.005 and np.max(np.abs(d.qvel[body_va]))<.005)
   body_stable=body_stable+m.opt.timestep if qualified else 0.
   if body_stable>=.5:break
  else:raise RuntimeError('V6 empty-hand straightening failed original stopped base and torso5mrad stability gate')
"""
 once(marker,tail if restore_after_preparation else marker)
 marker="fine=fullarrays['time']>=fine_started[0] if fine_started[0] is not None else np.zeros(len(full),bool)"
 once(marker,marker+";fine &= ~np.isin(fullarrays['phase'],['straighten_after_preparation','settle_straightened'])")
 return source,dict(V6_STANDING_BODY_CONFIG=config,V6_RESTORE_AFTER_PREPARATION=bool(restore_after_preparation))
