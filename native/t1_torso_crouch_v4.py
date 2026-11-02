"""Actuator-only four-torso crouch overlay. No live pose/model/gain writes."""
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np,mujoco,json

@dataclass(frozen=True)
class CrouchConfig:
 target_joint_rad:tuple=(.5,-1.,.5,0.)
 down_phase:str='crouch_before_pickup'
 up_phase:str='stand_after_preparation'
 duration_s:float=4.

def smooth(u):
 u=float(np.clip(u,0,1));return u*u*u*(10+u*(-15+6*u))

class TorsoCrouchOverlay:
 def __init__(self,m,config=None):
  self.config=config or CrouchConfig();self.js=np.array([m.joint('astribot_torso_joint_'+str(k)).id for k in range(1,5)]);self.qa=m.jnt_qposadr[self.js];self.va=m.jnt_dofadr[self.js];self.acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in self.js]);self.target=np.asarray(self.config.target_joint_rad,dtype=float)
  if np.any(self.target<m.jnt_range[self.js,0]+.001) or np.any(self.target>m.jnt_range[self.js,1]-.001):raise ValueError('Crouch target lacks joint planning margin')
  self.start=None;self.anchor=None;self.reference=None;self.mode='standing';self.rows=[];self.last_phase=None
  self.baseqa=int(m.joint('chassis_free').qposadr[0]);self.baseva=int(m.joint('chassis_free').dofadr[0]);self.top=m.body('astribot_torso_link_4').id
 def __call__(self,m,d,phase):
  if self.reference is None:self.reference=d.qpos[self.qa].copy()
  if phase==self.config.down_phase and self.last_phase!=phase:self.mode='lowering';self.start=float(d.time);self.anchor=d.qpos[self.qa].copy()
  if phase==self.config.up_phase and self.last_phase!=phase:self.mode='rising';self.start=float(d.time);self.anchor=d.qpos[self.qa].copy()
  if self.mode=='lowering':
   u=smooth((float(d.time)-self.start+m.opt.timestep)/self.config.duration_s);q=(1-u)*self.anchor+u*self.target
   if u>=1:self.mode='crouched'
  elif self.mode=='crouched':q=self.target.copy()
  elif self.mode=='rising':
   u=smooth((float(d.time)-self.start+m.opt.timestep)/self.config.duration_s);q=(1-u)*self.anchor+u*self.reference
   if u>=1:self.mode='standing'
  else:q=self.reference.copy()
  d.ctrl[self.acts]=q
  self.rows.append((float(d.time),str(phase),q.copy(),d.qpos[self.qa].copy(),d.qvel[self.va].copy(),d.xpos[self.top].copy(),d.qpos[self.baseqa:self.baseqa+2].copy(),d.qvel[self.baseva:self.baseva+6].copy()))
  self.last_phase=phase
 def save(self,out):
  out=Path(out);out.mkdir(exist_ok=True,parents=True);names=['time','phase','target_joint_rad','actual_joint_rad','actual_joint_velocity_rad_s','upper_torso_world_m','base_xy_m','base_velocity'];a={k:np.array([r[j] for r in self.rows]) for j,k in enumerate(names)};np.savez_compressed(out/'torso_crouch_actual.npz',**a)
  r=dict(scope=__doc__,config=asdict(self.config),samples=len(self.rows),mode=self.mode,reference_joint_rad=self.reference.tolist(),only_actuator_targets_written=True,runtime_qpos_model_or_force_changes=False)
  (out/'torso_crouch_overlay.json').write_text(json.dumps(r,indent=2));return r