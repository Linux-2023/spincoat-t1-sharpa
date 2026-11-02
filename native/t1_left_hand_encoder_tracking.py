"""Bounded joint-encoder tracking for new LEFT glass stage only, no tactile feedback."""
import json,numpy as np
from pathlib import Path
from t1_spincoat_scene import hand_indices
class LeftHandEncoderTracker:
 def __init__(self,out):self.out=Path(out);self.initialized=False;self.rows=[];self.phase=None;self.phase_start=0.;self.integrator_frozen=False
 def freeze_integrator(self):
  """Freeze the existing integral value; never clear it or write physical state."""
  if not self.initialized:raise RuntimeError('Cannot freeze an uninitialized encoder tracker')
  self.integrator_frozen=True
  return self.offset.copy()
 def step(self,m,d,phase):
  if not self.initialized:
   hj,self.hq,_=hand_indices(m,'left');self.acts=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in hj]);self.mask=np.array([any('/left_'+dig+'_' in m.joint(j).name for dig in ['thumb','index','middle']) for j in hj]);self.offset=np.zeros(len(hj));self.previous=m.actuator_ctrlrange[self.acts].copy();self.initialized=True;self.lasttime=float(d.time)
  nominal=d.ctrl[self.acts].copy();actual=d.qpos[self.hq].copy();dt=max(0.,min(.002,float(d.time)-self.lasttime));self.lasttime=float(d.time)
  if phase!=self.phase:self.phase=phase;self.phase_start=float(d.time)
  active=not any(phase.startswith(key) for key in ['coarse_dock','open_hand','approach','lower_outside','seat_','release_tube','withdraw_empty','return_to_original','settle_original'])
  if active:
   if not self.integrator_frozen:self.offset[self.mask]=np.clip(self.offset[self.mask]+3.*dt*(nominal-actual)[self.mask],-.03,.03)
  else:self.offset*=np.exp(-dt/.15)
  commanded=np.clip(nominal+self.offset,m.actuator_ctrlrange[self.acts,0],m.actuator_ctrlrange[self.acts,1]);d.ctrl[self.acts]=commanded;self.rows.append((float(d.time),phase,nominal,actual,self.offset.copy(),commanded,bool(self.integrator_frozen)))
 def save(self):
  if not self.rows:return {}
  names=['time','phase','nominal_target_rad','actual_joint_rad','encoder_offset_rad','actuator_ctrl_rad','integrator_frozen'];a={k:np.array([r[i] for r in self.rows]) for i,k in enumerate(names)};np.savez_compressed(self.out/'left_hand_encoder_tracking_1khz.npz',**a);report=dict(scope='NewLEFTglass only, bounded position integral using joint encoders; no contact force/tactile input and no physical property changes',ki_per_s=3.,offset_bound_rad=.03,actual_offset_max_rad=float(abs(a['encoder_offset_rad']).max()),samples=len(self.rows),timestamp='Immediately before mj_step; corresponding physics observer is one timestep later',gain_forcecap_friction_changed=False,integrator_frozen_final=bool(self.integrator_frozen),integrator_frozen_samples=int(a['integrator_frozen'].sum()));(self.out/'left_hand_encoder_tracking.json').write_text(json.dumps(report,indent=2));return report
