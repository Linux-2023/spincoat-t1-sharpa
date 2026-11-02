"""Opt-in phase/clock-only left index MCP_AA offset; never reads contact."""
import hashlib,json
from pathlib import Path
import numpy as np

def c2(u):
 u=float(np.clip(u,0.,1.));return u*u*u*(10.+u*(-15.+6.*u))

class LeftIndexClearanceOverlay:
 def __init__(self,out,offset_rad,*,ramp_s=.5):
  if not np.isfinite(offset_rad) or abs(offset_rad)>.1:raise ValueError('Explicit screened offset required, absolute bound .1 rad')
  if not np.isfinite(ramp_s) or ramp_s<=0:raise ValueError('Positive C2 ramp duration required')
  self.out=Path(out);self.offset=float(offset_rad);self.ramp=float(ramp_s);self.start=None;self.release=None;self.release_weight=None;self.rows=[];self.act=None
 def __call__(self,m,d,phase):
  if self.act is None:
   names=[m.joint(i).name for i in range(m.njnt)];matches=[i for i,n in enumerate(names) if n and n.startswith('sharpa_left/') and n.endswith('left_index_MCP_AA')]
   if len(matches)!=1:raise ValueError('Ambiguous left index MCP_AA joint')
   self.joint=matches[0];acts=np.flatnonzero(m.actuator_trnid[:,0]==self.joint)
   if len(acts)!=1:raise ValueError('Ambiguous index actuator')
   self.act=int(acts[0]);self.qadr=int(m.jnt_qposadr[self.joint])
  t=float(d.time)
  if self.start is None and phase.startswith('tilt_tube_'):self.start=t
  weight=0. if self.start is None else c2((t-self.start)/self.ramp)
  if phase=='release_tube' and self.release is None:self.release=t;self.release_weight=weight
  if self.release is not None:weight=self.release_weight*(1.-c2((t-self.release)/self.ramp))
  nominal=float(d.ctrl[self.act]);target=nominal+self.offset*weight
  lo,hi=m.actuator_ctrlrange[self.act]
  if target<lo or target>hi:raise RuntimeError('Index clearance target exceeds existing actuator ctrlrange')
  jlo,jhi=m.jnt_range[self.joint]
  if target<jlo or target>jhi:raise RuntimeError('Index clearance target exceeds existing joint range')
  # This runs before the unchanged encoder tracker. Its retained I offset is
  # applied exactly as before; final applied ctrl is in that tracker trace.
  d.ctrl[self.act]=target
  self.rows.append((t,str(phase),weight,nominal,target,float(d.qpos[self.qadr])))
 def save(self):
  keys=['time','phase','weight','nominal_ctrl_rad','overlay_target_ctrl_rad','actual_joint_rad']
  arrays={k:np.array([r[i] for r in self.rows]) for i,k in enumerate(keys)}
  np.savez_compressed(self.out/'index_clearance_overlay_1khz.npz',**arrays)
  p=Path(__file__);report=dict(scope='Explicit pre-screened index MCP_AA target overlay only, phase and physical clock inputs; not contact feedback',offset_rad=self.offset,ramp_s=self.ramp,first_tilt_time_s=self.start,release_time_s=self.release,samples=len(self.rows),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),forcecap_gain_friction_or_gate_changes=False,read_only_physical_state=True,final_ctrl_trace='left_hand_encoder_tracking_1khz.npz')
  (self.out/'index_clearance_overlay.json').write_text(json.dumps(report,indent=2));(self.out/'executed_index_clearance_overlay.py').write_bytes(p.read_bytes());return report
