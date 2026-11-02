"""Asset-only gravity settle provenance, applied once BEFORE episode t=0."""
import json,numpy as np,mujoco
from pathlib import Path

def initialize_naturally_settled_tube(m,d,handles,out,duration_s=15.,*,initial_tilt_deg=0.,tilt_axis_world=(1.,0.,0.)):
 if abs(float(d.time))>1e-12:raise ValueError('Natural tube initialization is RESET-only')
 if not np.isfinite(duration_s) or duration_s<.5:raise ValueError('Settling requires a finite duration and0.5s stability window')
 axis=np.asarray(tilt_axis_world,dtype=float)
 if axis.shape!=(3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis)<1e-12:raise ValueError('A finite nonzero tilt axis is required')
 if not np.isfinite(initial_tilt_deg) or abs(initial_tilt_deg)>5.:raise ValueError('RESET tilt assumption must be within5degrees')
 out=Path(out);j=m.joint('prep/reagent_tube_free');qa=int(j.qposadr[0]);va=int(j.dofadr[0]);w=mujoco.MjData(m);mujoco.mj_copyData(w,m,d);initialq=d.qpos.copy();rows=[]
 out.mkdir(parents=True,exist_ok=True)
 perturb=np.empty(4);tilted=np.empty(4);mujoco.mju_axisAngle2Quat(perturb,axis/np.linalg.norm(axis),np.deg2rad(initial_tilt_deg));mujoco.mju_mulQuat(tilted,perturb,w.qpos[qa+3:qa+7].copy());w.qpos[qa+3:qa+7]=tilted;perturbed=w.qpos[qa:qa+7].copy();tail=[];steps=round(duration_s/m.opt.timestep)
 # Separate asset test fixes non-target DOFs, not a robot execution trajectory.
 qm=np.ones(m.nq,bool);qm[qa:qa+7]=False;vm=np.ones(m.nv,bool);vm[va:va+6]=False
 for k in range(steps):
  w.qpos[qm]=initialq[qm];w.qvel[vm]=0.;w.qfrc_applied[:]=0.;mujoco.mj_forward(m,w);mujoco.mj_step(m,w)
  if k%20==0:rows.append((float(w.time),w.qpos[qa:qa+7].copy(),w.qvel[va:va+6].copy()))
  if k>=steps-round(.5/m.opt.timestep):tail.append(w.qvel[va:va+6].copy())
 velocity=np.asarray(tail);linear=float(np.max(np.linalg.norm(velocity[:,:3],axis=1)));angular=float(np.max(np.linalg.norm(velocity[:,3:],axis=1)));stable=linear<=1e-4 and angular<=1e-3
 final=w.qpos[qa:qa+7].copy()
 if stable:d.qpos[qa:qa+7]=final;d.qvel[va:va+6]=0.;mujoco.mj_forward(m,d)
 np.savez_compressed(out/'tube_reset_settle_trace.npz',time=np.array([r[0] for r in rows]),tube_qpos=np.array([r[1] for r in rows]),tube_qvel=np.array([r[2] for r in rows]),initial_tube_qpos=initialq[qa:qa+7],final_tube_qpos=final)
 report=dict(scope='Separate fixed-boundary asset gravity-settle; only target tube freejoint copied once to episode RESET t0. No runtime object writes.',duration_s=duration_s,episode_time_s=float(d.time),initial_tube_qpos=initialq[qa:qa+7].tolist(),settled_tube_qpos=final.tolist(),asset_end_velocity=w.qvel[va:va+6].tolist(),non_target_state_reset_difference_max=float(abs(d.qpos[qm]-initialq[qm]).max()),other_workdata_dofs_held_fixed=True)
 report.update(initial_tilt_deg=float(initial_tilt_deg),tilt_axis_world=axis.tolist(),perturbed_asset_initial_tube_qpos=perturbed.tolist(),perturbation_scope='Explicit engineering RESET assumption, not a measured rack pose',stability_window_s=.5,tail_linear_speed_max_m_s=linear,tail_angular_speed_max_rad_s=angular,linear_speed_limit_m_s=1e-4,angular_speed_limit_rad_s=1e-3,stability_pass=stable,copied_to_episode=stable)
 np.savez_compressed(out/'tube_reset_stability_tail.npz',qvel=velocity,timestep_s=m.opt.timestep)
 (out/'tube_reset_settle_provenance.json').write_text(json.dumps(report,indent=2))
 if not stable:raise RuntimeError('Tube settling failed; episode RESET unchanged; see provenance')
 return report
