"""Opt-in near-upright elliptical swirl. Returns arm targets only.
No finger/force/friction changes. Numerical parameters are engineering defaults,
not measured human motion. No live object/base/qpos/time writes.
"""
from dataclasses import dataclass,asdict
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.interpolate import make_interp_spline
from t1_spincoat_scene import arm_indices
from t1_human_tube_mixing import quintic

@dataclass(frozen=True)
class MixingConfig:
 duration_s:float=5.
 amplitude_deg:float=5.
 frequency_hz:float=.6
 ramp_s:float=.8
 knot_dt_s:float=.02
 orbit_radius_x_m:float=.015
 orbit_radius_y_m:float=.025
 phase_name:str='human_swirl_ellipse'
 tube_length_m:float=.150
 maximum_entry_tilt_deg:float=10.
 rotation_axis_world:tuple=(1.,0.,0.)
 # Penalize large wrist excursions more than proximal compensation, while
 # full hand Cartesian pose remains primary. This is not a human joint map.
 reference_joint_weights:tuple=(.05,.05,.05,.05,.05,.15,.15)

def angle_at(t,c):
 w=float(quintic(t/c.ramp_s)*quintic((c.duration_s-t)/c.ramp_s));return np.deg2rad(c.amplitude_deg)*w*np.sin(2*np.pi*c.frequency_hz*t)

def orbit_offset(t,c):
 w=float(quintic(t/c.ramp_s)*quintic((c.duration_s-t)/c.ramp_s));phi=2*np.pi*c.frequency_hz*t;return w*np.array([c.orbit_radius_x_m*(np.cos(phi)-1),c.orbit_radius_y_m*np.sin(phi),0.])

def plan_mouth_pivot_shake(m,d,handles,arm_command,*,config=None,side='left',ik_solver=None):
 c=config or MixingConfig()
 if side not in ['left','right']:raise ValueError('Explicit actual arm side required')
 if not 0<c.amplitude_deg<=10 or c.frequency_hz<=0:raise ValueError('Bounded open-tube rocking amplitude/frequency required')
 ids,qa,_=arm_indices(m,side);pb=m.body(f'sharpa_{side}/{side}_hand_C_MC').id;tb=m.body(handles['tube_body']).id;mouth=m.site(handles['tube_mouth_site']).id;work=mujoco.MjData(m);work.qpos[:]=d.qpos;mujoco.mj_kinematics(m,work);qref=d.qpos[qa].copy();Rh=work.xmat[pb].reshape(3,3).copy();Rt=work.xmat[tb].reshape(3,3).copy();pivot=work.site_xpos[mouth].copy();local=Rh.T@(pivot-work.xpos[pb]);relR=Rh.T@Rt;axis=np.asarray(c.rotation_axis_world,dtype=float);axis/=np.linalg.norm(axis);entry=float(np.degrees(np.arccos(np.clip(Rt[2,2],-1,1))))
 if entry>c.maximum_entry_tilt_deg or abs(axis@Rt[:,2])>.1:raise ValueError('Actual held tube must enter near upright with transverse rocking axis')
 lo=m.jnt_range[ids,0]+.05;hi=m.jnt_range[ids,1]-.05
 if (qref<lo).any() or (qref>hi).any():raise ValueError('Actual entry arm lacks original .05rad geometric IK margin')
 times=np.linspace(0,c.duration_s,int(np.ceil(c.duration_s/c.knot_dt_s))+1);qs=[];points=[];rots=[];errors=[];seed=qref.copy();weights=np.asarray(c.reference_joint_weights);fallbacks=0
 for t in times:
  theta=angle_at(t,c);A=Rotation.from_rotvec(axis*theta).as_matrix()@Rt@relR.T;P=pivot+orbit_offset(t,c)-A@local
  def pose(q):
   work.qpos[qa]=q;mujoco.mj_kinematics(m,work);return np.r_[work.xpos[pb]-P,Rotation.from_matrix(work.xmat[pb].reshape(3,3)@A.T).as_rotvec()]
  def residual(q):
   e=pose(q);return np.r_[3000*e[:3],300*e[3:],weights*(q-qref)]
  if abs(theta)<1e-12 and np.linalg.norm(orbit_offset(t,c))<1e-12:q=qref.copy()
  else:
   fit=least_squares(residual,np.clip(seed,lo+1e-10,hi-1e-10),bounds=(lo,hi),max_nfev=150,ftol=1e-11,xtol=1e-11,gtol=1e-11);q=fit.x.copy()
  er=pose(q);pe=float(np.linalg.norm(er[:3])*1000);re=float(np.degrees(np.linalg.norm(er[3:])))
  if (pe>.1 or re>.1) and ik_solver is not None:
   start,_=ik_solver(m,work,P,A,side=side,seed=seed);fit=least_squares(residual,np.clip(start,lo+1e-10,hi-1e-10),bounds=(lo,hi),max_nfev=200,ftol=1e-11,xtol=1e-11,gtol=1e-11);q=fit.x.copy();er=pose(q);pe=float(np.linalg.norm(er[:3])*1000);re=float(np.degrees(np.linalg.norm(er[3:])));fallbacks+=1
  if pe>.1 or re>.1:raise RuntimeError(f'Anchored swirl IK failed at t={t:.6f}s: pos={pe:.6f}mm rot={re:.6f}deg; unchanged .1mm/.1deg pose guard')
  qs.append(q);points.append(P);rots.append(A);errors.append((pe,re));seed=q
 qs=np.array(qs);commands=np.asarray(arm_command)[None,:]+qs-qref;spline=make_interp_spline(times,commands,k=5,bc_type=([(1,np.zeros(7)),(2,np.zeros(7))],[(1,np.zeros(7)),(2,np.zeros(7))]));dense=np.linspace(0,c.duration_s,round(c.duration_s/m.opt.timestep)+1);cmd=spline(dense);velocity=spline(dense,1);margin=float(np.minimum(cmd-m.jnt_range[ids,0],m.jnt_range[ids,1]-cmd).min());acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in ids]);limited=m.actuator_ctrllimited[acts].astype(bool);ctrlmargin=float(np.minimum(cmd[:,limited]-m.actuator_ctrlrange[acts[limited],0],m.actuator_ctrlrange[acts[limited],1]-cmd[:,limited]).min()) if limited.any() else None
 if abs(velocity).max()>2 or margin<.005 or (ctrlmargin is not None and ctrlmargin<0):raise RuntimeError('Rocking command violates existing speed/range guards')
 angles=np.array([angle_at(t,c) for t in dense]);omega=np.gradient(angles,dense);acc=np.gradient(omega,dense)
 report=dict(config=asdict(c),side=side,scope='Planning only until caller actually executes; original grasp/base/geometry/liquid gates remain required.',reference_sources=['https://www2.chem.wisc.edu/deptfiles/genchem/lab/labdocs/modules/mixing/mixman.htm','https://energy.wisc.edu/sites/default/files/2017-08/Fermentation%20Challenge%20Teacher.pdf'],motion_evidence='UW manual explicitly distinguishes swirl (base in small circles) from fixed-top rocking. University fermentation protocol specifies gentle test-tube swirl. Ellipse dimensions/frequency/5deg tilt are engineering defaults; not recovered human video measurements.',parameter_provenance='Explicit engineering candidate; no amplitude/frequency recovered from human video.',entry_mouth_world_m=pivot.tolist(),mouth_pivot_world_m=pivot.tolist(),mouth_is_stationary=False,maximum_nominal_horizontal_mouth_offset_mm=float(max(np.linalg.norm(orbit_offset(t,c))*1000 for t in dense)),nominal_tube_bottom_worldY_peak_to_peak_mm=float(np.ptp([orbit_offset(t,c)[1]+c.tube_length_m*np.sin(angle_at(t,c)) for t in dense])*1000),entry_tube_tilt_deg=entry,tilt_only_bottom_peak_excursion_m=c.tube_length_m*np.sin(np.deg2rad(c.amplitude_deg)),sampled_nominal_mouth_peak_speed_m_s=float(np.linalg.norm(np.gradient(np.array([orbit_offset(t,c) for t in dense]),dense,axis=0),axis=1).max()),sampled_nominal_mouth_peak_acceleration_m_s2=float(np.linalg.norm(np.gradient(np.gradient(np.array([orbit_offset(t,c) for t in dense]),dense,axis=0),dense,axis=0),axis=1).max()),sampled_peak_angular_velocity_rad_s=float(abs(omega).max()),sampled_peak_angular_acceleration_rad_s2=float(abs(acc).max()),maximum_IK_position_error_mm=max(x[0] for x in errors),maximum_IK_orientation_error_deg=max(x[1] for x in errors),sampled_physics_timestep_peak_command_velocity_rad_s=float(abs(velocity).max()),sampled_command_joint_margin_rad=margin,sampled_actuator_control_margin_rad=ctrlmargin,arm_joint_peak_to_peak_rad=np.ptp(qs,axis=0).tolist(),final_target_minus_entry_rad=(commands[-1]-np.asarray(arm_command)).tolist(),initial_command_jump_rad=float(abs(commands[0]-arm_command).max()),nullspace_reference='Fixed actual entry q throughout all cycles; no accumulated per-knot recentering and no forced terminal-knot overwrite.',reference_weights=weights.tolist(),fallback_seed_calls=fallbacks,runtime_object_pose_writes=False,hand_target_changes=False,tactile_feedback_control=False,chemical_homogeneity_certified=False)
 return dict(time=times,kinematic_q=qs,arm_commands=commands,palm_positions=np.array(points),palm_rotations=np.array(rots),spline=spline,report=report)

def advance_mouth_pivot_shake(m,d,handles,arm_command,step_callback,*,config=None,side='left',ik_solver=None):
 c=config or MixingConfig();plan=plan_mouth_pivot_shake(m,d,handles,arm_command,config=c,side=side,ik_solver=ik_solver);start=float(d.time);mouth=m.site(handles['tube_mouth_site']).id;pivot=np.array(plan['report']['mouth_pivot_world_m']);actual=[]
 for k in range(round(c.duration_s/m.opt.timestep)):
  before=float(d.time);step_callback(np.asarray(plan['spline'](min((k+1)*m.opt.timestep,c.duration_s))),c.phase_name,True)
  if abs(d.time-before-m.opt.timestep)>1e-9:raise RuntimeError('Caller must advance exactly one existing physics timestep')
  actual.append((float(d.time),d.site_xpos[mouth].copy()))
 final=plan['arm_commands'][-1].copy();report=dict(plan['report'],execution_start_s=start,execution_end_s=float(d.time),actual_mouth_excursion_max_mm=max(float(np.linalg.norm(P-pivot)*1000) for _,P in actual),execution_completed=True,physical_success_not_inferred=True)
 return final,report,dict(time=np.array([t for t,_ in actual]),mouth_world_m=np.array([P for _,P in actual]))
# Explicit v3 names; legacy-shaped aliases above allow an isolated stage adapter.
plan_elliptic_swirl=plan_mouth_pivot_shake
advance_elliptic_swirl=advance_mouth_pivot_shake
