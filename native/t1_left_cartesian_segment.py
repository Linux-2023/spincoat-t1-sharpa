"""Opt-in Cartesian LEFT arm target spline. Never writes live qpos/object/base/time."""
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices
from t1_left_continuous_ik import solve_arm

class CartesianArmSegment:
 def __init__(self,times,q):
  self.times=np.asarray(times);self.q=np.asarray(q);self.duration_s=float(self.times[-1]);self.spline=CubicSpline(self.times,self.q,bc_type=((1,np.zeros(7)),(1,np.zeros(7))))
 def __call__(self,elapsed_s):return np.asarray(self.spline(np.clip(elapsed_s,0,self.duration_s)))

def plan_cartesian_arm_segment(m,d,target_palm_world_m,target_palm_R_world,*,seed,duration_s=2.5,knot_dt_s=.02,side='left',ik_solver=None):
 if side!='left':raise ValueError('New opt-in LEFT segment only')
 if duration_s<=0 or knot_dt_s<=0:raise ValueError('positive duration/dt required')
 solver=solve_arm if ik_solver is None else ik_solver
 ids,qa,_=arm_indices(m,'left');pb=m.body('sharpa_left/left_hand_C_MC').id;work=mujoco.MjData(m);work.qpos[:]=d.qpos;work.qpos[qa]=np.asarray(seed);mujoco.mj_kinematics(m,work);P0=work.xpos[pb].copy();R0=work.xmat[pb].reshape(3,3).copy();target=np.asarray(target_palm_world_m);R=np.asarray(target_palm_R_world);rv=Rotation.from_matrix(R@R0.T).as_rotvec();N=max(2,int(np.ceil(duration_s/knot_dt_s)));times=np.linspace(0,duration_s,N+1);q=np.asarray(seed).copy();knots=[q.copy()];errors=[]
 for t in times[1:]:
  u=t/duration_s;s=u**3*(10+u*(-15+6*u));P=P0+(target-P0)*s;A=Rotation.from_rotvec(rv*s).as_matrix()@R0;q,e=solver(m,work,P,A,side='left',seed=q);knots.append(q.copy());errors.append(e)
 segment=CartesianArmSegment(times,np.asarray(knots));dense=np.arange(0,duration_s+.0005,.001);commands=segment.spline(dense);velocity=segment.spline(dense,1);margin=np.minimum(commands-m.jnt_range[ids,0],m.jnt_range[ids,1]-commands).min();report=dict(scope=__doc__,duration_s=duration_s,knot_dt_s=float(times[1]-times[0]),knot_count=len(times),target_palm_world_m=target.tolist(),target_palm_R_world=R.tolist(),maximum_knot_position_error_mm=max(e['position_error_mm'] for e in errors),maximum_knot_rotation_error_deg=max(e['orientation_error_deg'] for e in errors),sampled_1ms_command_joint_margin_rad=float(margin),sampled_1ms_command_peak_velocity_rad_s=float(np.abs(velocity).max()),command_boundary_velocity_rad_s=[np.asarray(segment.spline(0,1)).tolist(),np.asarray(segment.spline(duration_s,1)).tolist()],geometric_planning_only=True,actual_following_or_contact_validated=False)
 if margin<.049 or abs(velocity).max()>2.:raise RuntimeError('Cartesian arm target spline violates planning margin/speed bounds '+str(report))
 return segment,report

def plan_tube_descent(m,d,handles,arm_command,target_bottom_world_m,duration_s=2.5,ik_solver=None):
 """Runtime actual measured hand/tube transform -> LEFT target commands only.
 Current arm position-servo load offset is preserved differentially, like mixing.
 Returns time/arm_commands/geometric_arm_q/spline/report; no live state writes.
 """
 _,qa,_=arm_indices(m,'left');pb=m.body('sharpa_left/left_hand_C_MC').id;tb=m.body(handles['tube_body']).id;observed=mujoco.MjData(m);observed.qpos[:]=d.qpos;mujoco.mj_kinematics(m,observed);actual=d.qpos[qa].copy();current=np.asarray(arm_command,dtype=float).copy();Rh=observed.xmat[pb].reshape(3,3).copy();Rt=observed.xmat[tb].reshape(3,3).copy();root_local=Rh.T@(observed.xpos[tb]-observed.xpos[pb]);tube_local_R=Rh.T@Rt;targetR=tube_local_R.T;target=np.asarray(target_bottom_world_m)-targetR@root_local
 segment,report=plan_cartesian_arm_segment(m,d,target,targetR,seed=actual,duration_s=duration_s,knot_dt_s=.02,side='left',ik_solver=ik_solver);offset=current-actual;commands=segment.q+offset;spline=CubicSpline(segment.times,commands,bc_type=((1,np.zeros(7)),(1,np.zeros(7))))
 report.update(actual_hand_to_tube_position_in_palm_m=root_local.tolist(),actual_hand_to_tube_rotation_in_palm=tube_local_R.tolist(),actual_tube_bottom_world_m=observed.xpos[tb].tolist(),target_bottom_world_m=np.asarray(target_bottom_world_m).tolist(),preserved_initial_command_minus_actual_q_rad=offset.tolist(),initial_command_jump_rad=float(abs(commands[0]-current).max()),physical_object_base_time_pose_writes=False,pose_observation='Privileged actual simulator body transform, not RGB/tactile feedback',following_scope='Geometry path is raw geometric_arm_q; servo commands retain initial measured load offset. Fresh actual following/contact must be audited.')
 return dict(time=segment.times.copy(),arm_commands=commands,geometric_arm_q=segment.q.copy(),spline=spline,geometric_spline=segment.spline,report=report)
def plan_palm_cartesian_move(m,d,arm_command,target_palm_world_m,target_palm_R_world,duration_s,knot_dt_s=.02,ik_solver=None):
 """Opt-in post-held move; preserve current servo load offset, return targets only.
 Actual Cartesian pose starts the minimum-jerk trajectory. The current command
 is exactly preserved at t0, including its measured command-minus-q offset.
 """
 _,qa,_=arm_indices(m,'left');actual=d.qpos[qa].copy();current=np.asarray(arm_command,dtype=float).copy();segment,report=plan_cartesian_arm_segment(m,d,target_palm_world_m,target_palm_R_world,seed=actual,duration_s=duration_s,knot_dt_s=knot_dt_s,side='left',ik_solver=ik_solver);offset=current-actual;commands=segment.q+offset;spline=CubicSpline(segment.times,commands,bc_type=((1,np.zeros(7)),(1,np.zeros(7))))
 report.update(preserved_initial_command_minus_actual_q_rad=offset.tolist(),initial_command_jump_rad=float(abs(commands[0]-current).max()),physical_object_base_time_pose_writes=False,following_scope='Actual arm FK seeds geometric Cartesian move; current position-servo load offset is carried as a constant command overlay. Actual following and held-tool stability require a fresh physics rollout.')
 return dict(time=segment.times.copy(),arm_commands=commands,geometric_arm_q=segment.q.copy(),spline=spline,geometric_spline=segment.spline,report=report)
def plan_loaded_cartesian_transport(m,d,arm_command,goal_joint_target,duration_s,ik_solver=None):
 """Opt-in LEFT post-held transport from actual FK to original goal-q FK.
 The original target remains the same physical Cartesian endpoint. Only the
 intervening path changes; use an explicit phase whitelist in the caller.
 """
 _,qa,_=arm_indices(m,'left');pb=m.body('sharpa_left/left_hand_C_MC').id;goal=np.asarray(goal_joint_target,dtype=float)
 if goal.shape!=(7,) or not np.isfinite(goal).all():raise ValueError('LEFT goal must be7 finite joint targets')
 work=mujoco.MjData(m);work.qpos[:]=d.qpos;work.qpos[qa]=goal;mujoco.mj_kinematics(m,work);targetP=work.xpos[pb].copy();targetR=work.xmat[pb].reshape(3,3).copy();result=plan_palm_cartesian_move(m,d,arm_command,targetP,targetR,duration_s,ik_solver=ik_solver);result['commands']=result['arm_commands'];result['report'].update(endpoint_source='Original goal_joint_target FK on isolated current-state workData',original_goal_joint_target_rad=goal.tolist(),allowed_scope='Explicitly enabled LEFT post-held raise/carry/return transport only; does not alter finger targets, tilt schedule, forcecap, friction, model or original RIGHT control.')
 return result