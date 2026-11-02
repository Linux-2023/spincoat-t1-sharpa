"""Opt-in LEFT-only near-seed IK. Original right/global solver is unchanged.
This is geometric IK, not collision checking or motion dynamics validation.
"""
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices
from probe_t1_reagent_reach import solve_arm as original_solve_arm

def solve_arm(m,d,target,R,side='right',seed=None):
 if side!='left':return original_solve_arm(m,d,target,R,side=side,seed=seed)
 ids,qa,_=arm_indices(m,'left');pb=m.body('sharpa_left/left_hand_C_MC').id;work=mujoco.MjData(m);work.qpos[:]=d.qpos;seed=np.asarray(d.qpos[qa] if seed is None else seed,dtype=float).copy();target=np.asarray(target,dtype=float);R=np.asarray(R,dtype=float);lo=m.jnt_range[ids,0]+.05;hi=m.jnt_range[ids,1]-.05;seed=np.clip(seed,lo+1e-9,hi-1e-9);work.qpos[qa]=seed;mujoco.mj_kinematics(m,work);start=work.xpos[pb].copy();startR=work.xmat[pb].reshape(3,3).copy();rotvec=Rotation.from_matrix(R@startR.T).as_rotvec();N=max(1,int(np.ceil(np.linalg.norm(target-start)/.025)),int(np.ceil(np.linalg.norm(rotvec)/.12)));q=seed.copy();maxstep=0.
 for k in range(1,N+1):
  u=k/N;P=start+(target-start)*u;A=Rotation.from_rotvec(rotvec*u).as_matrix()@startR;anchor=q.copy()
  def pose(x):
   work.qpos[qa]=x;mujoco.mj_kinematics(m,work);return np.r_[work.xpos[pb]-P,Rotation.from_matrix(work.xmat[pb].reshape(3,3)@A.T).as_rotvec()]
  def residual(x):
   e=pose(x);return np.r_[3000*e[:3],300*e[3:],.05*(x-anchor),.003/(x-lo+.05),.003/(hi-x+.05)]
  candidates=[]
  starts=[anchor]
  for offset in [0.,.12,-.12]:
   trial=anchor.copy();trial[3]+=offset;starts.append(np.clip(trial,lo+1e-8,hi-1e-8))
  for q0 in starts:
   fit=least_squares(residual,np.clip(q0,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10);e=pose(fit.x);pe=float(np.linalg.norm(e[:3])*1000);re=float(np.degrees(np.linalg.norm(e[3:])));margin=float(np.minimum(fit.x-m.jnt_range[ids,0],m.jnt_range[ids,1]-fit.x).min())
   if pe<.1 and re<.1 and margin>=.05-1e-9:candidates.append((float(np.linalg.norm(fit.x-anchor)),fit.x.copy(),pe,re,margin))
  if not candidates:raise RuntimeError('LEFT near-seed IK no qualified solution at continuation '+str(k)+'/'+str(N)+'; original physical limits unchanged, required position<.1mm/orientation<.1deg/margin>=.05rad')
  _,q,pe,re,margin=min(candidates,key=lambda z:z[0]);maxstep=max(maxstep,float(np.max(abs(q-anchor))))
 e=pose(q);return q,dict(position_error_mm=float(np.linalg.norm(e[:3])*1000),orientation_error_deg=float(np.degrees(np.linalg.norm(e[3:]))),joint_margin_rad=margin,ik_method='left_near_seed_continuation_full_regularized_residual',continuation_steps=N,maximum_continuation_joint_step_rad=maxstep,minimum_requested_joint_margin_rad=.05,geometric_nullspace_barrier_weight=.003,qualified_candidate_selected_by='smallest joint distance to preceding seed among all pose-qualified candidates')
