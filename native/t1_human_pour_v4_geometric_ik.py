"""Fixed-progress geometric IK for independent v4 pour. No physics mutation."""
import numpy as np,mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices
def solve_geometric(m,d,target,R,*,seed):
 ids,qa,_=arm_indices(m,'left');w=mujoco.MjData(m);w.qpos[:]=d.qpos;p=m.body('sharpa_left/left_hand_C_MC').id;initial=np.asarray(seed).copy();physical_lo=m.jnt_range[ids,0];physical_hi=m.jnt_range[ids,1];lo=np.maximum(physical_lo+.051,initial-.12);hi=np.minimum(physical_hi-.051,initial+.12)
 def residual(q):
  w.qpos[qa]=q;mujoco.mj_kinematics(m,w)
  return np.r_[1000*(w.xpos[p]-target),100*Rotation.from_matrix(w.xmat[p].reshape(3,3)@np.asarray(R).T).as_rotvec()]
 def objective(q):
  marginlo=q-physical_lo+.05;marginhi=physical_hi-q+.05
  return float(.5*np.sum((q-initial)**2)+1e-4*np.sum(1/marginlo**2+1/marginhi**2))
 fit=minimize(objective,np.clip(initial,lo+1e-9,hi-1e-9),method='SLSQP',bounds=list(zip(lo,hi)),constraints=dict(type='eq',fun=residual),options=dict(maxiter=100,ftol=1e-11))
 e=residual(fit.x);pe=np.linalg.norm(e[:3]);re=np.degrees(np.linalg.norm(e[3:])/100);margin=np.minimum(fit.x-physical_lo,physical_hi-fit.x).min()
 if pe>=.1 or re>=.1 or margin<.05-1e-8:raise RuntimeError('V4 geometric IK rejected '+str((pe,re,margin)))
 return fit.x,dict(position_error_mm=float(pe),orientation_error_deg=float(re),joint_margin_rad=float(margin),ik_method='fixed_progress_near_seed_with_joint_limit_barrier',geometric_only=True)

