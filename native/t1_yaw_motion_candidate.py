"""Run an independent yaw candidate without editing scene or runner sources."""
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import run_t1_spincoat_episode as runner

def main():
 parser=argparse.ArgumentParser(add_help=False)
 parser.add_argument('--grasp-yaw-deg',type=float,default=0.)
 parser.add_argument('--transport-yaw-deg',type=float,default=0.)
 parser.add_argument('--transport-yaw-start-s',type=float,default=14.5)
 parser.add_argument('--transport-yaw-duration-s',type=float,default=3.)
 parser.add_argument('--ik-limit-margin-rad',type=float,default=0.)
 parser.add_argument('--smooth-nullspace',action='store_true')
 parser.add_argument('--nullspace-start-s',type=float,default=8.5)
 args,remaining=parser.parse_known_args()
 rotation=Rotation.from_euler('z',args.grasp_yaw_deg,degrees=True)
 quat=(rotation*Rotation.from_quat(np.array(runner.HAND_QUAT)[[1,2,3,0]])).as_quat()[[3,0,1,2]]
 offset=rotation.apply(runner.GRASP_WRIST_OFFSET)
 runner.HAND_QUAT=quat;runner.GRASP_WRIST_OFFSET=offset
 baseline_ik=runner.ik;applied=[0.]
 if args.ik_limit_margin_rad:
  def margin_ik(m,d,position,quaternion,seed=None,iterations=150):
   ids,qids,dofs=runner.arm_indices(m);work=mujoco.MjData(m);work.qpos[:]=d.qpos
   lower=m.jnt_range[ids,0]+args.ik_limit_margin_rad;upper=m.jnt_range[ids,1]-args.ik_limit_margin_rad
   work.qpos[qids]=np.clip(seed if seed is not None else d.qpos[qids],lower,upper)
   seed_reference=work.qpos[qids].copy()
   body=m.body('sharpa_right/right_hand_C_MC').id;jp=np.zeros((3,m.nv));jr=jp.copy()
   middle=(lower+upper)/2;half=(upper-lower)/2
   for _ in range(iterations):
    mujoco.mj_kinematics(m,work);mujoco.mj_comPos(m,work)
    ep=np.asarray(position)-work.xpos[body];eq=np.zeros(3);mujoco.mju_subQuat(eq,np.asarray(quaternion),work.xquat[body]);eq=work.xmat[body].reshape(3,3)@eq
    if np.linalg.norm(ep)<1e-6 and np.linalg.norm(eq)<1e-5 and work.qpos[qids[-1]]>-1.32:break
    mujoco.mj_jacBody(m,work,jp,jr,body);J=np.vstack([jp[:,dofs],.3*jr[:,dofs]])
    inverse=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-5,np.eye(6))
    posture=.008*(middle-work.qpos[qids])/(half*half)
    posture[6]+=.08*(-1.20-work.qpos[qids[6]])
    null_step=(np.eye(7)-inverse@J)@posture
    if args.smooth_nullspace:
     u=np.clip((d.time-args.nullspace_start_s)/3.,0.,1.);ramp=u**3*(10-15*u+6*u*u)
     null_step*=ramp
     # Bound the accumulated redundant motion per IK call, independently of
     # the primary Cartesian correction. This prevents 35 iterations from
     # integrating a large posture command at the prefix/online handover.
     budget=.003/max(iterations,1)
     null_step*=min(1.,budget/max(np.max(np.abs(null_step)),1e-12))
    dq=inverse@np.r_[ep,.3*eq]+null_step
    work.qpos[qids]=np.clip(work.qpos[qids]+np.clip(dq,-.04,.04),lower,upper)
   mujoco.mj_kinematics(m,work)
   return work.qpos[qids].copy(),float(np.linalg.norm(work.xpos[body]-position))
  baseline_ik=margin_ik
 def transport_ik(m,d,position,quaternion,seed=None,iterations=150):
  u=np.clip((d.time-args.transport_yaw_start_s)/args.transport_yaw_duration_s,0,1)
  target=np.deg2rad(args.transport_yaw_deg)*u**3*(10-15*u+6*u*u)
  increment=target-applied[0];applied[0]=target
  if abs(increment)>1e-12:
   rz=Rotation.from_rotvec([0,0,increment])
   tip=d.site_xpos[m.site('pipette/tip_site').id]
   position=tip+rz.apply(np.asarray(position)-tip)
   quaternion=(rz*Rotation.from_quat(np.asarray(quaternion)[[1,2,3,0]])).as_quat()[[3,0,1,2]]
  return baseline_ik(m,d,position,quaternion,seed=seed,iterations=iterations)
 if args.transport_yaw_deg or args.ik_limit_margin_rad:runner.ik=transport_ik
 sys.argv=[sys.argv[0],*remaining]
 runner.main()
 tag=remaining[remaining.index('--tag')+1]
 out=runner.ROOT/'logs/spincoat/t1'/tag
 provenance={'grasp_yaw_deg':args.grasp_yaw_deg,'hand_quaternion_wxyz':quat.tolist(),'grasp_wrist_offset_m':offset.tolist(),'transport_yaw_deg':args.transport_yaw_deg,'transport_yaw_start_s':args.transport_yaw_start_s,'transport_yaw_duration_s':args.transport_yaw_duration_s,'operation':'Rz(yaw) * Rx(90); wrist goal rotated about measured pipette tip during air transport; source files unchanged','runner_sha256':hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),'model_designed':True}
 (out/'yaw_candidate_parameters.json').write_text(json.dumps(provenance,indent=2))
 provenance['ik_limit_margin_rad']=args.ik_limit_margin_rad
 provenance['smooth_nullspace']=args.smooth_nullspace
 provenance['nullspace_start_s']=args.nullspace_start_s
 (out/'yaw_candidate_parameters.json').write_text(json.dumps(provenance,indent=2))
 print('YAW_CANDIDATE',json.dumps(provenance),flush=True)
if __name__=='__main__':main()
