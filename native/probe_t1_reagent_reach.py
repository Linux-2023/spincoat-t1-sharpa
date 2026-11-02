"""Independent left-hand prep kinematic seed; no live poses modified by IK."""
from pathlib import Path
import argparse,json,numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from preview_t1_reagent_preparation import build_preparation_scene,copy_named_state
from t1_spincoat_scene import ROOT,arm_indices,hand_indices

def solve_arm(m,d,point,R,side='left',seed=None):
 ids,qa,dof=arm_indices(m,side);work=mujoco.MjData(m);work.qpos[:]=d.qpos;pb=m.body('sharpa_'+side+'/'+side+'_hand_C_MC').id;lo=m.jnt_range[ids,0]+.005;hi=m.jnt_range[ids,1]-.005;q0=np.clip(d.qpos[qa] if seed is None else seed,lo,hi)
 def residual(q):
  work.qpos[qa]=q;mujoco.mj_kinematics(m,work);return np.r_[(work.xpos[pb]-point)*10,Rotation.from_matrix(work.xmat[pb].reshape(3,3)@R.T).as_rotvec(),.001*(q-q0)]
 starts=[q0,np.clip(np.array([-.6,-1.,.3,1.4,-.8,.2,-.5]),lo,hi),np.clip(np.array([.2,-1.4,-.7,1.5,.7,-.2,.6]),lo,hi)]
 fits=[least_squares(residual,start,bounds=(lo,hi),max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10) for start in starts];fit=min(fits,key=lambda v:np.linalg.norm(residual(v.x)[:6]));work.qpos[qa]=fit.x;mujoco.mj_forward(m,work);return fit.x,dict(position_error_mm=float(np.linalg.norm(work.xpos[pb]-point)*1000),orientation_error_deg=float(np.degrees(Rotation.from_matrix(work.xmat[pb].reshape(3,3)@R.T).magnitude())),joint_margin_rad=float(np.minimum(fit.x-m.jnt_range[ids,0],m.jnt_range[ids,1]-fit.x).min()))

def make_reagent_plan(m,d,h,side='right'):
 right=m.body('sharpa_right/right_hand_C_MC').id;pipe=m.body('pipette/pipette').id;pts=[]
 for c in d.contact:
  ns=[m.body(int(m.geom_bodyid[int(g)])).name for g in c.geom]
  if any(n.startswith('pipette/') for n in ns) and any(n.startswith('sharpa_right/right_') and 'thumb' not in n and 'hand_C' not in n for n in ns):pts.append(c.pos.copy())
 if not pts:raise RuntimeError('No physical reference support contacts')
 S=np.diag([1.,-1.,1.]) if side=='left' else np.eye(3);Rr=d.xmat[right].reshape(3,3);R=S@Rr@S;offset=S@(Rr.T@(np.mean(pts,axis=0)-d.xpos[right]));tube=d.xpos[m.body(h['tube_body']).id].copy();contact=tube+[0,0,.07];pick=contact-R@offset
 ids,qa,_=arm_indices(m,side);q,err=solve_arm(m,d,pick,R,side=side);lift=pick+[0,0,.10];ql,el=solve_arm(m,d,lift,R,side=side,seed=q);turn=Rotation.from_euler('x',100,degrees=True).as_matrix();Rt=turn;mouth=np.array([.4,.05,.965]);rootp=mouth-Rt@np.array([0,0,.11465]);grasp=rootp+Rt@np.array([0,0,.07]);Rp=turn@R;pourp=grasp-Rp@offset;qp,ep=solve_arm(m,d,pourp,Rp,side=side,seed=ql)
 return dict(side=side,reference_contact_mean_world_m=np.mean(pts,axis=0).tolist(),mirrored_contact_offset_palm_m=offset.tolist(),pickup_palm_world_m=pick.tolist(),pickup_palm_R_world=R.tolist(),lift_palm_world_m=lift.tolist(),pour_palm_world_m=pourp.tolist(),pour_palm_R_world=Rp.tolist(),pickup_arm_q=q.tolist(),lift_arm_q=ql.tolist(),pour_arm_q=qp.tolist(),pickup_ik=err,lift_ik=el,pour_ik=ep,scope='Work-data kinematic feasibility only, no support/dynamic/collision success.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',default='left_reach_01');p.add_argument('--side',default='right',choices=['right','left']);a=p.parse_args();out=ROOT/'research/hardware/reagent_preparation'/a.tag;out.mkdir(parents=True,exist_ok=False);from t1_reagent_preparation import PreparationConfig
 c=PreparationConfig(tube_xy_m=(.44,.16));s,m,d,children,h,ledger=build_preparation_scene(out,c);ref=ROOT/'logs/spincoat/t1/integrated_neutralB01_03';old=mujoco.MjModel.from_binary_path(str(ref/'model.mjb'));saved=np.load(ref/'handoff_grasp_end.npz');copy_named_state(old,saved,m,d);d.time=float(saved['time']);mujoco.mj_forward(m,d);plan=make_reagent_plan(m,d,h,side=a.side);(out/'reach_plan.json').write_text(json.dumps(plan,indent=2));np.savez_compressed(out/'independent_checkpoint_seed.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time);mujoco.mj_saveModel(m,str(out/'model.mjb'),None);print(json.dumps(plan,indent=2))
if __name__=='__main__':main()


