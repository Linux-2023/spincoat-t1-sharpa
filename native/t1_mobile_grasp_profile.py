"""Re-solve fixed-base ready approach at measured wheel-supported root height."""
import copy,json
import numpy as np,mujoco
from scipy.optimize import least_squares
from t1_spincoat_scene import ROOT,scene,arm_indices

def adapt_grasp_profile(m,d,profile_path,out):
    p=json.loads(profile_path.read_text());_,fixed,fd,_=scene()
    _,fq,_=arm_indices(fixed);ids,qa,_=arm_indices(m)
    body=m.body('sharpa_right/right_hand_C_MC').id;fb=fixed.body('sharpa_right/right_hand_C_MC').id
    work=mujoco.MjData(m);work.qpos[:]=d.qpos
    lo=m.jnt_range[ids,0]+.04;hi=m.jnt_range[ids,1]-.04
    audit=[]
    for frame in p['waypoints']:
        original=np.asarray(frame['arm_q_rad']);fd.qpos[fq]=original;mujoco.mj_kinematics(fixed,fd)
        target=fd.xpos[fb]+[0,.4,0];quat=fd.xquat[fb].copy()
        def residual(q):
            work.qpos[qa]=q;mujoco.mj_kinematics(m,work);e=np.zeros(3);mujoco.mju_subQuat(e,quat,work.xquat[body])
            return np.r_[100*(work.xpos[body]-target),10*e,.0001*(q-original)]
        fit=least_squares(residual,np.clip(original,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=150,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        rr=residual(fit.x);assert np.linalg.norm(rr[:3])<.01,(frame['name'],rr)
        frame['arm_q_rad']=fit.x.tolist();audit.append({'name':frame['name'],'target_world_m':target.tolist(),'position_error_mm':float(np.linalg.norm(rr[:3])*10),'orientation_error_deg':float(np.rad2deg(np.linalg.norm(rr[3:6])/10))})
    p['provenance']='Fixed-base profile world FK +Y0.4, re-solved on physical wheel-supported free-base reset; not base pose animation'
    p['mobile_fk_retarget_audit']=audit;p['base_reset_z_m']=float(d.qpos[m.joint('chassis_free').qposadr[0]+2])
    out.write_text(json.dumps(p,indent=2));d.qpos[qa]=p['waypoints'][0]['arm_q_rad'];mujoco.mj_forward(m,d)
    return out
