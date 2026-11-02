"""Post-held LEFT local 6D IK. Pre-held natural-pinch solver is untouched.

Every local continuation step stays within .15rad of its preceding seed.
Full original .1mm/.1deg pose accuracy and .05rad joint margins remain.
An extra .001rad internal reserve protects the unchanged .05rad spline gate.
Nearby feasible joints take priority over a global minimum wrist angle.
This is geometric planning only; no control, contact or physics-state edits.
"""
import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices

IK_REPORT=[]


def solve_arm(m,d,target,R,side='left',seed=None):
    if side!='left':
        raise ValueError('Independent loaded continuous solver is LEFT only')
    ids,qa,_=arm_indices(m,'left')
    w=mujoco.MjData(m);w.qpos[:]=d.qpos
    palm=m.body('sharpa_left/left_hand_C_MC').id
    elbow=m.body('astribot_arm_left_link_5').id
    wrist=m.body('astribot_arm_left_link_6').id
    middle=m.body('sharpa_left/left_middle_MCP_VL').id
    physical_lo=m.jnt_range[ids,0]+.05
    physical_hi=m.jnt_range[ids,1]-.05
    initial=np.asarray(d.qpos[qa] if seed is None else seed,dtype=float).copy()
    if np.any(initial<physical_lo-1e-6) or np.any(initial>physical_hi+1e-6):
        raise RuntimeError('Loaded entry seed lacks original .05rad geometric margin')
    initial=np.clip(initial,physical_lo+.001,physical_hi-.001)
    target=np.asarray(target,dtype=float);R=np.asarray(R,dtype=float)
    w.qpos[qa]=initial;mujoco.mj_kinematics(m,w)
    P0=w.xpos[palm].copy();R0=w.xmat[palm].reshape(3,3).copy()
    steps=[]

    def attempt(anchor,P,A):
        low=np.maximum(physical_lo+.001,anchor-.15)
        high=np.minimum(physical_hi-.001,anchor+.15)
        def pose(q):
            w.qpos[qa]=q;mujoco.mj_kinematics(m,w)
            return np.r_[w.xpos[palm]-P,Rotation.from_matrix(w.xmat[palm].reshape(3,3)@A.T).as_rotvec()]
        def wrist_cos(q):
            pose(q);u=w.xpos[wrist]-w.xpos[elbow];v=w.xpos[middle]-w.xpos[palm]
            return float(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1.,1.))
        def objective(q):
            dq=q-anchor
            return float(.5*(dq@dq)+.002*(1.-wrist_cos(q)))
        def equality(q):
            e=pose(q)
            return np.r_[1000.*e[:3],100.*e[3:]]
        fit=minimize(objective,np.clip(anchor,low+1e-10,high-1e-10),
                     method='SLSQP',bounds=list(zip(low,high)),
                     constraints=dict(type='eq',fun=equality),
                     options=dict(maxiter=180,ftol=1e-11))
        e=pose(fit.x)
        pe=float(np.linalg.norm(e[:3])*1000.)
        re=float(np.degrees(np.linalg.norm(e[3:])))
        margin=float(np.minimum(fit.x-m.jnt_range[ids,0],m.jnt_range[ids,1]-fit.x).min())
        jump=float(np.max(abs(fit.x-anchor)))
        if pe<.1 and re<.1 and margin>=.05-1e-8 and jump<=.15+1e-9:
            return fit.x.copy(),dict(position_error_mm=pe,orientation_error_deg=re,
                                     joint_margin_rad=margin,local_joint_step_rad=jump,
                                     forearm_palm_angle_deg=float(np.degrees(np.arccos(wrist_cos(fit.x)))))
        return None

    def advance(anchor,Pa,Aa,Pb,Ab,depth=0):
        result=attempt(anchor,Pb,Ab)
        if result is not None:
            q,info=result;steps.append(info);return q
        if depth>=5:
            raise RuntimeError('Loaded local 6D IK failed within .15rad trust region; original pose and margin guards retained')
        rv=Rotation.from_matrix(Ab@Aa.T).as_rotvec()
        Pmid=.5*(Pa+Pb);Amid=Rotation.from_rotvec(.5*rv).as_matrix()@Aa
        qmid=advance(anchor,Pa,Aa,Pmid,Amid,depth+1)
        return advance(qmid,Pmid,Amid,Pb,Ab,depth+1)

    rv=Rotation.from_matrix(R@R0.T).as_rotvec()
    count=max(1,int(np.ceil(np.linalg.norm(target-P0)/.01)),int(np.ceil(np.linalg.norm(rv)/.05)))
    q=initial.copy();Pa=P0;Aa=R0
    for i in range(1,count+1):
        u=i/count;Pb=P0+(target-P0)*u;Ab=Rotation.from_rotvec(rv*u).as_matrix()@R0
        q=advance(q,Pa,Aa,Pb,Ab);Pa=Pb;Aa=Ab
    info=dict(steps[-1],ik_method='loaded_local_seed_distance_then_weak_neutral_preference',
              continuation_steps=len(steps),maximum_local_joint_step_rad=max(x['local_joint_step_rad'] for x in steps),
              target_minus_entry_joint_max_rad=float(abs(q-initial).max()),
              trust_region_rad=.15,internal_joint_margin_rad=.051,seed_distance_weight=.5,neutral_preference_weight=.002,
              geometric_only=True)
    IK_REPORT.append(info)
    return q,info
