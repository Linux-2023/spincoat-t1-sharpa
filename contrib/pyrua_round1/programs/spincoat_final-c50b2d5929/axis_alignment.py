"""Align the rocking axis relative to the held palm, without copying an arm pose."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pyrua.codebox.outcome import Outcome

def plan_align(cb,wrist7=-.32194,count=150,delta=(0,0,0)):
    if cb.table_z is None:cb.table()
    obs=cb.robo.observe();m=cb.robot;arm=cb.arms['left'];base=dict(zip(cb.arm_names,obs.arm_q));base.update(zip(cb.dex_names,obs.dex_q));m.fk(base);P,R=m.pose(arm.palm);P0=P.copy();R0=R.copy();prev=obs.arm_q[7:].copy();rows=[];worst=0.;lowest=10.
    for i,value in enumerate(np.linspace(prev[6],wrist7,count)):
        goal_P=P0+np.asarray(delta)*(i+1)/count
        ref=prev.copy()
        def res(x):
            q=dict(base);q.update(zip(arm.names,x));m.fk(q);p,r=m.pose(arm.palm)
            return np.r_[p-goal_P,.15*Rotation.from_matrix(R0.T@r).as_rotvec(),.15*(x[6]-value),.00001*(x-ref)]
        fit=least_squares(res,np.clip(prev,arm.lo+1e-7,arm.hi-1e-7),bounds=(arm.lo+1e-8,arm.hi-1e-8),max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        err=res(fit.x);e=float(np.linalg.norm(err[:3]));angle=float(np.linalg.norm(err[3:6])/.15);lowest=min(lowest,float(arm.lowest()[0]));worst=max(worst,e)
        if e>.001 or angle>.005 or abs(fit.x[6]-value)>.005:
            return Outcome(False,'alignment_unreachable',{'value':float(value),'position_error':e,'rotation_error':angle},0),None
        rows.append(fit.x);prev=fit.x
    commands=np.tile(obs.arm_cmd,(len(rows),1));commands[:,7:]=rows
    limits=np.array([cb.spec['embodiment']['joint_speeds'][n] for n in arm.names])/25
    ratio=float(np.max(np.abs(np.diff(np.vstack([obs.arm_cmd,commands]),axis=0))[:,7:]/limits))
    data={'delta':list(delta),'target_joint7':wrist7,'final_joint6':float(prev[5]),'position_error':worst,'lowest':lowest,'speed_ratio':ratio,'final_arm':prev.tolist()}
    print('ALIGN_PLAN',data,flush=True)
    if ratio>1 or lowest<cb.table_z+.025:return Outcome(False,'alignment_guard',data,0),None
    return Outcome(True,'alignment_planned',data,0),{'rows':commands,'base':base,'P0':P0,'R0':R0}

def execute_align(cb,p):
    r=cb.robo.step(arm=p['rows'],log=True);cb.sync();maxp=0.
    for a in ([] if r.arm_log is None else r.arm_log):
        q=dict(p['base']);q.update(zip(cb.arm_names,a));cb.robot.fk(q);P,R=cb.robot.pose(cb.arms['left'].palm);maxp=max(maxp,float(np.linalg.norm(P-p['P0'])))
    print('ALIGN_TRACKING',{'max_palm_translation':maxp},flush=True)
    if r.episode_ended:return Outcome(False,'episode_ended',{},r.executed)
    cb.hold(25)
    if cb.ended:return Outcome(False,'episode_ended',{},r.executed+25)
    return Outcome(True,'aligned',{},r.executed+25)


def plan_adaptive(cb,amplitude=.22,wrist7=-.32194,count=150,max_shift=.04):
    """Choose the smallest palm shift that makes the selected rocking axis feasible."""
    if cb.table_z is None:cb.table()
    obs=cb.robo.observe();m=cb.robot;arm=cb.arms['left']
    base=dict(zip(cb.arm_names,obs.arm_q));base.update(zip(cb.dex_names,obs.dex_q));m.fk(base)
    P,R=m.pose(arm.palm);P0=P.copy();R0=R.copy();start=obs.arm_q[7:].copy()
    lower=arm.lo[5]+amplitude+.02;upper=arm.hi[5]-amplitude-.02
    def residual(x):
        q=dict(base);q.update(zip(arm.names,x));m.fk(q);p,r=m.pose(arm.palm)
        return np.r_[p-P0,5*Rotation.from_matrix(R0.T@r).as_rotvec(),5*(x[6]-wrist7),5*max(0,lower-x[5]),5*max(0,x[5]-upper),.00001*(x-start)]
    seeds=[start.copy(),start.copy()];seeds[1][6]=wrist7
    rng=np.random.default_rng(9)
    seeds.extend(rng.uniform(arm.lo,arm.hi) for _ in range(3))
    candidates=[];pose_shifts=[]
    for seed in seeds:
        fit=least_squares(residual,np.clip(seed,arm.lo+1e-7,arm.hi-1e-7),bounds=(arm.lo+1e-8,arm.hi-1e-8),max_nfev=600,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        residual(fit.x);p,r=m.pose(arm.palm);delta=p-P0;rotation=float(np.linalg.norm(Rotation.from_matrix(R0.T@r).as_rotvec()))
        valid_pose=rotation<.002 and abs(fit.x[6]-wrist7)<.002 and fit.x[5]-amplitude>arm.lo[5]+.005 and fit.x[5]+amplitude<arm.hi[5]-.005
        if valid_pose:
            pose_shifts.append(float(np.linalg.norm(delta)))
            if np.linalg.norm(delta)<=max_shift:candidates.append((float(np.linalg.norm(delta)),delta.copy(),fit.x.copy()))
    if not candidates:
        data={'shift_bound':max_shift,'minimum_required_shift':min(pose_shifts) if pose_shifts else None}
        print('ALIGNMENT_REFUSED',data,flush=True)
        return Outcome(False,'no_safe_rocking_configuration',data,0),None
    _,delta,endpoint=min(candidates,key=lambda x:x[0])
    print('ADAPTIVE_ALIGNMENT',{'delta':delta.tolist(),'endpoint_joint6':float(endpoint[5]),'endpoint_joint7':float(endpoint[6])},flush=True)
    result,package=plan_align(cb,wrist7=wrist7,count=count,delta=delta)
    ratio=result.data.get('speed_ratio',0.)
    if ratio>.25 and result.data.get('lowest',0)>cb.table_z+.025:
        slower=int(np.ceil((count-1)*ratio/.25))+1
        if slower>600:return Outcome(False,'alignment_rate_guard',{'rows_needed':slower},0),None
        result,package=plan_align(cb,wrist7=wrist7,count=slower,delta=delta)
    return result,package
