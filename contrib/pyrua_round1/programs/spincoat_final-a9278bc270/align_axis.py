# s02: measured-palm wrist-7 alignment plus 10 mm root-y shift used before mixed at 530 in exploration event 0170. The source retains its E002-derived wrist-7 preset. Source: seeds/s02/code/explore/align_axis.py
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
    for a in r.arm_log:
        q=dict(p['base']);q.update(zip(cb.arm_names,a));cb.robot.fk(q);P,R=cb.robot.pose(cb.arms['left'].palm);maxp=max(maxp,float(np.linalg.norm(P-p['P0'])))
    print('ALIGN_TRACKING',{'max_palm_translation':maxp},flush=True)
    if r.episode_ended:return Outcome(False,'episode_ended',{},r.executed)
    cb.hold(25)
    return Outcome(True,'aligned',{},r.executed+25)
