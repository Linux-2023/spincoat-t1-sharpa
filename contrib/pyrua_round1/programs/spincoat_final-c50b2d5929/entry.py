"""Reach a measured tube with bounded, checked orientation freedom when needed.
The free horizontal-opening solve follows the robot-only entry methods of s03/s04.
"""
import numpy as np
from scipy.optimize import least_squares
from pyrua.codebox.outcome import Outcome


def arrived(cb,out,target):
    if out:return out
    if out.reason=='off_target' and not cb.ended and out.data.get('error',1)<.015:
        cb.hold(10)
        error=float(np.linalg.norm(np.asarray(cb.state()['left']['grasp_point'])-target))
        if error<.004:return Outcome(True,'settled_at_target',{'error':error},out.steps+10)
    return out


def free_entry(cb,rim,nominal):
    opened=cb.shape('left','pad',width=.025,fingers=['index','middle'],clearance=.018)
    if not opened and opened.reason not in ('blocked',):return opened
    obs=cb.robo.observe();arm=cb.arms['left'];model=cb.robot
    grip=cb.grip['left'];point,opening,approach=(np.array(v,copy=True) for v in grip['path'].frame(grip['scale']))
    base=dict(zip(cb.arm_names,obs.arm_q));base.update(zip(cb.dex_names,obs.dex_q));start=obs.arm_q[7:].copy()
    candidates=[]
    for depth in (.025,.020):
        target=np.asarray(rim)-[0,0,depth]
        def residual(q):
            state=dict(base);state.update(zip(arm.names,q));model.fk(state);P,R=model.pose(arm.palm)
            return np.r_[10*(P+R@point-target),(R@opening)[2],.05*(R@approach-nominal),.005*(q-start)]
        fit=least_squares(residual,np.clip(start,arm.lo+1e-7,arm.hi-1e-7),bounds=(arm.lo+1e-8,arm.hi-1e-8),max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        residual(fit.x);P,R=model.pose(arm.palm);app=(R@approach).copy();op=(R@opening).copy();error=float(np.linalg.norm(P+R@point-target))
        turn=float(np.arccos(np.clip(app@nominal,-1,1)))
        if error>.0015 or abs(op[2])>.03 or turn>.4:continue
        pre=np.asarray(rim)-[0,0,.04]-.045*app
        def pre_residual(q):
            state=dict(base);state.update(zip(arm.names,q));model.fk(state);P,R=model.pose(arm.palm)
            return np.r_[10*(P+R@point-pre),R@approach-app,R@opening-op,.001*(q-fit.x)]
        before=least_squares(pre_residual,np.clip(fit.x,arm.lo+1e-7,arm.hi-1e-7),bounds=(arm.lo+1e-8,arm.hi-1e-8),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        e=pre_residual(before.x)
        if np.linalg.norm(e[:3])>.02 or np.linalg.norm(e[3:9])>.01:continue
        segments=[];previous=obs.arm_cmd.copy();minimum=10.;valid=True
        for end in (before.x,fit.x):
            target_arm=previous.copy();target_arm[7:]=end
            count=max(40,int(np.ceil(1.5*np.max(np.abs(end-previous[7:])/arm.speed)*25))+1)
            t=np.linspace(0,1,count);t=t*t*(3-2*t)
            rows=previous+(target_arm-previous)*t[:,None]
            for row in rows:
                state=dict(base);state.update(zip(cb.arm_names,row));model.fk(state);minimum=min(minimum,float(arm.lowest()[0]))
                if minimum<cb.table_z+.025:valid=False;break
            if not valid:break
            segments.append(rows);previous=target_arm
        if valid:
            candidates.append((turn,depth,target,app,op,minimum,segments))
    if not candidates:return Outcome(False,'entry_unreachable',{},0)
    _,depth,target,app,op,minimum,segments=min(candidates,key=lambda x:(0 if x[1]==.025 else 1,x[0]))
    print('FREE_ENTRY',{'depth':depth,'approach':app.tolist(),'opening':op.tolist(),'minimum_z':minimum},flush=True)
    begin=cb.steps_used
    for rows in segments:
        moved=cb.robo.step(arm=rows);cb.sync()
        if moved.episode_ended:return Outcome(False,'episode_ended',{},moved.executed)
        cb.hold(10)
    final=np.asarray(cb.state()['left']['grasp_point']);error=float(np.linalg.norm(final-target))
    if error>.005:return Outcome(False,'entry_off_target',{'error':error},cb.steps_used-begin)
    return Outcome(True,'entered_free',{'depth':depth,'error':error},cb.steps_used-begin)


def enter(cb,rim):
    nominal=np.array([1.,-1.,-1.])/np.sqrt(3);pre=np.asarray(rim)-[0,0,.04]-.045*nominal
    out=cb.move_to('left',pre,approach=nominal,open=[1,1,0],shape={'grasp':'pad','width':.025,'fingers':['index','middle'],'clearance':.018})
    out=arrived(cb,out,pre)
    if not out:
        if out.steps==0 and out.reason in ('ik_unreachable','needs_reconfigure'):
            return free_entry(cb,rim,nominal)
        return out
    for depth in (.025,.020):
        target=np.asarray(rim)-[0,0,depth]
        planned=cb.move_to('left',target,approach=nominal,open=[1,1,0],path='line',plan=True)
        if planned:
            print('ENTRY_DEPTH',depth,flush=True)
            return arrived(cb,cb.move_to('left',target,approach=nominal,open=[1,1,0],path='line'),target)
    return free_entry(cb,rim,nominal)
