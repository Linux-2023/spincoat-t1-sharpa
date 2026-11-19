"""Plan a measured tube entry; prefer the recorded s02 orientation.

Fallback30deg/30deg is the orientation that scored s06's original pickup.
No pose or joint target from an exploring layout is reused.
"""
import numpy as np
from pyrua.codebox.kin import Frame,Goal
from pyrua.codebox.outcome import Outcome


def plan_entry(cb,rim):
    arm=cb.arms['left'];hand=cb.hands['left'];path=hand.path('pad','index+middle')
    q=dict(zip(cb.arm_names,cb.q_arm));q.update(dict(zip(cb.dex_names,cb.q_dex)))
    virtual=q.copy();virtual.update(path.posture(.043));arm.base(virtual)
    point,opening,approach=path.frame(.043);frame=Frame(point,approach,opening)
    start=np.asarray(cb.q_arm[7:]).copy()
    templates=[(-45.,-35.2643897),(-30.,-30.),(-30.,-40.),(-45.,-50.)]
    attempts=[]
    hand_bodies=cb.robot.subtree(arm.palm)
    try:
        for yaw,pitch in templates:
            y,t=np.deg2rad([yaw,pitch])
            a=np.array([np.cos(y)*np.cos(t),np.sin(y)*np.cos(t),np.sin(t)])
            o=np.array([-np.sin(y),np.cos(y),0.])
            pre=np.asarray(rim)-[0,0,.04]-.045*a
            target=np.asarray(rim)-[0,0,.025]
            arm.at(start);p0,a0,o0=arm.where(frame)
            one,s1=arm.line(frame,start,Goal(p0,a0,o0),Goal(pre,a,o))
            if s1<.999:
                attempts.append([yaw,pitch,'pre',float(s1)]);continue
            arm.at(one[-1]);p0,a0,o0=arm.where(frame)
            two,s2=arm.line(frame,one[-1],Goal(p0,a0,o0),Goal(target,a,o))
            if s2<.999:
                attempts.append([yaw,pitch,'entry',float(s2)]);continue
            minimum=float('inf')
            for joints in one+two:
                arm.at(joints)
                height,_=cb.robot.lowest(hand_bodies)
                minimum=min(minimum,height)
            if minimum<max(cb.table_z+.10,float(rim[2])-.085):
                attempts.append([yaw,pitch,'clearance',minimum]);continue
            print('ENTRY_PLAN',yaw,pitch,'min_hand_z',minimum,'prior',attempts,flush=True)
            return Outcome(True,'entry_planned',{'approach':a.tolist(),'opening':o.tolist(),'pre':pre.tolist(),'target':target.tolist(),'yaw':yaw,'pitch':pitch,'min_hand_z':minimum})
        return Outcome(False,'entry_unreachable',{'attempts':attempts})
    finally:
        arm.base(q)
