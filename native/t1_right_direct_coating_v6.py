"""Opt-in v6 straight high-clearance coater transfer before vertical lowering.
Uses the existing RGB target and privileged simulator tip feedback, no touch.
Original precise-alignment, grasp, collision, fluid and press gates unchanged.
"""
import numpy as np
from t1_right_transport_low_v3 import adapt_coating_source as _low

class RightDirectCoatingPolicyV6:
    def __init__(self,transit_height_over_target_m=.024,maximum_tip_reference_speed_m_s=.05):
        self.height=float(transit_height_over_target_m);self.maximum_speed=float(maximum_tip_reference_speed_m_s);self.stage='raise';self.started=None;self.anchor=None;self.direct_at=None;self.direct_anchor=None;self.direct_duration=None;self.lower_at=None;self.lower_anchor=None;self.trace=[]
    @staticmethod
    def smooth(u):
        u=float(np.clip(u,0.,1.));return u**3*(10+u*(-15+6*u))
    def target(self,time,actual_tip,target):
        p=np.asarray(actual_tip);target=np.asarray(target);transit=target+np.array([.0001,0.,self.height]);final=target+np.array([.0001,0.,.012])
        if self.started is None:self.started=float(time);self.anchor=p.copy()
        if self.stage=='raise':
            raised=np.r_[self.anchor[:2],transit[2]];u=self.smooth((time-self.started)/.6);goal=(1-u)*self.anchor+u*raised
            if time-self.started>=.6 and abs(p[2]-transit[2])<.001:
                self.stage='direct_transfer';self.direct_at=float(time);self.direct_anchor=p.copy();self.direct_duration=max(2.,1.875*float(np.linalg.norm(transit-self.direct_anchor))/self.maximum_speed)
        if self.stage=='direct_transfer':
            age=time-self.direct_at;u=self.smooth(age/self.direct_duration);goal=(1-u)*self.direct_anchor+u*transit
            if age>=self.direct_duration and np.linalg.norm(p[:2]-transit[:2])<.0015 and abs(p[2]-transit[2])<.001:
                self.stage='vertical_lower';self.lower_at=float(time);self.lower_anchor=p.copy()
        if self.stage=='vertical_lower':
            u=self.smooth((time-self.lower_at)/2.);goal=(1-u)*self.lower_anchor+u*final
        self.trace.append(dict(time_s=float(time),phase=self.stage,actual_tip_world_m=p.tolist(),goal_tip_world_m=goal.tolist(),direct_duration_s=self.direct_duration));return goal

def adapt_direct_coating_source(source,config=None):
    source=_low(source)
    candidates=['    desired=d.qpos.copy();desired_hand=current_hand_controls(m,d,hj)','    desired=d.qpos.copy();desired_hand=close.copy()']
    matches=[text for text in candidates if source.count(text)==1]
    if len(matches)!=1:raise RuntimeError('Unique original/live coating desired-command anchor required')
    anchor=matches[0]
    replacement="    desired=d.qpos.copy()\n    for robot_act in range(m.nu):\n        robot_joint=int(m.actuator_trnid[robot_act,0])\n        if robot_joint>=0 and not m.actuator(robot_act).name.startswith('astribot_chassis_motor_') and m.joint(robot_joint).name.startswith(('astribot_','sharpa_')):\n            desired[int(m.jnt_qposadr[robot_joint])]=d.ctrl[robot_act]\n    desired_hand=current_hand_controls(m,d,hj)"
    source=source.replace(anchor,replacement)
    a='    for step in range(round((args.max_seconds+prefix)/.001)):'
    if source.count(a)!=1:raise RuntimeError('Original coater loop anchor must occur exactly once')
    source=source.replace(a,"    right_direct_coating_policy=RightDirectCoatingPolicyV6()\n    globals()['right_direct_coating_policy']=right_direct_coating_policy\n"+a)
    a='                # Slowly correct measured pipe orientation and nozzle position.'
    if source.count(a)!=1:raise RuntimeError('Original coater tool servo anchor must occur exactly once')
    source=source.replace(a,"                if phase in ('translate','lower','settle','wait_for_motion_ready','wait_for_base_stop'):\n                    goal_tip=right_direct_coating_policy.target(t,current_tip,target)\n"+a)
    return 'from t1_right_direct_coating_v6 import RightDirectCoatingPolicyV6\n'+source
