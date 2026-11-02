"""Lower RIGHT post-aspiration carry target; no scene/physics/gate edits.

Transform only the isolated executed aspiration adapter before its compilation.
The unchanged arm IK and physical grip carry this lower configuration during
mobile transfer; no runtime robot/object pose writes are introduced.
"""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class RightTransportLowConfig:
    receiver_rim_z_m: float = .8560005
    tip_clearance_over_rim_m: float = .034
    minimum_dry_clearance_m: float = .005
    stand_bypass_x_m: float = .055
    lateral_start_clearance_over_rim_m: float = .020
    lateral_duration_s: float = 2.
    coating_transit_above_target_m: float = .024

def right_withdrawal_goal(source,config=None):
    c=config or RightTransportLowConfig()
    z=max(float(c.receiver_rim_z_m+c.tip_clearance_over_rim_m),float(source.initial_surface_z_m+c.minimum_dry_clearance_m))
    return np.array([source.center_xy_m[0]+c.stand_bypass_x_m,source.center_xy_m[1],z],dtype=float)

class RightLowWithdrawalPolicy:
    """Clear the cup vertically before side-stepping the tall pipette stand."""
    def __init__(self,source,config=None):
        self.config=config or RightTransportLowConfig();self.final_goal=right_withdrawal_goal(source,self.config)
        self.vertical_goal=self.final_goal.copy();self.vertical_goal[:2]=source.center_xy_m
        self.lateral_at=None;self.lateral_anchor=None
    def target(self,time,age,anchor,actual_tip):
        def smooth(u):
            u=float(np.clip(u,0,1));return u*u*u*(10+u*(-15+6*u))
        if self.lateral_at is None and actual_tip[2]>=self.config.receiver_rim_z_m+self.config.lateral_start_clearance_over_rim_m:
            self.lateral_at=float(time);self.lateral_anchor=np.asarray(actual_tip).copy()
        if self.lateral_at is None:
            u=smooth(age/3.);return (1-u)*anchor+u*self.vertical_goal
        u=smooth((time-self.lateral_at)/self.config.lateral_duration_s)
        return (1-u)*self.lateral_anchor+u*self.final_goal

def adapt_aspiration_source(source,config=None):
    """Returns (source_text, globals, config). Caller compiles its private copy."""
    c=config or RightTransportLowConfig();anchor='withdrawal_goal=np.r_[source.center_xy_m,.98]'
    if source.count(anchor)!=1:raise ValueError('Expected exactly one original .98m withdrawal target')
    motion='goal=(1-smooth(age/3))*anchor+smooth(age/3)*withdrawal_goal'
    if source.count(motion)!=1:raise ValueError('Expected exactly one original withdrawal interpolation')
    updated=source.replace(anchor,'right_transport_policy=RightLowWithdrawalPolicy(source,RIGHT_TRANSPORT_LOW_CONFIG)\n    withdrawal_goal=right_transport_policy.final_goal.copy()')
    updated=updated.replace(motion,'goal=right_transport_policy.target(t,age,anchor,d.site_xpos[tip])')
    return updated,dict(RightLowWithdrawalPolicy=RightLowWithdrawalPolicy,RIGHT_TRANSPORT_LOW_CONFIG=c),asdict(c)

def adapt_coating_source(source,config=None):
    """Lower transit only; final12mm precise placement and its gates stay intact."""
    c=config or RightTransportLowConfig();anchor='target+[.0001,0,.08]'
    if source.count(anchor)!=1:raise ValueError('Expected exactly one coating transit target')
    return source.replace(anchor,'target+[.0001,0,'+repr(float(c.coating_transit_above_target_m))+']')
