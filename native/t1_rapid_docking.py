"""Optional faster-converging dock controller; original engineering gate intact."""
import types
import numpy as np
from t1_mobile_stage import MobileDockController,advance_mobile_stage

class RapidDockController(MobileDockController):
    """Change position/integral gains only; preserve route, speeds and gate."""
    def step(self,d,target_xy=None,enable_gate=True):
        if target_xy is not None:self.target=np.asarray(target_xy)
        a,v=self.qa,self.dv;yaw=self.driver._yaw_from_quat(*d.qpos[a+3:a+7])
        err=self.target-d.qpos[a:a+2]
        dt=float(d.time-self.last_time) if hasattr(self,'last_time') else 0.
        if enable_gate:self.integral=np.clip(self.integral+err*dt,-.005,.005)
        cmd=np.clip(3.*err+.2*self.integral-.5*d.qvel[v:v+2],-self.speed_limit,self.speed_limit)
        c,sn=np.cos(yaw),np.sin(yaw);body=np.array([[c,sn],[-sn,c]])@cmd
        wz=np.clip(-2*yaw-.4*d.qvel[v+5],-.2,.2)
        self.driver.apply(d,-np.r_[body,wz])
        stable=enable_gate and np.linalg.norm(err)<.001 and np.linalg.norm(d.qvel[v:v+2])<.001 and abs(d.qvel[v+5])<.002
        self.stable_duration=self.stable_duration+float(d.time-self.last_time) if stable and hasattr(self,'last_time') else 0.
        self.last_time=float(d.time);self.stopped=self.stable_duration>=.5
        return self.stopped

_globals=dict(advance_mobile_stage.__globals__)
_globals['MobileDockController']=RapidDockController
_advance=types.FunctionType(advance_mobile_stage.__code__,_globals,name='advance_rapid_docking',argdefs=advance_mobile_stage.__defaults__,closure=advance_mobile_stage.__closure__)
_advance.__kwdefaults__=advance_mobile_stage.__kwdefaults__

def advance_rapid_docking(*args,**kwargs):
    logs,audit,dock=_advance(*args,**kwargs)
    audit.update(dock_controller='RapidDockController',position_gain=3.,integral_gain=.2,velocity_damping=.5,
        original_gate_preserved='XY<1mm, planar speed<1mm/s, yaw rate<.002rad/s continuously .5s')
    return logs,audit,dock
