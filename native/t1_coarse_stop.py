"""Coarse station arrival, physical braking, then zero wheel-speed commands.

Position feedback is irrevocably disabled on entering the coarse region. A
short speed ramp is braking, not position refinement. All fine placement is
left to the visual arm controller. No freejoint pose/velocity assignments.
"""
import types
from pathlib import Path
import numpy as np
from t1_mobile_stage import MobileDockController,advance_mobile_stage

_original_wheel_servos={}
ZERO_SPEED_GAIN_SCALE=4.

class CoarseStopController(MobileDockController):
    def __init__(self,m,target_xy,speed_limit=.09,hold_gain_scale=None):
        super().__init__(m,target_xy,speed_limit=speed_limit)
        self.station_xy=np.asarray(target_xy,dtype=float).copy()
        self.wheel_ids=np.array([i for i in range(m.nu) if m.actuator(i).name.startswith('astribot_chassis_motor_')])
        self.model=m;self.hold_gain_scale=float(ZERO_SPEED_GAIN_SCALE if hold_gain_scale is None else hold_gain_scale)
        self.wheel_dofs=m.jnt_dofadr[m.actuator_trnid[self.wheel_ids,0]]
        self.gain_records=[]
        # A second station uses the SAME runtime model. Recover its original
        # route servo rather than multiplying an already strengthened hold.
        key=id(m)
        if key not in _original_wheel_servos:
            _original_wheel_servos[key]=(m,m.actuator_gainprm[self.wheel_ids].copy(),m.actuator_biasprm[self.wheel_ids].copy())
        _,self.route_gain,self.route_bias=_original_wheel_servos[key]
        m.actuator_gainprm[self.wheel_ids]=self.route_gain;m.actuator_biasprm[self.wheel_ids]=self.route_bias
        self.mode='approach';self.brake_start=None;self.arrived_at=None
        self.coarse_radius_m=.010;self.brake_duration_s=.4;self.control_records=[]
    def step(self,d,target_xy=None,enable_gate=True):
        a,v=self.qa,self.dv;now=float(d.time)
        error=float(np.linalg.norm(d.qpos[a:a+2]-self.station_xy))
        # station_xy is frozen to the FINAL station at construction, never the
        # moving route waypoint. Latch at first actual station-region entry,
        # even if the nominal route has not finished, then never steer again.
        if self.mode=='approach' and error<=self.coarse_radius_m:
            self.mode='braking';self.brake_start=now;self.brake_from=d.ctrl[self.wheel_ids].copy()
            self.latch_error_m=error;self.integral[:]=0.;self.stable_duration=0.
        if self.mode=='approach':
            # Original route tracking. Its precision-stop gate is not used.
            super().step(d,target_xy,enable_gate=False)
            self.stopped=False
        else:
            dt=max(0.,now-self.last_time) if hasattr(self,'last_time') else 0.
            u=float(np.clip((now-self.brake_start)/self.brake_duration_s,0,1))
            blend=u*u*u*(10+u*(-15+6*u))
            # Encoder velocity damping only. No wheel-position or world-pose
            # target is introduced, and actuator force limits are unchanged.
            gain_scale=1+(self.hold_gain_scale-1)*blend
            self.model.actuator_gainprm[self.wheel_ids,0]=self.route_gain[:,0]*gain_scale
            self.model.actuator_biasprm[self.wheel_ids,2]=self.route_bias[:,2]*gain_scale
            if u>=1.:
                self.mode='zero_speed_hold'
                if self.arrived_at is None:self.arrived_at=now
                d.ctrl[self.wheel_ids]=0.
            else:d.ctrl[self.wheel_ids]=(1-blend)*self.brake_from
            stable=(self.mode=='zero_speed_hold' and error<=self.coarse_radius_m
                and np.linalg.norm(d.qvel[v:v+2])<.001 and abs(d.qvel[v+5])<.002)
            self.stable_duration=self.stable_duration+dt if stable else 0.
            self.stopped=self.stable_duration>=.5;self.last_time=now
        self.control_records.append([now,*d.qpos[a:a+2],*d.qvel[v:v+2],d.qvel[v+5],*d.ctrl[self.wheel_ids],
            float(self.mode=='approach'),float(self.mode=='zero_speed_hold'),float(self.stopped),error])
        self.gain_records.append([*self.model.actuator_gainprm[self.wheel_ids,0],
            *self.model.actuator_biasprm[self.wheel_ids,2],*d.qvel[self.wheel_dofs],*d.actuator_force[self.wheel_ids]])
        return self.stopped

_globals=dict(advance_mobile_stage.__globals__);_globals['MobileDockController']=CoarseStopController
_advance=types.FunctionType(advance_mobile_stage.__code__,_globals,name='advance_coarse_stop',argdefs=advance_mobile_stage.__defaults__,closure=advance_mobile_stage.__closure__)
_advance.__kwdefaults__=advance_mobile_stage.__kwdefaults__

def advance_coarse_stop(*args,**kwargs):
    logs,audit,dock=_advance(*args,**kwargs)
    audit.update(dock_controller='CoarseStopController',position_precision_requirement_replaced_by_user=True,
        gate_definition='coarse workspace XY<=10mm; actual planar speed<1mm/s and yaw rate<.002rad/s continuously .5s; zero wheel commands after braking',
        position_feedback_disabled_at_s=dock.brake_start,zero_speed_hold_started_at_s=dock.arrived_at,
        brake_duration_s=dock.brake_duration_s,coarse_latch_error_m=getattr(dock,'latch_error_m',None))
    audit['zero_speed_velocity_servo_gain_scale']=dock.hold_gain_scale
    return logs,audit,dock

def save_coarse_control(dock,path):
    rows=np.asarray(dock.control_records)
    names=['time','base_x','base_y','base_vx','base_vy','yaw_rate']+[f'wheel_command_{i}' for i in range(len(dock.wheel_ids))]+['world_pi_active','coarse_latched','stopped_gate','workspace_error_m']
    arrays={name:rows[:,i] for i,name in enumerate(names)}
    gains=np.asarray(dock.gain_records);n=len(dock.wheel_ids)
    for col,prefix in enumerate(['wheel_velocity_gain','wheel_velocity_bias','actual_wheel_speed','actual_wheel_force']):
        for i in range(n):arrays[f'{prefix}_{i}']=gains[:,col*n+i]
    np.savez_compressed(Path(path),**arrays)
    hold=rows[:,-3]>.5
    return dict(position_feedback_disabled_at_s=dock.brake_start,zero_speed_hold_started_at_s=dock.arrived_at,
        samples=len(rows),hold_samples=int(hold.sum()),max_wheel_command_after_hold=float(np.max(np.abs(rows[hold,6:6+len(dock.wheel_ids)]))) if hold.any() else None,
        zero_speed_velocity_servo_gain_scale=dock.hold_gain_scale,
        position_feedback_reenabled_after_braking=bool(np.any(rows[rows[:,0]>=dock.brake_start,-4]>.5)) if dock.brake_start is not None else None,
        gate_definition='XY<=10mm coarse workspace + original speed/yaw thresholds held .5s; no XY fine servo')
