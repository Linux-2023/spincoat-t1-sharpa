"""V26 larger lateral wrist oscillation with failure-preserving partial traces.
First five arm targets stay fixed. No object-state or physical-parameter writes.
"""
from dataclasses import dataclass,asdict
import numpy as np
from t1_spincoat_scene import arm_indices

@dataclass(frozen=True)
class MixingConfig:
    tube_length_m:float=.15
    frequency_hz:float=3.
    amplitude_deg:float=5.
    record_dir:str=""
    steady_s:float=6.
    ramp_s:float=1.
    wrist6_weight:float=.12236061555
    wrist7_weight:float=1.
    prefilter_strength:float=1.
    phase_name:str='human_wrist_left_right_high_frequency'
    @property
    def duration_s(self):return self.steady_s+2*self.ramp_s

def angle_state(t,c):
    if t<=0 or t>=c.duration_s:return np.zeros(3)
    if t<c.ramp_s:
        u=t/c.ramp_s;sign=1.
    elif t>c.duration_s-c.ramp_s:
        u=(c.duration_s-t)/c.ramp_s;sign=-1.
    else:
        u=1.;sign=0.
    e=u**3*(10+u*(-15+6*u))
    de=sign*30*u*u*(1-u)**2/c.ramp_s
    dde=(60*u-180*u*u+120*u**3)/c.ramp_s**2 if sign else 0.
    w=2*np.pi*c.frequency_hz;a=np.radians(c.amplitude_deg)
    s=np.sin(w*t);co=np.cos(w*t)
    return a*np.array([e*s,de*s+e*w*co,dde*s+2*de*w*co-e*w*w*s])

def command_plan(m,d,arm_command,c):
    if not (0<c.frequency_hz<=8 and 0<c.amplitude_deg<=15 and c.ramp_s>0 and c.steady_s>0 and 0<=c.prefilter_strength<=1):
        raise ValueError('Invalid oscillation configuration')
    ids,qa,va=arm_indices(m,'left')
    acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in ids])
    dt=m.opt.timestep;count=round(c.duration_s/dt)
    states=np.array([angle_state(t,c) for t in np.arange(count+1)*dt])
    direction=np.array([0.,0.,0.,0.,0.,c.wrist6_weight,c.wrist7_weight])
    reference=np.asarray(arm_command)[None,:]+states[:,0,None]*direction
    commands=reference.copy()
    for j in (5,6):
        gain=float(m.actuator_gainprm[acts[j],0]);damp=-float(m.actuator_biasprm[acts[j],2])
        passive=float(m.dof_damping[va[j]]);inertia=float(m.dof_armature[va[j]])
        if gain<=0:raise ValueError('Positive servo gain required')
        correction=((damp+passive)*states[:,1]+inertia*states[:,2])*direction[j]/gain*c.prefilter_strength
        target=np.clip(reference[:,j]+correction,reference[:,j].min(),reference[:,j].max())
        for k in range(1,count+1):commands[k,j]=commands[k-1,j]+np.clip(target[k]-commands[k-1,j],-2*dt,2*dt)
    return ids,qa,va,acts,states,reference,commands

def advance_mouth_pivot_shake(m,d,handles,arm_command,step_callback,*,config=None,side='left',ik_solver=None):
    c=config or MixingConfig()
    if side!='left':raise ValueError('LEFT wrist only')
    ids,qa,va,acts,states,reference,commands=command_plan(m,d,arm_command,c)
    dt=m.opt.timestep;count=len(commands)-1
    margin=float(np.minimum(commands-m.jnt_range[ids,0],m.jnt_range[ids,1]-commands).min())
    limited=m.actuator_ctrllimited[acts].astype(bool)
    cm=float(np.minimum(commands[:,limited]-m.actuator_ctrlrange[acts[limited],0],m.actuator_ctrlrange[acts[limited],1]-commands[:,limited]).min())
    peak=float(np.max(abs(np.diff(commands,axis=0)/dt)))
    if peak>2+1e-9 or margin<.005 or cm<0:raise RuntimeError('Original command speed/range guards failed')
    mouth=m.site(handles['tube_mouth_site']).id;tube=m.body(handles['tube_body']).id
    origin=d.site_xpos[mouth].copy();start=float(d.time)
    shapes={'time':(), 'mouth_world_m':(3,), 'arm_qpos':(7,), 'arm_qvel':(7,), 'arm_actuator_force':(7,), 'bottom_world_m':(3,), 'arm_qacc':(7,), 'arm_passive_force':(7,), 'arm_bias_force':(7,), 'arm_applied_force':(7,), 'command_angle_velocity_acceleration':(3,), 'servo_target_rad':(7,), 'reference_target_rad':(7,)}
    arrays={key:np.empty((count,*shape)) for key,shape in shapes.items()}
    failure=None
    filled=0
    for k in range(count):
        before=float(d.time)
        try:step_callback(commands[k+1],c.phase_name,True)
        except (RuntimeError,ValueError) as exc:failure=exc
        if failure is not None and abs(d.time-before)<1e-12:break
        if abs(d.time-before-dt)>1e-9:raise RuntimeError('Expected one physical step')
        values=(d.time,d.site_xpos[mouth],d.qpos[qa],d.qvel[va],d.actuator_force[acts],d.xpos[tube],d.qacc[va],d.qfrc_passive[va],d.qfrc_bias[va],d.qfrc_applied[va],states[k+1],commands[k+1],reference[k+1])
        for key,value in zip(shapes,values):arrays[key][k]=value
        filled=k+1
        if failure is not None:break
    if filled==0 and failure is not None:raise failure
    if filled<count:arrays={key:value[:filled].copy() for key,value in arrays.items()}
    report=dict(config=asdict(c),duration_s=c.duration_s,execution_start_s=start,execution_end_s=float(d.time),execution_completed=failure is None,physical_success_not_inferred=True,actual_mouth_excursion_max_mm=float(np.linalg.norm(arrays['mouth_world_m']-origin,axis=1).max()*1000),actual_bottom_vertical_range_mm=float(np.ptp(arrays['bottom_world_m'][:,2])*1000),mouth_pivot_world_m=origin.tolist(),proximal_targets_unchanged=True,runtime_object_pose_writes=False,sampled_physics_timestep_peak_command_velocity_rad_s=peak,sampled_peak_angular_acceleration_rad_s2=float(np.abs(states[:,2]).max()),sampled_command_joint_margin_rad=margin,scope=__doc__)
    if failure is not None:
        import json
        from pathlib import Path
        report["failure"]=type(failure).__name__+": "+str(failure)
        if c.record_dir:
            out=Path(c.record_dir)
            (out/"mixing_report.json").write_text(json.dumps(report,indent=2))
            np.savez_compressed(out/"mixing_actual_trace.npz",**arrays)
        raise failure
    return commands[-1].copy(),report,arrays
