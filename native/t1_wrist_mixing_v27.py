"""V27 scratch-kinematics mouth-pivot compensation; unchanged physical limits.
Wrist-preferred seven-joint IK keeps tube mouth fixed while rotating about world X.
No live object state assignments. Simulator geometry is a planning input, not vision.
"""
from dataclasses import dataclass,asdict
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from t1_wrist_mixing_v26 import MixingConfig as BaseConfig,angle_state
from t1_spincoat_scene import arm_indices

@dataclass(frozen=True)
class MixingConfig(BaseConfig):
    pass

def command_plan(m,d,arm_command,c):
    if not (0<c.frequency_hz<=8 and 0<c.amplitude_deg<=15 and c.ramp_s>0 and c.steady_s>0 and 0<=c.prefilter_strength<=1):
        raise ValueError('Invalid oscillation configuration')
    ids,qa,va=arm_indices(m,'left')
    acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in ids])
    pb=m.body('sharpa_left/left_hand_C_MC').id
    mouth=m.site('prep/tube_mouth').id
    work=mujoco.MjData(m);work.qpos[:]=d.qpos
    mujoco.mj_kinematics(m,work)
    H=work.xmat[pb].reshape(3,3).copy()
    pivot=work.site_xpos[mouth].copy()
    local=H.T@(pivot-work.xpos[pb])
    q0=d.qpos[qa].copy();offset=np.asarray(arm_command)-q0
    angles=np.linspace(-np.radians(c.amplitude_deg),np.radians(c.amplitude_deg),81)
    knots=[];errors=[];rotation_errors=[]
    lo=m.jnt_range[ids,0]+.005
    hi=m.jnt_range[ids,1]-.005
    posture_weight=np.array([.1,.1,.1,.1,.1,.01,.01])
    for angle in angles:
        target_R=Rotation.from_rotvec(np.array([angle,0.,0.])).as_matrix()@H
        def residual(q):
            work.qpos[qa]=q;mujoco.mj_kinematics(m,work)
            R=work.xmat[pb].reshape(3,3)
            point=work.xpos[pb]+R@local
            return np.r_[1000*(point-pivot),100*Rotation.from_matrix(target_R@R.T).as_rotvec(),posture_weight*(q-q0)]
        fit=least_squares(residual,q0,bounds=(lo,hi),diff_step=1e-4,max_nfev=80,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        r=residual(fit.x)
        errors.append(np.linalg.norm(r[:3]));rotation_errors.append(np.linalg.norm(r[3:6])/100)
        knots.append(fit.x+offset)
    if max(errors)>.5:raise RuntimeError('Mouth-pivot planning residual exceeds0.5mm')
    if max(rotation_errors)>np.radians(.1):raise RuntimeError('Lateral orientation planning residual exceeds0.1deg')
    dt=m.opt.timestep;count=round(c.duration_s/dt)
    states=np.array([angle_state(t,c) for t in np.arange(count+1)*dt])
    reference=CubicSpline(angles,np.array(knots),axis=0)(states[:,0])
    speed=np.gradient(reference,dt,axis=0);acceleration=np.gradient(speed,dt,axis=0)
    commands=reference.copy()
    for j in range(7):
        gain=float(m.actuator_gainprm[acts[j],0]);damp=-float(m.actuator_biasprm[acts[j],2])
        if gain<=0:raise ValueError('Positive servo gain required')
        correction=((damp+m.dof_damping[va[j]])*speed[:,j]+m.dof_armature[va[j]]*acceleration[:,j])/gain*c.prefilter_strength
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
    report=dict(config=asdict(c),duration_s=c.duration_s,execution_start_s=start,execution_end_s=float(d.time),execution_completed=failure is None,physical_success_not_inferred=True,actual_mouth_excursion_max_mm=float(np.linalg.norm(arrays['mouth_world_m']-origin,axis=1).max()*1000),actual_bottom_vertical_range_mm=float(np.ptp(arrays['bottom_world_m'][:,2])*1000),mouth_pivot_world_m=origin.tolist(),proximal_targets_unchanged=False,runtime_object_pose_writes=False,sampled_physics_timestep_peak_command_velocity_rad_s=peak,sampled_peak_angular_acceleration_rad_s2=float(np.abs(states[:,2]).max()),sampled_command_joint_margin_rad=margin,scope=__doc__)
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
