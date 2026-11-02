"""Upper-tube-pivot mixing targets, inspired by university lab teaching.

Only robot actuator targets are returned. No live qpos/qvel/time/object pose is
assigned. Amplitude/frequency are engineering defaults, not video measurements.
The caller retains its physics step, fingers, dock, ledger and safety gates.
"""
from dataclasses import dataclass,asdict
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.interpolate import make_interp_spline
from t1_spincoat_scene import arm_indices
from probe_t1_reagent_reach import solve_arm


@dataclass(frozen=True)
class MixingConfig:
    duration_s:float=2.4
    amplitude_deg:float=3.
    frequency_hz:float=1.5
    ramp_s:float=.4
    knot_dt_s:float=.02
    tube_length_m:float=.150
    maximum_entry_tilt_deg:float=10.
    # Rotate around world X so the tube bottom moves approximately in world Y.
    rotation_axis_world:tuple=(1.,0.,0.)


def quintic(u):
    u=np.clip(u,0.,1.);return u*u*u*(10.+u*(-15.+6.*u))


def angle_at(t,c):
    """C2 envelope: no full-amplitude or acceleration jump at stage boundaries."""
    if c.duration_s<=2*c.ramp_s or c.ramp_s<=0:raise ValueError('duration must exceed twice positive ramp')
    t=float(np.clip(t,0,c.duration_s))
    envelope=quintic(t/c.ramp_s)*quintic((c.duration_s-t)/c.ramp_s)
    return np.deg2rad(c.amplitude_deg)*envelope*np.sin(2*np.pi*c.frequency_hz*t)


def plan_mouth_pivot_shake(m,d,handles,arm_command,*,config=None,side='right',ik_solver=None):
    """Static IK planning from the CURRENT held state; not a dynamic certificate.

    Actual mouth and hand-to-tube transform are used as simulation controller
    state, not presented as camera estimates or tactile policy feedback.
    """
    if side not in ['left','right']:raise ValueError('side must be left or right')
    c=config or MixingConfig();_,qa,_=arm_indices(m,side)
    work=mujoco.MjData(m);work.qpos[:]=d.qpos;mujoco.mj_kinematics(m,work)
    pb=m.body('sharpa_'+side+'/'+side+'_hand_C_MC').id;tb=m.body(handles['tube_body']).id
    mouth=m.site(handles['tube_mouth_site']).id
    Rh=work.xmat[pb].reshape(3,3).copy();Rt=work.xmat[tb].reshape(3,3).copy()
    pivot=work.site_xpos[mouth].copy();mouth_in_palm=Rh.T@(pivot-work.xpos[pb])
    tube_in_palm=Rh.T@Rt;axis=np.array(c.rotation_axis_world,dtype=float);axis/=np.linalg.norm(axis)
    entry_tilt=float(np.degrees(np.arccos(np.clip(Rt[2,2],-1,1))))
    if entry_tilt>c.maximum_entry_tilt_deg:raise ValueError(f'tube not near upright: {entry_tilt:.3f}deg')
    if abs(axis@Rt[:,2])>.1:raise ValueError('mixing rotation axis must be transverse to upright tube')
    times=np.linspace(0,c.duration_s,int(np.ceil(c.duration_s/c.knot_dt_s))+1)
    positions=[];rotations=[];q=[];errors=[];seed=work.qpos[qa].copy()
    for t in times:
        desired_tube=Rotation.from_rotvec(axis*angle_at(t,c)).as_matrix()@Rt
        desired_hand=desired_tube@tube_in_palm.T;point=pivot-desired_hand@mouth_in_palm
        solved,error=(solve_arm if ik_solver is None else ik_solver)(m,work,point,desired_hand,side=side,seed=seed)
        if error['position_error_mm']>.5 or error['orientation_error_deg']>.5:
            raise RuntimeError('mouth-pivot IK rejected '+str(error))
        positions.append(point);rotations.append(desired_hand);q.append(solved);errors.append(error);seed=solved
    q=np.asarray(q)
    # Preserve initial PD/grasp load: apply differential joint motion around
    # the existing command rather than abruptly unloading to actual joint q.
    commands=np.asarray(arm_command)[None,:]+q-q[0]
    # A closed Cartesian loop can end at another valid7DoF configuration.
    # Keep the continuous IK endpoint for the opt-in solver; resetting that
    # single knot would create a joint-space jump at the end of the shake.
    if ik_solver is None:commands[-1]=np.asarray(arm_command)
    spline=make_interp_spline(times,commands,k=5,bc_type=([(1,np.zeros(7)),(2,np.zeros(7))],[(1,np.zeros(7)),(2,np.zeros(7))]))
    dense=np.linspace(0,c.duration_s,max(2,int(round(c.duration_s/m.opt.timestep))+1))
    dense_commands=np.asarray(spline(dense));peak_speed=float(abs(spline(dense,1)).max())
    armids,_,_=arm_indices(m,side)
    joint_margin=float(np.minimum(dense_commands-m.jnt_range[armids,0],m.jnt_range[armids,1]-dense_commands).min())
    armacts=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in armids])
    limited=m.actuator_ctrllimited[armacts].astype(bool)
    ctrl_margin=float(np.minimum(dense_commands[:,limited]-m.actuator_ctrlrange[armacts[limited],0],m.actuator_ctrlrange[armacts[limited],1]-dense_commands[:,limited]).min()) if limited.any() else None
    if ik_solver is not None and (peak_speed>2. or joint_margin<.005 or (ctrl_margin is not None and ctrl_margin<0.)):
        raise RuntimeError('Opt-in mixing target spline rejected: speed/joint/actuator-range bound; no physics executed')
    report=dict(config=asdict(c),side=side,scope='IK target plan only; actuator execution and actual held/base/collision gates must be checked separately.',
                reference='https://www2.chem.wisc.edu/deptfiles/genchem/lab/labdocs/modules/mixing/mixman.htm',
                evidence='For less-than-half-full tubes university text describes keeping the upper tube in one place while moving its bottom left/right. Embedded video is a different two-hand finger-flick demonstration.',
                parameter_provenance='Amplitude, frequency and duration are explicit uncalibrated engineering defaults; not recovered from video.',
                mouth_pivot_world_m=pivot.tolist(),entry_tube_tilt_deg=entry_tilt,
                mouth_in_palm_m=mouth_in_palm.tolist(),tube_in_palm_rotation=tube_in_palm.tolist(),
                approximate_bottom_peak_excursion_m=c.tube_length_m*np.sin(np.deg2rad(c.amplitude_deg)),
                maximum_IK_position_error_mm=max(e['position_error_mm'] for e in errors),
                maximum_IK_orientation_error_deg=max(e['orientation_error_deg'] for e in errors),
                minimum_IK_joint_margin_rad=min(e['joint_margin_rad'] for e in errors),
                runtime_object_pose_writes=False,hand_target_changes=False,chemical_homogeneity_certified=False,
                ik_solver='original' if ik_solver is None else ik_solver.__module__+'.'+ik_solver.__name__,
                final_joint_target_forced_to_entry=ik_solver is None,
                final_target_minus_entry_rad=(commands[-1]-np.asarray(arm_command)).tolist(),
                sampled_physics_timestep_peak_command_velocity_rad_s=peak_speed,
                sampled_command_joint_margin_rad=joint_margin,sampled_actuator_control_margin_rad=ctrl_margin)
    return dict(time=times,kinematic_q=q,arm_commands=commands,palm_positions=np.asarray(positions),
                palm_rotations=np.asarray(rotations),spline=spline,report=report)


def advance_mouth_pivot_shake(m,d,handles,arm_command,step_callback,*,config=None,side='right',ik_solver=None):
    """Execute through caller's existing physics loop.

    step_callback(q_desired, phase, allow_mix) must assign arm actuator ctrl and
    call the existing sample()/mj_step once. It must retain finger preload,
    zero-wheel dock control, gravity compensation and original ledger checks.
    No reset, direct object-state write, or new tactile policy is introduced.
    """
    c=config or MixingConfig();plan=plan_mouth_pivot_shake(m,d,handles,arm_command,config=c,side=side,ik_solver=ik_solver)
    start=float(d.time);mouth=m.site(handles['tube_mouth_site']).id;pivot=np.asarray(plan['report']['mouth_pivot_world_m'])
    actual=[]
    for k in range(int(round(c.duration_s/m.opt.timestep))):
        t=min((k+1)*m.opt.timestep,c.duration_s)
        previous_time=float(d.time)
        step_callback(np.asarray(plan['spline'](t)), 'gentle_shake_mouth_pivot', True)
        if abs(float(d.time)-previous_time-m.opt.timestep)>1e-9:
            raise RuntimeError('mixing callback must advance the existing live physics by exactly one timestep')
        actual.append((float(d.time),d.site_xpos[mouth].copy()))
    report=dict(plan['report'],execution_start_s=start,execution_end_s=float(d.time),
                actual_mouth_excursion_max_mm=float(max(np.linalg.norm(p-pivot)*1000 for _,p in actual)),
                execution_completed=True,physical_success_not_inferred=True)
    final_command=plan['arm_commands'][-1].copy()
    armids,_,_=arm_indices(m,side);armacts=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in armids])
    report['returned_command_minus_actual_last_ctrl_max_rad']=float(abs(final_command-d.ctrl[armacts]).max())
    if ik_solver is not None and report['returned_command_minus_actual_last_ctrl_max_rad']>1e-8:
        raise RuntimeError('Mixing command handoff is discontinuous')
    return final_command,report,dict(time=np.asarray([t for t,_ in actual]),mouth_world_m=np.asarray([p for _,p in actual]))
