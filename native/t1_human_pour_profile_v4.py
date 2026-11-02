"""Independent v4 side-grip, near-rim human-pour planner. Profiles require own static+dynamic validation; original fluid/physical gates untouched."""
import numpy as np
from scipy.spatial.transform import Rotation

REFERENCES = [
 dict(title="Lisa Nichols, Butte College: Transferring Methods - Liquids",url="https://chem.libretexts.org/Bookshelves/Organic_Chemistry/Organic_Chemistry_Lab_Techniques_(Nichols)/01:_General_Techniques/1.03:_Transferring_Methods/1.3B:_Transferring_Methods_-_Liquids",basis="Direct transfer for volumes above5mL; receiving/donating glass close or touching controls pouring. This is a teaching principle, not measured robot trajectories or a mandate for glass contact."),
 dict(title="Claremont Colleges: Common Laboratory Techniques",url="https://ccdl.claremont.edu/digital/api/collection/cer/id/166/download",basis="A glass rod may guide liquid transfer and prevent liquid running down the vessel. No rod is present here; no claim this robot implements rod-guided technique."),
 dict(title="Beloit College Chem117: Using a chemical reagent bottle",url="https://chem.beloit.edu/classes/Chem117/SlideShow/slides/techniques/reagent_bottle_mov.html",basis="Primary public teaching video index, G.Lisensky,12s. Reference uses a reagent bottle, not this25x150mm tube; no quantitative motion was extracted.")]

def axial_yaw_deg(angle,profile):
    knots=profile.get('axial_yaw_waypoints_deg')
    if knots is None:return float(profile['transport_yaw_deg'])
    from scipy.interpolate import PchipInterpolator
    k=np.asarray(knots,dtype=float)
    return float(PchipInterpolator(k[:,0],k[:,1])(angle))

def mouth_position(angle,profile):
    if profile.get('mouth_xyz_waypoints_m') is not None:
        from scipy.interpolate import PchipInterpolator
        knots=np.asarray(profile['mouth_xyz_waypoints_m'],dtype=float)
        return np.asarray(PchipInterpolator(knots[:,0],knots[:,1:],axis=0)(angle))
    knots=np.asarray(profile['mouth_height_waypoints_m'],dtype=float)
    return np.r_[np.asarray(profile['mouth_xy_m']),np.interp(angle,knots[:,0],knots[:,1])]

def make_profile_v4(receiver_xy,rim_z_m,*,tilt_axis_azimuth_deg,transport_yaw_deg,mouth_gap_m=.020,axial_yaw_waypoints_deg=None):
    """Explicit engineering candidate, not a validated human trajectory or geometry pass."""
    az=np.deg2rad(tilt_axis_azimuth_deg);flow=np.array([np.sin(az),-np.cos(az)])
    low=float(rim_z_m+mouth_gap_m);knots=[[0.,1.02],[25.,.97],[50.,.92],[60.,.89],[75.,low],[100.,low]]
    return dict(name='human_pour_v4_side_grip_low_rim',receiver_xy_m=list(receiver_xy),mouth_xy_m=(np.asarray(receiver_xy)-.010*flow).tolist(),mouth_xy_offset_m=(-.010*flow).tolist(),tilt_axis_azimuth_deg=float(tilt_axis_azimuth_deg),transport_yaw_deg=float(transport_yaw_deg),axial_yaw_waypoints_deg=axial_yaw_waypoints_deg,max_tilt_deg=100.,mouth_height_waypoints_m=knots,upright_retreat_mouth_z_m=1.02,idle_segment_s=1.2,approach_duration_s=3.,tilt_duration_s=4.8,return_duration_s=3.6,retreat_duration_s=1.2,maximum_drain_wait_s=2.,grasp_or_physics_changes=False,flow_gate_changes=False,engineering_scope='Candidate side/overhand orientation with20mm rim gap. Crouch/layout and complete path require validation; no fluid coefficients, gains, friction or acceptance modifications.')

def target(angle_deg,profile,returning=False):
    """Tube orientation and mouth goal; returning reverses the same geometric path."""
    az=np.deg2rad(profile['tilt_axis_azimuth_deg']);axis=np.array([np.cos(az),np.sin(az),0.])
    tube_R=Rotation.from_rotvec(np.deg2rad(angle_deg)*axis).as_matrix() @ Rotation.from_euler('z',axial_yaw_deg(angle_deg,profile),degrees=True).as_matrix()
    return mouth_position(angle_deg,profile),tube_R


def smoothstep5(x):
    x=np.clip(x,0.,1.);return x*x*x*(10.+x*(-15.+6.*x))

def plan_pour_segment_v4(m,d,profile,*,segment,ik_solver,entry_arm_command=None,duration_s=4.8,knot_dt_s=.08):
    """Pure isolated-FK planner. Returning arrays does not execute or certify grip."""
    if profile.get('geometric_path_kind')=='joint_reference_curve':
        from t1_human_pour_v4_joint_reference import plan_segment
        from t1_spincoat_scene import arm_indices
        if entry_arm_command is None:entry_arm_command=np.asarray(d.qpos[arm_indices(m,'left')[1]]).copy()
        return plan_segment(m,d,profile,segment=segment,entry_arm_command=entry_arm_command,duration_s=duration_s)
    if profile.get('time_parameterization')=='joint_arclength_C2':
        from t1_human_pour_v4_arclength import plan_segment
        from t1_spincoat_scene import arm_indices
        if entry_arm_command is None:entry_arm_command=np.asarray(d.qpos[arm_indices(m,'left')[1]]).copy()
        return plan_segment(m,d,profile,segment=segment,entry_arm_command=entry_arm_command,duration_s=duration_s)
    import mujoco
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Slerp
    from t1_spincoat_scene import arm_indices
    _,qa,_=arm_indices(m,'left');w=mujoco.MjData(m);w.qpos[:]=d.qpos;mujoco.mj_kinematics(m,w)
    pb=m.body('sharpa_left/left_hand_C_MC').id;tb=m.body(profile.get('tube_body','prep/reagent_tube')).id;ms=m.site(profile.get('tube_mouth_site','prep/tube_mouth')).id
    H0=w.xmat[pb].reshape(3,3).copy();P0=w.xpos[pb].copy();E=H0.T@w.xmat[tb].reshape(3,3);ml=H0.T@(w.site_xpos[ms]-P0)
    # profile may freeze the measured transform for the same physical pour sequence.
    E=np.asarray(profile.get('measured_hand_to_tube_R',E));ml=np.asarray(profile.get('measured_mouth_in_hand_m',ml))
    maximum=float(profile.get('max_tilt_deg',100.));az=np.deg2rad(profile.get('tilt_axis_azimuth_deg',135.));axis=np.r_[np.cos(az),np.sin(az),0.];yaw=Rotation.from_euler('z',axial_yaw_deg(0.,profile),degrees=True).as_matrix();knots=np.asarray(profile['mouth_height_waypoints_m']);height=CubicSpline(knots[:,0],knots[:,1],bc_type='natural');xy=np.asarray(profile['mouth_xy_m']);times=np.linspace(0,duration_s,max(3,int(np.ceil(duration_s/knot_dt_s))+1));seed=np.asarray(d.qpos[qa] if entry_arm_command is None else entry_arm_command).copy();commands=[seed.copy()];goals=[];infos=[]
    Tstart=w.xmat[tb].reshape(3,3).copy();ang0=float(np.degrees(np.arccos(np.clip(Tstart[2,2],-1,1))))
    for t in times[1:]:
        u=smoothstep5(t/duration_s)
        if segment=='approach':
            T=yaw;H=T@E.T;mouth=mouth_position(0.,profile);P=mouth-H@ml;rot=Slerp([0,1],Rotation.from_matrix(np.array([H0,H])))(u).as_matrix();mouth0=P0+H0@ml;pos=mouth0+(mouth-mouth0)*u-rot@ml
        elif segment in ['tilt','return']:
            angle=maximum*u if segment=='tilt' else maximum*(1-u);yaw=Rotation.from_euler('z',axial_yaw_deg(angle,profile),degrees=True).as_matrix();T=Rotation.from_rotvec(np.deg2rad(angle)*axis).as_matrix()@yaw;rot=T@E.T;mouth=mouth_position(angle,profile);pos=mouth-rot@ml
        elif segment=='retreat':
            rot=yaw@E.T;mouth0=w.site_xpos[ms].copy();mouth=mouth0+(np.asarray(profile.get('upright_retreat_mouth_xyz_m',mouth_position(0.,profile)))-mouth0)*u;pos=mouth-rot@ml
        else:raise ValueError(segment)
        q,info=ik_solver(m,w,pos,rot,side='left',seed=seed);commands.append(q);goals.append((pos.copy(),rot.copy()));infos.append(info);seed=q;w.qpos[qa]=q;mujoco.mj_kinematics(m,w)
    commands=np.asarray(commands);spline=CubicSpline(times,commands,axis=0,bc_type=((1,np.zeros(7)),(1,np.zeros(7))));dense_t=np.arange(1,int(round(duration_s/m.opt.timestep))+1)*m.opt.timestep;denseq=spline(dense_t);velocity=spline(dense_t,1);ids,_,_=arm_indices(m,'left');margin=np.minimum(denseq-m.jnt_range[ids,0],m.jnt_range[ids,1]-denseq).min();peak=float(abs(velocity).max())
    if margin<.05-1e-6 or peak>2.:raise RuntimeError('Geometric spline rejected: margin='+str(margin)+' speed='+str(peak))
    return dict(time=dense_t,arm_commands=denseq,segment=segment,duration_s=duration_s,peak_joint_speed_rad_s=peak,min_joint_margin_rad=float(margin),measured_hand_to_tube_R=E.tolist(),measured_mouth_in_hand_m=ml.tolist(),knot_times=times,knot_commands=commands)


def advance_human_pour_v4(m,d,handles,arm_command,step_callback,*,profile,ledger,ik_solver):
    """Original controls live in callback. No mj_step,qpos,bodypose,clock or fluid writes.
    Callback must preserve its normal physics/ledger/base/support monitoring and apply the
    separately supplied idle-finger overlay on these named phases before its encoder tracker.
    External ledger.stop_requested() is required; no fixed5s or magic volume cap.
    """
    import mujoco
    if not callable(getattr(ledger,'stop_requested',None)):raise ValueError('External flow termination ledger.stop_requested() required')
    p=dict(profile);p['tube_body']=handles['tube_body'];p['tube_mouth_site']=handles['tube_mouth_site'];trace=[];reports=[];cmd=np.asarray(arm_command).copy()
    # Idle joints are controlled only by the independently supplied named-phase overlay.
    if not profile.get('idle_fold_already_applied',False):
        for phase in ['human_pour_idle_extend','human_pour_idle_fold']:
            for _ in range(int(round(profile.get('idle_segment_s',1.2)/m.opt.timestep))):step_callback(cmd,phase)
    for seg in ['approach','tilt']:
        if seg=='approach' and p.get('geometric_path_kind')=='joint_reference_curve' and p.get('skip_satisfied_joint_approach',False):
            first=np.asarray(p['joint_reference_keyframes_rad'][0][1:])
            if np.max(abs(cmd-first))<1e-6:
                reports.append(dict(segment='approach',duration_s=0.,native_steps=0,skipped='Coarse command already equals first natural q node; original upstream physical entry gates unchanged'))
                continue
        plan=plan_pour_segment_v4(m,d,p,segment=seg,ik_solver=ik_solver,entry_arm_command=cmd,duration_s=profile.get(seg+'_duration_s',4.8),knot_dt_s=float(profile.get('ik_knot_dt_s',.02)));p['measured_hand_to_tube_R']=plan['measured_hand_to_tube_R'];p['measured_mouth_in_hand_m']=plan['measured_mouth_in_hand_m'];reports.append({k:v for k,v in plan.items() if k not in ['time','arm_commands','knot_commands','knot_times','angle_deg']})
        for j,q in enumerate(plan['arm_commands']):
            cmd=q;step_callback(cmd,'human_pour_'+seg);trace.append((float(d.time),seg,cmd.copy()))
            if seg=='tilt' and ledger.stop_requested():
                p['max_tilt_deg']=float(plan['angle_deg'][j]) if plan.get('angle_deg') is not None else float(profile.get('max_tilt_deg',100.))*float(smoothstep5((j+1)*m.opt.timestep/plan['duration_s']));break
        # Until drain completion, keep actual final geometric target. Timeout is failure, never a success cap.
    waited=0.
    while not ledger.stop_requested():
        if waited>=float(profile.get('maximum_drain_wait_s',2.)):raise RuntimeError('External liquid completion not reached before diagnostic safety timeout')
        step_callback(cmd,'human_pour_drain_wait');waited+=m.opt.timestep;trace.append((float(d.time),'drain_wait',cmd.copy()))
    for seg in ['return','retreat']:
        plan=plan_pour_segment_v4(m,d,p,segment=seg,ik_solver=ik_solver,entry_arm_command=cmd,duration_s=profile.get(seg+'_duration_s',4.8),knot_dt_s=float(profile.get('ik_knot_dt_s',.02)));reports.append({k:v for k,v in plan.items() if k not in ['time','arm_commands','knot_commands','knot_times','angle_deg']})
        for q in plan['arm_commands']:
            cmd=q;step_callback(cmd,'human_pour_'+seg);trace.append((float(d.time),seg,cmd.copy()))
    for phase in ['human_pour_idle_restore_fold','human_pour_idle_restore_extend']:
        for _ in range(int(round(profile.get('idle_segment_s',1.2)/m.opt.timestep))):step_callback(cmd,phase)
    return cmd,dict(scope='Uncertified continuous motion adapter; callback retains original physics/gates',segments=reports,drain_wait_s=waited,flow_stop_source='external ledger.stop_requested',success=True),dict(time=np.array([v[0] for v in trace]),phase=np.array([v[1] for v in trace]),arm_command=np.array([v[2] for v in trace]))


class IdleFingerOverlay:
    """Opt-in9-joint nominal ctrl overlay, no contact feedback or primary-finger edits.
    Call after ordinary hand-control writes and before the existing encoder tracker.
    Extend MCP first; fold only distal joints; restore in the opposite order after retreat.
    """
    def __init__(self,m,idle_segment_s=1.2):
        from t1_spincoat_scene import hand_indices
        js,_,_=hand_indices(m,'left');acts=[int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in js];self.names=[];self.ids=[];self.extend=[];self.fold=[];self.start=None;self.phase=None;self.t0=0.;self.duration=float(idle_segment_s);self.anchor=None;self.last=None;self.restored=False
        for j,ac in zip(js,acts):
            name=m.joint(int(j)).name
            if not any('/left_'+di+'_' in name for di in ['ring','pinky']):continue
            self.names.append(name);self.ids.append(int(ac));val=-.15 if name.endswith('MCP_FE') else 1.4 if name.endswith('_PIP') else 1.0 if name.endswith('_DIP') else .05 if name.endswith('_CMC') else 0.;self.fold.append(val)
        self.ids=np.asarray(self.ids,dtype=int);self.fold=np.asarray(self.fold);self.rows=[]
    def __call__(self,m,d,phase):
        if self.restored and not phase.startswith('human_pour_idle_'):return
        if self.start is None:
            if phase!='human_pour_idle_extend':return
            self.start=d.ctrl[self.ids].copy();self.extend=self.start.copy()
            for i,name in enumerate(self.names):
                if name.endswith('MCP_FE'):self.extend[i]=-.15
                if name.endswith('MCP_AA'):self.extend[i]=0.
        if phase!=self.phase:self.phase=phase;self.t0=float(d.time);self.anchor=self.last.copy() if self.last is not None else d.ctrl[self.ids].copy()
        goal=self.extend if phase in ['human_pour_idle_extend','human_pour_idle_restore_fold'] else self.start if phase=='human_pour_idle_restore_extend' else self.fold
        u=smoothstep5((float(d.time)-self.t0+m.opt.timestep)/self.duration) if phase.startswith('human_pour_idle_') else 1.
        out=self.anchor+(goal-self.anchor)*u;d.ctrl[self.ids]=out;self.last=out.copy();self.restored=bool(phase=='human_pour_idle_restore_extend' and u>=1.-1e-9);self.rows.append((float(d.time),phase,out.copy()))

V4_REFERENCES=[dict(title='UniversityofWisconsin official Pouring Liquids',url='https://www2.chem.wisc.edu/deptfiles/genchem/lab/labdocs/modules/pouring/pouringliq.htm',video='https://www2.chem.wisc.edu/deptfiles/genchem/lab/labdocs/modules/pouring/mp4VIDEO/01045702.mp4',basis='Primary teaching video shows top/side vessel grasp and close guided transfer, with receiving vessel supported by otherhand. It is beaker-to-flask with a glassrod, not a measured25x150tube robot trajectory.'),REFERENCES[0]]
