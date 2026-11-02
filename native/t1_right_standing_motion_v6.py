"""Independent standing RIGHT v6 planning/runtime helpers; opt-in only.
No runtime root/object qpos writes. Original hand controls and gates retained.
"""
from pathlib import Path
import inspect,json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices,hand_indices,ROOT
from t1_right_interior_workstation_v6 import RightInteriorWorkstationV6

def _exact(s,old,new):
    if s.count(old)!=1:raise RuntimeError('v6 source anchor count: '+repr(old)+'='+str(s.count(old)))
    return s.replace(old,new)

def prepare_standing_right_v6(m,d,out,plan,*,time_scale=1.25,config=None):
    """Original station/hand factory, new fixed-dock C2 prefix on planning data.

    Called at RESET before LEFT. The station model reset remains RESET-only;
    robot planning state is isolated and is never assigned to live robot qpos.
    """
    import t1_continuous_approach as ca
    c=config or RightInteriorWorkstationV6()
    if float(d.time)!=0.:raise ValueError('v6 station factory is RESET-only')
    _,qa,_=arm_indices(m);hj,hq,_=hand_indices(m);bqa=int(m.joint('chassis_free').qposadr[0]);ready=d.qpos[qa].copy();before=d.qpos.copy();work=mujoco.MjData(m);work.qpos[:]=d.qpos;work.qvel[:]=d.qvel;work.ctrl[:]=d.ctrl
    work.qpos[bqa:bqa+2]=c.grasp_base_xy_m
    for i,x in enumerate(c.standing_torso_q_rad):work.qpos[int(m.joint('astribot_torso_joint_'+str(i+1)).qposadr[0])]=x
    mujoco.mj_forward(m,work);palm=m.body('sharpa_right/right_hand_C_MC').id;Pready=work.xpos[palm].copy();Rready=work.xquat[palm].copy()
    src=(ROOT/'research/hardware/direct_grasp/direct_oblique_07/executed_clearance_adapter.py').read_text(encoding='utf-8-sig')
    src=_exact(src,'a=types.SimpleNamespace(base_x=-.08,base_y=.4,base_approach=True)',"a=types.SimpleNamespace(base_x=V6_CONFIG.grasp_base_xy_m[0],base_y=V6_CONFIG.grasp_base_xy_m[1],base_approach=False)")
    src=_exact(src,'d.qpos[bqa:bqa+3]=[-.08,.4,.09117]','d.qpos[bqa:bqa+2]=V6_CONFIG.grasp_base_xy_m')
    line=next(x for x in src.splitlines() if x.startswith(' targets=['))
    src=_exact(src,line," targets=[('ready_hold',0.,V6_READY_P,V6_READY_Q),('grasp_settle',5.8,[.350,.01,1.002],[2**-.5,2**-.5,0,0])]")
    src=_exact(src,'dock=CoarseStopController(m,[0.,.4])','dock=CoarseStopController(m,V6_CONFIG.grasp_base_xy_m)')
    src=_exact(src,'future.qpos[bqa]=0.','future.qpos[bqa:bqa+2]=V6_CONFIG.grasp_base_xy_m')
    # These values must survive both source compilation environments.
    imports='from t1_right_standing_motion_v6 import _ACTIVE_CONFIG as V6_CONFIG, _READY_P as V6_READY_P, _READY_Q as V6_READY_Q\n'
    src=imports+src
    global _ACTIVE_CONFIG,_READY_P,_READY_Q
    _ACTIVE_CONFIG=c;_READY_P=Pready;_READY_Q=Rready
    baseline=ca.baseline_station_prepare;station=inspect.getsource(baseline)
    anchor="src=(ROOT/'research/hardware/direct_grasp/direct_oblique_07/executed_clearance_adapter.py').read_text(encoding='utf-8-sig')"
    station=_exact(station,anchor,'src=V6_CLEARANCE_SOURCE')
    factory_source=inspect.getsource(ca.adapted_runner)
    factory_anchor="    module=types.ModuleType('front_workflow_'+kind)"
    replacement="    if kind=='grasp':\n        needle='goal_tip[2]+=.03*'\n        if source.count(needle)!=1:raise RuntimeError('Original sole verticalize extra-lift reference required')\n        source=source.replace(needle,'goal_tip[2]+=' + repr(float(V6_EXTRA_VERTICALIZE_LIFT)) + '*')\n"+factory_anchor
    factory_source=_exact(factory_source,factory_anchor,replacement);factory_path=out/'v6_right_runner_factory.py';factory_path.write_text(factory_source);factory_env=dict(ca.adapted_runner.__globals__,V6_EXTRA_VERTICALIZE_LIFT=c.grasp_verticalize_extra_lift_m);exec(compile(factory_source,str(factory_path),'exec'),factory_env)
    station=_exact(station,'glob=dict(t1_support_grasp.prepare_support_grasp.__globals__);glob.update(',"glob=dict(t1_support_grasp.prepare_support_grasp.__globals__);glob['adapted_runner']=V6_ARM_FACTORY;glob.update(")
    if c.station_already_prepared:
        reset_anchor=' center,turn=reset_station(m,d,station_prepare.angle,out)'
        station=_exact(station,reset_anchor," center=V6_CANONICAL_PIPE_CENTER.copy();turn=Rotation.from_euler('z',station_prepare.angle,degrees=True).as_matrix()")
        standR=work.xmat[m.body('stand/pipette_stand').id].reshape(3,3)
        expected=Rotation.from_euler('z',c.stand_yaw_deg,degrees=True).as_matrix()
        if not np.allclose(standR,expected,atol=1e-9):raise ValueError('Canonical final stand yaw mismatch; refuse double-rotation')
        (out/'station_reset.json').write_text(json.dumps(dict(scope='Already compiled canonical RESET station; no rotation or live tool pose reset',station_yaw_deg=c.stand_yaw_deg,pivot_world_m=work.xpos[m.body('pipette/pipette').id].tolist()),indent=2))
    env=dict(baseline.__globals__,V6_CLEARANCE_SOURCE=src,V6_ARM_FACTORY=factory_env['adapted_runner'],V6_CANONICAL_PIPE_CENTER=work.xpos[m.body('pipette/pipette').id].copy());path=out/'v6_station_preparer.py';path.write_text(station);exec(compile(station,str(path),'exec'),env);fn=env[baseline.__name__];fn.angle=60.;fn.closing_scale=1.
    result=fn(m,work,out,plan,time_scale=time_scale);runner,path,callback,audit,clock=result
    profile=json.loads(path.read_text());end=np.asarray(profile['waypoints'][-1]['arm_q_rad']);times=np.array([.7,5.8]);q=np.array([ready,end]);profile['continuous_approach']=dict(time_s=times.tolist(),arm_q_rad=q.tolist(),method='v6 fixed-safe-dock direct C2 joint prefix; actual ready joints to new stationary pre-seat IK; pending actual geometry audit')
    profile['waypoints']=[dict(name='ready_hold',time_s=0.,arm_q_rad=ready.tolist()),dict(name='ready_hold',time_s=.7,arm_q_rad=ready.tolist()),dict(name='continuous_station_entry',time_s=5.8,arm_q_rad=end.tolist())];profile['hand_profile']['home_q_rad']=d.qpos[hq].tolist();profile['v6_working_posture']=dict(base_xy_m=c.grasp_base_xy_m,torso_q_rad=c.standing_torso_q_rad)
    path.write_text(json.dumps(profile,indent=2));runner.continuous_prefix_sample=ca.continuous_prefix_sample
    # Original station reset changed only the free pipette's RESET quaternion.
    pipe=m.body('pipette/pipette').id;j=int(m.body_jntadr[pipe]);pq=int(m.jnt_qposadr[j]);d.qpos[pq:pq+7]=work.qpos[pq:pq+7];mask=np.ones(m.nq,bool);mask[pq:pq+7]=False
    if not np.array_equal(before[mask],d.qpos[mask]):raise AssertionError('v6 planner changed live robot RESET state')
    mujoco.mj_forward(m,d)
    qp=mujoco.MjData(m);qp.qpos[:]=work.qpos;samples=[]
    for t in np.linspace(0,5.8,1161):qp.qpos[qa]=ca.continuous_prefix_sample(profile,t);samples.append(qp.qpos.copy())
    np.savez_compressed(out/'v6_conditional_prefix.npz',time=np.linspace(0,5.8,1161),qpos=np.array(samples))
    return result

_ACTIVE_CONFIG=RightInteriorWorkstationV6();_READY_P=None;_READY_Q=None

def advance_standing_dock_v6(m,d,target_xy,*,out,hold_gain_scale=32.,maximum_s=20.):
    """Original physical coarse controller then original .5s stopped gate.

    Holds existing joint actuator targets; only wheel commands move the base.
    This is a coarse transition, never a fine-position feedback controller.
    """
    from t1_coarse_stop import CoarseStopController,save_coarse_control
    out=Path(out);out.mkdir(parents=True,exist_ok=False);dock=CoarseStopController(m,target_xy,hold_gain_scale=hold_gain_scale);control=d.ctrl.copy();wheel=dock.wheel_ids;nonwheel=np.array([a for a in range(m.nu) if a not in wheel]);robot=np.unique([int(m.jnt_dofadr[int(m.actuator_trnid[a,0])]) for a in nonwheel if m.actuator_trnid[a,0]>=0 and m.joint(int(m.actuator_trnid[a,0])).name.startswith(('astribot_','sharpa_'))]);rows=[];events=[];start=float(d.time)
    for k in range(int(maximum_s/m.opt.timestep)):
        d.ctrl[nonwheel]=control[nonwheel];dock.step(d);d.qfrc_applied[:]=0.;d.qfrc_applied[robot]=d.qfrc_bias[robot];mujoco.mj_step(m,d)
        for ci,c in enumerate(d.contact):
            ns=[m.body(int(m.geom_bodyid[g])).name for g in c.geom]
            if any(n.startswith(('astribot_','sharpa_')) for n in ns) and any(n.startswith(('lab_bench','stand/','spincoater/','aspiration_source')) for n in ns):
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
                if np.linalg.norm(f[:3])>.001:events.append(dict(time_s=float(d.time),bodies=ns,force_N=float(np.linalg.norm(f[:3])),depth_mm=max(0.,float(-c.dist*1000))))
        if k%20==0 or dock.stopped:rows.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()))
        if dock.stopped:break
    raw={name:np.array([r[i] for r in rows]) for i,name in enumerate(['time','qpos','qvel','ctrl'])};report=dict(scope='Actual original coarse-only base transition; not fine arm alignment',target_xy_m=list(target_xy),start_time_s=start,end_time_s=float(d.time),stopped=bool(dock.stopped),environment_force_events=events,original_thresholds_unchanged=True,no_runtime_pose_writes=True,success=bool(dock.stopped and not events),control=save_coarse_control(dock,out/'coarse_control.npz'));np.savez_compressed(out/'actual_states.npz',**raw);(out/'report.json').write_text(json.dumps(report,indent=2));return raw,report,dock

def advance_direct_cup_v6(m,d,dock,*,out,config=None,duration_s=4.,settle_s=.5):
    """Actual arm-only straight tip approach with one smooth axial-yaw change.

    Begins only after original coarse stop; preserves captured finger controls.
    New reference uses actual start hand/tool geometry (privileged GT), not force.
    """
    from t1_workflow_ik import install_validated_ik
    import types
    c=config or RightInteriorWorkstationV6();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if not dock.stopped or dock.mode!='zero_speed_hold':raise RuntimeError('v6 cup approach requires original coarse stopped gate')
    ids,qa,_=arm_indices(m);armact=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in ids]);palm=m.body('sharpa_right/right_hand_C_MC').id;tip=m.site('pipette/tip_site').id;mujoco.mj_forward(m,d);actualR=d.xmat[palm].reshape(3,3).copy();actualT=d.site_xpos[tip].copy();rel=actualR.T@(actualT-d.xpos[palm]);ctrl=d.ctrl.copy();reference=mujoco.MjData(m);reference.qpos[:]=d.qpos;reference.qpos[qa]=ctrl[armact];mujoco.mj_kinematics(m,reference);R0=reference.xmat[palm].reshape(3,3).copy();T0=reference.xpos[palm]+R0@rel;goal=np.r_[c.cup_xy_m,.93];wheel=dock.wheel_ids;robot=np.unique([int(m.jnt_dofadr[int(m.actuator_trnid[a,0])]) for a in range(m.nu) if a not in wheel and m.actuator_trnid[a,0]>=0 and m.joint(int(m.actuator_trnid[a,0])).name.startswith(('astribot_','sharpa_'))]);solver=types.SimpleNamespace(arm_indices=arm_indices,workflow_nullspace_enabled=True);start=float(d.time);install_validated_ik(solver,start);qcmd=d.ctrl[armact].copy();rows=[];events=[];ik=[]
    for k in range(int(round((duration_s+settle_s)/m.opt.timestep))):
        t=float(d.time-start);u=float(np.clip(t/duration_s,0,1));u=u**3*(10+u*(-15+6*u));R=Rotation.from_euler('z',c.aspiration_tool_world_yaw_delta_deg*u,degrees=True).as_matrix()@R0;T=(1-u)*T0+u*goal
        if k%20==0:
            P=T-R@rel;quat=Rotation.from_matrix(R).as_quat()[[3,0,1,2]];qcmd,e=solver.ik(m,d,P,quat,seed=qcmd,iterations=150);ik.append(float(e))
            if e>.0001:raise RuntimeError('v6 direct cup IK failed original precision: '+str(e))
        d.ctrl[:]=ctrl;d.ctrl[armact]=qcmd;dock.step(d);d.qfrc_applied[:]=0.;d.qfrc_applied[robot]=d.qfrc_bias[robot];mujoco.mj_step(m,d)
        for ci,ct in enumerate(d.contact):
            ns=[m.body(int(m.geom_bodyid[g])).name for g in ct.geom]
            if any(n.startswith(('astribot_','sharpa_')) for n in ns) and any(n.startswith(('lab_bench','stand/','spincoater/','aspiration_source')) for n in ns):
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
                if np.linalg.norm(f[:3])>.001:events.append(dict(time_s=float(d.time),bodies=ns,force_N=float(np.linalg.norm(f[:3]))))
        if k%20==0 or k==int(round((duration_s+settle_s)/m.opt.timestep))-1:rows.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.site_xpos[tip].copy()))
    raw={name:np.array([r[i] for r in rows]) for i,name in enumerate(['time','qpos','qvel','ctrl','tip'])};err=float(np.linalg.norm(d.site_xpos[tip]-goal));report=dict(scope='Actual arm-only direct cup preposition after original coarse stop; single world-Z tool reorientation, GT geometry; no tactile inputs',start_time_s=start,end_time_s=float(d.time),target_tip_world_m=goal.tolist(),actual_tip_error_mm=err*1000,maximum_command_ik_error_mm=max(ik)*1000,environment_force_events=events,success=bool(err<.002 and not events and dock.stopped),no_runtime_pose_writes=True)
    np.savez_compressed(out/'actual_states.npz',**raw);(out/'report.json').write_text(json.dumps(report,indent=2));return raw,report

def make_standing_preparer_v6(config=None):
    c=config or RightInteriorWorkstationV6()
    def prepare(m,d,out,plan,*,time_scale=1.25):return prepare_standing_right_v6(m,d,out,plan,time_scale=time_scale,config=c)
    return prepare

def adapt_engine_source_v6(source,config=None):
    """Exact private engine anchors. New coarse phases precede each fine stage."""
    c=config or RightInteriorWorkstationV6();cfg=repr(dict(c.__dict__))
    a='    prepared_source=prep_ledger.source_geometry()\n'
    b=a+"    from t1_right_standing_motion_v6 import advance_standing_dock_v6,advance_direct_cup_v6,register_right_docks_v6,save_joined_right_coarse_control_v6\n    from t1_right_interior_workstation_v6 import RightInteriorWorkstationV6\n    right_v6_config=RightInteriorWorkstationV6(**"+cfg+")\n    right_dock_raw,right_dock_report,right_grasp_transfer_dock=advance_standing_dock_v6(m,d,right_v6_config.grasp_base_xy_m,out=out/'v6_right_grasp_dock')\n    np.savez_compressed(out/'v6_right_grasp_dock.npz',**right_dock_raw)\n    v6_early_reports={'v6_right_grasp_dock':right_dock_report};register_right_docks_v6(out,right_grasp_transfer_dock,grasp_audit.dock);handoff('v6_right_grasp_dock_end')\n    if not right_dock_report['success']:return\n"
    source=_exact(source,a,b)
    source=_exact(source,'    preparation_reanchor(m,d,mobile_profile)','    from t1_right_standing_motion_v6 import reanchor_prefix_at_actual_dock_v6\n    reanchor_prefix_at_actual_dock_v6(m,d,grasp,mobile_profile)')
    stage_line=next(x for x in source.splitlines() if x.startswith("    stages={'grasp':"))
    source=_exact(source,stage_line,stage_line+'\n    stages.update(v6_early_reports)')
    old_save="save_coarse_control(grasp_audit.dock,out/'left_coarse_control.npz')"
    if source.count(old_save)!=2:raise RuntimeError('Original two grasp/asp coarse-control save anchors required')
    source=source.replace(old_save,"save_joined_right_coarse_control_v6(grasp_audit.dock,out/'left_coarse_control.npz')")
    a='    original_asp=run_aspiration_on_state\n'
    b="    right_dock_raw,right_dock_report,right_asp_dock=advance_standing_dock_v6(m,d,right_v6_config.aspiration_base_xy_m,out=out/'v6_right_aspiration_dock')\n    np.savez_compressed(out/'v6_right_aspiration_dock.npz',**right_dock_raw)\n    stages['v6_right_aspiration_dock']=right_dock_report;handoff('v6_right_aspiration_dock_end')\n    if not right_dock_report['success']:return\n    right_cup_raw,right_cup_report=advance_direct_cup_v6(m,d,right_asp_dock,out=out/'v6_right_direct_cup',config=right_v6_config)\n    np.savez_compressed(out/'v6_right_direct_cup.npz',**right_cup_raw)\n    stages['v6_right_direct_cup']=right_cup_report;handoff('v6_right_direct_cup_end')\n    if not right_cup_report['success']:return\n    register_right_docks_v6(out,right_asp_dock)\n    grasp_audit.dock=right_asp_dock\n"+a
    source=_exact(source,a,b)
    source=_exact(source,'mobile,mr,dock=mobile_advance(m,d,[0,0],','mobile,mr,dock=mobile_advance(m,d,right_v6_config.coating_base_xy_m,')
    if c.skip_camera_clearance_motion:
        source=_exact(source,"    clearance,clearance_review=neutral_view_clearance(m,d,dock,ledger,out=out/'view_clearance')","    from t1_right_standing_motion_v6 import record_skipped_camera_clearance_v6\n    clearance,clearance_review=record_skipped_camera_clearance_v6(m,d,dock,ledger,out=out/'view_clearance')")
    return source

def adapt_neutral_source_v6(source,config=None):
    c=config or RightInteriorWorkstationV6();cfg=repr(dict(c.__dict__))
    source=_exact(source,'        preparation=prepare_smooth_station_grasp','        from t1_right_standing_motion_v6 import make_standing_preparer_v6\n        from t1_right_interior_workstation_v6 import RightInteriorWorkstationV6\n        preparation=make_standing_preparer_v6(RightInteriorWorkstationV6(**'+cfg+'))')
    a="    asp_text=(support/'executed_aspiration_adapter.py').read_text(encoding='utf-8-sig')"
    b="    from t1_right_standing_motion_v6 import adapt_engine_source_v6\n    from t1_right_interior_workstation_v6 import RightInteriorWorkstationV6\n    engine_text=adapt_engine_source_v6(engine_text,RightInteriorWorkstationV6(**"+cfg+"))\n"+a
    return _exact(source,a,b)

def adapt_launch_source(source,config=None):
    """Apply to v4.transformed_source AFTER existing v5 seat/fixed adapters.

    Caller owns coherent shared RESET scene; this owns only RIGHT controls.
    Planning/actual validation is still required for the selected config.
    """
    c=config or RightInteriorWorkstationV6();cfg=repr(dict(c.__dict__))
    a=" layout_config=RightWorkstationV4(station_shift_world_m=(a.v4_station_shift_x_m,a.v4_station_shift_y_m,0.))\n source=Path(baseline.__file__).read_text()\n"
    b=a+" from t1_right_standing_motion_v6 import adapt_neutral_source_v6\n from t1_right_interior_workstation_v6 import RightInteriorWorkstationV6\n source=adapt_neutral_source_v6(source,RightInteriorWorkstationV6(**"+cfg+"))\n"
    source=_exact(source,a,b)
    a='from t1_right_workstation_v4 import translated_station_preparer,relocate_workstation_at_reset as relocate_station_at_reset,RightWorkstationV4,adapt_supported_aspiration_source'
    # This line is itself generated-source text, so append a normal import in
    # the outer launcher immediately before its source compiler environment.
    anchor=' from t1_right_standing_motion_v6 import adapt_neutral_source_v6\n'
    source=_exact(source,anchor,' from t1_right_interior_workstation_v6 import adapt_vertical_aspiration_source as adapt_supported_aspiration_source\n from t1_right_direct_coating_v6 import adapt_direct_coating_source as adapt_coating_source\n'+anchor)
    return source

_RIGHT_DOCKS_V6={}
def register_right_docks_v6(out,*docks):
    """Retain every real controller; source arrays are never replaced by a new dock."""
    key=str(Path(out).resolve());registry=_RIGHT_DOCKS_V6.setdefault(key,[])
    for dock in docks:
        if not any(old is dock for old in registry):registry.append(dock)
    return registry

def save_joined_right_coarse_control_v6(dock,path):
    """Save each original control/gain ledger then stable-sort exact rows by time.

    Includes all samples and zeros; no deduplication, filtering or resampling.
    Explicit source indices make every merged row traceable to its own dock.
    """
    from t1_coarse_stop import save_coarse_control
    path=Path(path);registry=register_right_docks_v6(path.parent,dock);sources=[];audits=[]
    for i,old in enumerate(registry):
        source=path.parent/('v6_right_dock_control_'+str(i).zfill(2)+'.npz')
        if len(old.control_records):review=save_coarse_control(old,source)
        else:
            n=len(old.wheel_ids);names=['time','base_x','base_y','base_vx','base_vy','yaw_rate']+[f'wheel_command_{j}' for j in range(n)]+['world_pi_active','coarse_latched','stopped_gate','workspace_error_m']
            names +=[f'{prefix}_{j}' for prefix in ['wheel_velocity_gain','wheel_velocity_bias','actual_wheel_speed','actual_wheel_force'] for j in range(n)]
            np.savez_compressed(source,**{name:np.empty(0,float) for name in names});review=dict(samples=0,scope='Controller has no actual step yet; retained explicitly, not synthetic zero records')
        with np.load(source) as z:arrays={name:z[name].copy() for name in z.files}
        sources.append(arrays);audits.append(dict(index=i,file=source.name,samples=len(arrays['time']),original_controller_audit=review))
    fields=list(sources[0]);
    if any(set(x)!=set(fields) for x in sources):raise RuntimeError('Distinct coarse controller ledger schemas cannot be silently merged')
    arrays={name:np.concatenate([x[name] for x in sources]) for name in fields};origin=np.concatenate([np.full(len(x['time']),i,dtype=int) for i,x in enumerate(sources)]);row=np.concatenate([np.arange(len(x['time'])) for x in sources]);order=np.argsort(arrays['time'],kind='stable');arrays={name:x[order] for name,x in arrays.items()};origin=origin[order];row=row[order]
    for i,original in enumerate(sources):
        select=origin==i
        if any(not np.array_equal(arrays[name][select],original[name]) for name in fields):raise AssertionError('Coarse history merge altered/dropped/reordered source values')
    arrays['source_dock_index']=origin;arrays['source_row_index']=row;np.savez_compressed(path,**arrays)
    review=dict(scope='Actual dock ledgers, exact stable time merge without interpolation, filtering or deduplication',samples=len(order),sources=audits,all_source_rows_preserved_exactly=True,duplicate_timestamps_retained=int(np.count_nonzero(np.diff(arrays['time'])==0)),max_wheel_command_after_hold=max((x['original_controller_audit'].get('max_wheel_command_after_hold') or 0.) for x in audits),zero_speed_velocity_servo_gain_scale=float(dock.hold_gain_scale),original_gate_definition='speed<1mm/s, yaw<.002rad/s, workspace<=10mm; unchanged')
    (path.parent/'v6_right_coarse_history_merge.json').write_text(json.dumps(review,indent=2));return review

def reanchor_prefix_at_actual_dock_v6(m,d,module,path):
    """Replan ready→preseat using the measured stopped base, without moving it.

    Keeps original ready-return guard, .04rad margin IK and existing hand plan.
    Synchronizes the original prefix FK compensation reference with its new
    actual-dock plan, preventing a nominal-base command jump at formal close.
    """
    from t1_reagent_workflow_adapter import reanchor_ready_profile
    from t1_workflow_ik import install_validated_ik
    import types
    before=d.qpos.copy();commands=d.ctrl.copy();original=reanchor_ready_profile(m,d,path);profile=json.loads(Path(path).read_text());ids,qa,_=arm_indices(m);bq=int(m.joint('chassis_free').qposadr[0]);base_function=module.ik
    for _ in range(8):
        closure=inspect.getclosurevars(base_function).nonlocals
        if 'a' in closure and 'frames' in closure:break
        if not callable(closure.get('old')):raise RuntimeError('Original station close-entry closure required beneath opt-in wrappers')
        base_function=closure['old']
    else:raise RuntimeError('Station IK wrapper chain exceeds reviewed depth')
    env=base_function.__globals__
    P=np.asarray(env['station_point']([.350,.01,1.002],5.8));quat=np.asarray(env['station_quat']([2**-.5,2**-.5,0.,0.]));R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix();palm=m.body('sharpa_right/right_hand_C_MC').id;solver=types.SimpleNamespace(arm_indices=arm_indices,workflow_nullspace_enabled=True);install_validated_ik(solver,float(d.time));old=np.asarray(profile['continuous_approach']['arm_q_rad'][-1]);work=mujoco.MjData(m);work.qpos[:]=d.qpos;best=None
    for seed in [old,d.qpos[qa],(m.jnt_range[ids,0]+m.jnt_range[ids,1])/2]:
        q,e=solver.ik(m,work,P,quat,seed=seed,iterations=500);work.qpos[qa]=q;mujoco.mj_kinematics(m,work);er=float(Rotation.from_matrix(work.xmat[palm].reshape(3,3)@R.T).magnitude());candidate=(float(e),er,q.copy())
        if best is None or e+er*.01<best[0]+best[1]*.01:best=candidate
        if e<.0001 and er<np.radians(.1):break
    error,angle,end=best
    if error>=.0001 or angle>=np.radians(.1):raise RuntimeError('Actual-dock prefix IK failed original precision: '+repr((error,angle)))
    first=d.qpos[qa].copy();profile['continuous_approach']['arm_q_rad']=[first.tolist(),end.tolist()];profile['continuous_approach']['time_s']=[.7,5.8]
    for waypoint in profile['waypoints']:waypoint['arm_q_rad']=(first if waypoint['time_s']<=.7 else end).tolist()
    closure['a'].base_x=float(d.qpos[bq]);closure['a'].base_y=float(d.qpos[bq+1]);closure['frames'][0]['arm_q_rad']=first.tolist();closure['frames'][-1]['arm_q_rad']=end.tolist()
    report=dict(scope='Planning only at actual latched dock; no base refinement or pose/control write',actual_base_xy_m=d.qpos[bq:bq+2].tolist(),position_error_mm=error*1000,orientation_error_deg=float(np.degrees(angle)),ready_return_guard=original,old_endpoint_q_rad=old.tolist(),new_endpoint_q_rad=end.tolist());profile['v6_actual_dock_replan']=report;Path(path).write_text(json.dumps(profile,indent=2));(Path(path).parent/'v6_actual_dock_replan.json').write_text(json.dumps(report,indent=2))
    if not np.array_equal(before,d.qpos) or not np.array_equal(commands,d.ctrl):raise AssertionError('Actual-dock planning changed live state/commands')
    return report


def record_skipped_camera_clearance_v6(m,d,dock,ledger,out):
    """One actual state, zero integration/control writes. Fresh RGB gate follows.

    success means only that skipping the unnecessary arm retreat was recorded
    at an original stopped dock. It never certifies image validity/alignment.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if getattr(dock,'mode',None)!='zero_speed_hold' or not dock.stopped:
        raise RuntimeError('No-motion view record requires original stopped gate')
    tip=m.site('pipette/tip_site').id
    raw=dict(time=np.array([float(d.time)]),qpos=d.qpos[None,:].copy(),qvel=d.qvel[None,:].copy(),ctrl=d.ctrl[None,:].copy(),phase=np.array(['skipped_unnecessary_motion']),tip=d.site_xpos[tip][None,:].copy())
    report=dict(scope='Actual single-state record; unnecessary legacy +Y100mm retreat skipped; no integration or command writes',status='skipped_unnecessary_motion',start_time_s=float(d.time),end_time_s=float(d.time),actual_samples=1,motion_steps=0,success=True,success_scope='No-motion record only; fresh RGB measurement, pixel and fine alignment gates still required by caller',rgb_validated=False,original_rgb_gate_retained=True,final_stopped_gate=bool(dock.stopped),no_runtime_pose_or_control_writes=True)
    np.savez_compressed(out/'actual_no_motion_state.npz',**raw)
    (out/'view_clearance_report.json').write_text(json.dumps(report,indent=2))
    return raw,report
