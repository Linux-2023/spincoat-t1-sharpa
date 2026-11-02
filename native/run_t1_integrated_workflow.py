"""Independent continuous pick/aspirate/drive/RGB-relocalize/coat experiment.

The delivered 010 runner is loaded into an isolated module with explicit stage
adapters. Main runner, frozen fluid and its historical results are unchanged.
"""
import os,sys,json,argparse,types
os.environ['MUJOCO_GL']='egl';os.environ['T1_LAB_CONTEXT']='1';os.environ['T1_CHEMIST_COAT']='0'
from pathlib import Path
import numpy as np,mujoco
from t1_spincoat_scene import ROOT
from t1_mobile_stage import MobileDockController,advance_mobile_stage
from t1_visual_relocalization import capture_relocalization
from t1_aspiration_stage import SourceGeometry
from run_t1_aspiration_probe import build_aspiration_scene,run_aspiration_on_state

def adapted_runner(kind,fast_transfer=False,controlled_stop=False,rapid_dispense=False,left_gate=False):
    source=(ROOT/'scripts/run_t1_spincoat_episode.py').read_text()
    source=source.replace("robot_dofs=np.array([m.jnt_dofadr[j] for j in range(m.njnt) if not m.joint(j).name.startswith(('pipette/','spincoater/'))])",
      "robot_dofs=np.array(sorted({int(m.jnt_dofadr[m.actuator_trnid[a,0]]) for a in range(m.nu) if not m.actuator(a).name.startswith('astribot_chassis_motor_') and m.actuator_trnid[a,0]>=0}))")
    source=source.replace("seed=np.load(ROOT/'research/hardware/grasp/probe.npz')['qpos'][0][qa]", "seed=d.qpos[qa].copy()")
    if kind=='grasp':
        source=source.replace('base=PIPETTE_START+GRASP_WRIST_OFFSET','base=PIPETTE_START+GRASP_WRIST_OFFSET+np.array([0,.4,0])')
    else:
        if fast_transfer:
            source=source.replace('task_t=t-prefix', "task_t=t-prefix\n        e=max(0.,task_t-9.)\n        task_t=float(np.interp(e,[0,2.,3.5,4.,4.2],[9.,13.,16.,18.,21.])) if e<=4.2 else 21.+e-4.2")
        if rapid_dispense:
            source=source.replace('e,[0,2.,3.5,4.,4.2],[9.,13.,16.,18.,21.])) if e<=4.2 else 21.+e-4.2','e,[0,1.2,2.4],[9.,13.,21.])) if e<=2.4 else 21.+e-2.4')
            # Map descent directly to task16 at2.4s; omit fixed settle/thumb holds.
            source=source.replace('[9.,13.,21.]','[9.,13.,16.]')
            source=source.replace('else 21.+e-2.4','else 21.+e-2.4')
        source=source.replace("        if phase!=last_phase:","""        workflow_press_enabled[0]=False
        if phase=='press':
            workflow_press_enabled[0]=bool(workflow_base_ready())
            if workflow_press_enabled[0]:
                if workflow_press_start[0] is None:workflow_press_start[0]=d.geom_xpos[pad].copy()
                workflow_press_clock[0]+=.001
                task_t=21.+workflow_press_clock[0]
            else:phase='wait_for_base_stop'
        if phase!=last_phase:""")
        source=source.replace('q,error=ik(m,d,base,HAND_QUAT,seed=seed,iterations=400);assert error<.001,error','q=d.qpos[qa].copy();error=0.')
        source=source.replace('d.qpos[qa]=q;d.qpos[hq]=openq','pass  # live stage; never reset robot or object')
        source=source.replace('desired=d.qpos.copy();desired_hand=close.copy()', 'desired=d.qpos.copy();desired_hand=current_hand_controls(m,d,hj)')
        source=source.replace('t=step*.001\n','t=step*.001+9.\n')
        source=source.replace('target=d.site_xpos[targetid].copy()','audit_target=d.site_xpos[targetid].copy();target=visual_target.copy()')
        source=source.replace('impact[:2]-target[:2]','impact[:2]-audit_target[:2]')
        source=source.replace('if not launched and d.qpos[button_q]<-.005:', 'if not launched and d.qpos[button_q]<-.005 and ledger.delivered_m3>=.95*ledger.p.capacity_m3:')
        source=source.replace('F.SpincoatFluid(F.FluidParameters())','F.SpincoatFluid(F.FluidParameters(volume_m3=ledger.remaining_volume_m3))')
        source=source.replace("if not valid:failure_reason='invalid_or_premature_dispense';break", "if not valid:failure_reason='invalid_or_premature_dispense';break\n            fluid=F.SpincoatFluid(F.FluidParameters(volume_m3=ledger.delivered_m3))")
        source=source.replace("if phase in ('thumb_above','press','retract','spin'):","if phase in ('press','retract','spin'):")
        source=source.replace("elif phase=='press':thumb_goal=top+axis*(.010-.025*smooth((task_t-21)/3))", "elif phase=='press':\n                        u=smooth((task_t-21)/3);thumb_goal=(1-u)*workflow_press_start[0]+u*(top-axis*.015)")
    if rapid_dispense:
        source=source.replace("else:phase='wait_for_base_stop'","else:phase='wait_for_motion_ready'")
        source=source.replace('u=smooth((task_t-21)/3);thumb_goal=', 'u=smooth((task_t-21)/1.8);thumb_goal=')
    if controlled_stop:
        source=source.replace('if motor_at is not None:omega=20*np.pi*min(1,(t-motor_at)/.5)', 'if motor_at is not None:omega=20*np.pi*min(1,(t-motor_at)/.5)*(1-smooth((t-motor_at-4.2)/2.))')
        source=source.replace('if motor_at is not None and t-motor_at>=4.2:break','if motor_at is not None and t-motor_at>=6.7:break')
    source=source.replace("'fixed_base':True", "'fixed_base':False")
    source=source.replace("'starts_from_home':args.from_home,'home_approach_duration_s':prefix", "'starts_from_home':False,'home_approach_duration_s':0.,'task_ready_approach_duration_s':prefix")
    source=source.replace("'pipette_start':PIPETTE_START.tolist()", "'pipette_start':(PIPETTE_START+np.array([0,.4,0])).tolist()")
    source=source.replace("'target':target.tolist(),'events':events", "'target':target.tolist(),'controller_clock_offset_s':workflow_clock_offset_s,'time_basis':'stage_logical_seconds','events':events")
    if kind=='grasp' and left_gate:
        source=source.replace("        if phase!=last_phase:", "        if phase=='close' and last_phase!='close' and not workflow_left_gate():failure_reason='left_base_not_stopped_before_close';break\n        if phase!=last_phase:")
    module=types.ModuleType('front_workflow_'+kind);module.__file__=str(ROOT/'scripts/run_t1_spincoat_episode.py')
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    module.GRASP_WRIST_OFFSET=np.array([-.025,.09,.12])
    module.workflow_press_clock=[0.]
    module.workflow_press_enabled=[False]
    module.workflow_press_start=[None]
    module.workflow_base_ready=lambda:False
    module.workflow_clock_offset_s=0.
    return module

def current_hand_controls(m,d,joints):
    result=[]
    for j in joints:
        a=np.flatnonzero(m.actuator_trnid[:,0]==j)
        result.append(d.ctrl[a[0]] if len(a) else d.qpos[m.jnt_qposadr[j]])
    return np.array(result)

def run_child(module,s,m,d,children,tag,argv,step_callback):
    module.scene=lambda:(s,m,d,children)
    original_step=mujoco.mj_step
    def advance(model,data,*args,**kwargs):
        step_callback(model,data,False)
        result=original_step(model,data,*args,**kwargs)
        step_callback(model,data,True)
        return result
    oldargv=sys.argv;sys.argv=['stage','--tag',tag,*argv];mujoco.mj_step=advance
    try:module.main()
    finally:mujoco.mj_step=original_step;sys.argv=oldargv
    path=ROOT/'logs/spincoat/t1'/tag
    with np.load(path/'episode.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    return arrays,json.loads((path/'result.json').read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--noslip-iterations',type=int,default=10);p.add_argument('--fast-transfer',action='store_true');p.add_argument('--rapid-dispense',action='store_true');p.add_argument('--rapid-docking',action='store_true');p.add_argument('--coarse-stop',action='store_true');p.add_argument('--hold-gain-scale',type=float,default=4.);p.add_argument('--absolute-mcp-overlay',action='store_true'); p.add_argument('--stop-after',choices=['grasp','aspirate','mobile','full'],default='full');args=p.parse_args()
    if args.coarse_stop and args.rapid_docking:p.error('--coarse-stop and historical --rapid-docking are mutually exclusive')
    out=ROOT/'logs/spincoat/t1'/args.tag;out.mkdir(parents=True,exist_ok=False)
    if args.coarse_stop:
        import t1_coarse_stop
        t1_coarse_stop.ZERO_SPEED_GAIN_SCALE=args.hold_gain_scale
    s,m,d,children=build_aspiration_scene(source_xy=(.4,.05))
    m.opt.noslip_iterations=args.noslip_iterations
    # Scene reset only: align the tool/rack with the left initial dock.
    stand=m.body('stand/pipette_stand').id;m.body_pos[stand,1]+=.4;s.body('stand/pipette_stand').pos[1]+=.4
    pipeq=m.joint('pipette/free').qposadr[0];d.qpos[pipeq+1]+=.4
    config=json.loads((ROOT/'research/hardware/task_ready_natural_01_pinky_010mm.json').read_text())
    for name,value in config['initial_joint_positions'].items():d.qpos[m.joint(name).qposadr[0]]=value
    from t1_clearance05_grasp import prepare_clearance05
    from preview_lab_materials_v2 import appearance_assets,lighting
    appdir=(out/'appearance').resolve();appdir.mkdir();old_d=d
    appearance_assets(s,appdir);m=s.compile();d=mujoco.MjData(m)
    assert m.nq==old_d.qpos.size and m.nv==old_d.qvel.size
    d.qpos[:]=old_d.qpos;d.qvel[:]=old_d.qvel;d.ctrl[:]=old_d.ctrl
    m.opt.noslip_iterations=10;lighting(m,True,neutral=True);m.vis.global_.offwidth=2048;m.vis.global_.offheight=2048;mujoco.mj_forward(m,d)
    grasp,mobile_profile,grasp_callback,grasp_audit=prepare_clearance05(m,d,out,coarse_stop=args.coarse_stop,runner_factory=lambda kind:adapted_runner(kind,left_gate=args.coarse_stop),absolute_mcp_overlay=args.absolute_mcp_overlay)
    model_identity=id(m);data_identity=id(d);handoffs=[]
    def handoff(label):
        assert id(m)==model_identity and id(d)==data_identity
        handoffs.append({'stage':label,'physical_time_s':float(d.time),'model_identity':id(m),'data_identity':id(d)})
        np.savez_compressed(out/('handoff_'+label+'.npz'),qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
    handoff('initial')
    g,gr=run_child(grasp,s,m,d,children,args.tag+'_grasp',
        ['--motion-profile',str(mobile_profile),'--close-pinky-before-lift','--lab-context','--max-seconds','9'],grasp_callback)
    ga=grasp_audit();(out/'grasp_contact_audit.json').write_text(json.dumps(ga,indent=2));handoff('grasp_end')
    np.savez_compressed(out/'grasp.npz',**g)
    stages={'grasp':{'no_contact_violation':not gr['robot_station_contact_violation'],'failure_reason':gr['failure_reason'],'note':'partial stage; child full-task success is expected false'}}
    result={'success':False,'stages':stages,'noslip_iterations':args.noslip_iterations,'fast_transfer':args.fast_transfer,'rapid_dispense':args.rapid_dispense,'rapid_docking':args.rapid_docking,'coarse_stop':args.coarse_stop,'hold_gain_scale':args.hold_gain_scale,'absolute_mcp_overlay':args.absolute_mcp_overlay,'no_runtime_qpos_writes':True,'grasp_prefix_clear':ga['prefix_clear'],'initial_approach':'clearance05_actual_wheels_MCP_open_25mm_close_entry','asp_index_DIP_extra_preload_rad':.015,'appearance':'V2_materials_V3_neutral_same_model_for_RGB','live_handoffs':handoffs,'known_limitations':['provisional camera and planar colored-substrate detector','GT odometry and arm IK; not VLA','kinematic aspiration volume proxy','T1 physical arm/torso colliders incomplete','uncalibrated frozen reduced spin-coating fluid']}
    def fine_readiness(start,end):
        with np.load(out/'left_coarse_control.npz') as z:
            select=(z['time']>=start)&(z['time']<end)
            speed=np.hypot(z['base_vx'][select],z['base_vy'][select]);yaw=np.abs(z['yaw_rate'][select]);workspace=z['workspace_error_m'][select]
            violations=(speed>=.001)|(yaw>=.002)|(workspace>.01)
            times=z['time'][select]
            return {'scope':'actual1kHz base speed/yaw/workspace during fine-stage motions','samples':len(times),'instantaneous_motion_pass':bool(not violations.any()),'violation_samples':int(violations.sum()),'violation_times_s':times[violations].tolist(),'max_speed_mm_s':float(speed.max()*1000) if len(speed) else None,'max_yawrate_rad_s':float(yaw.max()) if len(yaw) else None,'max_workspace_mm':float(workspace.max()*1000) if len(workspace) else None}
    def save():
        (out/'result.json').write_text(json.dumps(result,indent=2))
        np.savez_compressed(out/'checkpoint.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
        mujoco.mj_saveModel(m,str(out/'model.mjb'),None)
    save()
    if args.coarse_stop:
        from t1_coarse_stop import save_coarse_control
        stages['left_coarse_control']=save_coarse_control(grasp_audit.dock,out/'left_coarse_control.npz');save()
        stages['left_grasp_fine_readiness']=fine_readiness(5.8,float(d.time));save()
    if gr['failure_reason'] or gr['robot_station_contact_violation'] or not ga['prefix_clear'] or args.stop_after=='grasp':return
    if args.coarse_stop and not grasp_audit.dock.stopped:
        result['failure_reason']='left_base_not_stopped_before_aspiration';save();return
    # Single-finger contact preload; bounded, smooth, no force-cap change.
    # The original strict all-four-bearing-fingers gate is retained unchanged.
    original_asp=run_aspiration_on_state
    asp_globals=dict(original_asp.__globals__);original_hold=asp_globals['hold_control'];asp_started=float(d.time)
    index_act=next(a for a in range(m.nu) if 'right_index_DIP' in m.joint(int(m.actuator_trnid[a,0])).name)
    def index_preload_hold(model,data,desired):
        original_hold(model,data,desired)
        u=float(np.clip((data.time-asp_started)/.5,0,1));u=u*u*u*(10+u*(-15+6*u))
        value=data.ctrl[index_act]+.015*u
        if model.actuator_ctrllimited[index_act]:value=np.clip(value,*model.actuator_ctrlrange[index_act])
        data.ctrl[index_act]=value
    asp_globals['hold_control']=index_preload_hold
    if args.coarse_stop:asp_globals['MobileDockController']=lambda model,target:grasp_audit.dock
    live_asp=types.FunctionType(original_asp.__code__,asp_globals,name=original_asp.__name__,argdefs=original_asp.__defaults__,closure=original_asp.__closure__)
    live_asp.__kwdefaults__=original_asp.__kwdefaults__
    asp,ar,ledger=live_asp(m,d,source_geometry=SourceGeometry((.4,.05),.0195,.84,np.pi*.0195**2*.04,.88),provenance={'entry':'continuous initial ready and physical grasp; no held-state reset'})
    handoff('aspiration_end');np.savez_compressed(out/'aspiration.npz',**asp);stages['aspiration']=ar;save()
    if args.coarse_stop:
        from t1_coarse_stop import save_coarse_control
        stages['left_coarse_control']=save_coarse_control(grasp_audit.dock,out/'left_coarse_control.npz');stages['left_asp_fine_readiness']=fine_readiness(asp_started,float(d.time));save()
    if not ar['success'] or args.stop_after=='aspirate':return
    # The liquid observer stays active during transport: accidental presses
    # debit spill and may never silently refill the tool.
    basehold=d.ctrl.copy()
    def transport_hold(model,data,qref):
        for a in range(model.nu):
            if not model.actuator(a).name.startswith('astribot_chassis_motor_'):data.ctrl[a]=basehold[a]
        if data.time>ledger._last_time:
            ledger.update(data.time,float(data.qpos[model.joint('pipette/pipette_button').qposadr[0]]),data.site_xpos[model.site('pipette/tip_site').id],allow_dispense=False)
    mobile_advance=advance_mobile_stage
    if args.rapid_docking:
        from t1_rapid_docking import advance_rapid_docking
        mobile_advance=advance_rapid_docking
    if args.coarse_stop:
        from t1_coarse_stop import advance_coarse_stop
        mobile_advance=advance_coarse_stop
    mobile,mr,dock=mobile_advance(m,d,[0,0],duration_s=18,hold_callback=transport_hold,stop_when_docked=args.fast_transfer)
    handoff('mobile_end');np.savez_compressed(out/'mobile.npz',**mobile);stages['mobile']=mr;stages['transport_liquid']=ledger.audit();save()
    stages['transport_liquid_acceptance']={'criterion':'remaining dose >=95% capacity; raw conservative proxy loss retained, not claimed zero spill','remaining_fraction':ledger.remaining_volume_m3/ledger.p.capacity_m3}
    if not dock.stopped or ledger.remaining_volume_m3<.95*ledger.p.capacity_m3 or args.stop_after=='mobile':save();return
    base=m.body('chassis_base').id
    m.geom_rgba[m.geom('spincoater/liquid_visual').id,3]=0
    vision=capture_relocalization(m,d,d.xpos[base],d.xmat[base].reshape(3,3),out=out/'vision')
    stages['vision']=vision;save()
    if not vision['measurement']['valid']:return
    visual=np.array(vision['measurement']['position_world_m'])
    audit_target=d.site_xpos[m.site('spincoater/dispense_target').id].copy()
    stages['vision']['audit_error_mm']=float(np.linalg.norm(visual[:2]-audit_target[:2])*1000)
    # Ground truth is logged for scoring only; image-fit quality decides whether
    # the controller proceeds. Never gate on privileged audit error.
    core=adapted_runner('dispense',fast_transfer=args.fast_transfer or args.rapid_dispense,controlled_stop=True,rapid_dispense=args.rapid_dispense);core.visual_target=visual;core.ledger=ledger;core.current_hand_controls=current_hand_controls
    from t1_workflow_ik import install_validated_ik
    install_validated_ik(core,d.time)
    alignment_qualified_since=[None];alignment_log=[]
    def rapid_alignment_ready():
        tid=m.site('pipette/tip_site').id;pid=m.body('pipette/pipette').id
        goal=visual+np.array([.0001,0,.012]);error=float(np.linalg.norm(d.site_xpos[tid]-goal))
        vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,tid,vel,0);speed=float(np.linalg.norm(vel[3:]))
        axis=d.xmat[pid].reshape(3,3)[:,2];tilt=float(np.arccos(np.clip(axis[2],-1,1)))
        ready=error<.0003 and speed<.0075 and tilt<.005
        if not ready:alignment_qualified_since[0]=None
        elif alignment_qualified_since[0] is None:alignment_qualified_since[0]=float(d.time)
        qualified=ready and d.time-alignment_qualified_since[0]>=.04
        alignment_log.append([float(d.time),error,speed,tilt,float(dock.stopped),float(qualified)])
        return qualified
    core.workflow_base_ready=lambda:bool(dock.stopped and (not args.rapid_dispense or core.workflow_press_start[0] is not None or rapid_alignment_ready()))
    started=d.time
    wheel_ids=np.array([a for a in range(m.nu) if m.actuator(a).name.startswith('astribot_chassis_motor_')],dtype=int)
    base_joint=m.joint('chassis_free');base_qadr=int(base_joint.qposadr[0]);base_vadr=int(base_joint.dofadr[0]);coating_base_trace=[]
    core.workflow_clock_offset_s=started-9.;handoff('coat_start')
    def coating_step(model,data,after):
        core.workflow_nullspace_enabled=data.time-started<(3.5 if args.fast_transfer else 7.)
        if not after:
            dock.step(data)
            coating_base_trace.append(np.r_[float(data.time),data.qpos[base_qadr:base_qadr+7],data.qvel[base_vadr:base_vadr+6],data.ctrl[wheel_ids],float(dock.stopped)])
        else:
            relative=data.time-started+9
            ledger.update(data.time,float(data.qpos[model.joint('pipette/pipette_button').qposadr[0]]),data.site_xpos[model.site('pipette/tip_site').id],allow_dispense=core.workflow_press_enabled[0] and dock.stopped)
    coat,cr=run_child(core,s,m,d,children,args.tag+'_coat',['--lab-context','--max-seconds','28'],coating_step)
    np.savez_compressed(out/'coating_base_wheel_trace.npz',samples=np.asarray(coating_base_trace),wheel_actuator_names=np.array([m.actuator(a).name for a in wheel_ids]),columns=np.array(['time','base_x','base_y','base_z','qw','qx','qy','qz','vx','vy','vz','wx','wy','wz',*[m.actuator(a).name for a in wheel_ids],'stopped_gate']))
    np.savez_compressed(out/'rapid_alignment_gate.npz',columns=np.array(['time','tip_error_m','tip_speed_m_s','tilt_rad','original_base_stopped','alignment_gate']),samples=np.asarray(alignment_log))
    coat['time']=coat['time']-9+started
    cr['events']=[dict(e,time_s=e['time_s']+started-9.) for e in cr['events']];cr['time_basis']='physical_data_time_s'
    coat['metadata_json']=np.array(json.dumps(cr))
    if args.coarse_stop:stages['center_coarse_control']=save_coarse_control(dock,out/'center_coarse_control.npz')
    handoff('coat_end');np.savez_compressed(out/'coating.npz',**coat);stages['coating']=cr;stages['final_liquid']=ledger.audit()
    rotord=int(m.joint('spincoater/rotor').dofadr[0]);tip=d.site_xpos[m.site('pipette/tip_site').id];target=d.site_xpos[m.site('spincoater/dispense_target').id]
    final_rpm=abs(float(d.qvel[rotord]))*30/np.pi
    lastmask=coat['time']>=coat['time'][-1]-.5
    stop_rpm_max=float(np.max(np.abs(coat['qvel'][lastmask,rotord]))*30/np.pi)
    stages['ending']={'rotor_model':'prescribed kinematic angular velocity, not motor/brake dynamics','spin_preserved_s':4.2,'smooth_deceleration_s':2.,'zero_hold_s':.5,'actual_final_rpm':final_rpm,'last_half_second_max_rpm':stop_rpm_max,'tip_clearance_m':float(tip[2]-target[2]),'no_substrate_pick':True}
    motor_start=next(e['time_s'] for e in cr['events'] if e['phase']=='motor_start') if cr['release'] else None
    if motor_start is not None:
        split=motor_start+4.2;tailmask=coat['time']>=split
        def slice_log(mask):return {k:(v[mask] if v.ndim>0 and len(v)==len(coat['time']) else v) for k,v in coat.items()}
        tail=slice_log(tailmask);body=slice_log(~tailmask)
        tail['phase']=np.where(tail['time']<split+2.,'spin_down','finish_hold')
        stages['finish']=dict(stages.pop('ending'),events=[{'time_s':split,'phase':'rotor_deceleration'},{'time_s':split+2.,'phase':'rotor_stopped_hold'}],time_basis='physical_data_time_s',same_live_core_no_reset=True,success=bool(stop_rpm_max<1 and tip[2]-target[2]>=.08))
        tail['metadata_json']=np.array(json.dumps(stages['finish']))
        np.savez_compressed(out/'finish.npz',**tail);np.savez_compressed(out/'coating.npz',**body)
    result['success']=bool((not args.coarse_stop or (stages['left_grasp_fine_readiness']['instantaneous_motion_pass'] and stages['left_asp_fine_readiness']['instantaneous_motion_pass'])) and cr['success'] and ledger.delivered_m3>=.95*ledger.p.capacity_m3 and stop_rpm_max<1 and tip[2]-target[2]>=.08)
    save();print('INTEGRATED_WORKFLOW_RESULT',json.dumps(result),flush=True)

if __name__=='__main__':main()

