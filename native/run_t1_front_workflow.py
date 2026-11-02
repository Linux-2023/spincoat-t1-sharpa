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

def adapted_runner(kind,fast_transfer=False):
    source=(ROOT/'scripts/run_t1_spincoat_episode.py').read_text()
    source=source.replace("robot_dofs=np.array([m.jnt_dofadr[j] for j in range(m.njnt) if not m.joint(j).name.startswith(('pipette/','spincoater/'))])",
      "robot_dofs=np.array(sorted({int(m.jnt_dofadr[m.actuator_trnid[a,0]]) for a in range(m.nu) if not m.actuator(a).name.startswith('astribot_chassis_motor_') and m.actuator_trnid[a,0]>=0}))")
    source=source.replace("seed=np.load(ROOT/'research/hardware/grasp/probe.npz')['qpos'][0][qa]", "seed=d.qpos[qa].copy()")
    if kind=='grasp':
        source=source.replace('base=PIPETTE_START+GRASP_WRIST_OFFSET','base=PIPETTE_START+GRASP_WRIST_OFFSET+np.array([0,.4,0])')
    else:
        if fast_transfer:
            source=source.replace('task_t=t-prefix', "task_t=t-prefix\n        e=max(0.,task_t-9.)\n        task_t=float(np.interp(e,[0,2.,3.5,4.,4.2],[9.,13.,16.,18.,21.])) if e<=4.2 else 21.+e-4.2")
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
    source=source.replace("'fixed_base':True", "'fixed_base':False")
    source=source.replace("'starts_from_home':args.from_home,'home_approach_duration_s':prefix", "'starts_from_home':False,'home_approach_duration_s':0.,'task_ready_approach_duration_s':prefix")
    source=source.replace("'pipette_start':PIPETTE_START.tolist()", "'pipette_start':(PIPETTE_START+np.array([0,.4,0])).tolist()")
    source=source.replace("'target':target.tolist(),'events':events", "'target':target.tolist(),'controller_clock_offset_s':workflow_clock_offset_s,'time_basis':'stage_logical_seconds','events':events")
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
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--noslip-iterations',type=int,default=0);p.add_argument('--fast-transfer',action='store_true');p.add_argument('--stop-after',choices=['grasp','aspirate','mobile','full'],default='full');args=p.parse_args()
    out=ROOT/'logs/spincoat/t1'/args.tag;out.mkdir(parents=True,exist_ok=False)
    s,m,d,children=build_aspiration_scene(source_xy=(.4,.05))
    m.opt.noslip_iterations=args.noslip_iterations
    # Scene reset only: align the tool/rack with the left initial dock.
    stand=m.body('stand/pipette_stand').id;m.body_pos[stand,1]+=.4;s.body('stand/pipette_stand').pos[1]+=.4
    pipeq=m.joint('pipette/free').qposadr[0];d.qpos[pipeq+1]+=.4
    config=json.loads((ROOT/'research/hardware/task_ready_natural_01_pinky_010mm.json').read_text())
    for name,value in config['initial_joint_positions'].items():d.qpos[m.joint(name).qposadr[0]]=value
    # Reset to wheel-supported height, then preserve original WORLD approach FK.
    # Runtime root motion remains wheel dynamics; no qpos writes in the stage.
    d.qpos[m.joint('chassis_free').qposadr[0]+2]=.09117
    mujoco.mj_forward(m,d)
    from t1_mobile_grasp_profile import adapt_grasp_profile
    mobile_profile=adapt_grasp_profile(m,d,ROOT/'research/hardware/task_ready_natural_01_pinky_010mm_profile.json',out/'mobile_grasp_profile.json')
    dock=MobileDockController(m,[0,.4])
    grasp=adapted_runner('grasp')
    g,gr=run_child(grasp,s,m,d,children,args.tag+'_grasp',
        ['--motion-profile',str(mobile_profile),'--close-pinky-before-lift','--lab-context','--max-seconds','9'],
        lambda model,data,after:None if after else dock.step(data))
    np.savez_compressed(out/'grasp.npz',**g)
    stages={'grasp':{'no_contact_violation':not gr['robot_station_contact_violation'],'failure_reason':gr['failure_reason'],'note':'partial stage; child full-task success is expected false'}}
    result={'success':False,'stages':stages,'noslip_iterations':args.noslip_iterations,'fast_transfer':args.fast_transfer,'no_runtime_qpos_writes':True,'known_limitations':['provisional camera and planar colored-substrate detector','GT odometry and arm IK; not VLA','kinematic aspiration volume proxy','T1 physical arm/torso colliders incomplete','uncalibrated frozen reduced spin-coating fluid']}
    def save():
        (out/'result.json').write_text(json.dumps(result,indent=2))
        np.savez_compressed(out/'checkpoint.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
        mujoco.mj_saveModel(m,str(out/'model.mjb'),None)
    save()
    if gr['failure_reason'] or gr['robot_station_contact_violation'] or args.stop_after=='grasp':return
    asp,ar,ledger=run_aspiration_on_state(m,d,source_geometry=SourceGeometry((.4,.05),.0195,.84,np.pi*.0195**2*.04,.88),provenance={'entry':'continuous initial ready and physical grasp; no held-state reset'})
    np.savez_compressed(out/'aspiration.npz',**asp);stages['aspiration']=ar;save()
    if not ar['success'] or args.stop_after=='aspirate':return
    # The liquid observer stays active during transport: accidental presses
    # debit spill and may never silently refill the tool.
    basehold=d.ctrl.copy()
    def transport_hold(model,data,qref):
        for a in range(model.nu):
            if not model.actuator(a).name.startswith('astribot_chassis_motor_'):data.ctrl[a]=basehold[a]
        if data.time>ledger._last_time:
            ledger.update(data.time,float(data.qpos[model.joint('pipette/pipette_button').qposadr[0]]),data.site_xpos[model.site('pipette/tip_site').id],allow_dispense=False)
    mobile,mr,dock=advance_mobile_stage(m,d,[0,0],duration_s=18,hold_callback=transport_hold,stop_when_docked=args.fast_transfer)
    np.savez_compressed(out/'mobile.npz',**mobile);stages['mobile']=mr;stages['transport_liquid']=ledger.audit();save()
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
    core=adapted_runner('dispense',fast_transfer=args.fast_transfer);core.visual_target=visual;core.ledger=ledger;core.current_hand_controls=current_hand_controls
    from t1_workflow_ik import install_validated_ik
    install_validated_ik(core,d.time);core.workflow_base_ready=lambda:dock.stopped
    started=d.time
    core.workflow_clock_offset_s=started-9.
    def coating_step(model,data,after):
        core.workflow_nullspace_enabled=data.time-started<(3.5 if args.fast_transfer else 7.)
        if not after:dock.step(data)
        else:
            relative=data.time-started+9
            ledger.update(data.time,float(data.qpos[model.joint('pipette/pipette_button').qposadr[0]]),data.site_xpos[model.site('pipette/tip_site').id],allow_dispense=core.workflow_press_enabled[0] and dock.stopped)
    coat,cr=run_child(core,s,m,d,children,args.tag+'_coat',['--lab-context','--max-seconds','28'],coating_step)
    coat['time']=coat['time']-9+started
    cr['events']=[dict(e,time_s=e['time_s']+started-9.) for e in cr['events']];cr['time_basis']='physical_data_time_s'
    coat['metadata_json']=np.array(json.dumps(cr))
    np.savez_compressed(out/'coating.npz',**coat);stages['coating']=cr;stages['final_liquid']=ledger.audit()
    result['success']=bool(cr['success'] and ledger.delivered_m3>=.95*ledger.p.capacity_m3)
    save();print('FRONT_WORKFLOW_RESULT',json.dumps(result),flush=True)

if __name__=='__main__':main()
