"""Opt-in read-only full-state recorder before original RIGHT aspiration.
Integration state plus complete native MjData pickle retain solver warmstart and
pre-step-lagged derived transforms. No forward/step/reset in the recorder.
Only trusted locally generated pickle files may be restored by independent probes.
"""
from pathlib import Path
import json,hashlib,gzip,pickle
import numpy as np,mujoco

def _plain(value):
    if value is None or isinstance(value,(bool,int,float,str)):return value
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(list,tuple)):return [_plain(x) for x in value]
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    return {'class':type(value).__module__+'.'+type(value).__name__,'not_serialized_as_json':True}

def save_right_boundary_snapshot_v6(m,d,dock,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    flag=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,flag));mujoco.mj_getState(m,d,state,flag)
    # mj_step leaves transforms at its solve time while qpos is integrated.
    # Re-running mj_forward during cold restore would change this first input.
    fields=['xpos','xquat','xmat','xipos','ximat','geom_xpos','geom_xmat','site_xpos','site_xmat','qfrc_bias','actuator_force','qacc','qacc_warmstart']
    derived={key:getattr(d,key).copy() for key in fields}
    np.savez_compressed(out/'integration_state.npz',state=state,state_spec=int(flag),dt=float(m.opt.timestep),time=float(d.time),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy(),**{'derived_'+key:value for key,value in derived.items()})
    mujoco.mj_saveModel(m,str(out/'model.mjb'),None)
    with gzip.open(out/'native_data_trusted.pkl.gz','wb',compresslevel=1) as stream:pickle.dump(d,stream,protocol=5)
    controller={key:_plain(value) for key,value in dock.__dict__.items() if key not in ['model','driver','control_records','gain_records']}
    controller['class']=type(dock).__module__+'.'+type(dock).__name__;controller['driver_class']=type(dock.driver).__module__+'.'+type(dock.driver).__name__;controller['driver_state']={key:_plain(value) for key,value in vars(dock.driver).items()};controller['prior_control_record_count']=len(dock.control_records);controller['prior_gain_record_count']=len(dock.gain_records)
    np.savez_compressed(out/'controller_history_prefix.npz',control_records=np.asarray(dock.control_records),gain_records=np.asarray(dock.gain_records))
    (out/'controller_state.json').write_text(json.dumps(controller,indent=2))
    after=np.empty_like(state);mujoco.mj_getState(m,d,after,flag)
    if not np.array_equal(state,after) or any(not np.array_equal(value,getattr(d,key)) for key,value in derived.items()):raise AssertionError('Read-only boundary recorder changed actual data')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();scripts=Path(__file__).parent
    report=dict(scope='Read-only THIS fresh actual boundary, no integration/forward/control/pose writes',time_s=float(d.time),state_spec=int(flag),state_size=len(state),dt=float(m.opt.timestep),mujoco_version=mujoco.__version__,all_integration_and_selected_derived_arrays_unchanged=True,full_native_data_pickle_preserves_pre_step_lagged_transforms=True,model_sha256=sha(out/'model.mjb'),integration_state_sha256=sha(out/'integration_state.npz'),native_data_sha256=sha(out/'native_data_trusted.pkl.gz'),controller_state_sha256=sha(out/'controller_state.json'),source_sha256={key:sha(scripts/key) for key in ['t1_right_boundary_recorder_v6.py','t1_right_standing_motion_v6.py','t1_right_interior_workstation_v6.py','t1_right_direct_coating_v6.py','t1_coarse_stop.py']},original_controller_mode=dock.mode,hold_gain_scale=float(dock.hold_gain_scale),snapshot_alone_is_not_acceptance=True)
    (out/'report.json').write_text(json.dumps(report,indent=2));return report

def load_trusted_boundary_snapshot_v6(out):
    out=Path(out)
    with gzip.open(out/'native_data_trusted.pkl.gz','rb') as stream:data=pickle.load(stream)
    model=data.model;z=np.load(out/'integration_state.npz');state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    if not np.array_equal(state,z['state']):raise AssertionError('Native pickle integration state changed on roundtrip')
    for key in z.files:
        if key.startswith('derived_') and not np.array_equal(z[key],getattr(data,key[8:])):raise AssertionError('Derived native roundtrip differs: '+key)
    return model,data,json.loads((out/'controller_state.json').read_text())

def adapt_engine_source(source):
    anchor='    original_asp=run_aspiration_on_state\n'
    if source.count(anchor)!=1:raise RuntimeError('Unique original aspiration boundary anchor required')
    source=source.replace(anchor,"    from t1_right_boundary_recorder_v6 import save_right_boundary_snapshot_v6\n    stages['v6_right_asp_boundary_snapshot']=save_right_boundary_snapshot_v6(m,d,right_asp_dock,out/'right_asp_boundary_snapshot')\n    save()\n"+anchor)
    anchor="    asp_globals['hold_control']=index_preload_hold\n"
    if source.count(anchor)!=1:raise RuntimeError('Original aspiration preload globals anchor required')
    source=source.replace(anchor,anchor+"    from t1_right_boundary_recorder_v6 import instrument_asp_globals_v6\n    finish_right_boundary_trace=instrument_asp_globals_v6(asp_globals,m,d,grasp_audit.dock,out/'right_asp_boundary_snapshot')\n")
    anchor="    handoff('aspiration_end');"
    if source.count(anchor)!=1:raise RuntimeError('Original aspiration completion anchor required')
    return source.replace(anchor,"    finish_right_boundary_trace()\n"+anchor)

def adapt_neutral_source(source):
    lines=[line for line in source.splitlines() if line.startswith('    engine_text=adapt_engine_source_v6(')]
    if len(lines)!=1:raise RuntimeError('Apply boundary recorder after v6 RIGHT engine adapter installation')
    anchor=lines[0];return source.replace(anchor,anchor+'\n    from t1_right_boundary_recorder_v6 import adapt_engine_source as adapt_right_boundary_engine\n    engine_text=adapt_right_boundary_engine(engine_text)',1)

def adapt_launch_source(source):
    lines=[line for line in source.splitlines() if line.startswith(' source=adapt_neutral_source_v6(')]
    if len(lines)!=1:raise RuntimeError('Apply boundary recorder after v6 RIGHT outer adapter')
    anchor=lines[0];return source.replace(anchor,anchor+'\n from t1_right_boundary_recorder_v6 import adapt_neutral_source as adapt_right_boundary_neutral\n source=adapt_right_boundary_neutral(source)',1)


def instrument_asp_globals_v6(asp_globals,m,d,dock,out):
    """Private module proxy records original step inputs/outputs; no global hook."""
    import types,atexit
    out=Path(out);original_module=asp_globals['mujoco'];original_step=original_module.mj_step;proxy=types.SimpleNamespace(**vars(original_module));rows=[]
    baseq=int(m.joint('chassis_free').qposadr[0]);basev=int(m.joint('chassis_free').dofadr[0]);armq=np.array([int(m.joint('astribot_arm_right_joint_'+str(i)).qposadr[0]) for i in range(1,8)]);armv=np.array([int(m.joint('astribot_arm_right_joint_'+str(i)).dofadr[0]) for i in range(1,8)])
    def observed_step(model,data,*args,**kwargs):
        if model is not m or data is not d:return original_step(model,data,*args,**kwargs)
        before=(float(data.time),data.ctrl.copy(),data.qpos[baseq:baseq+7].copy(),data.qvel[basev:basev+6].copy(),data.qpos[armq].copy(),data.qvel[armv].copy(),np.asarray(dock.integral).copy(),float(dock.stable_duration),float(dock.stopped),str(dock.mode))
        value=original_step(model,data,*args,**kwargs)
        rows.append((*before,float(data.time),data.qvel[basev:basev+6].copy(),data.ctrl.copy()))
        return value
    proxy.mj_step=observed_step;asp_globals['mujoco']=proxy
    def finish():
        if rows:
            fields=['time_pre','ctrl','base_qpos_pre','base_qvel_pre','arm_qpos_pre','arm_qvel_pre','dock_integral','dock_stable_duration','dock_stopped','dock_mode','time_post','base_qvel_post','ctrl_post']
            np.savez_compressed(out/'asp_step_inputs_outputs_1khz.npz',**{name:np.array([row[i] for row in rows]) for i,name in enumerate(fields)})
        report=dict(scope='Read-only private aspiration mj_step proxy; original step/observers called exactly once, original inputs/outputs unchanged',samples=len(rows),pre_post_timestamps_explicit=True,full_robot_and_finger_ctrl_recorded=True,force_or_contact_enters_policy=False,global_mujoco_step_not_modified=True)
        (out/'asp_step_trace_report.json').write_text(json.dumps(report,indent=2));return report
    atexit.register(finish);return finish
