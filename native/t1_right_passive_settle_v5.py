"""Opt-in bounded passive wait for the existing RIGHT handoff gate.
Default: a predeclared fixed0.5s with no contact-based timing. The adaptive gate_diagnostic mode is diagnostic only. No hand-target changes or threshold relaxation. All prior missing support remains.
"""
import json,hashlib
from pathlib import Path
import numpy as np
import mujoco


def _ready(records, timing, now, lift):
    held=[r for r in records if r[0]>=timing['hold_start_s']]
    last=[r for r in held if r[0]>=now-.5]
    c=np.asarray([r[2][1:] for r in last]);f=np.asarray([r[4][1:] for r in last])
    return bool(lift>=.06 and len(last)>=499 and (c>0).all() and (f>.01).all())


def passive_wait_before_gate(m,d,raw,records,timing,initial_tip_z,callback,out,*,max_wait_s=2.,fixed_wait_s=.5,wait_mode="fixed"):
    """Use the already-bound native observer; callback retains the existing dock.

    `records` is the original neutral observer list. The caller's mj_step records
    each real physics sample. Fixed mode never changes timing based on contacts.
    Only the explicitly diagnostic mode lets the original gate end the wait.
    """
    if not 0<=max_wait_s<=5:raise ValueError('Passive wait must be bounded within5s')
    if wait_mode not in ['fixed','gate_diagnostic'] or not 0<=fixed_wait_s<=max_wait_s:raise ValueError('Invalid wait mode/duration')
    out=Path(out);start=float(d.time);fixed=d.ctrl.copy();dt=float(m.opt.timestep)
    wheel=np.array([i for i in range(m.nu) if m.actuator(i).name.startswith('astribot_chassis_motor_')]);nonwheel=np.setdiff1d(np.arange(m.nu),wheel)
    if np.max(abs(fixed[wheel]))>1e-12:raise ValueError('Passive wait requires existing zero-speed wheel hold')
    rd=np.array(sorted({int(m.jnt_dofadr[m.actuator_trnid[a,0]]) for a in nonwheel if m.actuator_trnid[a,0]>=0}))
    tip=m.site('pipette/tip_site').id;palm=m.body('sharpa_right/right_hand_C_MC').id;button=int(m.joint('pipette/pipette_button').qposadr[0]);rows=[];tail=[];environment_events=[]
    fields=['actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange','body_mass','body_inertia','geom_size','geom_friction','geom_condim','geom_solref','geom_solimp']
    before={k:getattr(m,k).copy() for k in fields};history_before=len(records);initial_ready=_ready(records,timing,d.time,float(d.site_xpos[tip,2]-initial_tip_z))
    def snapshot():
        r=records[-1];counts=np.asarray(r[2]);forces=np.asarray(r[4]);values=dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),time=float(d.time),phase='right_passive_support_settle',tip=d.site_xpos[tip].copy(),palm=d.xpos[palm].copy(),button=float(d.qpos[button]),grip_contacts=int(counts.sum()),finger_contacts=counts.copy(),finger_forces_N=forces.copy())
        sample={}
        for k,v in raw.items():
            if not isinstance(v,np.ndarray) or v.ndim==0 or len(v)!=len(raw['time']):continue
            if k in values:sample[k]=values[k]
            elif k in ['film','fluid_time','drop_film_time_s','desired_qpos','ik_error_m']:sample[k]=v[-1].copy()
            else:raise ValueError('Unsupported raw tail field '+k)
        tail.append(sample)
    snapshot()
    duration=fixed_wait_s if wait_mode=="fixed" else max_wait_s
    for i in range(int(round(duration/dt))):
        if wait_mode=='gate_diagnostic' and _ready(records,timing,d.time,float(d.site_xpos[tip,2]-initial_tip_z)):break
        d.ctrl[:]=fixed;d.qfrc_applied[:]=0;d.qfrc_applied[rd]=d.qfrc_bias[rd]
        callback(m,d,False)
        if np.max(abs(d.ctrl[nonwheel]-fixed[nonwheel]))>1e-12 or np.max(abs(d.ctrl[wheel]))>1e-12:raise RuntimeError('Existing callback changed passive command')
        mujoco.mj_step(m,d);callback(m,d,True)
        if len(records)<=history_before+i:raise RuntimeError('Existing1k observer must record every waited physics step')
        if not np.isfinite(d.qpos).all():raise RuntimeError('nonfinite passive state')
        # Exact original child obstacle predicate and >.001N threshold.
        for ci,c in enumerate(d.contact):
            names=[m.body(int(m.geom_bodyid[int(g)])).name for g in c.geom]
            obstacle=next((n for n in names if n.startswith(('lab_bench','stand/','spincoater/'))),None)
            moving=any(n and not n.startswith(('lab_bench','stand/','spincoater/','pipette/')) for n in names)
            if obstacle and (moving or (not obstacle.startswith('stand/') and any(n.startswith('pipette/') for n in names))):
                force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);mag=float(np.linalg.norm(force[:3]))
                if mag>.001:environment_events.append(dict(time_s=float(d.time),bodies=names,force_N=mag,depth_mm=max(0.,-float(c.dist)*1000)))
        rows.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()))
        if (i+1)%20==0:snapshot()
    if rows and (not tail or tail[-1]['time']<d.time-1e-9):snapshot()
    for k in (tail[0] if tail else {}):raw[k]=np.concatenate([raw[k],np.asarray([s[k] for s in tail])],axis=0)
    if 'metadata_json' in raw:
        meta=json.loads(str(raw['metadata_json']));meta.setdefault('events',[]).append(dict(time_s=start,phase='right_passive_support_settle'));meta['bounded_passive_wait_s']=float(d.time-start);meta['passive_tail_time_basis']='actual physical poststep d.time matching qpos; original child historical time convention preserved';raw['metadata_json']=np.array(json.dumps(meta))
    same={k:bool(np.array_equal(before[k],getattr(m,k))) for k in fields}
    if not all(same.values()):raise RuntimeError('Passive wait changed physical arrays '+repr(same))
    held=[r for r in records if r[0]>=timing['hold_start_s']];missing=[r[0] for r in held if np.any(np.asarray(r[2])[1:]<=0) or np.any(np.asarray(r[4])[1:]<=.01)]
    report=dict(original_environment_violation=bool(environment_events),original_environment_force_events=environment_events,scope=__doc__,wait_mode=wait_mode,contact_changes_runtime_timing=(wait_mode=="gate_diagnostic"),fixed_wait_s=fixed_wait_s,new_tail_time_basis="poststep physical d.time with qpos; native contact/geom cache retains original up-to-one-step observer lag",maximum_wait_s=max_wait_s,start_s=start,end_s=float(d.time),waited_s=float(d.time-start),initial_original_gate_ready=initial_ready,final_original_gate_ready=_ready(records,timing,d.time,float(d.site_xpos[tip,2]-initial_tip_z)),all_history_records_preserved=True,historical_missing_support_times_s=missing,physics_arrays_unchanged=same,hand_targets_and_wheel_commands_constant=True,original_gate_thresholds_unchanged=True,prior_base_fine_violations_are_not_cleared=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'right_passive_support_settle.json').write_text(json.dumps(report,indent=2))
    if rows:np.savez_compressed(out/'right_passive_support_settle_1khz.npz',time=np.array([r[0] for r in rows]),qpos=np.array([r[1] for r in rows]),qvel=np.array([r[2] for r in rows]),ctrl=np.array([r[3] for r in rows]))
    return raw,report


def adapt_neutral_source(source,*,max_wait_s=2.,fixed_wait_s=.5):
    anchor="            tip_lift=float(d.site_xpos[m.site('pipette/tip_site').id,2]-initial_tip_z[0])"
    if source.count(anchor)!=1:raise RuntimeError('Exact neutral held-gate anchor required once')
    insert="            from t1_right_passive_settle_v5 import passive_wait_before_gate\n            raw,passive_settle=passive_wait_before_gate(m,d,raw,records,timing,initial_tip_z[0],callback,out,max_wait_s="+repr(float(max_wait_s))+',fixed_wait_s='+repr(float(fixed_wait_s))+',wait_mode="fixed")\n            result=dict(result,right_passive_support_settle=passive_settle)\n            if passive_settle["original_environment_violation"]:result=dict(result,robot_station_contact_violation=True,failure_reason="passive_wait_environment_contact")\n'
    return source.replace(anchor,insert+anchor)


def adapt_launch_source(source,*,max_wait_s=2.,fixed_wait_s=.5):
    anchor=" layout_config=RightWorkstationV4(station_shift_world_m=(a.v4_station_shift_x_m,a.v4_station_shift_y_m,0.))\n source=Path(baseline.__file__).read_text()\n"
    if source.count(anchor)!=1:raise RuntimeError('Exact outer neutral source-read anchor required once')
    insertion=" from t1_right_passive_settle_v5 import adapt_neutral_source as adapt_right_passive_settle_source\n source=adapt_right_passive_settle_source(source,max_wait_s="+repr(float(max_wait_s))+',fixed_wait_s='+repr(float(fixed_wait_s))+')\n'
    return source.replace(anchor,anchor+insertion)

