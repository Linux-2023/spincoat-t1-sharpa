"""Opt-in human workflow v2. Frozen neutral workflow/defaults remain unchanged."""
import argparse,functools,json,sys,hashlib
from pathlib import Path
import numpy as np
import types

def record_json(value,*args,**kwargs):
    def convert(item):
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,np.ndarray):return item.tolist()
        raise TypeError(type(item).__name__+' is not a supported record type')
    kwargs.setdefault('default',convert)
    return json.dumps(value,*args,**kwargs)


def install_v2(pour_profile_path,amplitude_deg=10.):
    import t1_left_glass_preparation_stage as left
    import t1_bimanual_glass_preparation as assets
    from t1_glass_preparation_stage import build_glass_stage as original_build
    from t1_gravity_pour_liquid_v2 import GravityPourLedger
    from t1_gravity_liquid_render_v2 import attach_gravity_visuals
    from t1_liquid_palette import LIQUID_RGBA
    from t1_human_tube_mixing_v2 import MixingConfig,advance_mouth_pivot_shake
    from t1_human_pour_profile_v2 import advance_human_pour_v2,IdleFingerOverlay
    from t1_left_continuous_ik import solve_arm as continuous_ik
    profile=json.loads(Path(pour_profile_path).read_text(encoding='utf-8-sig'))
    # Static certificates contain baseline transforms; the live controller measures its own.
    profile.pop('measured_hand_to_tube_R',None);profile.pop('measured_mouth_in_hand_m',None)
    original_assets=assets.add_reagent_preparation
    def add_assets(spec,config=None):
        handles=original_assets(spec,config)
        attach_gravity_visuals(spec)
        for g in spec.geoms:
            if any(token in g.name for token in ('liquid_visual','liquid_proxy','tube_bottom_liquid_','pour_visual_','fluid_','film_visual')):
                g.rgba[:3]=LIQUID_RGBA[:3]
        return handles
    assets.add_reagent_preparation=add_assets
    def build_stage(**kwargs):
        if kwargs['side']!='left':raise ValueError('Human v2 is LEFT only')
        hook=kwargs.get('control_hook');idle=[None]
        def control(m,d,phase):
            if hook is not None:hook(m,d,phase)
            if idle[0] is None:idle[0]=IdleFingerOverlay(m)
            idle[0](m,d,phase)
        kwargs.update(ledger_class=GravityPourLedger,control_hook=control)
        base=original_build(**kwargs);out=Path(kwargs['out'])
        source=(out/'executed_left_glass_stage.py').read_text()
        env=base.__globals__.copy()
        env['json']=types.SimpleNamespace(dumps=record_json,loads=json.loads)
        env.update(solve_arm=continuous_ik,MixingConfig=functools.partial(MixingConfig,amplitude_deg=amplitude_deg),advance_mouth_pivot_shake=advance_mouth_pivot_shake,advance_human_pour_v2=advance_human_pour_v2,HUMAN_POUR_PROFILE=profile)
        previous_transport=env['cartesian_transport_phase']
        env['cartesian_transport_phase']=lambda phase:phase.startswith('pre_mix_clearance_') or previous_transport(phase)
        begin=source.index('  clear_lift=rack_clearance_lift_m(plan,float(d.xpos[body,2]))')
        end=source.index("  events.append(dict(time_s=float(d.time),phase='gentle_shake_mouth_pivot'))",begin)
        pre_mix="""  for pre_index,waypoint in enumerate(plan.get('pre_mix_tube_bottom_waypoints_world_m',[])):
   measured_R=d.xmat[palm].reshape(3,3).copy();measured_root=measured_R.T@(d.xpos[body]-d.xpos[palm])
   phase('pre_mix_clearance_'+str(pre_index),2.,goal(np.asarray(waypoint)-measured_R@measured_root,measured_R))
"""
        source=source[:begin]+pre_mix+source[end:]
        mix_def='  def mixing_step(qtarget,phase_name,allow_mix):'
        assert source.count(mix_def)==1
        source=source.replace(mix_def,'  mixing_anchor=d.site_xpos[mouth].copy()\n'+mix_def)
        mix_sample='armcmd=np.asarray(qtarget).copy();sample(phase_name,mix=allow_mix)'
        assert source.count(mix_sample)==1
        source=source.replace(mix_sample,mix_sample+"\n   if np.linalg.norm(d.site_xpos[mouth]-mixing_anchor)>.05:raise RuntimeError('Mix mouth departed50mm diagnostic envelope; stop before subsequent pour')")
        held_anchor=';window=full[-500:];allfour='
        assert source.count(held_anchor)==1
        readiness="""
  readiness_start=float(d.time);readiness_limit=float(plan.get('original_held_readiness_max_extra_s',0.))
  if not 0.<=readiness_limit<=1.:raise ValueError('Extra original-gate readiness is bounded to0..1s')
  for readiness_step in range(round(readiness_limit/m.opt.timestep)):
   recent=full[-500:]
   support=len(recent)==500 and all(r[2][0]>0 and r[3][0]>.01 and int(((r[2][1:]>0)&(r[3][1:]>.01)).sum())>=2 for r in recent)
   if support:
    pp=np.asarray([r[4] for r in recent]);rr=np.asarray([r[5] for r in recent]);move=float(np.linalg.norm(pp-pp[0],axis=1).max());turn=float(np.degrees(Rotation.from_matrix(np.einsum('ij,njk->nik',rr[0].T,rr)).magnitude()).max())
    if move<.002 and turn<3. and d.xpos[body,2]-tube_initial[2]>=required_lift_m:break
   sample('wait_for_original_held_readiness')
  events.append(dict(time_s=float(d.time),phase='original_held_readiness_complete',extra_wait_s=float(d.time-readiness_start),maximum_extra_wait_s=readiness_limit,scope='Original privileged simulation acceptance window; actuator targets unchanged, no force/gain/friction or gate changes'))
  window=full[-500:];allfour="""
        source=source.replace(held_anchor,readiness)
        begin=source.index("  for angle in pour_profile['tilt_waypoints_deg']:")
        end=source.index('  clear_lift=rack_clearance_lift_m(plan,float(d.xpos[body,2]))',begin)
        pour="""  def human_pour_step(qtarget,phase_name):
   nonlocal armcmd
   armcmd=np.asarray(qtarget).copy();sample(phase_name,pour=True)
  armcmd,human_pour_report,human_pour_trace=advance_human_pour_v2(m,d,handles,armcmd,human_pour_step,profile=HUMAN_POUR_PROFILE,ledger=ledger,ik_solver=continuous_left_solve_arm)
  (out/'human_pour_report.json').write_text(json.dumps(human_pour_report,indent=2))
  np.savez_compressed(out/'human_pour_trace.npz',**human_pour_trace)
  if ledger.delivered_m3<c.target_pour_volume_m3*.99:raise RuntimeError('Gravity pour delivered less than original99percent target; no flow valve/cap')
"""
        source=source[:begin]+pour+source[end:]
        # Persist actual robot states before any report serialization can fail.
        assert source.count('fullarrays={name:')==1
        source=source.replace('fullarrays={name:',"np.savez_compressed(out/'actual_human_preparation.npz',**arrays);fullarrays={name:")
        path=out/'executed_human_glass_stage_v2.py';path.write_text(source)
        exec(compile(source,str(path),'exec'),env);compiled=env['advance_reagent_preparation']
        def wrapped(*args,**kw):
            import mujoco
            try:arrays,report,ledger=compiled(*args,**kw)
            finally:mujoco.mj_saveModel(args[0],str(out/'model.mjb'),None)
            report=json.loads(record_json(report))
            ledger.export_record(out/'gravity_liquid_record.json')
            report['human_workflow_v2']=dict(amplitude_deg=amplitude_deg,pour_profile=profile,liquid_record='gravity_liquid_record.json',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Actual actuator simulation, reduced hydrostatic/ballistic liquid, not CFD or trained VLA')
            if idle[0] is not None:
                rows=idle[0].rows
                np.savez_compressed(out/'idle_finger_overlay_v2.npz',time=np.array([r[0] for r in rows]),phase=np.array([r[1] for r in rows]),commands=np.array([r[2] for r in rows]))
            (out/'reagent_preparation_report.json').write_text(json.dumps(report,indent=2))
            return arrays,report,ledger
        wrapped.encoder_tracker=base.encoder_tracker
        return wrapped
    left.build_glass_stage=build_stage
    return build_stage


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--v2-pour-profile',type=Path,required=True)
    p.add_argument('--v2-shake-amplitude-deg',type=float,default=10.)
    p.add_argument('--v2-preflight-dir',type=Path)
    args,rest=p.parse_known_args()
    build=install_v2(args.v2_pour_profile,args.v2_shake_amplitude_deg)
    if args.v2_preflight_dir:
        from t1_bimanual_glass_preparation import BimanualGlassConfig,BimanualGlassLedger
        build(side='left',config_class=BimanualGlassConfig,ledger_class=BimanualGlassLedger,out=args.v2_preflight_dir)
        print('V2_STAGE_COMPILE_PASS');return
    import run_t1_neutral_integrated_workflow as original
    sys.argv=[str(Path(original.__file__))]+rest
    original.main()

if __name__=='__main__':main()
