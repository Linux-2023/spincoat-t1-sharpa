"""Four-line opt-in v3 integration. Baseline v2 and neutral defaults unchanged."""
import argparse,sys,types,json,hashlib
from pathlib import Path

def changed(source,old,new):
 if source.count(old)!=1:raise RuntimeError('Source anchor mismatch '+repr(old[:100])+' count='+str(source.count(old)))
 return source.replace(old,new)

def adapt_stage_extras(source):
 source=changed(source,'transport=plan_palm_cartesian_move(m,d,armcmd,target_p,target_R,duration,ik_solver=continuous_left_solve_arm)',"transport=plan_palm_cartesian_move(m,d,armcmd,target_p,target_R,duration,ik_solver=(V3_SAFE_EXIT_SOLVER if name.startswith('pre_mix_clearance_') else continuous_left_solve_arm))")
 source=changed(source,'except RuntimeError as exc:failure=str(exc)','except (RuntimeError,ValueError) as exc:failure=type(exc).__name__+": "+str(exc)')
 source=changed(source,'if ledger.delivered_m3<c.target_pour_volume_m3*.99:', 'if ledger.delivered_m3<max(c.target_pour_volume_m3,c.tube_initial_volume_m3)*.99:')
 return source

def install_left(profile):
 import t1_left_glass_preparation_stage as left
 import run_t1_human_workflow_v2 as baseline
 source=Path(baseline.__file__).read_text()
 source=changed(source,'from t1_human_tube_mixing_v2 import MixingConfig,advance_mouth_pivot_shake','from t1_human_tube_mixing_v3 import MixingConfig,advance_mouth_pivot_shake')
 source=changed(source,"report['human_workflow_v2']=","report['human_workflow_v3']=")
 source=changed(source,'from t1_left_continuous_ik import solve_arm as continuous_ik','from t1_left_grip_human_v3 import solve_arm as continuous_ik')
 source=changed(source,"path=out/'executed_human_glass_stage_v2.py';path.write_text(source)","from t1_coarse_fine_motion_v3 import adapt_left_stage_source\n        source=V3_ADAPT_STAGE_EXTRAS(source)\n        source,coarse_globals=adapt_left_stage_source(source)\n        env.update(coarse_globals)\n        from t1_swirl_postheld_ik_buffer import safe_exit_solve_arm\n        env['V3_SAFE_EXIT_SOLVER']=safe_exit_solve_arm\n        from t1_left_loaded_continuous_neutral_v3 import solve_arm as loaded_continuous_ik\n        env['continuous_left_solve_arm']=loaded_continuous_ik\n        path=out/'executed_human_glass_stage_v3.py';path.write_text(source)")
 namespace=dict(__name__='isolated_left_human_v3',__file__=baseline.__file__,V3_ADAPT_STAGE_EXTRAS=adapt_stage_extras)
 exec(compile(source,baseline.__file__+' [v3 opt-in adapter]','exec'),namespace)
 namespace['install_v2'](profile,5.)
 return source

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--v3-pour-profile',type=Path,required=True);p.add_argument('--v3-station-shift-x-m',type=float,default=-.075)
 a,rest=p.parse_known_args();tag=rest[rest.index('--tag')+1]
 from t1_spincoat_scene import ROOT
 out=ROOT/'logs/spincoat/t1'/tag
 archive=ROOT/'research/hardware/human_workflow_v3/launches'/tag;archive.mkdir(parents=True,exist_ok=False)
 left_source=install_left(a.v3_pour_profile)
 import run_t1_neutral_integrated_workflow as baseline
 from t1_right_transport_low_v3 import adapt_aspiration_source,adapt_coating_source,RightTransportLowConfig
 from t1_right_transport_layout_v3 import translated_station_preparer,relocate_station_at_reset,RightStationLayoutConfig
 config=RightTransportLowConfig(stand_bypass_x_m=0.)
 layout_config=RightStationLayoutConfig(station_shift_world_m=(a.v3_station_shift_x_m,0.,0.))
 source=Path(baseline.__file__).read_text()
 source=changed(source,"    asp_globals=dict(engine.run_aspiration_on_state.__globals__)","    asp_text,right_low_globals,right_low_config=adapt_aspiration_source(asp_text,RIGHT_LOW_CONFIG)\n    asp_globals=dict(engine.run_aspiration_on_state.__globals__)\n    asp_globals.update(right_low_globals)\n    (V3_ARCHIVE/'right_low_config_v3.json').write_text(json.dumps(right_low_config,indent=2))")
 source=changed(source,"        original_builder=engine.build_aspiration_scene", "        original_unmoved_builder=engine.build_aspiration_scene\n        def original_builder(*args,**kwargs):\n            scene,reset_layout=relocate_station_at_reset(original_unmoved_builder(*args,**kwargs),RIGHT_LAYOUT_CONFIG)\n            out.mkdir(parents=True,exist_ok=True)\n            (out/'right_station_layout_v3.json').write_text(json.dumps(reset_layout,indent=2))\n            return scene")
 # Only the isolated coating runner's copy is transformed.
 core_anchor="    source=(ROOT/'scripts/run_t1_spincoat_episode.py').read_text()"
 core_injection=core_anchor+"\n    if kind!='grasp':source=adapt_coating_source(source,RIGHT_LOW_CONFIG)"
 engine_anchor="    engine_text=(ROOT/'scripts/run_t1_integrated_workflow.py').read_text(encoding='utf-8-sig')"
 source=changed(source,engine_anchor,engine_anchor+"\n    assert engine_text.count("+repr(core_anchor)+")==1, 'V3 coating adapter anchor mismatch'\n    engine_text=engine_text.replace("+repr(core_anchor)+","+repr(core_injection)+")")
 exec_anchor="    exec(compile(engine_text,engine.__file__,'exec'),engine.__dict__)"
 source=changed(source,exec_anchor,exec_anchor+"\n    engine.adapt_coating_source=adapt_coating_source\n    engine.RIGHT_LOW_CONFIG=RIGHT_LOW_CONFIG")
 namespace=dict(__name__='isolated_human_v3_neutral',__file__=baseline.__file__,adapt_aspiration_source=adapt_aspiration_source,adapt_coating_source=adapt_coating_source,RIGHT_LOW_CONFIG=config,V3_ARCHIVE=archive,RIGHT_LAYOUT_CONFIG=layout_config,relocate_station_at_reset=relocate_station_at_reset)
 exec(compile(source,str(archive/'executed_neutral_human_v3.py'),'exec'),namespace)
 namespace['station_prepare']=translated_station_preparer(namespace['station_prepare'],archive/'translated_station_prepare_v3.py',layout_config)
 (archive/'executed_left_install_v3.py').write_text(left_source)
 (archive/'executed_neutral_human_v3.py').write_text(source)
 (archive/'v3_entry_provenance.json').write_text(json.dumps(dict(scope='Four-line opt-in actual candidate; no success predeclared',entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),left_install_sha256=hashlib.sha256(left_source.encode()).hexdigest(),neutral_adapter_sha256=hashlib.sha256(source.encode()).hexdigest(),argv=rest,original_neutral_unchanged=True,original_v2_unchanged=True),indent=2))
 import t1_continuous_approach as smooth
 from t1_right_approach_v3 import install_ready_retreat
 install_ready_retreat(smooth,archive/'right_approach_v3.py',.08,preclose_retreat_x_m=.002)
 sys.argv=[str(Path(baseline.__file__)),*rest]
 try:namespace['main']()
 finally:
  if out.exists():
   import shutil
   shutil.copytree(archive,out/'v3_launch_snapshot',dirs_exist_ok=True)
if __name__=='__main__':main()