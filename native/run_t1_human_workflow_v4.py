"""Independent v4 entry: supported layout, crouch, near-rim human pouring.

All transformations are checked against the preserved v3 source. Live control
uses actuator commands; no robot/object pose writes are introduced.
"""
from pathlib import Path
import run_t1_human_workflow_v3 as v3

def transformed_source():
 s=Path(v3.__file__).read_text(encoding='utf-8-sig')
 def change(old,new):
  nonlocal s
  s=v3.changed(s,old,new)
 # Extend the generated physical LEFT stage before it is compiled.
 change(' return source\n\ndef install_left',
        ' from t1_loaded_crouch_v4 import adapt_loaded_body_stage_source\n source,_=adapt_loaded_body_stage_source(source)\n return source\n\ndef install_left')
 marker=" source=changed(source,\"report['human_workflow_v2']=\",\"report['human_workflow_v3']=\")"
 extra='''
 source=changed(source,'from t1_human_pour_profile_v2 import advance_human_pour_v2,IdleFingerOverlay','from t1_human_pour_profile_v2 import IdleFingerOverlay\\n    from t1_human_pour_profile_v4 import advance_human_pour_v4 as advance_human_pour_v2')
 source=changed(source,"hook=kwargs.get('control_hook');idle=[None]","from t1_loaded_crouch_v4 import LoadedCrouchControlHook as CrouchControlHook\\n        body_hook=CrouchControlHook(kwargs.get('control_hook'))\\n        kwargs['control_hook']=body_hook\\n        hook=kwargs.get('control_hook');idle=[None]")
 source=changed(source,"ledger.export_record(out/'gravity_liquid_record.json')","ledger.export_record(out/'gravity_liquid_record.json')\\n            report['body_crouch_v4']=body_hook.save(out)")
'''
 change(marker,marker+extra.rstrip())
 change("        env.update(coarse_globals)\\n", "        env.update(coarse_globals)\\n        from t1_loaded_crouch_v4 import adapt_loaded_coarse_entry_source\\n        source=adapt_loaded_coarse_entry_source(source)\\n        from t1_loaded_mouth_transport_v4 import advance_loaded_transport\\n        env['V3_ADVANCE_COARSE']=advance_loaded_transport\\n        from t1_loaded_crouch_v4 import LoadedCrouchControlHook as CrouchControlHook\\n        env['V4_BODY_CONFIG']=CrouchControlHook().config\\n")
 change("p.add_argument('--v3-station-shift-x-m',type=float,default=-.075)","p.add_argument('--v4-station-shift-x-m',type=float,default=.04);p.add_argument('--v4-station-shift-y-m',type=float,default=.08)")
 change("--v3-pour-profile","--v4-pour-profile")
 change("a.v3_pour_profile","a.v4_pour_profile")
 change("from t1_right_transport_layout_v3 import translated_station_preparer,relocate_station_at_reset,RightStationLayoutConfig", "from t1_right_workstation_v4 import translated_station_preparer,relocate_workstation_at_reset as relocate_station_at_reset,RightWorkstationV4,adapt_supported_aspiration_source")
 change("layout_config=RightStationLayoutConfig(station_shift_world_m=(a.v3_station_shift_x_m,0.,0.))", "layout_config=RightWorkstationV4(station_shift_world_m=(a.v4_station_shift_x_m,a.v4_station_shift_y_m,0.))")
 change('adapt_aspiration_source(asp_text,RIGHT_LOW_CONFIG)', 'adapt_supported_aspiration_source(asp_text,RIGHT_LAYOUT_CONFIG)')
 change('adapt_aspiration_source=adapt_aspiration_source,adapt_coating_source=', 'adapt_supported_aspiration_source=adapt_supported_aspiration_source,adapt_coating_source=')
 marker=' source=Path(baseline.__file__).read_text()\n source=changed(source,"    asp_globals'
 inject=''' source=Path(baseline.__file__).read_text()
 source=changed(source,"            config=PreparationConfig(**json.loads(sources['preparation_config.json']))","            config=PreparationConfig(**json.loads(sources['preparation_config.json']))\\n            config=__import__('dataclasses').replace(config,receiver_xy_m=(.40,.30))")
 source=changed(source,"            scene,created=augment_preparation_scene(original_builder(*args,**kwargs),config,asset_builder=asset_builder)","            scene,created=augment_preparation_scene(original_builder(*args,**kwargs),config,asset_builder=asset_builder)\\n            from t1_right_workstation_v4 import audit_shared_receiver_coherence\\n            coherence=audit_shared_receiver_coherence(scene[1],scene[2],config.receiver_xy_m)\\n            (out/'shared_receiver_coherence_v4.json').write_text(json.dumps(coherence,indent=2))")
 source=changed(source,"    asp_globals'''
 change(marker,inject)
 # Record labels change only for this opt-in run; physics references stay v3.
 s=s.replace('human_workflow_v3/launches','human_workflow_v4/launches')
 s=s.replace('executed_human_glass_stage_v3.py','executed_human_glass_stage_v4.py')
 s=s.replace("report['human_workflow_v3']=","report['human_workflow_v4']=")
 s=s.replace("out/'v3_launch_snapshot'","out/'v4_launch_snapshot'")
 return s

if __name__=='__main__':
 s=transformed_source()
 compile(s,__file__+' [v4 opt-in]','exec')
 ns=dict(__name__='__main__',__file__=__file__)
 exec(compile(s,__file__+' [v4 opt-in]','exec'),ns)
