"""Private v6 canonical scene adapter. All geometry precedes final compile."""
from pathlib import Path
import json

def _once(source,old,new):
    if source.count(old)!=1:raise RuntimeError('V6 layout source anchor '+repr(old[:100])+' count='+str(source.count(old)))
    return source.replace(old,new)

def adapt_neutral_scene_source(source,layout_path):
    layout_path=Path(layout_path);layout=json.loads(layout_path.read_text());targets=layout['main_asset_targets_world_xy_m']
    cup=tuple(targets['shared_beaker']);tube=tuple(targets['rack_and_tubes']);stand=tuple(targets['pipette_stand_tool']);coater=tuple(targets['spincoater'])
    anchor="            scene,created=augment_preparation_scene(original_builder(*args,**kwargs),config,asset_builder=asset_builder)"
    new="            from t1_interior_scene_v6 import build_scene as build_canonical_v6_scene\n            scene,created,v6_scene_report=build_canonical_v6_scene(out/'scene_reset_v6',cup_xy="+repr(cup)+",tube_xy="+repr(tube)+",stand_xy="+repr(stand)+",stand_yaw=60.,coater_xy="+repr(coater)+",decor_proposal="+repr(str(layout_path))+",appearance=False,settled_tube=False)"
    source=_once(source,anchor,new)
    old="config=__import__('dataclasses').replace(config,receiver_xy_m=(.40,.30))"
    source=_once(source,old,"config=__import__('dataclasses').replace(config,receiver_xy_m="+repr(cup)+",tube_xy_m="+repr(tube)+")")
    engine_anchor="    engine_text=(ROOT/'scripts/run_t1_integrated_workflow.py').read_text(encoding='utf-8-sig')"
    removal="\n    from t1_interior_layout_launch_v6 import remove_legacy_station_reset_shift\n    engine_text=remove_legacy_station_reset_shift(engine_text)"
    source=_once(source,engine_anchor,engine_anchor+removal)
    compile(source,__file__+' [canonical neutral source]','exec')
    return source

def remove_legacy_station_reset_shift(source):
    source=_once(source,"    stand=m.body('stand/pipette_stand').id;m.body_pos[stand,1]+=.4;s.body('stand/pipette_stand').pos[1]+=.4","    # V6 canonical station is already placed in the fully compiled RESET scene.")
    source=_once(source,"    pipeq=m.joint('pipette/free').qposadr[0];d.qpos[pipeq+1]+=.4","    # V6 free pipette RESET pose already matches its inward stand.")
    return source

def adapt_launch_source(source,layout_path):
    anchor=" namespace=dict(__name__='isolated_human_v3_neutral',"
    new=" from t1_interior_layout_launch_v6 import adapt_neutral_scene_source\n source=adapt_neutral_scene_source(source,"+repr(str(Path(layout_path)))+")\n"+anchor
    source=_once(source,anchor,new)
    layout=json.loads(Path(layout_path).read_text())
    assignment=' layout_config=RightWorkstationV4(station_shift_world_m=(a.v4_station_shift_x_m,a.v4_station_shift_y_m,0.))'
    config=layout['right_runtime_config']
    replacement=' from t1_right_interior_workstation_v6 import RightInteriorWorkstationV6\n layout_config=RightInteriorWorkstationV6(**'+repr(config)+').supported_station_config()'
    source=_once(source,assignment,replacement)
    source=source.replace('human_workflow_v4/launches','human_workflow_v6/launches').replace('v4_launch_snapshot','v6_launch_snapshot')
    compile(source,__file__+' [canonical v6 outer]','exec')
    return source
