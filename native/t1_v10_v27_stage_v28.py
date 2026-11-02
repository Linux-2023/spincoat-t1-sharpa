"""V10-based continuous workflow with V27 motion and optional V30 hold candidates.
Explicit hold profiles select shorter timing and bounded finger commands only.
Original liquid proxy, physical gates and subsequent stages are retained.
"""
from t1_transition_flow_v8 import once

def adapt_left_stage(source):
    source=once(source,' import copy',' import copy\n from t1_wrist_mixing_v27 import MixingConfig,advance_mouth_pivot_shake\n from t1_radial_preload_v30 import RadialPreload\n from t1_balanced_pinch_v30 import BalancedPinch\n from t1_object_wrench_v30 import ObjectWrenchHold\n from t1_fullhand_preload_v31 import FullHandPreload\n from t1_frozen_transport_hold_v31 import FrozenTransportHold\n from t1_five_pad_hold_v31 import FivePadHold\n hold_config=(plan or {}).get("v30_hold")\n holder={"radial":RadialPreload,"balanced":BalancedPinch,"wrench":ObjectWrenchHold,"fullhand":FullHandPreload,"freeze":FrozenTransportHold,"fivepad":FivePadHold}[hold_config.get("controller","radial")](m,hold_config["extra_force_N"]) if hold_config else None')
    source=once(source,'config=MixingConfig(tube_length_m=c.tube_length_m)','config=MixingConfig(tube_length_m=c.tube_length_m,record_dir=str(out),**({key:hold_config[key] for key in ("steady_s","ramp_s")} if hold_config else {}))')
    source=once(source,'CONTROL_HOOK(m,d,phase) if CONTROL_HOOK is not None else None;','CONTROL_HOOK(m,d,phase) if CONTROL_HOOK is not None else None;d.ctrl[hacts]=handcmd if hold_config and hold_config.get("controller")=="fivepad" else d.ctrl[hacts];holder.apply(m,d,phase) if holder is not None else None;')
    source=once(source,'  nonlocal armcmd,handcmd','  nonlocal armcmd,handcmd\n  if name=="approach_outside_tube" and plan.get("v31_preshape_during_approach",False):goalhand=openq')
    source=once(source,'from t1_left_opposite_workflow_v7 import retarget_from_live,make_standing_transport','from t1_left_opposite_workflow_v7 import retarget_from_live,make_standing_transport\n  if V6_RUNTIME_PROFILE.get("v31_preturn_fraction"):\n   from t1_preturn_route_v31 import retarget_from_live')
    compile(source,'<V28 V10 grasp plus V27 motion full stage>','exec')
    return source

def adapt_launch(source):
    anchor="path=out/'executed_human_glass_stage_v4.py';path.write_text(source)"
    return once(source,anchor,'from t1_v10_v27_stage_v28 import adapt_left_stage as adapt_v28\\n        source=adapt_v28(source)\\n        '+anchor)
