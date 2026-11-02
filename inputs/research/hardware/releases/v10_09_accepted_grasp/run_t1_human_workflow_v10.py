"""V7 opposite LEFT pour and natural RIGHT wrist, fresh actual workflow.
Fresh original gates must pass; static or independent branches are not success.
"""
import json,sys,hashlib
from pathlib import Path
ROOT=Path('/home/agilex/spincoat-t1-sharpa')
DEFAULT_LAYOUT=ROOT/'research/hardware/human_workflow_v7/shared_layout_03_anatomical_mount.json'

def compose_source(layout_path,*,left_adapter=None,record_actual_clock=False,record_asp_boundary=False):
    import run_t1_human_workflow_v4 as base
    from t1_right_seat_alignment_v5 import adapt_launch_source as seat
    from t1_right_passive_settle_v5 import adapt_launch_source as settle
    from t1_deferred_idle_restore_v5 import adapt_launch_source as idle
    from t1_right_standing_motion_v6 import adapt_launch_source as right
    from t1_right_interior_workstation_v6 import RightInteriorWorkstationV6
    from t1_natural_layout_launch_v7 import adapt_launch_source as layout
    cfg=RightInteriorWorkstationV6(**json.loads(Path(layout_path).read_text())['right_runtime_config'])
    if left_adapter is None:
        from t1_left_opposite_workflow_v7 import adapt_launch_source as left_adapter
    source=base.transformed_source()
    source=seat(source);source=settle(source);source=idle(source)
    source=right(source,cfg)
    # The v7 RIGHT candidate only changes existing placement/free-axis yaw config.
    # No new runtime wrist optimizer or controller is installed.
    if record_actual_clock:
        from t1_actual_record_clock_v6 import adapt_launch_source as physical_clock
        source=physical_clock(source)
    if record_asp_boundary:
        from t1_right_boundary_recorder_v6 import adapt_launch_source as snapshot_boundary
        source=snapshot_boundary(source)
    source=left_adapter(source)
    if cfg.stand_yaw_deg == 75.0:
        from t1_right_station_yaw_launch_v7 import adapt_launch_source as station_yaw
        source=station_yaw(source)
    elif cfg.stand_yaw_deg != 60.0:
        raise ValueError('V7 has only explicit original60 or candidate75 station adapters')
    source=layout(source,layout_path)
    source=source.replace('human_workflow_v6/launches','human_workflow_v7/launches').replace('v6_launch_snapshot','v7_launch_snapshot')
    compile(source,__file__+' [v6 composed]','exec')
    return source

def main():
    layout_path=DEFAULT_LAYOUT
    if '--v7-layout' in sys.argv:
        i=sys.argv.index('--v7-layout');layout_path=Path(sys.argv[i+1]);del sys.argv[i:i+2]
    record_actual_clock='--record-actual-clock' in sys.argv
    record_asp_boundary='--record-asp-boundary' in sys.argv
    for option in ['--record-actual-clock','--record-asp-boundary']:
        if option in sys.argv:sys.argv.remove(option)
    left_adapter=None
    if '--v4-pour-profile' in sys.argv:
        selected_profile=Path(sys.argv[sys.argv.index('--v4-pour-profile')+1])
        selected_options=json.loads(selected_profile.read_text())
        if selected_options.get('v7_empty_ready_clearance',False):
            from t1_empty_hand_ready_v7 import adapt_workflow_launch_source as left_adapter
        elif selected_options.get('v7_actual_return_reanchor',False):
            from t1_actual_frame_return_v7 import adapt_workflow_launch_source as left_adapter
    source=compose_source(layout_path,left_adapter=left_adapter,record_actual_clock=record_actual_clock,record_asp_boundary=record_asp_boundary)
    from t1_transition_flow_v8 import adapt_launch
    source=adapt_launch(source)
    from t1_single_grasp_v10 import adapt_launch as adapt_single_grasp
    source=adapt_single_grasp(source)
    tag=sys.argv[sys.argv.index('--tag')+1];out=ROOT/'logs/spincoat/t1'/tag
    out_snapshot=ROOT/'research/hardware/human_workflow_v7/composed_sources';out_snapshot.mkdir(parents=True,exist_ok=True)
    (out_snapshot/(tag+'.py')).write_text(source)
    try:exec(compile(source,__file__+' [v6 composed]','exec'),dict(__name__='__main__',__file__=__file__))
    finally:
        if out.exists():
            report=dict(scope=__doc__,layout_path=str(layout_path),layout_sha256=hashlib.sha256(Path(layout_path).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),privileged_pose_geometry_not_vla=True,uncalibrated_reduced_liquid_not_cfd=True,original_acceptance_thresholds_retained=True,no_postaspiration_x_bypass=True,no_forced_leg_crouch=True,left_opposite_pour_and_shared_right_cup=True,right_natural_wrist_v7=True,left_mount_convention='RIGHT_anatomical_mirror_Rz180_provisional',hardware_mount_calibrated=False,poststep_physical_record_clock=record_actual_clock,read_only_full_asp_boundary_snapshot=record_asp_boundary)
            (out/'natural_workflow_v7_provenance.json').write_text(json.dumps(report,indent=2))
            (out/'interior_workflow_v6_provenance.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
