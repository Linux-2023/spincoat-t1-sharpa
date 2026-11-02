"""Fresh Rz60 neutral-wrist pickup through dispense/spin, with strict failures retained.

No acceptance threshold is relaxed. A complete functional replay may still fail
the original fine-base criteria, which remain part of the final engine result.
"""
import os,sys,json,hashlib,types,argparse
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco
from t1_spincoat_scene import ROOT
import t1_clearance05_grasp
from probe_t1_station_orientation_chain import station_prepare
from t1_neutral_view_clearance import advance_view_clearance


def main():
    options=argparse.ArgumentParser(add_help=False)
    options.add_argument('--contact-plan',type=Path)
    options.add_argument('--smooth-approach',action='store_true')
    options.add_argument('--reagent-preparation',action='store_true')
    options.add_argument('--tube-grasp-plan',type=Path)
    options.add_argument('--reagent-config',type=Path)
    options.add_argument('--traditional-test-tube',action='store_true')
    options.add_argument('--bimanual-preparation',action='store_true')
    options.add_argument('--left-index-clearance-rad',type=float,default=None)
    options.add_argument('--left-index-clearance-ramp-s',type=float,default=.5)
    options.add_argument('--settled-tube-reset',action='store_true')
    options.add_argument('--reset-tube-tilt-deg',type=float,default=0.)
    options.add_argument('--reset-tube-tilt-axis',choices=['x','y'],default='x')
    overrides,remaining=options.parse_known_args(sys.argv[1:])
    if overrides.reagent_preparation and overrides.tube_grasp_plan is None:
        options.error('--reagent-preparation requires an explicit --tube-grasp-plan')
    if overrides.reagent_preparation and overrides.reagent_config is None:
        options.error('--reagent-preparation requires an explicit --reagent-config')
    if overrides.traditional_test_tube and not overrides.reagent_preparation:
        options.error('--traditional-test-tube requires --reagent-preparation')
    if overrides.bimanual_preparation and not (overrides.reagent_preparation and overrides.traditional_test_tube):
        options.error('--bimanual-preparation requires --reagent-preparation and --traditional-test-tube')
    if overrides.settled_tube_reset and not overrides.bimanual_preparation:
        options.error('--settled-tube-reset requires --bimanual-preparation')
    if overrides.reset_tube_tilt_deg and not overrides.settled_tube_reset:
        options.error('--reset-tube-tilt-deg requires --settled-tube-reset')
    if overrides.left_index_clearance_rad is not None and not overrides.bimanual_preparation:
        options.error('--left-index-clearance-rad requires --bimanual-preparation')
    launch_argv=sys.argv[1:].copy()
    sys.argv=[sys.argv[0],*remaining]
    tag=sys.argv[sys.argv.index('--tag')+1];out=ROOT/'logs/spincoat/t1'/tag
    support=ROOT/'research/hardware/grasp_azimuth/support_staged_04'
    plan_path=overrides.contact_plan or support/'contact_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
    if not plan.get('planning_success'):raise ValueError('Contact plan must pass its planning checks.')
    engine_text=(ROOT/'scripts/run_t1_integrated_workflow.py').read_text(encoding='utf-8-sig')
    needle="'--max-seconds','9'";assert engine_text.count(needle)==1
    engine_text=engine_text.replace(needle,"'--max-seconds','14.375'")
    needle='    vision=capture_relocalization('
    assert engine_text.count(needle)==1
    engine_text=engine_text.replace(needle,"""    clearance,clearance_review=neutral_view_clearance(m,d,dock,ledger,out=out/'view_clearance')
    np.savez_compressed(out/'view_clearance.npz',**clearance)
    stages['view_clearance']=clearance_review;handoff('view_clearance_end');save()
    if not clearance_review['success']:return
"""+needle)
    if overrides.reagent_preparation:
        needle="    handoff('initial')"
        assert engine_text.count(needle)==1
        engine_text=engine_text.replace(needle,needle+"""
    from t1_coarse_stop import CoarseStopController
    prep_base=int(m.joint('chassis_free').qposadr[0])
    prep_dock=CoarseStopController(m,d.qpos[prep_base:prep_base+2].copy())
    prep,prep_review,prep_ledger=run_reagent_preparation(m,d,prep_dock,preparation_handles[0],out=out/'reagent_preparation')
    prep_review['pipette_clearance']=preparation_pipette_check(m,d)
    prep_review['success']=bool(prep_review['success'] and prep_review['pipette_clearance']['pass'])
    np.savez_compressed(out/'reagent_preparation.npz',**prep)
    handoff('reagent_preparation_end')
    if not prep_review['success']:
        mujoco.mj_saveModel(m,str(out/'model.mjb'),None)
        (out/'result.json').write_text(json.dumps({'success':False,'stages':{'reagent_preparation':prep_review},'live_handoffs':handoffs,'failure_reason':'reagent_preparation_failed'},indent=2))
        return
    prepared_source=prep_ledger.source_geometry()
    grasp_clock_origin[0]=float(d.time)
    mark_grasp_start(m,d)
    preparation_reanchor(m,d,mobile_profile)
    m.actuator_gainprm[grasp_audit.dock.wheel_ids]=grasp_audit.dock.route_gain
    m.actuator_biasprm[grasp_audit.dock.wheel_ids]=grasp_audit.dock.route_bias
    grasp_audit.dock.last_time=float(d.time)
    for key in grasp_timing:
        if key.endswith('_start_s') or key=='stage_end_s':grasp_timing[key]+=grasp_clock_origin[0]
    (out/'neutral_timing.json').write_text(json.dumps(dict(grasp_timing,time_basis='physical_data_time_s',grasp_clock_origin_s=grasp_clock_origin[0]),indent=2))
""")
        needle="    result={'success':False,'stages':stages,"
        assert engine_text.count(needle)==1
        engine_text=engine_text.replace(needle,"    stages['reagent_preparation']=prep_review\n"+needle)
        engine_text=engine_text.replace('fine_readiness(5.8,float(d.time))','fine_readiness(5.8+grasp_clock_origin[0],float(d.time))')
        needle='source_geometry=SourceGeometry((.4,.05),.0195,.84,np.pi*.0195**2*.04,.88)'
        assert engine_text.count(needle)==1
        engine_text=engine_text.replace(needle,'source_geometry=prepared_source')
        if overrides.bimanual_preparation:
            # The shared central beaker requires a longer lateral move and a
            # lower immersion target. Extend only the execution ceiling; the
            # observer's existing state timeouts and all gates remain intact.
            needle='asp,ar,ledger=live_asp(m,d,source_geometry=prepared_source,'
            assert engine_text.count(needle)==1
            engine_text=engine_text.replace(needle,'asp,ar,ledger=live_asp(m,d,duration=35.,source_geometry=prepared_source,')
    asp_text=(support/'executed_aspiration_adapter.py').read_text(encoding='utf-8-sig')
    if overrides.bimanual_preparation:
        # The professional shared beaker replaced the disabled legacy cup.
        # Keep the original force threshold while observing the actual walls.
        collision_filter="if 'aspiration_source' in names and any(n.startswith(('pipette/','sharpa_right/')) for n in names):"
        assert asp_text.count(collision_filter)==1
        asp_text=asp_text.replace(collision_filter,"if any(n=='aspiration_source' or n.endswith('/shared_griffin_beaker') for n in names) and any(n.startswith(('pipette/','sharpa_right/','sharpa_left/')) for n in names):")
    sources={'executed_integrated_engine.py':engine_text,'executed_aspiration_adapter.py':asp_text,
        'run_t1_neutral_integrated_workflow.py':Path(__file__).read_text(encoding='utf-8-sig'),
        'reference_clearance07.py':(ROOT/'research/hardware/direct_grasp/direct_oblique_07/executed_clearance_adapter.py').read_text(encoding='utf-8-sig')}
    for name in ['probe_t1_station_orientation_chain.py','t1_support_grasp.py','run_t1_integrated_workflow.py',
                 'run_t1_spincoat_episode.py','t1_coarse_stop.py','t1_workflow_ik.py','t1_yaw_motion_candidate.py',
                 't1_visual_relocalization.py','preview_lab_materials_v2.py','t1_neutral_view_clearance.py']:
        sources[name]=(ROOT/'scripts'/name).read_text(encoding='utf-8-sig')
    preparation=station_prepare
    if overrides.smooth_approach:
        from t1_continuous_approach import prepare_smooth_station_grasp
        preparation=prepare_smooth_station_grasp
        sources['t1_continuous_approach.py']=(ROOT/'scripts/t1_continuous_approach.py').read_text(encoding='utf-8-sig')
    if overrides.reagent_preparation:
        for name in ['t1_reagent_preparation.py','t1_reagent_preparation_stage.py','t1_reagent_workflow_adapter.py','probe_t1_reagent_reach.py']:
            sources[name]=(ROOT/'scripts'/name).read_text(encoding='utf-8-sig')
        sources['tube_grasp_plan.json']=overrides.tube_grasp_plan.read_text(encoding='utf-8-sig')
        sources['preparation_config.json']=overrides.reagent_config.read_text(encoding='utf-8-sig')
        if overrides.traditional_test_tube:
            for name in ['t1_glass_testtube_preparation.py','t1_glass_preparation_stage.py','t1_human_tube_mixing.py','t1_glass_pour_trajectory.py','t1_glass_rack_clearance.py']:
                sources[name]=(ROOT/'scripts'/name).read_text(encoding='utf-8-sig')
        if overrides.bimanual_preparation:
            for name in ['t1_bimanual_glass_preparation.py','t1_left_glass_preparation_stage.py','t1_bimanual_head_view.py','t1_left_hand_encoder_tracking.py','t1_left_continuous_ik.py','t1_left_cartesian_segment.py','t1_visible_view_clearance.py']:
                sources[name]=(ROOT/'scripts'/name).read_text(encoding='utf-8-sig')
            if overrides.left_index_clearance_rad is not None:
                sources['t1_left_index_clearance_overlay.py']=(ROOT/'scripts/t1_left_index_clearance_overlay.py').read_text(encoding='utf-8-sig')
            if overrides.settled_tube_reset:
                sources['settle_t1_glass_reset.py']=(ROOT/'scripts/settle_t1_glass_reset.py').read_text(encoding='utf-8-sig')
    engine=types.ModuleType('neutralB_integrated_engine');engine.__file__=str(ROOT/'scripts/run_t1_integrated_workflow.py')
    exec(compile(engine_text,engine.__file__,'exec'),engine.__dict__)
    engine.neutral_view_clearance=advance_view_clearance
    if overrides.bimanual_preparation:
        from t1_visible_view_clearance import advance_view_clearance_if_needed
        engine.neutral_view_clearance=advance_view_clearance_if_needed
    grasp_clock_origin=[0.]
    engine.grasp_clock_origin=grasp_clock_origin
    if overrides.reagent_preparation:
        from t1_reagent_workflow_adapter import augment_preparation_scene,relative_station_preparer,reanchor_ready_profile
        if overrides.traditional_test_tube:
            from t1_glass_testtube_preparation import GlassPreparationConfig as PreparationConfig,add_reagent_preparation as asset_builder
            from t1_glass_preparation_stage import advance_reagent_preparation
            if overrides.bimanual_preparation:
                from t1_bimanual_glass_preparation import BimanualGlassConfig as PreparationConfig,add_reagent_preparation as asset_builder
                from t1_left_glass_preparation_stage import advance_reagent_preparation
        else:
            from t1_reagent_preparation_stage import advance_reagent_preparation
            from t1_reagent_preparation import PreparationConfig,add_reagent_preparation as asset_builder
        original_builder=engine.build_aspiration_scene
        handles=[]
        def build_preparation(*args,**kwargs):
            config=PreparationConfig(**json.loads(sources['preparation_config.json']))
            scene,created=augment_preparation_scene(original_builder(*args,**kwargs),config,asset_builder=asset_builder)
            if overrides.settled_tube_reset:
                from settle_t1_glass_reset import initialize_naturally_settled_tube
                reset_out=out/'reagent_preparation'
                reset_out.mkdir(parents=True,exist_ok=True)
                axis=(1.,0.,0.) if overrides.reset_tube_tilt_axis=='x' else (0.,1.,0.)
                initialize_naturally_settled_tube(scene[1],scene[2],created,reset_out,
                    initial_tilt_deg=overrides.reset_tube_tilt_deg,tilt_axis_world=axis)
            handles.append(created)
            return scene
        engine.build_aspiration_scene=build_preparation
        engine.preparation_handles=handles
        tube_plan=json.loads(sources['tube_grasp_plan.json'])
        def run_reagent_preparation(*args,**kwargs):
            if overrides.bimanual_preparation:
                from t1_bimanual_head_view import BimanualHeadView
                head=BimanualHeadView(args[0],args[1])
                clearance=None
                control=head
                if overrides.left_index_clearance_rad is not None:
                    from t1_left_index_clearance_overlay import LeftIndexClearanceOverlay
                    clearance=LeftIndexClearanceOverlay(kwargs['out'],overrides.left_index_clearance_rad,ramp_s=overrides.left_index_clearance_ramp_s)
                    def control(m,d,phase):
                        head(m,d,phase)
                        clearance(m,d,phase)
                arrays,review,ledger=advance_reagent_preparation(*args,**kwargs,plan=tube_plan,control_hook=control)
                if clearance is not None:review['index_clearance_overlay']=clearance.save()
                review['head_view']=head.save(args[1],kwargs['out'])
                (Path(kwargs['out'])/'reagent_preparation_report.json').write_text(json.dumps(review,indent=2))
                return arrays,review,ledger
            return advance_reagent_preparation(*args,**kwargs,plan=tube_plan)
        engine.run_reagent_preparation=run_reagent_preparation
        engine.preparation_reanchor=reanchor_ready_profile
    asp_globals=dict(engine.run_aspiration_on_state.__globals__)
    exec(compile(asp_text,str(support/'executed_aspiration_adapter.py'),'exec'),asp_globals)
    engine.run_aspiration_on_state=asp_globals['run_aspiration_on_state']
    old_prepare=t1_clearance05_grasp.prepare_clearance05;timing={};initial_tip_z=[None];initial_pipe_position=[None]
    station_prepare.angle=60.;station_prepare.closing_scale=1.
    engine.grasp_timing=timing
    live_observation=[None,None];ignored_observation_steps=[0]
    def prepare(m,d,directory,coarse_stop=False,runner_factory=None,absolute_mcp_overlay=False):
        if not coarse_stop or not absolute_mcp_overlay:raise ValueError('Validated coarse-stop/MCP modes required.')
        live_observation[:]=[m,d]
        active_preparation=preparation
        if overrides.reagent_preparation:
            relative=relative_station_preparer(station_prepare,grasp_clock_origin,directory/'relative_station_prepare.py')
            if overrides.smooth_approach:
                import t1_continuous_approach
                old_station=t1_continuous_approach.baseline_station_prepare
                t1_continuous_approach.baseline_station_prepare=relative
                try:prepared=active_preparation(m,d,directory,plan,time_scale=1.25)
                finally:t1_continuous_approach.baseline_station_prepare=old_station
            else:prepared=relative(m,d,directory,plan,time_scale=1.25)
        else:prepared=active_preparation(m,d,directory,plan,time_scale=1.25)
        runner,path,callback,audit,clock=prepared
        timing.update(clock);mujoco.mj_forward(m,d)
        initial_tip_z[0]=float(d.site_xpos[m.site('pipette/tip_site').id,2])
        initial_pipe_position[0]=d.xpos[m.body('pipette/pipette').id].copy()
        (directory/'neutral_timing.json').write_text(json.dumps(clock,indent=2))
        if overrides.reagent_preparation:
            local_audit=audit
            def physical_audit():
                result=json.loads(json.dumps(local_audit()))
                for key in ('prefix_hand_tool_contacts_1khz','all_hand_tool_contacts_1khz'):
                    for contact in result.get(key,[]):contact['time_s']+=grasp_clock_origin[0]
                result['formal_close_time_s']+=grasp_clock_origin[0]
                result['grasp_clock_origin_s']=grasp_clock_origin[0]
                overlay=directory/'mcp_overlay_trace.npz'
                if overlay.exists():
                    with np.load(overlay) as saved:trace={key:saved[key].copy() for key in saved.files}
                    trace['samples'][:,0]+=grasp_clock_origin[0]
                    trace['time_basis']=np.array('physical_data_time_s')
                    np.savez_compressed(overlay,**trace)
                return result
            physical_audit.dock=local_audit.dock
            audit=physical_audit
        return runner,path,callback,audit
    t1_clearance05_grasp.prepare_clearance05=prepare
    native=mujoco.mj_step;records=[];exception=None;digits=['thumb','index','middle','ring','pinky']
    def observed_step(m,d,*args,**kwargs):
        result=native(m,d,*args,**kwargs)
        if m is not live_observation[0] or d is not live_observation[1]:
            ignored_observation_steps[0]+=1
            return result
        counts=np.zeros(5,int);normal=np.zeros(5);wrench=np.zeros(5);all_hand_contacts=0;all_hand_force=0.
        for ci,c in enumerate(d.contact):
            names=[m.body(int(m.geom_bodyid[int(g)])).name for g in c.geom]
            if not any(n.startswith('pipette/') for n in names):continue
            if any(n.startswith(('sharpa_right/','sharpa_left/')) for n in names):
                all_hand_contacts+=1;force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);all_hand_force+=float(np.linalg.norm(force[:3]))
            for j,name in enumerate(digits):
                if any(n.startswith('sharpa_right/right_'+name+'_') for n in names):
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                    counts[j]+=1;normal[j]+=abs(float(force[0]));wrench[j]+=float(np.linalg.norm(force[:3]))
        palm=m.body('sharpa_right/right_hand_C_MC').id;pipe=m.body('pipette/pipette').id
        f=d.xpos[m.body('astribot_arm_right_link_6').id]-d.xpos[m.body('astribot_arm_right_link_5').id]
        h=d.xpos[m.body('sharpa_right/right_middle_MCP_VL').id]-d.xpos[palm]
        angle=float(np.degrees(np.arccos(np.clip(f@h/(np.linalg.norm(f)*np.linalg.norm(h)),-1,1))))
        R=d.xmat[palm].reshape(3,3)
        records.append((float(d.time),float(d.qpos[m.joint('pipette/pipette_button').qposadr[0]]),counts,normal,wrench,
            angle,(R.T@(d.xpos[pipe]-d.xpos[palm])).copy(),(R.T@d.xmat[pipe].reshape(3,3)).copy(),all_hand_contacts,all_hand_force))
        return result
    def preparation_pipette_check(m,d):
        unexpected=[r for r in records if r[8]>0 and r[9]>.01]
        displacement=float(np.linalg.norm(d.xpos[m.body('pipette/pipette').id]-initial_pipe_position[0]))
        return dict(**{'pass':not unexpected},unexpected_force_contact_samples=len(unexpected),
            unexpected_contact_times_s=[r[0] for r in unexpected],pipette_end_translation_from_reset_mm=displacement*1000,
            scope='All recorded left/right hand-pipette contact above0.01N during preparation is unintended; static settling displacement separately retained.')
    def mark_grasp_start(m,d):
        initial_tip_z[0]=float(d.site_xpos[m.site('pipette/tip_site').id,2])
    engine.preparation_pipette_check=preparation_pipette_check
    engine.mark_grasp_start=mark_grasp_start
    original_child=engine.run_child
    def guarded_child(module,s,m,d,children,child_tag,argv,callback):
        raw,result=original_child(module,s,m,d,children,child_tag,argv,callback)
        if child_tag==tag+'_grasp':
            if overrides.reagent_preparation:
                raw['time']=raw['time']+grasp_clock_origin[0]
                metadata=json.loads(str(raw['metadata_json']))
                metadata['grasp_clock_origin_s']=grasp_clock_origin[0]
                metadata['time_basis']='physical_data_time_s'
                metadata['controller_clock_offset_s']=grasp_clock_origin[0]
                metadata['motion_profile_time_basis']='elapsed since grasp_clock_origin_s'
                for event in metadata.get('events',[]):event['time_s']+=grasp_clock_origin[0]
                raw['metadata_json']=np.array(json.dumps(metadata))
            tip_lift=float(d.site_xpos[m.site('pipette/tip_site').id,2]-initial_tip_z[0])
            held=[r for r in records if r[0]>=timing['hold_start_s']]
            counts=np.asarray([r[2][1:] for r in held]);forces=np.asarray([r[4][1:] for r in held])
            settled=[r for r in held if r[0]>=d.time-.5]
            settled_counts=np.asarray([r[2][1:] for r in settled]);settled_forces=np.asarray([r[4][1:] for r in settled])
            gate=dict(actual_tip_lift_m=tip_lift,required_tip_lift_m=.06,held_samples_1khz=len(held),
                all_four_contact_and_force=bool(len(held) and (counts>0).all() and (forces>.01).all()),
                historical_missing_support_samples=int(np.any((counts<=0)|(forces<=.01),axis=1).sum()) if len(held) else None,
                settled_support_interval_s=.5,settled_samples_1khz=len(settled),
                settled_all_four_contact_and_force=bool(len(settled)>=499 and (settled_counts>0).all() and (settled_forces>.01).all()),
                scope='Read-only handoff acceptance: actual lift and final0.5s stable support. Earlier transient gaps remain reported; not tactile control. All original base/aspiration/dispense criteria unchanged.')
            gate['pass']=bool(tip_lift>=.06 and gate['settled_all_four_contact_and_force'])
            (out/'actual_held_lift_gate.json').write_text(json.dumps(gate,indent=2))
            if not gate['pass']:result=dict(result,failure_reason='actual_held_lift_gate_failed')
        return raw,result
    engine.run_child=guarded_child;mujoco.mj_step=observed_step
    try:engine.main()
    except BaseException as exc:
        exception=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        mujoco.mj_step=native;t1_clearance05_grasp.prepare_clearance05=old_prepare
        out.mkdir(parents=True,exist_ok=True);snap=out/'executed_neutral_source';snap.mkdir(exist_ok=True)
        for name,content in sources.items():(snap/name).write_text(content,encoding='utf-8')
        (snap/'contact_plan.json').write_text(json.dumps(plan,indent=2))
        hashes={name:hashlib.sha256((snap/name).read_bytes()).hexdigest() for name in [*sources,'contact_plan.json']}
        if records:
            names=['time','button_qpos_m','finger_contact_count','finger_normal_force_N','finger_wrench_magnitude_N',
                'forearm_palm_angle_deg','relative_tool_position_in_palm_m','relative_tool_rotation_in_palm',
                'all_hand_pipette_contact_count','all_hand_pipette_contact_force_N']
            arrays={name:np.asarray([r[i] for r in records]) for i,name in enumerate(names)}
            np.savez_compressed(out/'fullrate_contact_observer.npz',**arrays,digit_names=np.array(digits))
            np.savez_compressed(out/'actual_button_1khz.npz',time=arrays['time'],button_qpos_m=arrays['button_qpos_m'])
        path=out/'result.json';result=json.loads(path.read_text()) if path.exists() else {'success':False}
        observer_scope=dict(binding='Actual model/data identity captured at prepare; independent RESET/workData steps execute natively but are not runtime records.',
            live_state_bound=live_observation[1] is not None,live_record_count=len(records),
            independent_workdata_step_count=ignored_observation_steps[0],
            live_physical_clock_strictly_increasing=bool(np.all(np.diff([r[0] for r in records])>0)) if records else None,
            live_physical_time_range_s=[records[0][0],records[-1][0]] if records else None)
        (out/'fullrate_observer_scope.json').write_text(json.dumps(observer_scope,indent=2))
        provenance=dict(entry='Fresh initial state; RESET-only joint rack/tool Rz60; same live model/data through all stages.',
            source_snapshot_sha256=hashes,contact_plan_sha256=hashes['contact_plan.json'],contact_plan_path=str(plan_path),argv=launch_argv,exception=exception,
            baseline_and_300_layouts_replaced=False,official_tool_adapter_changed=False,station_yaw_deg=60.,closing_scale=1.,
            runtime_object_or_root_pose_writes=False,observation_scope='Read-only contact/wrist audit; no tactile feedback input.',
            fullrate_observer_scope=observer_scope,
            smooth_approach=overrides.smooth_approach,
            reagent_preparation=overrides.reagent_preparation,
            traditional_test_tube=overrides.traditional_test_tube,
            bimanual_preparation=overrides.bimanual_preparation,
            settled_tube_reset=overrides.settled_tube_reset,
            reset_tube_tilt_deg=overrides.reset_tube_tilt_deg,
            reset_tube_tilt_axis=overrides.reset_tube_tilt_axis,
            preparation_pose_source=('Simulator ground-truth tube 6D alignment for geometric baseline; RGB estimation and learned policy are not validated by this episode.' if overrides.bimanual_preparation else None),
            acceptance='Original base/aspiration/dispense criteria retained. Functional completion does not override strict failure.',no_substrate_pick=True)
        result['neutralB_integration']=provenance
        (out/'neutralB_provenance.json').write_text(json.dumps(provenance,indent=2));path.write_text(json.dumps(result,indent=2))
        print('NEUTRAL_INTEGRATION_SUMMARY',json.dumps(dict(success=result['success'],stages=list(result.get('stages',{})),exception=exception)),flush=True)


if __name__=='__main__':main()
