"""Schedule original RIGHT preshape while LEFT returns its empty hand.

Install last in the frozen R08/TOPP candidate. The original full LEFT stage,
base docks, contact/loaded-motion clock and all acceptance gates remain intact.
"""
from __future__ import annotations

from functools import wraps
import inspect
import json
import linecache
from pathlib import Path
import textwrap

import numpy as np
from scipy.interpolate import BPoly

import v34_overlap_control as overlap
import v34_topp_core as core
import v34_topp_right as right
from v34_topp_right_paths import acquisition_task_path, closure, copy_callable, derivative_enclosure, exact, prefix_path
from v34_smoothtubereturn_bounds import restrict_curve

PATCHED_SYMBOLS = (
    't1_v10_v27_stage_v28.adapt_left_stage',
    'v34_speedrightpickup.retime_engine_source',
    'v34_speedrightpickup.retime_prepared',
    't1_continuous_approach.adapted_runner',
    'v34_topp_right._Acquisition',
    't1_right_standing_motion_v6.reanchor_prefix_at_actual_dock_v6',
    't1_right_standing_motion_v6.advance_standing_dock_v6',
    't1_left_glass_preparation_stage.advance_reagent_preparation',
    'run_t1_full_workflow_v28.main',
)
_INSTALLED = False
_BaseAcquisition = right._Acquisition


class OverlapAcquisition(_BaseAcquisition):
    def __init__(self, profile, command_timing):
        super().__init__(profile,command_timing)
        self.credited_home_s = 0.
        self.overlap_evidence = None

    def plan_at_actual_dock(self, path):
        if self.prefix_timing is not None:
            raise RuntimeError('Overlap remaining prefix was already frozen')
        state = overlap.session()
        state.require_completed()
        if not state.qualification['native_staging_tracking_guard_passed']:
            raise RuntimeError('Actual staging tracking guard is required before replanning')
        profile = json.loads(Path(path).read_text())
        full_duration = float(self.original_timing['seat_start_s'])
        full_reference, full_clock = overlap.full_entry_envelope(profile,state.original_ready,self,
            'overlap/unexecuted_actual_dock_full_entry_envelope')
        self.credited_home_s = float(self.profile['home_s'])
        trimmed = restrict_curve(full_clock,self.credited_home_s,full_duration)
        remaining_clock = BPoly(trimmed.c,trimmed.x-self.credited_home_s,extrapolate=False)
        source_duration = full_duration-self.credited_home_s
        native_tail = float(self.profile['entry_native_tail_s'])
        # Reuse the native knot; subtraction can create a one-ULP polynomial slice.
        self.prefix_timing = core.retime(prefix_path(profile),duration=source_duration,
            label='right/acquisition_remaining_after_overlap',clock=remaining_clock,
            vlim=self.profile['arm_rate_limit_rad_s'],alim=full_reference.report['acceleration_limits'],
            protected=((float(remaining_clock.x[-2]),source_duration),),minimum_duration=native_tail)
        self.prefix = self.prefix_timing.duration
        delta = self.prefix-full_duration
        self.command_timing.update({key:value+delta for key,value in self.original_timing.items()})
        self.close = self.command_timing['finger_close_start_s']
        self.verticalize = self.command_timing['verticalize_start_s']
        if self.physical_timing is None:
            raise RuntimeError('Missing native acquisition physical timing dictionary')
        self.physical_timing.update(self.command_timing)
        task = acquisition_task_path(self.profile)
        duration = float(task.x[-1])
        self.task_timing = core.retime(task,duration=duration,
            label='right/acquisition_coupled_contact_clock',vlim=derivative_enclosure(task),
            protected=((0.,duration),))
        self.overlap_evidence = dict(sidecar='dual_arm_overlap.json',credited_initial_hold_s=self.credited_home_s,
            credited_native_preshape_s=float(self.profile['preshape_duration_s']),
            original_near_seat_tail_s=native_tail,remaining_source_duration_s=source_duration,
            hand_program_physically_completed=True,original_full_entry_limits=full_reference.report['acceleration_limits'])
        profile['duration_s'] = self.prefix
        first = profile['continuous_approach']['arm_q_rad'][0]
        last = profile['continuous_approach']['arm_q_rad'][-1]
        profile['waypoints'] = [dict(name='overlap_staging',time_s=0.,arm_q_rad=first),
                                dict(name='remaining_station_entry',time_s=self.prefix,arm_q_rad=last)]
        profile['staged_grasp_timing'] = dict(profile['staged_grasp_timing'],**self.command_timing)
        profile['hand_profile']['time_basis'] = 'Original hand-program source age; already executed during LEFT overlap, not replayed by this child'
        profile['overlap_completed_hand_program'] = self.overlap_evidence
        profile['topp_right_acquisition'] = self.report()
        Path(path).write_text(json.dumps(profile,indent=2))
        state.event('remaining_prefix_planned',duration_s=self.prefix,
                    original_full_prefix_s=full_duration,credited_initial_hold_s=self.credited_home_s,
                    native_approach_tail_s=native_tail)
        state.save()

    def report(self):
        result = super().report()
        result['preshape_and_mcp_clocks'] = 'Original hand program executed once during LEFT tail; completed opening held until native seating fade'
        result['overlap'] = self.overlap_evidence
        return result


def adapt_left(source):
    if not source.startswith('def advance_reagent_preparation('):
        raise RuntimeError('Expected the original generated LEFT function')
    first,rest=source.split('\n',1)
    source=first+'\n import v34_overlap_control as _overlap\n'+rest
    source=exact(source,'d.ctrl[robotacts]=initial_ctrl[robotacts];d.ctrl[armacts]=armcmd',
                 '_overlap.left_defaults(m,d,robotacts,initial_ctrl);d.ctrl[armacts]=armcmd')
    source=exact(source,'mujoco.mj_step(m,d);obs=ledger.observe',
                 '_overlap.before_step(m,d,phase);mujoco.mj_step(m,d);_overlap.after_step(m,d,phase);obs=ledger.observe')
    source=exact(source,"phase('release_tube',2.,goalhand=openq);phase('withdraw_empty_hand'",
                 "phase('release_tube',2.,goalhand=openq);_overlap.release(m,d,dock,body,tube_initial);phase('withdraw_empty_hand'")
    source=exact(source,"  phase('settle_original_ready',.7)\n",
                 "  phase('settle_original_ready',.7)\n  _overlap.finish_left(m,d)\n")
    return source


def adapt_child_source(source):
    # Edit the actual factory input, never inspect a dynamically compiled main.
    source=exact(source,'def main():\n',
        'def main():\n    from v34_overlap_control import restore_hand_entry as _overlap_restore_hand_entry, completed_nominal_hand as _overlap_completed_nominal_hand\n')
    initial='    for a in range(4,m.nu):d.ctrl[a]=desired[act_q[a]]'
    source=exact(source,'\n'+initial+'\n','\n'+initial+'\n    _overlap_restore_hand_entry(m,d)\n')
    blend='u=smooth((t-hand_start)/(hand_end-hand_start));desired_hand=(1-u)*home_hand+u*openq'
    return exact(source,blend,'desired_hand=_overlap_completed_nominal_hand(m,d,openq)')


def prepare_runner(previous,*args,**kwargs):
    runner,path,callback,audit,clock=previous(*args,**kwargs)
    state=runner._topp_acquisition
    if not isinstance(state,OverlapAcquisition):
        raise RuntimeError('Overlap acquisition must be installed before native preparation')
    observed_cells=closure(callback)
    coherent=observed_cells['coherent']
    native_callback=closure(coherent)['old_callback']
    opening='u=np.clip((t-0.7)/1.1,0,1);u=u*u*u*(10+u*(-15+6*u))'
    native_callback=copy_callable(native_callback,[(opening,'u=_overlap_completed_opening(model,data)')],
        globals_={'_overlap_completed_opening':overlap.completed_opening},label='overlap_native_mcp')
    coherent=copy_callable(coherent,cells={'old_callback':native_callback},label='overlap_coherent_callback')
    callback=copy_callable(callback,cells={'coherent':coherent},label='overlap_observed_callback')
    speed_sampler=closure(runner.continuous_prefix_sample)['original_sample']

    @wraps(speed_sampler)
    def remaining_sampler(profile,seconds):
        if state.prefix_timing is None:
            raise RuntimeError('Remaining prefix sampled before genuine stopped-dock planning')
        return speed_sampler(profile,state.credited_home_s+state.prefix_timing.source_time(float(seconds)))

    runner.continuous_prefix_sample=remaining_sampler

    @wraps(audit)
    def combined_audit():
        report=audit()
        session=overlap.session()
        report['prefix_clear']=bool(report['prefix_clear'] and session.qualification['precontact_passed'])
        report['early_right_precontact']=dict(time_basis='absolute native physical time',
            sidecar='dual_arm_overlap.json',contact_events=session.contact_events,
            original_native_prefix_observer_retained=True)
        return report

    combined_audit.dock=audit.dock
    return runner,path,callback,combined_audit,clock


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    if not right._INSTALLED:
        raise RuntimeError('Dual-arm overlap requires the frozen TOPP chain first')
    import v34_speedrightpickup as pickup
    import t1_v10_v27_stage_v28 as stage
    import t1_right_standing_motion_v6 as motion
    import t1_left_glass_preparation_stage as left
    import t1_continuous_approach as approach
    import run_t1_full_workflow_v28 as workflow
    previous_stage=stage.adapt_left_stage
    previous_engine=pickup.retime_engine_source
    previous_prepared=pickup.retime_prepared
    previous_reanchor=motion.reanchor_prefix_at_actual_dock_v6
    previous_left=left.advance_reagent_preparation
    previous_main=workflow.main
    factory_anchor="    module=types.ModuleType('front_workflow_'+kind)"
    # Native downstream adapters require inspectable top-level factory source.
    factory_source=textwrap.dedent(inspect.getsource(approach.adapted_runner))
    factory_source=exact(factory_source,factory_anchor,
        "    if kind=='grasp':\n        from v34_dualarmoverlap import adapt_child_source\n        source=adapt_child_source(source)\n"+factory_anchor)
    factory_filename=__file__+' [overlap native factory]'
    linecache.cache[factory_filename]=(len(factory_source),None,factory_source.splitlines(True),factory_filename)
    exec(compile(factory_source,factory_filename,'exec'),approach.__dict__)

    @wraps(previous_stage)
    def left_source(source):
        return adapt_left(previous_stage(source))

    @wraps(previous_engine)
    def engine_source(source):
        source=previous_engine(source)
        anchor='model_identity=id(m);data_identity=id(d);handoffs=[]'
        return exact(source,anchor,anchor+'\n    from v34_overlap_control import bind as _bind_overlap\n    _bind_overlap(m,d,grasp,mobile_profile,out)')

    @wraps(previous_prepared)
    def prepared(*args,**kwargs):
        return prepare_runner(previous_prepared,*args,**kwargs)

    @wraps(previous_reanchor)
    def reanchor(m,d,module,path):
        overlap.session(m,d).prepare_remaining(path)
        return previous_reanchor(m,d,module,path)

    dock=copy_callable(motion.advance_standing_dock_v6,[
        ('    for k in range(int(maximum_s/m.opt.timestep)):',
         "    _observe_overlap_dock=Path(out).name=='v6_right_grasp_dock'\n    for k in range(int(maximum_s/m.opt.timestep)):") ,
        ('mujoco.mj_step(m,d)\n',
         'mujoco.mj_step(m,d)\n        if _observe_overlap_dock:_overlap_observe_dock(m,d)\n'),
    ],globals_={'_overlap_observe_dock':overlap.observe_dock},label='overlap_first_dock_observer')

    @wraps(previous_left)
    def left_advance(m,d,*args,**kwargs):
        from t1_spincoat_scene import arm_indices,hand_indices
        _,aq,_=arm_indices(m,'right');_,hq,_=hand_indices(m,'right')
        indices=np.r_[aq,hq]
        ready=d.qpos[indices].copy()
        arrays,report,ledger=previous_left(m,d,*args,**kwargs)
        session=overlap.session(m,d)
        report.pop('right_ready_passive_joint_deviation_max_rad',None)
        before=(arrays['time']<=session.start_tick*session.dt) if session.start_tick is not None else np.ones(len(arrays['time']),bool)
        report['right_pre_release_ready_deviation_max_rad']=float(np.max(np.abs(arrays['qpos'][before][:,indices]-ready)))
        report['right_arm_hand_targets']='Original ready until actual release; original preshape and bounded arm preposition then share the LEFT physical ticks'
        report['new_left_stage_dynamic_scope']='Complete original LEFT preparation and acceptance retained; secondary RIGHT preposition explicitly recorded before separate native RIGHT dock/grasp'
        report['dual_arm_overlap']='../dual_arm_overlap.json'
        out=kwargs.get('out',args[2] if len(args)>2 else None)
        if out is None:
            raise RuntimeError('Native LEFT output directory missing')
        (Path(out)/'reagent_preparation_report.json').write_text(json.dumps(report,indent=2))
        return arrays,report,ledger

    @wraps(previous_main)
    def main(*args,**kwargs):
        try:
            return previous_main(*args,**kwargs)
        finally:
            overlap.save_if_bound()

    right._Acquisition=OverlapAcquisition
    stage.adapt_left_stage=left_source
    pickup.retime_engine_source=engine_source
    pickup.retime_prepared=prepared
    motion.reanchor_prefix_at_actual_dock_v6=reanchor
    motion.advance_standing_dock_v6=dock
    left.advance_reagent_preparation=left_advance
    workflow.main=main
    _INSTALLED=True
