"""Whole LEFT R08 reference retiming, installed after all ten native plugins.

Clock-only cutover: actual physics, callback step ownership, live replans,
nonlinear R03 closure, mixing, fluid termination and acceptance stay native.
Strict source anchors deliberately fail on a different generated workflow.
"""
from functools import wraps

import numpy as np

import v34_topp_core as core
import v34_topp_left_paths as paths

_INSTALLED = False
PATCHED_SYMBOLS = (
    't1_v10_v27_stage_v28.adapt_left_stage',
    't1_torso_crouch_v4.TorsoCrouchOverlay.__init__',
    't1_torso_crouch_v4.TorsoCrouchOverlay.__call__',
    't1_human_pour_profile_v2.IdleFingerOverlay.__init__',
    't1_human_pour_profile_v2.IdleFingerOverlay.__call__',
    't1_left_standing_workflow_v6.make_standing_transport',
    't1_left_opposite_workflow_v7.make_standing_transport',
    't1_human_pour_v4_joint_reference.plan_segment',
    'v34_speedpour._clock',
    'v34_speedpour._retime',
    'v34_smoothtubereturn_reference.Reference.__init__',
    'v34_smoothtubereturn_reference._command_guards',
    'v34_smoothtubereturn_reference.plan_empty_ready',
    'v34_smoothtubereturn_reference.advance_empty',
    't1_actual_frame_return_v7.plan_actual_root_descent_free_yaw',
)

COVERAGE = [
    dict(stage='coarse_dock_before_prep', decision='native_feedback', reason='No geometric reference; original wheel controller, braking latch,0.4s brake and0.5s stopped gate untouched.'),
    dict(stage='standing_lean_before_pickup', decision='retime', reason='Exact four-joint quintic; native torso overlay and generated caller share one TOPP clock. Subsequent0.5s measured stability remains physical time.'),
    dict(stage='approach_outside_tube', decision='retime', reason='Original analytic arm chord and optional preshape share original source progress; actual tube6D reanchor remains a hard boundary.'),
    dict(stage='close_on_tube', decision='protected_reference', reason='Native Cartesian spline enters TOPP with the full4.5s interval protected. Original R03 nonlinear22-joint closing_target is still called at unchanged source progress; contact-sensitive qualification is not compressed.'),
    dict(stage='hold_before_tube_lift', decision='native_feedback', reason='Original1s acquisition, optional10Hz actual-tube tracking, environment gate and subsequent encoder freeze unchanged.'),
    dict(stage='lift_tube', decision='retime', reason='Original constructed Cartesian command CubicSpline, exact servo offset and IK knot spacing; no earlier freeze, reanchor or required actual lift changed.'),
    dict(stage='stabilize_lifted_tube / wait_for_original_held_readiness', decision='native_feedback', reason='Original stabilization, bounded extra wait, last500 native1ms contact samples, force/relative-pose/lift thresholds preserved.'),
    dict(stage='pre_mix_clearance_*', decision='retime_with_protected_window', reason='Native Cartesian spline with first0.6s source interval protected for original contact-qualified preload ramp; explicit original duration overrides remain minimum durations.'),
    dict(stage='gentle_shake_mouth_pivot / V27 oscillation', decision='physical_exposure', reason='Selected3Hz,+/-5degree,1s+2s+1s exposure and derivative-dependent prefilter/increment limiter unchanged, as is the mouth50mm gate.'),
    dict(stage='human_pour_idle_extend / human_pour_idle_fold', decision='retime', reason='Native ring/pinky goal selection and ordering retained; exact quintic source clock is shared by overlay and caller, while primary-finger hold/integral clocks remain physical.'),
    dict(stage='postmix_live_retarget', decision='native_reanchor', reason='R03 receiver/corridor and V31 fresh actual grasp retarget run unchanged in this episode.'),
    dict(stage='coarse_carry_to_receiver', decision='retime', reason='Actual native joint chord, not supplied Cartesian waypoint list; immediate predecessor factory/closure and original speed/range/wrist/upright/collision guards retained.'),
    dict(stage='settle_before_precise_pour', decision='native_feedback', reason='Original stopped-base predicate and4000-step failure bound; not a nominal4s delay.'),
    dict(stage='human_pour_approach', decision='conditional_retime', reason='Exact native quintic joint chord after predecessor planning; original1e-6 satisfied-target skip remains in native executor.'),
    dict(stage='human_pour_tilt', decision='retime_with_protected_window', reason='Native SpeedPour spatial BPoly and predecessor clock; active-flow interval after first planned55degree crossing is protected,0.55rad/s design cap retained, angle_deg follows the new source coordinate.'),
    dict(stage='human_pour_drain_wait', decision='physical_exposure', reason='Original ledger.stop_requested after actual native steps and unchanged2s drain timeout; no completion from planned duration.'),
    dict(stage='human_pour_return', decision='retime_with_protected_window', reason='Actual-return/offset-aware/SpeedPour chain retained; first and last0.5 native source seconds protected. Event-triggered measured-state reanchor is not blended away.'),
    dict(stage='human_pour_retreat', decision='physical_exposure', reason='Original constant-command0.1s ledger interval and subsequent99percent delivery gate unchanged.'),
    dict(stage='loaded_clearance / coarse_return / rack_side_entry', decision='retime_with_protected_window', reason='Same SmoothTubeReturn spatial polynomial; loaded clearance and entire>=2s side-entry protected, ordinary free corridor retimed. Recomputed joins, C2 endpoint/positive-join guards, traces, Bernstein bounds, dense FK and continuous sweep use new clock.'),
    dict(stage='return_above_holder', decision='conditional_retime', reason='Original actual<=0.5mm/0.2degree skip decision preserved; otherwise retime the existing Cartesian spline, not replanned IK.'),
    dict(stage='lower_into_holder', decision='protected_reference', reason='Actual-root free-yaw native spline enters TOPP and caller reads returned duration; full original2.5s contact stroke conservatively pinned, not claimed a universal physical minimum.'),
    dict(stage='release_tube', decision='protected_reference', reason='Original2s hand unloading, holder release and physical encoder decay retained; release precedes withdrawal.'),
    dict(stage='withdraw_empty_hand', decision='retime', reason='Original2s analytic joint chord to150mm-above-bottom target, distinct from subsequent empty-clearance reference.'),
    dict(stage='human_pour_idle_restore_fold / human_pour_idle_restore_extend', decision='retime', reason='Native opposite-order finger restoration remains after withdrawal, sharing overlay/caller clock.'),
    dict(stage='empty_hand_clearance_before_ready / return_to_original_ready_joints', decision='retime', reason='Same broad SmoothTubeReturn spatial curve and exact final targets; original safe_ready_hand_target source progress and offset decay are preserved together. Native collision screening and spatial all-time corridor certificates remain authoritative; no claim of a new empty-hand CCD certificate.'),
    dict(stage='settle_original_ready / LEFT acceptance', decision='physical_exposure', reason='Original0.25s settle and environment/liquid/wheel/tube-return gates unchanged; native sample owns one1ms physics step and real elapsed-time observations throughout.'),
]


_CARTESIAN_CALL = "   transport=plan_palm_cartesian_move(m,d,armcmd,target_p,target_R,duration,ik_solver=(solve_arm if (plan.get('v7_cartesian_pregrip',False) and name in ['seat_from_side','seat_open_realignment','lift_tube','close_on_tube']) else ((V3_SAFE_EXIT_SOLVER if name.startswith('pre_mix_clearance_') else continuous_left_solve_arm))))"
_PHASE_LOOP = "  for k in range(int(round(duration/m.opt.timestep))):\n   u=smooth((k+1)*m.opt.timestep/duration)"
_RETAINED = (
    "   sample(name,pour,mix)",
    "     from t1_single_grasp_v10 import closing_target;handcmd=closing_target(h0,closed,u)",
    "  if not held_gate['pass_']:raise RuntimeError('actual tube held/lift gate failed')",
    "  if not dock.stopped:raise RuntimeError('Actual base did not settle before precision pour')",
    "  if ledger.delivered_m3<max(c.target_pour_volume_m3,c.tube_initial_volume_m3)*.99:raise RuntimeError('Gravity pour delivered less than original99percent target; no flow valve/cap')",
    "  if rack_above_position_error<=.0005 and rack_above_orientation_error<=.2:",
    "   descent=plan_actual_root_descent_free_yaw(m,d,handles,armcmd,tube_initial+np.array([0,0,.001]),duration_s=2.5,ik_solver=continuous_left_solve_arm)",
    "  phase('release_tube',2.,goalhand=openq);phase('withdraw_empty_hand',2.,goal(tube_initial+[0,0,.15]-returnR@root_local,returnR));phase('human_pour_idle_restore_fold',float(V6_RUNTIME_PROFILE.get('idle_segment_s',1.2)));phase('human_pour_idle_restore_extend',float(V6_RUNTIME_PROFILE.get('idle_segment_s',1.2)))",
    "  phase('settle_original_ready',.7)",
    "d.qfrc_applied[robotdofs]=d.qfrc_bias[robotdofs];mujoco.mj_step(m,d);obs=ledger.observe(m,d,handles,allow_pour=pour,allow_mix=mix)",
)


def adapt_stage(source):
    for anchor in _RETAINED:
        if source.count(anchor) != 1:
            raise RuntimeError('TOPP LEFT retained native gate/callback anchor mismatch: ' + repr(anchor[:140]))
    replacements = [
        (' import copy', ' import copy\n import v34_topp_left_paths as _topp_left\n _topp_observations=_topp_left.LeftObservations()', 1),
        ('  return counts,forces', '  _topp_observations.observe(phase,m.opt.timestep,mix=mix)\n  return counts,forces', 1),
        ('  transport=None', '  transport=None\n  _topp_source_duration=float(duration);_topp_timing=None', 1),
        (_CARTESIAN_CALL, _CARTESIAN_CALL + "\n   transport=_topp_left.retime_cartesian(m,transport,name,duration,plan)\n   _topp_timing=transport['_topp_timing'];duration=_topp_timing.duration", 1),
        (_PHASE_LOOP,
         "  if transport is None:\n   _topp_timing=_topp_left.phase_timing(m,name,_topp_source_duration,q0,goalarm,h0,goalhand,plan,hand_ids=hj,safe_ready=HUMAN_POUR_PROFILE.get('v7_empty_ready_clearance',False))\n   duration=_topp_timing.duration if _topp_timing is not None else _topp_left.overlay_duration(m,d,name,duration,CONTROL_HOOK,body_config=V4_BODY_CONFIG,hand_command=h0,hand_actuators=hacts)\n  _topp_left.native_tick_count(m,duration)\n  for k in range(int(round(duration/m.opt.timestep))):\n   _topp_source_elapsed=_topp_left.source_elapsed(_topp_timing,(k+1)*m.opt.timestep)\n   u=smooth(_topp_source_elapsed/_topp_source_duration)", 1),
        ('handcmd=safe_ready_hand_target(m,hj,h0,goalhand,(k+1)*m.opt.timestep/duration)',
         'handcmd=safe_ready_hand_target(m,hj,h0,goalhand,_topp_source_elapsed/_topp_source_duration)', 1),
        ("    for idle_step in range(round(float(V6_RUNTIME_PROFILE.get('idle_segment_s',1.2))/m.opt.timestep)):coarse_transport_step(armcmd,idle_phase)",
         "    phase(idle_phase,float(V6_RUNTIME_PROFILE.get('idle_segment_s',1.2)))", 1),
        ('   for descent_step in range(round(2.5/m.opt.timestep)):',
         "   _topp_left.native_tick_count(m,descent['report']['duration_s'])\n   for descent_step in range(round(descent['report']['duration_s']/m.opt.timestep)):", 1),
        (" rec.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),'prep_end'",
         " _topp_observations.finish()\n rec.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),'prep_end'", 1),
    ]
    for old, new, count in replacements:
        source = paths.once(source, old, new, 'generated final preparation', count)
    for anchor in _RETAINED:
        if source.count(anchor) != 1:
            raise RuntimeError('TOPP LEFT modified a retained gate/callback: ' + repr(anchor[:140]))
    return source


def _install_overlays():
    import t1_torso_crouch_v4 as torso
    # The V4 motion stack still constructs its actual idle controller in install_v2.
    import t1_human_pour_profile_v2 as pour
    previous_torso_init = torso.TorsoCrouchOverlay.__init__
    previous_idle_init = pour.IdleFingerOverlay.__init__

    @wraps(previous_torso_init)
    def torso_init(self, m, config=None):
        previous_torso_init(self, m, config)
        paths.register_overlay('torso', self, m)

    @wraps(previous_idle_init)
    def idle_init(self, m, idle_segment_s=1.2):
        previous_idle_init(self, m, idle_segment_s)
        paths.register_overlay('idle', self, m)

    torso_call = paths.adapt_callable(torso.TorsoCrouchOverlay.__call__, [
        ('u=smooth((float(d.time)-self.start+m.opt.timestep)/self.config.duration_s)',
         "u=smooth(_topp_overlay_progress(self,m,d,phase,float(d.time)-self.start+m.opt.timestep,self.config.duration_s,'torso'))", 2),
    ], 'TorsoCrouchOverlay.__call__', {'_topp_overlay_progress': paths.overlay_progress})
    idle_call = paths.adapt_callable(pour.IdleFingerOverlay.__call__, [
        ("u=smoothstep5((float(d.time)-self.t0+m.opt.timestep)/self.duration) if phase.startswith('human_pour_idle_') else 1.",
         "u=smoothstep5(_topp_overlay_progress(self,m,d,phase,float(d.time)-self.t0+m.opt.timestep,self.duration,'idle')) if phase.startswith('human_pour_idle_') else 1.", 1),
    ], 'IdleFingerOverlay.__call__', {'_topp_overlay_progress': paths.overlay_progress})
    torso.TorsoCrouchOverlay.__init__ = torso_init
    torso.TorsoCrouchOverlay.__call__ = torso_call
    pour.IdleFingerOverlay.__init__ = idle_init
    pour.IdleFingerOverlay.__call__ = idle_call


def _install_carry():
    import t1_left_standing_workflow_v6 as standing
    import t1_left_opposite_workflow_v7 as opposite
    previous_factory = standing.make_standing_transport
    if opposite.make_standing_transport is not previous_factory:
        raise RuntimeError('TOPP LEFT carry aliases no longer share the installed immediate predecessor')

    @wraps(previous_factory)
    def make_standing_transport(profile, original):
        previous = previous_factory(profile, original)
        return paths.adapt_callable(previous, [
            ('ts=np.arange(round(duration/m.opt.timestep)+1)*m.opt.timestep;u=ts/duration;s=u**3*(10+u*(-15+6*u));commands=start[None,:]+s[:,None]*(end-start)[None,:]',
             '_topp,ts,commands,_topp_peak=_topp_carry_reference(m,start,end,duration);duration=_topp.duration', 1),
            ('speed=float(1.875*np.max(abs(end-start))/duration)', 'speed=_topp_peak', 1),
            ('maximum_command_speed_rad_s=speed,', 'maximum_command_speed_rad_s=speed,topp=_topp.report,source_duration_s=_topp.source_duration,', 1),
            ('dict(time=ts,arm_commands=commands,geometric_arm_q=commands,',
             'dict(time=ts,arm_commands=commands,geometric_arm_q=commands,source_elapsed_s=_topp.source_time(ts),reference_velocity_rad_s=_topp(ts,1),reference_acceleration_rad_s2=_topp(ts,2),', 1),
        ], 'native standing carry closure', {'_topp_carry_reference': paths.carry_reference})

    standing.make_standing_transport = make_standing_transport
    opposite.make_standing_transport = make_standing_transport


def _install_pour():
    import v34_speedpour as speed
    import t1_human_pour_v4_joint_reference as joint
    previous_clock = speed._clock
    previous_retime = speed._retime
    previous_segment = joint.plan_segment
    if not getattr(speed, '_INSTALLED', False):
        raise RuntimeError('TOPP LEFT requires the original SpeedPour plugin first')

    @wraps(previous_clock)
    def clock(spatial, source_time, source_rates, fixed_source_intervals, peak_design, dt):
        return paths.speedpour_clock(previous_clock, spatial, source_time, source_rates,
                                     fixed_source_intervals, peak_design, dt)

    @wraps(previous_retime)
    def retime(m, d, plan, entry):
        context = {'segment': plan['segment']}
        token = paths._POUR_CONTEXT.set(context)
        try:
            result = previous_retime(m, d, plan, entry)
        finally:
            paths._POUR_CONTEXT.reset(token)
        if 'timing' not in context:
            raise RuntimeError('TOPP LEFT native SpeedPour did not reach the exact spatial/clock seam')
        result['topp'] = context['timing'].report
        result['speedpour_retiming']['source_r08_clock'] = context['source_clock']
        result['time_law'] = 'unchanged_native_dense_C2_spatial_reference_with_TOPP_RA_certified_C2_clock'
        return result

    @wraps(previous_segment)
    def plan_segment(m, d, profile, *, segment, entry_arm_command, duration_s):
        result = previous_segment(m, d, profile, segment=segment,
                                  entry_arm_command=entry_arm_command, duration_s=duration_s)
        if segment == 'approach' and profile.get('geometric_path_kind') == 'joint_reference_curve':
            start = np.asarray(entry_arm_command)
            end = np.asarray(profile['joint_reference_keyframes_rad'][0][1:])
            original = core.quintic(start, end, duration_s)
            if np.max(np.abs(original(result['time']) - result['arm_commands'])) > 1e-9:
                raise RuntimeError('TOPP LEFT native pour approach no longer matches its exact analytic chord')
            timing = paths._retime(original, duration_s, 'left.human_pour_approach', 2.0)
            times = paths.native_times(m, timing.duration, include_start=False)
            from t1_spincoat_scene import arm_indices
            margin, control_margin = paths._command_bounds(m, original, arm_indices(m, 'left')[0], .05 - 1e-6, 'pour approach')
            result = dict(result)
            result.update(time=times, arm_commands=timing(times), duration_s=timing.duration,
                          peak_joint_speed_rad_s=float(np.abs(timing(times, 1)).max()),
                          min_joint_margin_rad=margin, actuator_control_hull_margin_rad=control_margin,
                          time_law='native_analytic_quintic_chord_with_TOPP_RA_C2_clock', topp=timing.report)
        return result

    speed._clock = clock
    speed._retime = retime
    joint.plan_segment = plan_segment


def _install_smooth_return():
    import v34_smoothtubereturn_reference as smooth
    previous_init = smooth.Reference.__init__
    previous_guards = smooth._command_guards

    @wraps(previous_init)
    def reference_init(self, spatial, length, joins, durations, timestep, *, join_rate_limits=None, fixed_durations=None):
        previous_init(self, spatial, length, joins, durations, timestep,
                      join_rate_limits=join_rate_limits, fixed_durations=fixed_durations)
        paths.smooth_reference(self, timestep, loaded=join_rate_limits is not None)

    @wraps(previous_guards)
    def command_guards(m, start, reference):
        report, trace = previous_guards(m, start, reference)
        report.update(topp=reference.topp_timing.report,
                      source_phase_durations_s=reference.topp_source_durations.tolist(),
                      source_join_times_s=reference.topp_source_join_times.tolist(),
                      source_fixed_phase_durations=reference.topp_source_fixed_durations.tolist(),
                      native_guards_recomputed_on_new_clock=True,
                      original_spatial_polynomial_preserved=True)
        trace['source_elapsed_s'] = reference.topp_timing.source_time(trace['time'])
        trace['reference_spatial_coordinate'] = reference.clock(trace['time'])
        return report, trace

    smooth.Reference.__init__ = reference_init
    smooth._command_guards = command_guards
    smooth.plan_empty_ready = paths.adapt_callable(smooth.plan_empty_ready, [
        ('hand_progress = 0.0 if time <= reference.join_times[0] else (time - reference.join_times[0]) / reference.durations[1]',
         'hand_progress = _topp_ready_hand_progress(reference, time)', 1),
        ("original_ready_hand_duration_s=3.0, executed_ready_hand_duration_s=float(reference.durations[1]),",
         "original_ready_hand_duration_s=3.0, source_ready_hand_duration_s=float(reference.topp_source_durations[1]), executed_ready_hand_duration_s=float(reference.durations[1]), safe_hand_progress_clock='original_reference_elapsed_time_under_same_TOPP_clock',", 1),
    ], 'SmoothTubeReturn.plan_empty_ready', {'_topp_ready_hand_progress': paths.ready_hand_progress})
    smooth.advance_empty = paths.adapt_callable(smooth.advance_empty, [
        ('progress = (time - reference.join_times[0]) / reference.durations[1]',
         'progress = _topp_ready_hand_progress(reference, time)', 1),
    ], 'SmoothTubeReturn.advance_empty', {'_topp_ready_hand_progress': paths.ready_hand_progress})


def _install_insertion():
    import t1_actual_frame_return_v7 as actual
    previous = actual.plan_actual_root_descent_free_yaw

    @wraps(previous)
    def plan_actual_root_descent_free_yaw(m, d, handles, arm_command, target_bottom_world_m, duration_s=2.5, ik_solver=None):
        result = previous(m, d, handles, arm_command, target_bottom_world_m,
                          duration_s=duration_s, ik_solver=ik_solver)
        return paths.retime_insertion(m, result, duration_s)

    actual.plan_actual_root_descent_free_yaw = plan_actual_root_descent_free_yaw


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import t1_v10_v27_stage_v28 as stage
    import v34_smoothtubereturn as smooth_plugin
    if not getattr(smooth_plugin, '_INSTALLED', False):
        raise RuntimeError('TOPP LEFT requires the original SmoothTubeReturn plugin first')
    previous_stage = stage.adapt_left_stage

    @wraps(previous_stage)
    def adapt_left_stage(source):
        return adapt_stage(previous_stage(source))

    _install_overlays()
    _install_carry()
    _install_pour()
    _install_smooth_return()
    _install_insertion()
    stage.adapt_left_stage = adapt_left_stage
    _INSTALLED = True
