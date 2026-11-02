"""Install phase-local original-speed windows without changing physical gates."""
from __future__ import annotations

import importlib.util
import types
import numpy as np

import v34_uncapped_clock as clock
from v34_local_speed_policy import (
    POLICY_ID, SOURCE_FRACTION_WINDOWS, TRANSITION_S, ORIGINAL_SPEEDPOUR_SEGMENTS,
    FAST_REFERENCE_FACTOR, REFERENCE_FACTOR_OVERRIDES,
    accelerated_reference, reference_factor,
)
from v34_window_clock import retime_local

PATCHED_SYMBOLS = (
    'v34_uncapped_clock.retime',
    'v34_topp_core.retime',
    'v34_uncapped_clock.save_report',
    'v34_topp_core.save_report',
    'v34_speedpour._retime',
    'v34_topp_left_paths.speedpour_clock',
    'v34_smoothtubereturn_reference.Reference.__init__',
    'v34_smoothtubereturn_reference.plan_loaded_return',
    'v34_overlap_control.OverlapSession.release',
    'v34_topp_right._cup_timing',
    'v34_topp_right._descent_progress',
    't1_workflow_ik.install_validated_ik',
    't1_right_standing_motion_v6.advance_direct_cup_v6',
    't1_right_interior_workstation_v6.make_validated_aspiration_ik_v6',
    't1_right_interior_workstation_v6.adapt_vertical_aspiration_source',
    't1_right_direct_coating_v6.adapt_direct_coating_source',
    'v34_uncapped_right._dispense_source',
    'v34_dualarmoverlap.OverlapAcquisition.plan_at_actual_dock',
    'v34_topp_right._Acquisition.task_clock',
    'v34_uncapped_right._grasp_source',
    'v34_uncapped_right._VelocityReference.__init__',
    'v34_uncapped_right._VelocityReference.sample',
    'v34_uncapped_right._retimed_transfer',
    'v34_topp_right.retimed_transfer',
    'v34_overlap_control.OverlapSession.before_step',
    'v34_overlap_control.OverlapSession.after_step',
    'v34_overlap_control.OverlapSession.save',
    'v34_dualarmoverlap.OverlapAcquisition.report',
    'v34_speedrightpickup.retime_prepared',
    'v34_speedrightpickup.retime_engine_source',
    'v34_topp_right._installed_precision_type',
    'v34_topp_right._installed_precision_type.__init__',
    'v34_topp_right._installed_precision_type.target',
    'v34_topp_right.COVERAGE',
    't1_actual_frame_return_v7.advance_human_pour_reanchored_return_v7',
    't1_human_pour_v4_joint_reference.plan_segment',
    't1_actual_frame_return_v7.plan_return',
    't1_v10_v27_stage_v28.adapt_left_stage',
)
_INSTALLED = False
_ORIGINAL_RETIME = None
_LOADED_SOURCE = None


def _localized_dispatch(path, *, duration, label, **kwargs):
    fractions = SOURCE_FRACTION_WINDOWS.get(str(label), ())
    windows = [(float(start)*duration, float(end)*duration) for start, end in fractions]
    kwargs['factor'] = reference_factor(label)
    timing = retime_local(_ORIGINAL_RETIME, path, duration=duration, label=label,
                          windows=windows, transition_s=TRANSITION_S, **kwargs)
    timing.report.update(local_speed_policy_id=POLICY_ID,
                         source_fraction_windows=[list(pair) for pair in fractions])
    clock.record('localized_reference_timing', str(label), policy_id=POLICY_ID,
                 source_fraction_windows=fractions, timing=timing.report,
                 physical_execution_passed=None)
    return timing


def _source_rate_program(native, source_time, fast, peak_design, segment):
    if segment not in ORIGINAL_SPEEDPOUR_SEGMENTS:
        return fast
    source_speed = np.max(np.abs(native(source_time, 1)), axis=1)
    return np.maximum(1.0, np.minimum(fast, peak_design / np.maximum(source_speed, 1e-12)))


def _speedpour_source_clock(previous, spatial, source_time, source_rates,
                            fixed_source_intervals, peak_design, dt):
    import v34_topp_left_paths as paths
    import v34_uncapped_left as uncapped_left
    context = paths._POUR_CONTEXT.get()
    if context is not None and context['segment'] in ORIGINAL_SPEEDPOUR_SEGMENTS:
        result = previous(spatial, source_time, source_rates,
                          fixed_source_intervals, peak_design, dt)
        clock.record('nominal_clock_restore', 'left.human_pour_'+context['segment'],
                     source='Preserved original SpeedPour source-clock callable',
                     source_duration_s=float(result[0].x[-1]),
                     fresh_spatial_reference=True, recorded_trajectory_replayed=False,
                     final_command_clipping_added=False, physical_success_inferred=False)
        return result
    return uncapped_left._speedpour_source_clock(
        spatial, source_time, source_rates, fixed_source_intervals, peak_design, dt)


def _loaded_source_init(reference, spatial, length, joins, durations, timestep,
                        *, join_rate_limits, fixed_durations):
    _LOADED_SOURCE.Reference.__init__(
        reference, spatial, length, joins, durations, timestep,
        join_rate_limits=join_rate_limits, fixed_durations=fixed_durations)
    clock.record('nominal_clock_restore', 'left.smooth_loaded_return',
                 source='Frozen original SmoothTubeReturn timing constructor',
                 source_duration_s=reference.duration,
                 source_phase_durations_s=reference.durations.tolist(),
                 fresh_spatial_reference=True, recorded_trajectory_replayed=False,
                 final_command_clipping_added=False, physical_success_inferred=False)


def _command_continuous_hand_seed(session):
    command = session.data.ctrl[session.hand_acts].copy()
    clock.record('reference_handoff', 'overlap/right_hand',
                 source='Incoming actuator command, not measured joint position',
                 incoming_command_rad=command.tolist(),
                 measured_joint_position_rad=session.data.qpos[session.hand_q].tolist(),
                 physical_state_written=False, command_clipping=False,
                 original_bounds_checks_unchanged=True)
    return command


def _synchronize_acquisition_clock(acquisition):
    timing = acquisition.task_timing
    source_prefix = float(acquisition.original_timing['seat_start_s'])
    acquisition.command_timing.update({
        key: acquisition.prefix + timing.new_time_for_source(float(value)-source_prefix)
        for key, value in acquisition.original_timing.items()
    })
    acquisition.close = acquisition.command_timing['finger_close_start_s']
    acquisition.verticalize = acquisition.command_timing['verticalize_start_s']
    acquisition.physical_timing.update(acquisition.command_timing)
    acquisition._localized_start_rate = float(timing(0., 1)[0])
    acquisition._localized_end_rate = float(
        timing.report.get('terminal_left_velocity', [0.8*timing.factor])[0])
    clock.record('localized_acquisition_schedule', 'right/acquisition_coupled_contact_clock',
                 source_timing=dict(acquisition.original_timing),
                 command_timing=dict(acquisition.command_timing),
                 terminal_logical_rate=acquisition._localized_end_rate,
                 source_prefix_s=source_prefix, executed_prefix_s=acquisition.prefix,
                 physical_timestep_unchanged=True, acceptance_unchanged=True)


def install():
    global _INSTALLED, _ORIGINAL_RETIME, _LOADED_SOURCE
    if _INSTALLED:
        raise RuntimeError('LocalizedSpeed must be installed once per frozen process')
    import v34_topp_core as core
    import v34_nativewristtiming as wrist
    import v34_speedpour as speed
    import v34_topp_left_paths as paths
    import v34_smoothtubereturn_reference as smooth
    import v34_overlap_control as overlap
    import v34_topp_right as right
    import v34_dualarmoverlap as dual
    import t1_workflow_ik as workflow_ik
    import t1_right_standing_motion_v6 as motion
    import t1_right_interior_workstation_v6 as interior
    import t1_right_direct_coating_v6 as coating
    import v34_uncapped_right as uncapped_right
    from v34_topp_right_paths import closure
    if not getattr(wrist, '_INSTALLED', False):
        raise RuntimeError('LocalizedSpeed requires the frozen native-wrist parent')
    if core.retime is not clock.retime or core.save_report is not clock.save_report:
        raise RuntimeError('Unexpected final reference-clock aliases')
    _ORIGINAL_RETIME = types.FunctionType(
        clock.retime.__code__, clock.retime.__globals__, clock.retime.__name__,
        clock.retime.__defaults__, clock.retime.__closure__)
    _ORIGINAL_RETIME.__kwdefaults__ = dict(clock.retime.__kwdefaults__ or {})
    clock.patch(speed._retime.__wrapped__, [(
        '    rates = fast\n',
        '    rates = _localized_source_rates(native, source_time, fast, peak_design, segment)\n', 1,
    )], 'selected original SpeedPour nominal rate program',
        helpers={'_localized_source_rates': _source_rate_program})
    clock.patch(paths.speedpour_clock, [(
        '    native = _uncapped_speedpour_source(spatial, source_time, source_rates, fixed_source_intervals, peak_design, dt)',
        '    native = _localized_speedpour_source(previous, spatial, source_time, source_rates, fixed_source_intervals, peak_design, dt)', 1,
    )], 'selected original SpeedPour source-clock construction',
        helpers={'_localized_speedpour_source': _speedpour_source_clock})
    source_spec = importlib.util.spec_from_file_location(
        '_localized_original_smooth_return', smooth.__file__)
    _LOADED_SOURCE = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(_LOADED_SOURCE)
    clock.patch(smooth.Reference.__init__.__wrapped__, [(
        '    self.spatial = as_bernstein(spatial)\n',
        '    if join_rate_limits is not None:\n'
        '        return _localized_loaded_source_init(\n'
        '            self, spatial, length, joins, durations, timestep,\n'
        '            join_rate_limits=join_rate_limits, fixed_durations=fixed_durations)\n'
        '    self.spatial = as_bernstein(spatial)\n', 1,
    )], 'original loaded-return nominal clock only',
        helpers={'_localized_loaded_source_init': _loaded_source_init})
    clock.patch(smooth.plan_loaded_return, [(
        'coarse_duration = float(CoarseFineConfig().minimum_duration_s)',
        'coarse_duration = transport_duration(remaining, np.linalg.norm(rotation_vector), config=CoarseFineConfig())', 1,
    ), ('(minimum_clearance_duration_s if minimum_clearance_duration_s > 0.0 else .5)',
        'max(.5, clear_lift / .075, minimum_clearance_duration_s)', 1)],
        'original loaded-return source durations on fresh geometry')
    clock.patch(overlap.OverlapSession.release, [(
        'self.home_hand = self.data.qpos[self.hand_q].copy()',
        'self.home_hand = _localized_hand_seed(self)', 1,
    )], 'command-continuous right handoff without clipping or relaxed bounds',
        helpers={'_localized_hand_seed': _command_continuous_hand_seed})
    clock.patch(dual.OverlapAcquisition.plan_at_actual_dock, [(
        "    self.overlap_evidence = dict(sidecar='dual_arm_overlap.json'",
        "    _localized_acquisition_schedule(self)\n"
        "    self.overlap_evidence = dict(sidecar='dual_arm_overlap.json'", 1,
    ), ('hand_program_physically_completed=True,reference_factor=_uncapped_factor,',
        'hand_program_physically_completed=True,base_reference_factor=_uncapped_factor,', 1)],
        'synchronized contact landmarks from the final local source clock',
        helpers={'_localized_acquisition_schedule': _synchronize_acquisition_clock})
    clock.patch(right._Acquisition.task_clock, [(
        'return .8 * _uncapped_factor * elapsed',
        'return self._localized_start_rate * elapsed', 1,
    ), ('+ .8 * _uncapped_factor * (elapsed - self.task_timing.duration)',
        '+ self._localized_end_rate * (elapsed - self.task_timing.duration)', 1)],
        'contact-clock continuation uses the actual endpoint source rate')
    clock.patch(right._cup_timing, [(
        'Source polynomial only; executed reference is uncapped factor two, not velocity/torque certified',
        'Source polynomial only; executed reference follows recorded local timing, not velocity/torque certified', 1,
    )], 'accurate dry-cup local timing evidence')
    clock.patch(right._descent_progress, [(
        'Original descent geometry composed once at factor two; command slew/rate caps removed, measured liquid/geometry gates retained',
        'Original descent geometry follows recorded local timing; measured liquid/geometry gates retained', 1,
    )], 'accurate pressed-descent local timing evidence')
    clock.patch(workflow_ik.install_validated_ik, [(
        'body=_uncapped_nullspace_source(body)',
        "if not getattr(runner, 'workflow_original_nullspace_timing', False):\n"
        '        body=_uncapped_nullspace_source(body)', 1,
    )], 'original redundant-joint timing only for explicitly marked references')
    cup_private = closure(motion.advance_direct_cup_v6)['previous_cup']
    clock.patch(cup_private, [(
        'workflow_nullspace_enabled=True)',
        'workflow_nullspace_enabled=True,workflow_original_nullspace_timing=True)', 1,
    )], 'original redundant-joint reference policy only during dry-cup motion')
    clock.record('original_reference_control', 'right/dry_cup_tip_and_axial_yaw',
                 original_nullspace_budget_rad_per_call=0.003,
                 scope='Dry-cup solver instance; other local solver policies are recorded separately',
                 physical_parameters_changed=False, acceptance_changed=False,
                 physical_success_inferred=False)
    clock.patch(interior.make_validated_aspiration_ik_v6, [(
        'workflow_nullspace_enabled=True)',
        'workflow_nullspace_enabled=True,workflow_original_nullspace_timing=True)', 1,
    )], 'original redundant-joint reference policy only for aspiration solvers')
    clock.patch(interior.adapt_vertical_aspiration_source, [(
        'Original joint margins, numerical iteration trust regions and posture gains; artificial per-call nullspace budget removed',
        'Original joint margins, numerical iteration trust regions and posture gains; original per-call nullspace reference budget retained during aspiration', 1,
    ), ('near_liquid_clock_rate=FACTOR', 'near_liquid_clock_rate=1.0', 1),
       ('executed_reference_factor=FACTOR',
        'base_reference_factor=FACTOR, guarded_descent_reference_factor=1.0', 1),
    ], 'accurate aspiration joint-reference and local timing evidence')
    clock.record('original_reference_control', 'right/aspiration_joint_reference',
                 original_nullspace_budget_rad_per_call=0.003,
                 scope='Aspiration solver instances only; motion-reference timing is recorded separately',
                 physical_parameters_changed=False, acceptance_changed=False,
                 physical_success_inferred=False)
    clock.patch(coating.adapt_direct_coating_source, [(
        'return "from v34_uncapped_right import _command_tick as _uncapped_command_tick\\n" + source',
        'return "workflow_original_nullspace_timing=True\\nfrom v34_uncapped_right import _command_tick as _uncapped_command_tick\\n" + source', 1,
    )], 'original redundant-joint reference policy only for the coating module')
    clock.patch(uncapped_right._dispense_source, [(
        '    old = found[0]\n'
        '    return _replace(source, old, old.replace("(task_t-21)", "2.0*(task_t-21)"),\n'
        '                    "enabled_exposure_thumb_reference")',
        '    return source', 1,
    )], 'original enabled-exposure thumb reference without changing physical gates')
    clock.patch(uncapped_right._grasp_source, [(
        '    source = _replace(source,\n'
        '        "Rotation.from_rotvec(rotation_error*min(1,.012/max(angle,1e-12)))",\n'
        '        "Rotation.from_rotvec(rotation_error)", "grasp_orientation_slew")\n'
        '    return _replace(source,\n'
        '        "                shift*=min(1,.0025/max(np.linalg.norm(shift),1e-12))\\n",\n'
        '        "", "grasp_cartesian_slew")',
        '    return source', 1,
    )], 'original angular and Cartesian reference increments during grasp verticalization')
    clock.record('original_reference_control', 'right/grasp_verticalization_reference',
                 angular_increment_rad_per_update=0.012,
                 cartesian_increment_m_per_update=0.0025,
                 control_period_s=0.02,
                 scope='Grasp-module post-lift correction only; shared acquisition phase clock remains locally accelerated',
                 physical_parameters_changed=False, acceptance_changed=False,
                 physical_success_inferred=False)
    clock.record('original_reference_control', 'right/coating_and_dispense_reference',
                 original_nullspace_budget_rad_per_call=0.003,
                 dispense_thumb_reference_factor=1.0,
                 physical_press_exposure_clock_unchanged=True,
                 scope='Coating module only; original nullspace enable window and all Cartesian motion clocks retained',
                 physical_parameters_changed=False, acceptance_changed=False,
                 physical_success_inferred=False)
    payload = dict(policy_id=POLICY_ID, base_reference_factor=FAST_REFERENCE_FACTOR,
                   reference_factor_overrides=REFERENCE_FACTOR_OVERRIDES,
                   factor_selection='Previously factor-two runtime references; explicit original-rate calls remain unchanged',
                   source_fraction_windows=SOURCE_FRACTION_WINDOWS,
                   transition_source_seconds=TRANSITION_S,
                   original_speedpour_source_segments=ORIGINAL_SPEEDPOUR_SEGMENTS,
                   original_loaded_return_source_clock=True,
                   right_hand_reference_handoff='Incoming actuator command; measured state and original bounds retained',
                   right_acquisition_landmarks='All contact phases invert the same final source clock; endpoint extension rates preserved',
                   source_restore_scope='Original timing law on freshly planned geometry; no trajectory replay or final command clipping',
                   window_coordinate='Nominal elapsed time within each named reference',
                   shaking_reference='Retained native-wrist parent timing',
                   physical_timestep_s=clock.DT, physical_model_parameters_unchanged=True,
                   requested_motion_geometry_changed=True,
                   original_and_experimental_acceptance_unchanged=True)
    dispatch = (
        '    if factor == 2.0 and _localized_runtime_reference(str(label)):\n'
        '        return _localized_timing_dispatch(\n'
        '            path, duration=duration, label=label, vlim=vlim, clock=clock,\n'
        '            alim=alim, protected=protected, maximum_rate=maximum_rate,\n'
        '            minimum_duration=minimum_duration, accept_clock=accept_clock, factor=factor)\n'
        '    import v34_topp_core as core\n')
    clock.patch(clock.retime, [('    import v34_topp_core as core\n', dispatch, 1)],
                'selected fourth-speed motion references with original-rate windows',
                helpers={'_localized_runtime_reference': accelerated_reference,
                         '_localized_timing_dispatch': _localized_dispatch})
    clock.patch(clock.save_report, [
        ("implementation='fixed_reference_factor_without_software_speed_caps'",
         "implementation='localized_original_speed_windows'", 1),
        ('requested_motion_factor=FACTOR,',
         "requested_motion_factor=_localized_policy_payload['base_reference_factor'],\n"
         "                  factor_scope='Per-reference acceleration with original-rate windows and rollback overrides',\n"
         '                  local_speed_policy=_localized_policy_payload,', 1),
    ], 'localized timing report identity', helpers={'_localized_policy_payload': payload})
    from v34_rate_consumers import install as install_rate_consumers
    from v34_wall_pour import install as install_wall_pour
    from v34_edge_center import install as install_edge_center
    install_rate_consumers()
    install_wall_pour()
    install_edge_center()
    clock.record('policy_install', 'LocalizedSpeed', **payload,
                 physical_success_inferred=False, main_or_default_promoted=False)
    _INSTALLED = True
    return PATCHED_SYMBOLS
