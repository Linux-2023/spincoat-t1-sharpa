"""LEFT NoLimits2x reference cutover after the frozen TOPP/overlap chain.

Only reference clocks and software speed policy change. The native callbacks,
nonlinear closure, actual-state replans, position/geometry guards, controller
feedback, liquid clocks and physical qualification windows remain authoritative.
"""
from functools import wraps
import inspect
import linecache
import textwrap

import numpy as np
from scipy.interpolate import BPoly

import v34_uncapped_clock as uncapped

_INSTALLED = False
PATCHED_SYMBOLS = (
    't1_v10_v27_stage_v28.adapt_left_stage',
    'v34_topp_left_paths.adapt_callable',
    'v34_topp_left_paths._retime',
    'v34_topp_left_paths.retime_cartesian',
    'v34_topp_left_paths.phase_timing',
    'v34_topp_left_paths.overlay_duration',
    'v34_topp_left_paths.carry_reference',
    'v34_topp_left_paths.retime_insertion',
    'v34_topp_left_paths.speedpour_clock',
    'v34_topp_left_paths.smooth_reference',
    'v34_topp_left_paths.LeftObservations.finish',
    'v34_topp_left.COVERAGE',
    't1_left_cartesian_segment.plan_cartesian_arm_segment',
    't1_left_standing_workflow_v6.make_standing_transport',
    't1_left_opposite_workflow_v7.make_standing_transport',
    't1_human_pour_v4_joint_reference.plan_segment',
    't1_actual_frame_return_v7.plan_return',
    't1_actual_frame_return_v7.plan_actual_root_descent_free_yaw',
    'v34_speedpour._retime',
    'v34_smoothtubereturn_reference._clock',
    'v34_smoothtubereturn_reference.Reference.__init__',
    'v34_smoothtubereturn_reference._command_guards',
    'v34_smoothtubereturn_reference.plan_loaded_return',
    'v34_smoothtubereturn_reference.plan_empty_ready',
)

MOTION_PHASES = (
    'standing_lean_before_pickup', 'approach_outside_tube', 'close_on_tube',
    'lift_tube', 'pre_mix_clearance_*', 'human_pour_idle_extend',
    'human_pour_idle_fold', 'coarse_carry_to_receiver', 'human_pour_approach',
    'human_pour_tilt', 'human_pour_return', 'raise_upright_clear_of_rack_before_return',
    'coarse_return_free_corridor', 'return_clearance_waypoint_*',
    'return_above_holder', 'lower_into_holder', 'release_tube',
    'withdraw_empty_hand', 'human_pour_idle_restore_fold',
    'human_pour_idle_restore_extend', 'empty_hand_clearance_before_ready',
    'return_to_original_ready_joints',
)
PHYSICAL_PHASES = (
    'coarse_dock_before_prep', 'settle_standing_lean', 'settle_straightened',
    'hold_before_tube_lift', 'stabilize_lifted_tube',
    'wait_for_original_held_readiness', 'settle_before_precise_pour',
    'human_pour_drain_wait', 'human_pour_retreat', 'pour_hold',
    'settle_original_ready',
)


def _patch(function, replacements, label, helpers=None):
    return uncapped.patch(function, replacements, label, helpers=helpers)


def _cells(function):
    return dict(zip(function.__code__.co_freevars,
                    (cell.cell_contents for cell in (function.__closure__ or ()))))


def _cache_factory_source(function, predecessor, replacements, label):
    """Describe an existing generated body; never compile or clone its callable."""
    source = textwrap.dedent(inspect.getsource(predecessor.__code__))
    for old, new, count in replacements:
        source = uncapped.replace_once(source, old, new, label, count=count)
    filename = function.__code__.co_filename
    if not filename.startswith('<'):
        raise RuntimeError('Expected the frozen generated source seam: ' + label)
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)


def _rate_enclosures(spatial, clock):
    """Finite mathematical bounds for diagnostics/sweep, never a speed gate."""
    from v34_smoothtubereturn_bounds import BernsteinRateBounds
    bounder = BernsteinRateBounds(spatial, clock)
    intervals = [(float(a), float(b), bounder.bound(float(a), float(b)))
                 for a, b in zip(clock.x[:-1], clock.x[1:])]
    return bounder, intervals


def _speedpour_source_clock(spatial, source_time, source_rates,
                            fixed_source_intervals, peak_design, dt):
    """Build the supplied uncapped rate program before the one final 2x clock.

    Harmonic/secant constraints below construct a monotone polynomial; they
    are dimensionless interpolation-shape constraints, not speed ceilings.
    Former fixed boundaries remain source knots, never protected durations.
    """
    from v34_speedpour_profile import PROFILE
    if abs(float(dt) - uncapped.DT) > 1e-12:
        raise RuntimeError('Uncapped LEFT requires the original 1ms reference grid')
    times = np.asarray(source_time, dtype=float)
    rates = np.asarray(source_rates, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or rates.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(rates).all()
            or np.any(np.diff(times) <= 0.) or np.any(rates <= 0.)):
        raise ValueError('Invalid supplied SpeedPour source clock')
    ticks = len(times) - 1
    stride = max(1, round(PROFILE['clock_knot_source_s'] / dt))
    cuts = {0, ticks, *range(stride, ticks, stride)}
    for a, b in fixed_source_intervals:
        cuts.update((a, b))
    indices = np.asarray(sorted(cuts), dtype=int)
    coordinates = times[indices]
    inverse = 1. / rates
    accumulated = np.r_[0., np.cumsum(np.diff(times) * (inverse[:-1] + inverse[1:]) * .5)]
    requested = np.array([uncapped.ceil_native_duration(float(value), dt)
                          for value in np.diff(accumulated[indices])])
    secants = np.diff(coordinates) / requested
    derivatives = np.r_[secants[0],
                        np.minimum(2. * secants[:-1] * secants[1:] / (secants[:-1] + secants[1:]),
                                   1.2 * np.minimum(secants[:-1], secants[1:])),
                        secants[-1]]
    clock = BPoly.from_derivatives(np.r_[0., np.cumsum(requested)],
                                  [[s, v, 0.] for s, v in zip(coordinates, derivatives)])
    if not np.isfinite(clock.c).all() or np.min(clock.derivative().c) < -1e-10:
        raise RuntimeError('SpeedPour source clock is nonfinite or not monotone')
    bounder, intervals = _rate_enclosures(spatial, clock)
    return clock, bounder, intervals, requested.copy(), requested, np.zeros(len(requested), dtype=bool)


def _observe_phase(self):
    if self.phase is not None:
        uncapped.record('left_phase_observation', 'left.' + self.phase,
                        physical_duration_s=float(self.seconds),
                        clock_kind=('physical_feedback_or_process' if self.phase in PHYSICAL_PHASES
                                    else 'mixing_owned_by_wrist_adapter' if self.mixing
                                    else 'motion_reference'),
                        old_speed_certificate_applicable=False,
                        protected_motion_duration_claim=False)
    self.phase = None
    self.seconds = 0.0
    self.mixing = False


def _install_path_clocks(paths):
    _patch(paths._retime, [(
        "return core.retime(path, duration=float(duration), label=label, vlim=vlim,\n"
        "                       clock=clock, alim=None, protected=protected,\n"
        "                       maximum_rate=MAXIMUM_RATE, minimum_duration=minimum_duration,\n"
        "                       accept_clock=accept_clock)",
        "return _uncapped_retime(path, duration=float(duration), label=label, clock=clock)", 1,
    )], 'LEFT final reference clock', {'_uncapped_retime': uncapped.retime})
    _patch(paths.retime_cartesian, [(
        "timing = _retime(original, duration, 'left.' + name, 2.0,\n"
        "                     protected=_protected_phase(name, duration),\n"
        "                     minimum_duration=duration if name in phase_plan.get('phase_duration_overrides_s', {}) else None)",
        "timing = _retime(original, duration, 'left.' + name, None)", 1,
    ), ('topp=timing.report, original_ik_knots_and_spacing_preserved=True',
        'uncapped_reference=timing.report, original_ik_knots_and_spacing_preserved=True, old_speed_certificate_applicable=False', 1)],
        'LEFT Cartesian final reference')
    _patch(paths.phase_timing, [(
        "limits = _engineering_velocity(path)\n"
        "    if goalarm is not None:\n"
        "        limits[:len(q0)] = 2.0\n"
        "    return _retime(path, duration, 'left.' + name, limits,\n"
        "                   protected=_protected_phase(name, duration),\n"
        "                   minimum_duration=duration if name in phase_plan.get('phase_duration_overrides_s', {}) else None)",
        "return _retime(path, duration, 'left.' + name, None)", 1,
    )], 'LEFT synchronized arm and hand clock')
    _patch(paths.overlay_duration, [(
        "timing = _retime(path, duration, 'left.overlay.' + phase, _engineering_velocity(path))",
        "timing = _retime(path, duration, 'left.overlay.' + phase, None)", 1,
    )], 'LEFT synchronized overlay clock')
    _patch(paths.carry_reference, [(
        "timing = _retime(core.quintic(start, end, duration), duration, 'left.coarse_carry_to_receiver', 2.0)",
        "timing = _retime(core.quintic(start, end, duration), duration, 'left.coarse_carry_to_receiver', None)", 1,
    ), ("bounds = BernsteinRateBounds(timing.path, timing.clock)\n"
        "    intervals = bounds.partition(2.0, require_native_tick(m))",
        "bounds, intervals = _uncapped_enclosures(timing.path, timing.clock)", 1),
        ("    if peak > 2.0:\n        raise RuntimeError('TOPP carry original all-time2rad/s guard rejected')\n", '', 1)],
        'LEFT carry diagnostic rate enclosure', {'_uncapped_enclosures': _rate_enclosures})
    _patch(paths.retime_insertion, [(
        'conservative_contact_stroke_clock_preserved=True',
        'contact_stroke_geometry_preserved=True, protected_motion_duration_claim=False', 1,
    )], 'LEFT insertion report identity')
    _patch(paths.speedpour_clock, [(
        'native = previous(spatial, source_time, source_rates, fixed_source_intervals, peak_design, dt)',
        'native = _uncapped_speedpour_source(spatial, source_time, source_rates, fixed_source_intervals, peak_design, dt)', 1,
    ), ("timing = _retime(spatial, old_clock.x[-1], 'left.human_pour_' + context['segment'], peak_design,\n"
        "                     clock=old_clock, protected=protected,\n"
        "                     accept_clock=_native_rate_acceptance(peak_design, dt))",
        "timing = _retime(spatial, old_clock.x[-1], 'left.human_pour_' + context['segment'], None, clock=old_clock)", 1),
        ('bounds = BernsteinRateBounds(spatial, timing.clock)\n    intervals = bounds.partition(peak_design, dt)',
         'bounds, intervals = _uncapped_enclosures(spatial, timing.clock)', 1),
        ("    if max(float(row[2].max()) for row in intervals) > peak_design:\n        raise RuntimeError('TOPP SpeedPour original design-speed certificate rejected')\n", '', 1),
        ('certified_piece_durations_s=old_durations.tolist()', 'nominal_piece_durations_s=old_durations.tolist()', 1),
        ('fixed_pieces=old_fixed.tolist(), protected_elapsed_intervals_s=[list(pair) for pair in protected]',
         'source_fixed_pieces=old_fixed.tolist(), protected_elapsed_intervals_s=[], old_speed_certificate_applicable=False', 1)],
        'LEFT uncapped SpeedPour composition',
        {'_uncapped_speedpour_source': _speedpour_source_clock, '_uncapped_enclosures': _rate_enclosures})
    _patch(paths.smooth_reference, [(
        "fixed = reference.fixed_durations.copy()\n"
        "    if loaded:\n"
        "        fixed[-1] = True  # Entire >=2s side-entry and incoming rack-rate limit.\n"
        "        if len(source_durations) == 3:\n"
        "            fixed[0] = True  # Native clearance minimum/explicit override.\n"
        "    protected = tuple((float(source_edges[i]), float(source_edges[i + 1])) for i in np.flatnonzero(fixed))",
        'fixed = np.zeros(len(source_durations), dtype=bool)', 1,
    ), ("timing = _retime(reference.spatial, source_duration,\n"
        "                     'left.smooth_loaded_return' if loaded else 'left.smooth_empty_ready',\n"
        "                     2.0, clock=reference.clock, protected=protected,\n"
        "                     accept_clock=_native_rate_acceptance(2.0, timestep))",
        "timing = _retime(reference.spatial, source_duration,\n"
        "                     'left.smooth_loaded_return' if loaded else 'left.smooth_empty_ready',\n"
        "                     None, clock=reference.clock)", 1),
        ('reference.rate_bounds = BernsteinRateBounds(reference.spatial, reference.clock)\n'
         '    reference.rate_intervals = reference.rate_bounds.partition(2.0, timestep)',
         'reference.rate_bounds, reference.rate_intervals = _uncapped_enclosures(reference.spatial, reference.clock)', 1),
        ("    if reference.rate_bound > 2.0:\n        raise RuntimeError('TOPP SmoothTubeReturn original Bernstein-rate guard rejected')\n", '', 1),
        ("    if loaded and reference.durations[-1] + 1e-9 < source_durations[-1]:\n        raise RuntimeError('TOPP SmoothTubeReturn shortened the protected rack-side entry')\n", '', 1)],
        'LEFT smooth final reference clock', {'_uncapped_enclosures': _rate_enclosures})
    _patch(paths.LeftObservations.finish, [(
        "if self.phase is not None:\n"
        "        reason = FIXED_PHASES.get(self.phase)\n"
        "        if reason is None and self.mixing:\n"
        "            reason = 'Original V27 frequency/amplitude/ramp/steady exposure and derivative-dependent prefilter; measured liquid observation uses actual elapsed time.'\n"
        "        if reason is not None:\n"
        "            core.note_fixed('left.' + self.phase, reason=reason, duration=self.seconds)\n"
        "    self.phase = None\n"
        "    self.seconds = 0.0\n"
        "    self.mixing = False",
        '_uncapped_observe_phase(self)', 1,
    )], 'LEFT physical observation identity', {'_uncapped_observe_phase': _observe_phase})


def _install_native_planners(paths):
    import t1_left_cartesian_segment as cartesian
    import t1_left_standing_workflow_v6 as standing
    import t1_left_opposite_workflow_v7 as opposite
    import t1_human_pour_v4_joint_reference as joint
    import t1_actual_frame_return_v7 as actual
    import v34_v34returnoffsetaware as offset

    _patch(cartesian.plan_cartesian_arm_segment, [(
        'if margin<.049 or abs(velocity).max()>2.:', 'if margin<.049:', 1,
    ), ('Cartesian arm target spline violates planning margin/speed bounds',
        'Cartesian arm target spline violates planning position margin', 1)],
        'LEFT native Cartesian position-only guard')
    if opposite.make_standing_transport is not standing.make_standing_transport:
        raise RuntimeError('LEFT standing carry aliases must share the installed TOPP factory')
    _patch(standing.make_standing_transport.__wrapped__, [(
        'if margin<.049 or speed>2 or cmargin<0:', 'if margin<.049 or cmargin<0:', 1,
    ), ('V6 own-state coarse entry original speed/margin/range guard failed',
        'V6 own-state coarse entry original margin/range guard failed', 1)],
        'LEFT actual standing carry factory')
    # This factory runs later. Its adapted child must retain the real mutable
    # native namespace and closure cells, rather than TOPP's dictionary clone.
    _patch(paths.adapt_callable, [(
        "if getattr(previous, '__wrapped__', None) is not None:\n"
        "        raise RuntimeError('TOPP LEFT source seam has an unexpected wrapper; refusing to bypass it: ' + label)\n"
        "    source = textwrap.dedent(inspect.getsource(previous))\n"
        "    for old, new, count in replacements:\n"
        "        source = once(source, old, new, label, count)\n"
        "    tree = ast.parse(source)\n"
        "    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]\n"
        "    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):\n"
        "        raise RuntimeError('TOPP LEFT expected one synchronous predecessor: ' + label)\n"
        "    function = functions[0]\n"
        "    if any(isinstance(node, ast.Nonlocal) for node in ast.walk(function)):\n"
        "        raise RuntimeError('TOPP LEFT cannot detach a mutating closure: ' + label)\n"
        "    function.decorator_list = []\n"
        "    environment = dict(previous.__globals__)\n"
        "    environment.update(inspect.getclosurevars(previous).nonlocals)\n"
        "    environment.update(helpers or {})\n"
        "    exec(compile(ast.fix_missing_locations(tree), '<topp-left:' + label + '>', 'exec'), environment)\n"
        "    result = environment[function.name]\n"
        "    update_wrapper(result, previous)\n"
        "    result._topp_predecessor = previous\n"
        "    result._topp_source_anchors = tuple(old for old, _, _ in replacements)\n"
        "    return result",
        "return _uncapped_patch(previous, replacements, label, helpers=helpers)", 1,
    )], 'LEFT future actual-callable source adaptation', {'_uncapped_patch': uncapped.patch})
    # These two wrappers remain in the execution chain; patch their native
    # predecessor object without unwrapping/replacing the installed entrypoint.
    _patch(joint.plan_segment.__wrapped__.__wrapped__, [(
        'if peak>2. or margin<.05-1e-6:', 'if margin<.05-1e-6:', 1,
    )], 'LEFT natural pour joint position guard')
    _patch(joint.plan_segment, [(
        "time_law='native_analytic_quintic_chord_with_TOPP_RA_C2_clock', topp=timing.report",
        "time_law='native_analytic_quintic_chord_with_uncapped_factor_two_clock', uncapped_reference=timing.report", 1,
    )], 'LEFT pour approach timing report')
    offset_wrapper = actual.plan_return.__wrapped__
    constrained = _cells(offset_wrapper)['constrained']
    _cache_factory_source(constrained, offset.ORIGINAL_PLAN_RETURN,
                          [(offset._BOUND_ANCHOR, offset._BOUND_REPLACEMENT, 1)],
                          'offset-aware existing constrained body')
    _patch(constrained, [(
        'if speed>2 or margin<.05 or(control_margin is not None and control_margin<0):',
        'if margin<.05 or(control_margin is not None and control_margin<0):', 1,
    ), ('Original return speed/margin/control-range failed',
        'Original return margin/control-range failed', 1)],
        'LEFT actual offset-aware return position guard')
    _patch(offset_wrapper, [(
        '"native_guards_unchanged": True,',
        '"native_position_geometry_control_guards_retained": True,\n'
        '            "software_velocity_rejection_removed": True,', 1,
    )], 'LEFT offset-aware report identity')
    _patch(actual.plan_actual_root_descent_free_yaw.__wrapped__, [(
        'if speed>2 or margin<.05:', 'if margin<.05:', 1,
    ), ('Original descent speed/margin reject', 'Original descent position-margin reject', 1)],
        'LEFT actual-root insertion position guard')


def _install_speedpour():
    import v34_speedpour as speed
    _patch(speed._retime.__wrapped__, [(
        'rates = np.maximum(1.0, np.minimum(fast, peak_design / np.maximum(source_speed, 1e-12)))',
        'rates = fast', 1,
    ), (' or peak > 2.0 or margin < minimum', ' or margin < minimum', 1),
        ('    source_speed = np.max(np.abs(native(source_time, 1)), axis=1)\n', '', 1),
        ('profile=dict(PROFILE), original_duration_s=',
         'source_profile=dict(PROFILE), source_profile_is_provenance_only=True, original_duration_s=', 1),
        ('certified_clock_piece_durations_s=durations.tolist()', 'executed_clock_piece_durations_s=durations.tolist()', 1),
        ('native_rate_fixed_clock_pieces=frozen.tolist()', 'protected_clock_piece_mask=frozen.tolist()', 1),
        ('design_peak_rad_s=peak_design', 'software_speed_limit_enforced=False', 1),
        ("certificate='Interval-local Bernstein Qprime(s(I))*sdot(I); exact same executed polynomials'",
         "rate_enclosure_scope='Diagnostic bound for the executed polynomial; not speed acceptance', old_speed_certificate_applicable=False", 1),
        ('certified_intervals=len(intervals)', 'enclosure_intervals=len(intervals)', 1),
        ("recovery_boundary_source_seconds=PROFILE['return_boundary_source_s']",
         "source_recovery_boundary_seconds=PROFILE['return_boundary_source_s'], recovery_duration_protected=False", 1)],
        'LEFT SpeedPour removes reference speed clipping')
    _patch(speed._retime, [(
        "result['topp'] = context['timing'].report", "result['uncapped_reference'] = context['timing'].report", 1,
    ), ("result['speedpour_retiming']['source_r08_clock'] = context['source_clock']",
        "result['speedpour_retiming']['nominal_uncapped_source_clock'] = context['source_clock']", 1),
        ("'unchanged_native_dense_C2_spatial_reference_with_TOPP_RA_certified_C2_clock'",
         "'unchanged_native_dense_C2_spatial_reference_with_uncapped_factor_two_clock'", 1)],
        'LEFT SpeedPour final timing identity')


def _install_smooth_return(paths):
    import v34_smoothtubereturn_reference as smooth
    _patch(smooth._clock, [(
        'for i, limit in enumerate(join_rate_limits, start=1):', 'for i in range(1, len(positions) - 1):', 1,
    ), ('min(2.0 * left * right / (left + right), 1.2 * min(left, right), limit)',
        'min(2.0 * left * right / (left + right), 1.2 * min(left, right))', 1)],
        'LEFT smooth monotone clock without rack rate ceiling')
    _patch(smooth.Reference.__init__.__wrapped__, [(
        "for _ in range(4):\n"
        "        if durations.sum() > 30.0:\n"
        "            raise RuntimeError(\"Continuous reference exceeds native30s planning bound\")\n"
        "        clock = _clock(self.length, self.joins, durations, limits)\n"
        "        rate_bounds = BernsteinRateBounds(self.spatial, clock)\n"
        "        intervals = rate_bounds.partition(2.0, timestep)\n"
        "        joint_upper = np.max(np.stack([row[2] for row in intervals]), axis=0)\n"
        "        rate_bound = float(joint_upper.max())\n"
        "        if rate_bound <= 2.0:\n"
        "            break\n"
        "        if fixed.all():\n"
        "            raise RuntimeError(\"Fixed phase timing cannot certify the original2rad/s bound\")\n"
        "        durations[~fixed] = np.ceil(durations[~fixed] * (rate_bound / 2.0) * (1.0 + 1e-8) / timestep) * timestep\n"
        "    else:\n"
        "        raise RuntimeError(\"Unresolved all-time2rad/s certification; no clipping or fixed-duration extension\")",
        "if durations.sum() > 30.0:\n"
        "        raise RuntimeError(\"Continuous reference exceeds native30s planning bound\")\n"
        "    clock = _clock(self.length, self.joins, durations, limits)\n"
        "    rate_bounds, intervals = _uncapped_enclosures(self.spatial, clock)\n"
        "    joint_upper = np.max(np.stack([row[2] for row in intervals]), axis=0)\n"
        "    rate_bound = float(joint_upper.max())", 1,
    )], 'LEFT smooth source removes rate-driven duration inflation',
        {'_uncapped_enclosures': _rate_enclosures})
    _patch(smooth._command_guards.__wrapped__, [(
        'if margin < .049 or float(np.abs(velocities).max()) > 2.0 + 1e-9:', 'if margin < .049:', 1,
    ), ('Native .049rad joint-margin/2rad/s command guard rejected continuous reference',
        'Native .049rad position-margin guard rejected continuous reference', 1),
        ("all_time_rate_bound_method='Interval-local Bernstein hulls of Qprime(s(I)) and clock derivative on the same time interval; no sampled-speed acceptance'",
         "all_time_rate_bound_method='Diagnostic Bernstein enclosure for exact executed polynomials; no software speed ceiling', old_speed_certificate_applicable=False, software_speed_limit_enforced=False", 1),
        ("trace['certified_rate_intervals_s_and_joint_bounds']", "trace['diagnostic_rate_enclosures_s_and_joint_bounds']", 1)],
        'LEFT smooth retains finite position and seam guards')
    _patch(smooth._command_guards, [(
        'report.update(topp=reference.topp_timing.report,', 'report.update(uncapped_reference=reference.topp_timing.report,', 1,
    ), ('native_guards_recomputed_on_new_clock=True,',
        'finite_position_endpoint_seam_guards_recomputed=True, protected_motion_duration_claim=False,', 1)],
        'LEFT smooth final report identity')
    _patch(smooth.plan_loaded_return, [(
        'coarse_duration = transport_duration(remaining, np.linalg.norm(rotation_vector), config=CoarseFineConfig())',
        'coarse_duration = float(CoarseFineConfig().minimum_duration_s)', 1,
    ), ('if not 0.0 <= minimum_clearance_duration_s <= 30.0 or not 2.0 <= side_entry_duration_s <= 30.0:',
        'if not 0.0 <= minimum_clearance_duration_s <= 30.0 or not 0.0 < side_entry_duration_s <= 30.0:', 1),
        ('max(.5, clear_lift / .075, minimum_clearance_duration_s)',
         '(minimum_clearance_duration_s if minimum_clearance_duration_s > 0.0 else .5)', 1),
        ('# The rack-side entry keeps its original >=2s duration, but crosses its\n'
         '        # incoming endpoint slowly rather than stopping: arc rate<=10mm/s.',
         '# Nominal source durations are composed once by the final clock.\n'
         '        # Legacy join limits identify the loaded path but are not enforced.', 1),
        ('        if max(rack_path_speed, rack_palm_speed) > .010 + 1e-10:\n'
         '            raise RuntimeError("Rack side-entry join exceeds declared10mm/s target/FK reference speed")\n', '', 1),
        ("scope='Clearance lift, free-return corridor and low-speed final rack side-entry; stop before unchanged actual alignment decision/insertion'",
         "scope='Clearance lift, free-return corridor and rack side-entry under one uncapped reference; stop before unchanged actual alignment decision/insertion'", 1),
        ('rack_entry_arc_rate_limit_m_s=.010, original_rack_entry_minimum_duration_s=2.0,',
         'rack_entry_speed_limit_enforced=False, source_rack_entry_duration_s=side_entry_duration_s, rack_entry_duration_protected=False,', 1)],
        'LEFT loaded clearance and rack entry remove design-speed policy')
    # TOPP compiled this body without retaining source text. Cache precisely
    # its two declared edits so patch() can use the ACTUAL code object.
    _cache_factory_source(smooth.plan_empty_ready, smooth.plan_empty_ready._topp_predecessor, [(
        'hand_progress = 0.0 if time <= reference.join_times[0] else (time - reference.join_times[0]) / reference.durations[1]',
        'hand_progress = _topp_ready_hand_progress(reference, time)', 1,
    ), ('original_ready_hand_duration_s=3.0, executed_ready_hand_duration_s=float(reference.durations[1]),',
        "original_ready_hand_duration_s=3.0, source_ready_hand_duration_s=float(reference.topp_source_durations[1]), executed_ready_hand_duration_s=float(reference.durations[1]), safe_hand_progress_clock='original_reference_elapsed_time_under_same_TOPP_clock',", 1)],
        'existing TOPP empty-hand source')
    _patch(smooth.plan_empty_ready, [(
        "safe_hand_progress_clock='original_reference_elapsed_time_under_same_TOPP_clock'",
        "safe_hand_progress_clock='original_reference_elapsed_time_under_same_uncapped_factor_two_clock'", 1,
    )], 'LEFT safe-hand source progress identity')


def _adapt_stage(source):
    retained = (
        "from t1_single_grasp_v10 import closing_target;handcmd=closing_target(h0,closed,u)",
        "_topp_source_elapsed=_topp_left.source_elapsed(_topp_timing,(k+1)*m.opt.timestep)",
        "_overlap.before_step(m,d,phase);mujoco.mj_step(m,d);_overlap.after_step(m,d,phase);obs=ledger.observe",
        "if not held_gate['pass_']:raise RuntimeError('actual tube held/lift gate failed')",
        "if not dock.stopped:raise RuntimeError('Actual base did not settle before precision pour')",
        "if ledger.delivered_m3<max(c.target_pour_volume_m3,c.tube_initial_volume_m3)*.99:raise RuntimeError('Gravity pour delivered less than original99percent target; no flow valve/cap')",
        "if rack_above_position_error<=.0005 and rack_above_orientation_error<=.2:",
        "_overlap.release(m,d,dock,body,tube_initial)",
        "_overlap.finish_left(m,d)",
        "if name=='settle_original_ready':duration=.25",
    )
    for anchor in retained:
        if source.count(anchor) != 1:
            raise RuntimeError('Uncapped LEFT retained source boundary mismatch: ' + anchor)
    replacements = [
        ("    if transport['report']['v10_retimed_peak_joint_speed_rad_s']>2.:raise RuntimeError('Retimed approach exceeds2rad/s')\n", '', 1),
        ('not requested_duration<=override<=30.', 'not 0.<override<=30.', 1),
        ('not 2.<=_smooth_override<=30.', 'not 0.<_smooth_override<=30.', 1),
        ('requested_duration_s=requested_duration,executed_duration_s=duration,',
         "requested_duration_s=requested_duration,source_duration_s=duration,final_duration_policy='uncapped_factor_two',", 1),
        ('native_minimum_override_s=_smooth_override,', 'nominal_source_override_s=_smooth_override,', 1),
        ('Original validated duration override retained in coherent reference; no command clipping',
         'Explicit source duration composed once with uncapped factor two; no protected-duration claim or command clipping', 1),
        ('Single4.5s Cartesian final approach plus C2 finger closure; no intermediate open/seat/realignment phases',
         'Single synchronized Cartesian and nonlinear hand closure; nominal4.5s source accelerated once; no intermediate open/seat/realignment phases', 1),
        ('Physical arm orientation correction during existing2s clearance; actual hand-tube transform read once, no object pose write or extra hold',
         'Physical arm orientation correction on the accelerated nominal2s clearance reference; actual hand-tube transform read once, no object pose write or extra hold', 1),
    ]
    for old, new, count in replacements:
        source = uncapped.replace_once(source, old, new, 'final LEFT source factory', count=count)
    for anchor in retained:
        if source.count(anchor) != 1:
            raise RuntimeError('Uncapped LEFT changed a physical boundary: ' + anchor)
    return source


def install():
    """Install after TOPP/overlap and Main's common core.retime cutover."""
    global _INSTALLED
    if _INSTALLED:
        return PATCHED_SYMBOLS
    import v34_topp_core as core
    import v34_topp_left as left
    import v34_topp_left_paths as paths
    import v34_dualarmoverlap as overlap
    import t1_v10_v27_stage_v28 as stage
    if not left._INSTALLED or not overlap._INSTALLED or core.retime is not uncapped.retime:
        raise RuntimeError('Uncapped LEFT requires the installed TOPP/overlap chain and shared uncapped core')
    _install_path_clocks(paths)
    _install_native_planners(paths)
    _install_speedpour()
    _install_smooth_return(paths)
    previous = stage.adapt_left_stage

    @wraps(previous)
    def adapt_left_stage(source):
        return _adapt_stage(previous(source))

    stage.adapt_left_stage = adapt_left_stage
    left.COVERAGE = [dict(stage=phase, decision='uncapped_reference_factor_two',
                          reason='One final shared motion clock; old speed and protected-duration certificates inapplicable.')
                     for phase in MOTION_PHASES]
    left.COVERAGE += [dict(stage=phase, decision='native_physical_clock',
                           reason='Native feedback, qualification, settling, fluid clock and failure gate retained.')
                      for phase in PHYSICAL_PHASES]
    left.COVERAGE.append(dict(stage='gentle_shake_mouth_pivot', decision='owned_by_wrist_adapter',
                              reason='Main owns waveform/prefilter/rate-limit cutover; actual liquid and callback clocks remain physical.'))
    uncapped.record('install', 'left.uncapped_reference', patched_symbols=list(PATCHED_SYMBOLS),
                    motion_reference_factor=uncapped.FACTOR,
                    old_speed_certificate_applicable=False,
                    protected_motion_duration_claim=False,
                    physical_timestep_s=uncapped.DT)
    _INSTALLED = True
    return PATCHED_SYMBOLS
