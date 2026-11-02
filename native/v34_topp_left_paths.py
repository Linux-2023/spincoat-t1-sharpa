"""Exact LEFT reference clocks for the R08 TOPP candidate.

This module changes reference evaluation only. Native callbacks still own every
1 ms physical step, actual-state reanchor, contact gate and liquid observation.
No universal torso/finger speed limit exists in R08: their added timing ceiling
is explicitly 1.5 times the original polynomial derivative enclosure, not a
hardware rating. Arm limits remain the original per-stage limits.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
from functools import update_wrapper
import inspect
import textwrap

import numpy as np
from scipy.interpolate import BPoly

import v34_topp_core as core
from v34_smoothtubereturn_bounds import BernsteinRateBounds

MAXIMUM_RATE = 1.5
_OVERLAYS = {}
_PENDING = {}
_POUR_CONTEXT = ContextVar('topp_left_pour_reference', default=None)

FIXED_PHASES = {
    'close_on_tube': 'Entire original R03 nonlinear synchronized closure/contact acquisition; conservative qualification policy, not a proven physical minimum.',
    'hold_before_tube_lift': 'Original acquisition hold, 10Hz actual-tube tracking when enabled, and subsequent encoder-freeze boundary.',
    'stabilize_lifted_tube': 'Original stabilization and last500 native-observation held qualification; no reference-sample substitution.',
    'wait_for_original_held_readiness': 'State-dependent actual held/contact readiness with original bounded timeout.',
    'release_tube': 'Original contact-unloading stroke, hold-controller release and physical encoder-offset decay; conservative unchanged2s policy.',
    'lower_into_holder': 'Original rack-contact insertion stroke preserved conservatively in full;2.5s is retained choreography, not asserted a universal physical minimum.',
    'settle_original_ready': 'Original0.25s ready settling before final tube/environment/liquid acceptance.',
    'coarse_dock_before_prep': 'Unmodified closed-loop docking/braking and actual0.5s stop predicate; no geometric reference.',
    'settle_standing_lean': 'Original base/torso5mrad qualification continuously0.5 physical seconds.',
    'settle_straightened': 'Original base/torso qualification; native continuous physical-time window.',
    'settle_before_precise_pour': 'Original state-dependent stopped-base gate before liquid manipulation.',
    'human_pour_drain_wait': 'Actual ledger termination and unchanged2s failure timeout, not nominal motion.',
    'human_pour_retreat': 'Original constant-command0.1s ledger interval before delivery acceptance.',
    'pour_hold': 'Original actual-pose feedback and fluid observation, not an exported fixed reference.',
}


def once(source, old, new, label, count=1):
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(f'TOPP LEFT {label}: expected {count} exact anchors, found {actual}: {old[:140]!r}')
    return source.replace(old, new, count)


def adapt_callable(previous, replacements, label, helpers=None):
    """Execute the immediate predecessor body with only checked clock edits.

    Read-only closure bindings are retained, not an unwrapped older function.
    The original guards/loops stay in that body; it is never run twice. This is
    needed for native nested carry and coefficient/hand seams without a public
    planner hook. Changed callable keeps __wrapped__ for source provenance.
    """
    if getattr(previous, '__wrapped__', None) is not None:
        raise RuntimeError('TOPP LEFT source seam has an unexpected wrapper; refusing to bypass it: ' + label)
    source = textwrap.dedent(inspect.getsource(previous))
    for old, new, count in replacements:
        source = once(source, old, new, label, count)
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise RuntimeError('TOPP LEFT expected one synchronous predecessor: ' + label)
    function = functions[0]
    if any(isinstance(node, ast.Nonlocal) for node in ast.walk(function)):
        raise RuntimeError('TOPP LEFT cannot detach a mutating closure: ' + label)
    function.decorator_list = []
    environment = dict(previous.__globals__)
    environment.update(inspect.getclosurevars(previous).nonlocals)
    environment.update(helpers or {})
    exec(compile(ast.fix_missing_locations(tree), '<topp-left:' + label + '>', 'exec'), environment)
    result = environment[function.name]
    update_wrapper(result, previous)
    result._topp_predecessor = previous
    result._topp_source_anchors = tuple(old for old, _, _ in replacements)
    return result


def require_native_tick(m):
    dt = float(m.opt.timestep)
    if abs(dt - .001) > 1e-12:
        raise RuntimeError('TOPP LEFT requires the unchanged original1ms timestep')
    return dt


def native_tick_count(m, duration):
    dt = require_native_tick(m)
    ticks = round(float(duration) / dt)
    if ticks < 1 or abs(ticks * dt - float(duration)) > 1e-9:
        raise RuntimeError('TOPP LEFT reference duration does not end on a native tick')
    return ticks


def native_times(m, duration, include_start=True):
    ticks = native_tick_count(m, duration)
    return np.arange(0 if include_start else 1, ticks + 1, dtype=float) * float(m.opt.timestep)


def source_elapsed(timing, elapsed):
    if timing is None:
        return elapsed
    return timing.source_time(np.clip(elapsed, 0.0, timing.duration))


def _engineering_velocity(path):
    bound = np.max(np.abs(path.derivative().c), axis=(0, 1))
    return np.maximum(MAXIMUM_RATE * bound, 1e-9)


def _retime(path, duration, label, vlim, *, clock=None, protected=(), minimum_duration=None, accept_clock=None):
    return core.retime(path, duration=float(duration), label=label, vlim=vlim,
                       clock=clock, alim=None, protected=protected,
                       maximum_rate=MAXIMUM_RATE, minimum_duration=minimum_duration,
                       accept_clock=accept_clock)


def _native_rate_acceptance(limit, timestep):
    def accept(path, clock):
        intervals = BernsteinRateBounds(path, clock).partition(limit, timestep)
        return all(float(row[2].max()) <= limit for row in intervals)
    return accept


def _protected_phase(name, duration):
    if name in FIXED_PHASES:
        return ((0.0, float(duration)),)
    if name.startswith('pre_mix_clearance_'):
        # Preload begins at this named phase entry; never scale its0.6s ramp.
        return ((0.0, min(float(duration), .6)),)
    return ()


def _command_bounds(m, path, joint_ids, minimum, label):
    coefficients = core.as_polynomial(path).c.reshape(-1, len(joint_ids))
    lower, upper = m.jnt_range[joint_ids, 0], m.jnt_range[joint_ids, 1]
    margin = float(np.minimum(coefficients - lower, upper - coefficients).min())
    acts = np.array([int(np.flatnonzero(m.actuator_trnid[:, 0] == joint)[0]) for joint in joint_ids])
    limited = m.actuator_ctrllimited[acts].astype(bool)
    control_margin = (float(np.minimum(coefficients[:, limited] - m.actuator_ctrlrange[acts[limited], 0],
                                       m.actuator_ctrlrange[acts[limited], 1] - coefficients[:, limited]).min())
                      if limited.any() else None)
    if not np.isfinite(coefficients).all() or margin < minimum or (control_margin is not None and control_margin < 0.0):
        raise RuntimeError(f'TOPP LEFT {label} original command-range guard rejected: {(margin, control_margin)}')
    return margin, control_margin


def retime_cartesian(m, planned, name, duration, phase_plan):
    """Retain the already constructed native IK spline and all its knots."""
    from t1_spincoat_scene import arm_indices
    original = planned['spline']
    timing = _retime(original, duration, 'left.' + name, 2.0,
                     protected=_protected_phase(name, duration),
                     minimum_duration=duration if name in phase_plan.get('phase_duration_overrides_s', {}) else None)
    times = native_times(m, timing.duration)
    source = np.asarray(source_elapsed(timing, times))
    result = dict(planned)
    report = dict(planned['report'])
    report['source_reference_report'] = dict(planned['report'])
    result.update(time=times, arm_commands=timing(times), spline=timing, _topp_timing=timing)
    if 'geometric_spline' in planned:
        result['geometric_arm_q'] = planned['geometric_spline'](source)
        result['geometric_spline'] = GeometricClock(planned['geometric_spline'], timing)
    else:
        offset = np.asarray(planned['arm_commands'][0]) - np.asarray(planned['geometric_arm_q'][0])
        result['geometric_arm_q'] = result['arm_commands'] - offset
    margin, control_margin = _command_bounds(m, original, arm_indices(m, 'left')[0], .049, name)
    report.update(duration_s=float(timing.duration), source_duration_s=float(duration),
                  sampled_1ms_command_peak_velocity_rad_s=float(np.abs(timing(times, 1)).max()),
                  sampled_1ms_command_joint_margin_rad=margin,
                  actuator_control_hull_margin_rad=control_margin,
                  command_boundary_velocity_rad_s=[timing(0., 1).tolist(), timing(timing.duration, 1).tolist()],
                  topp=timing.report, original_ik_knots_and_spacing_preserved=True)
    result['report'] = report
    return result


class GeometricClock:
    def __init__(self, curve, timing):
        self.curve, self.timing = curve, timing

    def __call__(self, elapsed, nu=0):
        time = np.clip(np.asarray(elapsed, dtype=float), 0., self.timing.duration)
        source = self.timing.clock(time)
        if nu == 0:
            return self.curve(source)
        velocity = np.asarray(self.timing.clock(time, 1))[..., None]
        if nu == 1:
            return self.curve(source, 1) * velocity
        if nu == 2:
            return (self.curve(source, 2) * velocity ** 2
                    + self.curve(source, 1) * np.asarray(self.timing.clock(time, 2))[..., None])
        raise ValueError('Only position, velocity and acceleration are geometric reference outputs')


def phase_timing(m, name, duration, q0, goalarm, h0, goalhand, phase_plan, *, hand_ids=None, safe_ready=False):
    """Exact native chord and safe-ready polynomial; R03 stays source-pinned."""
    if goalarm is None and goalhand is None:
        return None
    curves = []
    if goalarm is not None:
        curves.append(core.quintic(q0, goalarm, duration))
    if goalhand is not None:
        hand = core.quintic(h0, goalhand, duration)
        if safe_ready and name == 'return_to_original_ready_joints':
            from v34_smoothtubereturn_bounds import restrict_curve
            if hand_ids is None:
                raise RuntimeError('TOPP safe-ready path requires native hand-joint ordering')
            breaks = np.array([0., .2 * duration, .8 * duration, duration])
            pieces = [restrict_curve(hand, float(a), float(b)).c.copy() for a, b in zip(breaks[:-1], breaks[1:])]
            spread = {'left_index_MCP_AA': -.06, 'left_middle_MCP_AA': 0., 'left_ring_MCP_AA': .04}
            for index, joint in enumerate(hand_ids):
                name_in_hand = m.joint(int(joint)).name.split('/')[-1]
                if name_in_hand in spread:
                    middle = spread[name_in_hand]
                    pieces[0][:, 0, index] = core.quintic(h0[index], middle, .2 * duration).c[:, 0, 0]
                    pieces[1][:, 0, index] = middle
                    pieces[2][:, 0, index] = core.quintic(middle, goalhand[index], .2 * duration).c[:, 0, 0]
            hand = BPoly(np.concatenate(pieces, axis=1), breaks)
        curves.append(hand)
    if len(curves) == 1:
        path = curves[0]
    else:
        from v34_smoothtubereturn_bounds import restrict_curve
        breaks = np.unique(np.concatenate([curve.x for curve in curves]))
        pieces = [np.concatenate([restrict_curve(curve, float(a), float(b)).c for curve in curves], axis=2)
                  for a, b in zip(breaks[:-1], breaks[1:])]
        path = BPoly(np.concatenate(pieces, axis=1), breaks)
    limits = _engineering_velocity(path)
    if goalarm is not None:
        limits[:len(q0)] = 2.0
    return _retime(path, duration, 'left.' + name, limits,
                   protected=_protected_phase(name, duration),
                   minimum_duration=duration if name in phase_plan.get('phase_duration_overrides_s', {}) else None)


def register_overlay(kind, overlay, m):
    key = (id(m), kind)
    entries = _OVERLAYS.setdefault(key, [])
    if not any(value is overlay for value in entries):
        entries.append(overlay)


def _unique_overlay(m, kind):
    values = _OVERLAYS.get((id(m), kind), ())
    if len(values) > 1:
        raise RuntimeError('TOPP LEFT requires an unambiguous native ' + kind + ' overlay')
    return values[0] if values else None


def _body_config(hook):
    seen = set()
    while hook is not None and id(hook) not in seen:
        seen.add(id(hook))
        config = getattr(hook, 'config', None)
        if config is not None and all(hasattr(config, name) for name in ('down_phase', 'up_phase', 'target_joint_rad', 'duration_s')):
            return config
        hook = getattr(hook, 'original', None)
    return None


def overlay_duration(m, d, phase, duration, hook, *, body_config=None, hand_command=None, hand_actuators=None):
    """Prepare a clock BEFORE its native caller chooses its step count."""
    require_native_tick(m)
    config = body_config or _body_config(hook)
    kind, start, goal = None, None, None
    if config is not None and phase in (config.down_phase, config.up_phase):
        if abs(float(config.duration_s) - float(duration)) > 1e-9:
            raise RuntimeError('TOPP torso overlay and caller disagree about original duration')
        kind = 'torso'
        qa = np.array([int(m.joint('astribot_torso_joint_' + str(k)).qposadr[0]) for k in range(1, 5)])
        start = np.asarray(d.qpos[qa]).copy()
        overlay = _unique_overlay(m, kind)
        if phase == config.down_phase:
            goal = np.asarray(config.target_joint_rad)
        elif overlay is not None and overlay.reference is not None:
            goal = np.asarray(overlay.reference)
        else:
            raise RuntimeError('TOPP torso rise has no native initial reference')
    elif phase.startswith('human_pour_idle_'):
        kind = 'idle'
        overlay = _unique_overlay(m, kind)
        if overlay is None:
            raise RuntimeError('TOPP idle duration requires the original instantiated finger overlay')
        if abs(float(overlay.duration) - float(duration)) > 1e-9:
            raise RuntimeError('TOPP idle overlay and caller disagree about original duration')
        if overlay.start is None:
            if phase != 'human_pour_idle_extend' or hand_command is None or hand_actuators is None:
                raise RuntimeError('TOPP first idle fold requires its original nominal hand command')
            by_actuator = dict(zip(np.asarray(hand_actuators, dtype=int), np.asarray(hand_command)))
            original_start = np.array([by_actuator[int(act)] for act in overlay.ids])
            extended = original_start.copy()
            for i, name in enumerate(overlay.names):
                if name.endswith('MCP_FE'):
                    extended[i] = -.15
                if name.endswith('MCP_AA'):
                    extended[i] = 0.
        else:
            original_start, extended = overlay.start, overlay.extend
        start = overlay.last.copy() if overlay.last is not None else original_start.copy()
        goal = (extended if phase in ('human_pour_idle_extend', 'human_pour_idle_restore_fold')
                else original_start if phase == 'human_pour_idle_restore_extend' else overlay.fold)
    if kind is None:
        return float(duration)
    path = core.quintic(start, goal, float(duration))
    timing = _retime(path, duration, 'left.overlay.' + phase, _engineering_velocity(path))
    _PENDING[(id(d), phase, kind)] = dict(timing=timing, start=np.asarray(start).copy(), goal=np.asarray(goal).copy())
    native_times(m, timing.duration)
    return float(timing.duration)


def overlay_progress(overlay, m, d, phase, elapsed, source_duration, kind):
    pending = _PENDING.get((id(d), phase, kind))
    if pending is None and kind == 'torso':
        pending = getattr(overlay, '_topp_active', None)
    if pending is None:
        raise RuntimeError('TOPP ' + kind + ' motion reached without synchronized caller duration: ' + phase)
    if getattr(overlay, '_topp_active', None) is not pending:
        goal = (overlay.target if overlay.mode == 'lowering' else overlay.reference) if kind == 'torso' else (
            overlay.extend if phase in ('human_pour_idle_extend', 'human_pour_idle_restore_fold')
            else overlay.start if phase == 'human_pour_idle_restore_extend' else overlay.fold)
        if (np.max(np.abs(overlay.anchor - pending['start'])) > 1e-10 or
                np.max(np.abs(np.asarray(goal) - pending['goal'])) > 1e-10 or
                abs(pending['timing'].source_duration - float(source_duration)) > 1e-9):
            raise RuntimeError('TOPP overlay changed its native start/goal/source duration: ' + phase)
        overlay._topp_active = pending
    return float(source_elapsed(pending['timing'], elapsed)) / float(source_duration)


def carry_reference(m, start, end, duration):
    timing = _retime(core.quintic(start, end, duration), duration, 'left.coarse_carry_to_receiver', 2.0)
    times = native_times(m, timing.duration)
    commands = timing(times)
    bounds = BernsteinRateBounds(timing.path, timing.clock)
    intervals = bounds.partition(2.0, require_native_tick(m))
    peak = max(float(row[2].max()) for row in intervals)
    if peak > 2.0:
        raise RuntimeError('TOPP carry original all-time2rad/s guard rejected')
    return timing, times, commands, peak


def retime_insertion(m, planned, duration):
    """Native solver runs once at original knot spacing; caller uses new duration."""
    result = retime_cartesian(m, planned, 'lower_into_holder', duration, {})
    from t1_spincoat_scene import arm_indices
    margin, _ = _command_bounds(m, planned['spline'], arm_indices(m, 'left')[0], .05, 'rack insertion')
    result['report'].update(command_margin_rad=margin,
                            command_peak_velocity_rad_s=float(np.abs(result['_topp_timing'](result['time'], 1)).max()),
                            conservative_contact_stroke_clock_preserved=True)
    return result




def speedpour_clock(previous, spatial, source_time, source_rates, fixed_source_intervals, peak_design, dt):
    native = previous(spatial, source_time, source_rates, fixed_source_intervals, peak_design, dt)
    old_clock, _, _, old_requested, old_durations, old_fixed = native
    context = _POUR_CONTEXT.get()
    if context is None:
        raise RuntimeError('TOPP SpeedPour clock reached outside its native plan context')
    # Native fixed-piece flags identify exact knots. Root-finding the same
    # boundary can introduce a one-ULP sliver and destroy its derivative jet.
    protected = tuple((float(old_clock.x[i]), float(old_clock.x[i + 1]))
                      for i in np.flatnonzero(old_fixed))
    timing = _retime(spatial, old_clock.x[-1], 'left.human_pour_' + context['segment'], peak_design,
                     clock=old_clock, protected=protected,
                     accept_clock=_native_rate_acceptance(peak_design, dt))
    bounds = BernsteinRateBounds(spatial, timing.clock)
    intervals = bounds.partition(peak_design, dt)
    if max(float(row[2].max()) for row in intervals) > peak_design:
        raise RuntimeError('TOPP SpeedPour original design-speed certificate rejected')
    source_edges = np.asarray(timing.source_time(timing.clock.x))
    frozen = np.zeros(len(source_edges) - 1, dtype=bool)
    for a, b in protected:
        frozen |= (source_edges[:-1] >= a - 1e-10) & (source_edges[1:] <= b + 1e-10)
    context.update(timing=timing, source_clock=dict(
        physical_knots_s=old_clock.x.tolist(), source_knots_s=old_clock(old_clock.x).tolist(),
        requested_piece_durations_s=old_requested.tolist(), certified_piece_durations_s=old_durations.tolist(),
        fixed_pieces=old_fixed.tolist(), protected_elapsed_intervals_s=[list(pair) for pair in protected]))
    return timing.clock, bounds, intervals, np.diff(source_edges), np.diff(timing.clock.x), frozen


def smooth_reference(reference, timestep, loaded):
    """Replace the clock before native guards, trace generation and sweep."""
    if abs(float(timestep) - .001) > 1e-12:
        raise RuntimeError('TOPP SmoothTubeReturn requires native1ms references')
    source_duration = float(reference.duration)
    source_joins = reference.join_times.copy()
    source_durations = reference.durations.copy()
    source_edges = np.r_[0., source_joins, source_duration]
    fixed = reference.fixed_durations.copy()
    if loaded:
        fixed[-1] = True  # Entire >=2s side-entry and incoming rack-rate limit.
        if len(source_durations) == 3:
            fixed[0] = True  # Native clearance minimum/explicit override.
    protected = tuple((float(source_edges[i]), float(source_edges[i + 1])) for i in np.flatnonzero(fixed))
    timing = _retime(reference.spatial, source_duration,
                     'left.smooth_loaded_return' if loaded else 'left.smooth_empty_ready',
                     2.0, clock=reference.clock, protected=protected,
                     accept_clock=_native_rate_acceptance(2.0, timestep))
    reference.topp_timing = timing
    reference.topp_source_join_times = source_joins
    reference.topp_source_durations = source_durations
    reference.topp_source_fixed_durations = reference.fixed_durations.copy()
    reference.clock = timing.clock
    if not isinstance(reference.clock, BPoly):
        raise TypeError('SmoothTubeReturn requires a real scalar BPoly clock')
    reference.duration = float(timing.duration)
    reference.join_times = np.array([timing.new_time_for_source(float(t)) for t in source_joins])
    reference.durations = np.diff(np.r_[0., reference.join_times, reference.duration])
    reference.fixed_durations = fixed
    reference.rate_bounds = BernsteinRateBounds(reference.spatial, reference.clock)
    reference.rate_intervals = reference.rate_bounds.partition(2.0, timestep)
    reference.joint_rate_upper = np.max(np.stack([row[2] for row in reference.rate_intervals]), axis=0)
    reference.rate_bound = float(reference.joint_rate_upper.max())
    if reference.rate_bound > 2.0:
        raise RuntimeError('TOPP SmoothTubeReturn original Bernstein-rate guard rejected')
    if np.any(reference.durations <= 0.) or np.any(reference.clock(reference.join_times, 1) <= 0.):
        raise RuntimeError('TOPP SmoothTubeReturn cannot stop at a traversed internal join')
    if not np.allclose(reference.clock(reference.join_times), reference.joins, rtol=0., atol=1e-10):
        raise RuntimeError('TOPP SmoothTubeReturn changed a spatial join')
    if loaded and reference.durations[-1] + 1e-9 < source_durations[-1]:
        raise RuntimeError('TOPP SmoothTubeReturn shortened the protected rack-side entry')
    if abs(round(reference.duration / timestep) * timestep - reference.duration) > 1e-9:
        raise RuntimeError('TOPP SmoothTubeReturn total duration is not on a native tick')


def ready_hand_progress(reference, time):
    source = float(source_elapsed(reference.topp_timing, time))
    join = float(reference.topp_source_join_times[0])
    return float(np.clip((source - join) / reference.topp_source_durations[1], 0., 1.))


class LeftObservations:
    """Coverage of reached real-time intervals, without altering sample()."""
    def __init__(self):
        self.phase = None
        self.seconds = 0.0
        self.mixing = False

    def observe(self, phase, dt, *, mix=False):
        if phase != self.phase or bool(mix) != self.mixing:
            self.finish()
            self.phase = phase
            self.mixing = bool(mix)
        self.seconds += float(dt)

    def finish(self):
        if self.phase is not None:
            reason = FIXED_PHASES.get(self.phase)
            if reason is None and self.mixing:
                reason = 'Original V27 frequency/amplitude/ramp/steady exposure and derivative-dependent prefilter; measured liquid observation uses actual elapsed time.'
            if reason is not None:
                core.note_fixed('left.' + self.phase, reason=reason, duration=self.seconds)
        self.phase = None
        self.seconds = 0.0
        self.mixing = False
