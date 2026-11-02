"""Tail-flattened dry-pour clocks; original liquid callback remains authoritative.

The previous rejected variants multiplied the already-fast middle of recovery.
This variant spends its acceleration on slow source-time tails, caps the design
joint rate, and joins every clock piece in PVA. The native dense cubic is the
same C2 interpolant the original planner used to check command speed. Native
geometry/IK planners run first; every new command is additionally screened.
"""
from functools import wraps

import mujoco
import numpy as np
from scipy.interpolate import BPoly, CubicSpline, PchipInterpolator

from t1_spincoat_scene import arm_indices
from v34_smoothtubereturn_bounds import BernsteinRateBounds
from v34_speedpour_profile import PROFILE

CANDIDATE_ID = 'SpeedPour'
PATCHED_SYMBOLS = ('t1_human_pour_v4_joint_reference.plan_segment',
                   't1_actual_frame_return_v7.plan_return')
_INSTALLED = False


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _planned_tilt(m, d, plan, commands, source_time):
    _, qa, _ = arm_indices(m, 'left')
    work = mujoco.MjData(m)
    work.qpos[:] = d.qpos
    palm = m.body('sharpa_left/left_hand_C_MC').id
    relative = np.asarray(plan['measured_hand_to_tube_R'])
    selected = np.unique(np.r_[0, np.arange(10, len(source_time), 10), len(source_time) - 1])
    angles = []
    offset = np.asarray(plan['preserved_servo_offset_rad']) if plan['segment'] == 'return' else 0.0
    for tick in selected:
        work.qpos[qa] = commands[tick] - offset
        mujoco.mj_kinematics(m, work)
        tube_R = work.xmat[palm].reshape(3, 3) @ relative
        angles.append(np.degrees(np.arccos(np.clip(tube_R[2, 2], -1.0, 1.0))))
    return np.asarray(PchipInterpolator(source_time[selected], angles)(source_time))


def _clock(spatial, source_time, source_rates, fixed_source_intervals, peak_design, dt):
    ticks = len(source_time) - 1
    stride = max(1, round(PROFILE['clock_knot_source_s'] / dt))
    cuts = {0, ticks, *range(stride, ticks, stride)}
    for a, b in fixed_source_intervals:
        cuts.update((a, b))
    indices = np.asarray(sorted(cuts), dtype=int)
    coordinates = source_time[indices]
    inverse = 1.0 / source_rates
    accumulated = np.r_[0.0, np.cumsum(np.diff(source_time) * (inverse[:-1] + inverse[1:]) * .5)]
    durations = np.ceil(np.diff(accumulated[indices]) / dt - 1e-10) * dt
    fixed = np.zeros(len(durations), dtype=bool)
    for a, b in fixed_source_intervals:
        fixed |= (indices[:-1] >= a) & (indices[1:] <= b)
    durations[fixed] = np.diff(coordinates)[fixed]
    requested = durations.copy()
    for _ in range(8):
        secants = np.diff(coordinates) / durations
        rates = np.r_[min(1.0, secants[0]),
                      np.minimum(2.0 * secants[:-1] * secants[1:] / (secants[:-1] + secants[1:]),
                                 1.2 * np.minimum(secants[:-1], secants[1:])),
                      min(1.0, secants[-1])]
        # Frozen source intervals are exactly affine, including both endpoint
        # derivatives. Thus active flow and native recovery boundary response
        # are not accidentally sped up by normalized-clock rescaling.
        for index in np.flatnonzero(fixed):
            rates[index:index + 2] = 1.0
        times = np.r_[0.0, np.cumsum(durations)]
        curve = BPoly.from_derivatives(times, [[s, v, 0.0] for s, v in zip(coordinates, rates)])
        if np.min(curve.derivative().c) < -1e-10:
            raise RuntimeError('SpeedPour clock is not certified monotone; no stopped fallback')
        bounds = BernsteinRateBounds(spatial, curve)
        intervals = bounds.partition(peak_design, dt)
        peak = max(float(row[2].max()) for row in intervals)
        if peak <= peak_design:
            return curve, bounds, intervals, requested, durations, fixed
        if fixed.all() or durations.sum() > 30.0:
            raise RuntimeError('SpeedPour design-rate certificate unresolved in bounded timing family')
        durations[~fixed] = np.ceil(durations[~fixed] * (peak / peak_design) * (1.0 + 1e-8) / dt) * dt
    raise RuntimeError('SpeedPour unresolved all-time rate proof; no sampled-speed acceptance')


def _geometry_guards(m, d, plan, commands):
    _, qa, _ = arm_indices(m, 'left')
    work = mujoco.MjData(m)
    work.qpos[:] = d.qpos
    mujoco.mj_kinematics(m, work)
    palm = m.body('sharpa_left/left_hand_C_MC').id
    elbow = m.body('astribot_arm_left_link_5').id
    wrist = m.body('astribot_arm_left_link_6').id
    middle = m.body('sharpa_left/left_middle_MCP_VL').id
    returning = plan['segment'] == 'return'
    offset = np.asarray(plan['preserved_servo_offset_rad']) if returning else 0.0
    if returning:
        tube = m.body('prep/reagent_tube').id
        mouth = m.site('prep/tube_mouth').id
        hand_R = work.xmat[palm].reshape(3, 3)
        root_local = hand_R.T @ (work.xpos[tube] - work.xpos[palm])
        mouth_local = np.asarray(plan['measured_mouth_in_hand_m'])
        lower = np.array([.32, .10, min(.89, float(work.site_xpos[mouth, 2]) - .001)])
        upper = np.array([.75, .65, 1.16])
    peak_wrist = 0.0
    for command in commands:
        work.qpos[qa] = command - offset
        mujoco.mj_kinematics(m, work)
        u = work.xpos[wrist] - work.xpos[elbow]
        v = work.xpos[middle] - work.xpos[palm]
        angle = float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1.0, 1.0))))
        peak_wrist = max(peak_wrist, angle)
        if angle > 50.0 + (1e-5 if returning else 1e-6):
            raise RuntimeError('SpeedPour retimed native interpolant violates original wrist guard')
        if returning:
            H = work.xmat[palm].reshape(3, 3)
            M = work.xpos[palm] + H @ mouth_local
            root = work.xpos[palm] + H @ root_local
            if np.min(np.r_[M - lower, upper - M, root[2] - .82]) < -1e-5:
                raise RuntimeError('SpeedPour retimed native interpolant violates original mouth/root corridor')
    return peak_wrist


def _retime(m, d, plan, entry):
    dt = float(m.opt.timestep)
    if abs(dt - .001) > 1e-12:
        raise RuntimeError('SpeedPour requires original1ms callback timing')
    source_time = np.r_[0.0, np.asarray(plan['time'])]
    original = np.vstack((entry, np.asarray(plan['arm_commands'])))
    native = CubicSpline(source_time, original, axis=0,
                         bc_type=((1, np.zeros(7)), (1, np.zeros(7))))
    spatial = BPoly.from_power_basis(native)
    source_speed = np.max(np.abs(native(source_time, 1)), axis=1)
    tilt = _planned_tilt(m, d, plan, original, source_time)
    segment = plan['segment']
    peak_design = float(PROFILE['tilt_design_peak_rad_s' if segment == 'tilt' else 'return_design_peak_rad_s'])
    if segment == 'tilt':
        low, high = PROFILE['preflow_transition_deg']
        weight = 1.0 - _smooth((tilt - low) / (high - low))
        fast = 1.0 + (PROFILE['preflow_rate'] - 1.0) * weight
        reached = np.flatnonzero(tilt >= high)
        if not len(reached):
            raise RuntimeError('SpeedPour requires an explicit native-rate preflow/active boundary')
        slow_index = int(reached[0])
        fast[slow_index:] = 1.0
        fixed = [(slow_index, len(source_time) - 1)]
    else:
        fast = np.full(len(source_time), PROFILE['return_rate'])
        boundary = min(round(PROFILE['return_boundary_source_s'] / dt), (len(source_time) - 1) // 4)
        fast[:boundary + 1] = fast[-boundary - 1:] = 1.0
        fixed = [(0, boundary), (len(source_time) - 1 - boundary, len(source_time) - 1)]
    rates = np.maximum(1.0, np.minimum(fast, peak_design / np.maximum(source_speed, 1e-12)))
    clock, bounder, intervals, requested, durations, frozen = _clock(
        spatial, source_time, rates, fixed, peak_design, dt)
    duration = float(clock.x[-1])
    times = np.arange(1, round(duration / dt) + 1) * dt
    source_samples = clock(np.minimum(times, duration))
    source_samples[-1] = source_time[-1]
    commands = spatial(source_samples)
    ids, _, _ = arm_indices(m, 'left')
    hull = spatial.c.reshape(-1, 7)
    margin = float(np.minimum(hull - m.jnt_range[ids, 0], m.jnt_range[ids, 1] - hull).min())
    acts = np.array([np.flatnonzero(m.actuator_trnid[:, 0] == j)[0] for j in ids])
    limited = m.actuator_ctrllimited[acts].astype(bool)
    control_margin = float(np.minimum(hull[:, limited] - m.actuator_ctrlrange[acts[limited], 0],
                                      m.actuator_ctrlrange[acts[limited], 1] - hull[:, limited]).min()) if limited.any() else None
    peak = max(float(row[2].max()) for row in intervals)
    minimum = .05 - 1e-6 if segment == 'tilt' else .05
    if not np.isfinite(commands).all() or peak > 2.0 or margin < minimum or (control_margin is not None and control_margin < 0.0):
        raise RuntimeError('SpeedPour original command bound rejected ' + str((peak, margin, control_margin)))
    if np.max(np.abs(commands[-1] - original[-1])) > 1e-10 or np.max(np.abs(spatial(clock(0.0)) - entry)) > 1e-10:
        raise RuntimeError('SpeedPour changed an original command endpoint')
    wrist = _geometry_guards(m, d, plan, commands)
    coordinate = (PchipInterpolator(source_time, np.r_[0.0, plan['angle_deg']])(source_samples)
                  if segment == 'tilt' else None)
    result = dict(plan)
    result.update(time=times, arm_commands=commands, angle_deg=coordinate,
                  duration_s=duration, peak_joint_speed_rad_s=peak,
                  min_joint_margin_rad=margin, time_law='native_dense_C2_reference_with_tail_flattened_C2_clock')
    if 'v34_return_offset_aware' in result:
        result['v34_return_offset_aware'] = dict(result['v34_return_offset_aware'])
        result['v34_return_offset_aware']['command_min_margin_by_joint_rad'] = np.minimum(
            commands - m.jnt_range[ids, 0], m.jnt_range[ids, 1] - commands).min(axis=0).tolist()
    result['speedpour_retiming'] = dict(
        profile=dict(PROFILE), original_duration_s=float(plan['duration_s']), duration_s=duration,
        clock_source_knots_s=clock(clock.x).tolist(), clock_physical_knots_s=clock.x.tolist(),
        requested_clock_piece_durations_s=requested.tolist(), certified_clock_piece_durations_s=durations.tolist(),
        native_rate_fixed_clock_pieces=frozen.tolist(),
        conservative_all_time_speed_bound_rad_s=peak, design_peak_rad_s=peak_design,
        certificate='Interval-local Bernstein Qprime(s(I))*sdot(I); exact same executed polynomials',
        certified_intervals=len(intervals), source_reference='Original dense CubicSpline with original zero endpoint velocity',
        maximum_retained_wrist_deg=wrist, all_time_command_hull_margin_rad=margin,
        actuator_control_hull_margin_rad=control_margin,
        recovery_boundary_source_seconds=PROFILE['return_boundary_source_s'],
        previous_geometry_ik_and_offset_guards_completed=True,
        original_stop_dose_drain_and_retreat_unchanged=True,
        no_live_state_writes=True, grip_or_physical_success_not_inferred=True)
    return result


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import t1_human_pour_v4_joint_reference as reference
    import t1_actual_frame_return_v7 as actual_return
    previous_reference = reference.plan_segment
    previous_return = actual_return.plan_return

    @wraps(previous_reference)
    def plan_segment(m, d, profile, *, segment, entry_arm_command, duration_s):
        plan = previous_reference(m, d, profile, segment=segment,
                                  entry_arm_command=entry_arm_command, duration_s=duration_s)
        if segment != 'tilt' or profile.get('geometric_path_kind') != 'joint_reference_curve':
            return plan
        return _retime(m, d, plan, entry_arm_command)

    @wraps(previous_return)
    def plan_return(m, d, profile, *, entry_arm_command, duration_s=10.0):
        plan = previous_return(m, d, profile, entry_arm_command=entry_arm_command, duration_s=duration_s)
        return _retime(m, d, plan, entry_arm_command)

    reference.plan_segment = plan_segment
    actual_return.plan_return = plan_return
    _INSTALLED = True
