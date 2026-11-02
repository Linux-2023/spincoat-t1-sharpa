"""LEFT-only continuous return references; live controls remain in the native callback.

A single spatial reference and C2 path clock cross each selected safe join.
Planning MjData is isolated. No live pose, object, clock, model or gain writes.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.interpolate import BPoly, make_interp_spline
from scipy.spatial.transform import Rotation

from t1_coarse_fine_motion_v3 import (
    CoarseFineConfig, _distance_to_polyline, transport_duration,
)
from t1_empty_hand_ready_v7 import safe_ready_hand_target
from t1_spincoat_scene import arm_indices, hand_indices
from v34_smoothtubereturn_bounds import BernsteinRateBounds, as_bernstein, restrict_curve
from v34_smoothtubereturn_profile import PROFILE


CANDIDATE_ID = "SmoothTubeReturn"


def _smooth(u):
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _work(m, d):
    work = mujoco.MjData(m)
    work.qpos[:] = d.qpos
    mujoco.mj_kinematics(m, work)
    return work


def _curve(points, corner_scale):
    points = np.asarray(points, dtype=float)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths < 1e-9):
        raise ValueError("Continuous reference has duplicate spatial knots")
    arc = np.r_[0.0, np.cumsum(lengths)]
    directions = delta / lengths[:, None]
    tangent = np.empty_like(points)
    tangent[0], tangent[-1] = directions[0], directions[-1]
    for i in range(1, len(points) - 1):
        vector = directions[i - 1] + directions[i]
        norm = np.linalg.norm(vector)
        if norm < .1:
            raise RuntimeError("Reversal requires a stopped maneuver, not a blended join")
        tangent[i] = corner_scale * vector / norm
    curve = BPoly.from_derivatives(
        arc, [[q, v, np.zeros_like(q)] for q, v in zip(points, tangent)],
    )
    return arc, curve


def _bounded_cartesian_path(points):
    # Same 12mm corridor as native adaptive return, but never accept its zero-
    # tangent last resort: this candidate must actually traverse every join.
    for attempt, scale in enumerate((.35, .175, .0875)):
        arc, curve = _curve(points, scale)
        deviation = _distance_to_polyline(curve(np.linspace(0.0, arc[-1], 20001)), points)
        if float(deviation.max()) <= .0119:
            return arc, curve, dict(
                interior_tangent_scale=scale, halving_attempt=attempt,
                dense_selection_deviation_mm=float(deviation.max() * 1000),
                selection_samples=20001, original_corridor_limit_mm=12.0,
                zero_tangent_fallback_permitted=False, micro_corner_fallback_permitted=False,
                minimum_tangent_fraction_of_original=.25,
            )
    raise RuntimeError("No nonstopping C2 path satisfies the native return corridor")


def _clock(length, joins, durations, join_rate_limits):
    durations = np.asarray(durations, dtype=float)
    positions = np.r_[0.0, joins, length]
    if len(positions) != len(durations) + 1 or np.any(np.diff(positions) <= 0.0):
        raise ValueError("Traversed joins must be ordered strictly inside the path")
    secants = np.diff(positions) / durations
    rates = np.zeros(len(positions))
    for i, limit in enumerate(join_rate_limits, start=1):
        left, right = secants[i - 1], secants[i]
        rates[i] = min(2.0 * left * right / (left + right), 1.2 * min(left, right), limit)
        if rates[i] <= 0.0:
            raise ValueError("A continuous join requires a positive path rate")
    # Adjacent rates sum to <=2.4*the intervening secant, so quintic
    # Bernstein controls remain ordered even when both interval ends move.
    coefficients = np.stack((positions[:-1],
                             positions[:-1] + rates[:-1] * durations / 5.0,
                             positions[:-1] + 2.0 * rates[:-1] * durations / 5.0,
                             positions[1:] - 2.0 * rates[1:] * durations / 5.0,
                             positions[1:] - rates[1:] * durations / 5.0,
                             positions[1:]))
    return BPoly(coefficients, np.r_[0.0, np.cumsum(durations)])


class Reference:
    """Analytic composition q(s(t)); the join has only one q, qdot and qddot."""

    def __init__(self, spatial, length, joins, durations, timestep, *, join_rate_limits=None, fixed_durations=None):
        self.spatial = as_bernstein(spatial)
        self.length = float(length)
        self.joins = np.atleast_1d(np.asarray(joins, dtype=float))
        limits = np.full(len(self.joins), np.inf) if join_rate_limits is None else np.asarray(join_rate_limits, dtype=float)
        if limits.shape != self.joins.shape:
            raise ValueError("Each traversed join requires exactly one rate limit")
        requested = np.ceil(np.asarray(durations, dtype=float) / timestep) * timestep
        fixed = np.zeros(len(requested), dtype=bool) if fixed_durations is None else np.asarray(fixed_durations, dtype=bool)
        if fixed.shape != requested.shape:
            raise ValueError("Fixed-duration mask must match the phase clock")
        durations = requested.copy()
        for _ in range(4):
            if durations.sum() > 30.0:
                raise RuntimeError("Continuous reference exceeds native30s planning bound")
            clock = _clock(self.length, self.joins, durations, limits)
            rate_bounds = BernsteinRateBounds(self.spatial, clock)
            intervals = rate_bounds.partition(2.0, timestep)
            joint_upper = np.max(np.stack([row[2] for row in intervals]), axis=0)
            rate_bound = float(joint_upper.max())
            if rate_bound <= 2.0:
                break
            if fixed.all():
                raise RuntimeError("Fixed phase timing cannot certify the original2rad/s bound")
            durations[~fixed] = np.ceil(durations[~fixed] * (rate_bound / 2.0) * (1.0 + 1e-8) / timestep) * timestep
        else:
            raise RuntimeError("Unresolved all-time2rad/s certification; no clipping or fixed-duration extension")
        self.clock = clock
        self.durations = durations
        self.requested_durations = requested
        self.fixed_durations = fixed
        self.duration = float(durations.sum())
        self.join_times = np.cumsum(durations)[:-1]
        self.rate_bound = rate_bound
        self.joint_rate_upper = joint_upper
        self.rate_bounds = rate_bounds
        self.rate_intervals = intervals

    def __call__(self, t, derivative=0):
        s = self.clock(t)
        if derivative == 0:
            return self.spatial(s)
        v = self.clock(t, 1)
        if derivative == 1:
            return self.spatial(s, 1) * np.asarray(v)[..., None]
        if derivative == 2:
            return (self.spatial(s, 2) * np.asarray(v)[..., None] ** 2
                    + self.spatial(s, 1) * np.asarray(self.clock(t, 2))[..., None])
        raise ValueError("Only position, velocity and acceleration are reference outputs")


def _command_guards(m, start, reference):
    ids, _, _ = arm_indices(m, 'left')
    acts = np.array([int(np.flatnonzero(m.actuator_trnid[:, 0] == j)[0]) for j in ids])
    times = np.arange(round(reference.duration / m.opt.timestep) + 1) * m.opt.timestep
    commands = reference(times)
    velocities, accelerations = reference(times, 1), reference(times, 2)
    if not all(np.isfinite(v).all() for v in (commands, velocities, accelerations)):
        raise RuntimeError("Nonfinite continuous reference")
    margin = float(np.minimum(commands - m.jnt_range[ids, 0], m.jnt_range[ids, 1] - commands).min())
    limited = m.actuator_ctrllimited[acts].astype(bool)
    control_margin = float(np.minimum(
        commands[:, limited] - m.actuator_ctrlrange[acts[limited], 0],
        m.actuator_ctrlrange[acts[limited], 1] - commands[:, limited],
    ).min()) if limited.any() else None
    if margin < .049 or float(np.abs(velocities).max()) > 2.0 + 1e-9:
        raise RuntimeError("Native .049rad joint-margin/2rad/s command guard rejected continuous reference")
    if control_margin is not None and control_margin < 0.0:
        raise RuntimeError("Native actuator-control range rejected continuous reference")
    jump = float(np.max(np.abs(commands[0] - start)))
    if jump > 1e-10:
        raise RuntimeError("Continuous reference would jump from the incoming actuator command")
    if max(float(np.abs(reference(t, order)).max())
           for t in (0.0, reference.duration) for order in (1, 2)) > 1e-8:
        raise RuntimeError("Hard-stop endpoint derivatives are not zero")
    seams = []
    for join_time in reference.join_times:
        seams.append(dict(time_s=float(join_time),
                          position_rad=reference(join_time).tolist(),
                          velocity_rad_s=reference(join_time, 1).tolist(),
                          acceleration_rad_s2=reference(join_time, 2).tolist(),
                          one_shared_analytic_boundary=True))
        if np.linalg.norm(reference(join_time, 1)) < 1e-8:
            raise RuntimeError("Selected safe join accidentally retains a full stop")
    report = dict(
        candidate_id=CANDIDATE_ID, duration_s=reference.duration,
        phase_durations_s=reference.durations.tolist(),
        maximum_command_speed_rad_s=float(np.abs(velocities).max()),
        conservative_all_time_speed_bound_rad_s=reference.rate_bound,
        all_time_rate_bound_method='Interval-local Bernstein hulls of Qprime(s(I)) and clock derivative on the same time interval; no sampled-speed acceptance',
        all_time_rate_interval_count=len(reference.rate_intervals),
        all_time_joint_rate_bounds_rad_s=reference.joint_rate_upper.tolist(),
        requested_phase_durations_s=reference.requested_durations.tolist(),
        fixed_phase_durations=reference.fixed_durations.tolist(),
        maximum_command_acceleration_rad_s2=float(np.abs(accelerations).max()),
        command_joint_margin_rad=margin, actuator_control_margin_rad=control_margin,
        initial_command_jump_rad=jump, shared_boundaries=seams,
        hard_endpoints_velocity_and_acceleration_zero=True,
        native_timestep_s=float(m.opt.timestep), command_clipping=False,
        reference_semantics='One spatial joint curve composed with one monotone C2 path clock; not recording interpolation',
    )
    trace = dict(time=times, arm_commands=commands,
                 reference_velocity_rad_s=velocities, reference_acceleration_rad_s2=accelerations)
    trace['certified_rate_intervals_s_and_joint_bounds'] = np.asarray([[a, b, *bounds] for a, b, bounds in reference.rate_intervals])
    return report, trace


def _ik_path(m, work, qa, start, parameters, target, solver):
    q = np.asarray(start).copy()
    knots = [q.copy()]
    errors = []
    for value in parameters[1:]:
        point, rotation = target(float(value))
        q, error = solver(m, work, point, rotation, side='left', seed=q)
        if error['position_error_mm'] > .1 or error['orientation_error_deg'] > .1:
            raise RuntimeError("Original 0.1mm/0.1degree geometric IK guard failed")
        work.qpos[qa] = q
        mujoco.mj_kinematics(m, work)
        knots.append(q.copy())
        errors.append(error)
    return np.asarray(knots), dict(
        maximum_knot_position_error_mm=max(e['position_error_mm'] for e in errors),
        maximum_knot_rotation_error_deg=max(e['orientation_error_deg'] for e in errors),
    )


def plan_loaded_return(m, d, handles, arm_command, clear_lift, return_points, return_R, *, ik_solver, audit_directory, minimum_clearance_duration_s=0.0, side_entry_duration_s=2.0):
    """One measured grasp/servo offset at the preceding mandatory pour boundary."""
    from v34_smoothtubereturn_geometry import ReturnGeometryAudit

    _, qa, arm_dofs = arm_indices(m, 'left')
    pb = m.body('sharpa_left/left_hand_C_MC').id
    tb = m.body(handles['tube_body']).id
    tq = int(m.body(tb).jntadr[0])
    tq = int(m.jnt_qposadr[tq])
    work = _work(m, d)
    initial_P, initial_R = work.xpos[pb].copy(), work.xmat[pb].reshape(3, 3).copy()
    initial_q = work.qpos[qa].copy()
    offset = np.asarray(arm_command) - initial_q
    rel = initial_R.T @ (work.xpos[tb] - initial_P)
    E = initial_R.T @ work.xmat[tb].reshape(3, 3)
    geometry_audit = ReturnGeometryAudit(m, work, pb, handles['tube_body'], tq, rel, E,
                                       entry_time_s=float(d.time), audit_directory=audit_directory)
    with geometry_audit.preserve_failure():
        geometry_audit.observe(work, reference_time_s=0.0, origin='entry')
        if not np.isfinite(clear_lift) or clear_lift < 0.0:
            raise ValueError("Invalid native clearance lift")
        points = [initial_P]
        lifted = clear_lift > 1e-6
        if lifted:
            points.append(initial_P + np.array([0.0, 0.0, clear_lift]))
        for point in np.asarray(return_points):
            if np.linalg.norm(point - points[-1]) < .0005:
                raise RuntimeError("Redundant return waypoint requires explicit native review, not silent deletion")
            points.append(np.asarray(point).copy())
        points = np.asarray(points)
        arc, cartesian, corner_review = _bounded_cartesian_path(points)
        lift_join = float(arc[1]) if lifted else None
        rotation_start = lift_join if lifted else 0.0
        rack_join = float(arc[-2])
        rotation_vector = Rotation.from_matrix(np.asarray(return_R) @ initial_R.T).as_rotvec()
    
        def target(s):
            progress = (0.0 if s <= rotation_start else 1.0 if s >= rack_join
                        else _smooth((s - rotation_start) / (rack_join - rotation_start)))
            return cartesian(s), Rotation.from_rotvec(rotation_vector * progress).as_matrix() @ initial_R
    
        # Include all route knots exactly, with <=2mm intervening Cartesian arc.
        parameters = np.unique(np.r_[np.linspace(0.0, arc[-1], max(7, int(math.ceil(arc[-1] / .002)) + 1)), arc])
        geometric, ik_report = _ik_path(m, work, qa, initial_q, parameters, target, ik_solver)
        spatial = make_interp_spline(parameters, geometric + offset, k=5, axis=0)
        remaining = rack_join - rotation_start
        coarse_duration = transport_duration(remaining, np.linalg.norm(rotation_vector), config=CoarseFineConfig())
        joins = [lift_join, rack_join] if lifted else [rack_join]
        if not 0.0 <= minimum_clearance_duration_s <= 30.0 or not 2.0 <= side_entry_duration_s <= 30.0:
            raise ValueError("Invalid original phase duration override")
        durations = ([max(.5, clear_lift / .075, minimum_clearance_duration_s), coarse_duration, side_entry_duration_s]
                     if lifted else [coarse_duration, side_entry_duration_s])
        # The rack-side entry keeps its original >=2s duration, but crosses its
        # incoming endpoint slowly rather than stopping: arc rate<=10mm/s.
        limits = [np.inf, .010] if lifted else [.010]
        reference = Reference(spatial, arc[-1], joins, durations, m.opt.timestep, join_rate_limits=limits)
        report, trace = _command_guards(m, np.asarray(arm_command), reference)
        work = _work(m, d)
        planned_positions, max_position_error, max_rotation_error = [], 0.0, 0.0
        for time, command in zip(trace['time'], trace['arm_commands']):
            work.qpos[qa] = command - offset
            mujoco.mj_kinematics(m, work)
            s = float(reference.clock(time))
            expected_P, expected_R = target(s)
            position_error = float(np.linalg.norm(work.xpos[pb] - expected_P))
            rotation_error = float(np.degrees(Rotation.from_matrix(work.xmat[pb].reshape(3, 3) @ expected_R.T).magnitude()))
            max_position_error = max(max_position_error, position_error)
            max_rotation_error = max(max_rotation_error, rotation_error)
            if position_error > .0001 or rotation_error > .1:
                raise RuntimeError("Continuous loaded FK interpolation violates original geometric pose bounds")
            geometry_audit.observe(work, reference_time_s=time, origin='dense_reference')
            planned_positions.append(work.xpos[pb].copy())
        deviation = float(_distance_to_polyline(np.asarray(planned_positions), points).max())
        if deviation > .012:
            raise RuntimeError("Continuous loaded FK path leaves native 12mm corridor")
        from v34_smoothtubereturn_sweep import certify_loaded_environment_sweep
        sweep_report = certify_loaded_environment_sweep(m, d, reference, offset, handles['tube_body'], pb, rel, geometry_audit=geometry_audit)
        rack_time = float(reference.join_times[-1])
        rack_path_speed = float(np.linalg.norm(cartesian(rack_join, 1)) * reference.clock(rack_time, 1))
        work.qpos[qa] = reference(rack_time) - offset
        mujoco.mj_kinematics(m, work)
        mujoco.mj_comPos(m, work)
        jacobian = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, work, jacobian, None, pb)
        rack_palm_speed = float(np.linalg.norm(jacobian[:, arm_dofs] @ reference(rack_time, 1)))
        if max(rack_path_speed, rack_palm_speed) > .010 + 1e-10:
            raise RuntimeError("Rack side-entry join exceeds declared10mm/s target/FK reference speed")
        grip_audit_report = geometry_audit.save_completed()
        trace['geometric_arm_q'] = trace['arm_commands'] - offset
        trace['target_palm_world_m'] = np.asarray(planned_positions)
        report.update(ik_report)
        report.update(
            scope='Clearance lift, free-return corridor and low-speed final rack side-entry; stop before unchanged actual alignment decision/insertion',
            planned_palm_waypoints_world_m=points.tolist(), corner_review=corner_review,
            corner_deviation_max_mm=deviation * 1000,
            dense_fk_max_position_error_mm=max_position_error * 1000,
            dense_fk_max_orientation_error_deg=max_rotation_error,
            preserved_servo_offset_rad=offset.tolist(),
            frozen_actual_hand_to_tube_position_m=rel.tolist(),
            frozen_actual_hand_to_tube_rotation=E.tolist(),
            source_state_time_s=float(d.time), actual_grasp_reanchors_inside_span=0,
            planning_native_samples=len(trace['time']),
            planning_geometry_scope='Native environment/crossdigit rejection plus conservative continuous environment sweep; original helper held-tube nonpad/overwall outputs are a separate rejected/not-certified audit, not an original post-pour functional gate',
            grip_geometry_audit=grip_audit_report,
            entry_grip_geometry_audit_artifact='return_entry_grip_geometry_audit.json',
            grip_geometry_audit_artifact='return_grip_geometry_audit.json',
            enabled_native_environment_sweep=sweep_report,
            rack_entry_join_time_s=rack_time, rack_entry_join_palm_speed_m_s=rack_palm_speed,
            rack_entry_join_target_path_speed_m_s=rack_path_speed,
            rack_entry_arc_rate_limit_m_s=.010, original_rack_entry_minimum_duration_s=2.0,
            native_clearance_duration_override_s=minimum_clearance_duration_s,
            native_side_entry_duration_s=side_entry_duration_s,
            final_palm_target_world_m=np.asarray(return_points[-1]).tolist(),
            final_palm_rotation_world=np.asarray(return_R).tolist(),
            physics_success_not_inferred=True,
        )
        phases = (['raise_upright_clear_of_rack_before_return'] if lifted else [])
        phases += ['coarse_return_free_corridor', 'return_clearance_waypoint_' + str(len(return_points) - 1)]
        return dict(reference=reference, report=report, trace=trace, loaded=True, phases=phases)


def _empty_contacts(m, work):
    bad = []
    for contact in work.contact:
        if contact.dist >= 0.0:
            continue
        names = [m.body(int(m.geom_bodyid[int(g)])).name for g in contact.geom]
        left = any(n.startswith(('sharpa_left/', 'astribot_arm_left_')) for n in names)
        environment = any(n.startswith(('prep/', 'lab_bench', 'stand/', 'spincoater/', 'aspiration_source', 'pipette/')) for n in names)
        digits = [next((digit for digit in ('thumb', 'index', 'middle', 'ring', 'pinky')
                        if '/left_' + digit + '_' in name), None) for name in names]
        if (left and environment) or (all(v is not None for v in digits) and digits[0] != digits[1]):
            bad.append(dict(geoms=[m.geom(int(g)).name for g in contact.geom], depth_mm=-float(contact.dist) * 1000))
    return bad


def _quintic_jerk_coefficients(p0, p1, v0, v1, a0, a1):
    """Coefficients of d3q/du3 for normalized quintic endpoint p/v/a data."""
    r0 = p1 - p0 - v0 - .5 * a0
    r1 = v1 - v0 - a0
    r2 = a1 - a0
    return np.stack((6.0 * (10.0 * r0 - 4.0 * r1 + .5 * r2),
                     24.0 * (-15.0 * r0 + 7.0 * r1 - r2),
                     60.0 * (6.0 * r0 - 3.0 * r1 + .5 * r2)))


def _empty_join_derivatives(normal, linear, direction, width):
    """Exact two-variable minimum-jerk solve, constrained to the ready chord.

    x=width*qprime_parallel and y=width**2*qsecond_parallel. The outgoing
    quintic has scalar Bernstein controls [0,x/5,2*x/5+y/20,3*w/5,4*w/5,w].
    Their ordering is equivalent to x>=0, 4*x+y>=0, 8*x+y<=12*w. Requiring
    x>=w/4 additionally excludes a nearly stopped join as a geometry policy.
    The positive-definite quadratic has its minimum in the interior, on one
    face, or at an intersection of two faces; enumerate those seven cases.
    This solves for reference derivatives, not clipped execution commands.
    """
    gradient = linear @ direction
    free = -np.linalg.solve(normal, gradient)
    constraints = np.array([[1., 0.], [4., 1.], [-8., -1.]])
    targets = np.array([.25 * width, 0., -12. * width])
    feasible = []
    for active in ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)):
        value = free.copy()
        if active:
            rows = constraints[list(active)]
            inverse_rows = np.linalg.solve(normal, rows.T)
            value += inverse_rows @ np.linalg.solve(rows @ inverse_rows,
                                                    targets[list(active)] - rows @ free)
        padding = 256.0 * np.finfo(float).eps * max(1.0, width, float(np.abs(value).max()))
        if np.isfinite(value).all() and np.min(constraints @ value - targets) >= -padding:
            objective = float(.5 * value @ normal @ value + gradient @ value)
            feasible.append((objective, value, active))
    if not feasible:
        raise RuntimeError('No resolved chord-constrained minimum-jerk derivatives')
    _, value, active = min(feasible, key=lambda row: row[0])
    x, y = value
    controls = np.array([0., x / 5., 2. * x / 5. + y / 20.,
                         3. * width / 5., 4. * width / 5., width])
    return direction * (x / width), direction * (y / width ** 2), dict(
        shared_derivative_subspace='Original ready-chord direction; no transverse velocity or acceleration',
        scalar_spatial_join_rate=float(x / width), scalar_spatial_join_acceleration=float(y / width ** 2),
        minimum_spatial_join_rate_policy=.25, active_derivative_constraints=list(active),
        outgoing_scalar_bernstein_controls=controls.tolist(),
        constrained_objective='Exact integrated squared spatial jerk of both quintics, with fixed endpoint PVA and shared chord-parallel join VA',
    )


def _empty_bridge_progress_bound(bridge):
    """Certify finite forward progress across the broad bridge, not just its join."""
    direction = bridge(bridge.x[-1]) - bridge(bridge.x[0])
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    direction /= norm
    derivative = bridge.derivative()
    padding = 128.0 * len(derivative.c) * np.finfo(float).eps * max(1.0, float(np.abs(derivative.c).max()))
    # Restriction is bound-only: the two executed quintics never acquire extra
    # knots. Subdivide each bridge piece, never the unrelated straight tail.
    for subdivisions in (1, 2, 4, 8, 16, 32):
        lower = float('inf')
        for start, end in zip(bridge.x[:-1], bridge.x[1:]):
            cuts = np.linspace(start, end, subdivisions + 1)
            for a, b in zip(cuts[:-1], cuts[1:]):
                coefficients = restrict_curve(derivative, float(a), float(b)).c.reshape(-1, 7)
                bound = float(np.nextafter((coefficients @ direction).min() - padding, -np.inf))
                lower = min(lower, bound)
        if lower >= .125:
            return dict(all_time_forward_spatial_rate_lower_bound=lower,
                        minimum_forward_spatial_rate_policy=.125,
                        forward_direction=direction.tolist(), subdivisions_per_bridge_piece=subdivisions)
    return None


def _empty_clearance_certificate(m, d, qa, palm, spatial, offset, P0, P1, join):
    """All-time FK corridor enclosure on isolated data, not sampled acceptance.

    For every arm joint i, reach[i] bounds distance from its axis to the palm
    origin in every configuration. Thus ||FK(q)-FK(qmid)|| <= sum_i reach[i]
    *abs(q[i]-qmid[i]). Bernstein hulls enclose every q on each restricted
    polynomial piece. Distance to the closed Cartesian segment is 1-Lipschitz,
    so the anchor distance plus that displacement encloses the whole interval.
    """
    from v34_smoothtubereturn_sweep import _axis_reach_bound

    ids, _, _ = arm_indices(m, 'left')
    if np.any(m.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise RuntimeError('Empty FK enclosure requires the original seven revolute LEFT joints')
    # The inherited reach bound handles arbitrary hinge anchors, but not a
    # translating descendant or multiple joint axes in the same body frame.
    for joint in ids:
        body = int(palm)
        joint_body = int(m.jnt_bodyid[joint])
        while True:
            if body == 0:
                raise RuntimeError('Empty FK enclosure requires every LEFT joint to be a palm ancestor')
            first, count = int(m.body_jntadr[body]), int(m.body_jntnum[body])
            if count > 1 or (count and m.jnt_type[first] != mujoco.mjtJoint.mjJNT_HINGE):
                raise RuntimeError('Unsupported descendant joint in empty FK reach enclosure')
            if body == joint_body:
                break
            body = int(m.body_parentid[body])
    reaches = np.array([_axis_reach_bound(m, int(palm), int(j), 0.0) for j in ids])
    clearance = restrict_curve(spatial, 0.0, join)
    scale = np.maximum(1.0, np.abs(clearance.c).max(axis=(0, 1)))
    coefficient_padding = 128.0 * len(clearance.c) * np.finfo(float).eps * scale
    position_padding = 512.0 * np.finfo(float).eps * max(
        1.0, float(np.abs(np.r_[P0, P1]).max()), float(reaches.sum()))
    pending = [(float(a), float(b), 0) for a, b in zip(clearance.x[:-1], clearance.x[1:])]
    work = _work(m, d)
    segment = np.array([P0, P1])
    leaves = anchors = deepest = 0
    maximum_bound = maximum_anchor = 0.0
    while pending:
        a, b, depth = pending.pop()
        middle = .5 * (a + b)
        center = clearance(middle)
        coefficients = restrict_curve(clearance, a, b).c.reshape(-1, 7)
        joint_radius = np.nextafter(np.abs(coefficients - center).max(axis=0) + coefficient_padding, np.inf)
        displacement = float(np.nextafter(reaches @ joint_radius + position_padding, np.inf))
        work.qpos[qa] = center - offset
        mujoco.mj_kinematics(m, work)
        deviation = float(_distance_to_polyline(work.xpos[palm][None, :], segment)[0])
        bound = float(np.nextafter(deviation + displacement + position_padding, np.inf))
        anchors += 1
        maximum_anchor = max(maximum_anchor, deviation)
        if not np.isfinite(bound) or deviation > .012:
            return dict(certified=False, reason='Nonfinite enclosure or an FK anchor outside original12mm corridor',
                        interval=[a, b], anchor_deviation_mm=deviation * 1000.)
        if bound <= .012:
            leaves += 1
            maximum_bound = max(maximum_bound, bound)
            continue
        if depth >= 24 or not a < middle < b:
            return dict(certified=False, reason='Unresolved all-time12mm FK enclosure; no sampled-pass fallback',
                        interval=[a, b], upper_bound_mm=bound * 1000.)
        pending.extend(((a, middle, depth + 1), (middle, b, depth + 1)))
        deepest = max(deepest, depth + 1)
    return dict(
        certified=True, method='Per-piece Bernstein joint hull plus configuration-independent joint-axis reach times joint radius around exact FK anchors',
        covered_spatial_interval=[0.0, float(join)], certified_intervals=leaves,
        fk_anchor_count=anchors, deepest_subdivision=deepest,
        maximum_all_time_cartesian_corridor_bound_mm=maximum_bound * 1000.,
        maximum_fk_anchor_deviation_mm=maximum_anchor * 1000.,
        joint_axis_palm_reach_bounds_m=reaches.tolist(),
        original_cartesian_corridor_mm=12., bound_only_subdivision=True,
        assumptions='Original rigid hinge kinematics and fixed planning non-arm pose; constant entry servo offset on clearance',
        exclusions='Not actual tracking, collision, finger or environmental motion certification; unchanged native1ms checks and callback still follow',
    )


def _smooth_empty_spatial(m, d, qa, palm, commands, ready_arm, offset, P0, P1):
    """C4 clearance plus a broad, chord-constrained minimum-jerk C2 bridge."""
    lengths = np.linalg.norm(np.diff(commands, axis=0), axis=1)
    if np.any(lengths < 1e-9):
        raise RuntimeError('Duplicate clearance IK knots require explicit planning review')
    arc = np.r_[0.0, np.cumsum(lengths)]
    clearance = as_bernstein(make_interp_spline(arc, commands, k=5, axis=0))
    join = float(arc[-1])
    ready_chord = np.vstack((commands[-1], np.asarray(ready_arm)))
    ready_delta = ready_chord[1] - ready_chord[0]
    ready_length = float(np.linalg.norm(ready_delta))
    if ready_length < 1e-8:
        raise RuntimeError('Ready target already reached; no traversed join')
    direction = ready_delta / ready_length
    incoming = clearance(join, 1)
    if np.linalg.norm(incoming) < 1e-9 or np.linalg.norm(incoming / np.linalg.norm(incoming) + direction) < .1:
        raise RuntimeError('Reversal requires a stopped maneuver, not a blended join')
    length = join + ready_length
    zero = np.zeros(7)
    # J(u)=j0+j1*u+j2*u^2; same exact squared-jerk objective as interval04.
    gram = np.kron(np.eye(2), 1.0 / (1.0 + np.arange(3)[:, None] + np.arange(3)[None, :]))
    influence = np.array([[-24., 3.], [168., -24.], [-180., 30.],
                          [-36., -9.], [192., 36.], [-180., -30.]])
    normal = influence.T @ gram @ influence
    work = _work(m, d)
    rejected = []
    for fraction in (.5, .4, .32, .25):
        # Turn over at least a quarter of the short leg, not a terminal IK cell.
        # Constrain shared derivatives instead of shrinking a transverse ready
        # overshoot into a tiny corner. Arm and fingers share the final clock.
        width = fraction * min(join, ready_length)
        before, after = join - width, join + width
        start = clearance(before)
        velocity = clearance(before, 1)
        acceleration = clearance(before, 2)
        end = commands[-1] + width * direction
        baseline = np.vstack((_quintic_jerk_coefficients(start, commands[-1], velocity * width, zero, acceleration * width ** 2, zero),
                              _quintic_jerk_coefficients(commands[-1], end, zero, direction * width, zero, zero)))
        join_velocity, join_acceleration, derivative_review = _empty_join_derivatives(
            normal, influence.T @ gram @ baseline, direction, width)
        prefix = restrict_curve(clearance, 0.0, before)
        bridge = BPoly.from_derivatives([before, join, after],
                                       [[start, velocity, acceleration],
                                        [commands[-1], join_velocity, join_acceleration],
                                        [end, direction, zero]])
        progress_review = _empty_bridge_progress_bound(bridge)
        if progress_review is None:
            rejected.append(dict(fraction=fraction, reason='Broad bridge forward-progress enclosure unresolved'))
            continue
        tail = BPoly.from_derivatives([after, length],
                                     [[end, direction, zero], [ready_arm, direction, zero]])
        spatial = BPoly(np.concatenate((prefix.c, bridge.c, tail.c), axis=1),
                        np.r_[prefix.x, bridge.x[1:], tail.x[1:]])
        ready_part = restrict_curve(spatial, join, length)
        # Chord-parallel shared VA and ordered scalar controls put the ENTIRE
        # outgoing piece on the original chord. Keep the original all-time
        # joint-corridor guard on its actual Bernstein hull, including the tail.
        # No global uniform grid can hide or underresolve a short curved piece.
        coefficients = ready_part.c.reshape(-1, 7)
        padding = 128.0 * len(ready_part.c) * np.finfo(float).eps * max(1.0, float(np.abs(coefficients).max()))
        ready_hull_deviation = float(np.nextafter(_distance_to_polyline(coefficients, ready_chord).max() + padding, np.inf))
        if not np.isfinite(ready_hull_deviation) or ready_hull_deviation > .012:
            rejected.append(dict(fraction=fraction, reason='Original all-time.012rad ready-chord guard rejected',
                                 ready_hull_bound_rad=ready_hull_deviation))
            continue
        clearance_deviation = 0.0
        for value in np.unique(np.r_[np.linspace(0.0, join, 201), before, join]):
            work.qpos[qa] = spatial(value) - offset
            mujoco.mj_kinematics(m, work)
            clearance_deviation = max(clearance_deviation, float(_distance_to_polyline(work.xpos[palm][None, :], np.array([P0, P1]))[0]))
        if clearance_deviation > .012:
            rejected.append(dict(fraction=fraction, reason='Original12mm FK sample guard rejected',
                                 clearance_deviation_mm=clearance_deviation * 1000.))
            continue
        clearance_certificate = _empty_clearance_certificate(m, d, qa, palm, spatial, offset, P0, P1, join)
        if not clearance_certificate['certified']:
            rejected.append(dict(fraction=fraction, clearance_certificate=clearance_certificate))
            continue
        return spatial, length, join, ready_chord, dict(
            method='Smooth quintic clearance interpolation, chord-constrained minimum-spatial-jerk PVA bridge through exact clearance target, monotone original ready chord',
            blend_parameter_start=before, blend_parameter_join=join, blend_parameter_end=after,
            balanced_blend_width=width, short_leg_fraction=fraction,
            minimum_geometry_policy_fraction=.25, executed_bridge_polynomial_pieces=2,
            incoming_bridge_joint_chord_length_rad=float(np.linalg.norm(commands[-1] - start)),
            outgoing_bridge_joint_chord_length_rad=width,
            shared_spatial_velocity=join_velocity.tolist(), shared_spatial_acceleration=join_acceleration.tolist(),
            constrained_derivatives=derivative_review, bridge_forward_progress=progress_review,
            ready_joint_chord_all_time_control_hull_bound_rad=ready_hull_deviation,
            ready_corridor_proof_subdivisions=1, clearance_all_time_certificate=clearance_certificate,
            selection_clearance_fk_deviation_mm=clearance_deviation * 1000,
            rejected_broad_candidates=rejected,
            forced_zero_second_derivative_at_ik_samples=False,
            scope='Geometry design and all-time planned corridor enclosures, not measured motion acceptance. Native1ms corridor/collision/command checks still follow; safe-ready finger ordering uses the same retimed ready leg.',
        )
    raise RuntimeError('No certified broad chord-constrained C2 empty corner in the declared geometry family; '
                       'not a proof of geometric impossibility, and no micro-corner, stopped or sampled-pass fallback: ' + str(rejected))


def plan_empty_ready(m, d, arm_command, clearance_goal, ready_arm, hand_command, ready_hand, *, ik_solver):
    """Preserve native clearance, exact targets and safe-hand normalized ordering."""
    _, qa, _ = arm_indices(m, 'left')
    hj, hq, _ = hand_indices(m, 'left')
    pb = m.body('sharpa_left/left_hand_C_MC').id
    work = _work(m, d)
    P0, R0 = work.xpos[pb].copy(), work.xmat[pb].reshape(3, 3).copy()
    actual = work.qpos[qa].copy()
    offset = np.asarray(arm_command) - actual
    work.qpos[qa] = np.asarray(clearance_goal)
    mujoco.mj_kinematics(m, work)
    P1, R1 = work.xpos[pb].copy(), work.xmat[pb].reshape(3, 3).copy()
    rotation_vector = Rotation.from_matrix(R1 @ R0.T).as_rotvec()
    parameters = np.linspace(0.0, 1.0, 51)
    work = _work(m, d)

    def target(u):
        return P0 + (P1 - P0) * u, Rotation.from_rotvec(rotation_vector * u).as_matrix() @ R0

    geometric, ik_report = _ik_path(m, work, qa, actual, parameters, target, ik_solver)
    commands = geometric + offset
    spatial, length, join, ready_chord, corner_review = _smooth_empty_spatial(m, d, qa, pb, commands, ready_arm, offset, P0, P1)
    # Three seconds was a nominal movement duration, not a physical gate.
    # Every planning and live finger consumer uses reference.durations[1].
    reference = Reference(spatial, length, join, [PROFILE['empty_clearance_s'], PROFILE['ready_s']], m.opt.timestep)
    report, trace = _command_guards(m, np.asarray(arm_command), reference)
    work = _work(m, d)
    max_clearance_deviation = 0.0
    max_ready_joint_deviation = 0.0
    geometric_trace = []
    for time, command in zip(trace['time'], trace['arm_commands']):
        s = float(reference.clock(time))
        offset_progress = 0.0 if s <= join else _smooth((s - join) / (length - join))
        work.qpos[qa] = command - offset * (1.0 - offset_progress)
        hand_progress = 0.0 if time <= reference.join_times[0] else (time - reference.join_times[0]) / reference.durations[1]
        work.qpos[hq] = (np.asarray(hand_command) if hand_progress == 0.0 else
                         safe_ready_hand_target(m, hj, hand_command, ready_hand, hand_progress))
        mujoco.mj_kinematics(m, work)
        if s <= join:
            deviation = float(_distance_to_polyline(work.xpos[pb][None, :], np.array([P0, P1]))[0])
            max_clearance_deviation = max(max_clearance_deviation, deviation)
            if deviation > .012:
                raise RuntimeError("Blended empty clearance leaves the conservative 12mm native corridor")
        else:
            deviation = float(_distance_to_polyline(command[None, :], ready_chord)[0])
            max_ready_joint_deviation = max(max_ready_joint_deviation, deviation)
            if deviation > .012:
                raise RuntimeError("Ready corner departs more than .012rad from original joint chord")
        mujoco.mj_comPos(m, work)
        mujoco.mj_collision(m, work)
        bad = _empty_contacts(m, work)
        if bad:
            raise RuntimeError("Continuous empty-hand native collision screen failed at " + str(float(time)) + ': ' + str(bad[:3]))
        geometric_trace.append(work.qpos[qa].copy())
    if np.max(np.abs(trace['arm_commands'][-1] - ready_arm)) > 1e-10:
        raise RuntimeError("Exact original ready joint target was not preserved")
    trace['geometric_arm_q'] = np.asarray(geometric_trace)
    report.update(ik_report)
    report.update(
        scope='Already-withdrawn, restored empty-hand clearance through exact original ready joints; safe finger ordering and original native collision screens retained',
        final_arm_target_rad=np.asarray(ready_arm).tolist(),
        final_hand_target_rad=np.asarray(ready_hand).tolist(),
        clearance_palm_target_world_m=P1.tolist(),
        clearance_palm_target_rotation_world=R1.tolist(),
        original_safe_ready_hand_function='t1_empty_hand_ready_v7.safe_ready_hand_target',
        original_ready_hand_duration_s=3.0, executed_ready_hand_duration_s=float(reference.durations[1]),
        aggressive_profile=dict(PROFILE), ready_finger_order_and_endpoints_preserved=True,
        clearance_fk_deviation_max_mm=max_clearance_deviation * 1000,
        ready_joint_chord_deviation_max_rad=max_ready_joint_deviation,
        empty_spatial_corner=corner_review,
        preserved_entry_servo_offset_rad=offset.tolist(),
        offset_scope='One entry offset carried through clearance, C2-decayed on the ready leg so final command remains exactly the original ready target',
        actual_pose_reanchors_inside_span=0, planning_native_samples=len(trace['time']),
        planning_geometry_scope='Native enabled collision screen at every physics-reference sample with planned native safe-ready finger targets; not CCD/full visual or actual tracking certification',
        physics_success_not_inferred=True,
    )
    return dict(reference=reference, report=report, trace=trace, loaded=False,
                hj=hj, hand_start=np.asarray(hand_command).copy(), hand_goal=np.asarray(ready_hand).copy())


def advance_loaded(m, d, planned, step_callback, events):
    reference = planned['reference']
    previous_phase = None
    start = float(d.time)
    for time, command in zip(planned['trace']['time'][1:], planned['trace']['arm_commands'][1:]):
        phase_index = int(np.searchsorted(reference.join_times, time, side='left'))
        phase = planned['phases'][phase_index]
        if phase != previous_phase:
            events.append(dict(time_s=float(d.time), phase=phase, continuous_reference=CANDIDATE_ID))
            previous_phase = phase
        before = float(d.time)
        step_callback(command, phase)
        if abs(float(d.time) - before - m.opt.timestep) > 1e-9:
            raise RuntimeError("Native callback must advance exactly one unchanged physics step")
    planned['report'].update(execution_start_s=start, execution_end_s=float(d.time), execution_completed=True)
    return planned['trace']['arm_commands'][-1].copy(), planned['report'], planned['trace']


def advance_empty(m, d, planned, step_callback, events):
    reference = planned['reference']
    previous_phase = None
    start = float(d.time)
    hand = planned['hand_start'].copy()
    for time, command in zip(planned['trace']['time'][1:], planned['trace']['arm_commands'][1:]):
        phase = 'empty_hand_clearance_before_ready' if time <= reference.join_times[0] else 'return_to_original_ready_joints'
        if phase != previous_phase:
            events.append(dict(time_s=float(d.time), phase=phase, continuous_reference=CANDIDATE_ID))
            previous_phase = phase
        if time > reference.join_times[0]:
            progress = (time - reference.join_times[0]) / reference.durations[1]
            hand = safe_ready_hand_target(m, planned['hj'], planned['hand_start'], planned['hand_goal'], progress)
        before = float(d.time)
        step_callback(command, hand, phase)
        if abs(float(d.time) - before - m.opt.timestep) > 1e-9:
            raise RuntimeError("Native callback must advance exactly one unchanged physics step")
    planned['report'].update(execution_start_s=start, execution_end_s=float(d.time), execution_completed=True)
    return planned['trace']['arm_commands'][-1].copy(), hand.copy(), planned['report'], planned['trace']
