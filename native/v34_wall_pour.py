"""Slow, noncontact wall/rim approach on native18's ACTIVE joint pour branch.

Install after every native18 restoration. The installed joint planner and its
SpeedPour clock run first, once, on the current episode's freshly retargeted
profile. Their times and orientation reference are retained. A separate explicit
1x quintic blends the *lower inner lip*, not the tube-mouth center, toward an
inside-rim point at/after nominal coordinate 80 through 96. Source-relative IK changes
only the late tilt translation; the original prefix and hand policy are retained.
The original return dispatcher retains its external guards and full servo offset,
with a 0.1mm extra mouth-floor planning reserve to cover joint interpolation.
Rack alignment preserves its original FK goal and reserves the native 0.0501rad
IK margin before the unchanged descent; skipping alignment also requires that
margin. Its added screen rejects geometric regressions and records inherited
native rejections explicitly, never as a clean-geometry or strict-grip pass.
No successful episode/trajectory is loaded by this module.

The rolled outer lip is kept above the highest enabled receiver collider. The
whole held tube/hand is additionally screened by the unchanged native helper on
private workData. These are conditional planning checks, not actual grip, liquid,
continuous collision, visual-geometry or physical-success certificates. Original
ledger termination may interrupt this approach; it is never delayed for the pose.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from scipy.interpolate import BPoly, CubicSpline
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

PATCHED_SYMBOLS = (
    't1_actual_frame_return_v7.advance_human_pour_reanchored_return_v7',
    't1_human_pour_v4_joint_reference.plan_segment',
    't1_actual_frame_return_v7.plan_return',
    't1_v10_v27_stage_v28.adapt_left_stage',
)
_INSTALLED = False
_CONTEXT = ContextVar('v34_wall_pour_context', default=None)
START_COORDINATE = 80.0
END_COORDINATE = 96.0
MINIMUM_APPROACH_S = 1.5
INNER_WALL_INSET_M = 0.003
DESIGN_RIM_CLEARANCE_M = 0.004
MINIMUM_RIM_CLEARANCE_M = 0.002
POSITION_TOLERANCE_M = 0.0001
ORIENTATION_TOLERANCE_DEG = 0.01
RACK_COMMAND_MARGIN_RAD = 0.0501


def _finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Wall pour requires finite ' + name + ' of shape ' + str(shape))
    return result


def receiver_geometry(m, d, handles, profile):
    """Read existing handles/config and compiled geometry; never alter either.

    Rolled-lip sections are the exact engineering profile in
    t1_glass_testtube_preparation._add_glass_tube, relative to tube_mouth.
    Both those visual sections and the compiled wall-box corner radius are
    enclosed. Receiver dimensions are checked against all 64 actual wall boxes.
    """
    import mujoco

    c = handles['config']
    receiver_name = handles['receiver_body']
    tube_name = handles['tube_body']
    mouth_name = handles['tube_mouth_site']
    receiver = m.body(receiver_name).id
    tube = m.body(tube_name).id
    mouth = m.site(mouth_name).id
    rotation = _finite(d.xmat[receiver].reshape(3, 3), (3, 3), 'receiver rotation')
    if not np.allclose(rotation, np.eye(3), rtol=0., atol=1e-10):
        raise ValueError('Wall pour retains the parent world-aligned receiver contract')
    if int(m.site_bodyid[mouth]) != tube:
        raise ValueError('Wall pour tube-mouth site is not on the held tube')
    origin = _finite(d.xpos[receiver], (3,), 'receiver origin').copy()
    radius = float(c['tube_outer_radius_m'])
    wall = float(c['tube_wall_thickness_m'])
    length = float(c['tube_length_m'])
    inner = float(c['receiver_inner_radius_m'])
    height = float(c['beaker_height_m'])
    prefix = str(c['prefix'])
    if not (0. < wall < radius < length and 0. < INNER_WALL_INSET_M < inner):
        raise ValueError('Invalid existing tube/receiver dimensional contract')
    if not np.allclose(m.site_pos[mouth], [0., 0., length], rtol=0., atol=1e-9):
        raise ValueError('Unexpected tube-mouth local frame')
    if not np.allclose(profile['receiver_xy_m'], origin[:2], rtol=0., atol=1e-8):
        raise ValueError('Wall pour requires the already receiver-relative live profile')
    receiver_top = float('-inf')
    for k in range(64):
        g = m.geom(prefix + 'griffin_wall_' + str(k)).id
        if int(m.geom_bodyid[g]) != receiver or int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError('Unexpected existing receiver wall geometry')
        actual_inner = np.linalg.norm(m.geom_pos[g, :2]) - m.geom_size[g, 0]
        if abs(float(actual_inner) - inner) > 1e-9:
            raise ValueError('Receiver config disagrees with compiled inner wall')
        expected_top = height - (0.004 if k in (15, 16, 17) else 0.)
        if abs(float(m.geom_pos[g, 2] + m.geom_size[g, 2]) - expected_top) > 1e-9:
            raise ValueError('Receiver config disagrees with compiled wall height')
    # Include the raised spout, not merely the circular wall/rim height.
    for g in np.flatnonzero(np.asarray(m.geom_bodyid) == receiver):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        local_R = Rotation.from_quat(np.roll(m.geom_quat[g], -1)).as_matrix()
        size = np.asarray(m.geom_size[g])
        kind = int(m.geom_type[g])
        nz = abs(float(local_R[2, 2]))
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent = float(np.abs(local_R[2]) @ size)
        elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            extent = float(size[0] * np.sqrt(max(0., 1. - nz * nz)) + size[1] * nz)
        elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            extent = float(size[0] + size[1] * nz)
        else:
            raise ValueError('Unclassified enabled receiver collider: ' + str(m.geom(int(g)).name))
        receiver_top = max(receiver_top, float(m.geom_pos[g, 2]) + extent)
    if not np.isfinite(receiver_top) or receiver_top < height:
        raise ValueError('Missing receiver solid-rim envelope')
    sections = [
        [radius, -0.001], [radius + 0.00045, -0.00055],
        [radius + 0.00025, 0.], [radius - wall - 0.0001, 0.],
        [radius - wall, -0.001],
    ]
    collision_radius = 0.
    for k in range(48):
        g = m.geom(tube_name + '_wall_' + str(k)).id
        if int(m.geom_bodyid[g]) != tube or int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError('Unexpected held-tube wall geometry')
        outer = float(np.linalg.norm(m.geom_pos[g, :2]) + m.geom_size[g, 0])
        if abs(outer - radius) > 1e-9:
            raise ValueError('Tube config disagrees with compiled outer wall')
        if abs(float(m.geom_pos[g, 2] + m.geom_size[g, 2]) - length) > 1e-9:
            raise ValueError('Tube mouth plane disagrees with compiled wall top')
        collision_radius = max(collision_radius, float(np.hypot(outer, m.geom_size[g, 1])))
    # A bounding circle also covers the wall boxes' corners at their top plane.
    sections.append([collision_radius, 0.])
    inner_lip_radius = radius - wall - 0.0001
    if inner_lip_radius <= 0.:
        raise ValueError('Invalid inner outlet radius')
    offset = _finite(profile['mouth_xy_offset_m'], (2,), 'receiver-relative mouth offset')
    norm = float(np.linalg.norm(offset))
    if norm < 1e-8:
        raise ValueError('No unambiguous parent receiver-wall direction')
    direction = offset / norm
    rim = origin + np.r_[inner * direction, height]
    # For any tilt, inner-low minus outer-low <= |r-ri|+|zoffset|.
    # This bounds every listed rolled-rim section, not only a sampled azimuth.
    envelope_allowance = max(abs(r - inner_lip_radius) + abs(z) for r, z in sections)
    goal = origin + np.r_[
        (inner - INNER_WALL_INSET_M) * direction,
        receiver_top + DESIGN_RIM_CLEARANCE_M + envelope_allowance,
    ]
    return dict(
        receiver_body=receiver_name, tube_body=tube_name, tube_mouth_site=mouth_name,
        tube_joint=handles['tube_joint'], receiver_origin_world_m=origin.tolist(),
        receiver_rotation_world=rotation.tolist(), receiver_inner_radius_m=inner,
        receiver_rim_height_m=height, receiver_rim_world_z_m=float(origin[2] + height),
        receiver_solid_top_world_z_m=float(origin[2] + receiver_top),
        intended_wall_direction_world_xy=direction.tolist(),
        intended_inner_wall_rim_world_m=rim.tolist(), desired_inner_lip_world_m=goal.tolist(),
        tube_length_m=length, tube_outer_radius_m=radius,
        tube_inner_lip_radius_m=inner_lip_radius, tube_collision_top_radius_m=collision_radius,
        tube_rolled_lip_sections_radius_axial_m=sections,
        full_lip_envelope_allowance_m=envelope_allowance,
        inner_wall_inset_m=INNER_WALL_INSET_M,
        design_clearance_above_all_receiver_solids_m=DESIGN_RIM_CLEARANCE_M,
        minimum_required_planned_lip_clearance_m=MINIMUM_RIM_CLEARANCE_M,
        geometry_source='Actual handles/config, receiver pose, 64 receiver and 48 tube wall boxes; original rolled visual-lip profile; raised spout included',
        scope='Privileged existing simulation geometry for LEFT pouring, not a visual localization claim',
    )


def lip_geometry(mouth_world_m, tube_R_world, geometry):
    """Lowest liquid outlet and conservative complete rolled-lip clearance.

    The lower lip is the downward projection into the tilted mouth plane.
    Vertical clearance is a sufficient noncontact bound for the whole lip above
    every receiver solid. Full-tube/hand clearance is a separate native check.
    """
    mouth = _finite(mouth_world_m, (3,), 'mouth center')
    rotation = _finite(tube_R_world, (3, 3), 'tube rotation')
    axis = rotation[:, 2]
    sine = float(np.linalg.norm(axis[:2]))
    if sine < 1e-8:
        # Upright has no unique pouring edge. This deterministic in-plane edge
        # is only for inspection; the wall approach requires a tilted mouth.
        down = -rotation[:, 0]
    else:
        down = (axis[2] * axis - np.array([0., 0., 1.])) / sine
    lip = mouth + geometry['tube_inner_lip_radius_m'] * down
    lower_offset = min(z * axis[2] - r * sine
                       for r, z in geometry['tube_rolled_lip_sections_radius_axial_m'])
    origin = np.asarray(geometry['receiver_origin_world_m'])
    radial = float(np.linalg.norm(lip[:2] - origin[:2]))
    return dict(
        inner_lip_world_m=lip.tolist(),
        inner_lip_distance_to_intended_rim_m=float(np.linalg.norm(
            lip - np.asarray(geometry['intended_inner_wall_rim_world_m']))),
        inner_lip_radial_inset_m=float(geometry['receiver_inner_radius_m'] - radial),
        complete_lip_lowest_world_z_m=float(mouth[2] + lower_offset),
        complete_lip_vertical_clearance_m=float(
            mouth[2] + lower_offset - geometry['receiver_solid_top_world_z_m']),
        tube_tilt_deg=float(np.degrees(np.arccos(np.clip(axis[2], -1., 1.)))),
    )


def wall_mouth_target(source_mouth_world_m, tube_R_world, progress, geometry):
    """Blend the lower lip toward the inside-rim target; preserve orientation."""
    progress = float(progress)
    if not np.isfinite(progress) or not 0. <= progress <= 1.:
        raise ValueError('Wall approach progress must lie in [0,1]')
    source = np.asarray(source_mouth_world_m, dtype=float)
    source_lip = np.asarray(lip_geometry(source, tube_R_world, geometry)['inner_lip_world_m'])
    return source + progress * (np.asarray(geometry['desired_inner_lip_world_m']) - source_lip)


def approach_reference(duration_s):
    """Explicit original-rate C2 0..1 approach clock, independent of fast labels."""
    import v34_topp_core as core

    duration = float(duration_s)
    if not np.isfinite(duration) or duration < MINIMUM_APPROACH_S:
        raise RuntimeError('Native tilt provides less than 1.5s for the slow wall approach; no accelerated or center-pour fallback')
    return core.retime(core.quintic(np.array([0.]), np.array([1.]), duration),
                       duration=duration, label='left.wall_pour_approach', factor=1.0)


def _rack_command_bounds(m):
    from t1_spincoat_scene import arm_indices

    ids, qa, _ = arm_indices(m, 'left')
    lower = np.asarray(m.jnt_range[ids, 0]) + RACK_COMMAND_MARGIN_RAD
    upper = np.asarray(m.jnt_range[ids, 1]) - RACK_COMMAND_MARGIN_RAD
    acts = np.array([np.flatnonzero(m.actuator_trnid[:, 0] == joint)[0] for joint in ids])
    limited = m.actuator_ctrllimited[acts].astype(bool)
    lower[limited] = np.maximum(lower[limited], m.actuator_ctrlrange[acts[limited], 0])
    upper[limited] = np.minimum(upper[limited], m.actuator_ctrlrange[acts[limited], 1])
    lower, upper = np.nextafter(lower, np.inf), np.nextafter(upper, -np.inf)
    if np.any(lower >= upper):
        raise RuntimeError('No original-range rack alignment interval')
    return ids, qa, lower, upper


def _rack_alignment_ready(m, command):
    command = _finite(command, (7,), 'rack alignment command')
    _, _, lower, upper = _rack_command_bounds(m)
    return bool(np.all(command >= lower) and np.all(command <= upper))


def _rack_alignment_goal(m, d, handles, command):
    """Preserve the original FK goal while reserving the native IK margin.

    This is an endpoint for the existing smooth alignment reference, not an
    immediate control write. Only the optimizer seed is projected into bounds;
    the solved command is never clipped and all live state remains untouched.
    """
    import mujoco
    import v34_uncapped_clock as clock
    from t1_left_standing_workflow_v6 import _native

    command = _finite(command, (7,), 'original rack alignment goal').copy()
    ids, qa, lower, upper = _rack_command_bounds(m)
    if np.all(command >= lower) and np.all(command <= upper):
        return command
    palm = m.body('sharpa_left/left_hand_C_MC').id
    work = mujoco.MjData(m)
    work.qpos[:] = d.qpos
    work.qpos[qa] = command
    mujoco.mj_kinematics(m, work)
    target = work.xpos[palm].copy()
    rotation = work.xmat[palm].reshape(3, 3).copy()
    seed = np.clip(command, lower + 1e-9, upper - 1e-9)
    lower_seed = .003 / (seed - lower + .05)
    upper_seed = .003 / (upper - seed + .05)

    def residual(q):
        work.qpos[qa] = q
        mujoco.mj_kinematics(m, work)
        orientation = Rotation.from_matrix(work.xmat[palm].reshape(3, 3) @ rotation.T).as_rotvec()
        return np.r_[3000. * (work.xpos[palm] - target), 300. * orientation,
                     .05 * (q - seed),
                     .003 / (q - lower + .05) - lower_seed,
                     .003 / (upper - q + .05) - upper_seed]

    fit = least_squares(residual, seed, bounds=(lower, upper), max_nfev=180,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not fit.success:
        raise RuntimeError('Rack alignment reserve IK did not converge: ' + fit.message)
    q = _finite(fit.x, (7,), 'reserved rack alignment command')
    work.qpos[qa] = q
    mujoco.mj_kinematics(m, work)
    pe = float(np.linalg.norm(work.xpos[palm] - target))
    oe = float(np.degrees(Rotation.from_matrix(work.xmat[palm].reshape(3, 3) @ rotation.T).magnitude()))
    if pe > POSITION_TOLERANCE_M or oe > ORIENTATION_TOLERANCE_DEG or not _rack_alignment_ready(m, q):
        raise RuntimeError('Rack alignment reserve failed original pose/range requirements: ' + str((pe, oe)))
    tube = m.body(handles['tube_body']).id
    H = d.xmat[palm].reshape(3, 3)
    rel = H.T @ (d.xpos[tube] - d.xpos[palm])
    E = H.T @ d.xmat[tube].reshape(3, 3)
    tq = int(m.joint(handles['tube_joint']).qposadr[0])
    bad = _native(m, work, palm, handles['tube_body'], tq, rel, E)
    work.qpos[:] = d.qpos
    work.qpos[qa] = command
    mujoco.mj_kinematics(m, work)
    original_bad = _native(m, work, palm, handles['tube_body'], tq, rel, E)

    def rejection_metrics(rows):
        metrics = {}
        for row in rows:
            key = (row['kind'], tuple(sorted(row['geoms'])))
            count, depth = metrics.get(key, (0, 0.))
            metrics[key] = (count + 1, max(depth, float(row['depth_mm'])))
        return metrics

    # This rack phase retains its original live acceptance, not the pour-only
    # zero-rejection screen. Preserve inherited rejections explicitly and reject
    # any new/deeper geometry from the refinement; 1e-9 mm is roundoff only.
    previous = rejection_metrics(original_bad)
    for key, (count, depth) in rejection_metrics(bad).items():
        old = previous.get(key)
        if old is None or count > old[0] or depth > old[1] + 1e-9:
            raise RuntimeError('Rack alignment introduced a native-geometry regression: '
                               + str((key, old, (count, depth))))
    margin = float(np.minimum(q-m.jnt_range[ids, 0], m.jnt_range[ids, 1]-q).min())
    clock.record('rack_alignment_reference', 'left.return_above_holder',
                 source_state_time_s=float(d.time), original_command_rad=command.tolist(),
                 reserved_command_rad=q.tolist(), minimum_command_margin_rad=margin,
                 required_planning_margin_rad=RACK_COMMAND_MARGIN_RAD,
                 original_pose_position_error_m=pe, original_pose_orientation_error_deg=oe,
                 maximum_command_change_rad=float(np.max(np.abs(q-command))),
                 reference_clock_unchanged=True, original_descent_guard_unchanged=True,
                 original_native_geometry_rejections=original_bad,
                 reserved_native_geometry_rejections=bad,
                 native_geometry_clean=not bad, native_geometry_nonregression=True,
                 geometry_comparison_roundoff_mm=1e-9, strict_grip_certified=False,
                 live_state_or_controls_written=False, physical_execution_passed=None)
    return q


def _rack_alignment_source(source):
    import ast
    import v34_uncapped_clock as clock

    source = clock.replace_once(source,
        "phase('return_above_holder',3.,goal(rack_above_target,returnR))",
        "phase('return_above_holder',3.,_wall_rack_alignment_goal(m,d,handles,goal(rack_above_target,returnR)))",
        'rack alignment command reserve', count=1)
    source = clock.replace_once(source,
        'if rack_above_position_error<=.0005 and rack_above_orientation_error<=.2:',
        'if rack_above_position_error<=.0005 and rack_above_orientation_error<=.2 and _wall_rack_alignment_ready(m,armcmd):',
        'rack alignment skip preserves descent margin', count=1)
    tree = ast.parse(source)
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != 'advance_reagent_preparation':
        raise RuntimeError('Rack alignment requires the original generated LEFT function')
    # Keep the single-function source contract used by later adapters.
    lines = source.splitlines(keepends=True)
    first_body_line = tree.body[0].body[0].lineno - 1
    indent = lines[first_body_line][:len(lines[first_body_line]) - len(lines[first_body_line].lstrip())]
    lines.insert(first_body_line, indent + 'from v34_wall_pour import _rack_alignment_goal as _wall_rack_alignment_goal, '
                 '_rack_alignment_ready as _wall_rack_alignment_ready\n')
    result = ''.join(lines)
    compile(result, '<wall_rack_alignment_source>', 'exec')
    return result


@contextmanager
def planning_context(m, d, handles, *, profile, ik_solver, **unused):
    """Explicit context for the installed planner smoke or the native executor."""
    if _CONTEXT.get() is not None:
        raise RuntimeError('Nested wall-pour planning context is not supported')
    if profile.get('geometric_path_kind') != 'joint_reference_curve':
        raise RuntimeError('Wall pour requires native18 active joint_reference_curve')
    if not profile.get('v7_actual_return_reanchor', False):
        raise RuntimeError('Wall pour requires the unchanged actual-entry return planner')
    if not callable(ik_solver):
        raise TypeError('Wall pour requires the existing live-entry IK solver')
    geometry = receiver_geometry(m, d, handles, profile)
    token = _CONTEXT.set(dict(model=m, data=d, geometry=geometry))
    try:
        yield geometry
    finally:
        _CONTEXT.reset(token)


@contextmanager
def _execution_context(m, d, handles, arm_command, step_callback, *, profile, ledger, ik_solver):
    # No callback, physics, ledger, hand or live-state writes are made here.
    with planning_context(m, d, handles, profile=profile, ik_solver=ik_solver) as geometry:
        yield geometry


def _replan_wall(m, d, profile, source_plan, entry_arm_command, segment):
    """Run after, never instead of, the active native18 joint/SpeedPour chain."""
    if segment != 'tilt':
        return source_plan
    import mujoco
    import v34_topp_core as core
    import v34_uncapped_clock as clock
    import v34_speedpour as speed
    from t1_left_standing_workflow_v6 import _native
    from t1_spincoat_scene import arm_indices
    from v34_topp_left_paths import _command_bounds

    context = _CONTEXT.get()
    if context is None or context['model'] is not m or context['data'] is not d:
        raise RuntimeError('Active wall-pour tilt needs its actual-entry handles/IK context')
    if profile.get('geometric_path_kind') != 'joint_reference_curve':
        raise RuntimeError('Wall-pour hook reached a different geometric branch')
    dt = float(m.opt.timestep)
    if abs(dt - 0.001) > 1e-12:
        raise RuntimeError('Wall pour retains the original 1ms control/physics cadence')
    times = np.r_[0., np.asarray(source_plan['time'], dtype=float)]
    coordinates = np.r_[0., np.asarray(source_plan['angle_deg'], dtype=float)]
    entry = _finite(entry_arm_command, (7,), 'entry arm command').copy()
    source_commands = np.vstack((entry, np.asarray(source_plan['arm_commands'])))
    if (source_commands.shape != (len(times), 7) or coordinates.shape != times.shape
            or not np.isfinite(source_commands).all() or not np.isfinite(coordinates).all()
            or not np.allclose(np.diff(times), dt, rtol=0., atol=1e-10)
            or np.any(np.diff(coordinates) < -1e-9)):
        raise ValueError('Unexpected active native18 tilt reference grid')
    starts = np.flatnonzero(coordinates >= START_COORDINATE)
    ends = np.flatnonzero(coordinates >= END_COORDINATE)
    if not len(starts) or not len(ends):
        raise RuntimeError('Native tilt does not reach the declared wall-approach coordinates')
    first, finish = int(starts[0]), int(ends[0])
    if float(profile.get('v7_free_mouth_below_tilt_deg', 80.)) > START_COORDINATE:
        raise RuntimeError('Wall motion would intrude into the original free-preflow corridor')
    source = CubicSpline(times, source_commands, axis=0,
                         bc_type=((1, np.zeros(7)), (1, np.zeros(7))))
    ids, qa, _ = arm_indices(m, 'left')
    palm = m.body('sharpa_left/left_hand_C_MC').id
    geometry = context['geometry']
    tube = m.body(geometry['tube_body']).id
    mouth = m.site(geometry['tube_mouth_site']).id
    tq = int(m.joint(geometry['tube_joint']).qposadr[0])
    H0 = d.xmat[palm].reshape(3, 3).copy()
    E = H0.T @ d.xmat[tube].reshape(3, 3)
    ml = H0.T @ (d.site_xpos[mouth] - d.xpos[palm])
    rel = H0.T @ (d.xpos[tube] - d.xpos[palm])
    reference_work, work = mujoco.MjData(m), mujoco.MjData(m)
    reference_work.qpos[:] = d.qpos
    work.qpos[:] = d.qpos

    def source_pose(time):
        q = source(time)
        reference_work.qpos[qa] = q
        mujoco.mj_kinematics(m, reference_work)
        H = reference_work.xmat[palm].reshape(3, 3).copy()
        M = reference_work.xpos[palm] + H @ ml
        return q, H, M

    # Preserve the native outward component before starting an inward approach.
    # Selecting its first inward sample avoids reversing the lip at the join,
    # without changing the original prefix, endpoint time or minimum duration.
    eligible_start_time = float(times[first])

    def source_rim_distance(index):
        _, rotation, mouth_position = source_pose(float(times[index]))
        return lip_geometry(mouth_position, rotation @ E, geometry)[
            'inner_lip_distance_to_intended_rim_m']

    previous_distance = source_rim_distance(first)
    while first < finish:
        next_distance = source_rim_distance(first + 1)
        if next_distance <= previous_distance:
            break
        first += 1
        previous_distance = next_distance
    start_time, finish_time = float(times[first]), float(times[finish])
    blend = approach_reference(finish_time - start_time)
    if abs(blend.duration - (finish_time - start_time)) > 1e-9:
        raise RuntimeError('Explicit 1x approach changed the protected source-clock interval')

    def goal(time):
        q, H, M = source_pose(time)
        progress = float(blend(np.clip(time - start_time, 0., blend.duration))[0])
        desired = wall_mouth_target(M, H @ E, progress, geometry)
        return q, H, desired, progress

    lower = np.asarray(m.jnt_range[ids, 0]) + 0.05
    upper = np.asarray(m.jnt_range[ids, 1]) - 0.05

    def solve_wall_pose(qsource, previous_correction, target, rotation):
        # Transport the previous correction with the original joint curve.
        # Center both barrier residuals on this seed: zero requested correction
        # must not create the unrelated nullspace drift of the absolute barrier.
        # The original residual scales and hard 0.05rad margins are retained.
        anchor = np.clip(qsource + previous_correction, lower + 1e-9, upper - 1e-9)
        lower_anchor = 0.003 / (anchor - lower + 0.05)
        upper_anchor = 0.003 / (upper - anchor + 0.05)

        def residual(q):
            work.qpos[qa] = q
            mujoco.mj_kinematics(m, work)
            orientation = Rotation.from_matrix(
                work.xmat[palm].reshape(3, 3) @ rotation.T).as_rotvec()
            return np.r_[3000. * (work.xpos[palm] - target), 300. * orientation,
                         0.05 * (q - anchor),
                         0.003 / (q - lower + 0.05) - lower_anchor,
                         0.003 / (upper - q + 0.05) - upper_anchor]

        fit = least_squares(residual, anchor, bounds=(lower, upper), max_nfev=180,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not fit.success:
            raise RuntimeError('Source-relative wall IK did not converge: ' + fit.message)
        return _finite(fit.x, (7,), 'fresh wall IK command')

    stride = round(0.005 / dt)
    knot_indices = np.unique(np.r_[first, np.arange(first, len(times), stride), finish, len(times) - 1])
    knot_times = times[knot_indices]
    deltas = [np.zeros(7)]
    native_checks = 0
    for index in knot_indices[1:]:
        qsource, H, desired, _ = goal(float(times[index]))
        q = solve_wall_pose(qsource, deltas[-1], desired - H @ ml, H)
        work.qpos[qa] = q
        mujoco.mj_kinematics(m, work)
        actual_H = work.xmat[palm].reshape(3, 3)
        pe = float(np.linalg.norm(work.xpos[palm] + actual_H @ ml - desired))
        oe = float(np.degrees(Rotation.from_matrix(actual_H @ H.T).magnitude()))
        if pe > POSITION_TOLERANCE_M or oe > ORIENTATION_TOLERANCE_DEG:
            raise RuntimeError('Fresh wall-pour IK failed original pose precision: ' + str((float(times[index]), pe, oe)))
        bad = _native(m, work, palm, geometry['tube_body'], tq, rel, E)
        native_checks += 1
        if bad:
            raise RuntimeError('Fresh wall-pour native geometry rejected: ' + str((float(times[index]), bad)))
        deltas.append(q - qsource)
    deltas = np.asarray(deltas)
    jets = CubicSpline(knot_times, deltas, axis=0,
                       bc_type=((1, np.zeros(7)), (1, np.zeros(7))))
    values = np.stack((deltas, jets(knot_times, 1), jets(knot_times, 2)), axis=1)
    values[0, 1:] = 0.
    values[-1, 1:] = 0.
    correction = BPoly.from_derivatives(knot_times, values)
    # The correction is exactly zero with zero first/second derivatives at its
    # join. Preserve the entire already-guarded native prefix, not an IK refit.
    all_jets = np.stack([source(times, derivative) for derivative in (0, 1, 2)], axis=1)
    for derivative in (0, 1, 2):
        all_jets[first:, derivative] += correction(times[first:], derivative)
    path = BPoly.from_derivatives(times, all_jets)
    timing = core.retime(path, duration=float(times[-1]),
                         label='left.wall_pour_joint_reference', factor=1.0)
    if abs(timing.duration - float(source_plan['duration_s'])) > 1e-9:
        raise RuntimeError('Wall geometry changed the protected native18 tilt duration')
    commands = np.asarray(timing(times[1:]))
    if np.max(np.abs(commands[:first] - source_commands[1:first + 1])) > 1e-10:
        raise RuntimeError('Wall pour changed its original pre-approach prefix')
    margin, control_margin = _command_bounds(m, path, ids, 0.05 - 1e-6, 'wall pour')
    result = dict(source_plan)
    result.update(measured_hand_to_tube_R=E.tolist(), measured_mouth_in_hand_m=ml.tolist())
    # Re-run the original SpeedPour wrist/finite-position guards on NEW commands.
    wrist = speed._geometry_guards(m, d, result, commands)
    desired_distances, planned_distances, clearances = [], [], []
    peak_position_error, peak_orientation_error = 0., 0.
    sample_rows = []
    sample_indices = set(np.linspace(first, len(times) - 1, 101, dtype=int)) | {first, finish, len(times) - 1}
    for index in range(first, len(times)):
        time = float(times[index])
        _, H, desired, progress = goal(time)
        work.qpos[qa] = commands[index - 1]
        mujoco.mj_kinematics(m, work)
        actual_H = work.xmat[palm].reshape(3, 3).copy()
        actual_M = work.xpos[palm] + actual_H @ ml
        pe = float(np.linalg.norm(actual_M - desired))
        oe = float(np.degrees(Rotation.from_matrix(actual_H @ H.T).magnitude()))
        peak_position_error = max(peak_position_error, pe)
        peak_orientation_error = max(peak_orientation_error, oe)
        if pe > POSITION_TOLERANCE_M or oe > ORIENTATION_TOLERANCE_DEG:
            raise RuntimeError('Wall-pour interpolant failed original pose precision: ' + str((time, pe, oe)))
        if H[2, 0] > -0.25 + 1e-8:
            raise RuntimeError('Wall pour violates the original >=75-degree palm-normal guard')
        intended = lip_geometry(desired, H @ E, geometry)
        observed = lip_geometry(actual_M, actual_H @ E, geometry)
        clearance = observed['complete_lip_vertical_clearance_m']
        if clearance < MINIMUM_RIM_CLEARANCE_M:
            raise RuntimeError('Wall pour has insufficient complete tilted-lip clearance: ' + str((time, clearance)))
        desired_distances.append(intended['inner_lip_distance_to_intended_rim_m'])
        planned_distances.append(observed['inner_lip_distance_to_intended_rim_m'])
        clearances.append(clearance)
        if index == first or (index - first) % round(0.01 / dt) == 0 or index == len(times) - 1:
            bad = _native(m, work, palm, geometry['tube_body'], tq, rel, E)
            native_checks += 1
            if bad:
                raise RuntimeError('Wall-pour interpolated native geometry rejected: ' + str((time, bad)))
        if index in sample_indices:
            sample_rows.append(dict(time_s=time, nominal_coordinate=float(coordinates[index]),
                                    approach_progress=progress, commanded_mouth_world_m=desired.tolist(),
                                    planned_mouth_world_m=actual_M.tolist(),
                                    source_palm_rotation_world=H.tolist(),
                                    planned_palm_rotation_world=actual_H.tolist(),
                                    target_lip=intended, planned_lip=observed))
    desired_distances = np.asarray(desired_distances)
    planned_distances = np.asarray(planned_distances)
    desired_increase = float(max(0., np.max(np.diff(desired_distances))))
    planned_increase = float(max(0., np.max(np.diff(planned_distances))))
    if desired_increase > 1e-7 or desired_distances[0] - desired_distances[-1] < 0.005:
        raise RuntimeError('Wall-pour desired lip path is not a decreasing >=5mm approach to the selected rim')
    if planned_increase > 2. * POSITION_TOLERANCE_M:
        raise RuntimeError('Wall-pour fresh IK lip path exceeds its pose-precision monotonicity envelope')
    metadata = dict(
        geometry=geometry, active_branch='joint_reference_curve (takes precedence over joint_arclength_C2)',
        source='Installed native18 joint/SpeedPour plan from current-episode live retarget; not an archived command trajectory',
        source_state_time_s=float(d.time), profile_retarget_time_s=profile.get('source_state_time_s'),
        slow_segment=dict(label='left.wall_pour_approach', reference_factor=float(blend.factor),
                          start_s=start_time, end_s=finish_time, duration_s=blend.duration,
                          source_coordinate_interval=[float(coordinates[first]), float(coordinates[finish])],
                          earliest_eligible_start_s=eligible_start_time,
                          inward_join_delay_s=start_time - eligible_start_time,
                          join_reason='Preserve the original outward component before starting the inward approach',
                          law='Quintic C2 lower-inner-lip blend; final lip stays fixed while source tilt continues',
                          timing=blend.report),
        protected_native18_tilt_clock_unchanged=True, protected_return_clock_unchanged=True,
        native_source_timing=source_plan.get('uncapped_reference'),
        native_speedpour_source_report=source_plan.get('speedpour_retiming'),
        final_joint_reference_timing=timing.report, fresh_ik_knot_count=len(knot_times) - 1,
        fresh_ik_source='Source-relative continuation with seed-centered geometric barrier residuals; original residual scales, hard 0.05rad joint margins and pose/native guards retained',
        first_fresh_ik_knot_time_s=float(knot_times[1]),
        first_fresh_ik_correction_max_abs_rad=float(np.max(np.abs(deltas[1]))),
        maximum_fresh_ik_correction_rad=float(np.max(np.abs(deltas))),
        measured_hand_to_tube_R=E.tolist(), measured_mouth_in_hand_m=ml.tolist(),
        desired_lip_rim_distance_start_end_m=[float(desired_distances[0]), float(desired_distances[-1])],
        planned_lip_rim_distance_start_end_m=[float(planned_distances[0]), float(planned_distances[-1])],
        desired_maximum_distance_increase_per_1ms_m=desired_increase,
        planned_maximum_distance_increase_per_1ms_m=planned_increase,
        planned_monotonicity_numerical_envelope_m=2. * POSITION_TOLERANCE_M,
        planned_complete_lip_clearance_range_m=[float(min(clearances)), float(max(clearances))],
        maximum_mouth_position_error_m=peak_position_error,
        maximum_source_palm_orientation_error_deg=peak_orientation_error,
        correction_boundary_velocity_rad_s=[correction(start_time, 1).tolist(), correction(times[-1], 1).tolist()],
        correction_boundary_acceleration_rad_s2=[correction(start_time, 2).tolist(), correction(times[-1], 2).tolist()],
        original_command_prefix_preserved=True, native_geometry_screen_count=native_checks,
        native_geometry_scope='Unchanged _native categories/thresholds at fresh IK knots and <=10ms; original SpeedPour wrist guard at every 1ms command. No continuous collision certificate.',
        samples=sample_rows, hand_control_changed=False, model_or_live_state_written=False,
        flow_model_and_ledger_termination_unchanged=True,
        early_liquid_stop_may_interrupt_approach=True,
        physical_execution_passed=None, strict_grip_certified=False,
        runtime_requirement='Observe actual lower/outer lip, receiver clearance, grasp/support and original liquid completion on a fresh same-episode run; also inspect early-stop coverage and actual return-entry continuity.',
    )
    result.update(time=times[1:], arm_commands=commands,
                  peak_joint_speed_rad_s=float(np.max(np.abs(timing(times, 1)))),
                  min_joint_margin_rad=margin, actuator_control_hull_margin_rad=control_margin,
                  planned_forearm_palm_max_deg=wrist,
                  time_law='fresh_wall_IK_C2_translation_on_preserved_native18_tilt_time_and_orientation',
                  uncapped_reference=timing.report, wall_pour=metadata)
    # Old bounds describe the parent geometry, not these freshly solved commands.
    result['speedpour_retiming'] = dict(source_plan['speedpour_retiming'],
                                       scope='Preserved source clock and original source plan; final geometry checks are wall_pour',
                                       final_wall_reference_bounds_inherited=False)
    clock.record('wall_pour_reference', 'left.wall_pour_approach', **metadata)
    return result


def plan_wall_pour(m, d, handles, profile, *, entry_arm_command, ik_solver, duration_s=None):
    """Installed planning-smoke entry; uses the actual active dispatch, no physics."""
    if not _INSTALLED:
        raise RuntimeError('Install wall pour after the complete native18 chain first')
    import t1_human_pour_profile_v4 as api

    with planning_context(m, d, handles, profile=profile, ik_solver=ik_solver):
        return api.plan_pour_segment_v4(
            m, d, profile, segment='tilt', ik_solver=ik_solver,
            entry_arm_command=entry_arm_command,
            duration_s=float(profile['tilt_duration_s'] if duration_s is None else duration_s),
        )


def install():
    """Patch existing callable identities after native18; return exact symbols."""
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('Wall pour must be installed once per frozen process')
    import v34_uncapped_clock as clock
    import v34_speedpour as speed
    import v34_topp_left as left
    import t1_actual_frame_return_v7 as actual
    import t1_human_pour_v4_joint_reference as joint
    import t1_v10_v27_stage_v28 as stage
    import inspect

    if not getattr(speed, '_INSTALLED', False) or not getattr(left, '_INSTALLED', False):
        raise RuntimeError('Wall pour requires the installed native18 SpeedPour/TOPP chain')
    # Inspect the current code object, not inspect.unwrap or an obsolete disk body.
    installed = inspect.getsource(joint.plan_segment.__code__)
    if 'previous_segment(m, d, profile' not in installed or 'uncapped_reference=timing.report' not in installed:
        raise RuntimeError('Unexpected final native18 active pour wrapper')
    from v34_uncapped_left import _cells

    return_wrapper = actual.plan_return
    while 'constrained' not in _cells(return_wrapper):
        return_wrapper = getattr(return_wrapper, '__wrapped__', None)
        if return_wrapper is None:
            raise RuntimeError('Missing installed offset-aware return solver')
    constrained_return = _cells(return_wrapper)['constrained']
    # The original knot constraint can be active while its joint interpolant
    # falls below it. Reserve one existing position tolerance in the planner;
    # keep the external corridor, timing, servo offset and all live guards.
    clock.patch(constrained_return, [(
        'min(.89,M0[2]-.001)',
        'min(.89,M0[2]-.001)+_wall_return_geometry_reserve_m', 1,
    )], 'wall-return geometric interpolation reserve',
        helpers={'_wall_return_geometry_reserve_m': POSITION_TOLERANCE_M})
    clock.record('planning_guard', 'left.wall_return_interpolation_reserve',
                 mouth_floor_planning_reserve_m=POSITION_TOLERANCE_M,
                 original_external_corridor_unchanged=True,
                 original_timing_and_servo_offset_retained=True,
                 physical_parameters_unchanged=True,
                 physical_execution_passed=None)

    clock.patch(stage.adapt_left_stage, [(
        'return adapt_stage(previous(source))',
        'return _wall_rack_alignment_source(adapt_stage(previous(source)))', 1,
    )], 'rack alignment reserve after the complete installed parent source-adapter chain',
        helpers={'_wall_rack_alignment_source': _rack_alignment_source})

    clock.patch(actual.advance_human_pour_reanchored_return_v7, [(
        " install_return_planner()\n from t1_deferred_idle_restore_v5 import advance_human_pour_deferred_restore_v5\n return advance_human_pour_deferred_restore_v5(*args,**kwargs)",
        " with _wall_pour_execution_context(*args,**kwargs):\n  install_return_planner()\n  from t1_deferred_idle_restore_v5 import advance_human_pour_deferred_restore_v5\n  return advance_human_pour_deferred_restore_v5(*args,**kwargs)", 1,
    )], 'wall-pour actual handles and IK context without executor/gate changes',
        helpers={'_wall_pour_execution_context': _execution_context})
    clock.patch(joint.plan_segment, [(
        '    return result\n',
        '    return _wall_pour_replan(m, d, profile, result, entry_arm_command, segment)\n', 1,
    )], 'wall-pour active joint-reference geometry after native18 source clocks',
        helpers={'_wall_pour_replan': _replan_wall})
    _INSTALLED = True
    return PATCHED_SYMBOLS
