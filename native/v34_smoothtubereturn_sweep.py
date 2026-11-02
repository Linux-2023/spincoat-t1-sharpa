"""Conservative continuous enabled-native environment clearance on planning data.

This is a Lipschitz displacement enclosure, not endpoint-only collision sampling.
It certifies the declared rigid-grip/static-environment reference, not actual
tracking, grip, disabled meshes, full-visual geometry or all self-collision pairs.
"""
import numpy as np
import mujoco

from t1_spincoat_scene import arm_indices
from v34_smoothtubereturn_distance import prepare_sweep_distance


def _axis_reach_bound(m, body, joint, radius):
    """Configuration-independent triangle bound from a joint axis to a geom."""
    joint_body = int(m.jnt_bodyid[joint])
    while body != joint_body:
        if body == 0:
            return 0.0  # This arm joint is not an ancestor of the moving geom.
        radius += float(np.linalg.norm(m.body_pos[body]))
        first, count = int(m.body_jntadr[body]), int(m.body_jntnum[body])
        # Rotations about non-origin joint anchors can translate a body frame
        # by at most twice the anchor radius; do not assume all anchors are0.
        if count:
            radius += 2.0 * float(np.linalg.norm(m.jnt_pos[first:first + count], axis=1).sum())
        body = int(m.body_parentid[body])
    return radius + float(np.linalg.norm(m.jnt_pos[joint]))


def certify_loaded_environment_sweep(m, d, reference, offset, tube_name, palm, rel, *, geometry_audit):
    ids, qa, _ = arm_indices(m, 'left')
    if np.any(m.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise RuntimeError('Swept enclosure requires the original seven revolute LEFT joints')
    names = [m.body(int(body)).name for body in m.geom_bodyid]
    moving = [g for g, name in enumerate(names)
              if name == tube_name or name.startswith(('sharpa_left/', 'astribot_arm_left_'))]
    environment = [g for g, name in enumerate(names)
                   if name != tube_name and name.startswith(('prep/', 'lab_bench', 'stand/', 'spincoater/', 'aspiration_source', 'pipette/'))]
    explicit = {tuple(sorted((int(a), int(b)))) for a, b in zip(m.pair_geom1, m.pair_geom2)}
    supported = {int(getattr(mujoco.mjtGeom, 'mjGEOM_' + name))
                 for name in ('PLANE', 'SPHERE', 'CAPSULE', 'ELLIPSOID', 'CYLINDER', 'BOX', 'MESH')}
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    pairs = []
    motion_reaches = np.empty((len(moving), len(ids)))
    local_motion_rates = np.empty(len(moving))
    for moving_index, g in enumerate(moving):
        radius = float(np.linalg.norm(m.geom_pos[g]) + m.geom_rbound[g])
        body = int(m.geom_bodyid[g])
        if names[g] == tube_name:
            # The conditional tool is rigid relative to the palm, not its live
            # free joint. Its geometry radius must include the measured offset.
            body = int(palm)
            radius += float(np.linalg.norm(rel))
        motion_reaches[moving_index] = [_axis_reach_bound(m, body, int(j), radius) for j in ids]
        for h in environment:
            enabled = ((int(m.geom_contype[g]) & int(m.geom_conaffinity[h]))
                       or (int(m.geom_contype[h]) & int(m.geom_conaffinity[g]))
                       or tuple(sorted((g, h))) in explicit)
            if not enabled:
                continue
            # Never treat an unsupported mj_geomDistance pair as clear merely
            # because that API returns distmax for an unsupported type.
            if int(m.geom_type[g]) not in supported or int(m.geom_type[h]) not in supported or int(m.geom_type[g]) == plane:
                raise RuntimeError('Unsupported enabled geometry in swept clearance enclosure')
            pairs.append((g, h, moving_index))
    if not pairs:
        raise RuntimeError('No enabled native environment pairs available for a sweep certificate')
    work = mujoco.MjData(m)
    work.qpos[:] = d.qpos
    with prepare_sweep_distance(m, work, scratch_parent=geometry_audit.audit_directory) as geom_distance:
        pending = [(a, b, bounds, 0) for a, b, bounds in reference.rate_intervals]
        minimum_interval = float(m.opt.timestep) / 128.0
        accepted = queries = sampled = deepest = 0
        minimum_margin = float('inf')
        while pending:
            a, b, joint_rate_bounds, depth = pending.pop()
            np.dot(motion_reaches, joint_rate_bounds, out=local_motion_rates)
            work.qpos[qa] = reference(a) - offset
            mujoco.mj_kinematics(m, work)
            geometry_audit.observe(work, reference_time_s=a, origin='sweep_interval_start')
            sampled += 1
            interval_margin = float('inf')
            unresolved = None
            for g, h, moving_index in pairs:
                displacement = local_motion_rates[moving_index] * (b - a)
                # Bounding spheres are a conservative inexpensive rejection. Plane
                # geoms are infinite and must use the native distance routine.
                distance = -float('inf')
                if int(m.geom_type[h]) != plane:
                    distance = float(np.linalg.norm(work.geom_xpos[g] - work.geom_xpos[h])
                                     - m.geom_rbound[g] - m.geom_rbound[h])
                if distance <= displacement + 1e-9:
                    distance = float(geom_distance(g, h, displacement + 1e-8, None))
                    queries += 1
                margin = distance - displacement
                interval_margin = min(interval_margin, margin)
                if not np.isfinite(margin) or margin <= 1e-9:
                    unresolved = (g, h, distance, displacement)
                    break
            if unresolved is not None:
                if b - a <= minimum_interval:
                    g, h, distance, displacement = unresolved
                    raise RuntimeError('Unresolved swept native clearance; no sampling-pass fallback: '
                                       + str(dict(interval_s=[a, b], geoms=[m.geom(g).name, m.geom(h).name],
                                                  distance_m=distance, maximum_displacement_m=displacement)))
                middle = .5 * (a + b)
                # Both the parent and each new local enclosure are valid upper
                # bounds. Their intersection tightens a proof, not a command.
                left_rates = np.minimum(joint_rate_bounds, reference.rate_bounds.bound(a, middle))
                right_rates = np.minimum(joint_rate_bounds, reference.rate_bounds.bound(middle, b))
                pending.extend(((a, middle, left_rates, depth + 1), (middle, b, right_rates, depth + 1)))
                deepest = max(deepest, depth + 1)
            else:
                accepted += 1
                minimum_margin = min(minimum_margin, interval_margin)
        return dict(
            method='Native convex/primitive signed distance minus a joint-axis reach times analytic joint-rate displacement enclosure; recursive interval subdivision',
            covered_time_s=[0.0, reference.duration], enabled_environment_pair_count=len(pairs),
            certified_intervals=accepted, planning_pose_samples=sampled, native_distance_queries=queries,
            deepest_subdivision=deepest, minimum_interval_s=minimum_interval,
            minimum_certified_environment_clearance_margin_m=minimum_margin,
            analytic_joint_rate_bounds_rad_s=reference.joint_rate_upper.tolist(),
            displacement_bound_scope='Per-interval Bernstein joint-rate enclosure; descendants intersect parent and restricted local bounds, never a product of unrelated global maxima',
            reused_rate_certificate_intervals=len(reference.rate_intervals),
            conditional_assumptions='Fixed planning base/torso/fingers/environment; rigid measured hand-to-tube transform. These are not enforced on live state.',
            exclusions='Disabled/full-visual meshes, actual tracking/contact/friction, environmental motion and general robot self-collision are not certified. Shared classifier still rejects crossdigits/environment; held-tube nonpad/overwall remains explicitly rejected/not-certified grip audit, not a post-pour functional gate.',
            shared_geometry_classifier='v34_smoothtubereturn_geometry.classify_return_geometry',
            grip_certified=False,
            native_distance_query=geom_distance.metadata,
            unresolved_interval_policy='Reject, never accept because samples happened to be clear',
        )
