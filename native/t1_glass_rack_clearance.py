"""Explicit world-height clearance targets for the new glass rack route."""
import numpy as np


def rack_clearance_lift_m(plan, actual_tube_bottom_z_m):
    if 'rack_clearance_bottom_z_m' not in plan:
        return 0.
    target = float(plan['rack_clearance_bottom_z_m'])
    value = max(0., target-float(actual_tube_bottom_z_m))
    if not np.isfinite(target) or not np.isfinite(actual_tube_bottom_z_m) or value > .25:
        raise ValueError('Rack clearance requires finite heights and at most0.25m extra lift')
    return value


def rack_return_lift_m(plan, initial_tube_bottom_z_m):
    return max(.10, rack_clearance_lift_m(plan, initial_tube_bottom_z_m))


def route_waypoints(plan, key):
    points = np.asarray(plan.get(key, []), dtype=float)
    if points.size == 0:
        return []
    if points.ndim != 2 or points.shape[1] != 3 or len(points) > 8 or not np.all(np.isfinite(points)):
        raise ValueError('Explicit rack route requires at most8 finite worldXYZ waypoints')
    return points


def cartesian_transport_phase(name):
    return (name in {'raise_clear_of_rack', 'carry_above_receiver',
                     'raise_upright_clear_of_rack_before_return', 'return_above_holder'} or
            name.startswith(('carry_clearance_waypoint_', 'return_clearance_waypoint_')))


def left_palm_target_fk(model, data, joint_target):
    import mujoco
    from t1_spincoat_scene import arm_indices
    _, qa, _ = arm_indices(model, 'left')
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    work.qpos[qa] = np.asarray(joint_target)
    mujoco.mj_kinematics(model, work)
    palm = model.body('sharpa_left/left_hand_C_MC').id
    return work.xpos[palm].copy(), work.xmat[palm].reshape(3, 3).copy()
