"""Reset-only scene augmentation for the existing continuous workflow.

Extra free joints are initialized from the augmented model's default pose.
Every pre-existing state and actuator command is copied by name, not offset.
"""
import mujoco
import numpy as np
import inspect
import json


def augment_preparation_scene(scene_tuple, config=None, *, asset_builder=None):
    if asset_builder is None:
        from t1_reagent_preparation import add_reagent_preparation
        asset_builder = add_reagent_preparation

    spec, previous, state, children = scene_tuple
    if state.time != 0.:
        raise ValueError('Preparation assets can only be added at scene reset.')
    handles = asset_builder(spec, config)
    model = spec.compile()
    data = mujoco.MjData(model)
    for joint in range(previous.njnt):
        name = previous.joint(joint).name
        target = model.joint(name)
        kind = previous.jnt_type[joint]
        nq = 7 if kind == mujoco.mjtJoint.mjJNT_FREE else 4 if kind == mujoco.mjtJoint.mjJNT_BALL else 1
        nv = 6 if kind == mujoco.mjtJoint.mjJNT_FREE else 3 if kind == mujoco.mjtJoint.mjJNT_BALL else 1
        oldq, oldv = int(previous.jnt_qposadr[joint]), int(previous.jnt_dofadr[joint])
        newq, newv = int(target.qposadr[0]), int(target.dofadr[0])
        data.qpos[newq:newq+nq] = state.qpos[oldq:oldq+nq]
        data.qvel[newv:newv+nv] = state.qvel[oldv:oldv+nv]
    for actuator in range(previous.nu):
        data.ctrl[model.actuator(previous.actuator(actuator).name).id] = state.ctrl[actuator]
    data.time = state.time
    model.opt.noslip_iterations = previous.opt.noslip_iterations
    model.vis.global_.offwidth = max(2048, previous.vis.global_.offwidth)
    model.vis.global_.offheight = max(2048, previous.vis.global_.offheight)
    mujoco.mj_forward(model, data)
    return (spec, model, data, [*children, handles['child_spec']]), handles


def relative_station_preparer(baseline, origin, source_path):
    """Give clearance control an elapsed clock while its dock keeps live time.

    Only expressions inside the generated clearance adapter change. The actual
    MjData.time and the coarse controller's clock are never written or wrapped.
    Source is saved so further isolated source adapters can inspect it normally.
    """
    source = inspect.getsource(baseline)
    needle = 'env=dict(station_point=transform_point,station_quat=transform_quat,'
    if source.count(needle) != 1:
        raise ValueError('Unsupported station preparation source.')
    source = source.replace(needle, 'env=dict(station_clock_origin=station_clock_origin,station_point=transform_point,station_quat=transform_quat,')
    needle = ' glob=dict(t1_support_grasp.prepare_support_grasp.__globals__);glob.update('
    if source.count(needle) != 1:
        raise ValueError('Unsupported clearance adapter source boundary.')
    # Apply elapsed time only AFTER support preparation has scaled seat timing.
    # Earlier replacement would prevent its literal 1.5s -> 1.875s patches.
    save_statement = "(out/'executed_clearance_adapter.py').write_text(src)"
    # Match the child runner's integer step clock at phase boundaries. Raw
    # subtraction can yield 5.799999999986 while its task loop is already5.8,
    # otherwise skipping the first25mm seat offset for an entire IK update.
    elapsed = '(round((data.time-station_clock_origin[0])/model.opt.timestep)*model.opt.timestep)'
    elapsed_statement = 'src=src.replace('+repr('data.time')+','+repr(elapsed)+');' + save_statement
    late_patch = '\n'.join([
        ' assert prepare.count('+repr(save_statement)+')==1',
        ' prepare=prepare.replace('+repr(save_statement)+','+repr(elapsed_statement)+')',
    ])+'\n'
    source = source.replace(needle, late_patch + needle)
    source_path.write_text(source, encoding='utf-8')
    namespace = dict(baseline.__globals__, station_clock_origin=origin)
    exec(compile(source, str(source_path), 'exec'), namespace)
    function = namespace[baseline.__name__]
    function.angle = baseline.angle
    function.closing_scale = baseline.closing_scale
    return function


def reanchor_ready_profile(model, data, profile_path):
    """Start a planned approach at the actual returned ready pose, not a reset."""
    from t1_spincoat_scene import arm_indices, hand_indices

    _, arm_q, _ = arm_indices(model)
    _, hand_q, _ = hand_indices(model)
    profile = json.loads(profile_path.read_text())
    actual = data.qpos[arm_q].copy()
    old = np.asarray(profile['waypoints'][0]['arm_q_rad'])
    delta = actual - old
    if np.max(abs(delta)) > .03:
        raise ValueError('Preparation did not return sufficiently close to the ready pose.')
    for waypoint in profile['waypoints']:
        if float(waypoint['time_s']) <= .7 + 1e-8:
            waypoint['arm_q_rad'] = actual.tolist()
    if 'continuous_approach' in profile:
        knots = profile['continuous_approach']
        times = np.asarray(knots['time_s'])
        q = np.asarray(knots['arm_q_rad'])
        # Carry the measured return residual into the first second of motion
        # with a quintic taper, preserving the original terminal grasp target.
        u = np.clip((times - .7) / 1., 0., 1.)
        taper = 1. - u**3 * (10. + u*(-15. + 6.*u))
        updated = q + taper[:, None]*delta
        knots['arm_q_rad'] = updated.tolist()
        for waypoint in profile['waypoints']:
            t = float(waypoint['time_s'])
            if t >= .7:
                waypoint['arm_q_rad'] = [float(np.interp(t, times, updated[:, k])) for k in range(updated.shape[1])]
    profile['hand_profile']['home_q_rad'] = data.qpos[hand_q].tolist()
    profile['preparation_return_reanchor'] = {
        'actual_time_s': float(data.time), 'arm_residual_rad': delta.tolist(),
        'scope': 'Plan starts at measured returned state; no live pose/time writes.'}
    profile_path.write_text(json.dumps(profile, indent=2), encoding='utf-8')
    return profile['preparation_return_reanchor']
