"""RIGHT acquisition retiming with unchanged physical observation and gates.

The existing private-source/closure convention is reused. No dock, LEFT,
model, force, preload, native stepping, or acceptance implementation is changed.
The hand receives the original analytic closure at1kHz through the one existing
callback; the original50Hz arm loop and all original observers remain owners.
"""
from __future__ import annotations

from functools import wraps
import inspect
import json
import linecache
from pathlib import Path
import textwrap

import numpy as np

from v34_right_pickup_schedule import PROFILE, entry_clock, prefix_duration, smooth, task_clock, timing, validate_profile

CANDIDATE_ID = "SpeedRightPickup"
NEUTRAL_PREPARE_ANCHOR = "        timing.update(clock);mujoco.mj_forward(m,d)"
NEUTRAL_ENGINE_ANCHOR = "    exec(compile(engine_text,engine.__file__,'exec'),engine.__dict__)"
ENGINE_DURATION_ANCHOR = "['--motion-profile',str(mobile_profile),'--close-pinky-before-lift','--lab-context','--max-seconds','11.625']"
ENGINE_FINE_ANCHOR = "fine_readiness(5.8+grasp_clock_origin[0],float(d.time))"
_INSTALLED = False


def _once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{CANDIDATE_ID} exact anchor count {count}: {old!r}")
    return source.replace(old, new, 1)


def _closure(function):
    return dict(zip(function.__code__.co_freevars, (c.cell_contents for c in function.__closure__ or ())))


def _copy_callable(function, replacements, *, directory, cells=None, global_updates=None):
    source = textwrap.dedent(inspect.getsource(function))
    for old, new in replacements:
        source = _once(source, old, new)
    captured = _closure(function)
    captured.update(cells or {})
    factory = "def _bind(" + ",".join(captured) + "):\n" + textwrap.indent(source, "    ")
    factory += "    return " + function.__name__ + "\n"
    path = Path(directory) / ("aggressive_r07_" + function.__name__ + ".py")
    path.write_text(factory, encoding="utf-8")
    namespace = dict(function.__globals__, **(global_updates or {}))
    filename = str(path)
    linecache.cache[filename] = (len(factory), None, factory.splitlines(True), filename)
    exec(compile(factory, filename, "exec"), namespace)
    result = namespace["_bind"](**captured)
    result.__dict__.update(function.__dict__)
    return result


def _actuators(model, joints):
    result = []
    for joint in joints:
        found = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(found) != 1:
            raise RuntimeError("Expected one original position actuator per RIGHT joint")
        result.append(int(found[0]))
    return np.asarray(result)


def _check_positions(model, joints, actuators, values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite RIGHT command")
    limited = model.jnt_limited[joints].astype(bool)
    if np.any(values[limited] < model.jnt_range[joints[limited], 0] - 1e-10) or np.any(values[limited] > model.jnt_range[joints[limited], 1] + 1e-10):
        raise RuntimeError("RIGHT command exceeds original joint bounds")
    limited = model.actuator_ctrllimited[actuators].astype(bool)
    if np.any(values[limited] < model.actuator_ctrlrange[actuators[limited], 0] - 1e-10) or np.any(values[limited] > model.actuator_ctrlrange[actuators[limited], 1] + 1e-10):
        raise RuntimeError("RIGHT command exceeds original actuator bounds")


def _native_hand_endpoints(runner, model, joints):
    """Exact native runner initialization, including original thumb hover."""
    fit = runner.support_fit
    close = np.array([fit['joint_targets'][model.joint(j).name.split('/')[1]] for j in joints])
    for index, joint in enumerate(joints):
        name = model.joint(joint).name
        if 'thumb' not in name and any(part in name for part in ('MCP_FE', '_DIP')):
            close[index] = min(close[index] + .12, model.jnt_range[joint, 1])
        if 'thumb' in name:
            close[index] = fit['thumb_hover'][name.split('/')[1]]
    opened = runner.support_open.copy()
    for index, joint in enumerate(joints):
        name = model.joint(joint).name
        if 'MCP_FE' in name and 'thumb' not in name:
            opened[index] = 0.
        if 'thumb' in name:
            opened[index] = close[index]
    return opened, close


def retime_prepared(runner, path, callback, audit, clock, directory, model):
    validate_profile()
    expected = dict(seat_start_s=5.8, finger_close_start_s=7.675, hold_start_s=10.175,
                    lift_start_s=11.425, postlift_hold_start_s=15.175,
                    verticalize_start_s=16.425, stage_end_s=17.425)
    for key, value in expected.items():
        if abs(float(clock[key]) - value) > 1e-9:
            raise ValueError("Frozen acquisition timing mismatch: " + key)
    if runner.ik.__name__ != 'shifted_ik' or callback.__name__ != 'callback' or audit.__name__ != 'audit_summary':
        raise RuntimeError("Expected original seat/slot/clearance callable chain")
    prefix = prefix_duration()
    command_timing = timing()
    verticalize = command_timing['verticalize_start_s']
    shifted = _closure(runner.ik)
    close_entry, aligned = shifted['old'], shifted['native']
    if close_entry.__name__ != 'close_entry_ik' or aligned.__name__ != 'aligned_native':
        raise RuntimeError("Unexpected original RIGHT geometric closure chain")
    helpers = dict(_r07_task_clock=task_clock)
    aligned_copy = _copy_callable(aligned, [
        ('elapsed<5.8', f'elapsed<{prefix!r}'),
        ('(elapsed-5.8)/(1.5*time_scale)', f'_r07_task_clock(elapsed-{prefix!r})/1.5'),
    ], directory=directory, cells=dict(verticalize=verticalize), global_updates=helpers)
    physical_elapsed = '(round((data.time-station_clock_origin[0])/model.opt.timestep)*model.opt.timestep)'
    close_copy = _copy_callable(close_entry, [
        (f'({physical_elapsed}-5.8)/1.875', f'_r07_task_clock({physical_elapsed}-{prefix!r})/1.5'),
        ('if 5.8<=', f'if {prefix!r}<='),
        ('<7.675:', f'<{command_timing["finger_close_start_s"]!r}:'),
    ], directory=directory, cells=dict(original_ik=aligned_copy),
       global_updates=dict(helpers, station_verticalize_start=verticalize))
    runner.ik = _copy_callable(runner.ik, [
        ('task=(elapsed-5.8)/time_scale', f'task=_r07_task_clock(elapsed-{prefix!r})'),
    ], directory=directory, cells=dict(old=close_copy, native=aligned_copy, vert=verticalize), global_updates=helpers)
    old_callback = _copy_callable(callback, [
        ('<5.8 and', f'<{prefix!r} and'),
        ('(t-1.9)/1.1', f'(t-{PROFILE["preshape_start_s"]!r})/1.1'),
        ('(t-5.8)/1.0', f'_r07_task_clock(t-{prefix!r})/.8'),
        ('or act not in mcp_nominal:', f'or {prefix!r}<=t<{verticalize!r} or act not in mcp_nominal:'),
    ], directory=directory, global_updates=helpers)
    audit_copy = _copy_callable(audit, [
        ("c['time_s']<5.8", f"c['time_s']<{prefix!r}"),
        ("'formal_close_time_s':5.8", f"'formal_close_time_s':{prefix!r}"),
    ], directory=directory)
    runner.station_task_clock = task_clock
    old_sample = runner.continuous_prefix_sample
    from t1_spincoat_scene import arm_indices, hand_indices
    arm_joints, _, _ = arm_indices(model)
    hand_joints, _, _ = hand_indices(model)
    arm_joints, hand_joints = np.asarray(arm_joints), np.asarray(hand_joints)
    arm_acts, hand_acts = _actuators(model, arm_joints), _actuators(model, hand_joints)
    opened, closed = _native_hand_endpoints(runner, model, hand_joints)
    origin = close_copy.__globals__['station_clock_origin']
    stats = dict(maximum_arm_update_rate_rad_s=0., maximum_hand_update_rate_rad_s=0.,
                 prefix_conservative_rate_bound_rad_s=None, fine_hand_samples=0)
    checked_profiles = set()

    @wraps(old_sample)
    def accelerated_prefix(profile, seconds):
        key = id(profile)
        if key not in checked_profiles:
            knots = np.asarray(profile['continuous_approach']['arm_q_rad'])
            if knots.shape != (2, len(arm_joints)) or not np.allclose(profile['continuous_approach']['time_s'], [.7, 5.8], atol=1e-12, rtol=0.):
                raise RuntimeError("Expected original two-knot stopped-dock prefix")
            for knot in knots:
                _check_positions(model, arm_joints, arm_acts, knot)
            fast = PROFILE['entry_native_s'] - PROFILE['entry_native_tail_s'] - PROFILE['entry_saved_s']
            bound = float(np.max(np.abs(knots[1] - knots[0]))) * 1.875 / 5.1 * (1. + 1.875 * PROFILE['entry_saved_s'] / fast)
            if bound > PROFILE['arm_rate_limit_rad_s']:
                raise RuntimeError("Retimed prefix conservative command-rate bound exceeded: " + str(bound))
            stats['prefix_conservative_rate_bound_rad_s'] = bound
            checked_profiles.add(key)
        return old_sample(profile, entry_clock(seconds))

    runner.continuous_prefix_sample = accelerated_prefix
    last_arm = [None, None]
    last_hand = [None, None]

    @wraps(old_callback)
    def coherent_callback(m, d, after):
        elapsed = round((float(d.time) - origin[0]) / m.opt.timestep) * m.opt.timestep
        in_hand_span = prefix <= elapsed < verticalize
        if not after and in_hand_span:
            u = smooth((task_clock(elapsed - prefix) - 1.5) / 2.)
            d.ctrl[hand_acts] = (1. - u) * opened + u * closed
        old_callback(m, d, after)
        if after:
            return
        # Validate commands, never clip or substitute them. The original
        # callback has already applied the original MCP envelope and recorded it.
        _check_positions(m, arm_joints, arm_acts, d.ctrl[arm_acts])
        _check_positions(m, hand_joints, hand_acts, d.ctrl[hand_acts])
        if round(elapsed / .001) % 20 == 0:
            now = float(d.time)
            if last_arm[0] is not None and now > last_arm[0]:
                rate = float(np.max(np.abs(d.ctrl[arm_acts] - last_arm[1]))) / (now - last_arm[0])
                stats['maximum_arm_update_rate_rad_s'] = max(stats['maximum_arm_update_rate_rad_s'], rate)
                if rate > PROFILE['arm_rate_limit_rad_s'] + 1e-8:
                    raise RuntimeError("Retimed acquisition arm-rate bound exceeded: " + str(rate))
            last_arm[:] = [now, d.ctrl[arm_acts].copy()]
        if in_hand_span:
            now = float(d.time)
            if last_hand[0] is not None:
                rate = float(np.max(np.abs(d.ctrl[hand_acts] - last_hand[1]))) / (now - last_hand[0])
                stats['maximum_hand_update_rate_rad_s'] = max(stats['maximum_hand_update_rate_rad_s'], rate)
                if rate > PROFILE['hand_rate_limit_rad_s'] + 1e-8:
                    raise RuntimeError("Retimed acquisition hand-rate bound exceeded: " + str(rate))
            last_hand[:] = [now, d.ctrl[hand_acts].copy()]
            stats['fine_hand_samples'] += 1

    @wraps(audit_copy)
    def coherent_audit():
        result = audit_copy()
        result['aggressive_right_acquisition'] = dict(profile=PROFILE, command_timing=command_timing, **stats)
        return result

    coherent_audit.dock = audit_copy.dock
    clock = dict(clock, **command_timing)
    profile = json.loads(Path(path).read_text())
    profile['duration_s'] = prefix
    for waypoint in profile['waypoints']:
        if abs(waypoint['time_s'] - 5.8) < 1e-9:
            waypoint['time_s'] = prefix
    profile['hand_profile']['start_time_s'] = PROFILE['preshape_start_s']
    profile['hand_profile']['end_time_s'] = PROFILE['preshape_start_s'] + PROFILE['preshape_duration_s']
    profile['staged_grasp_timing'] = clock
    profile['speed_right_pickup'] = dict(PROFILE)
    Path(path).write_text(json.dumps(profile, indent=2), encoding='utf-8')
    report = dict(candidate_id=CANDIDATE_ID, profile=PROFILE, original_timing=expected,
                  command_timing=command_timing, unchanged_physical_observation=True,
                  original_support_windows_s=dict(prelift=1.25, postlift=1.25, passive=0.5),
                  nominal_schedule_saved_s=expected['stage_end_s'] - command_timing['stage_end_s'],
                  base_docks_changed=False, fine_readiness_starts_at_actual_seating_s=prefix,
                  command_scope='Original targets and geometry; one coherent C2 task clock plus1kHz native hand formula',
                  physics_success_not_inferred=True)
    (Path(directory) / 'speed_right_pickup_timing.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return runner, path, coherent_callback, coherent_audit, clock


def retime_engine_source(source):
    duration = timing()['stage_end_s'] - prefix_duration()
    source = _once(source, ENGINE_DURATION_ANCHOR, ENGINE_DURATION_ANCHOR.replace("'11.625'", repr(str(duration))))
    return _once(source, ENGINE_FINE_ANCHOR, f'fine_readiness({prefix_duration()!r}+grasp_clock_origin[0],float(d.time))')


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    validate_profile()
    # Byte-identical R06 component; native rejection still precedes settling.
    import v34_nativefailurepropagation
    v34_nativefailurepropagation.install()
    import t1_right_station_yaw_launch_v7 as yaw
    previous = yaw.adapt_neutral_source_v7_yaw

    @wraps(previous)
    def adapted_neutral(source, config):
        source = previous(source, config)
        insert = '        from v34_speedrightpickup import retime_prepared\n        runner,path,callback,audit,clock=retime_prepared(runner,path,callback,audit,clock,directory,m)\n'
        source = _once(source, NEUTRAL_PREPARE_ANCHOR, insert + NEUTRAL_PREPARE_ANCHOR)
        insert = '    from v34_speedrightpickup import retime_engine_source\n    engine_text=retime_engine_source(engine_text)\n'
        return _once(source, NEUTRAL_ENGINE_ANCHOR, insert + NEUTRAL_ENGINE_ANCHOR)

    yaw.adapt_neutral_source_v7_yaw = adapted_neutral
    _INSTALLED = True
