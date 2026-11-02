"""Offset-aware command bounds for the native V31 actual-frame return planner.

Only geometric IK bounds change: intersect the original feasible q interval
with the original command limits translated by the complete measured servo
offset. The native q+offset construction, interpolation, geometry constraints,
0.05 rad/2 rad/s/control-range guards, duration, and gains remain unchanged.
This is a planning candidate, not evidence of physical hold or task success.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
from pathlib import Path

import numpy as np

CANDIDATE_ID = "V34ReturnOffsetAware"
PATCHED_SYMBOL = "t1_actual_frame_return_v7.plan_return"
ORIGINAL_MODULE_SHA256 = "8a88d26317fbb486c33bab2777bba381fec80952d0cda91cbc8a4752bf1dcd63"
_BOUND_ANCHOR = "lo=m.jnt_range[ids,0]+.0501;hi=m.jnt_range[ids,1]-.0501;rows=[];qs=[];states=[]"
_BOUND_REPLACEMENT = "lo,hi=_v34_offset_bounds(m,ids,q0,offset);rows=[];qs=[];states=[]"
_GUARD_ANCHOR = "if speed>2 or margin<.05 or(control_margin is not None and control_margin<0):raise RuntimeError('Original return speed/margin/control-range failed '+str((speed,margin,control_margin)))"
_COMMAND_ANCHOR = "commands=interp(arc[-1]*progress)+offset"
ORIGINAL_PLAN_RETURN = None


def _actuators(m, ids):
    return np.array([np.flatnonzero(m.actuator_trnid[:, 0] == j)[0] for j in ids])


def _v34_offset_bounds(m, ids, q0, offset):
    # .0501 is the frozen planner's existing numerical reserve, not a new gate.
    geometric_lo = m.jnt_range[ids, 0] + .0501
    geometric_hi = m.jnt_range[ids, 1] - .0501
    command_lo = geometric_lo.copy()
    command_hi = geometric_hi.copy()
    acts = _actuators(m, ids)
    limited = m.actuator_ctrllimited[acts].astype(bool)
    command_lo[limited] = np.maximum(command_lo[limited], m.actuator_ctrlrange[acts[limited], 0])
    command_hi[limited] = np.minimum(command_hi[limited], m.actuator_ctrlrange[acts[limited], 1])
    lo = np.maximum(geometric_lo, command_lo - offset)
    hi = np.minimum(geometric_hi, command_hi - offset)
    entry = q0 + offset
    entry_margin = np.minimum(entry - m.jnt_range[ids, 0], m.jnt_range[ids, 1] - entry)
    if (not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo > hi) or np.any(entry_margin < .05)):
        raise RuntimeError("Offset-aware return has no admissible entry/bounds; compensation is not clipped")
    if limited.any() and (np.any(entry[limited] < m.actuator_ctrlrange[acts[limited], 0])
                          or np.any(entry[limited] > m.actuator_ctrlrange[acts[limited], 1])):
        raise RuntimeError("Offset-aware return entry violates original actuator range")
    return lo, hi


def install():
    """Install one isolated native-function source adaptation; never edit V31."""
    global ORIGINAL_PLAN_RETURN
    import t1_actual_frame_return_v7 as native

    original = native.plan_return
    if getattr(original, "_v34_return_offset_aware", False):
        return
    if original.__module__ != native.__name__ or hasattr(original, "__wrapped__"):
        raise RuntimeError("Another return planner is installed; candidates are alternatives, not composable")
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != ORIGINAL_MODULE_SHA256:
        raise RuntimeError("Offset-aware candidate requires the immutable V31 return module")
    source = inspect.getsource(original)
    for anchor in (_BOUND_ANCHOR, _GUARD_ANCHOR, _COMMAND_ANCHOR):
        if source.count(anchor) != 1:
            raise RuntimeError("Native return source anchor mismatch: " + anchor)
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    adapted = source.replace(_BOUND_ANCHOR, _BOUND_REPLACEMENT, 1)
    namespace = dict(original.__globals__)
    namespace["_v34_offset_bounds"] = _v34_offset_bounds
    exec(compile(adapted, "<v34_return_offset_aware_native_bounds>", "exec"), namespace)
    constrained = namespace["plan_return"]
    ORIGINAL_PLAN_RETURN = original

    @functools.wraps(original)
    def plan_return(m, d, profile, *, entry_arm_command, duration_s=10.):
        plan = constrained(m, d, profile, entry_arm_command=entry_arm_command, duration_s=duration_s)
        ids, qa, _ = native.arm_indices(m, "left")
        offset = np.asarray(plan["preserved_servo_offset_rad"])
        lo, hi = _v34_offset_bounds(m, ids, np.asarray(d.qpos[qa]), offset)
        commands = plan["arm_commands"]
        geometric = plan["geometric_arm_q"]
        plan["v34_return_offset_aware"] = {
            "candidate_id": CANDIDATE_ID,
            "original_function_sha256": source_sha256,
            "source_servo_offset_rad": offset.tolist(),
            "geometric_lower_bounds_rad": lo.tolist(),
            "geometric_upper_bounds_rad": hi.tolist(),
            "geometric_knot_min_margin_rad": float(np.minimum(
                geometric - m.jnt_range[ids, 0], m.jnt_range[ids, 1] - geometric).min()),
            "command_min_margin_by_joint_rad": np.minimum(
                commands - m.jnt_range[ids, 0], m.jnt_range[ids, 1] - commands).min(axis=0).tolist(),
            "compensation_preserved_without_clipping": True,
            "native_guards_unchanged": True,
            "physics_success_not_inferred": True,
        }
        return plan

    plan_return._v34_return_offset_aware = True
    native.plan_return = plan_return
