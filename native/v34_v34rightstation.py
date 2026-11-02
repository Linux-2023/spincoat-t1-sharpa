"""V34RightStation: preserve the canonical tool-to-coarse-dock XY relation.

The immutable V31 layout moves the stand/tool by at most 20 mm per world axis
but leaves grasp_base_xy_m fixed at (0.02, 0.20). This plugin translates that
existing coarse destination by the same bounded object displacement. Both the
prestart workData IK and the existing physical coarse phase consume the same
configuration. Initial robot state, torso, aspiration/coating docks, timing,
IK solvers, clearance gates, and fine-stage zero-speed control are untouched.

This is a candidate planning change, not a physical or hardware certificate.
"""
from dataclasses import replace
from functools import wraps
import math

CANDIDATE_ID = "V34RightStation"
PATCHED_SYMBOL = "t1_right_station_yaw_launch_v7.adapt_neutral_source_v7_yaw"
CANONICAL_STAND_XY_M = (0.68, 0.0)
CANONICAL_GRASP_BASE_XY_M = (0.02, 0.20)
MAX_OFFSET_PER_AXIS_M = 0.02
_APPLIED_PLANS = []
_INSTALLED = False


def station_config(config):
    """Return a bounded coarse destination; do not inspect or mutate live data."""
    stand = tuple(float(value) for value in config.stand_center_xy_m)
    old_base = tuple(float(value) for value in config.grasp_base_xy_m)
    if len(stand) != 2 or len(old_base) != 2:
        raise ValueError("V34RightStation requires two-dimensional station fields")
    if old_base != CANONICAL_GRASP_BASE_XY_M:
        raise ValueError("V34RightStation requires the unchanged canonical grasp dock")
    delta = tuple(value - nominal for value, nominal in zip(stand, CANONICAL_STAND_XY_M))
    if any(not math.isfinite(value) or abs(value) > MAX_OFFSET_PER_AXIS_M + 1e-12 for value in delta):
        raise ValueError("V34RightStation rejects offsets outside the authorized SAME20 domain")
    target = tuple(base + shift for base, shift in zip(old_base, delta))
    # Exact control no-op, including original list/tuple representation.
    derived = config if all(shift == 0.0 for shift in delta) else replace(config, grasp_base_xy_m=target)
    report = {
        "candidate_id": CANDIDATE_ID,
        "scope": "Object-relative destination for the existing RIGHT grasp coarse phase only",
        "stand_center_xy_m": list(stand),
        "original_grasp_base_xy_m": list(old_base),
        "grasp_base_xy_m": list(derived.grasp_base_xy_m),
        "translation_world_xy_m": list(delta),
        "translation_norm_m": math.hypot(*delta),
        "max_offset_per_axis_m": MAX_OFFSET_PER_AXIS_M,
        "zero_offset_exact_noop": derived is config,
        "original_initialization_and_gates_retained": True,
        "aspiration_and_coating_docks_unchanged": True,
        "no_fine_stage_base_feedback_added": True,
    }
    return derived, report


def applied_plans():
    return [dict(plan) for plan in _APPLIED_PLANS]


def install():
    """Wrap the existing yaw75 adapter before the native workflow is composed."""
    global _INSTALLED
    if _INSTALLED:
        return
    import t1_right_station_yaw_launch_v7 as native

    original = native.adapt_neutral_source_v7_yaw

    @wraps(original)
    def object_relative_station(source, config):
        derived, report = station_config(config)
        transformed = original(source, derived)
        _APPLIED_PLANS.append(report)
        return transformed

    native.adapt_neutral_source_v7_yaw = object_relative_station
    _INSTALLED = True
