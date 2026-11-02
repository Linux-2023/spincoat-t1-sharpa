"""V34 SpeedAspiration: fast dry positioning, unchanged guarded liquid work.

The original direct-cup routine executes its C2 quintic at a shorter duration
only in its named, dry 0.93m preposition phase, with the original 0.5s settle.
The original aspiration source is composed first. Its sole IMMERSE descent
clock then gains time strictly ABOVE both the rim and surface guard planes.
A C2 clock warp returns to unit rate before the guard plane, reproducing the
original final descent velocity/acceleration rather than scaling liquid work.

Docking, fine stopped-base checks, button/plunger commands, observer volume
accounting, fluid parameters, 1ms dynamics, contact gates, release, withdrawal,
and the original READY+3.5s terminal hold are not changed. Withdrawal remains
native because its dry acceleration cannot shorten that mandatory hold.
No original module, live pose/velocity/time, or acceptance gate is replaced.
"""
from functools import lru_cache, wraps
from pathlib import Path

from v34_speedaspiration_profile import PROFILE

CANDIDATE_ID = "SpeedAspiration"
PATCHED_SYMBOLS = (
    "t1_right_standing_motion_v6.advance_direct_cup_v6",
    "t1_right_interior_workstation_v6.adapt_vertical_aspiration_source",
)
DESCENT_ANCHOR = "u=smooth((t-immersion_lower_at)/3)"
DESCENT_REPLACEMENT = (
    "u=_v34_aspiration_descent_progress("
    "t-immersion_lower_at,immersion_anchor,immersed_goal,smooth)"
)
_INSTALLED = False


def _quintic(value):
    value = min(1.0, max(0.0, float(value)))
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


def _make_descent_clock(profile, rim_z):
    compression = float(profile["far_descent_compression"])
    surface_clearance = float(profile["surface_slow_clearance_m"])
    rim_clearance = float(profile["rim_slow_clearance_m"])

    @lru_cache(maxsize=8)
    def timing(start_z, goal_z):
        distance = start_z - goal_z
        # immersed_goal is the original surface-5mm target. The surface guard
        # therefore needs the unchanged 5mm immersion depth as well.
        guard_z = max(goal_z + 0.005 + surface_clearance, rim_z + rim_clearance)
        if compression == 1.0 or distance <= 0.0 or start_z <= guard_z:
            return 0.0, 0.0
        guarded_progress = min(1.0, (start_z - guard_z) / distance)
        low, high = 0.0, 1.0
        for _ in range(48):
            middle = (low + high) * 0.5
            if _quintic(middle) < guarded_progress:
                low = middle
            else:
                high = middle
        guard_age = 3.0 * (low + high) * 0.5
        fast_duration = guard_age / compression
        return fast_duration, guard_age - fast_duration

    def progress(elapsed, anchor, goal, original_smooth):
        fast_duration, saved_time = timing(float(anchor[2]), float(goal[2]))
        elapsed = float(elapsed)
        if saved_time == 0.0 or elapsed <= 0.0:
            return original_smooth(elapsed / 3.0)
        # s=t+A*H(t/T): s'=1 and s''=0 at both seams. Past the guard,
        # ds/dt=1 exactly, retaining the original near-liquid trajectory.
        shifted = elapsed + saved_time * _quintic(elapsed / fast_duration)
        return original_smooth(shifted / 3.0)

    return progress


def install():
    """Chain the frozen R03/native callables before private source composition."""
    global _INSTALLED
    if _INSTALLED:
        return
    profile = dict(PROFILE)
    duration = float(profile["direct_cup_duration_s"])
    compression = float(profile["far_descent_compression"])
    if not 1.0 <= duration <= 4.0 or not 1.0 <= compression <= 1.5:
        raise ValueError("SpeedAspiration profile exceeds reviewed timing bounds")
    if min(float(profile[key]) for key in ("surface_slow_clearance_m", "rim_slow_clearance_m")) < 0.015:
        raise ValueError("SpeedAspiration requires 15mm guarded rim/surface clearance")

    import t1_right_standing_motion_v6 as motion
    import t1_right_interior_workstation_v6 as interior
    from t1_right_transport_low_v3 import RightTransportLowConfig

    previous_cup = motion.advance_direct_cup_v6
    previous_adapter = interior.adapt_vertical_aspiration_source
    rim_z = RightTransportLowConfig().receiver_rim_z_m
    descent_progress = _make_descent_clock(profile, rim_z)

    @wraps(previous_cup)
    def dry_cup_preposition(m, d, dock, *, out, config=None, duration_s=4.0, settle_s=0.5):
        own_phase = Path(out).name == "v6_right_direct_cup" and float(duration_s) == 4.0
        # Accelerate only a high dry entry. A lower entry keeps the complete
        # original motion, never a faster fine path or an acceptance bypass.
        entry_z = float(d.site_xpos[m.site("pipette/tip_site").id, 2]) if own_phase else None
        dry_guard_passed = own_phase and min(entry_z, 0.93) >= rim_z + 0.030
        selected_duration = duration if dry_guard_passed else duration_s
        arrays, report = previous_cup(
            m, d, dock, out=out, config=config,
            duration_s=selected_duration, settle_s=settle_s,
        )
        if own_phase:
            report["speed_aspiration"] = {
                "profile": dict(profile),
                "motion_duration_s": selected_duration,
                "original_settle_s": settle_s,
                "entry_tip_z_m": entry_z,
                "dry_guard_z_m": rim_z + 0.030,
                "dry_guard_passed": dry_guard_passed,
                "unchanged_goal_z_m": 0.93,
                "unchanged_original_success": report["success"],
                "scope": "Dry high preposition only; immersion uses separate guarded native controller",
            }
        return arrays, report

    @wraps(previous_adapter)
    def guarded_aspiration_source(source, config=None):
        transformed, environment, report = previous_adapter(source, config)
        if transformed.count(DESCENT_ANCHOR) != 1:
            raise RuntimeError("SpeedAspiration requires the unique native IMMERSE descent clock")
        if "_v34_aspiration_descent_progress" in environment:
            raise RuntimeError("SpeedAspiration source environment already contains its timing helper")
        transformed = transformed.replace(DESCENT_ANCHOR, DESCENT_REPLACEMENT, 1)
        environment = dict(environment, _v34_aspiration_descent_progress=descent_progress)
        report = dict(report)
        report["speed_aspiration"] = {
            "profile": dict(profile),
            "source_anchor": DESCENT_ANCHOR,
            "previous_adapter_called_first": True,
            "near_liquid_clock_rate": 1.0,
            "withdrawal_and_ready_hold_unchanged": True,
            "no_ledger_or_button_writes": True,
        }
        return transformed, environment, report

    motion.advance_direct_cup_v6 = dry_cup_preposition
    interior.adapt_vertical_aspiration_source = guarded_aspiration_source
    _INSTALLED = True
