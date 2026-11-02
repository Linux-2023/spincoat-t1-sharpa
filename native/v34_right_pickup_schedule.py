"""One command clock for RIGHT seat, fingers, collar escape and lift.

These are reference seconds, never physical/observer seconds. Original hold,
postlift and final passive-support windows are not compressed.
"""
from __future__ import annotations

import math

PROFILE = {
    "name": "aggressive_r07_01",
    "home_s": 0.7,
    "entry_saved_s": 1.6,
    "entry_native_s": 5.1,
    "entry_native_tail_s": 1.5,
    "preshape_start_s": 0.7,
    "preshape_duration_s": 3.2,
    "seat_s": 1.5,
    "close_s": 1.75,
    "hold_s": 1.25,
    "lift_s": 3.0,
    "postlift_s": 1.25,
    "verticalize_s": 1.0,
    "arm_rate_limit_rad_s": 2.0,
    "hand_rate_limit_rad_s": 3.0,
}


def smooth(value):
    value = min(1.0, max(0.0, float(value)))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def validate_profile():
    if any(not math.isfinite(v) for v in PROFILE.values() if isinstance(v, (int, float))):
        raise ValueError("Nonfinite RIGHT reference profile")
    ranges = {"entry_saved_s": (1.2, 1.8), "seat_s": (1.25, 1.875),
              "close_s": (1.6, 2.5), "lift_s": (2.75, 3.75)}
    for key, (low, high) in ranges.items():
        if not low <= PROFILE[key] <= high:
            raise ValueError("RIGHT timing exceeds declared bracket: " + key)
    fixed = {"home_s": .7, "entry_native_s": 5.1, "entry_native_tail_s": 1.5,
             "preshape_start_s": .7, "preshape_duration_s": 3.2,
             "hold_s": 1.25, "postlift_s": 1.25, "verticalize_s": 1.,
             "arm_rate_limit_rad_s": 2., "hand_rate_limit_rad_s": 3.}
    for key, value in fixed.items():
        if PROFILE[key] != value:
            raise ValueError("Protected RIGHT profile value changed: " + key)
    if PROFILE["preshape_start_s"] + PROFILE["preshape_duration_s"] > prefix_duration():
        raise ValueError("Preshape must finish before physical seating begins")


def prefix_duration():
    return PROFILE["home_s"] + PROFILE["entry_native_s"] - PROFILE["entry_saved_s"]


def entry_clock(seconds):
    """Same two-knot joint chord; native final1.5s is reproduced at unit rate."""
    fast = PROFILE["entry_native_s"] - PROFILE["entry_native_tail_s"] - PROFILE["entry_saved_s"]
    return float(seconds) + PROFILE["entry_saved_s"] * smooth((seconds - PROFILE["home_s"]) / fast)


def task_clock(seconds):
    """C2 monotone clock with native0.8 logical-s/s at every operation seam.

    Each span adds its saved logical progress via a quintic. Thus the slot
    escape crossing close/hold uses exactly the same C2 clock as the fingers,
    rather than the contradictory old elapsed/1.25 clock.
    """
    seconds = float(seconds)
    start = logical = 0.0
    for name, delta in (("seat_s", 1.5), ("close_s", 2.0), ("hold_s", 1.0),
                        ("lift_s", 3.0), ("postlift_s", 1.0)):
        duration = PROFILE[name]
        if seconds < start + duration:
            elapsed = seconds - start
            return logical + .8 * elapsed + (delta - .8 * duration) * smooth(elapsed / duration)
        start += duration
        logical += delta
    return logical + .8 * (seconds - start)


def timing():
    result = {"seat_start_s": prefix_duration()}
    elapsed = prefix_duration()
    for duration, key in (("seat_s", "finger_close_start_s"), ("close_s", "hold_start_s"),
                          ("hold_s", "lift_start_s"), ("lift_s", "postlift_hold_start_s"),
                          ("postlift_s", "verticalize_start_s"), ("verticalize_s", "stage_end_s")):
        elapsed += PROFILE[duration]
        result[key] = elapsed
    return result
