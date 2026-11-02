"""V34PourClosure: bounded relative timing inside the original V10 closure.

Index reaches its existing closed target slightly ahead of middle during the
same arm/hand approach. Both retain the exact original start, final targets,
phase duration, encoder integral/freeze, radial preload and release behavior.
This tests path-dependent acquisition, not a new grasp or postmix pose repair.
"""
from functools import wraps

import numpy as np

CANDIDATE_ID = "V34PourClosure"
MAX_PROGRESS_SHIFT = 0.04
MAX_COMMAND_DELTA_RAD = 0.02
# V10 hand_indices order: thumb[0:5], index[5:9], middle[9:13].
DIGITS = ((slice(5, 9), 1.0), (slice(9, 13), -1.0))
_stats = {"calls": 0, "changed_calls": 0, "maximum_delta_rad": 0.0}
_installed = False


def shape_closure(original, start, closed, u):
    """C2 zero-endpoint relative timing, bounded against the native command."""
    hand = original(start, closed, u)
    start = np.asarray(start)
    closed = np.asarray(closed)
    if hand.shape != (22,) or start.shape != (22,) or closed.shape != (22,):
        raise ValueError("V34PourClosure requires the canonical 22-joint LEFT hand")
    u = float(u)
    if not np.isfinite(u) or not 0.0 <= u <= 1.0:
        raise ValueError("V34PourClosure requires original normalized closure progress")
    # Peak one, zero value/first/second derivatives at both phase endpoints.
    bump = 64.0 * u**3 * (1.0-u)**3
    maximum_delta = 0.0
    for indices, direction in DIGITS:
        span = closed[indices] - start[indices]
        largest_span = float(np.max(np.abs(span)))
        shift = min(MAX_PROGRESS_SHIFT, MAX_COMMAND_DELTA_RAD / max(largest_span, 1e-12))
        shifted_u = u + direction * shift * bump
        # These bounds preserve the original per-joint command segment exactly.
        if not 0.0 <= shifted_u <= 1.0:
            raise ValueError("Relative closure left the original command segment")
        delta = direction * shift * bump * span
        hand[indices] += delta
        maximum_delta = max(maximum_delta, float(np.max(np.abs(delta))))
    _stats["calls"] += 1
    _stats["changed_calls"] += int(maximum_delta > 0.0)
    _stats["maximum_delta_rad"] = max(_stats["maximum_delta_rad"], maximum_delta)
    return hand


def evidence():
    return {
        "candidate_id": CANDIDATE_ID,
        "command_scope": "Original close_on_tube only; index ahead/middle behind, all other joints unchanged",
        "retained_hold_scope": "Original held targets and encoder/preload/freeze behavior; no new hold phase",
        "maximum_progress_shift_per_digit": MAX_PROGRESS_SHIFT,
        "maximum_per_joint_delta_bound_rad": MAX_COMMAND_DELTA_RAD,
        "exact_original_endpoints": True,
        **_stats,
    }


def install():
    global _installed
    if _installed:
        return
    import t1_single_grasp_v10 as single
    original = single.closing_target

    @wraps(original)
    def closing_target(start, closed, u):
        return shape_closure(original, start, closed, u)

    single.closing_target = closing_target
    _installed = True
