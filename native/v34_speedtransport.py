"""V34 mobile-only route timing and fast/slow speed-envelope candidate.

The original coarse controller retains its 10 mm latch, 0.4 s physical brake,
existing wheel-servo policy, and continuously stopped 0.5 s handoff gate. This
module only shapes the moving reference duration and approach speed limit.
Earlier docks and every post-latch controller call use their original path.
The mobile-only one-second nominal route lead-in is removed AFTER the native
aspiration/readiness handoff, not by advancing the physical clock or a gate.
"""
from contextvars import ContextVar
from functools import wraps
import inspect
import math
import textwrap

import numpy as np


_ACTIVE_TRANSFER = ContextVar("v34_speedtransport_active", default=None)
PROFILE = {
    "profile_id": "aggressive_r07_01",
    "initial_route_hold_s": 0.0,
    "route_s": 1.0,
    "far_speed_limit_m_s": 0.3,
    "near_speed_limit_m_s": 0.035,
    "deceleration_radius_m": 0.16,
    "slow_radius_m": 0.025,
    "speed_limit_slew_m_s2": 0.4
}


def _smoothstep(u):
    u = min(1.0, max(0.0, float(u)))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _retimed_native_transfer(original, profile):
    """Private native loop copy; no shared runner file or physics step replacement."""
    if original.__code__.co_name != "advance_mobile_stage":
        raise RuntimeError("Expected native coarse transfer loop before R07 retiming")
    source = textwrap.dedent(inspect.getsource(original))
    for before, after in (
        ("(elapsed-1)/route_s", "(elapsed-" + repr(profile["initial_route_hold_s"]) + ")/route_s"),
        ("enable_gate=elapsed>=1+route_s", "enable_gate=elapsed>=" + repr(profile["initial_route_hold_s"]) + "+route_s"),
    ):
        if source.count(before) != 1:
            raise RuntimeError("SpeedTransport exact native route anchor required: " + before)
        source = source.replace(before, after, 1)
    namespace = dict(original.__globals__)
    exec(compile(source, __file__ + " [private native transfer reference]", "exec"), namespace)
    return namespace[original.__code__.co_name]


def install():
    import t1_coarse_stop as coarse

    old_advance = coarse.advance_coarse_stop
    if getattr(old_advance, "_v34_speedtransport", False):
        return
    old_step = coarse.CoarseStopController.step
    profile = dict(PROFILE)
    if not all(math.isfinite(v) for v in profile.values() if isinstance(v, (float, int))):
        raise ValueError("Nonfinite RIGHT transport profile")
    if not 0.0 <= profile["initial_route_hold_s"] <= .25 or not .8 <= profile["route_s"] <= 1.25:
        raise ValueError("Transport reference outside declared aggressive bracket")
    fixed = dict(far_speed_limit_m_s=.3, near_speed_limit_m_s=.035,
                 deceleration_radius_m=.16, slow_radius_m=.025, speed_limit_slew_m_s2=.4)
    if any(profile[key] != value for key, value in fixed.items()):
        raise ValueError("R06 physical approach envelope must remain unchanged")
    old_inner = coarse._advance
    timed_inner = _retimed_native_transfer(old_inner, profile)

    @wraps(old_inner)
    def inner(*args, **kwargs):
        scope = _ACTIVE_TRANSFER.get()
        data = args[1] if len(args) > 1 else kwargs.get("d")
        target = args[2] if len(args) > 2 else kwargs.get("target_xy")
        if scope is None or scope["data"] is not data or not np.array_equal(target, scope["target"]):
            return old_inner(*args, **kwargs)
        return timed_inner(*args, **kwargs)

    @wraps(old_step)
    def step(self, data, target_xy=None, enable_gate=True):
        scope = _ACTIVE_TRANSFER.get()
        if (scope is None or scope["data"] is not data
                or self.mode != "approach"
                or not np.array_equal(self.station_xy, scope["target"])):
            return old_step(self, data, target_xy, enable_gate=enable_gate)

        # Only the native post-aspiration transfer owns this context. The
        # position-error controller itself is unchanged, and the coarse latch
        # still irreversibly removes world-position feedback at first entry.
        distance = float(np.linalg.norm(data.qpos[self.qa:self.qa + 2]
                                        - self.station_xy))
        blend = _smoothstep((distance - profile["slow_radius_m"])
                            / (profile["deceleration_radius_m"]
                               - profile["slow_radius_m"]))
        requested_limit = (profile["near_speed_limit_m_s"]
                           + blend * (profile["far_speed_limit_m_s"]
                                      - profile["near_speed_limit_m_s"]))
        now = float(data.time)
        dt = max(0.0, now - scope["last_time"])
        previous_limit = scope["speed_limit"]
        bound = profile["speed_limit_slew_m_s2"] * dt
        limit = max(previous_limit - bound,
                    min(previous_limit + bound, requested_limit))
        scope["last_time"] = now
        scope["speed_limit"] = limit
        scope["minimum_limit_m_s"] = min(scope["minimum_limit_m_s"], limit)
        scope["maximum_limit_m_s"] = max(scope["maximum_limit_m_s"], limit)
        scope["approach_steps"] += 1
        if distance <= profile["deceleration_radius_m"]:
            if scope["deceleration_started_at_s"] is None:
                scope["deceleration_started_at_s"] = now
            scope["deceleration_steps"] += 1
        if distance <= profile["slow_radius_m"]:
            if scope["slow_region_entered_at_s"] is None:
                scope["slow_region_entered_at_s"] = now
            scope["slow_steps"] += 1
        old_limit = self.speed_limit
        self.speed_limit = limit
        try:
            stopped = old_step(self, data, target_xy, enable_gate=enable_gate)
        finally:
            self.speed_limit = old_limit
        if self.mode != "approach":
            scope["latched_speed_m_s"] = float(
                np.linalg.norm(data.qvel[self.dv:self.dv + 2]))
            scope["latched_limit_m_s"] = limit
        return stopped

    @wraps(old_advance)
    def advance(*args, **kwargs):
        # This name is taken from the frozen executed_integrated_engine.py,
        # not guessed from target coordinates or absolute physical time.
        hold = kwargs.get("hold_callback", args[5] if len(args) > 5 else None)
        if getattr(hold, "__name__", None) != "transport_hold":
            return old_advance(*args, **kwargs)
        data = args[1] if len(args) > 1 else kwargs["d"]
        target = args[2] if len(args) > 2 else kwargs["target_xy"]
        route = list(args)
        changed = dict(kwargs)
        if len(route) > 4:
            route[4] = profile["route_s"]
        else:
            changed["route_s"] = profile["route_s"]
        if len(route) > 7:
            route[7] = profile["far_speed_limit_m_s"]
        else:
            changed["speed_limit"] = profile["far_speed_limit_m_s"]
        scope = dict(data=data, target=np.asarray(target),
                     last_time=float(data.time),
                     speed_limit=profile["near_speed_limit_m_s"],
                     minimum_limit_m_s=profile["near_speed_limit_m_s"],
                     maximum_limit_m_s=profile["near_speed_limit_m_s"],
                     approach_steps=0, deceleration_steps=0, slow_steps=0,
                     deceleration_started_at_s=None,
                     slow_region_entered_at_s=None,
                     latched_speed_m_s=None, latched_limit_m_s=None)
        token = _ACTIVE_TRANSFER.set(scope)
        try:
            logs, audit, dock = old_advance(*route, **changed)
        finally:
            _ACTIVE_TRANSFER.reset(token)
        audit["v34_speedtransport"] = dict(
            profile=profile,
            scope="post-aspiration mobile transfer only; transport_hold callback",
            original_initial_hold_s=1.0,
            executed_initial_route_hold_s=profile["initial_route_hold_s"],
            prior_aspiration_and_base_predicates_unchanged=True,
            original_route_reference="quintic minimum jerk",
            cap_schedule="C2 actual-distance blend with bounded per-step slew",
            original_coarse_latch_and_brake_retained=True,
            original_stopped_half_second_gate_retained=True,
            fine_base_alignment_added=False,
            earlier_docks_changed=False,
            **{key: value for key, value in scope.items()
               if key not in ("data", "target", "last_time", "speed_limit")})
        return logs, audit, dock

    advance._v34_speedtransport = True
    coarse._advance = inner
    coarse.CoarseStopController.step = step
    coarse.advance_coarse_stop = advance
