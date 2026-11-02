"""Exact R08 RIGHT reference polynomials; no sampled path fitting.

Only command progress is retimed. These envelopes are not joint/torque proofs:
all native measured-tool IK, command-rate rejection and actuator caps remain
in their original controllers. VelocityPath parts retain their original-time
meaning for audit; sample() returns the executed reference and its derivatives.
"""
from __future__ import annotations

import ast
import inspect
import linecache
import math
import textwrap
from itertools import count

import numpy as np
from scipy.interpolate import BPoly, PPoly, make_interp_spline

import v34_topp_core as core

_COPY_IDS = count()


def exact(source, anchor, replacement):
    found = source.count(anchor)
    if found != 1:
        raise RuntimeError(f"TOPP RIGHT requires one native anchor ({found}): {anchor!r}")
    return source.replace(anchor, replacement, 1)


def closure(function):
    return dict(zip(function.__code__.co_freevars,
                    (cell.cell_contents for cell in function.__closure__ or ())))


def copy_callable(function, replacements=(), *, cells=None, globals_=None, label):
    """Copy the immediate callable, not inspect's __wrapped__ target.

    This is the native private-source convention. Capture the SAME callback,
    observer and mutable geometry objects; never mutate a shared source/global
    dictionary or a simulator clock. Keeping original closure names preserves
    the native actual-dock replan's explicit old/a/frames traversal.
    """
    source = textwrap.dedent(inspect.getsource(function.__code__))
    node = ast.parse(source).body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise RuntimeError("Expected a native function definition")
    source = "\n".join(source.splitlines()[node.lineno - 1:]) + "\n"
    for anchor, replacement in replacements:
        source = exact(source, anchor, replacement)
    captured = closure(function)
    captured.update(cells or {})
    factory = ("def _bind(" + ",".join(captured) + "):\n"
               + textwrap.indent(source, "    ")
               + "    return " + node.name + "\n")
    filename = f"{__file__} [{label}:{next(_COPY_IDS)}]"
    linecache.cache[filename] = (len(factory), None, factory.splitlines(True), filename)
    environment = dict(function.__globals__, **(globals_ or {}))
    exec(compile(factory, filename, "exec"), environment)
    result = environment["_bind"](**captured)
    result.__dict__.update(function.__dict__)
    result.__name__ = function.__name__
    result.__qualname__ = function.__qualname__
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def affine_clock(duration, offset=0.):
    return BPoly(np.asarray([[offset], [offset + duration]], dtype=float),
                 [0., duration], extrapolate=False)


def shifted_clock(duration, fast, saved, start=0.):
    """Exact t + saved*H((t-start)/fast), including both unit-rate tails."""
    if saved == 0.:
        return affine_clock(duration)
    if not (0. <= start < start + fast <= duration):
        raise ValueError("Invalid native quintic-clock interval")
    breaks = [0.]
    values = [[0., 1., 0.]]
    if start:
        breaks.append(start)
        values.append([start, 1., 0.])
    breaks.append(start + fast)
    values.append([start + fast + saved, 1., 0.])
    if start + fast < duration:
        breaks.append(duration)
        values.append([duration + saved, 1., 0.])
    return BPoly.from_derivatives(breaks, values, extrapolate=False)


def derivative_enclosure(path, clock=None):
    """Conservative native command-speed ceiling, not an actuator limit.

    Used only where R08 supplies no Cartesian reference-speed ceiling. Native
    explicit precision/retract limits are passed separately and never enlarged.
    """
    curve = core.as_polynomial(path)
    bounds = np.max(np.abs(curve.derivative().c), axis=(0, 1))
    if clock is not None:
        bounds = bounds * float(np.max(np.abs(clock.derivative().c)))
    bounds = np.asarray(bounds, dtype=float)
    return np.where(bounds > 0., np.nextafter(bounds, np.inf), 1.)


def prefix_path(profile):
    native = profile["continuous_approach"]
    times = np.asarray(native["time_s"], dtype=float)
    q = np.asarray(native["arm_q_rad"], dtype=float)
    if times.shape != (2,) or not np.array_equal(times, [.7, 5.8]) or q.shape[0] != 2:
        raise RuntimeError("Expected the R08 two-knot actual-stopped-dock prefix")
    zero = np.zeros(q.shape[1])
    spline = make_interp_spline(times, q, k=5,
                               bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]))
    curve = core.as_polynomial(spline)
    if curve.x[0] != .7 or curve.x[-1] != 5.8:
        raise RuntimeError("Native acquisition spline domain changed")
    hold = np.broadcast_to(q[0], (curve.c.shape[0], 1, q.shape[1])).copy()
    return BPoly(np.concatenate((hold, curve.c), axis=1),
                 np.r_[0., curve.x], extrapolate=False)


def acquisition_task_path(profile):
    times, states = [0.], [[0., .8, 0.]]
    logical = 0.
    for name, delta in (("seat_s", 1.5), ("close_s", 2.), ("hold_s", 1.),
                        ("lift_s", 3.), ("postlift_s", 1.), ("verticalize_s", .8)):
        times.append(times[-1] + float(profile[name]))
        logical += delta
        states.append([logical, .8, 0.])
    return BPoly.from_derivatives(times, states, extrapolate=False)


def inverse_quintic(fraction):
    if fraction <= 0.:
        return 0.
    if fraction >= 1.:
        return 1.
    lo, hi = 0., 1.
    for _ in range(64):
        mid = (lo + hi) * .5
        value = mid**3 * (10. + mid * (-15. + 6. * mid))
        if value < fraction:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * .5


def smooth(value):
    value = min(1., max(0., float(value)))
    return value**3 * (10. + value * (-15. + 6. * value))


def velocity_polynomial(parts):
    """Integrate each actual VelocityPath phase analytically, degree six.

    x(t)=d+v0*t+(v1-v0)*(2.5*t^4/T^3-3*t^5/T^4+t^6/T^5).
    This is exactly native _integrated_smooth, not a Hermite/sample fit.
    """
    if not parts:
        raise ValueError("An empty native path has no polynomial interval")
    coefficients, breaks = [], [float(parts[0][0])]
    for start, duration, distance, v0, v1, phase in parts:
        start, duration = float(start), float(duration)
        if not math.isfinite(start + duration) or duration <= 0. or abs(start - breaks[-1]) > 1e-12:
            raise RuntimeError("Discontinuous native VelocityPath partition")
        delta = float(v1 - v0)
        coefficients.append([delta / duration**5, -3. * delta / duration**4,
                             2.5 * delta / duration**3, 0., 0., float(v0), float(distance)])
        breaks.append(start + duration)
    return BPoly.from_power_basis(PPoly(np.asarray(coefficients).T, breaks, extrapolate=False))


class RetimedVelocityPath:
    """Native sample(age) API with per-phase bounds and through-speed joins."""

    def __init__(self, native, *, label, limits):
        self.native = native
        self.parts = native.parts
        self.distance = native.distance
        self.velocity = native.velocity
        self.source_duration = native.duration
        self.duration = 0.
        self.blocks = []
        self.report = dict(parts_time_basis="original native reference seconds",
                           geometry="exact degree-six integral of native velocity phase coefficients",
                           nonzero_internal_velocity_preserved=True, blocks=[])
        if not native.parts:
            core.note_fixed(label, reason="Exact zero-distance native reference; no motion interval",
                            duration=0., kind="zero_length_reference")
            return
        groups = []
        for part in native.parts:
            phase = part[-1]
            if phase not in limits:
                raise RuntimeError("Unsupported native velocity phase: " + phase)
            if not groups or groups[-1][0] != phase:
                groups.append((phase, []))
            groups[-1][1].append(part)
        for phase, parts in groups:
            source_start = float(parts[0][0])
            source_end = float(parts[-1][0] + parts[-1][1])
            span = source_end - source_start
            curve = velocity_polynomial(parts)
            velocity, acceleration = limits[phase]
            timing = core.retime(curve, duration=span, label=label + "/" + phase,
                                 clock=affine_clock(span, source_start),
                                 vlim=velocity, alim=acceleration)
            self.blocks.append((self.duration, source_start, source_end, phase, timing))
            self.duration += timing.duration
            self.report["blocks"].append(dict(phase=phase, source_start_s=source_start,
                                               source_end_s=source_end, executed_duration_s=timing.duration,
                                               timing=timing.report))

    def new_time_for_source(self, source):
        source = float(source)
        if source <= 0.:
            return 0.
        for start, source_start, source_end, phase, timing in self.blocks:
            if source <= source_end:
                return start + timing.new_time_for_source(source - source_start)
        return self.duration

    def sample(self, age):
        age = float(age)
        if not self.blocks:
            return self.native.sample(age)
        if age <= 0.:
            return self.native.sample(age)
        if age >= self.duration:
            return self.distance, self.velocity, 0., self.parts[-1][-1]
        for start, source_start, source_end, phase, timing in self.blocks:
            if age < start + timing.duration:
                elapsed = age - start
                return (float(timing(elapsed)[0]), float(timing(elapsed, 1)[0]),
                        float(timing(elapsed, 2)[0]), phase)
        raise RuntimeError("Retimed VelocityPath has an uncovered interval")


def retimed_transfer(previous, profile, distance):
    native, split, through = previous(distance)
    path = RetimedVelocityPath(native, label="right/precision_transfer", limits={
        "clear_transfer": (profile.clear_speed_m_s, profile.clear_acceleration_m_s2),
        "fine_transfer": (profile.fine_speed_m_s, profile.fine_acceleration_m_s2),
    })
    return path, path.new_time_for_source(split), through


def retimed_retract(native, split, profile):
    path = RetimedVelocityPath(native, label="right/retract", limits={
        "retract_initial": (profile.retract_initial_speed_m_s, profile.retract_initial_acceleration_m_s2),
        "retract_clear": (profile.retract_clear_speed_m_s, profile.retract_clear_acceleration_m_s2),
    })
    return path, path.new_time_for_source(split)
