"""Exact local-window retiming; no policy selection or runtime installation.

``windows`` and ``transition_s`` are NOMINAL ELAPSED seconds within one named
reference, not geometric path coordinates or episode time. The elapsed clock
runs at one in the windows and at the selected exterior rate (two or four)
elsewhere except adjacent shoulders. A shoulder integrates a quintic rate
over source_width / mean(endpoint_rates). Its rate and first two derivatives
join the constant-rate pieces continuously. Shoulders shrink at domain ends
and gap midpoints; they never enlarge the selected original-rate windows.

The supplied nominal clock is composed piece by piece in Bernstein form, not
fitted or sampled. Spatial coefficients and physical acceptance are untouched.
The only zero elapsed rate is an explicit final native-grid hold of <=1 ms.
The geometric clock can additionally inherit stationary pieces of its source.
Empty windows call the preserved original implementation without any changes.

Numerics: float64 BPoly storage, bracketed normalized inverses, long-double
Bernstein composition. Unresolvable intervals fail explicitly rather than
snapping a source knot, changing geometry, or falling back to another policy.
The old interval enclosures bound derivatives within polynomial pieces, not
impulses at pre-existing discontinuities or a nonstationary terminal hold.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import BPoly
from scipy.optimize import brentq

DT = 0.001
_EPS = np.finfo(float).eps


def _windows(values, duration):
    requested = []
    for pair in values:
        start, end = pair  # Preserve malformed-pair errors, not a guessed span.
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end)
                and 0. <= start < end <= duration):
            raise ValueError('Windows require finite 0 <= start < end <= duration')
        requested.append((start, end))
    requested.sort()
    merged = []
    for start, end in requested:
        if merged and start < merged[-1][1]:
            raise ValueError('Local timing windows must not overlap')
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return requested, merged


def _spans(windows, duration, shoulder, factor):
    """Return an exact source partition; no silently slowed intervening spans."""
    spans = []

    def add(start, end, kind, r0, r1):
        if end > start:
            spans.append(dict(old_start=start, old_end=end, decision=kind,
                              nominal_rate_start=r0, nominal_rate_end=r1))

    first = windows[0][0]
    lead = max(0., first - shoulder)
    add(0., lead, 'accelerated_exterior', factor, factor)
    add(lead, first, 'transition_shoulder', factor, 1.)
    for index, (start, end) in enumerate(windows):
        add(start, end, 'native_rate_window', 1., 1.)
        if index + 1 < len(windows):
            following = windows[index + 1][0]
            middle = end + (following - end) / 2.
            up_end = min(end + shoulder, middle)
            down_start = max(following - shoulder, middle)
            add(end, up_end, 'transition_shoulder', 1., factor)
            add(up_end, down_start, 'accelerated_exterior', factor, factor)
            add(down_start, following, 'transition_shoulder', factor, 1.)
        else:
            up_end = min(duration, end + shoulder)
            add(end, up_end, 'transition_shoulder', 1., factor)
            add(up_end, duration, 'accelerated_exterior', factor, factor)
    # A positive shoulder smaller than a floating-point spacing must not
    # silently disappear, leaving a rate discontinuity instead of C2 joins.
    for left, right in zip(spans[:-1], spans[1:]):
        if (left['old_end'] != right['old_start']
                or left['nominal_rate_end'] != right['nominal_rate_start']):
            raise ValueError('Transition shoulder is below elapsed-time resolution')
    return spans


def _join(pieces, core):
    degree = max(len(piece.c) - 1 for piece in pieces)
    coefficients = np.concatenate([core._elevate(p.c, degree) for p in pieces], axis=1)
    knots = np.r_[np.concatenate([p.x[:-1] for p in pieces]), pieces[-1].x[-1]]
    if (not np.isfinite(coefficients).all() or not np.isfinite(knots).all()
            or np.any(np.diff(knots) <= 0.)):
        raise ValueError('Unresolvable or nonfinite composed polynomial')
    return BPoly(coefficients, knots, extrapolate=False)


def _elapsed_clock(blocks, duration, core):
    pieces, widths = [], []
    resolution = 128. * _EPS * max(1., duration)
    for block in blocks:
        start, end = block['old_start'], block['old_end']
        r0, r1 = block['nominal_rate_start'], block['nominal_rate_end']
        width = (end - start) / ((r0 + r1) / 2.)
        if width <= resolution:
            raise ValueError('Local timing interval is below elapsed-time resolution')
        left = math.fsum(widths)
        widths.append(width)
        right = math.fsum(widths)
        shape = (np.array([0., 1.]) if r0 == r1 else
                 np.r_[0., np.cumsum([r0, r0, r0, r1, r1, r1])] / (3. * (r0+r1)))
        c = (np.longdouble(start) + np.longdouble(end - start)
             * shape.astype(np.longdouble)).astype(float)
        c[0], c[-1] = start, end
        pieces.append(BPoly(c[:, None], [left, right], extrapolate=False))
        block.update(new_start=left, new_end=right)
    return _join(pieces, core), resolution


def _inverse_elapsed(clock, source, source_edges):
    """Invert only elapsed time; inverting geometry loses flat nominal spans."""
    if source <= source_edges[0]:
        return float(clock.x[0])
    if source >= source_edges[-1]:
        return float(clock.x[-1])
    index = int(np.searchsorted(source_edges, source, side='right')) - 1
    if source == source_edges[index]:
        return float(clock.x[index])
    start, end = source_edges[index:index + 2]
    unit = (clock.c[:, index].astype(np.longdouble) - np.longdouble(start)) / (end - start)
    unit = BPoly(np.asarray(unit, dtype=float)[:, None], [0., 1.], extrapolate=False)
    target = (source - start) / (end - start)
    fraction = brentq(lambda x: float(unit(x)) - target, 0., 1.,
                      xtol=4. * _EPS, rtol=4. * _EPS)
    return float(clock.x[index] + fraction * (clock.x[index + 1] - clock.x[index]))


def _product(a, b):
    """Bernstein product, with no ill-conditioned conversion to power basis."""
    n, m = len(a) - 1, len(b) - 1
    result = np.zeros(n + m + 1, dtype=np.longdouble)
    for k in range(n + m + 1):
        denominator = np.longdouble(math.comb(n + m, k))
        for i in range(max(0, k - m), min(n, k) + 1):
            weight = (np.longdouble(math.comb(n, i))
                      * np.longdouble(math.comb(m, k - i)) / denominator)
            result[k] += weight * a[i] * b[k - i]
    return result


def _compose_coefficients(outer, inner):
    # Polynomial de Casteljau: each affine blend becomes two Bernstein products.
    # The inner hull lies in [0,1], so the exact result stays in the outer hull.
    z = np.asarray(inner, dtype=np.longdouble)
    if np.all(outer == outer[0]):
        return np.full((len(outer) - 1) * (len(inner) - 1) + 1, outer[0]), 0.
    rows = [np.array([value], dtype=np.longdouble) for value in outer]
    while len(rows) > 1:
        rows = [_product(a, 1. - z) + _product(b, z)
                for a, b in zip(rows[:-1], rows[1:])]
    result = np.asarray(rows[0], dtype=float)
    low, high = float(np.min(outer)), float(np.max(outer))
    violation = max(0., low - float(np.min(result)), float(np.max(result)) - high)
    tolerance = 64. * len(result) * _EPS * max(1., abs(low), abs(high))
    if not np.isfinite(result).all() or violation > tolerance:
        raise ValueError('Bernstein composition exceeded its analytic convex hull')
    # Only restore a proven coefficient hull after floating-point roundoff;
    # never clip a sampled trajectory or alter the supplied spatial reference.
    return np.clip(result, low, high), violation


def _compose_clock(baseline, elapsed, source_edges, resolution, core):
    from v34_smoothtubereturn_bounds import _restrict_coefficients

    pieces, maximum_correction, normalization_correction = [], 0., 0.
    for index, (a, b) in enumerate(zip(source_edges[:-1], source_edges[1:])):
        source = np.r_[a, baseline.x[(baseline.x > a) & (baseline.x < b)], b]
        executed = [_inverse_elapsed(elapsed, float(s), source_edges) for s in source]
        for low, high, left, right in zip(source[:-1], source[1:], executed[:-1], executed[1:]):
            if high - low <= resolution or right - left <= resolution:
                raise ValueError('Nominal knot and window edge are numerically inseparable')
            j = int(np.searchsorted(baseline.x, low, side='right')) - 1
            j = min(j, len(baseline.x) - 2)
            nominal_width = baseline.x[j + 1] - baseline.x[j]
            time_width = elapsed.x[index + 1] - elapsed.x[index]
            u = 0. if left == elapsed.x[index] else (left - elapsed.x[index]) / time_width
            v = 1. if right == elapsed.x[index + 1] else (right - elapsed.x[index]) / time_width
            restricted = _restrict_coefficients(elapsed.c[:, index], float(u), float(v))
            normalized = ((restricted.astype(np.longdouble) - np.longdouble(baseline.x[j]))
                          / nominal_width).astype(float)
            # Bracket inversion and restriction have finite roundoff. The
            # exact inner interval lies in this one nominal polynomial piece.
            error = max(0., -float(normalized.min()), float(normalized.max()) - 1.)
            allowed = 256. * _EPS * max(1., abs(a), abs(b)) / nominal_width
            if not np.isfinite(normalized).all() or error > allowed:
                raise ValueError('Composed elapsed clock escapes its nominal piece')
            first = (low - baseline.x[j]) / nominal_width
            last = (high - baseline.x[j]) / nominal_width
            normalization_correction = max(normalization_correction, nominal_width * max(
                error, abs(float(normalized[0]) - first), abs(float(normalized[-1]) - last)))
            normalized = np.clip(normalized, 0., 1.)
            normalized[0], normalized[-1] = first, last
            c, correction = _compose_coefficients(baseline.c[:, j], normalized)
            maximum_correction = max(maximum_correction, correction)
            pieces.append(BPoly(c[:, None], [left, right], extrapolate=False))
    return _join(pieces, core), maximum_correction, float(normalization_correction)


def _native_end(reference_duration):
    ticks = max(1, math.ceil(reference_duration / DT))
    duration = ticks * DT
    if duration < reference_duration:
        ticks += 1
        duration = ticks * DT
    if ticks > 1 and (ticks - 1) * DT >= reference_duration:
        duration = (ticks - 1) * DT
    hold = duration - reference_duration
    if not 0. <= hold <= DT:
        raise ValueError('Terminal native-grid hold is not representable within 1 ms')
    return duration, hold


def _hold(clock, duration, value):
    if duration == clock.x[-1]:
        return clock
    return BPoly(np.c_[clock.c, np.full(len(clock.c), value)],
                 np.r_[clock.x, duration], extrapolate=False)


@dataclass
class WindowTiming:
    """Timing-compatible fields plus an invertible nominal-elapsed clock.

    ``factor`` means exterior rate, not a global scale. ``reference_duration``
    ends motion; ``duration`` includes the final hold. Like the original API,
    inverse(source_duration) returns duration (the right end of that hold).
    """
    path: BPoly
    clock: BPoly
    baseline_clock: BPoly
    duration: float
    source_duration: float
    blocks: list
    report: dict
    identity_source_clock: bool
    factor: float
    reference_duration: float
    elapsed_clock: BPoly
    _motion_elapsed_clock: BPoly
    _source_edges: np.ndarray

    def __call__(self, time, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('Only position, velocity and acceleration are supported')
        values = np.asarray(time, dtype=float)
        t = np.clip(values, 0., self.duration)
        source = self.clock(t)
        if derivative == 0:
            return self.path(source)
        rate = np.asarray(self.clock(t, 1))[..., None]
        first = self.path(source, 1)
        result = first * rate
        if derivative == 2:
            result = self.path(source, 2) * rate**2 + first * np.asarray(self.clock(t, 2))[..., None]
        outside = (values < 0.) | (values > self.reference_duration)
        return np.where(outside[..., None], 0., result)

    def source_time(self, time):
        values = np.asarray(time, dtype=float)
        result = np.asarray(self.elapsed_clock(np.clip(values, 0., self.duration)))
        result = np.clip(result, 0., self.source_duration)
        return float(result) if result.ndim == 0 else result

    def new_time_for_source(self, time):
        values = np.clip(np.asarray(time, dtype=float), 0., self.source_duration)
        if np.isnan(values).any():
            raise ValueError('NaN nominal elapsed time cannot be inverted')
        result = np.asarray([
            self.duration if source >= self.source_duration else
            _inverse_elapsed(self._motion_elapsed_clock, float(source), self._source_edges)
            for source in values.reshape(-1)
        ]).reshape(values.shape)
        return float(result) if result.ndim == 0 else result


def retime_local(original_retime, path, *, duration, label, windows, transition_s=0.2, **kwargs):
    """Compose a preserved uncapped retimer with explicit native-rate windows.

    All original kwargs/errors and callback classification go through the
    preserved function. Nonempty windows accept exterior factor 2 or 4.
    Empty windows return its object untouched, including intentional factor=1 calls.
    The existing report dict is updated in place so the original record sink
    retains the FINAL clock rather than the intermediate validation clock.
    """
    windows = tuple(windows)
    if not windows:
        return original_retime(path, duration=duration, label=label, **kwargs)
    duration, transition_s = float(duration), float(transition_s)
    if not math.isfinite(duration) or duration <= 0.:
        raise ValueError('Positive finite source duration required')
    if not math.isfinite(transition_s) or transition_s <= 0.:
        raise ValueError('Positive finite transition_s required for C2 shoulders')
    exterior_factor = float(kwargs.get('factor', 2.))
    if exterior_factor not in (2., 4.):
        raise ValueError('Nonempty local windows require exterior factor 2 or 4')
    requested, merged = _windows(windows, duration)
    base = original_retime(path, duration=duration, label=label, **kwargs)
    import v34_topp_core as core

    baseline = base.baseline_clock
    if (baseline.x[0] != 0. or baseline.x[-1] != duration
            or np.any(np.diff(baseline.x) <= 0.)):
        raise ValueError('Local composition requires an exact nominal [0,duration] domain')
    blocks = _spans(merged, duration, transition_s, exterior_factor)
    motion_elapsed, resolution = _elapsed_clock(blocks, duration, core)
    source_edges = np.r_[[block['old_start'] for block in blocks], duration]
    rate_coefficients = motion_elapsed.derivative().c
    rate_low, rate_high = float(rate_coefficients.min()), float(rate_coefficients.max())
    allowance = 1024. * _EPS * max(1., duration) / float(np.min(np.diff(motion_elapsed.x)))
    if (not np.isfinite(rate_coefficients).all() or rate_low <= 0.
            or rate_low < 1. - allowance or rate_high > exterior_factor + allowance):
        raise ValueError('Elapsed-clock rate lacks a finite positive Bernstein enclosure')
    moving_clock, correction, normalization = _compose_clock(
        baseline, motion_elapsed, source_edges, resolution, core)
    reference_duration = float(motion_elapsed.x[-1])
    native_duration, hold = _native_end(reference_duration)
    clock = _hold(moving_clock, native_duration, float(baseline(duration)))
    elapsed = _hold(motion_elapsed, native_duration, duration)
    bounds = core._Certificate(base.path, clock).partition()
    endpoint = float(moving_clock(reference_duration))
    terminal_velocity = base.path(endpoint, 1) * float(moving_clock(reference_duration, 1))
    terminal_acceleration = (base.path(endpoint, 2) * float(moving_clock(reference_duration, 1))**2
                             + base.path(endpoint, 1) * float(moving_clock(reference_duration, 2)))
    shoulders = [dict(block) for block in blocks if block['decision'] == 'transition_shoulder']
    exterior = [[block['old_start'], block['old_end']] for block in blocks
                if block['decision'] == 'accelerated_exterior']
    if hold > 0.:
        blocks.append(dict(old_start=duration, old_end=duration,
                           new_start=reference_duration, new_end=native_duration,
                           decision='terminal_grid_hold', nominal_rate_start=0., nominal_rate_end=0.))
    report = dict(base.report)
    report.update(
        method='local_native_rate_windows_exact_bernstein_composition',
        duration_s=native_duration, requested_reference_duration_s=reference_duration,
        requested_factor=None, exterior_reference_factor=exterior_factor, window_reference_factor=1.,
        factor_field_meaning='exterior rate; no single global factor',
        average_motion_reference_factor=duration / reference_duration,
        saved_s=duration - native_duration, terminal_grid_hold_s=hold, native_timestep_s=DT,
        requested_windows_source_elapsed_s=[list(pair) for pair in requested],
        native_rate_windows_source_elapsed_s=[list(pair) for pair in merged],
        transition_source_width_s=transition_s, transition_shoulders=shoulders,
        exterior_intervals_source_elapsed_s=exterior,
        window_coordinate_basis='nominal elapsed seconds within this named reference',
        transition_method='quintic rate, integrated degree-six elapsed clock; C2 rate joins',
        source_clock_basis='unchanged supplied nominal clock composed with explicit local elapsed clock',
        path_preservation='original geometric polynomial object; no spatial fitting or modification',
        elapsed_rate_analytic_bounds_before_hold=[1., exterior_factor],
        elapsed_rate_bernstein_bounds_before_hold=[rate_low, rate_high],
        elapsed_rate_roundoff_allowance=allowance,
        finite_positive_elapsed_rate_before_hold=True,
        polynomial_composition_max_hull_roundoff_correction=correction,
        nominal_piece_max_normalization_roundoff_s=normalization,
        minimum_resolvable_interval_s=resolution,
        all_time_velocity_bound=np.asarray(bounds['velocity']).tolist(),
        all_time_acceleration_bound=np.asarray(bounds['acceleration']).tolist(),
        derivative_enclosure_method='original interval Bernstein certificate on final BPoly',
        derivative_enclosure_scope='within polynomial pieces; no impulsive-derivative or physical-success certificate',
        terminal_left_velocity=np.asarray(terminal_velocity).tolist(),
        terminal_left_acceleration=np.asarray(terminal_acceleration).tolist(),
        clock_knots=clock.x.tolist(), clock_coefficients=clock.c.tolist(),
        source_elapsed_clock_knots=elapsed.x.tolist(), source_elapsed_clock_coefficients=elapsed.c.tolist(),
        blocks=blocks, software_speed_limits_enforced=False, old_topp_certificate_valid=False,
        physical_limits_unchanged=True, physical_acceptance_unchanged=True, physical_execution_passed=None)
    base.report.clear()
    base.report.update(report)
    return WindowTiming(base.path, clock, baseline, native_duration, duration, blocks,
                        base.report, base.identity_source_clock, exterior_factor, reference_duration,
                        elapsed, motion_elapsed, source_edges)
