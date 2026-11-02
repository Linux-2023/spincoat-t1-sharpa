"""Path-preserving TOPP-RA with C2 scalar clocks and interval certificates.

This changes reference timing only. Physical clocks, actuator limits and task
acceptance remain owned by the native workflow. Acceleration ceilings derived
from a reference are engineering constraints, not hardware certification.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import BPoly, BSpline, PPoly
from scipy.optimize import brentq
import toppra as ta
import toppra.algorithm as algorithm
import toppra.constraint as constraint

from v34_smoothtubereturn_bounds import _hull, as_bernstein, restrict_curve

RECORDS = []
QUANTUM = .001


def as_polynomial(curve):
    if isinstance(curve, BPoly):
        result = curve
    elif isinstance(curve, PPoly):
        result = BPoly.from_power_basis(curve)
    elif isinstance(curve, BSpline):
        result = as_bernstein(curve)
    else:
        raise TypeError('TOPP requires an exact polynomial reference, not fitted samples')
    if result.axis != 0 or result.c.ndim not in (2, 3):
        raise ValueError('Expected scalar or axis-zero vector polynomial')
    if result.c.ndim == 2:
        result = BPoly(result.c[..., None], result.x.copy())
    if not np.isfinite(result.c).all() or np.any(np.diff(result.x) <= 0):
        raise ValueError('Invalid polynomial coefficients/domain')
    return result


def quintic(start, end, duration):
    start, end = np.atleast_1d(start).astype(float), np.atleast_1d(end).astype(float)
    if duration <= 0 or start.shape != end.shape:
        raise ValueError('Invalid quintic endpoints/duration')
    zero = np.zeros_like(start)
    return BPoly.from_derivatives([0., float(duration)], [[start, zero, zero], [end, zero, zero]])


def _product(a, b):
    products = np.stack((a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]))
    padding = 16*np.finfo(float).eps*np.maximum(1., np.abs(products).max(axis=0))
    return np.nextafter(products.min(axis=0)-padding, -np.inf), np.nextafter(products.max(axis=0)+padding, np.inf)


def _absolute(interval):
    return np.maximum(np.abs(interval[0]), np.abs(interval[1]))


def _roundoff_allowance(limit):
    # Equality of analytically saturated references is represented in float64.
    # Keep this separate from the unchanged physical limits and expose it.
    return None if limit is None else 8192*np.finfo(float).eps*np.maximum(1., np.abs(limit))


class _Certificate:
    def __init__(self, path, clock):
        self.path, self.clock = path, clock
        self.p1, self.p2 = path.derivative(), path.derivative(2)
        self.c1, self.c2 = clock.derivative(), clock.derivative(2)
        if clock.c.min() < path.x[0] or clock.c.max() > path.x[-1]:
            raise ValueError('Clock coefficient hull escapes the original path')

    def bound(self, a, b):
        slo, shi = _hull(self.clock, a, b)
        slo = max(float(self.path.x[0]), float(slo))
        shi = min(float(self.path.x[-1]), float(shi))
        p1, p2 = _hull(self.p1, slo, shi), _hull(self.p2, slo, shi)
        c1, c2 = _hull(self.c1, a, b), _hull(self.c2, a, b)
        v = _product(p1, c1)
        sq = (0. if c1[0] <= 0 <= c1[1] else min(c1[0]**2, c1[1]**2), max(c1[0]**2, c1[1]**2))
        aa, ab = _product(p2, sq), _product(p1, c2)
        acceleration = (np.nextafter(aa[0]+ab[0], -np.inf), np.nextafter(aa[1]+ab[1], np.inf))
        return _absolute(v), _absolute(acceleration)

    def partition(self, vlim=None, alim=None, *, initial_width=.02, minimum_width=.00025):
        v_roundoff, a_roundoff = _roundoff_allowance(vlim), _roundoff_allowance(alim)
        pending = [(float(a), float(b)) for a, b in zip(self.clock.x[:-1], self.clock.x[1:])]
        leaves = []
        while pending:
            a, b = pending.pop()
            velocity, acceleration = self.bound(a, b)
            if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
                raise RuntimeError('Nonfinite derivative enclosure')
            unsafe = ((vlim is not None and np.any(velocity > vlim+v_roundoff)) or
                      (alim is not None and np.any(acceleration > alim+a_roundoff)))
            if unsafe:
                # Samples can disprove feasibility, never prove acceptance.
                # Avoid subdividing a genuinely over-limit constant span millions of times.
                times = np.asarray([a, (a+b)/2, b])
                source = self.clock(times)
                rates = self.c1(times)[..., None]
                first = self.p1(source)
                actual_v = np.abs(first*rates)
                actual_a = np.abs(self.p2(source)*rates**2+first*self.c2(times)[..., None])
                disproved = ((vlim is not None and np.any(actual_v > vlim+v_roundoff)) or
                             (alim is not None and np.any(actual_a > alim+a_roundoff)))
                if disproved:
                    return dict(passed=False, strict_interval_passed=False,
                        velocity=velocity, acceleration=acceleration,
                        intervals=[(a, b, velocity, acceleration)],
                        velocity_roundoff=v_roundoff, acceleration_roundoff=a_roundoff,
                        full_partition_completed=False,
                        counterexample=dict(times=times.tolist(), velocity=actual_v.tolist(), acceleration=actual_a.tolist()))
            if b-a > initial_width or (unsafe and b-a > minimum_width):
                middle = (a+b)/2
                if not a < middle < b:
                    raise RuntimeError('Unresolved interval enclosure')
                pending.extend(((a, middle), (middle, b)))
            else:
                leaves.append((a, b, velocity, acceleration))
        leaves.sort(key=lambda row: row[0])
        velocity = np.max([row[2] for row in leaves], axis=0)
        acceleration = np.max([row[3] for row in leaves], axis=0)
        strict = ((vlim is None or np.all(velocity <= vlim)) and
                  (alim is None or np.all(acceleration <= alim)))
        passed = ((vlim is None or np.all(velocity <= vlim+v_roundoff)) and
                  (alim is None or np.all(acceleration <= alim+a_roundoff)))
        return dict(passed=bool(passed), strict_interval_passed=bool(strict),
                    velocity=velocity, acceleration=acceleration, intervals=leaves,
                    velocity_roundoff=v_roundoff, acceleration_roundoff=a_roundoff,
                    full_partition_completed=True, counterexample=None)


def _limits(value, dimensions, name):
    array = np.broadcast_to(np.asarray(value, dtype=float), (dimensions,)).copy()
    if np.any(array <= 0) or not np.isfinite(array).all():
        raise ValueError('Invalid '+name)
    return array


def _elevate(coefficients, degree):
    original = len(coefficients)-1
    if degree == original:
        return coefficients
    result = np.zeros((degree+1,)+coefficients.shape[1:])
    for i in range(degree+1):
        for j in range(max(0, i-degree+original), min(original, i)+1):
            result[i] += math.comb(original, j)*math.comb(degree-original, i-j)/math.comb(degree, i)*coefficients[j]
    return result


def _concatenate(clocks):
    degree = max(len(clock.c)-1 for clock in clocks)
    return BPoly(np.concatenate([_elevate(clock.c, degree) for clock in clocks], axis=1),
                 np.r_[np.concatenate([clock.x[:-1] for clock in clocks]), clocks[-1].x[-1]])


def _shift(clock, start):
    return BPoly(clock.c.copy(), clock.x-clock.x[0]+start)


def _monotone(clock):
    # A nonnegative derivative coefficient hull certifies the whole interval.
    return bool(np.min(clock.derivative().c) >= -1e-12)


def _inverse(clock, value):
    values = np.asarray(clock(clock.x), dtype=float)
    if value <= values[0]:
        return float(clock.x[0])
    if value >= values[-1]:
        return float(clock.x[-1])
    i = min(len(clock.x)-2, max(0, int(np.searchsorted(values, value, side='right'))-1))
    return float(brentq(lambda time: float(clock(time))-value, float(clock.x[i]), float(clock.x[i+1]), xtol=2e-13))


@dataclass
class Timing:
    path: BPoly
    clock: BPoly
    baseline_clock: BPoly
    duration: float
    source_duration: float
    blocks: list
    report: dict
    identity_source_clock: bool

    def __call__(self, time, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('Only position, velocity and acceleration are supported')
        t = np.clip(np.asarray(time, dtype=float), 0., self.duration)
        source = self.clock(t)
        if derivative == 0:
            return self.path(source)
        speed = np.asarray(self.clock(t, 1))[..., None]
        result = self.path(source, 1)*speed
        if derivative == 2:
            result = self.path(source, 2)*speed**2+self.path(source, 1)*np.asarray(self.clock(t, 2))[..., None]
        outside = (np.asarray(time) < 0) | (np.asarray(time) > self.duration)
        return np.where(np.asarray(outside)[..., None], 0., result)

    def source_time(self, time):
        array = np.asarray(time, dtype=float)
        def one(value):
            value = float(np.clip(value, 0., self.duration))
            for block in self.blocks:
                if value <= block['new_end']+1e-12:
                    if block['decision'] != 'retimed':
                        return float(np.clip(block['old_start']+value-block['new_start'], block['old_start'], block['old_end']))
                    source = float(self.clock(value))
                    return source if self.identity_source_clock else _inverse(self.baseline_clock, source)
            return self.source_duration
        result = np.asarray([one(v) for v in array.reshape(-1)]).reshape(array.shape)
        return float(result) if result.ndim == 0 else result

    def new_time_for_source(self, time):
        value = float(np.clip(time, 0., self.source_duration))
        for block in self.blocks:
            if value <= block['old_end']+1e-12:
                if block['decision'] != 'retimed':
                    return block['new_start']+value-block['old_start']
                return _inverse(self.clock, float(self.baseline_clock(value)))
        return self.duration


class _AugmentedPath(ta.interpolator.AbstractGeometricPath):
    """Normalize solver coordinates only; physical references stay untouched."""
    def __init__(self, path, start, end, scale):
        self.path = path
        self._interval = np.asarray([start, end], dtype=float)
        self._scale = np.asarray(scale, dtype=float)
        self._offset = np.r_[path(start), 0.]

    @property
    def dof(self):
        return self.path.c.shape[-1]+1

    @property
    def path_interval(self):
        return self._interval

    def __call__(self, values, order=0):
        values = np.asarray(values, dtype=float)
        joint = self.path(values, order)
        clock_coordinate = values-self._interval[0] if order == 0 else np.ones_like(values) if order == 1 else np.zeros_like(values)
        result = np.concatenate((joint, clock_coordinate[..., None]), axis=-1)
        return (result-self._offset if order == 0 else result)/self._scale


def _free_block(path, baseline, start, end, vlim, alim, maximum_rate, minimum, accept_clock):
    original = restrict_curve(baseline, start, end)
    span = end-start
    s0, s1 = float(baseline(start)), float(baseline(end))
    common = dict(old_start=start, old_end=end, original_duration=span)
    if s1-s0 <= 1e-13 or minimum >= span-1e-12:
        return original, dict(common, decision='fixed', reason='Zero progress or binding minimum-duration constraint')
    cvel = _hull(baseline.derivative(), start, end)
    cacc = _hull(baseline.derivative(2), start, end)
    clock_speed = maximum_rate*max(float(cvel[1]), (s1-s0)/span)
    clock_acceleration = max(float(_absolute(cacc)), 4*clock_speed/span)
    velocity = np.r_[vlim, clock_speed]
    acceleration = np.r_[alim, clock_acceleration]
    augmented = _AugmentedPath(path, s0, s1, velocity)
    normalized_velocity = np.ones_like(velocity)
    normalized_acceleration = acceleration/velocity
    points = np.unique(np.r_[np.linspace(s0, s1, 161), path.x[(path.x > s0) & (path.x < s1)]])
    solver = algorithm.TOPPRA([
        constraint.JointVelocityConstraint(np.c_[-normalized_velocity, normalized_velocity]),
        constraint.JointAccelerationConstraint(np.c_[-normalized_acceleration, normalized_acceleration], discretization_scheme=constraint.DiscretizationType.Interpolation)
    ], augmented, gridpoints=points, solver_wrapper='seidel', parametrizer='ParametrizeConstAccel')
    sdd, sd, _ = solver.compute_parameterization(max(0., float(baseline(start, 1))), max(0., float(baseline(end, 1))))
    if sd is None or sdd is None or not np.isfinite(sd).all() or not np.isfinite(sdd).all():
        raise RuntimeError('TOPP-RA failed: '+str(solver.problem_data.return_code))
    denominators = sd[:-1]+sd[1:]
    if np.any(denominators <= 0):
        raise RuntimeError('Untraversable TOPP interval')
    raw_dt = 2*np.diff(points)/denominators
    raw_t = np.r_[0., np.cumsum(raw_dt)]
    raw_duration = float(raw_t[-1])
    nodal_acceleration = np.zeros_like(sd)
    if len(sd) > 2:
        nodal_acceleration[1:-1] = (sdd[:-1]*raw_dt[1:]+sdd[1:]*raw_dt[:-1])/(raw_dt[:-1]+raw_dt[1:])
    nodal_acceleration[0], nodal_acceleration[-1] = baseline(start, 2), baseline(end, 2)
    raw_clock = BPoly.from_derivatives(raw_t, [[float(s), float(v), float(a)]
                                              for s, v, a in zip(points, sd, nodal_acceleration)])
    unit = np.unique(np.r_[raw_t/raw_duration, (original.x-start)/span])
    details = dict(common, solver='toppra0.6.10/seidel', raw_topp_duration=raw_duration,
                   raw_path_grid=points.tolist(), raw_path_velocity=sd.tolist(), raw_path_acceleration=sdd.tolist(),
                   clock_speed_limit=clock_speed, clock_acceleration_limit=clock_acceleration,
                   solver_coordinate_scale=velocity.tolist(),
                   solver_coordinate_scope='Velocity-limit coordinate scaling; original physical inequalities and path retained',
                   smoothing='C2 scalar-clock homotopy toward the original feasible clock; exact spatial polynomial retained')
    candidates = []
    for fraction in (0., .02, .05, .1, .2, .4, .65, .85, .95, .98, .995):
        requested = max(minimum, raw_duration*1.005+(span-raw_duration*1.005)*fraction)
        savings = math.floor((span-requested)/QUANTUM+1e-10)*QUANTUM
        if savings < QUANTUM/2:
            continue
        duration = span-savings
        if duration <= 0 or duration >= span:
            continue
        blend = min(1., max(0., (duration-raw_duration)/(span-raw_duration)))
        ts = unit*duration
        derivatives = [(1-blend)*raw_clock(unit*raw_duration, order)*(raw_duration/duration)**order
                       + blend*baseline(start+unit*span, order)*(span/duration)**order
                       for order in (0, 1, 2)]
        derivatives[0][0], derivatives[0][-1] = s0, s1
        for order in (1, 2):
            derivatives[order][0], derivatives[order][-1] = baseline(start, order), baseline(end, order)
        clock = BPoly.from_derivatives(ts, np.column_stack(derivatives).tolist())
        if (not _monotone(clock) or clock.c.min() < path.x[0] or clock.c.max() > path.x[-1]):
            candidates.append(dict(duration=duration, rejected='clock_monotonicity_or_domain'))
            continue
        certificate = _Certificate(path, clock).partition(vlim, alim)
        extra_passed = None
        accepted = certificate['passed']
        if accepted and accept_clock is not None:
            extra_passed = bool(accept_clock(path, clock))
            accepted = extra_passed
        candidates.append(dict(duration=duration, baseline_clock_blend=blend,
                               passed=accepted, derivative_certificate_passed=certificate['passed'],
                               additional_clock_constraint_passed=extra_passed,
                               strict_interval_passed=certificate['strict_interval_passed'],
                               velocity_bound=certificate['velocity'].tolist(), acceleration_bound=certificate['acceleration'].tolist()))
        if accepted:
            return clock, dict(details, decision='retimed', duration=duration, proposals=candidates)
    return original, dict(details, decision='constrained_no_gain', duration=span,
                          reason='No faster C2 clock certified within declared bounds; original feasible clock retained', proposals=candidates)


def retime(path, *, duration, label, vlim, clock=None, alim=None, protected=(), maximum_rate=1.5,
           minimum_duration=None, accept_clock=None):
    """Certify derivative limits and any caller-owned clock constraint before selection."""
    path = as_polynomial(path)
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(maximum_rate) or maximum_rate < 1:
        raise ValueError('Invalid source duration/rate cap')
    identity = clock is None
    if clock is None:
        clock = BPoly(np.asarray([[0.], [duration]]), [0., duration])
    if not isinstance(clock, BPoly) or clock.c.ndim != 2 or abs(clock.x[0]) > 1e-12 or abs(clock.x[-1]-duration) > 1e-10:
        raise ValueError('Expected scalar baseline clock over [0,duration]')
    if not _monotone(clock):
        raise ValueError('Original clock is not certified monotone')
    velocity = _limits(vlim, path.c.shape[-1], 'velocity ceiling')
    baseline = _Certificate(path, clock).partition()
    acceleration = _limits(np.maximum(1.01*baseline['acceleration'], 1e-8) if alim is None else alim,
                           path.c.shape[-1], 'acceleration ceiling')
    baseline_proof = _Certificate(path, clock).partition(velocity, acceleration, minimum_width=1e-7)
    if not baseline_proof['passed']:
        raise ValueError('Original reference violates requested ceilings; reconcile constraints before retiming')
    if accept_clock is not None and not accept_clock(path, clock):
        raise ValueError('Original reference violates the additional native clock constraint')
    intervals = []
    for a, b in sorted((float(a), float(b)) for a, b in protected):
        if not 0 <= a < b <= duration:
            raise ValueError('Protected interval outside original elapsed-time domain')
        if intervals and a <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(b, intervals[-1][1]))
        else:
            intervals.append((a, b))
    minimum = 0. if minimum_duration is None else float(minimum_duration)
    if not 0 <= minimum <= duration+1e-10:
        raise ValueError('Minimum duration must be within the original reference duration')
    boundaries = sorted(set([0., duration]+[value for interval in intervals for value in interval]))
    chunks = [(a, b, any(low <= a and b <= high for low, high in intervals)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    free_total = sum(b-a for a, b, fixed in chunks if not fixed)
    fixed_total = duration-free_total
    free_minimum_ratio = min(1., max(0., minimum-fixed_total)/free_total) if free_total else 1.
    pieces, blocks = [], []
    elapsed = 0.
    for a, b, fixed in chunks:
        if fixed:
            piece, report = restrict_curve(clock, a, b), dict(old_start=a, old_end=b, decision='fixed', reason='Protected original physical/reference clock')
        else:
            piece, report = _free_block(path, clock, a, b, velocity, acceleration, maximum_rate,
                                        (b-a)*free_minimum_ratio, accept_clock)
        length = float(piece.x[-1]-piece.x[0])
        shifted = _shift(piece, elapsed)
        if pieces:
            for order in (0, 1, 2):
                if abs(float(pieces[-1](pieces[-1].x[-1], order))-float(shifted(shifted.x[0], order))) > 2e-8:
                    raise RuntimeError('Retimed scalar clock lost C2 boundary continuity')
        pieces.append(shifted)
        report.update(new_start=elapsed, new_end=elapsed+length)
        blocks.append(report)
        elapsed += length
    combined = _concatenate(pieces)
    if not _monotone(combined):
        raise RuntimeError('Combined clock is not monotone')
    proof = _Certificate(path, combined).partition(velocity, acceleration, minimum_width=1e-7)
    if not proof['passed']:
        raise RuntimeError('Combined clock lacks an all-time derivative certificate')
    if accept_clock is not None and not accept_clock(path, combined):
        raise RuntimeError('Combined clock violates the additional native clock constraint')
    report = dict(label=str(label), source_duration_s=duration, duration_s=elapsed, saved_s=duration-elapsed,
                  method='TOPP-RA with certified C2 scalar-clock reconstruction', path_preservation='Exact polynomial; no fitted joint samples',
                  velocity_limits=velocity.tolist(), acceleration_limits=acceleration.tolist(),
                  acceleration_limit_source='Original reference Bernstein enclosure times1.01; additional engineering ceiling, not hardware rating' if alim is None else 'Explicit existing/declared stage ceiling',
                  all_time_velocity_bound=proof['velocity'].tolist(), all_time_acceleration_bound=proof['acceleration'].tolist(),
                  certificate='Outward-inflated Bernstein intervals with separately declared float64 equality allowance; unsafe leaves rejected',
                  strict_interval_passed=proof['strict_interval_passed'],
                  velocity_roundoff_allowance=proof['velocity_roundoff'].tolist(),
                  acceleration_roundoff_allowance=proof['acceleration_roundoff'].tolist(),
                  physical_limits_unchanged=True,
                  additional_clock_constraint_applied=accept_clock is not None,
                  certificate_intervals=[[a, b, v.tolist(), acc.tolist()] for a, b, v, acc in proof['intervals']],
                  baseline_velocity_bound=baseline['velocity'].tolist(), baseline_acceleration_bound=baseline['acceleration'].tolist(),
                  protected_source_time_intervals=intervals, maximum_source_rate_factor=maximum_rate,
                  clock_knots=combined.x.tolist(), clock_coefficients=combined.c.tolist(), blocks=blocks,
                  physical_execution_passed=None, physics_parameters_or_gates_changed=False)
    RECORDS.append(report)
    return Timing(path, combined, clock, elapsed, duration, blocks, report, identity)


def note_fixed(label, *, reason, duration=None, kind='physical_constraint'):
    RECORDS.append(dict(label=str(label), decision='fixed_native_clock', kind=str(kind), reason=str(reason),
                        source_duration_s=None if duration is None else float(duration), physical_execution_passed=None))


def save_report(directory):
    path = Path(directory)/'topp_report.json'
    temporary = path.with_suffix('.json.writing')
    temporary.write_text(json.dumps(dict(schema_version=1, implementation='toppra0.6.10/seidel', records=RECORDS,
        scope='Reference planning and whole-workflow fixed-clock coverage; not a physical-success verdict'), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
    return path
