"""Local Bernstein enclosures for q(t)=Q(s(t)); no sampled-speed acceptance.

De Casteljau restriction preserves the polynomial. A Bernstein curve lies in
its coefficient hull, so interval products of Q'(s(I)) and s'(I) enclose every
joint velocity on I. Unsafe bounds are refined or retimed, never accepted.
"""
import numpy as np
from scipy.interpolate import BPoly, BSpline, PPoly

_EPS = np.finfo(float).eps


def as_bernstein(curve):
    """Canonical polynomial used for BOTH execution and its subsequent bound."""
    if isinstance(curve, BPoly):
        return curve
    if not isinstance(curve, BSpline) or curve.axis != 0 or curve.c.ndim != 2:
        raise TypeError('Expected a vector BPoly or axis-zero BSpline')
    columns = []
    for joint in range(curve.c.shape[1]):
        power = PPoly.from_spline((curve.t, curve.c[:, joint], curve.k))
        bernstein = BPoly.from_power_basis(power)
        keep = np.diff(bernstein.x) > 0.0
        columns.append(bernstein.c[:, keep])
    breaks = np.r_[bernstein.x[:-1][keep], bernstein.x[1:][keep][-1]]
    return BPoly(np.stack(columns, axis=-1), breaks)


def _split(coefficients, fraction):
    work = np.asarray(coefficients).copy()
    left, right = [work[0].copy()], [work[-1].copy()]
    for _ in range(len(coefficients) - 1):
        work = (1.0 - fraction) * work[:-1] + fraction * work[1:]
        left.append(work[0].copy())
        right.append(work[-1].copy())
    return np.asarray(left), np.asarray(right[::-1])


def _restrict_coefficients(coefficients, low, high):
    if not 0.0 <= low <= high <= 1.0:
        raise RuntimeError('Invalid Bernstein restriction interval')
    if low == high:
        point = _split(coefficients, low)[0][-1]
        return np.broadcast_to(point, coefficients.shape).copy()
    result = coefficients
    if low > 0.0:
        result = _split(result, low)[1]
    if high < 1.0:
        result = _split(result, (high - low) / (1.0 - low))[0]
    return result


def _pieces(curve, low, high):
    if not curve.x[0] <= low <= high <= curve.x[-1]:
        raise RuntimeError('Polynomial bound requested outside its declared domain')
    first = min(len(curve.x) - 2, max(0, int(np.searchsorted(curve.x, low, side='right')) - 1))
    last = min(len(curve.x) - 2, int(np.searchsorted(curve.x, high, side='left')))
    for i in range(first, last + 1):
        a, b = max(float(low), float(curve.x[i])), min(float(high), float(curve.x[i + 1]))
        if a > b:
            continue
        length = float(curve.x[i + 1] - curve.x[i])
        if length <= 0.0:
            raise RuntimeError('Zero-width polynomial interval')
        u = 0.0 if a == curve.x[i] else (a - curve.x[i]) / length
        v = 1.0 if b == curve.x[i + 1] else (b - curve.x[i]) / length
        yield a, b, _restrict_coefficients(curve.c[:, i], float(u), float(v)), curve.c[:, i]


def restrict_curve(curve, low, high):
    """Exact polynomial slice, used to retain untouched clearance geometry."""
    pieces = [(a, b, c) for a, b, c, _ in _pieces(curve, low, high) if b > a]
    if not pieces:
        raise RuntimeError('Empty polynomial slice')
    return BPoly(np.stack([row[2] for row in pieces], axis=1),
                 np.r_[[row[0] for row in pieces], pieces[-1][1]])


def _hull(curve, low, high):
    lower, upper = None, None
    for _, _, coefficients, original in _pieces(curve, low, high):
        # Each restriction uses two degree<=5 convex de Casteljau triangles.
        # Inflate from the ORIGINAL coefficient magnitude, not a cancelled
        # local result; outward rounding is not a motion/acceptance tolerance.
        padding = 64.0 * len(original) * _EPS * np.maximum(1.0, np.abs(original).max(axis=0))
        lo = np.nextafter(coefficients.min(axis=0) - padding, -np.inf)
        hi = np.nextafter(coefficients.max(axis=0) + padding, np.inf)
        lower = lo if lower is None else np.minimum(lower, lo)
        upper = hi if upper is None else np.maximum(upper, hi)
    if lower is None or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RuntimeError('Unresolved or nonfinite polynomial enclosure')
    return lower, upper


class BernsteinRateBounds:
    """Prepared derivative polynomials, shared by timing and environment sweep."""

    def __init__(self, spatial, clock):
        self.spatial = spatial
        self.clock = clock
        self.spatial_derivative = spatial.derivative()
        self.clock_derivative = clock.derivative()
        self.cache = {}
        if (not np.isfinite(spatial.c).all() or not np.isfinite(clock.c).all()
                or clock.c.min() < spatial.x[0] or clock.c.max() > spatial.x[-1]):
            raise RuntimeError('Clock/reference domain is not enclosed')

    def bound(self, start, end):
        key = (float(start), float(end))
        if key in self.cache:
            return self.cache[key]
        s_low, s_high = _hull(self.clock, *key)
        # The clock coefficient hull above is independently known to lie in
        # the spatial domain. Intersecting enclosures is not command clipping.
        s_low = max(float(self.spatial.x[0]), float(s_low))
        s_high = min(float(self.spatial.x[-1]), float(s_high))
        q_low, q_high = _hull(self.spatial_derivative, s_low, s_high)
        v_low, v_high = _hull(self.clock_derivative, *key)
        products = np.stack((q_low * v_low, q_low * v_high, q_high * v_low, q_high * v_high))
        upper = np.abs(products).max(axis=0)
        upper = np.nextafter(upper + 8.0 * _EPS * np.maximum(1.0, upper), np.inf)
        if not np.isfinite(upper).all():
            raise RuntimeError('Nonfinite joint-rate enclosure')
        self.cache[key] = upper
        return upper

    def partition(self, limit, refinement_dt):
        """Return enclosing intervals; caller MUST reject/retime any bound>limit.

        Native-tick width only ends refinement of an unsafe timing proposal.
        Such an interval is NOT certified: it still forces retiming, followed
        by a new all-time proof. No samples substitute for a bound.
        """
        pending = [(float(a), float(b), 0) for a, b in zip(self.clock.x[:-1], self.clock.x[1:])]
        leaves = []
        while pending:
            start, end, depth = pending.pop()
            bound = self.bound(start, end)
            if float(bound.max()) <= limit or end - start <= refinement_dt:
                leaves.append((start, end, bound))
                continue
            middle = .5 * (start + end)
            if depth >= 32 or not start < middle < end:
                raise RuntimeError('Unresolved all-time velocity certification')
            pending.extend(((start, middle, depth + 1), (middle, end, depth + 1)))
        leaves.sort(key=lambda row: row[0])
        return leaves
