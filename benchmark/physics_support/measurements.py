"""Passive measurement gates for the newly identified benchmark judge.

Held lift ports native/t1_right_passive_settle_v5.py:10-14. Mixing is a NEW
actual-motion criterion, conservatively derived from public V27/V28 3 Hz,
+/-5 degree, 1+2+1 s settings; it is not a chemistry model or motion policy.
"""
from __future__ import annotations

import math
import numpy as np

from .contract import CRITERIA, DT

PIPETTE = CRITERIA['pipette']
MIXING = CRITERIA['mixing']


class HeldLift:
    def __init__(self, initial_tip_z):
        self.initial_tip_z = float(initial_tip_z)
        if not math.isfinite(self.initial_tip_z):
            raise ValueError('Initial tip height must be finite')
        self.last_time = None
        self.support_samples = 0
        self.acquired = False
        self.lift_m = 0.
        self.acquired_at_s = None

    def observe(self, now, tip_z, counts, forces):
        counts, forces = np.asarray(counts), np.asarray(forces)
        if counts.shape != (5,) or forces.shape != (5,) or not np.isfinite([now, tip_z, *counts, *forces]).all():
            raise ValueError('Invalid held-lift measurement')
        if self.last_time is not None and abs(float(now) - self.last_time - DT) > 1e-10:
            raise ValueError('Held-lift requires consecutive native samples')
        self.last_time = float(now)
        support = bool(np.all(counts[1:] > 0) and np.all(forces[1:] > PIPETTE['contact_force_N']))
        self.support_samples = self.support_samples + 1 if support else 0
        self.lift_m = float(tip_z - self.initial_tip_z)
        # 500 consecutive 1ms observations represent the native half-second
        # window (the historical floating-clock implementation admitted >=499).
        ready = self.lift_m >= PIPETTE['minimum_tip_lift_m'] and self.support_samples >= PIPETTE['support_samples']
        new = ready and not self.acquired
        if new:
            self.acquired, self.acquired_at_s = True, float(now)
        return new

    def status(self):
        return dict(acquired=self.acquired, acquired_at_s=self.acquired_at_s,
                    actual_tip_lift_m=self.lift_m, required_tip_lift_m=PIPETTE['minimum_tip_lift_m'],
                    consecutive_support_samples=self.support_samples, required_samples=PIPETTE['support_samples'])


class MixingMotion:
    """Recognize real repeated axis motion, never a declared stage or command.

One uninterrupted, force-supported near-pivot window before any pouring: at
least 2 s; >=2.094 rad axis travel (half the ideal 3Hz/5deg/2s plateau);
>=6 significant direction reversals on either transverse tilt coordinate,
with >=5deg peak/trough excursions (half nominal peak-to-peak). The mouth
may move at most 50 mm, matching the source diagnostic envelope. Sub-degree
contact jitter, a static hold, a one-way tilt or translated carry cannot pass.
"""
    duration_s = MIXING['duration_s']
    travel_rad = MIXING['axis_travel_rad']
    excursion_rad = MIXING['reversal_excursion_rad']
    reversals_required = MIXING['significant_reversals']
    mouth_limit_m = MIXING['mouth_excursion_m']

    def __init__(self):
        self.acquired = False
        self.acquired_at_s = None
        self.anchor = self.basis = self.previous_axis = None
        self.started = self.last_time = None
        self.travel = self.max_mouth = 0.
        self.direction = np.zeros(2, dtype=int)
        self.extreme = np.zeros(2)
        self.reversals = np.zeros(2, dtype=int)
        self.best = None

    def _begin(self, now, mouth, rotation):
        self.anchor, self.basis = mouth.copy(), rotation.copy()
        self.previous_axis = rotation[:, 2].copy()
        self.started = self.last_time = now
        self.travel = self.max_mouth = 0.
        self.direction[:] = self.extreme[:] = self.reversals[:] = 0

    def observe(self, now, mouth, rotation, eligible):
        now = float(now)
        mouth, rotation = np.asarray(mouth), np.asarray(rotation)
        if mouth.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite([now, *mouth, *rotation.ravel()]).all():
            raise ValueError('Invalid actual mixing pose')
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('Mixing observations must advance time')
        if not eligible:
            self.anchor = None
            self.last_time = now
            return False
        if self.anchor is None:
            self._begin(now, mouth, rotation)
            return False
        excursion = float(np.linalg.norm(mouth - self.anchor))
        if excursion > self.mouth_limit_m:
            self._begin(now, mouth, rotation)
            return False
        axis = rotation[:, 2]
        self.max_mouth = max(self.max_mouth, excursion)
        # atan2 is stable for tiny actual motion and does not turn dot roundoff
        # near one into accumulating fictitious arccos travel.
        self.travel += float(np.arctan2(np.linalg.norm(np.cross(self.previous_axis, axis)),
                                       np.clip(self.previous_axis @ axis, -1., 1.)))
        local = self.basis.T @ axis
        tilt = np.arctan2(local[:2], local[2])
        for i in range(2):
            if self.direction[i] == 0:
                if abs(tilt[i]) >= self.excursion_rad:
                    self.direction[i] = 1 if tilt[i] > 0 else -1
                    self.extreme[i] = tilt[i]
            elif self.direction[i] * (tilt[i] - self.extreme[i]) >= 0:
                self.extreme[i] = tilt[i]
            elif self.direction[i] * (self.extreme[i] - tilt[i]) >= self.excursion_rad:
                self.reversals[i] += 1
                self.direction[i] *= -1
                self.extreme[i] = tilt[i]
        self.previous_axis, self.last_time = axis.copy(), now
        measured = {'duration_s': now - self.started, 'axis_travel_rad': self.travel,
                    'significant_reversals': int(self.reversals.max()), 'mouth_excursion_m': self.max_mouth}
        if self.best is None or measured['significant_reversals'] > self.best['significant_reversals'] or (measured['significant_reversals'] == self.best['significant_reversals'] and measured['axis_travel_rad'] > self.best['axis_travel_rad']):
            self.best = measured.copy()
        passed = (measured['duration_s'] >= self.duration_s and self.travel >= self.travel_rad and
                  measured['significant_reversals'] >= self.reversals_required)
        new = passed and not self.acquired
        if new:
            self.acquired, self.acquired_at_s = True, now
        return new

    def status(self):
        return {'acquired': self.acquired, 'acquired_at_s': self.acquired_at_s,
                'best_window': self.best, 'material_homogeneity_verified': False}
