"""Read-only pipette aspiration observer and conservative volume bookkeeping.

This is a kinematic volume proxy, NOT fluid simulation, pipette calibration,
or a robot controller. It never writes MuJoCo qpos, ctrl, or object poses.
The caller supplies measured button/tip samples and source geometry. Ground
truth adapters are for scripted simulation/validation, not visual policy input.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path


class Stage(str, Enum):
    PRESS = "press_above_liquid"
    IMMERSE = "immerse_with_button_pressed"
    RELEASE = "release_while_immersed"
    WITHDRAW = "withdraw_after_aspiration"
    READY = "aspiration_complete"
    FAILED = "failed"


@dataclass(frozen=True)
class AspirationParameters:
    capacity_m3: float = 30e-9
    metering_stroke_m: float = .005
    released_qpos_tolerance_m: float = .0002
    min_immersion_m: float = .001
    bottom_clearance_m: float = .001
    radial_clearance_m: float = .001
    press_hold_s: float = .08
    immersion_hold_s: float = .10
    aspiration_hold_s: float = .15
    withdrawal_hold_s: float = .10
    withdrawal_clearance_m: float = .005
    min_fill_fraction: float = .95
    reverse_motion_tolerance_m: float = .0001
    timeout_s: float = 30.

    def __post_init__(self):
        positive = (self.capacity_m3, self.metering_stroke_m,
                    self.min_immersion_m, self.bottom_clearance_m,
                    self.timeout_s)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Capacity, stroke, clearances and timeout must be positive")
        if not 0 < self.min_fill_fraction <= 1:
            raise ValueError("min_fill_fraction must be in (0,1]")
        if not 0 <= self.released_qpos_tolerance_m < self.metering_stroke_m:
            raise ValueError("Release tolerance must be below metering stroke")
        nonnegative = (self.radial_clearance_m, self.press_hold_s,
                       self.immersion_hold_s, self.aspiration_hold_s,
                       self.withdrawal_hold_s, self.withdrawal_clearance_m,
                       self.reverse_motion_tolerance_m)
        if any(not math.isfinite(v) or v < 0 for v in nonnegative):
            raise ValueError("Holds/tolerances must be finite and nonnegative")


@dataclass(frozen=True)
class SourceGeometry:
    center_xy_m: tuple[float, float]
    inner_radius_m: float
    bottom_z_m: float
    initial_volume_m3: float
    # Cylinder proxy. Geometry must be supplied from the assembled scene.
    initial_surface_z_m: float

    def __post_init__(self):
        values = (*self.center_xy_m, self.inner_radius_m, self.bottom_z_m,
                  self.initial_volume_m3, self.initial_surface_z_m)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Source geometry must be finite")
        if self.inner_radius_m <= 0 or self.initial_volume_m3 <= 0:
            raise ValueError("Source radius/volume must be positive")
        if self.initial_surface_z_m <= self.bottom_z_m:
            raise ValueError("Liquid surface must be above source bottom")


class AspirationStage:
    """Observe every physics/control sample; no preloaded liquid is allowed.

    update returns enable_dispense, remaining_volume_m3, and events. The former
    means aspiration/withdrawal is complete; the task must ALSO gate base settle,
    relative visual localization and coater readiness before allow_dispense=True.
    Recorded subsequent compression debits liquid even outside a valid target;
    such volume is charged to spilled_m3 rather than silently restored.
    """

    def __init__(self, source: SourceGeometry,
                 parameters: AspirationParameters | None = None):
        self.source = source
        self.p = parameters or AspirationParameters()
        if source.inner_radius_m <= self.p.radial_clearance_m:
            raise ValueError("Source too narrow for configured clearance")
        if source.initial_volume_m3 < self.p.capacity_m3:
            raise ValueError("Source cannot supply the configured dose")
        self.state = Stage.PRESS
        self.failure = None
        self.remaining_volume_m3 = 0.
        self.source_remaining_m3 = source.initial_volume_m3
        self.aspirated_m3 = self.delivered_m3 = self.spilled_m3 = 0.
        self.events = []
        self._time0 = self._last_time = self._last_q = None
        self._hold = 0.
        self._press_q = None
        self._max_release_fraction = 0.

    def liquid_surface_z_m(self):
        # Conservative cylinder depletion from the configured initial surface.
        area = math.pi * self.source.inner_radius_m ** 2
        return max(self.source.bottom_z_m,
                   self.source.initial_surface_z_m - self.aspirated_m3 / area)

    def _event(self, t, kind, **details):
        event = {"time_s": float(t), "kind": kind, **details}
        self.events.append(event)

    def _transition(self, t, state):
        self.state = state
        self._hold = 0.
        self._event(t, "state", state=state.value)

    def _fail(self, t, reason):
        self.failure = reason
        self._transition(t, Stage.FAILED)
        self._event(t, "failure", code=reason)

    def update(self, time_s: float, button_qpos_m: float, tip_xyz_m,
               *, allow_dispense: bool = False):
        """Observe actual samples. Negative button qpos means compression.

        allow_dispense is an external task gate, not a command to the pipette.
        Return incremental delivered/spilled volume for the caller's fluid model.
        Always log source+pipette+delivered+spilled for conservation checks.
        """
        t, q = float(time_s), float(button_qpos_m)
        xyz = tuple(float(v) for v in tip_xyz_m)
        if len(xyz) != 3 or not all(math.isfinite(v) for v in (t, q, *xyz)):
            raise ValueError("Finite time, button qpos and 3-D tip required")
        if self._last_time is not None and t <= self._last_time:
            raise ValueError("Samples must have strictly increasing timestamps")
        previous_event_count = len(self.events)
        delivered_before, spilled_before = self.delivered_m3, self.spilled_m3
        dt = 0. if self._last_time is None else t-self._last_time
        if self._time0 is None:
            self._time0 = t
        compression = max(0., min(self.p.metering_stroke_m, -q))
        surface = self.liquid_surface_z_m()
        radial = math.hypot(xyz[0]-self.source.center_xy_m[0],
                            xyz[1]-self.source.center_xy_m[1])
        aligned = radial <= self.source.inner_radius_m-self.p.radial_clearance_m
        immersed = (aligned and
                    self.source.bottom_z_m+self.p.bottom_clearance_m <= xyz[2]
                    <= surface-self.p.min_immersion_m)
        dry_above = xyz[2] >= surface+self.p.withdrawal_clearance_m
        pressed = q <= -self.p.metering_stroke_m
        released = q >= -self.p.released_qpos_tolerance_m
        if self.state not in (Stage.READY, Stage.FAILED) and t-self._time0 > self.p.timeout_s:
            self._fail(t, "aspiration_timeout")
        if self.state == Stage.PRESS:
            # Immersing before prepress is rejected, not mistaken for aspiration.
            if immersed:
                self._fail(t, "immersion_before_prepress")
            else:
                self._hold = self._hold+dt if pressed and dry_above else 0.
                if self._hold >= self.p.press_hold_s:
                    self._press_q = q
                    self._transition(t, Stage.IMMERSE)
        elif self.state == Stage.IMMERSE:
            if not pressed:
                self._fail(t, "button_released_before_stable_immersion")
            else:
                self._hold = self._hold+dt if immersed else 0.
                if self._hold >= self.p.immersion_hold_s:
                    self._transition(t, Stage.RELEASE)
        elif self.state == Stage.RELEASE:
            if not immersed:
                self._fail(t, "tip_left_liquid_during_release")
            elif self._last_q is not None and q < self._last_q-self.p.reverse_motion_tolerance_m:
                self._fail(t, "recompression_during_aspiration")
            else:
                fraction = 1.-compression/self.p.metering_stroke_m
                fraction = max(self._max_release_fraction, min(1., fraction))
                requested = (fraction-self._max_release_fraction)*self.p.capacity_m3
                taken = min(requested, self.source_remaining_m3,
                            self.p.capacity_m3-self.remaining_volume_m3)
                self.remaining_volume_m3 += taken
                self.source_remaining_m3 -= taken
                self.aspirated_m3 += taken
                self._max_release_fraction = fraction
                if taken > 0:
                    self._event(t, "aspirated", volume_m3=taken)
                full = self.remaining_volume_m3 >= self.p.capacity_m3*self.p.min_fill_fraction
                self._hold = self._hold+dt if released and full else 0.
                if self._hold >= self.p.aspiration_hold_s:
                    self._transition(t, Stage.WITHDRAW)
        elif self.state == Stage.WITHDRAW:
            if not released:
                self._fail(t, "recompression_before_withdrawal")
            else:
                self._hold = self._hold+dt if dry_above else 0.
                if self._hold >= self.p.withdrawal_hold_s:
                    self._transition(t, Stage.READY)
                    self._event(t, "aspiration_ready", volume_m3=self.remaining_volume_m3)
        elif self.state == Stage.READY and self._last_q is not None:
            previous_compression = max(0., min(self.p.metering_stroke_m, -self._last_q))
            displaced = max(0., compression-previous_compression)
            volume = min(self.remaining_volume_m3,
                         displaced/self.p.metering_stroke_m*self.p.capacity_m3)
            self.remaining_volume_m3 -= volume
            if volume > 0:
                if allow_dispense:
                    self.delivered_m3 += volume
                    self._event(t, "dispensed", volume_m3=volume)
                else:
                    self.spilled_m3 += volume
                    self._event(t, "uncommanded_discharge", volume_m3=volume)
        # Failure freezes the ledger; the failed episode must stop or hand off to
        # an explicit recovery model. No subsequent fill/dispense is credited.
        self._last_time, self._last_q = t, q
        audit = self.audit()
        return {"stage": self.state.value, "failure": self.failure,
                "enable_dispense": self.state == Stage.READY and self.remaining_volume_m3 > 0.,
                "remaining_volume_m3": self.remaining_volume_m3,
                "source_remaining_m3": self.source_remaining_m3,
                "delivered_increment_m3": self.delivered_m3-delivered_before,
                "spilled_increment_m3": self.spilled_m3-spilled_before,
                "events": self.events[previous_event_count:],
                "mass_balance_error_m3": audit["mass_balance_error_m3"]}

    def audit(self):
        total = (self.source_remaining_m3+self.remaining_volume_m3+
                 self.delivered_m3+self.spilled_m3)
        return {"model": "kinematic_stroke_volume_proxy_not_resolved_fluid",
                "initial_pipette_volume_m3": 0.,
                "initial_source_volume_m3": self.source.initial_volume_m3,
                "source_remaining_m3": self.source_remaining_m3,
                "remaining_volume_m3": self.remaining_volume_m3,
                "aspirated_m3": self.aspirated_m3,
                "delivered_m3": self.delivered_m3,
                "spilled_m3": self.spilled_m3,
                "mass_balance_error_m3": total-self.source.initial_volume_m3,
                "stage": self.state.value, "failure": self.failure,
                "parameters": asdict(self.p), "source_geometry": asdict(self.source)}


def read_mujoco_sample(model, data, *, button_joint="pipette/pipette_button",
                       tip_site="pipette/tip_site"):
    """Read-only GT sample adapter; imports no MuJoCo until used by caller."""
    qadr = int(model.joint(button_joint).qposadr[0])
    sid = int(model.site(tip_site).id)
    return {"time_s": float(data.time), "button_qpos_m": float(data.qpos[qadr]),
            "tip_xyz_m": tuple(float(v) for v in data.site_xpos[sid])}


def load_stage(path: str | Path, *, source_geometry: SourceGeometry | None = None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    params = AspirationParameters(**raw["parameters"])
    if source_geometry is None and raw.get("source_geometry") is None:
        raise ValueError("Supply final scene SourceGeometry; config has no guessed geometry")
    geometry = source_geometry or SourceGeometry(**raw["source_geometry"])
    return AspirationStage(geometry, params)


def self_check():
    """Independent synthetic observer checks, NOT a robot rollout."""
    import copy
    source = SourceGeometry((.4, -.05), .012, .80, 1e-6, .82)
    stage = AspirationStage(source)
    t = 0.
    def hold(q, z, count=20, allowed=False):
        nonlocal t
        result = None
        for _ in range(count):
            t += .02
            result = stage.update(t, q, (.4, -.05, z), allow_dispense=allowed)
        return result
    hold(-.0052, .84)
    hold(-.0052, .815)
    for i in range(26):
        hold(-.005*(1-i/25), .815, count=1)
    hold(0., .815)
    ready = hold(0., .84)
    assert ready["enable_dispense"] and stage.aspirated_m3 > 29e-9
    spill = copy.deepcopy(stage)
    spill.update(t+.02, -.0052, (.4, -.05, .86), allow_dispense=False)
    assert spill.spilled_m3 == spill.aspirated_m3 and spill.delivered_m3 == 0.
    assert abs(spill.audit()["mass_balance_error_m3"]) < 1e-18
    hold(-.0052, .86, allowed=True)
    assert math.isclose(stage.delivered_m3, 30e-9, abs_tol=1e-15)
    hold(0., .86)
    hold(-.0052, .86, allowed=True)
    assert math.isclose(stage.delivered_m3, 30e-9, abs_tol=1e-15)
    assert stage.remaining_volume_m3 == 0.  # Pumping in air cannot refill.
    assert abs(stage.audit()["mass_balance_error_m3"]) < 1e-18
    # Empty gun cannot dispense, and wrong order cannot create liquid.
    wrong = AspirationStage(source)
    negative = wrong.update(.02, 0., (.4, -.05, .815), allow_dispense=True)
    assert negative["failure"] == "immersion_before_prepress"
    assert wrong.aspirated_m3 == wrong.delivered_m3 == 0.
    # Leaving the liquid during return stroke is a hard failure.
    partial = AspirationStage(source)
    for time, q, z in ((.02,-.006,.84),(.12,-.006,.84),(.14,-.006,.815),
                       (.26,-.006,.815),(.28,-.003,.815),(.30,0.,.84)):
        partial.update(time,q,(.4,-.05,z))
    assert partial.failure == "tip_left_liquid_during_release"
    assert 0. < partial.aspirated_m3 < 30e-9
    assert abs(partial.audit()["mass_balance_error_m3"]) < 1e-18
    return {"synthetic_checks_passed": True, "physical_rollout_tested": False,
            "positive_sequence": stage.audit(), "wrong_order": wrong.audit(),
            "uncommanded_discharge": spill.audit(),
            "no_refill_or_double_dispense_on_second_air_stroke": True,
            "withdraw_during_release": partial.audit()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.self_check:
        parser.error("Use --self-check, or import AspirationStage in the runner")
    report = self_check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
