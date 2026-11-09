"""Private, measurement-driven laboratory judge; never a task policy.

New torque-apparatus/task identity, not the historical expert's 35 checks.
Original gravity, aspiration and film leaves remain unmodified. The coupling
releases every actual metered increment and feeds measured shaft velocity.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np

from .physics_support.contract import TASK_ID, CRITERIA, DT
from .physics_support.measurements import HeldLift, MixingMotion

BASE = CRITERIA["base"]
TUBE = CRITERIA["tube"]
PIPE = CRITERIA["pipette"]
DISPENSE = CRITERIA["dispense"]
ROTOR = CRITERIA["rotor"]
SAFETY = CRITERIA["safety"]

TUBE_FIELDS = ("tube_volume_m3", "receiver_volume_m3", "spilled_m3", "delivered_m3",
               "parcels", "last_time", "mixing_proxy_angular_travel_rad",
               "_previous_axis", "_previous_mouth", "surface_z", "flow_rate")
ASP_FIELDS = ("failure", "remaining_volume_m3", "source_remaining_m3", "aspirated_m3",
              "delivered_m3", "spilled_m3", "_time0", "_last_time", "_last_q",
              "_hold", "_press_q", "_max_release_fraction")
FLUID_FIELDS = ("h", "time_s", "substeps", "initial_volume_m3", "escaped_volume_m3",
                "rim_outflow_bins_m3", "doses", "rim_reservoir", "direct_miss_m3",
                "parcels", "phase_rad", "_phase_start", "_advance_start", "delivered_m3",
                "shape_rate_s", "stable_age_s", "first_uniform_time_s", "_last_shape")


def withdrawal_height(config, surface_z):
    """Public receiver rim and liquid clearance, not a planned lateral pose."""
    rim = config["tabletop_z_m"] + config["beaker_height_m"]
    return max(rim + PIPE["withdrawal_over_rim_m"],
               surface_z + CRITERIA["aspiration_parameters"]["withdrawal_clearance_m"])


def dispense_alignment(tip, velocity, rotation, target):
    point = np.asarray(target) + np.array([0., 0., DISPENSE["standoff_m"]])
    error = float(np.linalg.norm(np.asarray(tip) - point))
    speed = float(np.linalg.norm(velocity))
    tilt = float(np.arccos(np.clip(np.asarray(rotation)[2, 2], -1., 1.)))
    if not np.isfinite([error, speed, tilt]).all():
        raise FloatingPointError("Nonfinite dispense geometry")
    return error, speed, tilt, bool(error < DISPENSE["alignment_error_m"] and
        speed < DISPENSE["alignment_speed_m_s"] and tilt < DISPENSE["alignment_tilt_rad"])


def plain(value):
    """Finite private diagnostics, not an imputation of missing measurements."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise FloatingPointError("Nonfinite live physical diagnostic")
    return value


def event(operation, **payload):
    return {"phase": "post_step", "event_type": "environment_command",
            "payload": {"source": "live_task_evaluator", "operation": operation, **plain(payload)}}


class LiveTask:
    def __init__(self, sources, config, handles, model, data, *, stations):
        self.sources = sources
        self.config = copy.deepcopy(config)
        self.handles = copy.deepcopy(handles)
        self.station_xy = {key: np.asarray(value, dtype=float).copy() for key, value in stations.items()}
        if set(self.station_xy) != {"preparation", "grasp", "aspiration", "coating"} or any(
                xy.shape != (2,) or not np.isfinite(xy).all() for xy in self.station_xy.values()):
            raise ValueError("Four finite public-case station regions required")
        self.liquid = sources.gravity.GravityPourLedger(SimpleNamespace(**config), sources.gravity.PourPhysics())
        self.aspiration_parameters = sources.aspiration.AspirationParameters(**CRITERIA["aspiration_parameters"])
        if asdict(self.aspiration_parameters) != asdict(sources.aspiration.AspirationParameters()):
            raise ValueError("Benchmark must preserve the original aspiration constants")
        self.aspiration = None
        self.fluid = sources.fluid.SpincoatFluid(sources.fluid.FluidParameters())
        self.bind(model)
        self.tick = 0
        self.initial_tube = data.xpos[self.tube_body].copy()
        self.initial_mouth = data.site_xpos[self.mouth_site].copy()
        self.initial_pipe = data.xpos[self.pipe_body].copy()
        self.pipette_held = HeldLift(data.site_xpos[self.tip_site, 2])
        self.mixing = MixingMotion()
        self.tube_window = np.zeros((TUBE["support_samples"], 15), dtype=np.float64)
        self.window_cursor = self.window_count = 0
        self.tube_held = False
        self.tube_held_time = None
        self.tube_returned = self.preparation_complete = self.aspiration_complete = False
        self.aspiration_samples = 0
        self.aspiration_contact_min = None
        self.right_grip_1khz_ok = self.tube_grip_1khz_ok = True
        self.base_stable = {key: 0 for key in self.station_xy}
        self.base_fine_violations = 0
        self.preparation_base_ready_at_contact = None
        self.grasp_base_ready_at_contact = None
        self.preparation_wheel_violation_samples = 0
        self.geometry = {"crossdigit_entries": 0, "nonpad_tube_entries": 0,
                         "wall_innerface_normal_flip_entries": 0,
                         "wall_full_thickness_exceed_entries": 0,
                         "max_hand_tube_penetration_m": 0., "max_wall_penetration_m": 0.}
        self.alignment_since = None
        self.press_eligible = self.dispense_geometry_ok = False
        self.pending_discharge_m3 = 0.
        self.ballistic = None
        self.secondary_ballistics = []
        self.next_release_index = 0
        self.landed_m3 = self.maximum_impact_error_m = 0.
        self.all_releases_valid = True
        self.film_origin_time_s = self.film_sample_time_s = None
        self.film_velocity_integral_rad = 0.
        self.motor_at = None
        self.motor_diagnostics = None
        self.omega = 0.
        self.rotor_window = np.zeros(ROTOR["stop_samples"], dtype=np.float64)
        self.rotor_cursor = self.rotor_count = 0
        self.rated_speed_samples = self.maximum_rated_speed_samples = 0
        self.last_button = float(data.qpos[self.button_q])
        self.preparation_contact_violations = self.source_contact_violations = self.station_contact_violations = 0
        self.current_contact_violations = {"preparation": 0, "source": 0, "station": 0}
        self.phase = "preparation"
        self.terminal = self.functional = False
        self.terminal_reason = None
        self.metrics = {}
        self.measured_gates = {}
        self.failure_diagnostics = {}
        # Reset-only ledger priming, no physical step or liquid preload.
        self.liquid.advance_pose(data.time, data.site_xpos[self.mouth_site],
                                 data.xmat[self.tube_body].reshape(3, 3))
        self.liquid.records.clear()
        self.liquid.events.clear()

    def bind(self, model):
        self.tube_body = int(model.body(self.handles["tube_body"]).id)
        self.mouth_site = int(model.site(self.handles["tube_mouth_site"]).id)
        self.left_palm = int(model.body("sharpa_left/left_hand_C_MC").id)
        self.pipe_body = int(model.body("pipette/pipette").id)
        self.tip_site = int(model.site("pipette/tip_site").id)
        self.target_site = int(model.site("spincoater/dispense_target").id)
        self.button_q = int(model.joint("pipette/pipette_button").qposadr[0])
        rotor = model.joint("spincoater/rotor")
        self.rotor_q, self.rotor_d = int(rotor.qposadr[0]), int(rotor.dofadr[0])
        self.rotor_parent = int(model.body_parentid[int(model.jnt_bodyid[int(rotor.id)])])
        base = model.joint("chassis_free")
        self.base_q, self.base_d = int(base.qposadr[0]), int(base.dofadr[0])
        self.wheels = np.asarray([a for a in range(model.nu)
                                  if model.actuator(a).name.startswith("astribot_chassis_motor_")])
        self.body_names = tuple(model.body(b).name or "" for b in range(model.nbody))
        self.geom_bodies = model.geom_bodyid.copy()
        self.geom_names = tuple(model.geom(g).name or "" for g in range(model.ngeom))

    def fail(self, reason, events):
        if self.terminal:
            return
        self.terminal, self.functional = True, False
        self.terminal_reason = reason
        self.failure_diagnostics[reason] = {"tick": self.tick}
        events.append(event("task_failed", reason=reason, terminal=True))

    def invalidate(self, error):
        self.terminal, self.functional = True, False
        self.terminal_reason = "infrastructure_error"
        self.phase = "failed"
        self.failure_diagnostics["infrastructure_error"] = {
            "tick": self.tick, "type": type(error).__name__, "message": str(error)}

    def _contacts(self, model, data, events):
        left_counts, right_counts = np.zeros(5, dtype=np.int64), np.zeros(5, dtype=np.int64)
        left_forces, right_forces = np.zeros(5), np.zeros(5)
        self.current_contact_violations = {"preparation": 0, "source": 0, "station": 0}
        digits = ("thumb", "index", "middle", "ring", "pinky")
        wrench = np.zeros(6)
        for ci, contact in enumerate(data.contact):
            names = [self.body_names[int(self.geom_bodies[g])] for g in (contact.geom1, contact.geom2)]
            geoms = [self.geom_names[g] for g in (contact.geom1, contact.geom2)]
            if float(contact.dist) < 0.:
                for side in ("left", "right"):
                    prefix = "sharpa_"+side+"/"+side+"_"
                    fingers = [next((finger for finger in digits if name.startswith(prefix+finger+"_")), None)
                               for name in names]
                    if all(fingers) and fingers[0] != fingers[1]:
                        self.geometry["crossdigit_entries"] += 1
                        events.append(event("strict_geometry_violation", kind="crossdigit", geoms=geoms,
                                            penetration_m=-float(contact.dist)))
                if self.handles["tube_body"] in names and any(n.startswith("sharpa_left/") for n in names):
                    tube_index = names.index(self.handles["tube_body"])
                    hand_index = 1-tube_index
                    depth = -float(contact.dist)
                    self.geometry["max_hand_tube_penetration_m"] = max(self.geometry["max_hand_tube_penetration_m"], depth)
                    violations = []
                    if not geoms[hand_index].endswith("_elastomer"):
                        self.geometry["nonpad_tube_entries"] += 1
                        violations.append("nonpad_tube")
                    if geoms[tube_index].startswith(self.config["prefix"]+"reagent_tube_wall_"):
                        self.geometry["max_wall_penetration_m"] = max(self.geometry["max_wall_penetration_m"], depth)
                        if depth > self.config["tube_wall_thickness_m"]:
                            self.geometry["wall_full_thickness_exceed_entries"] += 1
                            violations.append("wall_full_thickness_exceed")
                        axis = data.xmat[self.tube_body].reshape(3, 3)[:, 2]
                        radial = contact.pos-data.xpos[self.tube_body]
                        radial = radial-axis*float(radial@axis)
                        radius = float(np.linalg.norm(radial))
                        if radius > 0.:
                            inward_normal = np.asarray(contact.frame[:3])*(1. if tube_index == 1 else -1.)
                            if float(inward_normal@(radial/radius)) > 0.:
                                self.geometry["wall_innerface_normal_flip_entries"] += 1
                                violations.append("wall_innerface_normal_flip")
                    for violation in violations:
                        events.append(event("strict_geometry_violation", kind=violation, geoms=geoms,
                                            penetration_m=depth))
            left = any(n.startswith("sharpa_left/") for n in names)
            right = any(n.startswith("sharpa_right/") for n in names)
            tube = self.handles["tube_body"] in names
            pipe = any(n.startswith("pipette/") for n in names)
            source = any(n == "aspiration_source" or n.endswith("/shared_griffin_beaker") for n in names)
            obstacle = next((n for n in names if n.startswith(("lab_bench", "stand/", "spincoater/"))), None)
            moving = any(n and not n.startswith(("lab_bench", "stand/", "spincoater/", "pipette/")) for n in names)
            prep_obstacle = any((n.startswith(self.config["prefix"]) and n != self.handles["tube_body"])
                                or n.startswith(("aspiration_source", "stand/", "spincoater/", "lab_bench", "pipette/")) for n in names)
            if not ((left and (tube or prep_obstacle)) or (right and pipe) or
                    (source and (left or right or pipe)) or (obstacle and (moving or pipe))):
                continue
            mujoco.mj_contactForce(model, data, ci, wrench)
            force = float(np.linalg.norm(wrench[:3]))
            for fi, digit in enumerate(digits):
                if tube and any(n.startswith("sharpa_left/left_"+digit+"_") for n in names):
                    left_counts[fi] += 1
                    left_forces[fi] += force
                if pipe and any(n.startswith("sharpa_right/") and digit in n for n in names):
                    right_counts[fi] += 1
                    right_forces[fi] += force
            if left and prep_obstacle and force > SAFETY["preparation_contact_N"] and not self.preparation_complete:
                self.preparation_contact_violations += 1
                self.current_contact_violations["preparation"] += 1
                events.append(event("preparation_contact", bodies=names, force_N=force,
                                    penetration_m=max(0., -float(contact.dist))))
            if source and (left or right or pipe) and force > SAFETY["source_contact_N"]:
                self.source_contact_violations += 1
                self.current_contact_violations["source"] += 1
                events.append(event("source_collision", bodies=names, force_N=force))
            if obstacle and (moving or (not obstacle.startswith("stand/") and pipe)) and force > SAFETY["station_contact_N"]:
                self.station_contact_violations += 1
                self.current_contact_violations["station"] += 1
                events.append(event("station_collision", bodies=names, force_N=force,
                                    penetration_m=max(0., -float(contact.dist))))
        return left_counts, left_forces, right_counts, right_forces

    def _tube_gate(self, data, counts, forces, events):
        rotation = data.xmat[self.left_palm].reshape(3, 3)
        relative = rotation.T@(data.xpos[self.tube_body]-data.xpos[self.left_palm])
        relative_rotation = rotation.T@data.xmat[self.tube_body].reshape(3, 3)
        active = (counts > 0) & (forces > TUBE["contact_force_N"])
        support = bool(active[0] and np.sum(active[1:]) >= TUBE["minimum_nonthumb_support"])
        row = self.tube_window[self.window_cursor]
        row[:3], row[3:12], row[12:] = relative, relative_rotation.ravel(), [support, np.all(active[1:]), data.time]
        samples = TUBE["support_samples"]
        self.window_cursor = (self.window_cursor+1) % samples
        self.window_count = min(samples, self.window_count+1)
        lift = float(data.xpos[self.tube_body, 2]-self.initial_tube[2])
        required = max(TUBE["minimum_lift_m"],
                       float(self.handles["rack_top_world_z_m"])+TUBE["rack_clearance_m"]-self.initial_tube[2])
        movement = angle = None
        supported_window = self.window_count == samples and bool(np.all(self.tube_window[:, 12]))
        if not self.tube_held and lift >= required and supported_window:
            oldest = self.tube_window[self.window_cursor]
            movement = float(np.linalg.norm(self.tube_window[:, :3]-oldest[:3], axis=1).max())
            rotations = self.tube_window[:, 3:12].reshape(-1, 3, 3)
            traces = np.einsum("ij,nij->n", oldest[3:12].reshape(3, 3), rotations)
            angle = float(np.arccos(np.clip((traces-1.)/2., -1., 1.)).max())
            if movement < TUBE["relative_translation_m"] and angle < TUBE["relative_rotation_rad"]:
                self.tube_held = True
                self.tube_held_time = float(data.time)
                events.append(event("tube_held_gate", lift_m=lift, translation_m=movement,
                                    rotation_rad=angle, native_window_samples=samples))
        delivered = self.liquid.delivered_m3 >= TUBE["captured_fraction"]*max(
            self.config["target_pour_volume_m3"], self.config["tube_initial_volume_m3"])
        return_error = float(np.linalg.norm(data.xpos[self.tube_body]-self.initial_tube))
        self.tube_returned = bool(self.tube_held and delivered and return_error < TUBE["return_error_m"])
        if self.tube_held and not self.tube_returned and not support:
            self.tube_grip_1khz_ok = False
        preparation_ready = (self.preparation_base_ready_at_contact and
                             not self.preparation_wheel_violation_samples and
                             not self.preparation_contact_violations and self.mixing.acquired)
        if (self.tube_returned and not self.preparation_complete and preparation_ready and
                np.all(data.ctrl[self.wheels] == 0.)):
            self.preparation_complete = True
            events.append(event("preparation_completed", return_error_m=return_error,
                                delivered_m3=self.liquid.delivered_m3, material_homogeneity_verified=False))
        return {"tube_lift_m": lift, "required_lift_m": required, "support_now": support,
                "supported_halfsecond": supported_window, "relative_translation_m": movement,
                "relative_rotation_rad": angle, "tube_return_error_m": return_error}

    def _base_gate(self, data):
        speed = float(np.linalg.norm(data.qvel[self.base_d:self.base_d+2]))
        yaw = abs(float(data.qvel[self.base_d+5]))
        stationary = (speed < BASE["planar_speed_m_s"] and yaw < BASE["yaw_rate_rad_s"] and
                      bool(np.all(data.ctrl[self.wheels] == 0.)))
        errors = {}
        for key, xy in self.station_xy.items():
            errors[key] = float(np.linalg.norm(data.qpos[self.base_q:self.base_q+2]-xy))
            self.base_stable[key] = self.base_stable[key]+1 if stationary and errors[key] <= BASE["radius_m"] else 0
        return speed, yaw, errors
    def _source_geometry(self):
        volume = float(self.liquid.receiver_volume_m3)
        aspirated = 0. if self.aspiration is None else self.aspiration.aspirated_m3
        budget = volume+aspirated
        c = self.config
        return self.sources.aspiration.SourceGeometry(tuple(c["receiver_xy_m"]),
            c["receiver_inner_radius_m"], c["receiver_bottom_z_m"], budget,
            c["receiver_bottom_z_m"]+budget/(np.pi*c["receiver_inner_radius_m"]**2))

    def _aspiration_step(self, data, tip, button, right_counts, events):
        module, c, parameters = self.sources.aspiration, self.config, self.aspiration_parameters
        surface = c["receiver_bottom_z_m"]+self.liquid.receiver_volume_m3/(np.pi*c["receiver_inner_radius_m"]**2)
        in_source = bool(np.linalg.norm(tip[:2]-c["receiver_xy_m"]) <= c["receiver_inner_radius_m"] and
                         tip[2] <= surface-parameters.min_immersion_m)
        if (self.aspiration is None and self.liquid.receiver_volume_m3 >= parameters.capacity_m3 and
                (button < -parameters.released_qpos_tolerance_m or in_source)):
            self.aspiration = module.AspirationStage(self._source_geometry(), parameters)
            events.append(event("aspiration_activated", source_volume_m3=self.liquid.receiver_volume_m3,
                                reason="measured_button_compression_or_tip_immersion"))
        if self.aspiration is None:
            return None
        stage = self.aspiration
        # Shared receiver is authoritative; cumulative intake reconstructs the
        # unchanged observer's source budget even while fresh pour parcels land.
        stage.source = self._source_geometry()
        stage.source_remaining_m3 = self.liquid.receiver_volume_m3
        before = stage.source_remaining_m3
        result = stage.update(float(data.time), button, tip, allow_dispense=(
            self.press_eligible and self.dispense_geometry_ok and
            self.base_stable["coating"] >= BASE["stable_samples"]))
        self.liquid.receiver_volume_m3 = stage.source_remaining_m3
        if before != stage.source_remaining_m3:
            events.append(event("shared_receiver_transfer", before_m3=before,
                                after_m3=stage.source_remaining_m3,
                                pipette_increment_m3=before-stage.source_remaining_m3))
        for value in result["events"]:
            events.append(event("aspiration_observer", measurement=value))
        stage.events.clear()
        self.pending_discharge_m3 += float(result["delivered_increment_m3"])
        if not self.aspiration_complete and self.base_stable["aspiration"] < BASE["stable_samples"]:
            self.base_fine_violations += 1
        if not np.all(right_counts[1:] > 0):
            self.right_grip_1khz_ok = False
        if self.tick % PIPE["contact_sample_stride"] == 1 and not self.aspiration_complete:
            self.aspiration_contact_min = (right_counts.copy() if self.aspiration_contact_min is None
                                           else np.minimum(self.aspiration_contact_min, right_counts))
            self.aspiration_samples += 1
        required_z = withdrawal_height(c, stage.liquid_surface_z_m())
        if (not self.aspiration_complete and stage.state == module.Stage.READY and tip[2] >= required_z and
                self.pipette_held.acquired and self.aspiration_contact_min is not None and
                np.all(self.aspiration_contact_min[1:] > 0) and not self.source_contact_violations):
            self.aspiration_complete = True
            events.append(event("aspiration_completed", tip_z_m=tip[2], required_tip_z_m=required_z,
                                contact_sample_period_s=PIPE["contact_sample_stride"]*DT,
                                volume_m3=stage.remaining_volume_m3))
        if result["failure"]:
            self.fail(result["failure"], events)
        if result["spilled_increment_m3"] > 0.:
            self.fail("transport_or_premature_discharge_dose_loss", events)
        return {"stage": stage.state.value, "remaining_volume_m3": stage.remaining_volume_m3,
                "failure": stage.failure, "withdrawal_margin_m": float(tip[2]-required_z),
                "mass_balance_error_m3": result["mass_balance_error_m3"]}

    def _release(self, data, tip, velocity, target, button, events):
        # Each real stroke-volume increment is airborne immediately. No dose
        # waits in a hidden bucket until an expert-like full-stroke checkpoint.
        if self.pending_discharge_m3 > 0.:
            height = float(tip[2]-target[2])
            gravity = float(self.fluid.p.gravity_m_s2)
            if height <= 0.:
                raise FloatingPointError("Accepted dispense has no future substrate-plane crossing")
            flight = float((velocity[2]+np.sqrt(velocity[2]**2+2.*gravity*height))/gravity)
            impact = tip+velocity*flight
            impact[2] -= .5*gravity*flight**2
            error = float(np.linalg.norm(impact[:2]-target[:2]))
            valid = bool(self.press_eligible and self.dispense_geometry_ok and
                         not self.station_contact_violations and error <= DISPENSE["maximum_impact_error_m"])
            drop = {"release_position": tip.copy(), "release_velocity": velocity.copy(),
                    "release_time_s": float(data.time), "volume_m3": self.pending_discharge_m3,
                    "landing_time_s": float(data.time)+flight, "impact_position": impact,
                    "target_position": target.copy(), "impact_error_m": error,
                    "gravity_m_s2": gravity, "landed": False, "valid": valid,
                    "release_index": self.next_release_index}
            self.next_release_index += 1
            self.pending_discharge_m3 = 0.
            self.maximum_impact_error_m = max(self.maximum_impact_error_m, error)
            self.all_releases_valid = self.all_releases_valid and valid
            if self.ballistic is None:
                self.ballistic = drop
            else:
                self.secondary_ballistics.append(drop)
            events.append(event("ballistic_release", **drop))
            if not valid:
                self.fail("invalid_or_premature_dispense", events)
        if self.ballistic is None:
            return
        due = [drop for drop in (self.ballistic, *self.secondary_ballistics)
               if not drop["landed"] and float(data.time) >= drop["landing_time_s"]]
        for drop in sorted(due, key=lambda value: (value["landing_time_s"], value["release_index"])):
            drop["landed"] = True
            self.landed_m3 += drop["volume_m3"]
            if self.film_origin_time_s is None:
                self.film_origin_time_s = drop["landing_time_s"]
                self.film_sample_time_s = self.film_origin_time_s
            yaw = self._station_yaw(data)+float(data.qpos[self.rotor_q])
            x, y = drop["impact_position"][:2]-drop["target_position"][:2]
            xy = (np.cos(yaw)*x+np.sin(yaw)*y, -np.sin(yaw)*x+np.cos(yaw)*y)
            self.fluid.deposit(xy, volume_m3=drop["volume_m3"])
            self.fluid.doses[-1]["start"] = drop["landing_time_s"]-self.film_origin_time_s
            events.append(event("film_deposit", release_index=drop["release_index"],
                                volume_m3=drop["volume_m3"], substrate_xy_m=xy,
                                landing_time_s=drop["landing_time_s"]))
        self.secondary_ballistics = [drop for drop in self.secondary_ballistics if not drop["landed"]]
        dose = self.aspiration_parameters.capacity_m3*self.aspiration_parameters.min_fill_fraction
        if (self.motor_at is None and self.landed_m3 >= dose and not self.secondary_ballistics and
                self.ballistic["landed"] and tip[2]-target[2] > DISPENSE["motor_start_clearance_m"]):
            self.motor_at = float(data.time)
            events.append(event("motor_start", reason="landed_dose_and_measured_tip_clearance"))
        if self.film_origin_time_s is not None:
            now = float(data.time)
            self.film_velocity_integral_rad += self.omega*(now-self.film_sample_time_s)
            self.film_sample_time_s = now
            if self.tick % 20 == 1:
                duration = now-self.film_origin_time_s-self.fluid.time_s
                if duration > 0.:
                    average = self.film_velocity_integral_rad/duration
                    self.fluid.advance(duration, average, rotor_angle_start=(
                        self._station_yaw(data)+float(data.qpos[self.rotor_q])-self.film_velocity_integral_rad))
                    self.film_velocity_integral_rad = 0.
                    self.fluid.doses = [dose for dose in self.fluid.doses if dose["fraction"] < 1.]
                    events.append(event("film_advance", duration_s=duration, measured_omega_rad_s=average,
                                        fluid_time_s=self.fluid.time_s))
    def _station_yaw(self, data):
        rotation = data.xmat[self.rotor_parent].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _completion(self, now, clearance, events):
        dose = self.aspiration_parameters.capacity_m3*self.aspiration_parameters.min_fill_fraction
        self.measured_gates = {
            "preparation": bool(self.preparation_complete and self.tube_returned),
            "mixing_motion": self.mixing.acquired,
            "pipette_held_lift": self.pipette_held.acquired,
            "aspiration": self.aspiration_complete,
            "grasp_base": bool(self.grasp_base_ready_at_contact),
            "aspiration_base": not self.base_fine_violations,
            "dose_landed": self.landed_m3 >= dose,
            "release_geometry": bool(self.ballistic is not None and self.all_releases_valid and
                                     self.maximum_impact_error_m <= DISPENSE["maximum_impact_error_m"]),
            "no_flights": bool(self.ballistic is not None and self.ballistic["landed"] and
                               not self.secondary_ballistics and self.pending_discharge_m3 == 0.),
            "measured_spin": self.maximum_rated_speed_samples >= ROTOR["rated_speed_samples"],
            "measured_stop": bool(self.rotor_count == ROTOR["stop_samples"] and
                                  self.rotor_window.max() < ROTOR["stop_maximum_rpm"]),
            "finish_clearance": clearance >= DISPENSE["finish_clearance_m"],
            "apparatus_contact": not (self.preparation_contact_violations or
                                      self.source_contact_violations or self.station_contact_violations)}
        if self.terminal or self.motor_at is None or now-self.motor_at < ROTOR["finish_earliest_age_s"]:
            return
        self.functional = all(self.measured_gates.values())
        self.terminal = True
        self.terminal_reason = "completed" if self.functional else "completion_gates_failed"
        events.append(event("task_finished", functional=self.functional, measured_gates=self.measured_gates,
                            rotor_last_halfsecond_max_rpm=float(self.rotor_window.max()),
                            tip_clearance_m=clearance))

    def _contact_failures(self, events):
        if self.current_contact_violations["preparation"]:
            self.fail("unexpected_preparation_environment_contact", events)
        if self.current_contact_violations["station"]:
            self.fail("robot_station_contact_violation", events)
        if self.current_contact_violations["source"]:
            self.fail("aspiration_source_collision", events)

    def advance(self, model, data):
        if self.terminal:
            raise RuntimeError("Cannot advance a terminal evaluator")
        events = []
        self.tick += 1
        now = float(data.time)
        self.omega = float(data.qvel[self.rotor_d])
        counts, forces, right_counts, right_forces = self._contacts(model, data, events)
        speed, yaw, base_errors = self._base_gate(data)
        if self.preparation_base_ready_at_contact is None and counts[0] > 0 and forces[0] > TUBE["contact_force_N"]:
            self.preparation_base_ready_at_contact = self.base_stable["preparation"] >= BASE["stable_samples"]
            events.append(event("preparation_base_at_first_grip", ready=self.preparation_base_ready_at_contact))
        if self.grasp_base_ready_at_contact is None and np.any((right_counts[1:] > 0) & (right_forces[1:] > PIPE["contact_force_N"])):
            self.grasp_base_ready_at_contact = self.base_stable["grasp"] >= BASE["stable_samples"]
            events.append(event("pipette_base_at_first_grip", ready=self.grasp_base_ready_at_contact))
        if (self.preparation_base_ready_at_contact is not None and not self.preparation_complete and
                np.any(data.ctrl[self.wheels] != 0.)):
            self.preparation_wheel_violation_samples += 1
        mouth = data.site_xpos[self.mouth_site].copy()
        rotation = data.xmat[self.tube_body].reshape(3, 3)
        liquid_result = self.liquid.advance_pose(now, mouth, rotation,
            allow_mix=self.tube_held and not self.tube_returned)
        self.liquid.records.clear()
        self.liquid.events.clear()
        tube_metrics = self._tube_gate(data, counts, forces, events)
        mixing_eligible = bool(self.tube_held and not self.tube_returned and tube_metrics["support_now"] and
            self.liquid.delivered_m3 == 0. and self.liquid.spilled_m3 == 0. and
            self.liquid.airborne_m3 == 0. and self.liquid.flow_rate == 0.)
        if self.mixing.observe(now, mouth, rotation, mixing_eligible):
            events.append(event("measured_mixing_acquired", **self.mixing.status()))
        tip = data.site_xpos[self.tip_site].copy()
        target = data.site_xpos[self.target_site].copy()
        button = float(data.qpos[self.button_q])
        if self.pipette_held.observe(now, tip[2], right_counts, right_forces):
            events.append(event("pipette_held_lift", **self.pipette_held.status()))
        velocity6 = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, self.tip_site, velocity6, 0)
        velocity = velocity6[3:].copy()
        alignment_error, tip_speed, tilt, alignment = dispense_alignment(
            tip, velocity, data.xmat[self.pipe_body].reshape(3, 3), target)
        if not alignment:
            self.alignment_since = None
        elif self.alignment_since is None:
            self.alignment_since = now
        qualified = self.alignment_since is not None and now-self.alignment_since >= DISPENSE["alignment_hold_s"]
        if self.base_stable["coating"] < BASE["stable_samples"]:
            self.press_eligible = False
        if (self.aspiration_complete and qualified and self.base_stable["coating"] >= BASE["stable_samples"] and
                button < self.last_button and not self.press_eligible):
            self.press_eligible = True
            events.append(event("measured_dispense_eligible", alignment_error_m=alignment_error,
                                tip_speed_m_s=tip_speed, tilt_rad=tilt))
        clearance = float(tip[2]-target[2])
        self.dispense_geometry_ok = bool(DISPENSE["minimum_height_m"] <= clearance <= DISPENSE["maximum_height_m"] and
            tip_speed < DISPENSE["maximum_release_speed_m_s"] and not self.station_contact_violations)
        aspiration_result = self._aspiration_step(data, tip, button, right_counts, events)
        self._release(data, tip, velocity, target, button, events)
        self.last_button = button
        rpm = abs(self.omega)*30./np.pi
        self.rotor_window[self.rotor_cursor] = rpm
        self.rotor_cursor = (self.rotor_cursor+1) % ROTOR["stop_samples"]
        self.rotor_count = min(ROTOR["stop_samples"], self.rotor_count+1)
        rated = self.motor_at is not None and ROTOR["minimum_measured_rpm"] <= rpm <= ROTOR["maximum_measured_rpm"]
        self.rated_speed_samples = self.rated_speed_samples+1 if rated else 0
        self.maximum_rated_speed_samples = max(self.maximum_rated_speed_samples, self.rated_speed_samples)
        if float(data.xpos[self.pipe_body, 2]) < PIPE["minimum_body_z_m"]:
            self.fail("pipette_dropped", events)
        self._contact_failures(events)
        balance = self.balance_error()
        if not np.isfinite(balance) or abs(balance) > SAFETY["volume_balance_tolerance_m3"]:
            raise FloatingPointError("Live liquid conservation failed")
        self._completion(now, clearance, events)
        previous_phase = self.phase
        self.phase = self._phase(now)
        if self.phase != previous_phase:
            events.append(event("measured_task_phase", previous=previous_phase, phase=self.phase))
        self.metrics = plain({**tube_metrics, "liquid": liquid_result,
            "tube_remaining_m3": self.liquid.tube_volume_m3, "receiver_volume_m3": self.liquid.receiver_volume_m3,
            "tube_airborne_m3": self.liquid.airborne_m3, "preparation_spilled_m3": self.liquid.spilled_m3,
            "mixing_motion": self.mixing.status(), "pipette_held_lift": self.pipette_held.status(),
            "material_homogeneity_verified": False, "aspiration": aspiration_result,
            "base_speed_m_s": speed, "base_yaw_rate_rad_s": yaw, "base_workspace_errors_m": base_errors,
            "base_stable_samples": self.base_stable, "base_fine_violation_samples": self.base_fine_violations,
            "preparation_base_ready_at_contact": self.preparation_base_ready_at_contact,
            "grasp_base_ready_at_contact": self.grasp_base_ready_at_contact,
            "preparation_wheel_violation_samples": self.preparation_wheel_violation_samples,
            "task_failures": self.failure_diagnostics, "current_contact_violations": self.current_contact_violations,
            "strict_geometry": self.geometry, "measured_gates": self.measured_gates,
            "alignment_error_m": alignment_error, "alignment_qualified": qualified,
            "tip_speed_m_s": tip_speed, "pipette_tilt_rad": tilt,
            "left_contact_counts": counts, "left_contact_forces_N": forces,
            "right_contact_counts": right_counts, "right_contact_forces_N": right_forces,
            "rotor_rpm": rpm, "motor": self.motor_diagnostics,
            "maximum_rated_speed_samples": self.maximum_rated_speed_samples,
            "tip_clearance_m": clearance, "film": self.fluid.metrics(),
            "global_balance_error_m3": balance, "pending_discharge_m3": self.pending_discharge_m3,
            "landed_dose_m3": self.landed_m3, "maximum_impact_error_m": self.maximum_impact_error_m,
            "secondary_flights": len(self.secondary_ballistics),
            "strict_right_grip_1khz": self.right_grip_1khz_ok,
            "strict_left_support_1khz": self.tube_grip_1khz_ok})
        return events, copy.deepcopy(self.metrics)

    def _phase(self, now):
        if self.terminal:
            return "complete" if self.functional else "failed"
        if self.motor_at is not None:
            age = now-self.motor_at
            return "finish_hold" if age >= 6.2 else "spin_down" if age >= 4.2 else "coating"
        if self.ballistic is not None:
            return "tip_withdrawal" if self.ballistic["landed"] else "droplet_flight"
        if self.press_eligible:
            return "dispensing"
        if self.aspiration_complete:
            return "alignment" if self.base_stable["coating"] >= BASE["stable_samples"] else "transport"
        if self.aspiration is not None:
            return "aspiration:"+self.aspiration.state.value
        if self.preparation_complete:
            return "pipette_grasp"
        if self.liquid.delivered_m3 >= TUBE["captured_fraction"]*self.config["target_pour_volume_m3"]:
            return "tube_return"
        if self.liquid.flow_rate > 0. or self.liquid.airborne_m3 > 0.:
            return "pouring"
        return "tube_held" if self.tube_held else "preparation"

    def balance_error(self):
        stage = self.aspiration
        pipe = 0. if stage is None else stage.remaining_volume_m3
        discharged_spill = 0. if stage is None else stage.spilled_m3
        flight = 0. if self.ballistic is None or self.ballistic["landed"] else self.ballistic["volume_m3"]
        flight += sum(drop["volume_m3"] for drop in self.secondary_ballistics)
        # Fluid initial_volume includes film + finite-delivery queue + all
        # parcels/catcher/spill. Cumulative delivered is not counted twice.
        total = (self.liquid.tube_volume_m3+self.liquid.receiver_volume_m3+self.liquid.airborne_m3+
                 self.liquid.spilled_m3+pipe+discharged_spill+self.pending_discharge_m3+
                 flight+self.fluid.initial_volume_m3)
        return float(self.config["tube_initial_volume_m3"]+self.config["receiver_initial_volume_m3"]-total)

    def status(self):
        configured = bool(self.functional and self.right_grip_1khz_ok and self.tube_grip_1khz_ok and
            not any(self.geometry[key] for key in ("crossdigit_entries", "nonpad_tube_entries",
                "wall_innerface_normal_flip_entries", "wall_full_thickness_exceed_entries")))
        return {"task_id": TASK_ID, "terminal": self.terminal, "reason": self.terminal_reason, "task_phase": self.phase,
                "outcome": {"functional": self.functional, "configured_contact_checks": configured,
                            "strict_physics": None, "hardware_certified": None,
                            "reason": self.terminal_reason or "incomplete"},
                "measured_gates": copy.deepcopy(self.measured_gates), "metrics": copy.deepcopy(self.metrics)}

    def snapshot(self):
        """Read-only complete evaluator state for renderer mutation guards.

        This is NOT a runtime restore API; every benchmark begins at a fresh
        reset. Modules and immutable numerical meshes are represented by their
        bound configuration; mutable liquid/controller-independent state is copied.
        """
        tube = {key: copy.deepcopy(getattr(self.liquid, key)) for key in TUBE_FIELDS}
        for name in ("_hydro_at", "_record_at"):
            value = float(getattr(self.liquid, name))
            tube[name] = {"initialized": bool(np.isfinite(value)), "value_s": value if np.isfinite(value) else None}
        aspiration = None
        if self.aspiration is not None:
            aspiration = {key: copy.deepcopy(getattr(self.aspiration, key)) for key in ASP_FIELDS}
            aspiration.update(source=asdict(self.aspiration.source), parameters=asdict(self.aspiration.p),
                              state=self.aspiration.state.value, events=copy.deepcopy(self.aspiration.events))
        excluded = {"sources", "liquid", "aspiration", "fluid", "pipette_held", "mixing", "aspiration_parameters"}
        task = {key: copy.deepcopy(value) for key, value in self.__dict__.items() if key not in excluded}
        return {"physics": asdict(self.liquid.physics), "liquid": tube, "aspiration": aspiration,
                "fluid_parameters": asdict(self.fluid.p),
                "fluid": {key: copy.deepcopy(getattr(self.fluid, key)) for key in FLUID_FIELDS},
                "pipette_held": copy.deepcopy(self.pipette_held.__dict__),
                "mixing": copy.deepcopy(self.mixing.__dict__), "task": task}

    def render_state(self, data):
        liquid = self.liquid.state(float(data.time), data.xpos[self.tube_body],
                                   data.xmat[self.tube_body].reshape(3, 3))
        return {"liquid": {"state": liquid, "config": copy.deepcopy(self.config),
                           "physics": asdict(self.liquid.physics), "parcels": copy.deepcopy(self.liquid.parcels)},
                "fluid": {**{name: copy.deepcopy(getattr(self.fluid, name)) for name in FLUID_FIELDS},
                          "parameters": asdict(self.fluid.p)},
                "ballistic": copy.deepcopy(self.ballistic),
                "ballistic_parcels": copy.deepcopy(self.secondary_ballistics)}
