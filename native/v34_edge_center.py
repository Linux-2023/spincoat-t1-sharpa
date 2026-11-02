"""Disk-edge arrival, then original-rate centering on the active native18 policy.

Install after native18's control restorations and Main's per-label dispatcher.
The existing RGB target (including its original 100 um alignment offset) owns
position; only the compiled rotating substrate's radius/identity is consulted.
No simulator state, model, gain, physical clock or acceptance gate is written.
The original raise, lowering, terminal/measured readiness handshake, thumb
exposure, retract and finish implementations retain ownership.

Planning helpers: disk_geometry(model, data), approach_geometry(...), and
plan_reference(...). EdgeCenterReference.advance() uses the same measured
transition_conditions() exposed to planning smoke; sample() alone cannot
advance past the edge. Runtime evidence is saved by the existing
smooth_right_precision_policy.json writer under reference_audit.edge_center.
"""
from __future__ import annotations

import math

import numpy as np

PATCHED_SYMBOLS = (
    "v34_topp_right._installed_precision_type.__init__",
    "v34_topp_right._installed_precision_type.target",
    "v34_topp_right.COVERAGE",
)
FAST_LABEL = "right/precision_fast_disk_edge"
SLOW_LABEL = "right/precision_slow_edge_to_center"
DISK_GEOM = "spincoater/substrate"
ROTOR_BODY = "spincoater/rotor_body"
ROTOR_JOINT = "spincoater/rotor"
TARGET_SITE = "spincoater/dispense_target"
# Unchanged native measured-position corridor, now also required at the edge.
XY_CORRIDOR_M = .0015
Z_CORRIDOR_M = .001
_INSTALLED = False


def disk_geometry(model, data=None):
    """Read shape/identity, never a privileged world-space target position."""
    import mujoco

    geom = int(model.geom(DISK_GEOM).id)
    body = int(model.body(ROTOR_BODY).id)
    joint = int(model.joint(ROTOR_JOINT).id)
    site = int(model.site(TARGET_SITE).id)
    if (int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_CYLINDER)
            or int(model.geom_bodyid[geom]) != body
            or int(model.site_bodyid[site]) != body
            or int(model.jnt_bodyid[joint]) != body
            or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_HINGE)):
        raise RuntimeError("Expected the rotating cylindrical substrate, not the catch bowl")
    radius, half_thickness = map(float, model.geom_size[geom, :2])
    if not (math.isfinite(radius + half_thickness) and 0. < half_thickness < radius):
        raise RuntimeError("Invalid rotating substrate dimensions")
    if (not np.allclose(model.geom_quat[geom, 1:3], 0., rtol=0., atol=1e-12)
            or not np.allclose(model.jnt_axis[joint], [0., 0., 1.], rtol=0., atol=1e-12)
            or not np.allclose(model.site_pos[site] - model.geom_pos[geom],
                               [0., 0., half_thickness], rtol=0., atol=1e-12)):
        raise RuntimeError("Visual dispense target must identify the disk top-center")
    normal = None
    if data is not None:
        normal = np.asarray(data.geom_xmat[geom]).reshape(3, 3)[:, 2]
        if not np.allclose(normal, [0., 0., 1.], rtol=0., atol=1e-10):
            raise RuntimeError("Native vertical-clearance policy requires a horizontal disk")
    return dict(geom_name=DISK_GEOM, geom_id=geom, geom_type="cylinder",
                rotor_body_name=ROTOR_BODY, rotor_body_id=body,
                rotor_joint_name=ROTOR_JOINT, rotor_joint_id=joint,
                target_site_name=TARGET_SITE, target_site_id=site,
                radius_m=radius, half_thickness_m=half_thickness,
                surface_normal_world=None if normal is None else normal.tolist(),
                dimensions_source="compiled model.geom_size; no bowl radius or guessed waypoint",
                center_source="original RGB target argument; no geom_xpos/site_xpos read")


def approach_geometry(incoming, visual_target, radius_m, transit_height_m=.024):
    """Split the original straight incoming line at the perceived disk perimeter.

    The center is the unmodified RGB target. The endpoint keeps the native
    +100 um X calibration. Intersect that same line with the disk circle, so
    the offset does not introduce a radial detour or move the boundary radius.
    An entry already inside the disk only centers slowly; it never backs out.
    """
    incoming = np.asarray(incoming, dtype=float)
    visual = np.asarray(visual_target, dtype=float)
    radius, height = float(radius_m), float(transit_height_m)
    if (incoming.shape != (3,) or visual.shape != (3,)
            or not np.isfinite(incoming).all() or not np.isfinite(visual).all()
            or not math.isfinite(radius + height) or radius <= .0001 or height <= 0.):
        raise ValueError("Finite incoming/RGB points, disk radius and clearance required")
    center = visual + np.array([.0001, 0., height])
    raised = np.r_[incoming[:2], center[2]]
    delta = center - raised
    distance = float(np.linalg.norm(delta))
    direction = delta / distance if distance else np.zeros(3)
    incoming_radius = float(np.linalg.norm(raised[:2] - visual[:2]))
    fast = 0.
    mode = "already_inside_disk_no_outward_detour"
    if incoming_radius > radius:
        outward = -direction[:2]
        offset = center[:2] - visual[:2]
        projection = float(np.dot(offset, outward))
        slow = -projection + math.sqrt(projection**2 + radius**2 - float(np.dot(offset, offset)))
        fast = distance - slow
        if fast <= 0. or slow <= 0.:
            raise RuntimeError("Incoming line did not enter the disk perimeter")
        mode = "nearest_incoming_line_perimeter_intersection"
    else:
        slow = distance
    edge = raised + fast * direction
    return dict(visual_target_world_m=visual.tolist(),
                incoming_world_m=incoming.tolist(), raised_world_m=raised.tolist(),
                edge_world_m=edge.tolist(), center_world_m=center.tolist(),
                final_world_m=(visual + [.0001, 0., .012]).tolist(),
                direction_world=direction.tolist(), radius_m=radius,
                incoming_radius_m=incoming_radius,
                edge_radius_m=float(np.linalg.norm(edge[:2] - visual[:2])),
                clearance_over_visual_target_m=height,
                native_alignment_offset_world_m=[.0001, 0., 0.],
                distance_m=distance, fast_distance_m=fast, slow_distance_m=slow,
                boundary_mode=mode)


class _Leg:
    """Exact native C2 velocity envelope; zero derivatives at both endpoints."""

    def __init__(self, distance, label, phase, speed, acceleration, factor, envelope):
        import v34_topp_core as core
        from v34_topp_right_paths import velocity_polynomial

        self.distance = float(distance)
        self.phase = phase
        native = envelope.VelocityPath()
        envelope._append_velocity_envelope(native, self.distance, 0., speed, acceleration, phase)
        self.parts = native.parts
        self.timing = None
        self.duration = 0.
        self.report = dict(label=label, distance_m=self.distance,
                           source_speed_m_s=float(speed), source_acceleration_m_s2=float(acceleration),
                           source_duration_s=native.duration, requested_factor=factor,
                           source_parts=native.parts, terminal_speed_m_s=0., terminal_acceleration_m_s2=0.,
                           physical_execution_passed=None)
        if native.parts:
            polynomial = velocity_polynomial(native.parts)
            if factor == 1.0:
                self.timing = core.retime(polynomial, duration=native.duration,
                                          label=label, factor=1.0)
            else:
                self.timing = core.retime(polynomial, duration=native.duration,
                                          label=label, factor=2.0)
            self.duration = float(self.timing.duration)
            self.report.update(duration_s=self.duration, executed_factor=float(self.timing.factor),
                               timing=self.timing.report)
        else:
            self.report.update(duration_s=0., executed_factor=None,
                               skipped_reason="exact zero-distance span")

    def sample(self, age):
        age = float(age)
        if not math.isfinite(age):
            raise ValueError("Nonfinite edge-center reference age")
        if age <= 0.:
            return 0., 0., 0.
        if age >= self.duration:
            return self.distance, 0., 0.
        return tuple(float(np.asarray(self.timing(age, derivative)).reshape(-1)[0])
                     for derivative in (0, 1, 2))


def transition_conditions(reference, age, actual_tip, *, raise_complete):
    """Actual transition predicates; no elapsed-only or cached-readiness bypass."""
    age = float(age)
    actual = np.asarray(actual_tip, dtype=float)
    if not math.isfinite(age) or actual.shape != (3,) or not np.isfinite(actual).all():
        raise ValueError("Finite reference age and measured tip required")
    edge_xy = float(np.linalg.norm(actual[:2] - reference.edge[:2]))
    edge_z = float(abs(actual[2] - reference.edge[2]))
    center_xy = float(np.linalg.norm(actual[:2] - reference.center[:2]))
    center_z = float(abs(actual[2] - reference.center[2]))
    progress = reference.sample(age)[0]
    nominal = reference.anchor + progress * reference.direction
    nominal_xy = float(np.linalg.norm(nominal[:2] - reference.center[:2]))
    edge_ready = bool(age >= reference.fast.duration and raise_complete
                      and edge_xy < XY_CORRIDOR_M and edge_z < Z_CORRIDOR_M)
    center_complete = reference.center_complete(age)
    return dict(fast_reference_complete=bool(age >= reference.fast.duration),
                native_raise_complete=bool(raise_complete),
                actual_edge_xy_error_m=edge_xy, actual_edge_z_error_m=edge_z,
                edge_ready=edge_ready, slow_center_reference_complete=center_complete,
                actual_center_xy_error_m=center_xy, actual_center_z_error_m=center_z,
                nominal_center_xy_error_m=nominal_xy,
                lower_ready=bool(center_complete and raise_complete
                                 and center_xy < XY_CORRIDOR_M and center_z < Z_CORRIDOR_M
                                 and nominal_xy < XY_CORRIDOR_M),
                xy_threshold_m=XY_CORRIDOR_M, z_threshold_m=Z_CORRIDOR_M,
                comparison="strictly less than; existing native corridor unchanged")


class EdgeCenterReference:
    """Separate reference clocks with a physical measured-arrival seam."""

    def __init__(self, geometry, disk, envelope, *, started_at=0.):
        self.geometry = geometry
        self.anchor = np.asarray(geometry["raised_world_m"], dtype=float)
        self.edge = np.asarray(geometry["edge_world_m"], dtype=float)
        self.center = np.asarray(geometry["center_world_m"], dtype=float)
        self.direction = np.asarray(geometry["direction_world"], dtype=float)
        self.distance = float(geometry["distance_m"])
        self.velocity = 0.
        self.started_at = float(started_at)
        profile = envelope.PROFILE
        self.fast = _Leg(geometry["fast_distance_m"], FAST_LABEL, "clear_transfer",
                         profile.clear_speed_m_s, profile.clear_acceleration_m_s2, 2.0, envelope)
        self.slow = _Leg(geometry["slow_distance_m"], SLOW_LABEL, "fine_transfer",
                         profile.fine_speed_m_s, profile.fine_acceleration_m_s2, 1.0, envelope)
        # Parts are source-time diagnostics only, with the separate clock maps
        # and actual measured hold explicitly described in report.
        self.parts = list(self.fast.parts) + [
            (start + self.fast.report["source_duration_s"], duration, distance + self.fast.distance,
             v0, v1, phase) for start, duration, distance, v0, v1, phase in self.slow.parts]
        self.duration = self.fast.duration + self.slow.duration
        self.center_started_age = None
        self.last_age = None
        self.phase = "fast_edge" if self.fast.distance else "edge_wait"
        self.conditions = None
        self.report = dict(
            geometry=dict(geometry), disk_geometry=dict(disk),
            fast_reference=self.fast.report, slow_reference=self.slow.report,
            minimum_reference_duration_s=self.duration, executed_schedule_duration_s=None,
            parts_time_basis="concatenated source seconds; measured edge hold excluded",
            edge_transition=None, center_reference_completed_at_s=None,
            geometry_rule="one straight line, actual disk radius, original RGB center and calibration",
            seam_rule="fast endpoint at rest AND measured edge corridor AND completed native raise",
            lower_rule="complete slow reference AND every original actual/nominal lower-entry predicate",
            dispense_rule="untouched native terminal-reference AND current measured-center handshake",
            source_envelopes_are_not_physical_certificates=True,
            physical_execution_passed=None)

    def center_complete(self, age):
        return bool(self.center_started_age is not None
                    and float(age) >= self.center_started_age + self.slow.duration)

    def sample(self, age):
        """Hold the edge until advance() observes the real arrival predicate."""
        if self.center_started_age is None:
            distance, velocity, acceleration = self.fast.sample(age)
            phase = "clear_transfer" if self.fast.distance else "fine_transfer"
        else:
            distance, velocity, acceleration = self.slow.sample(float(age) - self.center_started_age)
            distance += self.fast.distance
            phase = "fine_transfer"
        return distance, velocity, acceleration, phase

    def advance(self, age, actual_tip, *, raise_complete):
        age = float(age)
        if self.last_age is not None and age < self.last_age:
            raise RuntimeError("Edge-center physical reference age moved backwards")
        conditions = transition_conditions(self, age, actual_tip, raise_complete=raise_complete)
        self.last_age = age
        if self.center_started_age is None:
            if conditions["edge_ready"]:
                self.center_started_age = age
                self.duration = age + self.slow.duration
                self.report["executed_schedule_duration_s"] = self.duration
                self.report["edge_transition"] = dict(
                    controller_time_s=self.started_at + age, age_s=age,
                    measured_wait_after_fast_s=age - self.fast.duration,
                    actual_tip_world_m=np.asarray(actual_tip).tolist(), conditions=conditions)
            else:
                self.duration = max(self.duration, age + self.slow.duration)
        if self.center_started_age is None:
            self.phase = "fast_edge" if age < self.fast.duration else "edge_wait"
        else:
            self.phase = "center_hold" if self.center_complete(age) else "slow_center"
        self.conditions = transition_conditions(self, age, actual_tip, raise_complete=raise_complete)
        self.report["last_transition_conditions"] = self.conditions
        if self.center_complete(age) and self.report["center_reference_completed_at_s"] is None:
            self.report["center_reference_completed_at_s"] = self.started_at + age
        return self.sample(age)


def plan_reference(incoming, visual_target, disk, *, transit_height_m=.024, started_at=0.):
    """Build fresh references without loading a model or touching live state."""
    import v34_smoothrightenvelope as envelope

    geometry = approach_geometry(incoming, visual_target, disk["radius_m"], transit_height_m)
    return EdgeCenterReference(geometry, disk, envelope, started_at=started_at)


def _plan_policy(policy, transit, visual_target, distance):
    path = plan_reference(policy.direct_anchor, visual_target, policy._edge_center_disk,
                          transit_height_m=policy.height, started_at=policy.direct_at)
    if (not np.array_equal(path.anchor, policy.direct_anchor)
            or not np.array_equal(path.center, transit)
            or abs(path.distance - distance) > 1e-12):
        raise RuntimeError("Edge-center split changed the original straight-line endpoints")
    policy.reference_audit["edge_center"] = path.report
    return path, path.fast.duration, 0.


def _sample_policy(policy, time, actual_tip):
    path = policy.path
    result = path.advance(float(time) - policy.direct_at, actual_tip,
                          raise_complete=float(time) - policy.started >= policy._topp_raise.duration)
    if path.center_started_age is not None:
        policy.transfer_split_at = path.center_started_age
        policy.direct_duration = path.center_started_age
        policy.reference_audit["transfer_split_at_s"] = path.center_started_age
    policy.reference_audit["continuous_transfer_duration_s"] = path.duration
    return result


def _observe_policy(policy):
    path = policy.path
    if path is None:
        return
    if policy.trace:
        policy.trace[-1].update(edge_center_phase=path.phase,
                               edge_center_transition_conditions=path.conditions)
    if policy.stage == "vertical_lower" and "native_lower_entry" not in path.report:
        path.report["native_lower_entry"] = dict(controller_time_s=policy.lower_at,
                                                conditions=dict(path.conditions))


def install():
    """Patch actual installed callables in place, preserving aliases/closures."""
    global _INSTALLED
    if _INSTALLED:
        return PATCHED_SYMBOLS
    import v34_topp_core as core
    import v34_topp_right as right
    import v34_uncapped_right as uncapped
    import v34_uncapped_clock as clock
    from v34_topp_right_paths import closure

    if not (right._INSTALLED and uncapped._INSTALLED and core.retime is clock.retime):
        raise RuntimeError("Edge-center requires the installed native18 chain and final reference dispatcher")
    precision = right._installed_precision_type
    native_target = closure(precision.target).get("target")
    if not callable(native_target):
        raise RuntimeError("Missing active ToppPrecisionPolicy native-target closure")
    clock.patch(precision.__init__, [(
        'self._topp_lower = None',
        'self._topp_lower = None\n'
        '    self._edge_center_disk = _edge_center_disk_geometry(model, data)', 1,
    )], "right/edge_center_compiled_disk_identity", helpers={
        "_edge_center_disk_geometry": disk_geometry})
    clock.patch(native_target, [(
        'self.path, self.transfer_split_at, self.pass_speed = transfer_path(distance)',
        'self.path, self.transfer_split_at, self.pass_speed = _edge_center_plan_policy(self, transit, target, distance)', 1,
    ), (
        'distance, speed, accel, path_phase = self.path.sample(age)',
        'distance, speed, accel, path_phase = _edge_center_sample_policy(self, time, p)', 1,
    ), (
        'if (np.linalg.norm(p[:2] - transit[:2]) < .0015',
        'if (self.path.center_complete(age)\n'
        '                    and np.linalg.norm(p[:2] - transit[:2]) < .0015', 1,
    )], "right/edge_center_active_native_target", helpers={
        "_edge_center_plan_policy": _plan_policy,
        "_edge_center_sample_policy": _sample_policy})
    clock.patch(precision.target, [(
        'result = target(self, time, actual_tip, target_point)',
        'result = target(self, time, actual_tip, target_point)\n'
        '    _edge_center_observe_policy(self)', 1,
    )], "right/edge_center_existing_evidence", helpers={
        "_edge_center_observe_policy": _observe_policy})
    coverage = [row for row in right.COVERAGE if row["stage"] == "precision_clear_fine_transfer"]
    if len(coverage) != 1:
        raise RuntimeError("Expected one installed precision-transfer coverage record")
    coverage[0].update(
        decision="fast_disk_edge_then_original_rate_center",
        reason="Compiled rotating-substrate radius, original RGB center; eligible fast edge reference, measured arrival, explicit factor-one fine centering, then unchanged lowering/readiness/dispense.",
        reference_labels=[FAST_LABEL, SLOW_LABEL], physical_execution_passed=None)
    clock.record("installation", "right/edge_center_reference", symbols=list(PATCHED_SYMBOLS),
                 nested_patch="closure(v34_topp_right._installed_precision_type.target)['target']",
                 fast_label=FAST_LABEL, fast_requested_factor=2.0,
                 slow_label=SLOW_LABEL, slow_requested_factor=1.0,
                 original_visual_target_retained=True,
                 external_phases=["raise", "clear_transfer", "fine_transfer", "vertical_lower"],
                 original_lower_and_terminal_gates_retained=True,
                 native_physics_and_command_cadence_unchanged=True,
                 physical_execution_passed=None)
    _INSTALLED = True
    return PATCHED_SYMBOLS
