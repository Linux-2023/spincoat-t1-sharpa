"""Uncapped RIGHT reference composition after frozen R08/TOPP/overlap.

Only reference clocks and artificial command slew/rate policies change. Native
1 ms steps, command cadence, actual-scene factories, IK trust regions, position
bounds, contact/geometry gates and physical qualification clocks remain owned
by the original workflow. Source-envelope numbers are not executed limits or
certificates. Main installs the shared deterministic clock before this adapter.
"""
from __future__ import annotations

from functools import wraps

import numpy as np

from v34_uncapped_clock import FACTOR, patch, record, replace_once, retime
from v34_topp_right_paths import closure, velocity_polynomial


PATCHED_SYMBOLS = (
    "v34_overlap_control.full_entry_envelope",
    "v34_overlap_control.OverlapSession.release",
    "v34_overlap_control.OverlapSession.before_step",
    "v34_overlap_control.OverlapSession.after_step",
    "v34_overlap_control.OverlapSession.save",
    "v34_dualarmoverlap.OverlapAcquisition.plan_at_actual_dock",
    "v34_dualarmoverlap.OverlapAcquisition.report",
    "v34_dualarmoverlap.adapt_child_source",
    "v34_topp_right._Acquisition.task_clock",
    "v34_speedrightpickup.retime_prepared",
    "v34_speedrightpickup.retime_engine_source",
    "t1_workflow_ik.install_validated_ik",
    "v34_topp_right._cup_timing",
    "v34_topp_right._route_timing",
    "v34_topp_right._descent_progress",
    "v34_smoothrightwithdrawal._command_rate",
    "t1_right_interior_workstation_v6.adapt_vertical_aspiration_source",
    "v34_topp_right.retimed_transfer",
    "v34_topp_right.retimed_retract",
    "v34_smoothrightenvelope.lower_bounds",
    "v34_smoothrightenvelope.checked_ik",
    "v34_topp_right._installed_precision_type",
    "v34_topp_right.COVERAGE",
    "t1_right_direct_coating_v6.adapt_direct_coating_source",
)
_INSTALLED = False


def _replace(source, old, new, label):
    return replace_once(source, old, new, "right/" + label)


def _command_tick(time, policy):
    """Keep physical command-cadence accounting without Cartesian slew clipping."""
    time = float(time)
    previous = getattr(policy, "_lookahead_previous_time", None)
    policy._lookahead_previous_time = time
    policy._lookahead_update_dt = None if previous is None else time - previous
    if policy.trace and abs(policy.trace[-1]["controller_time_s"] - time) < 1e-8:
        policy.trace[-1]["command_update_dt_s"] = policy._lookahead_update_dt
        policy.trace[-1]["cartesian_slew_clipped"] = False


def _uncapped_nullspace_source(source):
    # These are per-call redundant-command caps, NOT the numerical +/- .04rad
    # iteration trust region or the physical joint-margin projection.
    source = _replace(source, "budget = 0.003 / max(iterations, 1)", "",
                      "nullspace_call_budget")
    return _replace(source,
                    "null_step *= min(1.0, budget / max(np.max(np.abs(null_step)), 1e-12))",
                    "", "nullspace_call_clipping")


def _grasp_source(source):
    source = _replace(source,
        "Rotation.from_rotvec(rotation_error*min(1,.012/max(angle,1e-12)))",
        "Rotation.from_rotvec(rotation_error)", "grasp_orientation_slew")
    return _replace(source,
        "                shift*=min(1,.0025/max(np.linalg.norm(shift),1e-12))\n",
        "", "grasp_cartesian_slew")


def _dispense_source(source):
    """Runs at the final native factory seam, AFTER enabled-exposure adaptation."""
    candidates = ("u=smooth((task_t-21)/1.8);thumb_goal=",
                  "u=smooth((task_t-21)/3);thumb_goal=")
    found = [text for text in candidates if source.count(text) == 1]
    if len(found) != 1:
        raise RuntimeError("Expected exactly one native enabled-exposure thumb interpolation")
    old = found[0]
    return _replace(source, old, old.replace("(task_t-21)", "2.0*(task_t-21)"),
                    "enabled_exposure_thumb_reference")


class _VelocityReference:
    """One factor-two clock across all native velocity phases, including joins.

    A single clock avoids per-block tick rounding/terminal holds at the nonzero
    clear/fine and near/clear-retract seam. Parts remain explicitly source-time
    diagnostics; the returned derivatives and seam times are executed values.
    """
    def __init__(self, native, label):
        self.native = native
        self.parts = native.parts
        self.distance = native.distance
        self.velocity = FACTOR * native.velocity
        self.source_duration = float(native.duration)
        self.timing = None
        self.duration = 0.
        self.report = dict(
            reference_factor=FACTOR,
            parts_time_basis="unexecuted native source seconds",
            source_envelope_certificate_valid_for_execution=False,
            geometry="exact degree-six integral of original native velocity phases",
            nonzero_internal_velocity_preserved=True,
        )
        if native.parts:
            self.timing = retime(velocity_polynomial(native.parts),
                                 duration=self.source_duration, label=label)
            self.duration = self.timing.duration
            self.report["timing"] = self.timing.report
        else:
            record("zero_length_reference", label, duration_s=0.,
                   source_envelope_certificate_valid_for_execution=False)

    def new_time_for_source(self, source):
        return 0. if self.timing is None else self.timing.new_time_for_source(float(source))

    def sample(self, age):
        if self.timing is None:
            return self.native.sample(age)
        age = float(age)
        if not np.isfinite(age):
            raise ValueError("Nonfinite RIGHT velocity-reference age")
        if age <= 0.:
            p, v, a, phase = self.native.sample(0.)
            return p, FACTOR * v, FACTOR**2 * a, phase
        if age >= self.duration:
            return self.distance, self.velocity, 0., self.parts[-1][-1]
        source_age = self.timing.source_time(age)
        phase = self.parts[-1][-1]
        for start, duration, distance, v0, v1, name in self.parts:
            if source_age < start + duration:
                phase = name
                break
        return (float(np.asarray(self.timing(age)).reshape(-1)[0]),
                float(np.asarray(self.timing(age, 1)).reshape(-1)[0]),
                float(np.asarray(self.timing(age, 2)).reshape(-1)[0]), phase)


def _retimed_transfer(previous, profile, distance):
    native, split, through = previous(distance)
    path = _VelocityReference(native, "right/precision_clear_fine_uncapped")
    return path, path.new_time_for_source(split), FACTOR * through


def _retimed_retract(native, split, profile):
    path = _VelocityReference(native, "right/retract_uncapped")
    return path, path.new_time_for_source(split)


def _install_overlap(overlap, dual, right):
    patch(overlap.full_entry_envelope, [
        ("clock=clock, vlim=p['arm_rate_limit_rad_s'],\n                         protected=((0., duration),)",
         "clock=clock, factor=1.0", 1),
        ("Unexecuted original full-entry envelope; limits retained for smaller scheduled motions.",
         "Unexecuted source-clock construction only; factor=1, no executed speed certificate.", 1),
    ], "right/unexecuted_overlap_envelope")
    patch(overlap.OverlapSession.release, [
        ("round((self.preshape_start+self.preshape_duration)/self.dt)",
         "20*int(np.ceil((self.preshape_start+self.preshape_duration)/(_uncapped_factor*self.dt*20)))", 1),
        ("label='overlap/right_arm_preposition', vlim=pickup.PROFILE['arm_rate_limit_rad_s'],\n        alim=envelope.report['acceleration_limits'], protected=((0.,PREPOSITION_SECONDS),)",
         "label='overlap/right_arm_preposition'", 1),
    ], "right/overlap_release_clock", helpers={"_uncapped_factor": FACTOR})
    patch(overlap.OverlapSession.before_step, [
        ("u = smooth((age-self.preshape_start)/self.preshape_duration)",
         "u = smooth((_uncapped_factor*age-self.preshape_start)/self.preshape_duration)", 1),
        ("opening = smooth((age-self.preshape_start)/1.1)",
         "opening = smooth((_uncapped_factor*age-self.preshape_start)/1.1)", 1),
        ("if age < self.home_s else", "if _uncapped_factor*age < self.home_s else", 1),
    ], "right/overlap_hand_clock", helpers={"_uncapped_factor": FACTOR})
    patch(overlap.OverlapSession.after_step, [
        ("round(self.home_s/self.dt)", "round(self.home_s/(_uncapped_factor*self.dt))", 1),
        ("after the original initial hold; actual joint states retained in trace",
         "after the compressed source hold; actual joint states retained in trace", 1),
    ], "right/overlap_physical_observation", helpers={"_uncapped_factor": FACTOR})
    patch(overlap.OverlapSession.save, [
        ("original_initial_hold_s=.7,original_preshape_s=3.2,original_mcp_ramp_s=1.1,",
         "source_initial_hold_s=.7,source_preshape_s=3.2,source_mcp_ramp_s=1.1,\n"
         "            reference_factor=_uncapped_factor,executed_initial_hold_s=.7/_uncapped_factor,\n"
         "            executed_preshape_s=3.2/_uncapped_factor,executed_mcp_ramp_s=1.1/_uncapped_factor,\n"
         "            hand_terminal_command_tick=getattr(self,'hand_terminal_tick',None),\n"
         "            hand_terminal_command_s=(self.hand_terminal_tick*self.dt if hasattr(self,'hand_terminal_tick') else None),\n"
         "            superseded_speed_certificates_valid=False,", 1),
    ], "right/overlap_evidence", helpers={"_uncapped_factor": FACTOR})
    patch(dual.OverlapAcquisition.plan_at_actual_dock, [
        ("vlim=self.profile['arm_rate_limit_rad_s'],alim=full_reference.report['acceleration_limits'],\n        protected=((float(remaining_clock.x[-2]),source_duration),),minimum_duration=native_tail",
         "factor=_uncapped_factor", 1),
        ("delta = self.prefix-full_duration\n    self.command_timing.update({key:value+delta for key,value in self.original_timing.items()})",
         "self.command_timing.update({key:self.prefix+(value-full_duration)/_uncapped_factor\n"
         "                                    for key,value in self.original_timing.items()})", 1),
        ("label='right/acquisition_coupled_contact_clock',vlim=derivative_enclosure(task),\n        protected=((0.,duration),)",
         "label='right/acquisition_coupled_contact_clock',factor=_uncapped_factor", 1),
        ("hand_program_physically_completed=True,original_full_entry_limits=full_reference.report['acceleration_limits']",
         "hand_program_physically_completed=True,reference_factor=_uncapped_factor,\n"
         "            source_envelope_executed=False,superseded_speed_certificates_valid=False,\n"
         "            collar_escape_hold_span_compressed=True,passive_held_gate_s=.5", 1),
    ], "right/remaining_and_coupled_acquisition", helpers={"_uncapped_factor": FACTOR})
    patch(dual.OverlapAcquisition.report, [
        ("Original hand program executed once during LEFT tail; completed opening held until native seating fade",
         "Factor-two hand reference executed once during LEFT tail; completed opening held until the shared factor-two seating fade", 1),
    ], "right/acquisition_evidence")
    patch(right._Acquisition.task_clock, [
        ("return .8 * elapsed", "return .8 * _uncapped_factor * elapsed", 1),
        ("+ .8 * (elapsed - self.task_timing.duration)",
         "+ .8 * _uncapped_factor * (elapsed - self.task_timing.duration)", 1),
    ], "right/acquisition_clock_extension", helpers={"_uncapped_factor": FACTOR})
    previous_child = dual.adapt_child_source

    @wraps(previous_child)
    def child_source(source):
        return _grasp_source(previous_child(source))

    dual.adapt_child_source = child_source


def _install_acquisition(pickup):
    previous_prepared = pickup.retime_prepared
    previous_engine = pickup.retime_engine_source

    @wraps(previous_prepared)
    def prepared(runner, path, callback, audit, clock, directory, model):
        runner, path, callback, audit, clock = previous_prepared(
            runner, path, callback, audit, clock, directory, model)
        # Sanitize the emitted source report only after the frozen factories
        # finish. Recompiling native_prepared here would AST-normalize its
        # nested callbacks before downstream exact-source adapters see them.
        import json
        from pathlib import Path
        source_report_path = Path(directory) / "speed_right_pickup_timing.json"
        source_report = json.loads(source_report_path.read_text())
        for old, new in (("profile", "source_profile"),
                         ("command_timing", "source_command_timing"),
                         ("original_support_windows_s", "source_support_windows_s"),
                         ("nominal_schedule_saved_s", "source_nominal_schedule_saved_s"),
                         ("fine_readiness_starts_at_actual_seating_s", "source_seat_start_s")):
            source_report[new] = source_report.pop(old)
        source_report.update(
            timing_scope="Unexecuted source template; final factor-two timing is in motion_profile.json and grasp_contact_audit.json",
            superseded_speed_certificates_valid=False)
        source_report_path.write_text(json.dumps(source_report, indent=2))
        # These are the actual final callables returned by overlap, not disk
        # copies of runner.main. Mutate code in place so every captured alias
        # retains the actual mutable module globals and original closure cells.
        coherent = closure(callback)["coherent"]
        sampler = closure(runner.continuous_prefix_sample)["speed_sampler"]
        patch(sampler, [
            ("if bound > PROFILE['arm_rate_limit_rad_s']:", "if not np.isfinite(bound):", 1),
            ("Retimed prefix conservative command-rate bound exceeded: ",
             "Nonfinite source prefix rate observation: ", 1),
        ], "right/acquisition_prefix_rate_guard")
        patch(coherent, [
            ("if rate > PROFILE['arm_rate_limit_rad_s'] + 1e-8:",
             "if not np.isfinite(rate):", 1),
            ("Retimed acquisition arm-rate bound exceeded: ", "Nonfinite acquisition arm command rate: ", 1),
            ("if rate > PROFILE['hand_rate_limit_rad_s'] + 1e-8:",
             "if not np.isfinite(rate):", 1),
            ("Retimed acquisition hand-rate bound exceeded: ", "Nonfinite acquisition hand command rate: ", 1),
        ], "right/acquisition_rate_guards")

        @wraps(audit)
        def uncapped_audit():
            result = audit()
            source_report = result["aggressive_right_acquisition"]
            source_report["source_profile"] = source_report.pop("profile")
            source_report["software_rate_limits_enforced"] = False
            source_report["source_rate_bound_is_execution_certificate"] = False
            result["uncapped_right_acquisition"] = dict(
                reference_factor=FACTOR, software_arm_hand_rate_limits_enforced=False,
                source_rate_statistics_are_not_certificates=True,
                original_position_finite_contact_gates_retained=True,
                coupled_hold_includes_collar_escape=True,
                passive_support_qualification_s=.5)
            return result

        uncapped_audit.dock = audit.dock
        return runner, path, callback, uncapped_audit, clock

    @wraps(previous_engine)
    def engine_source(source):
        source = previous_engine(source)
        source_duration = pickup.timing()["stage_end_s"] - pickup.prefix_duration()
        old = pickup.ENGINE_DURATION_ANCHOR.replace("'11.625'", repr(str(source_duration)))
        new = old.replace(repr(str(source_duration)), "str(grasp._topp_acquisition.task_timing.duration)")
        source = _replace(source, old, new, "acquisition_child_reference_endpoint")
        source = _replace(source, "(data.time-asp_started)/.5",
                          "2.0*(data.time-asp_started)/.5", "index_preload_reference")
        anchor = "    module=types.ModuleType('front_workflow_'+kind)"
        source = _replace(source, anchor,
            "    if kind!='grasp':\n"
            "        from v34_uncapped_right import _dispense_source\n"
            "        source=_dispense_source(source)\n" + anchor,
            "final_dispense_factory")
        return source

    pickup.retime_prepared = prepared
    pickup.retime_engine_source = engine_source


def _install_aspiration(interior, withdrawal, right):
    patch(withdrawal._command_rate, [
        ("if not np.isfinite(rate) or rate > limit + 1e-9:", "if not np.isfinite(rate):", 1),
        ("Aspiration command-rate bound failed: ", "Nonfinite aspiration command rate: ", 1),
    ], "right/aspiration_command_rates")
    patch(right._cup_timing, [
        ("curve = core.quintic(np.r_[start, 0.]",
         "if not dry:\n        raise RuntimeError('Uncapped cup reference requires the original dry-entry clearance')\n"
         "    curve = core.quintic(np.r_[start, 0.]", 1),
        ("Original straight-tip/axial-yaw polynomial Bernstein enclosure; not a joint/torque bound",
         "Source polynomial only; executed reference is uncapped factor two, not velocity/torque certified", 1),
    ], "right/dry_cup_reference")
    patch(right._route_timing, [
        ("Native moving-target polynomial bound, NOT actual chassis speed; original SpeedTransport physical caps remain separate",
         "Source moving-target geometry only; uncapped factor-two reference is not actual chassis-speed certification", 1),
    ], "right/loaded_route_reference")
    patch(right._descent_progress, [
        ("Original guarded reference Bernstein enclosure; native1.5mm Cartesian lookahead and2rad/s guarded IK remain independent",
         "Original descent geometry composed once at factor two; command slew/rate caps removed, measured liquid/geometry gates retained", 1),
    ], "right/descent_reference_evidence")
    previous = interior.adapt_vertical_aspiration_source
    withdrawal_type = closure(previous)["withdrawal_type"]
    patch(withdrawal_type._ensure_topp_withdrawal, [
        ("Original withdrawal Bernstein enclosure; native guarded IK remains authoritative",
         "Persistent factor-two withdrawal geometry; no inherited velocity/acceleration certificate", 1),
    ], "right/persistent_withdrawal_evidence")

    @wraps(previous)
    def aspiration_source(source, config=None):
        source, environment, report = previous(source, config)
        hover = float(withdrawal.PROFILE["dry_thumb_hover_s"])
        original = ("u=smooth(t/" + repr(hover) + ");above=top+axis*.014;thumb_goal=(1-u)*initial_thumb+u*above if t<"
                    + repr(hover) + " else top+axis*(.014-.029*smooth((t-" + repr(hover) + ")/2.5))")
        replacement = ("u=smooth(2.0*t/" + repr(hover) + ");above=top+axis*.014;thumb_goal=(1-u)*initial_thumb+u*above if 2.0*t<"
                       + repr(hover) + " else top+axis*(.014-.029*smooth((2.0*t-" + repr(hover) + ")/2.5))")
        source = _replace(source, original, replacement, "aspiration_hover_press")
        source = _replace(source, "axis*.024*smooth(age/3)",
                          "axis*.024*smooth(2.0*age/3)", "immersed_thumb_release")
        source = _replace(source,
            "Rotation.from_rotvec(rv*min(1.,.010/max(angle,1e-12)))",
            "Rotation.from_rotvec(rv)", "aspiration_orientation_slew")
        source = _replace(source,
            "shift=goal-current;shift*=min(1.,.0015/max(np.linalg.norm(shift),1e-12))",
            "shift=goal-current", "aspiration_cartesian_slew")
        source = _replace(source, "float(np.clip(.08*stroke_error,-.00008,.00008))",
                          "float(.08*stroke_error)", "thumb_feedback_slew")
        source = _replace(source,
            "one original3s geometric reference with a persistent TOPP clock across WITHDRAW and READY",
            "one source3s geometric reference with a persistent uncapped factor-two clock across WITHDRAW and READY",
            "withdrawal_clock_description")
        report = dict(report)
        report["ik_method"] = "Original joint margins, numerical iteration trust regions and posture gains; artificial per-call nullspace budget removed"
        report["speed_aspiration"] = dict(report["speed_aspiration"],
            near_liquid_clock_rate=FACTOR, withdrawal_motion_unchanged=False,
            withdrawal_and_ready_hold_unchanged=False,
            source_envelope_certificate_valid_for_execution=False)
        motion_report = dict(report["smooth_right_withdrawal"])
        motion_report.pop("unchanged_press_stroke_s")
        motion_report.pop("unchanged_release_stroke_s")
        motion_report.update(dry_thumb_hover_s=hover / FACTOR,
            source_hover_s=hover, source_press_stroke_s=2.5, source_release_stroke_s=3.,
            press_reference_duration_s=2.5 / FACTOR, release_reference_duration_s=3. / FACTOR,
            thumb_feedback_gain=.08, thumb_feedback_slew_clip_removed=True,
            reference_factor=FACTOR, source_envelope_certificate_valid_for_execution=False)
        report["smooth_right_withdrawal"] = motion_report
        report["topp_right"] = dict(report["topp_right"],
            dry_descent_and_withdrawal_reference_only=False,
            superseded_speed_certificates_valid=False, executed_reference_factor=FACTOR)
        return source, environment, report

    interior.adapt_vertical_aspiration_source = aspiration_source


def _install_precision(envelope, coating, right):
    # The existing native velocity-path constructor supplies source shape and
    # distance invariants, not an executed envelope. One deterministic clock
    # composes that source exactly once and scales through-speed metadata.
    right.retimed_transfer = _retimed_transfer
    right.retimed_retract = _retimed_retract
    patch(envelope.lower_bounds, [
        ("if bounds['speed_m_s'] > PROFILE.fine_speed_m_s or bounds['acceleration_m_s2'] > PROFILE.fine_acceleration_m_s2:",
         "if not all(math.isfinite(value) for value in bounds.values()):", 1),
        ("Lower reference exceeds unchanged fine envelope: ", "Nonfinite lower source reference: ", 1),
    ], "right/lower_source_bound_rejection")
    patch(envelope.checked_ik, [
        ("# Retain R05's2rad/s rejection, now on all replacement trajectory updates.",
         "# Retain finite command rates and the original physical command cadence.", 1),
        ("if not np.isfinite(rate) or rate > PROFILE.command_rate_limit_rad_s + 1e-9:",
         "if not np.isfinite(rate):", 1),
        ("SmoothRightEnvelope command-rate bound failed: ", "Nonfinite coating command rate: ", 1),
    ], "right/precision_command_rate")
    precision_type = right._installed_precision_type
    native_target = closure(precision_type.target)["target"]
    patch(native_target, [
        ("self.reference_audit['lower_unchanged_terminal_tail_s'] = PROFILE.lower_native_tail_s",
         "self.reference_audit['lower_source_terminal_tail_s'] = PROFILE.lower_native_tail_s\n"
         "                self.reference_audit['lower_executed_terminal_tail_s'] = PROFILE.lower_native_tail_s / _uncapped_factor\n"
         "                self.reference_audit['superseded_speed_certificates_valid'] = False", 1),
    ], "right/precision_clock_evidence", helpers={"_uncapped_factor": FACTOR})
    patch(precision_type._lower_timing, [
        ('self.reference_audit["lower_all_time_reference_bounds"] = dict(\n'
         '            speed_m_s=self._topp_lower.report["all_time_velocity_bound"][0],\n'
         '            acceleration_m_s2=self._topp_lower.report["all_time_acceleration_bound"][0])',
         '_source_bounds = _uncapped_lower_source_bounds(delta)\n'
         '        self.reference_audit["lower_all_time_reference_bounds"] = dict(\n'
         '            speed_m_s=_uncapped_factor * _source_bounds["speed_m_s"],\n'
         '            acceleration_m_s2=_uncapped_factor**2 * _source_bounds["acceleration_m_s2"],\n'
         '            used_for_acceptance=False)', 1),
    ], "right/lower_observational_bounds", helpers={
        "_uncapped_factor": FACTOR, "_uncapped_lower_source_bounds": envelope.lower_bounds})
    patch(precision_type._lower_clock, [
        ("return age, 1., 0.", "return _uncapped_factor * age, _uncapped_factor, 0.", 1),
        ("return profile.lower_duration_s + age - timing.duration, 1., 0.",
         "return profile.lower_duration_s + _uncapped_factor * (age - timing.duration), _uncapped_factor, 0.", 1),
    ], "right/lower_clock_extension", helpers={"_uncapped_factor": FACTOR})
    previous = coating.adapt_direct_coating_source

    @wraps(previous)
    def coating_source(source, config=None):
        source = previous(source, config)
        source = _replace(source,
            "Rotation.from_rotvec(rotation_error*min(1,.012/max(angle,1e-12)))",
            "Rotation.from_rotvec(rotation_error)", "coating_orientation_slew")
        source = _replace(source,
            "shift*=min(1,coarse_lookahead(t,right_direct_coating_policy,current_tip,target,phase)/max(np.linalg.norm(shift),1e-12))",
            "_uncapped_command_tick(t,right_direct_coating_policy)", "coating_cartesian_slew")
        source = _replace(source,
            "profile=asdict(SMOOTH_RIGHT_PROFILE),time_basis='stage_logical_seconds',",
            "source_profile=asdict(SMOOTH_RIGHT_PROFILE),reference_factor=2.0,time_basis='stage_logical_seconds',",
            "precision_source_profile_scope")
        source = _replace(source, "original_thresholds_retained=True,",
            "original_physical_thresholds_retained=True,software_command_rate_caps_removed=True,superseded_speed_certificates_valid=False,",
            "precision_certificate_scope")
        return "from v34_uncapped_right import _command_tick as _uncapped_command_tick\n" + source

    coating.adapt_direct_coating_source = coating_source


def install():
    """Install once on the real frozen chain, before native factory execution."""
    global _INSTALLED
    if _INSTALLED:
        return
    import v34_topp_core as core
    import v34_topp_right as right
    import v34_dualarmoverlap as dual
    import v34_overlap_control as overlap
    import v34_speedrightpickup as pickup
    import v34_smoothrightwithdrawal as withdrawal
    import v34_smoothrightenvelope as envelope
    import t1_right_interior_workstation_v6 as interior
    import t1_right_direct_coating_v6 as coating
    import t1_workflow_ik as workflow_ik
    if not (right._INSTALLED and dual._INSTALLED and core.retime is retime):
        raise RuntimeError("Uncapped RIGHT requires frozen R08/TOPP/overlap and Main's uncapped clock first")
    if overlap.SESSION is not None:
        raise RuntimeError("Uncapped RIGHT must install before the physical overlap session is bound")
    _install_overlap(overlap, dual, right)
    _install_acquisition(pickup)
    patch(workflow_ik.install_validated_ik, [
        ("module=ast.parse(body);ast.fix_missing_locations(module)",
         "body=_uncapped_nullspace_source(body)\n    module=ast.parse(body);ast.fix_missing_locations(module)", 1),
    ], "right/validated_solver_factory", helpers={"_uncapped_nullspace_source": _uncapped_nullspace_source})
    _install_aspiration(interior, withdrawal, right)
    _install_precision(envelope, coating, right)
    physical_stages = {
        "standing_torso_and_base_docks", "passive_support_and_held_lift",
        "transport_liquid_and_view", "flight_landing_spin_finish",
    }
    # Mutate existing dictionaries too: the frozen report may have captured
    # individual entries rather than the module's list object.
    for coverage in right.COVERAGE:
        physical = coverage["stage"] in physical_stages
        coverage.update(
            decision="native_physical_qualification" if physical else "uncapped_factor_two_reference",
            reason=("Physical qualifications/process clocks unchanged; Main owns torso/base policies."
                    if physical else
                    "Final motion reference factor2, physical observation and acceptance clocks unchanged; source speed envelopes superseded. See right_inventory.json for exact stage coverage."),
            superseded_speed_certificates_valid=False)
    record("installation", "right/uncapped_adapters", symbols=list(PATCHED_SYMBOLS),
           reference_factor=FACTOR, physics_dt_s=.001,
           superseded_speed_certificates_valid=False,
           source_factory_bindings_preserved=True,
           physical_qualification_clocks_unchanged=True)
    _INSTALLED = True
