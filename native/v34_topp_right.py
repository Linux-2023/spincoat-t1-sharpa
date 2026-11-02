"""Whole-workflow RIGHT TOPP reference composition for the qualified R08 chain.

Install AFTER all ten original R08 plugins. Only exact command references are
retimed. Native1ms stepping,50Hz guarded IK,1kHz acquisition fingers, actual
measured-tool transforms, every observer/gate and actuator cap retain ownership.
In particular coating t stays physical elapsed+9; task classification, enabled
press exposure, flight, landing, rotor/film clocks and finish are never warped.

The actual-dock prefix is replanned from this episode's legitimate measured
state. Its fine1.5s tail and the independent3.2s preshape remain native. The
shared seat/close/slot/lift/verticalize clock is wholly protected: these are
coupled contact/loaded-feedback strokes, not free joint-space trajectories.
Explicit per-stage precision/retract Cartesian limits remain distinct from
joint-rate checks and do not claim joint-acceleration/torque certification.
"""
from __future__ import annotations

from functools import wraps
import json
from pathlib import Path

import numpy as np

import v34_topp_core as core
from v34_topp_right_paths import (
    acquisition_task_path, closure, copy_callable, derivative_enclosure, exact,
    inverse_quintic, prefix_path, retimed_retract, retimed_transfer, shifted_clock, smooth,
)

PATCHED_SYMBOLS = (
    "v34_speedrightpickup.retime_prepared",
    "v34_speedrightpickup.retime_engine_source",
    "t1_right_station_yaw_launch_v7.adapt_neutral_source_v7_yaw",
    "t1_right_standing_motion_v6.reanchor_prefix_at_actual_dock_v6",
    "t1_right_standing_motion_v6.advance_direct_cup_v6",
    "t1_right_passive_settle_v5.passive_wait_before_gate",
    "t1_right_interior_workstation_v6.adapt_vertical_aspiration_source",
    "t1_coarse_stop._advance",
    "t1_right_direct_coating_v6.adapt_direct_coating_source",
)

COVERAGE = [
    dict(stage="standing_torso_and_base_docks", decision="native_feedback",
         reason="Shared torso belongs to LEFT; all base feedback predicates/coverage belong to parent. No base class/gain patch."),
    dict(stage="acquisition_actual_dock_prefix", decision="exact_joint_spline_topp",
         reason="Actual-dock two-knot quintic spline, existing R08 entry clock, protected home and native final1.5s; independent preshape finishes by new seating."),
    dict(stage="preshape_and_mcp_opening", decision="protected_physical_reference",
         reason="Original .7+3.2s preshape and .7+1.1s MCP envelope use actual acquisition elapsed time."),
    dict(stage="seat_ingress_close_slot_lift_verticalize", decision="protected_exact_scalar_path",
         reason="One native C2 logical clock for measured seat alignment,25mm ingress, closure,12mm collar escape, lift and verticalization; coupled contact strokes and1.25s pre/postlift windows are not compressed."),
    dict(stage="passive_support_and_held_lift", decision="native_physical_qualification",
         reason="Actual .5s stepping, >=499 samples with all four fingers/force>.01N and measured lift>=60mm; prior missing support retained."),
    dict(stage="dry_cup_position_and_yaw", decision="exact_cartesian_axial_path_topp",
         reason="Native live-tool FK anchor, original straight tip path and axial yaw; dry guard, validated IK, .5s settlement and all original contact/error predicates retained."),
    dict(stage="aspiration_hover_press_preload", decision="protected_physical_reference_and_feedback",
         reason="Native .9s hover,2.5s press and .015rad/.5s preload;80um/call thumb servo and actual stroke feedback unchanged."),
    dict(stage="aspiration_guarded_descent", decision="exact_quintic_path_topp",
         reason="Native R08 analytic descent clock; final reference below both15mm guards remains at unit source-clock rate. Original lateral<1.5mm gate retained."),
    dict(stage="aspiration_immersed_release", decision="protected_physical_reference_and_qualification",
         reason="Native3s release and actual .08s/.10s/.15s liquid qualifications, margins, recompression rejection and30s timeout retained."),
    dict(stage="withdrawal_withdraw_to_ready", decision="persistent_exact_quintic_path_topp",
         reason="One latched anchor/clock across observer reset; wet exit protected, dry tail eligible. Complete terminal+current READY+error<2mm continuous.5s gate inherited unchanged."),
    dict(stage="loaded_mobile_reference", decision="exact_quintic_reference_topp",
         reason="Only SpeedTransport scoped native moving target changes. Physical .30/.035m/s caps,.4m/s2 slew,10mm irreversible latch,.4s brake,.5s stopped and18s ceiling remain native."),
    dict(stage="transport_liquid_and_view", decision="native_physical_qualification",
         reason="Full-rate spill observer and >=95% remaining dose; zero-motion view record is not a removable wait; fresh RGB target gates entry."),
    dict(stage="precision_raise", decision="exact_quintic_reference_topp",
         reason="Original command-FK anchor/target and measured+nominal1mm corridor; keep outgoing vertical derivatives during lateral overlap."),
    dict(stage="precision_clear_fine_transfer", decision="exact_native_velocity_integral_topp",
         reason="Degree-six polynomial from real phase coefficients; clear .18/.5, fine .05/.12; nonzero through velocity and original corner collider screens retained."),
    dict(stage="precision_lower", decision="exact_quintic_reference_topp",
         reason="Native2s geometry and R08 .8+.75 clock; exact final.75s unit-rate tail, fine .05/.12 limits, measured XY<1.5mm/Z<1mm entry predicates."),
    dict(stage="readiness_and_dispense", decision="native_physical_qualification_and_exposure",
         reason="Terminal reference AND actual<.3mm,<.0075m/s,<.005rad for40ms with stopped base;1.8s enabled press exposure accrues .001 per physical tick; actual button and >=95% dose gates unchanged."),
    dict(stage="retract", decision="exact_native_velocity_integral_topp",
         reason="Same final command anchor and100mm geometry; first30mm .066/.24 then .12/.5, nonzero internal velocity retained."),
    dict(stage="flight_landing_spin_finish", decision="native_physical_clocks",
         reason="Real ballistic/landing/film clocks; landed AND actual tip>40mm motor gate;4.2s spin including.5s ramp then2s slowdown+.5s observation, final<1rpm/80mm clearance. Native rotor qvel prescription remains an explicit limitation."),
]
_INSTALLED = False


class _PhysicalCoverage:
    """Record actually reached native phases, never substitute for their gates."""

    def __init__(self, label):
        self.label = label
        self.phase = None
        self.started = None
        self.last = None

    def touch(self, time, phase):
        time = float(time)
        phase = str(getattr(phase, "value", phase))
        if self.last is not None and time < self.last - 1e-10:
            raise RuntimeError("RIGHT physical coverage clock moved backwards")
        self.last = time
        if phase != self.phase:
            self.finish(time)
            self.phase, self.started = phase, time
            core.note_fixed(self.label + "/" + phase + "/entered",
                            reason="Reached native state; all original contact/feedback/elapsed-time predicates remain authoritative. Entry is not qualification success.",
                            kind="native_predicate_entry")

    def finish(self, time):
        if self.phase is not None:
            core.note_fixed(self.label + "/" + self.phase + "/elapsed",
                            reason="Observed actual elapsed span; any reference progress is separately planned. Native success predicate is unchanged and is not inferred here.",
                            duration=float(time) - self.started, kind="physical_elapsed")
            self.phase = None


class _Acquisition:
    def __init__(self, profile, command_timing):
        self.profile = profile
        self.original_timing = dict(command_timing)
        self.command_timing = dict(command_timing)
        self.prefix = float(command_timing["seat_start_s"])
        self.verticalize = float(command_timing["verticalize_start_s"])
        self.close = float(command_timing["finger_close_start_s"])
        self.physical_timing = None
        self.prefix_timing = None
        self.task_timing = None
        self.coverage = _PhysicalCoverage("right/acquisition")

    def plan_at_actual_dock(self, path):
        if self.prefix_timing is not None:
            raise RuntimeError("RIGHT actual-dock reference was already frozen")
        profile = json.loads(Path(path).read_text())
        curve = prefix_path(profile)
        source_duration = self.original_timing["seat_start_s"]
        fast = (self.profile["entry_native_s"] - self.profile["entry_native_tail_s"]
                - self.profile["entry_saved_s"])
        clock = shifted_clock(source_duration, fast, self.profile["entry_saved_s"],
                              start=self.profile["home_s"])
        self.prefix_timing = core.retime(
            curve, duration=source_duration, label="right/acquisition_actual_dock_prefix",
            clock=clock, vlim=self.profile["arm_rate_limit_rad_s"],
            protected=((0., self.profile["home_s"]),
                       (self.profile["home_s"] + fast, source_duration)),
            minimum_duration=self.profile["preshape_start_s"] + self.profile["preshape_duration_s"])
        self.prefix = self.prefix_timing.duration
        delta = self.prefix - source_duration
        self.command_timing.update({key: value + delta for key, value in self.original_timing.items()})
        self.close = self.command_timing["finger_close_start_s"]
        self.verticalize = self.command_timing["verticalize_start_s"]
        if self.physical_timing is None:
            raise RuntimeError("Missing original neutral acquisition timing dictionary")
        self.physical_timing.update(self.command_timing)
        task = acquisition_task_path(self.profile)
        task_duration = float(task.x[-1])
        self.task_timing = core.retime(
            task, duration=task_duration, label="right/acquisition_coupled_contact_clock",
            vlim=derivative_enclosure(task), protected=((0., task_duration),))
        profile["duration_s"] = self.prefix
        for waypoint in profile["waypoints"]:
            if abs(float(waypoint["time_s"]) - source_duration) < 1e-9:
                waypoint["time_s"] = self.prefix
        profile["staged_grasp_timing"] = dict(profile["staged_grasp_timing"], **self.command_timing)
        profile["topp_right_acquisition"] = self.report()
        Path(path).write_text(json.dumps(profile, indent=2))

    def task_clock(self, elapsed):
        if self.task_timing is None:
            raise RuntimeError("RIGHT task clock used before actual-dock planning")
        elapsed = float(elapsed)
        if elapsed < 0.:
            return .8 * elapsed
        if elapsed > self.task_timing.duration:
            return float(self.task_timing(self.task_timing.duration)[0]) + .8 * (elapsed - self.task_timing.duration)
        return float(self.task_timing(elapsed)[0])

    def observe(self, elapsed):
        phase = "prefix"
        for key, name in (("seat_start_s", "seat_ingress"), ("finger_close_start_s", "close_slot_escape"),
                          ("hold_start_s", "prelift_support"), ("lift_start_s", "loaded_lift"),
                          ("postlift_hold_start_s", "postlift_support"), ("verticalize_start_s", "verticalize"),
                          ("stage_end_s", "post_child_support")):
            if elapsed >= self.command_timing[key]:
                phase = name
        self.coverage.touch(elapsed, phase)

    def report(self):
        return dict(command_timing=dict(self.command_timing),
                    actual_physical_seat_start_relative_s=self.prefix,
                    preshape_and_mcp_clocks="unchanged actual elapsed acquisition time",
                    prefix=None if self.prefix_timing is None else self.prefix_timing.report,
                    coupled_task=None if self.task_timing is None else self.task_timing.report,
                    joint_acceleration_or_torque_certification=False)


def _prepare_acquisition(previous, *args, **kwargs):
    import v34_speedrightpickup as pickup
    runner, path, callback, audit, clock = previous(*args, **kwargs)
    state = _Acquisition(pickup.PROFILE, pickup.timing())
    runner._topp_acquisition = state
    old_prefix = repr(state.prefix)
    old_close = repr(state.close)
    old_verticalize = repr(state.verticalize)
    helpers = dict(_topp_acq=state)
    shifted = closure(runner.ik)
    close_entry, aligned = shifted["old"], shifted["native"]
    aligned_copy = copy_callable(aligned, [
        ("elapsed<" + old_prefix, "elapsed<_topp_acq.prefix"),
        ("elapsed>=verticalize", "elapsed>=_topp_acq.verticalize"),
        ("_r07_task_clock(elapsed-" + old_prefix + ")", "_topp_acq.task_clock(elapsed-_topp_acq.prefix)"),
    ], globals_=helpers, label="acquisition_aligned")
    elapsed = '(round((data.time-station_clock_origin[0])/model.opt.timestep)*model.opt.timestep)'
    close_copy = copy_callable(close_entry, [
        ("_r07_task_clock(" + elapsed + "-" + old_prefix + ")", "_topp_acq.task_clock(" + elapsed + "-_topp_acq.prefix)"),
        ("if " + old_prefix + "<=", "if _topp_acq.prefix<="),
        ("<" + old_close + ":", "<_topp_acq.close:"),
        ("<station_verticalize_start:", "<_topp_acq.verticalize:"),
    ], cells=dict(original_ik=aligned_copy), globals_=helpers, label="acquisition_ingress")
    runner.ik = copy_callable(runner.ik, [
        ("_r07_task_clock(elapsed-" + old_prefix + ")", "_topp_acq.task_clock(elapsed-_topp_acq.prefix)"),
        ("elapsed<vert", "elapsed<_topp_acq.verticalize"),
    ], cells=dict(old=close_copy, native=aligned_copy), globals_=helpers, label="acquisition_slot_escape")
    callback_cells = closure(callback)
    old_callback = copy_callable(callback_cells["old_callback"], [
        ("<" + old_prefix + " and", "<_topp_acq.prefix and"),
        ("_r07_task_clock(t-" + old_prefix + ")", "_topp_acq.task_clock(t-_topp_acq.prefix)"),
        ("or " + old_prefix + "<=t<" + old_verticalize + " or", "or _topp_acq.prefix<=t<_topp_acq.verticalize or"),
    ], globals_=helpers, label="acquisition_native_callback")
    coherent = copy_callable(callback, [
        ("in_hand_span = prefix <= elapsed < verticalize", "in_hand_span = _topp_acq.prefix <= elapsed < _topp_acq.verticalize"),
        ("task_clock(elapsed - prefix)", "_topp_acq.task_clock(elapsed - _topp_acq.prefix)"),
    ], cells=dict(old_callback=old_callback), globals_=helpers, label="acquisition_coherent_callback")
    origin = callback_cells["origin"]

    @wraps(callback)
    def observed_callback(m, d, after):
        if not after:
            state.observe(round((float(d.time) - origin[0]) / m.opt.timestep) * m.opt.timestep)
        return coherent(m, d, after)

    original_sample = runner.continuous_prefix_sample

    @wraps(original_sample)
    def sample(profile, seconds):
        if state.prefix_timing is None:
            raise RuntimeError("RIGHT prefix sampled before actual-stopped-dock plan")
        # Chain the native sampler/range validation at its own source time.
        return original_sample(profile, state.prefix_timing.source_time(float(seconds)))

    runner.continuous_prefix_sample = sample
    runner.station_task_clock = state.task_clock
    audit_cells = closure(audit)
    audit_copy = copy_callable(audit_cells["audit_copy"], [
        ("c['time_s']<" + old_prefix, "c['time_s']<_topp_acq.prefix"),
        ("'formal_close_time_s':" + old_prefix, "'formal_close_time_s':_topp_acq.prefix"),
    ], globals_=helpers, label="acquisition_contact_audit")
    coherent_audit = copy_callable(audit, cells=dict(audit_copy=audit_copy, command_timing=state.command_timing),
                                  label="acquisition_coherent_audit")

    @wraps(audit)
    def observed_audit():
        result = coherent_audit()
        if state.coverage.last is not None:
            state.coverage.finish(state.coverage.last)
        result["topp_right_acquisition"] = state.report()
        return result

    observed_audit.dock = audit.dock
    return runner, path, observed_callback, observed_audit, clock


def _cup_timing(m, d, start, goal, config, duration, out):
    from t1_right_transport_low_v3 import RightTransportLowConfig
    rim = RightTransportLowConfig().receiver_rim_z_m
    tip_z = float(d.site_xpos[m.site("pipette/tip_site").id, 2])
    dry = Path(out).name == "v6_right_direct_cup" and min(tip_z, .93) >= rim + .030
    curve = core.quintic(np.r_[start, 0.], np.r_[goal, np.deg2rad(config.aspiration_tool_world_yaw_delta_deg)], duration)
    timing = core.retime(curve, duration=duration, label="right/dry_cup_tip_and_axial_yaw",
                         vlim=derivative_enclosure(curve),
                         protected=() if dry else ((0., float(duration)),))
    timing.report["velocity_limit_source"] = "Original straight-tip/axial-yaw polynomial Bernstein enclosure; not a joint/torque bound"
    timing.report["dry_entry_guard_passed"] = dry
    return timing


def _cup_progress(timing, elapsed):
    return smooth(float(timing.clock(min(timing.duration, max(0., elapsed)))) / timing.source_duration)


def _route_timing(start, target, route_s):
    curve = core.quintic(start, np.asarray(target), float(route_s))
    timing = core.retime(curve, duration=route_s, label="right/loaded_mobile_moving_reference",
                         vlim=derivative_enclosure(curve))
    timing.report["velocity_limit_source"] = "Native moving-target polynomial bound, NOT actual chassis speed; original SpeedTransport physical caps remain separate"
    return timing


def _descent_progress(previous):
    native_timing = closure(previous).get("timing")
    if native_timing is None:
        raise RuntimeError("Expected the frozen R08 guarded-descent timing closure")
    planned = {}

    def progress(elapsed, anchor, goal, original_smooth):
        key = (tuple(np.asarray(anchor, dtype=float)), tuple(np.asarray(goal, dtype=float)))
        if key not in planned:
            fast, saved = native_timing(float(anchor[2]), float(goal[2]))
            duration = 3. - saved
            clock = shifted_clock(duration, fast, saved)
            curve = core.quintic(anchor, goal, 3.)
            timing = core.retime(curve, duration=duration, label="right/aspiration_guarded_descent",
                                 clock=clock, vlim=derivative_enclosure(curve, clock),
                                 protected=((float(fast), duration),))
            timing.report["velocity_limit_source"] = "Original guarded reference Bernstein enclosure; native1.5mm Cartesian lookahead and2rad/s guarded IK remain independent"
            planned[key] = timing
        timing = planned[key]
        return original_smooth(float(timing.clock(min(timing.duration, max(0., float(elapsed))))) / 3.)

    return progress


def _withdrawal_policy(previous):
    original_target = copy_callable(previous.target, [
        ("elapsed = time - self.started", "elapsed = time - self.started\n    self._ensure_topp_withdrawal()"),
        ("u = min(1., max(0., elapsed / 3.))", "u = min(1., max(0., float(self._topp_withdrawal.clock(min(self._topp_withdrawal.duration, max(0., elapsed)))) / 3.))"),
        ("self.reference_completed = u >= 1.", "self.reference_completed = elapsed >= self._topp_withdrawal.duration"),
        ("goal = self.anchor + smooth * delta", "goal = self._topp_withdrawal(elapsed)"),
        ("velocity = delta * 30. * u**2 * (1. - u)**2 / 3.", "velocity = self._topp_withdrawal(elapsed, 1)"),
        ("acceleration = delta * 60. * u * (1. - u) * (1. - 2.*u) / 9.", "acceleration = self._topp_withdrawal(elapsed, 2)"),
    ], label="withdrawal_persistent_native_target")

    class ToppContinuousWithdrawal(previous):
        def __init__(self, source, config=None):
            super().__init__(source, config)
            from t1_right_transport_low_v3 import RightTransportLowConfig
            native_config = config or RightTransportLowConfig()
            self._topp_guard_z = max(native_config.receiver_rim_z_m + .015,
                                     source.initial_surface_z_m + .015)
            self._topp_withdrawal = None
            self._topp_terminal_noted = False

        def _ensure_topp_withdrawal(self):
            if self._topp_withdrawal is not None:
                return
            distance = float(self.final_goal[2] - self.anchor[2])
            if distance <= 0.:
                guard_age = 3.
            else:
                guard_age = 3. * inverse_quintic((self._topp_guard_z - self.anchor[2]) / distance)
            curve = core.quintic(self.anchor, self.final_goal, 3.)
            self._topp_withdrawal = core.retime(curve, duration=3., label="right/persistent_vertical_withdrawal",
                                              vlim=derivative_enclosure(curve),
                                              protected=((0., guard_age),) if guard_age > 0. else ())
            self._topp_withdrawal.report["velocity_limit_source"] = "Original withdrawal Bernstein enclosure; native guarded IK remains authoritative"
            self._topp_withdrawal.report["anchor_contract"] = "First WITHDRAW anchor latched once; READY never replans or resets reference"

        def target(self, time, age, anchor, actual_tip):
            return original_target(self, time, age, anchor, actual_tip)

        def terminal_ready(self, elapsed, observer_ready, withdrawal_error):
            ready = super().terminal_ready(elapsed, observer_ready, withdrawal_error)
            if ready and not self._topp_terminal_noted:
                core.note_fixed("right/withdrawal_terminal_continuous_ready",
                                reason="Native predicate actually returned true: emitted terminal reference AND current READY AND error<2mm continuously.5s; resets and timeout behavior unchanged",
                                duration=float(elapsed) - self.terminal_since, kind="native_predicate_returned_true")
                self._topp_terminal_noted = True
            return ready

    return ToppContinuousWithdrawal


def _precision_policy(previous, envelope):
    native_transfer = previous.target.__globals__["transfer_path"]
    profile = envelope.PROFILE
    target = copy_callable(previous.target, [
        ("time - self.started < PROFILE.raise_duration_s", "time - self.started < self._topp_raise.duration"),
        ("time - self.started >= PROFILE.raise_duration_s", "time - self.started >= self._topp_raise.duration"),
        ("PROFILE.lower_far_duration_s + PROFILE.lower_native_tail_s", "self._lower_timing().duration"),
        ("lower_clock(time - self.lower_at)", "self._lower_clock(time - self.lower_at)"),
        ("['lower_all_time_reference_bounds']", "['lower_original_all_time_reference_bounds']"),
    ], globals_=dict(transfer_path=lambda distance: retimed_transfer(native_transfer, profile, distance)),
       label="precision_native_target")

    class ToppPrecisionPolicy(previous):
        def __init__(self, model, data, desired, qa, palm, tip):
            super().__init__(model, data, desired, qa, palm, tip)
            self._topp_native_data = data
            self._topp_clock_origin = float(data.time)
            self.retract, self.retract_split_at = retimed_retract(self.retract, self.retract_split_at, profile)
            self._topp_raise = None
            self._topp_lower = None
            self.reference_audit["topp_retract"] = self.retract.report
            self.reference_audit["parts_time_basis"] = "Native source seconds; see topp block maps for executed seconds"
            self.reference_audit["joint_acceleration_or_torque_certification"] = False

        def _raise_state(self, time):
            if self._topp_raise is None:
                curve = core.quintic(self.anchor, self.raised, profile.raise_duration_s)
                self._topp_raise = core.retime(curve, duration=profile.raise_duration_s,
                                               label="right/precision_raise", vlim=derivative_enclosure(curve))
                self._topp_raise.report["velocity_limit_source"] = "Native vertical-raise polynomial bound; no invented joint or horizontal-envelope claim"
                self.reference_audit["topp_raise"] = self._topp_raise.report
            age = float(time) - self.started
            return self._topp_raise(age), self._topp_raise(age, 1), self._topp_raise(age, 2)

        def _lower_timing(self):
            if self._topp_lower is None:
                delta = float(self.final[2] - self.lower_anchor[2])
                curve = core.quintic(0., delta, profile.lower_duration_s)
                duration = profile.lower_far_duration_s + profile.lower_native_tail_s
                saved = profile.lower_duration_s - duration
                clock = shifted_clock(duration, profile.lower_far_duration_s, saved)
                self._topp_lower = core.retime(curve, duration=duration, label="right/precision_lower",
                                               clock=clock, vlim=profile.fine_speed_m_s,
                                               alim=profile.fine_acceleration_m_s2,
                                               protected=((profile.lower_far_duration_s, duration),))
                self.reference_audit["topp_lower"] = self._topp_lower.report
                self.reference_audit["lower_all_time_reference_bounds"] = dict(
                    speed_m_s=self._topp_lower.report["all_time_velocity_bound"][0],
                    acceleration_m_s2=self._topp_lower.report["all_time_acceleration_bound"][0])
            return self._topp_lower

        def _lower_clock(self, age):
            timing = self._lower_timing()
            age = float(age)
            if age < 0.:
                return age, 1., 0.
            if age >= timing.duration:
                return profile.lower_duration_s + age - timing.duration, 1., 0.
            return float(timing.clock(age)), float(timing.clock(age, 1)), float(timing.clock(age, 2))

        def target(self, time, actual_tip, target_point):
            physical_time = float(self._topp_native_data.time) - self._topp_clock_origin + 9.
            if abs(float(time) - physical_time) > 1e-7:
                raise RuntimeError("Coating reference clock differs from native physical elapsed+9")
            result = target(self, time, actual_tip, target_point)
            if self.path is not None and "topp_transfer" not in self.reference_audit:
                self.reference_audit["topp_transfer"] = self.path.report
            return result

    return ToppPrecisionPolicy


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import v34_speedrightpickup as pickup
    import v34_smoothrightwithdrawal as withdrawal
    import v34_smoothrightenvelope as envelope
    import t1_right_station_yaw_launch_v7 as yaw
    import t1_right_standing_motion_v6 as motion
    import t1_right_passive_settle_v5 as passive
    import t1_right_interior_workstation_v6 as interior
    import t1_right_direct_coating_v6 as coating
    import t1_coarse_stop as coarse
    if not (pickup._INSTALLED and withdrawal._INSTALLED and envelope._INSTALLED
            and getattr(coarse.advance_coarse_stop, "_v34_speedtransport", False)):
        raise RuntimeError("TOPP RIGHT must install after all original R08 RIGHT plugins")
    previous_prepared = pickup.retime_prepared
    previous_engine = pickup.retime_engine_source
    previous_yaw = yaw.adapt_neutral_source_v7_yaw
    previous_reanchor = motion.reanchor_prefix_at_actual_dock_v6
    previous_passive = passive.passive_wait_before_gate
    previous_aspiration = interior.adapt_vertical_aspiration_source
    previous_coating = coating.adapt_direct_coating_source

    @wraps(previous_prepared)
    def prepared(*args, **kwargs):
        return _prepare_acquisition(previous_prepared, *args, **kwargs)

    @wraps(previous_engine)
    def engine_source(source):
        transformed = previous_engine(source)
        anchor = f"fine_readiness({pickup.prefix_duration()!r}+grasp_clock_origin[0],float(d.time))"
        return exact(transformed, anchor, "fine_readiness(grasp._topp_acquisition.prefix+grasp_clock_origin[0],float(d.time))")

    @wraps(previous_yaw)
    def neutral_source(source, config):
        transformed = previous_yaw(source, config)
        anchor = "        timing.update(clock);mujoco.mj_forward(m,d)"
        return exact(transformed, anchor, anchor + "\n        runner._topp_acquisition.physical_timing=timing")

    @wraps(previous_reanchor)
    def reanchor(m, d, module, path):
        result = previous_reanchor(m, d, module, path)
        module._topp_acquisition.plan_at_actual_dock(path)
        return result

    @wraps(previous_passive)
    def passive_wait(m, d, *args, **kwargs):
        started = float(d.time)
        result = previous_passive(m, d, *args, **kwargs)
        core.note_fixed("right/passive_support_and_held_lift",
                        reason="Native passive routine completed with unchanged captured controls, full-rate observer and actual held/lift gate; prior contact failures remain reported",
                        duration=float(d.time) - started, kind="physical_elapsed")
        return result

    # Preserve SpeedAspiration's named-phase/dry-entry wrapper and its report.
    previous_cup = motion.advance_direct_cup_v6
    cup_cells = closure(previous_cup)
    if "previous_cup" not in cup_cells:
        raise RuntimeError("Expected installed SpeedAspiration dry-cup wrapper")
    native_cup = cup_cells["previous_cup"]
    cup_loop = "    for k in range(int(round((duration_s+settle_s)/m.opt.timestep))):"
    cup_private = copy_callable(native_cup, [
        (cup_loop, "    _topp_cup=_topp_cup_timing(m,d,T0,goal,c,duration_s,out)\n"
                   "    _topp_duration=_topp_cup.duration\n"
                   "    for k in range(int(round((_topp_duration+settle_s)/m.opt.timestep))):"),
        ("u=float(np.clip(t/duration_s,0,1));u=u**3*(10+u*(-15+6*u));", "u=_topp_cup_progress(_topp_cup,t);"),
        ("k==int(round((duration_s+settle_s)/m.opt.timestep))-1", "k==int(round((_topp_duration+settle_s)/m.opt.timestep))-1"),
        ("    np.savez_compressed(out/'actual_states.npz',**raw);", "    report['topp_right_cup']=_topp_cup.report\n"
         "    _topp_core.note_fixed('right/dry_cup_settlement',reason='Native post-reference settlement completed; stopped-dock/error/contact predicates retained',duration=settle_s,kind='physical_elapsed')\n"
         "    np.savez_compressed(out/'actual_states.npz',**raw);"),
    ], globals_=dict(_topp_cup_timing=_cup_timing, _topp_cup_progress=_cup_progress, _topp_core=core),
       label="native_direct_cup")
    cup_wrapper = copy_callable(previous_cup, cells=dict(previous_cup=cup_private), label="dry_cup_wrapper")

    withdrawal_type = _withdrawal_policy(withdrawal.ContinuousVerticalWithdrawal)

    @wraps(previous_aspiration)
    def aspiration_source(source, config=None):
        transformed, environment, report = previous_aspiration(source, config)
        if environment.get("ContinuousVerticalWithdrawal") is not withdrawal.ContinuousVerticalWithdrawal:
            raise RuntimeError("Expected persistent R08 withdrawal policy in final aspiration adapter")
        if "_v34_aspiration_descent_progress" not in environment:
            raise RuntimeError("Expected native guarded-descent helper")
        transformed = exact(transformed, "    started_at=float(d.time)",
                            "    started_at=float(d.time)\n    _topp_exposure=_TOPPPhysicalCoverage('right/aspiration')")
        transformed = exact(transformed, "        t=float(d.time)-started_at",
                            "        t=float(d.time)-started_at\n        _topp_exposure.touch(t,observer.state)")
        transformed = exact(transformed, "'clock':'one original3s withdrawal reference across WITHDRAW and READY'",
                            "'clock':'one original3s geometric reference with a persistent TOPP clock across WITHDRAW and READY'")
        transformed = exact(transformed, "    arrays['metadata_json']=json.dumps(result)",
                            "    _topp_exposure.finish(float(d.time)-started_at)\n"
                            "    result['topp_withdrawal']=None if right_transport_policy._topp_withdrawal is None else right_transport_policy._topp_withdrawal.report\n"
                            "    arrays['metadata_json']=json.dumps(result)")
        environment = dict(environment, ContinuousVerticalWithdrawal=withdrawal_type,
                           _v34_aspiration_descent_progress=_descent_progress(environment["_v34_aspiration_descent_progress"]),
                           _TOPPPhysicalCoverage=_PhysicalCoverage)
        return transformed, environment, dict(report, topp_right=dict(
            previous_adapter_called_first=True, clock="unchanged actual aspiration elapsed seconds",
            dry_descent_and_withdrawal_reference_only=True, all_native_callbacks_and_predicates_retained=True))

    # Only the installed SpeedTransport's scoped inner native reference is copied.
    # Its native coarse class, physical caps, .4s slew/brake and .5s gate are not patched.
    # CoarseStopController ignores enable_gate and owns its distance/speed clock.
    # Keep the unused precision-base gate disabled, not tied to route completion.
    previous_inner = coarse._advance
    transfer_cells = closure(previous_inner)
    if set(("old_inner", "timed_inner")) - set(transfer_cells):
        raise RuntimeError("TOPP RIGHT requires installed SpeedTransport inner closure")
    native_inner = transfer_cells["old_inner"]
    if native_inner.__code__.co_name != "advance_mobile_stage":
        raise RuntimeError("Unexpected native loaded transfer loop")
    if native_inner.__globals__.get("MobileDockController") is not coarse.CoarseStopController:
        raise RuntimeError("Loaded reference requires the unchanged actual-distance coarse latch")
    native_route = copy_callable(native_inner, [
        ("    for k in range(int(duration_s/m.opt.timestep)):",
         "    _topp_route=_topp_route_timing(start,target_xy,route_s)\n    for k in range(int(duration_s/m.opt.timestep)):"),
        ("elapsed=float(d.time-t0);u=np.clip((elapsed-1)/route_s,0,1);h=u**3*(10-15*u+6*u*u)",
         "elapsed=float(d.time-t0);_topp_target=_topp_route(elapsed)"),
        ("start+h*(np.asarray(target_xy)-start),enable_gate=elapsed>=1+route_s",
         "_topp_target,enable_gate=False"),
        ("    return logs,audit,dock", "    audit['topp_loaded_reference']=_topp_route.report\n    return logs,audit,dock"),
    ], globals_=dict(_topp_route_timing=_route_timing), label="native_loaded_mobile_reference")
    transfer_wrapper = copy_callable(previous_inner, cells=dict(timed_inner=native_route), label="scoped_transport_inner")
    precision_type = _precision_policy(envelope.SmoothRightEntryPolicy, envelope)

    @wraps(previous_coating)
    def coating_source(source, config=None):
        transformed = previous_coating(source, config)
        # The native factory adds +9 after this adapter; validate the incoming
        # tick clock here and the final physical clock when the policy runs.
        if transformed.count("t=step*.001\n") != 1:
            raise RuntimeError("Expected native pre-offset coating tick clock")
        anchor = "    right_direct_coating_policy=SmoothRightEntryPolicy(m,d,desired,qa,palm,tip)"
        transformed = exact(transformed, anchor,
                            "    from v34_topp_right import _installed_precision_type,_PhysicalCoverage\n"
                            "    right_direct_coating_policy=_installed_precision_type(m,d,desired,qa,palm,tip)\n"
                            "    _topp_exposure=_PhysicalCoverage('right/coating')")
        transformed = exact(transformed, "        if phase!=last_phase:",
                            "        _topp_exposure.touch(t,phase)\n        if phase!=last_phase:")
        transformed = exact(transformed, "    arrays={k:np.array([r[k] for r in records]) for k in records[0]}",
                            "    _topp_exposure.finish(t+.001)\n"
                            "    arrays={k:np.array([r[k] for r in records]) for k in records[0]}")
        return transformed

    global _installed_precision_type
    _installed_precision_type = precision_type
    pickup.retime_prepared = prepared
    pickup.retime_engine_source = engine_source
    yaw.adapt_neutral_source_v7_yaw = neutral_source
    motion.reanchor_prefix_at_actual_dock_v6 = reanchor
    motion.advance_direct_cup_v6 = cup_wrapper
    passive.passive_wait_before_gate = passive_wait
    interior.adapt_vertical_aspiration_source = aspiration_source
    coarse._advance = transfer_wrapper
    coating.adapt_direct_coating_source = coating_source
    _INSTALLED = True
