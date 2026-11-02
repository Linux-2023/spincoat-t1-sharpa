"""Carry the original RIGHT withdrawal reference through the READY observer seam."""
from __future__ import annotations

from functools import wraps

import numpy as np

from t1_right_interior_workstation_v6 import VerticalCupWithdrawalV6
from v34_speedaspiration_profile import PROFILE


_INSTALLED = False


class ContinuousVerticalWithdrawal(VerticalCupWithdrawalV6):
    """Original3s vertical geometry, one persistent command clock and anchor.

    READY is the liquid observer's dry-above+.10s qualification. It is not a
    new arm operation. The obsolete READY+3.5s movement schedule is replaced
    by completed persistent reference plus consecutive READY/2mm for0.5s.
    No measured pose is substituted at this semantic state transition.
    """

    def __init__(self, source, config=None):
        super().__init__(source, config)
        self.started = None
        self.anchor = None
        self.last_age = None
        self.trace = []
        self.ready_seam = None
        self.command_audit = dict(maximum_arm_rate_rad_s=0., maximum_hand_rate_rad_s=0.)
        self.reference_completed = False
        self.terminal_since = None
        self.terminal_passed_at = None
        self.terminal_samples = 0

    def target(self, time, age, anchor, actual_tip):
        time, age = float(time), float(age)
        if self.started is None:
            self.started = time - age
            # WITHDRAW follows the unchanged immersed release/hold boundary.
            self.anchor = np.asarray(anchor).copy()
        elapsed = time - self.started
        u = min(1., max(0., elapsed / 3.))
        self.reference_completed = u >= 1.
        delta = self.final_goal - self.anchor
        smooth = u**3 * (10. + u * (-15. + 6. * u))
        goal = self.anchor + smooth * delta
        velocity = delta * 30. * u**2 * (1. - u)**2 / 3.
        acceleration = delta * 60. * u * (1. - u) * (1. - 2.*u) / 9.
        if self.last_age is not None and age < self.last_age:
            if self.ready_seam is not None:
                raise RuntimeError("Unexpected repeated withdrawal observer clock reset")
            self.ready_seam = dict(time_s=time, persistent_elapsed_s=elapsed,
                                   reference_position_m=goal.tolist(),
                                   reference_velocity_m_s=velocity.tolist(),
                                   reference_acceleration_m_s2=acceleration.tolist())
        self.last_age = age
        self.trace.append(dict(time_s=time, observer_age_s=age,
                               persistent_elapsed_s=elapsed,
                               actual_tip_world_m=np.asarray(actual_tip).tolist(),
                               reference_tip_world_m=goal.tolist(),
                               reference_velocity_m_s=velocity.tolist(),
                               reference_acceleration_m_s2=acceleration.tolist()))
        return goal

    def terminal_ready(self, elapsed, observer_ready, withdrawal_error):
        """Called after every native step, in the SAME relative clock as target.

        Completion is latched only when target emitted its exact3s endpoint.
        The full0.5s window restarts on any lost READY/2mm sample; no prior
        qualified sample is cached as a replacement for the current state.
        """
        eligible = self.reference_completed and observer_ready and withdrawal_error < .002
        if not eligible:
            self.terminal_since = None
            self.terminal_samples = 0
            return False
        elapsed = float(elapsed)
        if self.terminal_since is None:
            self.terminal_since = elapsed
        self.terminal_samples += 1
        ready = elapsed - self.terminal_since >= .5
        if ready and self.terminal_passed_at is None:
            self.terminal_passed_at = elapsed
        return ready


def _exact(source, old, new):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"SmoothRightWithdrawal anchor count {count}: {old!r}")
    return source.replace(old, new, 1)


def _command_bounds(model, joints, values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite aspiration command")
    for joint, value in zip(joints, values):
        if model.jnt_limited[joint] and not model.jnt_range[joint, 0] <= value <= model.jnt_range[joint, 1]:
            raise RuntimeError("Aspiration command exceeds native joint range")
        acts = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(acts) != 1:
            raise RuntimeError("Expected one native RIGHT position actuator")
        actuator = int(acts[0])
        if model.actuator_ctrllimited[actuator] and not model.actuator_ctrlrange[actuator, 0] <= value <= model.actuator_ctrlrange[actuator, 1]:
            raise RuntimeError("Aspiration command exceeds native actuator range")


def _command_rate(policy, kind, data, previous, proposed, limit):
    now = float(data.time)
    key = '_last_' + kind + '_command_time'
    last = getattr(policy, key, None)
    if last is not None:
        dt = now - last
        if abs(dt - .020) > 1e-6:
            raise RuntimeError("Aspiration command cadence must remain50Hz")
        rate = float(np.max(np.abs(np.asarray(proposed) - previous))) / dt
        metric = 'maximum_' + kind + '_rate_rad_s'
        policy.command_audit[metric] = max(policy.command_audit[metric], rate)
        if not np.isfinite(rate) or rate > limit + 1e-9:
            raise RuntimeError("Aspiration command-rate bound failed: " + repr((kind, rate)))
    setattr(policy, key, now)


def checked_arm_ik(original, model, data, goal, quat, *, seed, iterations, policy):
    previous = np.array(seed, copy=True)
    result = original(model, data, goal, quat, seed=seed, iterations=iterations)
    from t1_spincoat_scene import arm_indices
    joints = arm_indices(model)[0]
    _command_bounds(model, joints, result[0])
    _command_rate(policy, 'arm', data, previous, result[0], 2.)
    return result


def checked_thumb_ik(original, model, data, goal, hand, joints, qadr, policy):
    previous = np.array(hand, copy=True)
    result = original(model, data, goal, hand, joints, qadr)
    _command_bounds(model, joints, result)
    _command_rate(policy, 'hand', data, previous, result, 3.)
    return result


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    # Keep the frozen guarded-descent implementation, selecting only its
    # bounded dry-positioning parameters from this immutable profile.
    import v34_speedaspiration
    v34_speedaspiration.install()
    import t1_right_interior_workstation_v6 as interior
    previous = interior.adapt_vertical_aspiration_source

    @wraps(previous)
    def continuous_withdrawal_source(source, config=None):
        transformed, environment, report = previous(source, config)
        if environment.get('RightLowWithdrawalPolicy') is not VerticalCupWithdrawalV6:
            raise RuntimeError("Expected R05 vertical-only withdrawal policy; no bypass fallback")
        transformed = _exact(transformed,
            'right_transport_policy=RightLowWithdrawalPolicy(source,RIGHT_TRANSPORT_LOW_CONFIG)',
            'right_transport_policy=ContinuousVerticalWithdrawal(source,RIGHT_TRANSPORT_LOW_CONFIG)')
        hover = float(PROFILE['dry_thumb_hover_s'])
        if not .75 <= hover <= 1.5:
            raise ValueError("Dry thumb approach exceeds declared timing bracket")
        old_thumb = 'u=smooth(t/1.5);above=top+axis*.014;thumb_goal=(1-u)*initial_thumb+u*above if t<1.5 else top+axis*(.014-.029*smooth((t-1.5)/2.5))'
        new_thumb = ('u=smooth(t/' + repr(hover) + ');above=top+axis*.014;thumb_goal=(1-u)*initial_thumb+u*above if t<' + repr(hover) + ' else top+axis*(.014-.029*smooth((t-' + repr(hover) + ')/2.5))')
        transformed = _exact(transformed, old_thumb, new_thumb)
        transformed = _exact(transformed,
            'armq,error=ik(m,d,handgoal,quat,seed=armq,iterations=35)',
            'armq,error=checked_arm_ik(ik,m,d,handgoal,quat,seed=armq,iterations=35,policy=right_transport_policy)')
        transformed = _exact(transformed,
            'hand=thumb_ik(m,d,thumb_goal,hand,hj,hq)',
            'hand=checked_thumb_ik(thumb_ik,m,d,thumb_goal,hand,hj,hq,right_transport_policy)')
        legacy_terminal = (
            "        if observer.state==Stage.READY:\n"
            "            if ready_at is None:ready_at=float(d.time)\n"
            "            if d.time-ready_at>3.5 and np.linalg.norm(d.site_xpos[tip]-withdrawal_goal)<.002:break")
        qualified_terminal = (
            "        if observer.state==Stage.READY and ready_at is None:ready_at=float(d.time)\n"
            "        if right_transport_policy.terminal_ready(float(d.time)-started_at,observer.state==Stage.READY,float(np.linalg.norm(d.site_xpos[tip]-withdrawal_goal))):break")
        transformed = _exact(transformed, legacy_terminal, qualified_terminal)
        # Reaching the unchanged overall time ceiling cannot masquerade as a
        # qualified handoff if the consecutive terminal window never completed.
        native_success = "observer.state==Stage.READY and not failure and not collision_events and withdrawal_error<.002 and bool(np.all(grip_min[1:]>0))"
        transformed = _exact(transformed, native_success,
            native_success + " and right_transport_policy.terminal_passed_at is not None")
        evidence = (
            "    result['smooth_right_withdrawal']={"
            "'clock':'one original3s withdrawal reference across WITHDRAW and READY',"
            "'replaced_nominal_ready_delay_s':3.5,'terminal_continuous_ready_window_s':.5,"
            "'ready_seam':right_transport_policy.ready_seam,"
            "'terminal_clock':'elapsed native aspiration seconds',"
            "'terminal_reference_completed':right_transport_policy.reference_completed,"
            "'terminal_qualified_since_s':right_transport_policy.terminal_since,"
            "'terminal_passed_at_s':right_transport_policy.terminal_passed_at,"
            "'terminal_consecutive_samples':right_transport_policy.terminal_samples,"
            "'trace':right_transport_policy.trace,'command_audit':right_transport_policy.command_audit}\n")
        transformed = _exact(transformed, "    arrays['metadata_json']=json.dumps(result)",
                             evidence + "    arrays['metadata_json']=json.dumps(result)")
        environment = dict(environment, ContinuousVerticalWithdrawal=ContinuousVerticalWithdrawal,
                           checked_arm_ik=checked_arm_ik, checked_thumb_ik=checked_thumb_ik)
        report = dict(report, smooth_right_withdrawal=dict(
            previous_adapter_called_first=True,
            unchanged_vertical_geometry=True,
            replaced_nominal_ready_delay_s=3.5,
            terminal_continuous_ready_window_s=.5,
            terminal_predicate='persistent reference complete AND current READY AND current tip error<2mm, continuously0.5s',
            dry_thumb_hover_s=hover,
            unchanged_press_stroke_s=2.5,
            unchanged_release_stroke_s=3.0,
            original_liquid_qualification_windows_unchanged=True,
            no_lateral_bypass=True,
            no_control_or_physics_clock_change=True))
        report['speed_aspiration'] = dict(report['speed_aspiration'],
            withdrawal_and_ready_hold_unchanged=False,
            withdrawal_motion_unchanged=True,
            ready_schedule_owner='SmoothRightWithdrawal aggressive terminal qualification')
        return transformed, environment, report

    interior.adapt_vertical_aspiration_source = continuous_withdrawal_source
    _INSTALLED = True
