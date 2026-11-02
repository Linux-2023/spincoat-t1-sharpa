"""RIGHT-only continuous transit references; native physical gates stay authoritative."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps
import math

import mujoco
import numpy as np

from t1_right_direct_coating_v6 import RightDirectCoatingPolicyV6


@dataclass(frozen=True)
class PrecisionProfile:
    name: str = "aggressive_r07_01"
    coarse_lookahead_m: float = .005
    clear_speed_m_s: float = .180
    clear_acceleration_m_s2: float = .500
    fine_distance_m: float = .020
    fine_speed_m_s: float = .050
    fine_acceleration_m_s2: float = .120
    pass_speed_m_s: float = .050
    raise_duration_s: float = .40
    lower_duration_s: float = 2.00
    lower_native_tail_s: float = .75
    lower_far_duration_s: float = .80
    retract_initial_distance_m: float = .030
    retract_initial_speed_m_s: float = .066
    retract_initial_acceleration_m_s2: float = .240
    retract_clear_speed_m_s: float = .120
    retract_clear_acceleration_m_s2: float = .500
    command_rate_limit_rad_s: float = 2.0


PROFILE = PrecisionProfile()
_INSTALLED = False


def _exact(source, anchor, replacement):
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"SmoothRightEnvelope anchor count {count}: {anchor!r}")
    return source.replace(anchor, replacement, 1)


def _smooth(u):
    u = min(1., max(0., float(u)))
    return u**3 * (10. + u * (-15. + 6. * u))


def _integrated_smooth(u):
    return u**4 * (2.5 + u * (-3. + u))


def lower_clock(age):
    """Retain the exact native final0.75s of lowering, not merely its endpoint.

    The2s polynomial remains the spatial reference. Only its distant prefix
    gains time, with unit clock rate and zero acceleration at both seams.
    """
    far = PROFILE.lower_far_duration_s
    saved = PROFILE.lower_duration_s - PROFILE.lower_native_tail_s - far
    age = float(age)
    if age <= 0.:
        return age, 1., 0.
    if age >= far:
        return age + saved, 1., 0.
    u = age / far
    return (age + saved * _smooth(u),
            1. + saved * 30. * u**2 * (1. - u)**2 / far,
            saved * 60. * u * (1. - u) * (1. - 2.*u) / far**2)


def lower_bounds(distance):
    """Conservative all-time chain-rule bounds, checked before lowering."""
    far = PROFILE.lower_far_duration_s
    native = PROFILE.lower_duration_s
    saved = native - PROFILE.lower_native_tail_s - far
    if saved < 0. or far <= 0.:
        raise ValueError("Invalid lower clock")
    rate = 1. + 1.875 * saved / far
    accel = (10. / math.sqrt(3.)) * saved / far**2
    v = abs(distance) * 1.875 / native
    a = abs(distance) * (10. / math.sqrt(3.)) / native**2
    bounds = dict(speed_m_s=v * rate, acceleration_m_s2=a * rate**2 + v * accel)
    if bounds['speed_m_s'] > PROFILE.fine_speed_m_s or bounds['acceleration_m_s2'] > PROFILE.fine_acceleration_m_s2:
        raise RuntimeError("Lower reference exceeds unchanged fine envelope: " + repr(bounds))
    return bounds


class VelocityPath:
    """Exact integral of C2 velocity ramps: position is C3 at every join.

    A ramp v0->v1 has distance (v0+v1)*T/2 and maximum acceleration
    1.875*abs(v1-v0)/T. There is no interpolation of recorded states.
    """

    def __init__(self):
        self.parts = []
        self.duration = 0.
        self.distance = 0.
        self.velocity = 0.

    def add(self, velocity, duration, phase):
        velocity, duration = float(velocity), float(duration)
        if not math.isfinite(velocity + duration) or velocity < 0. or duration < 0.:
            raise ValueError("Invalid continuous reference span")
        if duration == 0.:
            if abs(velocity - self.velocity) > 1e-12:
                raise ValueError("Zero-duration velocity discontinuity")
            return
        self.parts.append((self.duration, duration, self.distance,
                           self.velocity, velocity, phase))
        self.distance += .5 * (self.velocity + velocity) * duration
        self.duration += duration
        self.velocity = velocity

    def sample(self, age):
        age = float(age)
        if not self.parts:
            return 0., 0., 0., "fine_transfer"
        if age <= 0.:
            return 0., 0., 0., self.parts[0][-1]
        if age >= self.duration:
            return self.distance, self.velocity, 0., self.parts[-1][-1]
        for start, duration, distance, v0, v1, phase in self.parts:
            if age < start + duration:
                u = (age - start) / duration
                delta = v1 - v0
                return (distance + duration * (v0 * u + delta * _integrated_smooth(u)),
                        v0 + delta * _smooth(u),
                        delta / duration * 30. * u**2 * (1. - u)**2,
                        phase)
        raise RuntimeError("Continuous reference has an uncovered time interval")


def _append_velocity_envelope(path, distance, terminal_speed, speed_limit,
                              acceleration_limit, phase):
    """Append a feasible C2 velocity envelope with arbitrary endpoint speeds.

    Quintic velocity ramps have duration K*abs(v1-v0)/A, K=1.875.
    The minimum distance connecting endpoint speeds is
        Dmin = K*abs(v1-v0)*(v1+v0)/(2*A).
    Any remaining distance can support a peak above BOTH endpoint speeds, not
    merely a long deceleration from a tiny incoming speed. The two regimes
    (distance-limited peak; speed-capped peak plus cruise) meet continuously.
    """
    initial = path.velocity
    values = (distance, terminal_speed, speed_limit, acceleration_limit, initial)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Nonfinite velocity envelope parameter")
    if (distance < 0. or terminal_speed < 0. or initial < 0.
            or speed_limit <= 0. or acceleration_limit <= 0.
            or max(initial, terminal_speed) > speed_limit):
        raise ValueError("Invalid velocity envelope bounds")
    k = 1.875
    def ramps(peak):
        up = k * (peak - initial) / acceleration_limit
        down = k * (peak - terminal_speed) / acceleration_limit
        travel = (.5 * (initial + peak) * up
                  + .5 * (terminal_speed + peak) * down)
        return up, down, travel

    base = max(initial, terminal_speed)
    minimum_distance = ramps(base)[2]
    if distance < minimum_distance:
        raise ValueError("Infeasible endpoint speeds for reference distance: "
                         + repr((distance, initial, terminal_speed, minimum_distance)))
    if distance == 0.:
        return
    cap_up, cap_down, cap_distance = ramps(speed_limit)
    if distance >= cap_distance:
        # This branch proves the cruise distance is nonnegative. No clipping.
        peak = speed_limit
        up, down = cap_up, cap_down
        cruise = (distance - cap_distance) / peak
    else:
        # D(peak)=K*(2*peak²-initial²-terminal²)/(2*A) is monotone.
        # Keep the feasible lower bracket, using the same ramp arithmetic as
        # emission. This avoids rounding an analytic square root above a cap
        # or turning a tiny negative cruise into zero. No tolerance/epsilon
        # defines a geometric branch; stop only at adjacent float values.
        low, high = base, speed_limit
        while True:
            middle = low + .5 * (high - low)
            if middle == low or middle == high:
                break
            if ramps(middle)[2] <= distance:
                low = middle
            else:
                high = middle
        peak = low
        up, down, ramp_distance = ramps(peak)
        cruise = (distance - ramp_distance) / peak
    if peak > speed_limit or min(up, down, cruise) < 0.:
        raise RuntimeError("Velocity envelope calculation violated its bounds")
    path.add(peak, up, phase)
    path.add(peak, cruise, phase)
    path.add(terminal_speed, down, phase)


def transfer_path(distance):
    """One continuous feasible line under explicit coarse and fine envelopes.

    The split-speed rule reserves at least half of each nonempty span for
    acceleration/deceleration beyond its endpoint-speed connection. It is a
    geometric feasibility rule, not an epsilon special case at20mm.
    A low incoming speed may accelerate within the fine-side caps before
    decelerating to rest, avoiding a duration proportional to1/incoming_speed.
    """
    distance = float(distance)
    if not math.isfinite(distance) or distance < 0.:
        raise ValueError("Invalid coating transfer distance")
    path = VelocityPath()
    fine = min(PROFILE.fine_distance_m, distance)
    clear = distance - fine
    if clear == 0.:
        # Same general endpoint-speed envelope as the fine span below. An
        # empty coarse span is an exact structural case, not an epsilon band.
        _append_velocity_envelope(path, fine, 0., PROFILE.fine_speed_m_s,
                                  PROFILE.fine_acceleration_m_s2, "fine_transfer")
        return path, 0., 0.
    k = 1.875
    through = min(PROFILE.pass_speed_m_s, PROFILE.clear_speed_m_s,
                  PROFILE.fine_speed_m_s,
                  math.sqrt(clear * PROFILE.clear_acceleration_m_s2 / k),
                  math.sqrt(fine * PROFILE.fine_acceleration_m_s2 / k))
    _append_velocity_envelope(path, clear, through, PROFILE.clear_speed_m_s,
                              PROFILE.clear_acceleration_m_s2, "clear_transfer")
    split_at = path.duration
    _append_velocity_envelope(path, fine, 0., PROFILE.fine_speed_m_s,
                              PROFILE.fine_acceleration_m_s2, "fine_transfer")
    if abs(path.distance - distance) > 1e-10:
        raise RuntimeError("Continuous transfer distance mismatch")
    return path, split_at, through


def retract_path():
    """Keep slow near-substrate clearance and pass through30mm without stopping."""
    path = VelocityPath()
    slow = PROFILE.retract_initial_speed_m_s
    first = PROFILE.retract_initial_distance_m
    ramp = 1.875 * slow / PROFILE.retract_initial_acceleration_m_s2
    cruise = (first - .5 * slow * ramp) / slow
    if cruise < 0.:
        raise RuntimeError("Invalid initial retract clearance envelope")
    path.add(slow, ramp, "retract_initial")
    path.add(slow, cruise, "retract_initial")
    split_at = path.duration
    accel = PROFILE.retract_clear_acceleration_m_s2
    remaining = .10 - first
    peak = min(PROFILE.retract_clear_speed_m_s,
               math.sqrt((remaining * accel / .9375 + slow**2) / 2.))
    up = 1.875 * (peak - slow) / accel
    down = 1.875 * peak / accel
    cruise = (remaining - .5 * (slow + peak) * up - .5 * peak * down) / peak
    if cruise < -1e-10 or up < 0.:
        raise RuntimeError("Invalid clear retract envelope")
    path.add(peak, up, "retract_clear")
    path.add(peak, max(0., cruise), "retract_clear")
    path.add(0., down, "retract_clear")
    if abs(path.distance - .10) > 1e-10:
        raise RuntimeError("Continuous retract distance mismatch")
    return path, split_at


class SmoothRightEntryPolicy(RightDirectCoatingPolicyV6):
    """C2 rounded arrival corners gated by original measured-position predicates."""

    def __init__(self, model, data, desired, qa, palm, tip):
        super().__init__()
        self.profile = PROFILE
        if not .12 <= PROFILE.clear_speed_m_s <= .20 or not .30 <= PROFILE.clear_acceleration_m_s2 <= .55:
            raise ValueError("Coarse coating profile exceeds aggressive bracket")
        if not .4 <= PROFILE.raise_duration_s <= .6 or not .8 <= PROFILE.lower_far_duration_s <= 1.0:
            raise ValueError("Coating timing exceeds aggressive bracket")
        if (PROFILE.lower_duration_s != 2. or PROFILE.lower_native_tail_s != .75
                or PROFILE.fine_distance_m != .02 or PROFILE.fine_speed_m_s != .05
                or PROFILE.fine_acceleration_m_s2 != .12 or PROFILE.coarse_lookahead_m != .005
                or PROFILE.command_rate_limit_rad_s != 2.):
            raise ValueError("Protected R06 fine envelope changed")
        # FK only on isolated planning data; live qpos/qvel/time remain untouched.
        work = mujoco.MjData(model)
        work.qpos[:] = data.qpos
        work.qpos[qa] = desired[qa]
        mujoco.mj_kinematics(model, work)
        actual_rotation = data.xmat[palm].reshape(3, 3)
        local_tip = actual_rotation.T @ (data.site_xpos[tip] - data.xpos[palm])
        self.initial_command = work.xpos[palm] + work.xmat[palm].reshape(3, 3) @ local_tip
        gap = float(np.linalg.norm(self.initial_command - data.site_xpos[tip]))
        if not np.isfinite(gap) or gap > .005:
            raise RuntimeError("Coating entry command/actual mismatch exceeds5mm: " + str(gap))
        self.command_goal = self.initial_command.copy()
        self.final = None
        self.path = None
        self.retract, self.retract_split_at = retract_path()
        self.transfer_split_at = None
        self.pass_speed = None
        self.reference_audit = dict(entry_command_actual_gap_m=gap)
        self.terminal_reference_ready = False
        self.reference_terminal_at_s = None
        self.first_measured_ready_physical_s = None
        self.handshake_passed_physical_s = None
        self.sweep_work = work
        self.qa = np.asarray(qa)
        self.palm = palm
        self.pipe = model.body('pipette/pipette').id
        self.pipe_qadr = int(model.jnt_qposadr[int(model.body_jntadr[self.pipe])])
        self.corner_active = False

    def _raise_state(self, time):
        u = min(1., max(0., (time - self.started) / PROFILE.raise_duration_s))
        delta = self.raised - self.anchor
        return (self.anchor + _smooth(u) * delta,
                delta * 30. * u**2 * (1. - u)**2 / PROFILE.raise_duration_s,
                delta * 60. * u * (1. - u) * (1. - 2.*u) / PROFILE.raise_duration_s**2)

    def target(self, time, actual_tip, target):
        time = float(time)
        p = np.asarray(actual_tip)
        transit = np.asarray(target) + np.array([.0001, 0., self.height])
        final = np.asarray(target) + np.array([.0001, 0., .012])
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        if self.started is None:
            self.started = time
            self.anchor = self.initial_command.copy()
            self.raised = np.r_[self.anchor[:2], transit[2]]
            self.final = final.copy()
        elif not np.array_equal(final, self.final):
            raise RuntimeError("Coating RGB target changed inside a continuous span")
        raised_goal, raised_velocity, raised_acceleration = self._raise_state(time)
        self.corner_active = False
        if self.stage == "raise":
            goal, velocity, acceleration = raised_goal, raised_velocity, raised_acceleration
            # The native1mm height gate is positional, not a stationary hold.
            # Finish its vertical polynomial while starting the lateral component
            # only inside the same measured AND nominal clearance corridor.
            # Its shorter polynomial is still completed before lower begins.
            if abs(p[2] - transit[2]) < .001 and abs(goal[2] - transit[2]) < .001:
                self.direct_at = time
                self.direct_anchor = self.raised.copy()
                delta = transit - self.direct_anchor
                distance = float(np.linalg.norm(delta))
                self.direction = delta / distance if distance else np.zeros(3)
                self.path, self.transfer_split_at, self.pass_speed = transfer_path(distance)
                self.direct_duration = self.transfer_split_at
                self.stage = "clear_transfer" if self.transfer_split_at else "fine_transfer"
                old_clear = max(1.2, 1.875 * max(0., distance - .020) / .160,
                                math.sqrt(10./math.sqrt(3.) * max(0., distance - .020) / .300))
                self.reference_audit.update(
                    raise_to_transfer_at_s=time,
                    raise_to_transfer_velocity_m_s=raised_velocity.tolist(),
                    transfer_distance_m=distance,
                    continuous_transfer_duration_s=self.path.duration,
                    previous_split_transfer_duration_s=old_clear + 1.4,
                    transfer_split_at_s=self.transfer_split_at,
                    transfer_pass_speed_m_s=self.pass_speed,
                    transfer_parts=self.path.parts)
        if self.stage in ("clear_transfer", "fine_transfer", "vertical_lower"):
            age = time - self.direct_at
            distance, speed, accel, path_phase = self.path.sample(age)
            goal = self.direct_anchor + distance * self.direction
            velocity = speed * self.direction
            acceleration = accel * self.direction
            if self.stage != "vertical_lower":
                self.stage = path_phase
                goal[2], velocity[2], acceleration[2] = raised_goal[2], raised_velocity[2], raised_acceleration[2]
                self.corner_active = time - self.started < PROFILE.raise_duration_s
                # Both original actual position gates are unchanged. There is
                # no original zero-velocity predicate here. Keep the lateral
                # deceleration tail instead of stopping at the right-angle knot.
                if (np.linalg.norm(p[:2] - transit[:2]) < .0015
                        and abs(p[2] - transit[2]) < .001
                        and np.linalg.norm(goal[:2] - transit[:2]) < .0015
                        and time - self.started >= PROFILE.raise_duration_s):
                    self.stage = "vertical_lower"
                    self.lower_at = time
                    self.lower_anchor = transit.copy()
                    self.reference_audit['lower_all_time_reference_bounds'] = lower_bounds(self.final[2] - self.lower_anchor[2])
                    self.reference_audit['lower_native_duration_s'] = PROFILE.lower_duration_s
                    self.reference_audit['lower_executed_duration_s'] = PROFILE.lower_far_duration_s + PROFILE.lower_native_tail_s
                    self.reference_audit['lower_unchanged_terminal_tail_s'] = PROFILE.lower_native_tail_s
                    self.reference_audit.update(
                        transfer_to_lower_at_s=time,
                        transfer_to_lower_velocity_m_s=velocity.tolist(),
                        transfer_to_lower_acceleration_m_s2=acceleration.tolist(),
                        transfer_to_lower_actual_xy_error_m=float(np.linalg.norm(p[:2] - transit[:2])),
                        transfer_to_lower_actual_z_error_m=float(abs(p[2] - transit[2])))
            if self.stage == "vertical_lower":
                native_age, clock_rate, clock_acceleration = lower_clock(time - self.lower_at)
                u = min(1., max(0., native_age / PROFILE.lower_duration_s))
                delta_z = self.final[2] - self.lower_anchor[2]
                goal[2] = self.lower_anchor[2] + _smooth(u) * delta_z
                native_velocity = delta_z * 30. * u**2 * (1. - u)**2 / PROFILE.lower_duration_s
                native_acceleration = delta_z * 60. * u * (1. - u) * (1. - 2.*u) / PROFILE.lower_duration_s**2
                velocity[2] = native_velocity * clock_rate
                acceleration[2] = native_acceleration * clock_rate**2 + native_velocity * clock_acceleration
                self.corner_active = age < self.path.duration
                self.terminal_reference_ready = bool(u >= 1. and age >= self.path.duration)
                if self.terminal_reference_ready:
                    goal = self.final.copy()
                    velocity.fill(0.)
                    acceleration.fill(0.)
                    if self.reference_terminal_at_s is None:
                        self.reference_terminal_at_s = time
                        self.reference_audit["reference_terminal_at_s"] = time
        if not np.isfinite(goal).all():
            raise RuntimeError("Nonfinite coating reference")
        self.command_goal = goal.copy()
        self.trace.append(dict(controller_time_s=time, phase=self.stage,
                               actual_tip_world_m=p.tolist(), goal_tip_world_m=goal.tolist(),
                               reference_velocity_m_s=velocity.tolist(),
                               reference_acceleration_m_s2=acceleration.tolist(),
                               rounded_corner_active=self.corner_active,
                               terminal_reference_ready=self.terminal_reference_ready))
        return goal

    def retract_target(self, time, actual_tip):
        if not hasattr(self, "retract_at"):
            self.retract_at = float(time)
            # The terminal-reference handshake ensures the preceding native
            # press starts and ends at this same zero-derivative reference.
            self.retract_anchor = self.final.copy()
        distance, speed, accel, phase = self.retract.sample(float(time) - self.retract_at)
        goal = self.retract_anchor + np.array([0., 0., distance])
        self.trace.append(dict(controller_time_s=float(time), phase=phase,
                               actual_tip_world_m=np.asarray(actual_tip).tolist(),
                               goal_tip_world_m=goal.tolist(),
                               reference_velocity_m_s=[0., 0., speed],
                               reference_acceleration_m_s2=[0., 0., accel]))
        self.corner_active = False
        return goal


def check_corner_sweep(model, data, previous, proposed, policy):
    """Reject enabled-collider intersections on isolated planning data only.

    The tool pose below is a hypothetical measured hand/tool transform for a
    local command sweep, never a live weld, force, attachment or state write.
    Actual grip/contact/liquid observers still see every real physics sample.
    This is not a whole-mesh or continuous-collision certificate.
    """
    if not policy.corner_active:
        return
    work = policy.sweep_work
    palm, pipe = policy.palm, policy.pipe
    actual_rotation = data.xmat[palm].reshape(3, 3)
    relative_position = actual_rotation.T @ (data.xpos[pipe] - data.xpos[palm])
    relative_rotation = actual_rotation.T @ data.xmat[pipe].reshape(3, 3)
    quaternion = np.empty(4)
    for fraction in (0., .25, .5, .75, 1.):
        work.qpos[:] = data.qpos
        work.qpos[policy.qa] = previous + fraction * (proposed - previous)
        mujoco.mj_kinematics(model, work)
        planned_rotation = work.xmat[palm].reshape(3, 3)
        qadr = policy.pipe_qadr
        work.qpos[qadr:qadr + 3] = work.xpos[palm] + planned_rotation @ relative_position
        mujoco.mju_mat2Quat(quaternion, (planned_rotation @ relative_rotation).reshape(9))
        work.qpos[qadr + 3:qadr + 7] = quaternion
        mujoco.mj_fwdPosition(model, work)
        for contact in work.contact:
            names = [model.body(int(model.geom_bodyid[int(g)])).name for g in contact.geom]
            moving = any(n.startswith(('astribot_', 'sharpa_', 'pipette/')) for n in names)
            obstacle = any(n.startswith(('lab_bench', 'stand/', 'spincoater/', 'aspiration_source')) for n in names)
            if moving and obstacle and contact.dist < 0.:
                raise RuntimeError("SmoothRightEnvelope rounded-corner planning collision: " + repr(names))
    if policy.trace:
        policy.trace[-1]['enabled_collider_corner_sweep_samples'] = 5


def coarse_lookahead(time, policy, actual_tip, target, phase):
    """R05's exact5mm envelope; no further lookahead increase."""
    time = float(time)
    previous = getattr(policy, '_lookahead_previous_time', None)
    policy._lookahead_previous_time = time
    policy._lookahead_update_dt = None if previous is None else time - previous
    limit = .0025
    if phase in ('translate', 'lower') and policy.stage == 'clear_transfer':
        age = time - float(policy.direct_at)
        allowed = max(0., min(float(policy.direct_duration),
                             2.2 - (float(policy.direct_at) - float(policy.started))))
        if allowed > 0.:
            p = np.asarray(actual_tip)
            target = np.asarray(target)
            remaining = float(np.linalg.norm(p[:2] - target[:2]))
            height = float(p[2] - target[2])
            weight = (_smooth(age / .40) * _smooth((allowed - age) / .60)
                      * _smooth((remaining - .040) / .030)
                      * _smooth((height - .020) / .003))
            limit += (PROFILE.coarse_lookahead_m - .0025) * weight
    policy._lookahead_current_limit = limit
    if policy.trace and abs(policy.trace[-1]['controller_time_s'] - time) < 1e-8:
        policy.trace[-1]['cartesian_lookahead_m'] = limit
        policy.trace[-1]['command_update_dt_s'] = policy._lookahead_update_dt
    return limit


def checked_ik(original, model, data, hand_goal, quat, *, seed, iterations, policy):
    previous = np.array(seed, copy=True)
    result = original(model, data, hand_goal, quat, seed=seed, iterations=iterations)
    dt = policy._lookahead_update_dt
    if dt is not None:
        if abs(dt - .020) > 1e-6:
            raise RuntimeError("SmoothRightEnvelope requires unchanged50Hz native commands")
        velocity = (np.asarray(result[0]) - previous) / dt
        rate = float(np.max(np.abs(velocity)))
        # Retain R05's2rad/s rejection, now on all replacement trajectory updates.
        if not np.isfinite(rate) or rate > PROFILE.command_rate_limit_rad_s + 1e-9:
            raise RuntimeError("SmoothRightEnvelope command-rate bound failed: " + str(rate))
        if policy.trace:
            policy.trace[-1]['arm_target_rate_rad_s'] = rate
            old_velocity = getattr(policy, '_previous_arm_velocity', None)
            if old_velocity is not None:
                policy.trace[-1]['arm_target_acceleration_rad_s2'] = float(np.max(np.abs(velocity - old_velocity)) / dt)
        policy._previous_arm_velocity = velocity
    check_corner_sweep(model, data, previous, np.asarray(result[0]), policy)
    return result


def terminal_reference_gate(original, policy, data):
    """Require commanded arrival AND the untouched actual readiness predicate.

    Always evaluate the previous gate first, even while the reference moves.
    Its real40ms continuous qualification therefore accrues normally; an old
    qualified sample is never cached as a substitute for the current predicate.
    The first press command cannot truncate the outgoing lowering polynomial.
    """
    if not callable(original):
        raise RuntimeError("Missing native coating motion-readiness predicate")

    @wraps(original)
    def ready():
        measured_ready = bool(original())
        if measured_ready and policy.first_measured_ready_physical_s is None:
            policy.first_measured_ready_physical_s = float(data.time)
        ready_now = measured_ready and policy.terminal_reference_ready
        if ready_now and policy.handshake_passed_physical_s is None:
            policy.handshake_passed_physical_s = float(data.time)
        return ready_now

    return ready


def _adapt(source):
    source = _exact(source, "    right_direct_coating_policy=RightDirectCoatingPolicyV6()",
                    "    right_direct_coating_policy=SmoothRightEntryPolicy(m,d,desired,qa,palm,tip)\n"
                    "    globals()['workflow_base_ready']=terminal_reference_gate(workflow_base_ready,right_direct_coating_policy,d)")
    source = _exact(source,
                    "                    if retract_at is None:retract_at=t;retract_anchor=current_tip.copy()\n"
                    "                    goal_tip=retract_anchor+np.array([0,0,.10*smooth((t-retract_at)/2)])",
                    "                    goal_tip=right_direct_coating_policy.retract_target(t,current_tip)")
    source = _exact(source, 'shift*=min(1,.0025/max(np.linalg.norm(shift),1e-12))',
                    'shift*=min(1,coarse_lookahead(t,right_direct_coating_policy,current_tip,target,phase)/max(np.linalg.norm(shift),1e-12))')
    source = _exact(source, 'q,error=ik(m,d,hand_goal,quat,seed=desired[qa],iterations=35)',
                    'q,error=checked_ik(ik,m,d,hand_goal,quat,seed=desired[qa],iterations=35,policy=right_direct_coating_policy)')
    arrays = "    arrays={k:np.array([r[k] for r in records]) for k in records[0]}"
    evidence = (
        "    (out/'smooth_right_precision_policy.json').write_text(json.dumps(dict("
        "profile=asdict(SMOOTH_RIGHT_PROFILE),time_basis='stage_logical_seconds',"
        "controller_clock_offset_s=workflow_clock_offset_s,original_thresholds_retained=True,"
        "reference_audit=right_direct_coating_policy.reference_audit,"
        "reference_terminal_at_s=right_direct_coating_policy.reference_terminal_at_s,"
        "first_measured_ready_physical_s=right_direct_coating_policy.first_measured_ready_physical_s,"
        "handshake_passed_physical_s=right_direct_coating_policy.handshake_passed_physical_s,"
        "retract_parts=right_direct_coating_policy.retract.parts,"
        "retract_split_at_s=right_direct_coating_policy.retract_split_at,"
        "retract_duration_s=right_direct_coating_policy.retract.duration,"
        "trace=right_direct_coating_policy.trace),indent=2))\n")
    source = _exact(source, arrays, evidence + arrays)
    return ("from dataclasses import asdict\n"
            "from v34_smoothrightenvelope import SmoothRightEntryPolicy,coarse_lookahead,checked_ik,terminal_reference_gate,PROFILE as SMOOTH_RIGHT_PROFILE\n"
            + source)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import t1_right_direct_coating_v6 as coating
    previous = coating.adapt_direct_coating_source

    @wraps(previous)
    def continuous_precision_source(source, config=None):
        return _adapt(previous(source, config))

    coating.adapt_direct_coating_source = continuous_precision_source
    _INSTALLED = True
