"""Policy-free extraction of the existing encoder and wheel-brake leaves.

No object pose, tactile preload, station approach, or inverse kinematics enters
this controller. Normal all-zero wheel requests select the original physical
brake/hold schedule; nonzero requests select original route gains. Safety holds
retain acquired non-wheel commands, rather than unloading fingers to qpos.
"""
from __future__ import annotations

import numpy as np

from .physics_support.contract import CONTRACT, CRITERIA

EXECUTION = CONTRACT['execution_controller']
ROTOR = CRITERIA['rotor']


def smooth(value):
    u = float(np.clip(value, 0., 1.))
    return u*u*u*(10.+u*(-15.+6.*u))


class ExecutionController:
    def __init__(self, model, initial_ctrl, route, *, encoder_active=True,
                 encoder_integrator_frozen=False):
        self.wheels = np.asarray(route["wheel_ids"], dtype=np.int64)
        actual = [a for a in range(model.nu)
                  if model.actuator(a).name.startswith("astribot_chassis_motor_")]
        if actual != self.wheels.tolist():
            raise ValueError("Original wheel registry mismatch")
        self.route_gain = np.asarray(route["route_gain"], dtype=np.float64).copy()
        self.route_bias = np.asarray(route["route_bias"], dtype=np.float64).copy()
        self.hold_gain_scale = float(route["hold_gain_scale"])
        self.brake_duration_s = float(route["brake_duration_s"])
        self.left = np.asarray([a for a in range(model.nu)
                               if model.joint(int(model.actuator_trnid[a, 0])).name.startswith("sharpa_left/")], dtype=np.int64)
        self.left_q = model.jnt_qposadr[model.actuator_trnid[self.left, 0]].copy()
        self.mask = np.asarray([any("/left_"+finger+"_" in
                                   model.joint(int(model.actuator_trnid[a, 0])).name
                                   for finger in ("thumb", "index", "middle"))
                                for a in self.left], dtype=bool)
        self.bias_dofs = np.asarray(sorted({int(model.jnt_dofadr[int(model.actuator_trnid[a, 0])])
                                           for a in range(model.nu) if a not in self.wheels and
                                           model.joint(int(model.actuator_trnid[a, 0])).name.startswith(("astribot_", "sharpa_"))}), dtype=np.int64)
        self.offset = np.zeros(len(self.left), dtype=np.float64)
        self.encoder_active = bool(encoder_active)
        self.integrator_frozen = bool(encoder_integrator_frozen)
        self.lasttime = None
        self.last_nominal = np.asarray(initial_ctrl, dtype=np.float64).copy()
        self.last_final = self.last_nominal.copy()
        self.held = self.last_final.copy()
        self.safety_active = False
        self.mode = "route"
        self.brake_start = None
        self.brake_from = np.zeros(len(self.wheels), dtype=np.float64)
        self.gain_scale = 1.

    def _wheel_control(self, model, data, nominal, stopping, events):
        old_mode, old_scale = self.mode, self.gain_scale
        if stopping:
            if self.mode == "route":
                self.brake_start = float(data.time)
                self.brake_from = self.last_final[self.wheels].copy()
                self.mode = "braking"
            blend = smooth((float(data.time)-self.brake_start)/self.brake_duration_s)
            self.gain_scale = 1.+(self.hold_gain_scale-1.)*blend
            data.ctrl[self.wheels] = (1.-blend)*self.brake_from
            if float(data.time)-self.brake_start >= self.brake_duration_s:
                self.mode = "zero_speed_hold"
                data.ctrl[self.wheels] = 0.
        else:
            self.mode, self.gain_scale = "route", 1.
            self.brake_start = None
            data.ctrl[self.wheels] = nominal[self.wheels]
        if self.gain_scale != old_scale:
            before_gain = model.actuator_gainprm[self.wheels].copy()
            before_bias = model.actuator_biasprm[self.wheels].copy()
            model.actuator_gainprm[self.wheels] = self.route_gain
            model.actuator_biasprm[self.wheels] = self.route_bias
            model.actuator_gainprm[self.wheels, 0] = self.route_gain[:, 0]*self.gain_scale
            model.actuator_biasprm[self.wheels, 2] = self.route_bias[:, 2]*self.gain_scale
            events.append({"phase": "pre_step", "event_type": "environment_command", "payload": {
                "source": "wheel_velocity_servo", "operation": "gain_bias_schedule",
                "actuator_indices": self.wheels.tolist(), "gain_scale": self.gain_scale,
                "before_gainprm": before_gain.tolist(), "before_biasprm": before_bias.tolist(),
                "gainprm": model.actuator_gainprm[self.wheels].tolist(),
                "biasprm": model.actuator_biasprm[self.wheels].tolist(),
                "force_caps_unchanged": True}})
        if old_mode != self.mode:
            events.append({"phase": "pre_control", "event_type": "controller_mode", "payload": {
                "source": "wheel_velocity_servo", "previous": old_mode, "mode": self.mode,
                "reason": "rejected_request_hold" if self.safety_active else "native_wheel_request"}})

    def apply(self, model, data, requested, reject_reason):
        events = []
        safety = reject_reason is not None
        now = float(data.time)
        dt = 0. if self.lasttime is None else max(0., min(.002, now-self.lasttime))
        self.lasttime = now
        if safety:
            if not self.safety_active:
                self.held = self.last_final.copy()
                events.append({"phase": "pre_control", "event_type": "controller_mode", "payload": {
                    "source": "grip_preserving_safety_hold", "mode": "acquired_command_hold",
                    "reason": reject_reason, "encoder_integral_frozen_during_hold": True}})
            self.safety_active = True
            data.ctrl[:] = self.held
            nominal = self.held
            # Do not add the encoder offset to a command that already includes it.
        else:
            if self.safety_active:
                events.append({"phase": "pre_control", "event_type": "controller_mode", "payload": {
                    "source": "grip_preserving_safety_hold", "mode": "teacher_control_resumed"}})
            self.safety_active = False
            nominal = np.asarray(requested, dtype=np.float64)
            self.last_nominal = nominal.copy()
            data.ctrl[:] = nominal
            if self.encoder_active:
                if not self.integrator_frozen:
                    error = nominal[self.left]-data.qpos[self.left_q]
                    bound = EXECUTION['encoder_bound_rad']
                    self.offset[self.mask] = np.clip(self.offset[self.mask]+EXECUTION['encoder_integral_per_s']*dt*error[self.mask], -bound, bound)
            else:
                self.offset *= np.exp(-dt/.15)
            data.ctrl[self.left] = np.clip(nominal[self.left]+self.offset,
                                          model.actuator_ctrlrange[self.left, 0],
                                          model.actuator_ctrlrange[self.left, 1])
        self._wheel_control(model, data, nominal, safety or bool(np.all(nominal[self.wheels] == 0.)), events)
        data.qfrc_applied[:] = 0.
        data.qfrc_applied[self.bias_dofs] = data.qfrc_bias[self.bias_dofs]
        # No controller writes xfrc_applied; preserve actual external inputs.
        self.last_final = data.ctrl.copy()
        transform = None
        if not safety:
            changed = np.flatnonzero(data.ctrl != nominal)
            if len(changed):
                transform = {"indices": changed.tolist(), "requested": nominal[changed].tolist(),
                             "applied": data.ctrl[changed].tolist(),
                             "reason": "bounded_left_encoder_tracking_and_native_zero_wheel_brake"}
        return transform, events


CONTROLLER_ID = EXECUTION['id']


MOTOR_ID = ROTOR['id']


def rotor_target(age):
    """Original device program, now a velocity reference rather than a write."""
    if age is None:
        return 0.
    if not np.isfinite(age):
        raise ValueError("Nonfinite rotor program age")
    return float(ROTOR['target_rpm'] * np.pi / 30. * min(1., max(0., age) / ROTOR['ramp_s']) *
                 (1. - smooth((age - ROTOR['deceleration_at_s']) / ROTOR['deceleration_duration_s'])))


def velocity_motor_torque(target, measured, inertia, damping, dt, limit, response_s=ROTOR['response_s']):
    """Bounded generalized torque; no state projection or integral windup.

    With an isolated shaft, explicit feedback has a pole in (0,1). Existing
    implicit native damping further reduces the one-step gain. Native viscous
    drag is compensated within the same torque cap, never deleted from model.
    """
    values = (target, measured, inertia, damping, dt, limit, response_s)
    if not np.isfinite(values).all() or min(inertia, dt, limit, response_s) <= 0 or damping < 0:
        raise ValueError("Invalid physical motor input")
    requested = inertia * (target - measured) / (response_s + dt) + damping * measured
    return float(np.clip(requested, -limit, limit))


class RotorMotor:
    """Declared new shaft motor; robot/model physics arrays remain unchanged."""
    def __init__(self, model, data):
        import mujoco
        joint = model.joint("spincoater/rotor")
        self.dof = int(joint.dofadr[0])
        self.mass_address = int(model.dof_Madr[self.dof])
        if int(model.jnt_type[joint.id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Rotor motor requires the original hinge")
        if int(model.dof_parentid[self.dof]) != -1:
            raise ValueError("Rotor motor requires the original fixed-parent isolated shaft")
        self.inertia = float(data.qM[self.mass_address])
        self.damping = float(model.dof_damping[self.dof])
        self.frictionloss = float(model.dof_frictionloss[self.dof])
        self.dt = float(model.opt.timestep)
        self.response_s = ROTOR['response_s']
        # A declared engineering sizing law: 2x no-load ramp inertial + native
        # rated-speed viscous + dry friction demand, frozen at this reset.
        target = ROTOR['target_rpm'] * np.pi / 30.
        self.limit = ROTOR['torque_headroom'] * (self.inertia * target / ROTOR['ramp_s'] +
                                               self.damping * target + self.frictionloss)
        velocity_motor_torque(0., 0., self.inertia, self.damping, self.dt, self.limit)
        self.last = None

    def specification(self):
        return {"id": MOTOR_ID, "joint": "spincoater/rotor", "dof": self.dof,
                "input": "native qfrc_applied generalized torque N*m",
                "inertia_kg_m2": self.inertia, "native_damping_Nm_s_rad": self.damping,
                "native_frictionloss_Nm": self.frictionloss,
                "response_s": self.response_s, "torque_limit_Nm": self.limit,
                "limit_law": "2*(I_reset*20*pi/0.5 + damping*20*pi + frictionloss)",
                "target_rpm": ROTOR['target_rpm'], "ramp_s": ROTOR['ramp_s'],
                "deceleration_at_s": ROTOR['deceleration_at_s'],
                "deceleration_duration_s": ROTOR['deceleration_duration_s'],
                "native_physics_equivalent_to_prescribed_velocity": False}

    def apply(self, model, data, motor_at):
        target = rotor_target(None if motor_at is None else float(data.time) - motor_at)
        measured = float(data.qvel[self.dof])
        inertia = float(data.qM[self.mass_address])
        torque = velocity_motor_torque(target, measured, inertia, self.damping,
                                       self.dt, self.limit, self.response_s)
        before = float(data.qfrc_applied[self.dof])
        data.qfrc_applied[self.dof] += torque
        self.last = {"id": MOTOR_ID, "target_rad_s": target, "torque_Nm": torque,
                     "measured_rad_s": measured, "inertia_kg_m2": inertia,
                     "torque_limit_Nm": self.limit, "motor_started_at_s": motor_at,
                     "dof": self.dof, "before_qfrc_Nm": before,
                     "actual_qfrc_Nm": float(data.qfrc_applied[self.dof])}
        return {"phase": "pre_step", "event_type": "environment_command",
                "payload": {"source": MOTOR_ID, "operation": "bounded_shaft_torque", **self.last}}
