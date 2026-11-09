"""Evaluator-owned native world: requested controls -> plant -> passive judge.

There is deliberately no expert, IK planner, detector, stage-action, history
restore or runtime pose/velocity/time-write interface in this backend.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import threading

import mujoco
import numpy as np

from .bootstrap import initialize
from .native_controller import CONTROLLER_ID, ExecutionController, RotorMotor
from .native_sources import ROOT, PhysicalSources, file_hash, identity, robot_actuators
from .native_task import TASK_ID, LiveTask, plain
from .recording import save_json


def current_kinematics(model, data, work):
    """Private FK refresh, retaining solver contacts and every live data byte."""
    mujoco.mj_copyData(work, model, data)
    mujoco.mj_kinematics(model, work)
    mujoco.mj_comPos(model, work)
    mujoco.mj_comVel(model, work)
    return work


class Backend:
    dt_ns = 1_000_000

    def __init__(self, assets: Path, output: Path, case: dict):
        self.owner_thread = threading.get_ident()
        self.closed = False
        self.tick = self.attempted_steps = 0
        self.integrity_error = None
        self.last_applied = None
        self.last_events = []
        self.output = Path(output).resolve()
        reset = initialize(assets, self.output, case)
        self.model, self.data, self.handles = reset.model, reset.data, reset.handles
        self.workspace, self.model_path = reset.workspace, reset.model_path
        self.model_sha256 = reset.provenance['model_sha256']
        self.sources = PhysicalSources(self.workspace)
        self.initial_controls = self.data.ctrl.copy()
        self.actuators = robot_actuators(self.model)
        self.joint_ids = np.asarray([int(self.model.joint(row['joint']).id) for row in self.actuators])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids].copy()
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids].copy()
        self.base_body = int(self.model.body('chassis_base').id)
        self.palms = {side: int(self.model.body(f'sharpa_{side}/{side}_hand_C_MC').id)
                      for side in ('left', 'right')}
        self.controller = ExecutionController(self.model, self.initial_controls, reset.route)
        self.motor = RotorMotor(self.model, self.data)
        self.task = LiveTask(self.sources, reset.config, self.handles, self.model, self.data, stations=reset.stations)
        self.measurement_data = mujoco.MjData(self.model)
        self.proprio_data = mujoco.MjData(self.model)
        self._validate(self.initial_controls)
        source_paths = [Path(__file__), Path(__file__).with_name('bootstrap.py'),
                        Path(__file__).with_name('native_controller.py'), Path(__file__).with_name('native_task.py'),
                        Path(__file__).with_name('native_sources.py'), Path(__file__).with_name('task_contract.json'),
                        Path(__file__).parent / 'physics_support/measurements.py',
                        Path(__file__).parent / 'physics_support/contract.py',
                        Path(__file__).parent / 'physics_support/source_provenance.json',
                        Path(__file__).parent / 'cases/reference.json']
        manifest = {'schema_version': 1, 'task_id': TASK_ID, 'controller_id': CONTROLLER_ID,
                    'case_identity': identity(case), 'model_sha256': self.model_sha256,
                    'initialization_manifest_sha256': file_hash(self.output / 'initialization_manifest.json'),
                    'source_files': {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
                    'physical_leaves': self.sources.hashes, 'apparatus': self.motor.specification(),
                    'controller': {'encoder_integral_per_s': 3., 'encoder_bound_rad': .03,
                                   'encoder_active': True, 'phase_gating_removed': True,
                                   'wheel_hold_scale': 32., 'wheel_brake_s': .4,
                                   'bias_compensation_dofs': self.controller.bias_dofs.tolist(),
                                   'route': reset.route},
                    'measurement_clock': 'Endpoint qpos/qvel with private current FK; native contact forces from the just-completed solver step (up to 1ms contact-geometry lag).',
                    'runtime_state_overrides': False, 'expert_executed': False}
        self.source_identity = identity(manifest)
        manifest['source_identity'] = self.source_identity
        save_json(self.output / 'backend_manifest.json', manifest)
        save_json(self.output / 'control_schema.json', self.control_schema())

    def _owner(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('Backend requires its constructing physics owner thread')
        if self.closed:
            raise RuntimeError('Backend is closed')

    def _validate(self, ctrl):
        value = np.asarray(ctrl, dtype=np.float64)
        if value.shape != (self.model.nu,) or not np.isfinite(value).all():
            raise ValueError('Expected one finite complete native 68-control vector')
        limited = self.model.actuator_ctrllimited.astype(bool)
        if (np.any(value[limited] < self.model.actuator_ctrlrange[limited, 0]) or
                np.any(value[limited] > self.model.actuator_ctrlrange[limited, 1])):
            raise ValueError('Requested controls exceed original actuator control range')
        return value

    def control_schema(self):
        return {'controller_id': CONTROLLER_ID, 'actuators': copy.deepcopy(self.actuators),
                'initial_controls': self.initial_controls.tolist(),
                'proprioception': {'joint_order': [row['joint'] for row in self.actuators],
                    'joint_qpos': 'actuated robot joints only; rad (hinge) or m (slide)',
                    'joint_qvel': 'same order; rad/s or m/s',
                    'eef': {'left': 'sharpa_left/left_hand_C_MC', 'right': 'sharpa_right/right_hand_C_MC'},
                    'eef_frame': 'chassis_base', 'eef_rotation': '3x3 row-major; base_R_eef',
                    'excluded': ['touch', 'contact', 'object state', 'liquid state', 'task metrics', 'world base pose']},
                'execution_transform': {'left_thumb_index_middle': 'bounded encoder integral: 3*dt*error, +/-0.03 rad; native clipping',
                    'all_zero_wheels': '0.4s physical brake followed by zero-speed hold, gain scale 32; no force-cap change',
                    'nonzero_wheels': 'original route servo; no route planner',
                    'robot_bias_compensation': 'non-wheel actuated robot DOFs only'},
                'pipette_button': 'Passive spring joint driven only by real robot contact; not an action actuator'}

    def step(self, ctrl):
        self._owner()
        if self.task.terminal or self.integrity_error is not None:
            raise RuntimeError('Cannot advance a terminal or infrastructure-invalid world')
        requested = self._validate(ctrl).copy()
        events = []
        self.last_events = events
        self.last_applied = None
        before = float(self.data.time)
        called = False
        try:
            transform, controller_events = self.controller.apply(self.model, self.data, requested, None)
            events.extend(controller_events)
            if transform is not None:
                events.append({'phase': 'pre_step', 'event_type': 'control_transform', 'payload': transform})
            events.append(self.motor.apply(self.model, self.data, self.task.motor_at))
            self.last_applied = {name: getattr(self.data, name).copy()
                                 for name in ('ctrl', 'qfrc_applied', 'xfrc_applied')}
            if any(not np.isfinite(value).all() for value in self.last_applied.values()):
                raise FloatingPointError('Nonfinite actual native force/control input')
            called = True
            self.attempted_steps += 1
            mujoco.mj_step(self.model, self.data)
            # Count the native call BEFORE any post-step evaluator can raise.
            self.tick += 1
            if (not np.isfinite(self.data.time) or abs(float(self.data.time)-before-.001) > 1e-10 or
                    any(not np.isfinite(getattr(self.data, name)).all() for name in
                        ('qpos', 'qvel', 'act', 'qacc', 'qacc_warmstart'))):
                raise FloatingPointError('Native step reset, failed or became nonfinite')
            measured = current_kinematics(self.model, self.data, self.measurement_data)
            self.task.motor_diagnostics = dict(self.motor.last, measured_after_rad_s=float(self.data.qvel[self.motor.dof]))
            task_events, diagnostics = self.task.advance(self.model, measured)
            events.extend(task_events)
            if self.task.tick != self.tick:
                raise RuntimeError('Task and physical tick diverged')
            for name, value in self.last_applied.items():
                if not np.array_equal(getattr(self.data, name), value):
                    raise RuntimeError('Actual step inputs changed after native integration: ' + name)
            plain(events)
            return events
        except BaseException as error:
            if called and self.tick < self.attempted_steps:
                # A raised native call is quarantined as attempted, never
                # silently described as zero motion or replayed.
                self.tick = self.attempted_steps
            self.integrity_error = {'type': type(error).__name__, 'message': str(error),
                                    'tick': self.tick, 'native_called': called,
                                    'before_time_s': before,
                                    'after_time_s': float(self.data.time) if np.isfinite(self.data.time) else None}
            self.task.tick = self.tick
            self.task.invalidate(error)
            # Preserve raw nonfinite arrays losslessly for forensic review;
            # never coerce a broken transition into a finite successful one.
            payload = {name: getattr(self.data, name).copy() for name in
                       ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'qacc', 'qacc_warmstart')}
            payload.update(time=np.array(self.data.time), requested=requested, tick=np.array(self.tick))
            if self.last_applied is not None:
                payload.update({'applied_' + key: value for key, value in self.last_applied.items()})
            with (self.output / 'invalid_transition.npz').open('xb') as stream:
                np.savez_compressed(stream, **payload)
            save_json(self.output / 'integrity_error.json', self.integrity_error)
            raise

    def proprioception(self):
        self._owner()
        if self.integrity_error is not None:
            raise RuntimeError('Invalid physical state cannot supply an observation')
        work = current_kinematics(self.model, self.data, self.proprio_data)
        base_R = work.xmat[self.base_body].reshape(3, 3)
        eef = {side: {'position_m': (base_R.T @ (work.xpos[body] - work.xpos[self.base_body])).tolist(),
                      'rotation': (base_R.T @ work.xmat[body].reshape(3, 3)).tolist()}
               for side, body in self.palms.items()}
        return plain({'tick': self.tick, 't_ns': self.tick*self.dt_ns,
                      'joint_names': [row['joint'] for row in self.actuators],
                      'joint_qpos': self.data.qpos[self.qpos_ids],
                      'joint_qvel': self.data.qvel[self.dof_ids], 'eef_frame': 'chassis_base', 'eef': eef})

    def outcome(self):
        report = self.task.status()
        report['source_identity'] = self.source_identity
        report['physics_steps'] = self.tick
        report['native_attempted_steps'] = self.attempted_steps
        report['infrastructure_valid'] = self.integrity_error is None
        if self.integrity_error is not None:
            report['outcome']['functional'] = False
            report['outcome']['configured_contact_checks'] = False
            report['integrity_error'] = copy.deepcopy(self.integrity_error)
        return plain(report)

    def close(self):
        if self.closed:
            return
        self._owner()
        self.closed = True
        self.measurement_data = self.proprio_data = None
        self.data = self.model = None
