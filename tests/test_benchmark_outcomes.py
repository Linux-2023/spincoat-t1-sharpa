"""Numerical/boundary regressions, not a successful laboratory rollout.

Tests use real public liquid leaves and small native MuJoCo plants. No private
assets, historical episode, expert planner or mocked physical success is used.
The full fresh-world/C14 acceptance smoke belongs to the integration owner.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest

import mujoco
import numpy as np

from benchmark.backend import Backend, current_kinematics
from benchmark.bootstrap import case_inputs, station_regions
from benchmark.native_controller import ExecutionController, RotorMotor, rotor_target, velocity_motor_torque
from benchmark.native_sources import ROOT, PhysicalSources, robot_actuators
from benchmark.native_task import LiveTask, dispense_alignment, plain, withdrawal_height
from benchmark.physics_support.contract import CRITERIA, DT
from benchmark.physics_support.measurements import HeldLift, MixingMotion


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def native_backend(output, task_type=LiveTask):
    # A deliberately small unit plant, NOT the benchmark apparatus or a
    # replacement initializer. Names exercise the real private adapters.
    model = mujoco.MjModel.from_xml_string('''
        <mujoco>
          <option timestep=".001" gravity="0 0 0"/>
          <default><geom type="sphere" size=".02" mass=".1" contype="0" conaffinity="0"/></default>
          <worldbody>
            <body name="chassis_base"><freejoint name="chassis_free"/><geom mass="10"/>
              <body name="wheel" pos="0 0 .1"><joint name="wheel_RF_Joint"/><geom/></body>
              <body name="sharpa_left/left_hand_C_MC" pos=".3 .3 .6">
                <joint name="astribot_arm_left_joint_1"/><geom/>
                <body name="sharpa_left/left_index_link" pos=".03 0 0">
                  <joint name="sharpa_left/left_index_joint"/><geom/></body>
              </body>
              <body name="sharpa_right/right_hand_C_MC" pos=".3 -.3 .6"><geom/></body>
            </body>
            <body name="prep/reagent_tube" pos=".6 .3 .84">
              <freejoint name="prep/tube_free"/><geom/>
              <site name="prep/tube_mouth" pos="0 0 .15"/>
            </body>
            <body name="pipette/pipette" pos=".6 0 .95">
              <freejoint name="pipette/free"/><geom/>
              <site name="pipette/tip_site" pos="0 0 -.1"/>
              <body name="pipette/button_body" pos="0 0 .05">
                <joint name="pipette/pipette_button" type="slide" range="-.006 0"/>
                <geom/></body>
            </body>
            <body name="spincoater/base" pos="0 -.4 .8">
              <body name="spincoater/shaft">
                <joint name="spincoater/rotor" damping=".0001"/>
                <geom type="cylinder" size=".03 .002" mass=".01"/>
                <site name="spincoater/dispense_target" pos="0 0 .005"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <velocity name="astribot_chassis_motor_0" joint="wheel_RF_Joint" kv=".1" ctrlrange="-2 2"/>
            <position name="arm" joint="astribot_arm_left_joint_1" kp="1" ctrlrange="-1 1"/>
            <position name="finger" joint="sharpa_left/left_index_joint" kp="1" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    case = json.loads((ROOT / 'benchmark/cases/reference.json').read_text())
    layout, config = case_inputs(case)
    sources = PhysicalSources()
    handles = {'tube_body': 'prep/reagent_tube', 'tube_mouth_site': 'prep/tube_mouth',
               'rack_top_world_z_m': .8880005, 'config': config}
    b = Backend.__new__(Backend)
    b.owner_thread = threading.get_ident()
    b.closed = False
    b.tick = b.attempted_steps = 0
    b.integrity_error = b.last_applied = None
    b.last_events = []
    b.output = Path(output)
    b.model, b.data, b.handles, b.sources = model, data, handles, sources
    b.initial_controls = data.ctrl.copy()
    b.actuators = robot_actuators(model)
    b.joint_ids = np.array([model.joint(row['joint']).id for row in b.actuators], dtype=int)
    b.qpos_ids = model.jnt_qposadr[b.joint_ids].copy()
    b.dof_ids = model.jnt_dofadr[b.joint_ids].copy()
    b.base_body = model.body('chassis_base').id
    b.palms = {side: model.body(f'sharpa_{side}/{side}_hand_C_MC').id for side in ('left', 'right')}
    wheels = [row['index'] for row in b.actuators if row['semantics'] == 'joint_velocity']
    route = {'wheel_ids': wheels, 'route_gain': model.actuator_gainprm[wheels].tolist(),
             'route_bias': model.actuator_biasprm[wheels].tolist(), 'hold_gain_scale': 32., 'brake_duration_s': .4}
    b.controller = ExecutionController(model, data.ctrl, route)
    b.motor = RotorMotor(model, data)
    b.task = task_type(sources, config, handles, model, data, stations=station_regions(layout))
    b.measurement_data = mujoco.MjData(model)
    b.proprio_data = mujoco.MjData(model)
    b.source_identity = 'unit-plant-not-laboratory-acceptance'
    return b


class HeldLiftTests(unittest.TestCase):
    def test_requires_500_supported_samples_and_actual_lift(self):
        gate = HeldLift(0.)
        counts, forces = np.ones(5), np.full(5, .02)
        for i in range(1, 500):
            self.assertFalse(gate.observe(i*DT, .06, counts, forces))
        self.assertTrue(gate.observe(.5, .06, counts, forces))
        self.assertTrue(gate.acquired)
        short = HeldLift(0.)
        for i in range(1, 601):
            short.observe(i*DT, .059999, counts, forces)
        self.assertFalse(short.acquired)

    def test_force_boundary_and_missing_sample(self):
        gate = HeldLift(0.)
        counts, forces = np.ones(5), np.full(5, .02)
        for i in range(1, 500):
            gate.observe(i*DT, .061, counts, forces)
        forces[3] = .01
        self.assertFalse(gate.observe(.5, .061, counts, forces))
        self.assertEqual(gate.support_samples, 0)
        with self.assertRaises(ValueError):
            gate.observe(.502, .061, counts, forces)
        with self.assertRaises(ValueError):
            HeldLift(float('nan'))


class MixingTests(unittest.TestCase):
    def observe_motion(self, amplitude, frequency=3., translated=False, interrupted=False):
        gate = MixingMotion()
        for i in range(4001):
            t = i*DT
            mouth = np.array([.1*t if translated else 0., 0., 1.])
            eligible = not (interrupted and i % 1000 == 0)
            gate.observe(t, mouth, rotation_x(np.deg2rad(amplitude)*np.sin(2*np.pi*frequency*t)), eligible)
        return gate

    def test_repeated_supported_motion_without_homogeneity_claim(self):
        gate = self.observe_motion(6.)
        self.assertTrue(gate.acquired)
        self.assertFalse(gate.status()['material_homogeneity_verified'])

    def test_static_jitter_and_translated_carry_do_not_pass(self):
        for amplitude, frequency, translated in ((0., 3., False), (.2, 30., False), (6., 3., True)):
            with self.subTest(amplitude=amplitude, translated=translated):
                self.assertFalse(self.observe_motion(amplitude, frequency, translated).acquired)
        self.assertFalse(self.observe_motion(6., interrupted=True).acquired)

    def test_nonfinite_or_repeated_clock_rejected(self):
        gate = MixingMotion()
        gate.observe(0., [0., 0., 0.], np.eye(3), True)
        with self.assertRaises(ValueError):
            gate.observe(0., [0., 0., 0.], np.eye(3), True)
        with self.assertRaises(ValueError):
            gate.observe(.001, [float('nan'), 0., 0.], np.eye(3), True)


class GeometryTests(unittest.TestCase):
    def test_shifted_right_dock_and_unknown_cases(self):
        case = json.loads((ROOT / 'benchmark/cases/reference.json').read_text())
        layout, config = case_inputs(case)
        stations = station_regions(layout)
        stand = np.array(layout['main_asset_targets_world_xy_m']['pipette_stand_tool'])
        expected = np.array(layout['right_runtime_config']['grasp_base_xy_m']) + stand - [.68, 0.]
        np.testing.assert_allclose(stations['grasp'], expected, atol=1e-15)
        self.assertGreater(np.linalg.norm(np.array(stations['grasp']) - layout['right_runtime_config']['grasp_base_xy_m']), .001)
        case['layout_perturbations'] = {'stand_dx_m': .001}
        with self.assertRaises(ValueError):
            case_inputs(case)
        self.assertAlmostEqual(withdrawal_height(config, .82), config['tabletop_z_m']+config['beaker_height_m']+.034)
        self.assertAlmostEqual(withdrawal_height(config, 1.0), 1.005)

    def test_public_standoff_and_strict_speed_boundary(self):
        target = np.zeros(3)
        tip = [0., 0., .012]
        self.assertTrue(dispense_alignment(tip, [0., 0., 0.], np.eye(3), target)[3])
        self.assertFalse(dispense_alignment([.000301, 0., .012], [0., 0., 0.], np.eye(3), target)[3])
        self.assertFalse(dispense_alignment(tip, [.0075, 0., 0.], np.eye(3), target)[3])
        self.assertFalse(dispense_alignment(tip, [0., 0., 0.], rotation_x(.00501), target)[3])
        with self.assertRaises(FloatingPointError):
            dispense_alignment([np.nan, 0., .012], [0., 0., 0.], np.eye(3), target)


class MotorTests(unittest.TestCase):
    def test_program_and_bounded_bidirectional_torque(self):
        self.assertEqual(rotor_target(None), 0.)
        self.assertAlmostEqual(rotor_target(.25), 10*np.pi)
        self.assertAlmostEqual(rotor_target(.5), 20*np.pi)
        self.assertEqual(rotor_target(6.2), 0.)
        self.assertEqual(velocity_motor_torque(1e6, 0., 1., 0., .001, .1), .1)
        self.assertEqual(velocity_motor_torque(0., 1e6, 1., 0., .001, .1), -.1)
        for bad in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                velocity_motor_torque(1., 0., bad, 0., .001, .1)

    def test_motor_is_force_input_not_velocity_override(self):
        with tempfile.TemporaryDirectory() as output:
            b = native_backend(output)
            try:
                b.data.time = .5  # Test reset, before any stepping.
                before_q, before_v, before_time = b.data.qpos.copy(), b.data.qvel.copy(), b.data.time
                b.data.qfrc_applied[:] = 0.
                b.motor.apply(b.model, b.data, 0.)
                np.testing.assert_array_equal(b.data.qpos, before_q)
                np.testing.assert_array_equal(b.data.qvel, before_v)
                self.assertEqual(b.data.time, before_time)
                self.assertGreater(b.motor.last['torque_Nm'], 0.)
                self.assertLessEqual(abs(b.motor.last['torque_Nm']), b.motor.limit)
                mujoco.mj_step(b.model, b.data)
                self.assertGreater(b.data.qvel[b.motor.dof], 0.)
                self.assertLess(b.data.qvel[b.motor.dof], rotor_target(.5))
                self.assertFalse(b.motor.specification()['native_physics_equivalent_to_prescribed_velocity'])
            finally:
                b.close()


class LiquidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = PhysicalSources()

    def test_actual_stroke_conserves_volume_and_air_cannot_refill(self):
        m = self.sources.aspiration
        stage = m.AspirationStage(m.SourceGeometry((0., 0.), .02, .8, 1e-6, .82))
        t = 0.
        def hold(q, z, count, allowed=False):
            nonlocal t
            for _ in range(count):
                t += DT
                stage.update(t, q, [0., 0., z], allow_dispense=allowed)
        hold(-.0051, .84, 100)
        hold(-.0051, .815, 120)
        for i in range(101):
            hold(-.005*(1-i/100), .815, 1)
        hold(0., .815, 180)
        hold(0., .84, 120)
        self.assertEqual(stage.state, m.Stage.READY)
        self.assertAlmostEqual(stage.remaining_volume_m3, 30e-9, delta=1e-18)
        hold(-.0051, .84, 1, allowed=True)
        self.assertAlmostEqual(stage.delivered_m3, 30e-9, delta=1e-18)
        hold(0., .84, 1)
        hold(-.0051, .84, 1, allowed=True)
        self.assertEqual(stage.remaining_volume_m3, 0.)
        self.assertAlmostEqual(stage.delivered_m3, 30e-9, delta=1e-18)
        self.assertAlmostEqual(stage.audit()['mass_balance_error_m3'], 0., delta=1e-18)

    def test_wrong_order_never_creates_dose(self):
        m = self.sources.aspiration
        stage = m.AspirationStage(m.SourceGeometry((0., 0.), .02, .8, 1e-6, .82))
        stage.update(DT, 0., [0., 0., .815])
        self.assertEqual(stage.failure, 'immersion_before_prepress')
        self.assertEqual(stage.remaining_volume_m3, 0.)
        self.assertEqual(stage.delivered_m3, 0.)


class BackendBoundaryTests(unittest.TestCase):
    def test_one_native_tick_and_robot_only_current_fk(self):
        with tempfile.TemporaryDirectory() as output:
            b = native_backend(output)
            try:
                ctrl = b.initial_controls.copy()
                ctrl[1] = .2
                b.step(ctrl)
                self.assertEqual(b.tick, 1)
                self.assertAlmostEqual(b.data.time, DT)
                cached = b.data.xmat.copy()
                observation = b.proprioception()
                work = current_kinematics(b.model, b.data, mujoco.MjData(b.model))
                expected = work.xmat[b.base_body].reshape(3,3).T @ work.xmat[b.palms['left']].reshape(3,3)
                np.testing.assert_allclose(observation['eef']['left']['rotation'], expected, atol=1e-15)
                np.testing.assert_array_equal(b.data.xmat, cached)
                self.assertEqual(set(observation), {'tick','t_ns','joint_names','joint_qpos','joint_qvel','eef_frame','eef'})
                self.assertIsNone(b.outcome()['outcome']['strict_physics'])
                self.assertIsNone(b.outcome()['outcome']['hardware_certified'])
                self.assertFalse(b.outcome()['outcome']['functional'])
            finally:
                b.close()
                b.close()

    def test_invalid_request_does_not_advance(self):
        with tempfile.TemporaryDirectory() as output:
            b = native_backend(output)
            try:
                with self.assertRaises(ValueError):
                    b.step(np.full(b.model.nu, np.nan))
                self.assertEqual(b.tick, 0)
                self.assertEqual(b.data.time, 0.)
                self.assertIsNone(b.integrity_error)
            finally:
                b.close()

    def test_poststep_failure_preserves_attempt_and_raw_state(self):
        class FailedDiagnostic(LiveTask):
            def advance(self, model, data):
                super().advance(model, data)
                raise FloatingPointError('injected post-step diagnostic failure')
        with tempfile.TemporaryDirectory() as output:
            b = native_backend(output, FailedDiagnostic)
            try:
                with self.assertRaises(FloatingPointError):
                    b.step(b.initial_controls)
                self.assertEqual(b.tick, 1)
                self.assertEqual(b.attempted_steps, 1)
                self.assertAlmostEqual(b.data.time, DT)
                self.assertFalse(b.outcome()['infrastructure_valid'])
                self.assertFalse(b.outcome()['outcome']['functional'])
                with np.load(Path(output)/'invalid_transition.npz') as evidence:
                    np.testing.assert_array_equal(evidence['qpos'], b.data.qpos)
                    self.assertEqual(int(evidence['tick']), 1)
                with self.assertRaises(RuntimeError):
                    b.step(b.initial_controls)
            finally:
                b.close()

    def test_renderer_payload_is_detached_and_read_only(self):
        with tempfile.TemporaryDirectory() as output:
            b = native_backend(output)
            try:
                before = json.dumps(plain(b.task.snapshot()), sort_keys=True)
                payload = b.task.render_state(b.data)
                payload['fluid']['h'][:] = 1.
                payload['liquid']['config']['tube_initial_volume_m3'] = 1.
                self.assertEqual(before, json.dumps(plain(b.task.snapshot()), sort_keys=True))
                with self.assertRaises(FloatingPointError):
                    plain({'broken': np.array([np.nan])})
            finally:
                b.close()


if __name__ == '__main__':
    unittest.main()
