"""Consumer-visible action/recording invariants using real native MuJoCo steps."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from benchmark.recording import file_sha256, save_json
from benchmark.session import ProtocolError, Session


class SmallPhysicalBackend:
    """Small deterministic physical fixture, not a replacement benchmark world."""
    dt_ns = 1_000_000

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string('''
        <mujoco><option timestep="0.001" gravity="0 0 0"/>
        <worldbody><body><joint name="arm" type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.1" mass="1"/></body>
        <body pos="0 1 0"><joint name="finger" type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.1" mass="1"/></body></worldbody>
        <actuator><motor name="arm" joint="arm" ctrllimited="true" ctrlrange="-1 1"/>
        <motor name="finger" joint="finger" ctrllimited="true" ctrlrange="0 1"/>
        </actuator></mujoco>''')
        self.data = mujoco.MjData(self.model)
        self.tick = 0
        self.task = SimpleNamespace(terminal=False)

    def control_schema(self):
        return {"actuators": [
            {"name": "arm", "index": 0, "limited": True, "range": [-1, 1]},
            {"name": "finger", "index": 1, "limited": True, "range": [0, 1]}],
            "initial_controls": [0., 0.]}

    def step(self, control):
        self.data.ctrl[:] = control
        mujoco.mj_step(self.model, self.data)
        self.tick += 1
        return []

    def outcome(self):
        return {"outcome": {"functional": False}, "reason": "fixture_has_no_laboratory_task"}


def fixture_identity(directory):
    directory = Path(directory)
    save_json(directory / "case.json", {"fixture": "two native MuJoCo sliders"})
    save_json(directory / "source_manifest.json", {
        "schema_version": 1,
        "files": {"test_benchmark_protocol.py": file_sha256(__file__)},
        "runtime_manifest_sha256": file_sha256(Path(__file__).parents[1] / "runtime_manifest.json"),
        "case_sha256": file_sha256(directory / "case.json")})


class ActionContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.backend = SmallPhysicalBackend()
        self.time = [0.]
        fixture_identity(self.directory.name)
        self.session = Session(self.backend, None, Path(self.directory.name), {
            "id": "physical-fixture",
            "budgets": {"max_physics_steps": 15, "max_wall_seconds": 60}},
            clock=lambda: self.time[0])

    def tearDown(self):
        self.session.finish()
        self.session.recorder.close()
        self.directory.cleanup()

    def request(self, ident, controls, steps):
        return {"request_id": ident, "actions": [{"controls": controls, "steps": steps}]}

    def assert_stationary(self, tick, qpos):
        self.assertEqual(self.backend.tick, tick)
        np.testing.assert_array_equal(self.backend.data.qpos, qpos)

    def test_entire_batch_validated_before_first_physics_step(self):
        qpos = self.backend.data.qpos.copy()
        payload = {"request_id": "bad-later-row", "actions": [
            {"controls": {"arm": 0.5}, "steps": 3},
            {"controls": {"finger": 2}, "steps": 2}]}
        with self.assertRaises(ProtocolError):
            self.session.act(payload)
        self.assert_stationary(0, qpos)
        np.testing.assert_array_equal(self.session.requested, [0., 0.])

    def test_same_request_id_is_exactly_once(self):
        payload = self.request("once", [0.5, 0.25], 10)
        first = self.session.act(payload)
        qpos = self.backend.data.qpos.copy()
        self.assertGreater(qpos[0], 0.)
        self.assertEqual(first["advanced_steps"], 10)
        self.assertEqual(self.session.act(payload), first)
        self.assert_stationary(10, qpos)
        self.assertEqual(self.session.accepted_requests, 1)

    def test_reusing_id_for_different_action_is_not_a_second_motion(self):
        self.session.act(self.request("bound", [0.5, 0.25], 5))
        qpos = self.backend.data.qpos.copy()
        with self.assertRaises(ProtocolError) as raised:
            self.session.act(self.request("bound", [-0.5, 0.25], 5))
        self.assertEqual(raised.exception.status, 409)
        self.assert_stationary(5, qpos)

    def test_budget_records_executed_prefix_not_unexecuted_suffix(self):
        self.session.act(self.request("first", [0.5, 0.25], 12))
        receipt = self.session.act(self.request("prefix", [-0.5, 0.25], 8))
        self.assertEqual((receipt["planned_steps"], receipt["advanced_steps"]), (8, 3))
        self.assertEqual(self.backend.tick, 15)
        self.session.finish()
        chunks = sorted((Path(self.directory.name) / "evidence").glob("native_*.npz"))
        with np.load(chunks[-1], allow_pickle=False) as last:
            np.testing.assert_array_equal(last["tick"], [12, 13, 14, 15])
            np.testing.assert_array_equal(last["qpos"][-1], self.backend.data.qpos)
            np.testing.assert_array_equal(last["applied_ctrl"], [[-0.5, 0.25]] * 3)
            self.assertEqual(last["qpos"].dtype, np.float64)

    def test_named_updates_retain_other_commands(self):
        self.session.act(self.request("full", [0.5, 0.25], 2))
        self.session.act(self.request("partial", {"arm": -0.25}, 2))
        np.testing.assert_array_equal(self.backend.data.ctrl, [-0.25, 0.25])
        np.testing.assert_array_equal(self.session.requested, [-0.25, 0.25])

    def test_invalid_shapes_nonfinite_and_boolean_values_never_advance(self):
        qpos = self.backend.data.qpos.copy()
        for controls in ([0.], [0., 0., 0.], [True, 0.], [float("nan"), 0.],
                         {"object_pose": 0.}, {"arm": "0.2"}, {"arm": float("inf")},
                         {"arm": 10 ** 1000}, [10 ** 1000, 0.]):
            with self.subTest(controls=controls), self.assertRaises(ProtocolError):
                self.session.act(self.request("invalid", controls, 1))
            self.assert_stationary(0, qpos)

    def test_wall_budget_and_submission_never_reset_the_world(self):
        self.session.act(self.request("before", [0.5, 0.25], 3))
        qpos = self.backend.data.qpos.copy()
        self.time[0] = 61.
        with self.assertRaises(ProtocolError):
            self.session.act(self.request("late", [-0.5, 0.25], 3))
        self.assert_stationary(3, qpos)
        result = self.session.finish()
        self.assertEqual(result["termination"], "wall_time_budget")
        with self.assertRaises(ProtocolError):
            self.session.act(self.request("after-finish", [0., 0.], 1))
        self.assert_stationary(3, qpos)


if __name__ == "__main__":
    unittest.main()
