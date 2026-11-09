"""Attempt completeness and evidence integrity with real native physical transitions."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark.aggregate import aggregate, plan
from benchmark.recording import file_sha256, save_json
from benchmark.session import Session
from test_benchmark_protocol import SmallPhysicalBackend, fixture_identity


class MeasuredGoalBackend(SmallPhysicalBackend):
    def __init__(self, fail_at=None):
        super().__init__()
        self.fail_at = fail_at

    def step(self, controls):
        if self.tick == self.fail_at:
            raise RuntimeError("Injected native-step failure")
        return super().step(controls)

    def outcome(self):
        return {"outcome": {"functional": bool(self.data.qpos[0] > 0.000001)}}


class AttemptEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.spec = {"id": "native-slider-goal",
                     "budgets": {"max_physics_steps": 20, "max_wall_seconds": 60}}
        fixture_identity(self.root)
        save_json(self.root / "task.json", self.spec)
        self.suite = plan(self.root / "task.json", [self.root / "case.json"], repetitions=2)

    def tearDown(self):
        self.temporary.cleanup()

    def attempt(self, index, *, controls=0.5, steps=10, fail_at=None):
        directory = self.root / self.suite["attempts"][index]["attempt_id"]
        directory.mkdir()
        fixture_identity(directory)
        identity = json.loads((directory / "source_manifest.json").read_text())
        identity["files"]["test_benchmark_aggregate.py"] = file_sha256(__file__)
        save_json(directory / "source_manifest.json", identity)
        backend = MeasuredGoalBackend(fail_at=fail_at)
        session = Session(backend, None, directory, self.spec)
        if steps:
            session.act({"request_id": "motion", "actions": [
                {"controls": [controls, 0.0], "steps": steps}]})
        session.finish()
        session.recorder.close()
        return directory

    def test_missing_attempt_prevents_an_official_score(self):
        self.attempt(0)
        result = aggregate(self.suite, self.root)
        self.assertEqual(result["functional_successes"], 1)
        self.assertEqual(result["missing_attempts"], 1)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["official_functional_success_rate"])

    def test_task_failure_stays_in_the_denominator(self):
        self.attempt(0)
        self.attempt(1, controls=-0.5)
        result = aggregate(self.suite, self.root)
        self.assertEqual(result["valid_attempts"], 2)
        self.assertEqual(result["functional_successes"], 1)
        self.assertEqual(result["official_functional_success_rate"], 0.5)

    def test_infrastructure_failure_is_not_a_valid_task_failure(self):
        self.attempt(0)
        directory = self.attempt(1, fail_at=2)
        result = aggregate(self.suite, self.root)
        self.assertEqual(result["infrastructure_errors"], 1)
        self.assertIsNone(result["official_functional_success_rate"])
        self.assertEqual(json.loads((directory / "result.json").read_text())["physics_steps"], 2)

    def test_zero_action_attempt_retains_its_native_boundary_states(self):
        directory = self.attempt(0, steps=0)
        self.attempt(1, steps=0)
        result = aggregate(self.suite, self.root)
        self.assertEqual(result["official_functional_success_rate"], 0.0)
        with np.load(directory / "evidence/initial_state.npz", allow_pickle=False) as initial:
            with np.load(directory / "evidence/final_state.npz", allow_pickle=False) as final:
                for field in ("tick", "time", "qpos", "qvel", "act", "ctrl",
                              "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
                    np.testing.assert_array_equal(initial[field], final[field])
                np.testing.assert_array_equal(final["tick"], [0])

    def test_changed_source_manifest_invalidates_the_attempt(self):
        directory = self.attempt(0)
        self.attempt(1)
        identity = json.loads((directory / "source_manifest.json").read_text())
        identity["case_sha256"] = file_sha256(self.root / "task.json")
        save_json(directory / "source_manifest.json", identity)
        result = aggregate(self.suite, self.root)
        self.assertEqual(result["invalid_evidence"], 1)
        self.assertIsNone(result["official_functional_success_rate"])

    def test_damaged_native_transition_invalidates_the_attempt(self):
        directory = self.attempt(0)
        self.attempt(1)
        manifest = json.loads((directory / "evidence/manifest.json").read_text())
        chunk = directory / "evidence" / manifest["chunks"][0]["path"]
        chunk.write_bytes(b"damaged native evidence")
        result = aggregate(self.suite, self.root)
        self.assertEqual(result["invalid_evidence"], 1)
        self.assertIsNone(result["official_functional_success_rate"])


if __name__ == "__main__":
    unittest.main()
