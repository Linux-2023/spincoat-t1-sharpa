"""HTTP privacy and parsing boundaries exercised against real native transitions."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from benchmark.server import BenchmarkServer
from benchmark.session import Session
from test_benchmark_protocol import SmallPhysicalBackend, fixture_identity


class HTTPBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.backend = SmallPhysicalBackend()
        fixture_identity(self.directory.name)
        self.session = Session(self.backend, None, Path(self.directory.name), {
            "id": "physical-http-fixture", "description": "Fixture only",
            "budgets": {"max_physics_steps": 100, "max_wall_seconds": 60}})
        self.token = "test-only-credential-not-used-for-any-external-service"
        self.server = BenchmarkServer(("127.0.0.1", 0), self.session, self.token)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01})
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        self.session.finish()
        self.session.recorder.close()
        self.directory.cleanup()

    def request(self, method, path, body=None, *, authorized=True, content_type="application/json"):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        headers = {"Content-Type": content_type}
        if authorized:
            headers["Authorization"] = "Bearer " + self.token
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        status, value = response.status, json.loads(response.read())
        connection.close()
        return status, value

    def test_unauthorized_action_does_not_execute(self):
        request = json.dumps({"request_id": "forbidden", "actions": [
            {"controls": [0.5, 0.25], "steps": 10}]})
        qpos = self.backend.data.qpos.copy()
        status, _ = self.request("POST", "/v1/act", request, authorized=False)
        self.assertEqual(status, 401)
        self.assertEqual(self.backend.tick, 0)
        np.testing.assert_array_equal(self.backend.data.qpos, qpos)

    def test_ambiguous_or_nonfinite_json_cannot_advance(self):
        payloads = [
            b'{"request_id":"a","request_id":"b","actions":[{"controls":[0.5,0.2],"steps":5}]}',
            b'{"request_id":"a","actions":[{"controls":[NaN,0.2],"steps":5}]}',
            b'{"request_id":"a","actions":[{"controls":[1e400,0.2],"steps":5}]}',
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                status, _ = self.request("POST", "/v1/act", payload)
                self.assertEqual(status, 400)
                self.assertEqual(self.backend.tick, 0)
                self.assertEqual(self.session.lifecycle, "running")

    def test_private_paths_and_undelivered_images_are_inaccessible(self):
        private = Path(self.directory.name) / "private.json"
        private.write_text('{"object_pose":[0.1,0.2,0.3]}')
        for path in ("/private.json", "/v1/private.json", "/v1/images/../private.json",
                     "/v1/images/000000/cam_head.png", "/v1/model.mjb"):
            with self.subTest(path=path):
                status, body = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertNotIn("object_pose", body)
        status, public = self.request("GET", "/v1/info")
        self.assertEqual(status, 200)
        self.assertNotIn("metrics", public)
        self.assertNotIn("object_pose", public)
        self.assertEqual(self.backend.tick, 0)

    def test_wrong_media_type_is_rejected_without_killing_episode(self):
        status, _ = self.request("POST", "/v1/act", b"{}", content_type="text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(self.session.lifecycle, "running")
        valid = json.dumps({"request_id": "valid", "actions": [
            {"controls": [0.5, 0.25], "steps": 10}]})
        status, receipt = self.request("POST", "/v1/act", valid)
        self.assertEqual(status, 200)
        self.assertEqual(receipt["advanced_steps"], 10)
        self.assertGreater(self.backend.data.qpos[0], 0.)


if __name__ == "__main__":
    unittest.main()
