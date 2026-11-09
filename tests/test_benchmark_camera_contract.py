"""Boundary regressions only; real C14 pixels/depth require parent smoke.

No asset world, EGL context, scripted controller or historical episode is needed
for these tests. Mock render objects exercise lifetime/error boundaries, never
stand in for optical or physical acceptance.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from benchmark.rendering import LiveC14, _immutable_binding, _visual_model_fields
from benchmark.camera_support import (CAMERAS, REPOSITORY, WIDTH, HEIGHT,
                                      array_sha, state_sha, validate_mounts, verified_inputs, visual_leaves)
from benchmark.camera_support.visuals import LiveVisuals
from model_identity import model_fingerprint
import mujoco


class TaskSnapshot:
    def __init__(self):
        self.film_time = .02
        self.seen_data = None

    def snapshot(self):
        return {"film_time": self.film_time}

    def render_state(self, data):
        # Object identity is instrumentation, not physical state in this fixture.
        self.seen_data = data
        return {"fluid": {"time_s": self.film_time}, "tip": data.xpos[1].copy()}


def native_fixture():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".001"/><size nuserdata="2"/>
      <worldbody>
        <body name="object"><freejoint name="free"/>
          <geom type="sphere" size=".03" mass="1"/></body>
        <body name="jointed" pos="1 0 0"><joint name="hinge"/>
          <geom type="sphere" size=".02" mass="1"/></body>
        <body name="mocap" mocap="true" pos="0 0 2"/>
      </worldbody>
      <actuator><general joint="hinge" dyntype="filter" dynprm=".1"/></actuator>
    </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    task = TaskSnapshot()
    backend = SimpleNamespace(model=model, data=data, task=task, tick=7, dt_ns=1_000_000)
    data.time = .007
    data.qpos[0] = .25  # Intentionally leave live derived FK stale.
    data.qvel[:] = .03
    data.ctrl[:] = .12
    data.act[:] = .11
    data.qacc_warmstart[:] = .02
    data.mocap_pos[:] = [0.1, 0.2, 2.0]
    data.qfrc_applied[:] = .07
    data.xfrc_applied[1] = [.01, .02, .03, 0, 0, 0]
    data.userdata[:] = [2, 3]
    render = LiveC14.__new__(LiveC14)
    render.backend = backend
    render._source_model, render._source_data, render._source_task = model, data, task
    render._owner_thread = threading.get_ident()
    render._gpu_thread = None
    render._dt_ns, render._time_origin = backend.dt_ns, 0.0
    render._state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    render._fingerprint = model_fingerprint
    render.model, render.data = copy.copy(model), None
    render.data = mujoco.MjData(render.model)
    before = render._live_evidence()
    render._binding = _immutable_binding(before["model"])
    render._source_names_sha = before["names_sha256"]
    render._source_visual = _visual_model_fields(model)
    render._camera_arrays = {}
    render._renderer = render._mesh_lists = None
    render.metadata = {}
    return render, before


class CameraIdentityTests(unittest.TestCase):
    def mounts(self):
        path = REPOSITORY / "replay_support/inputs/reference_native_manifest.json"
        return json.loads(path.read_text())["mounts"]

    def test_exact_left_and_right_are_not_mirrored(self):
        mounts = self.mounts()
        validate_mounts(mounts, self.mounts())
        changed = copy.deepcopy(mounts)
        changed[1]["quaternion_camera_gl_in_mount_wxyz"] = changed[2]["quaternion_camera_gl_in_mount_wxyz"]
        with self.assertRaisesRegex(ValueError, "rotation"):
            validate_mounts(changed, mounts)

    def test_quaternion_sign_is_equivalent_but_head_binding_is_not(self):
        expected = self.mounts()
        changed = copy.deepcopy(expected)
        changed[1]["quaternion_camera_gl_in_mount_wxyz"] = [
            -value for value in changed[1]["quaternion_camera_gl_in_mount_wxyz"]]
        validate_mounts(changed, expected)
        changed[0]["original"]["body"] = "another_head"
        with self.assertRaisesRegex(ValueError, "head binding"):
            validate_mounts(changed, expected)

    def test_camera_order_is_required(self):
        mounts = self.mounts()
        with self.assertRaisesRegex(ValueError, "order"):
            validate_mounts([mounts[0], mounts[2], mounts[1]], mounts)

    def test_public_lock_rejects_changed_camera_spec(self):
        # Only the preceding input and corrupted target are needed: failure
        # must happen at the spec, before any later file or drawing import.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("replay_support/onboard_cameras.py")
            (root / relative).parent.mkdir(parents=True)
            shutil.copyfile(REPOSITORY / relative, root / relative)
            target = root / "replay_support/inputs/camera_spec_return.json"
            target.parent.mkdir()
            target.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch: replay_support/inputs/camera_spec"):
                verified_inputs(root)

    def test_leaf_loader_never_executes_module_launchers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "leaves.py"
            path.write_text("raise RuntimeError('launcher must never run')" + chr(10)
                            + "def leaf(): return 17" + chr(10))
            self.assertEqual(visual_leaves(path, ("leaf",))["leaf"](), 17)
            with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
                visual_leaves(path, ("absent",))

    def test_only_execution_gains_are_excluded_from_between_tick_binding(self):
        render, before = native_fixture()
        baseline = _immutable_binding(before["model"])
        render.backend.model.actuator_gainprm[0, 0] += 1
        self.assertEqual(_immutable_binding(model_fingerprint(render.backend.model)), baseline)
        render.backend.model.geom_contype[0] = 0
        self.assertNotEqual(_immutable_binding(model_fingerprint(render.backend.model)), baseline)


class NativeNoninterferenceTests(unittest.TestCase):
    def test_complete_state_copy_uses_private_forward_and_current_fk(self):
        render, before = native_fixture()
        received = []
        render._visuals = SimpleNamespace(prepare=received.append, _gravity_mesh=0, _film_mesh=1)
        render._renderer = SimpleNamespace(_gl_context=SimpleNamespace(make_current=mock.Mock()),
                                            _mjr_context=object())
        render._mesh_lists = SimpleNamespace(after_uploads=mock.Mock())
        render._onboard = SimpleNamespace(palm_transforms=mock.Mock(return_value={}))
        render._ensure_gpu = mock.Mock()
        original_forward = mujoco.mj_forward

        def private_forward(model, data):
            self.assertIs(model, render.model)
            self.assertIs(data, render.data)
            self.assertIsNot(data, render.backend.data)
            original_forward(model, data)

        with mock.patch("benchmark.rendering.mujoco.mj_forward", side_effect=private_forward), \
                mock.patch("benchmark.rendering.mujoco.mjr_uploadMesh"):
            render._prepare(before)
        self.assertIs(render.backend.task.seen_data, render.data)
        self.assertEqual(received[0]["fluid"]["time_s"], .02)
        self.assertEqual(received[0]["tip"][0], .25)
        self.assertEqual(render.backend.data.xpos[1, 0], 0.)
        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat",
                     "qfrc_applied", "xfrc_applied", "userdata", "qacc_warmstart"):
            np.testing.assert_array_equal(getattr(render.data, name), getattr(render.backend.data, name))
        self.assertTrue(render._assert_live_unchanged(before)["unchanged"])

    def test_allocated_island_cache_mutation_is_rejected(self):
        render, _ = native_fixture()
        model, data = render.backend.model, render.backend.data
        model.body_pos[2, 0] = .29
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
        mujoco.mj_forward(model, data)
        self.assertGreater(data.nisland, 0)
        before = render._live_evidence()
        data.iM[0] += 1.
        with self.assertRaises(RuntimeError):
            render._assert_live_unchanged(before)

    def test_guards_include_model_task_and_derived_native_arrays(self):
        for field in ("model", "task", "native"):
            with self.subTest(field=field):
                render, before = native_fixture()
                if field == "model":
                    render.backend.model.geom_friction[0, 0] *= 2
                elif field == "task":
                    render.backend.task.film_time += .02
                else:
                    render.backend.data.xpos[1, 0] = 100.
                with self.assertRaisesRegex(RuntimeError, "Rendering changed live state"):
                    render._assert_live_unchanged(before)

    def test_guard_runs_on_render_exception_without_repairing_live_state(self):
        render, _ = native_fixture()
        def fail(_):
            render.backend.data.ctrl[0] = .9
            raise ValueError("draw failed")
        render._prepare = fail
        with self.assertRaisesRegex(RuntimeError, "Rendering changed live state"):
            render.observe()
        self.assertEqual(render.backend.data.ctrl[0], .9)

    def test_replaced_backend_data_is_not_accepted(self):
        render, _ = native_fixture()
        render.backend.data = mujoco.MjData(render.backend.model)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            render._live_evidence()

    def test_wrong_thread_rejected_before_native_or_gl_work(self):
        render, _ = native_fixture()
        errors = []
        def another_thread():
            try:
                render.observe()
            except RuntimeError as error:
                errors.append(str(error))
        thread = threading.Thread(target=another_thread)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("owner thread", errors[0])

    def test_array_evidence_preserves_dtype_shape_and_no_nan_coercion(self):
        self.assertNotEqual(array_sha(np.zeros(4)), array_sha(np.zeros((2, 2))))
        with self.assertRaises(ValueError):
            state_sha({"measurement": float("nan")})


class DrawingBoundaryTests(unittest.TestCase):
    def test_fluid_capacity_reserves_all_36_assembly_parts(self):
        visual = LiveVisuals.__new__(LiveVisuals)
        visual.flying, visual.gravity_parcels, visual.releases = 0, [], []
        scene = SimpleNamespace(ngeom=100, maxgeom=167)
        with self.assertRaisesRegex(RuntimeError, "nothing truncated"):
            visual.append(scene, reserved_parts=36)
        self.assertEqual(scene.ngeom, 100)

    def test_probe_failure_rebuilds_full_scene_before_raising(self):
        render = LiveC14.__new__(LiveC14)
        render._build_scene = mock.Mock(return_value=({}, {"full_scene_geometry_sha256": "full"}))
        render._optical_probe = mock.Mock(side_effect=RuntimeError("optical failure"))
        with self.assertRaisesRegex(RuntimeError, "optical failure"):
            render._render_view({"id": CAMERAS[0]}, 0, 0, probe=True)
        self.assertEqual(render._build_scene.call_count, 2)

    def test_rendered_arrays_are_owned_and_native_resolution(self):
        render = LiveC14.__new__(LiveC14)
        buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        render._renderer = SimpleNamespace(render=lambda: buffer)
        render._build_scene = lambda spec: ({"id": spec["id"]}, {})
        images = [render._render_view({"id": camera}, 3, 3_000_000, probe=False)[0] for camera in CAMERAS]
        images[0][0, 0, 0] = 255
        self.assertEqual(buffer[0, 0, 0], 0)
        self.assertEqual(images[1][0, 0, 0], 0)
        self.assertTrue(all(image.shape == (720, 1280, 3) for image in images))

    def test_mesh_lists_close_first_and_reopen_reisolates(self):
        render = LiveC14.__new__(LiveC14)
        render._owner_thread = threading.get_ident()
        render._renderer = render._mesh_lists = render._gpu_thread = None
        render.model, render.parts, render._max_geom = object(), [object()] * 36, 22000
        render.metadata = {}
        calls = []
        def context():
            return SimpleNamespace(_gl_context=SimpleNamespace(make_current=lambda: calls.append("current")),
                                   close=lambda: calls.append("renderer.close"))
        def lists(model, renderer, parts):
            calls.append("lists.new")
            return SimpleNamespace(report={}, close=lambda: calls.append("lists.close"))
        render._onboard = SimpleNamespace(
            isolate_mesh_display_lists=lambda model, renderer: calls.append("isolate"), MeshLists=lists)
        with mock.patch("benchmark.rendering.mujoco.Renderer", side_effect=lambda *a, **k: context()):
            render._ensure_gpu()
            render.close()
            render.close()
            render._ensure_gpu()
            render.close()
        self.assertEqual(calls, ["current", "isolate", "lists.new", "lists.close", "renderer.close"] * 2)

    def test_context_closes_even_if_mesh_cleanup_raises(self):
        render = LiveC14.__new__(LiveC14)
        render._owner_thread = threading.get_ident()
        close = mock.Mock()
        render._renderer = SimpleNamespace(close=close)
        render._mesh_lists = SimpleNamespace(close=mock.Mock(side_effect=RuntimeError("list failure")))
        with self.assertRaisesRegex(RuntimeError, "list failure"):
            render.close()
        close.assert_called_once()
        self.assertIsNone(render._renderer)
        render.close()


if __name__ == "__main__":
    unittest.main()
