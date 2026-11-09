"""Three current-tick C14 RGB views from an independent native workspace.

The caller serializes physics, observe/optical_proof and close on one owner
thread. No task policy, saved trajectory, PNG export, HUD or montage is here.
See camera_support/inputs.lock.json for the exact public drawing dependencies.
Optical/assembly acceptance does not certify installation or hardware.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import threading

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ["MUJOCO_GL"] in ("egl", "osmesa"):
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco
import numpy as np

from .camera_support import (CAMERAS, WIDTH, HEIGHT, LOCK_SHA256, REPOSITORY,
                             array_sha, geometry_snapshot, native_numeric_evidence,
                             sha256, state_sha, validate_mounts, verified_inputs, visual_leaves)
from .camera_support.visuals import LiveVisuals

CAMERA_ARRAYS = ("cam_bodyid", "cam_mode", "cam_targetbodyid", "cam_pos", "cam_quat",
                 "cam_fovy", "cam_ipd", "cam_sensorsize")


def _immutable_binding(fingerprint):
    # The native wheel brake changes only these execution gains between ticks.
    result = dict(fingerprint, arrays=dict(fingerprint["arrays"]))
    for name in ("actuator_gainprm", "actuator_biasprm"):
        result["arrays"].pop(name)
    return state_sha(result)


def _visual_model_fields(model):
    result = {}
    for label, owner in (("stat", model.stat), *[
            ("vis." + name, getattr(model.vis, name))
            for name in ("global_", "quality", "headlight", "map", "scale", "rgba")]):
        values = {}
        for name in dir(owner):
            if name.startswith("_"):
                continue
            value = getattr(owner, name)
            if isinstance(value, np.ndarray):
                values[name] = array_sha(value)
            elif isinstance(value, (int, float)):
                values[name] = value
        result[label] = values
    return result


class LiveC14:
    """LiveC14(backend, output); close releases GL and observe can reopen it.

    output belongs to the parent recorder; this renderer never writes images or
    an MJB. All evidence/metadata is private, while observe exposes only the
    three RGB8 arrays, current camera calibration and integer tick/time.
    """

    def __init__(self, backend, output: Path):
        self.backend, self.output = backend, Path(output)
        self._owner_thread = threading.get_ident()
        self._renderer = self._mesh_lists = None
        self._gpu_thread = None
        self._source_model, self._source_data, self._source_task = backend.model, backend.data, backend.task
        self._state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self._dt_ns = int(backend.dt_ns)
        if (self._dt_ns <= 0 or self._dt_ns != backend.dt_ns
                or abs(float(backend.model.opt.timestep) * 1e9 - self._dt_ns) > 1e-6):
            raise ValueError("Native timestep and declared integer clock disagree")
        self._time_origin = float(backend.data.time) - int(backend.tick) * self._dt_ns / 1e9
        self.inputs = verified_inputs()
        # This module contains drawing leaves only. No replay driver is imported.
        from replay_support import onboard_cameras
        import model_identity
        for module, relative in ((onboard_cameras, "replay_support/onboard_cameras.py"),
                                 (model_identity, "model_identity.py")):
            if Path(module.__file__).resolve() != (REPOSITORY / relative).resolve():
                raise ValueError("Loaded camera dependency is not the pinned public input: " + relative)
        self._onboard = onboard_cameras
        self._fingerprint = model_identity.model_fingerprint
        before = self._live_evidence()
        self._binding = _immutable_binding(before["model"])
        self._source_names_sha = before["names_sha256"]
        self._source_visual = before["visual_model"]
        try:
            self.model = copy.copy(backend.model)
            if self.model is backend.model or self._fingerprint(self.model) != before["model"]:
                raise ValueError("Renderer model must be an exact independent native copy")
            for name in before["model"]["arrays"]:
                live, private = getattr(backend.model, name), getattr(self.model, name)
                if live.size and np.shares_memory(live, private):
                    raise ValueError("Renderer model aliases live native memory: " + name)
            self.data = mujoco.MjData(self.model)
            if np.shares_memory(self.data.qpos, backend.data.qpos):
                raise ValueError("Renderer data aliases live native memory")
            source = REPOSITORY / "replay_support/inputs"
            spec = json.loads((source / "camera_spec_return.json").read_text(encoding="utf-8"))
            self.specs = spec["cameras"]
            if [s["id"] for s in self.specs] != list(CAMERAS):
                raise ValueError("C14 camera IDs/order changed")
            reference = json.loads((source / "reference_native_manifest.json").read_text(encoding="utf-8"))
            self.mounts = [onboard_cameras.configure_camera(self.model, s) for s in self.specs]
            validate_mounts(self.mounts, reference["mounts"])
            self.near_by_camera = reference["near_clip_m_by_camera"]
            for s in self.specs:
                if float(s.get("near_clip_m", .003)) != self.near_by_camera[s["id"]]:
                    raise ValueError("C14 per-camera near clip changed")
            self._camera_arrays = {name: getattr(self.model, name).copy() for name in CAMERA_ARRAYS}
            self.assembly, self.parts = onboard_cameras.load_assembly(source / "assembly/assembly_manifest.json")
            if (len(self.parts) != 36
                    or sha256(source / "assembly" / self.assembly["camera_spec"])
                    != self.inputs["files"]["replay_support/inputs/camera_spec_return.json"]):
                raise ValueError("C14 assembly/spec identity differs")
            self.model.vis.global_.offwidth = max(WIDTH, self.model.vis.global_.offwidth)
            self.model.vis.global_.offheight = max(HEIGHT, self.model.vis.global_.offheight)
            self.model.vis.global_.ipd = 0.0
            self._options = mujoco.MjvOption()
            self._options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
            self._options.sitegroup[:] = 0
            self._options.frame = mujoco.mjtFrame.mjFRAME_NONE
            self._visuals = LiveVisuals(self.model, self.data)
            self._projection = visual_leaves(REPOSITORY / "replay_support/references/projection_reference.py",
                                              ("project", "check_projection"), mujoco=mujoco)
            self._max_geom = max(22000, int(self.model.ngeom) + 36 + 256)
            self.metadata = {
                "identity": "benchmark-live-c14-v1", "mujoco_version": mujoco.__version__,
                "input_lock_sha256": LOCK_SHA256, "input_sha256": self.inputs["files"],
                "source_sha256": {str(p.relative_to(REPOSITORY)): sha256(p) for p in (
                    Path(__file__), Path(__file__).with_name("camera_support") / "__init__.py",
                    Path(__file__).with_name("camera_support") / "visuals.py")},
                "source_model_semantic_sha256": state_sha(before["model"]),
                "source_model_names_sha256": self._source_names_sha,
                "state_spec": self._state_spec, "mounts": self.mounts,
                "near_clip_m_by_camera": self.near_by_camera, "image_size_wh": [WIDTH, HEIGHT],
                "assembly_parts": [{"id": p.id, "sha256": p.digest} for p in self.parts],
                "assembly_limitations": self.assembly["limitations"],
                "hardware_mechanical_installation_certified": False,
                "optical_projection_verified": False, "display_list_layout": None,
                "state_source": "complete current integration snapshot; private forward for current FK",
                "fluid_clock": "current task film clock, not replaced by native time or advanced here",
                "occlusion": "full native geometry prefix plus all 36 C14 parts; no wrist/hand replacement",
                "mutable_native_model_fields_between_ticks": ["actuator_gainprm", "actuator_biasprm"],
            }
            self.last_observation_evidence = None
            self.last_optical_proof = None
        finally:
            self._assert_live_unchanged(before)

    def _assert_owner(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("LiveC14 must run on its serialized physics/EGL owner thread")

    def _native_state(self):
        state = np.empty(mujoco.mj_stateSize(self.backend.model, self._state_spec), dtype=np.float64)
        mujoco.mj_getState(self.backend.model, self.backend.data, state, self._state_spec)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite complete native integration state")
        return state

    def _live_evidence(self):
        b = self.backend
        if (b.model is not self._source_model or b.data is not self._source_data
                or b.task is not self._source_task):
            raise RuntimeError("Bound live model/data/task identity changed")
        if b.dt_ns != self._dt_ns or type(b.tick) is not int or b.tick < 0:
            raise ValueError("Invalid native tick/clock")
        return {"tick": b.tick, "dt_ns": b.dt_ns, "state": self._native_state(),
                "model": self._fingerprint(b.model),
                "names_sha256": hashlib.sha256(b.model.names).hexdigest(),
                "visual_model": _visual_model_fields(b.model),
                "native_numeric_sha256": native_numeric_evidence(b.data),
                "task_snapshot_sha256": state_sha(b.task.snapshot())}

    def _assert_live_unchanged(self, before):
        after = self._live_evidence()
        if state_sha(before) != state_sha(after):
            changes = [key for key in before if state_sha(before[key]) != state_sha(after[key])]
            raise RuntimeError("Rendering changed live state or ran concurrently: " + ", ".join(changes))
        return {"unchanged": True, "tick": before["tick"],
                "native_state_sha256": array_sha(before["state"]),
                "native_numeric_sha256": before["native_numeric_sha256"],
                "model_sha256": state_sha(before["model"]),
                "model_visual_sha256": state_sha(before["visual_model"]),
                "task_snapshot_sha256": before["task_snapshot_sha256"],
                "scope": "full semantic model, mjSTATE_INTEGRATION, exposed numeric/contact/solver caches, task.snapshot"}

    def _prepare(self, before):
        if (_immutable_binding(before["model"]) != self._binding
                or before["names_sha256"] != self._source_names_sha
                or before["visual_model"] != self._source_visual):
            raise ValueError("Bound native model/camera/topology changed; create a new renderer/world")
        for name, expected in self._camera_arrays.items():
            if not np.array_equal(getattr(self.model, name), expected):
                raise ValueError("Private C14 local camera transform changed: " + name)
        state = before["state"]
        if state.shape != (mujoco.mj_stateSize(self.model, self._state_spec),):
            raise ValueError("Native/render integration dimensions differ")
        # These setters and both forwards target private data only. The first
        # forward avoids stale post-mj_step FK in render_state's tube transform.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, state, self._state_spec)
        expected_time = self._time_origin + before["tick"] * self._dt_ns / 1e9
        if abs(float(self.data.time) - expected_time) > max(1e-9, (abs(expected_time) + 1) * 1e-10):
            raise ValueError("Native snapshot is not at the declared tick")
        mujoco.mj_forward(self.model, self.data)
        payload = copy.deepcopy(self.backend.task.render_state(self.data))
        self._visuals.prepare(payload)
        mujoco.mj_forward(self.model, self.data)
        self._ensure_gpu()
        self._renderer._gl_context.make_current()
        for mesh in (self._visuals._gravity_mesh, self._visuals._film_mesh):
            mujoco.mjr_uploadMesh(self.model, self._renderer._mjr_context, mesh)
        self._mesh_lists.after_uploads()
        self._transforms = self._onboard.palm_transforms(self.model, self.data)
        return state_sha(payload)

    def _ensure_gpu(self):
        self._assert_owner()
        if self._renderer is not None:
            if self._gpu_thread != threading.get_ident():
                raise RuntimeError("EGL context belongs to another worker")
            return
        renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH, max_geom=self._max_geom)
        try:
            renderer._gl_context.make_current()
            isolation = self._onboard.isolate_mesh_display_lists(self.model, renderer)
            lists = self._onboard.MeshLists(self.model, renderer, self.parts)
        except BaseException:
            renderer.close()
            raise
        self._renderer, self._mesh_lists = renderer, lists
        self._gpu_thread = threading.get_ident()
        self.metadata["display_list_layout"] = {"isolation": isolation, "assembly": lists.report}

    def _build_scene(self, spec):
        renderer = self._renderer
        near = self._onboard.set_camera_near_clip(self.model, spec)
        warning = self.data.warning[mujoco.mjtWarning.mjWARN_VGEOMFULL]
        previous_warning = int(warning.number)
        renderer.update_scene(self.data, camera=spec["native_camera"], scene_option=self._options)
        if int(warning.number) != previous_warning or renderer.scene.ngeom >= renderer.scene.maxgeom:
            raise RuntimeError("Native scene capacity would omit original occluders")
        cal = self._onboard.native_calibration(self.model, self.data, renderer.scene, spec, WIDTH, HEIGHT)
        if not np.isclose(cal["near_m"], near, rtol=0, atol=2e-8):
            raise ValueError("Native near clipping differs from the C14 camera contract")
        count = int(renderer.scene.ngeom)
        native_prefix = geometry_snapshot(renderer.scene, count)
        self._visuals.append(renderer.scene, reserved_parts=len(self.parts))
        with_liquid = int(renderer.scene.ngeom)
        liquid_prefix = geometry_snapshot(renderer.scene, with_liquid)
        self._mesh_lists.append(self.parts, self._transforms)
        if (renderer.scene.ngeom != with_liquid + 36
                or geometry_snapshot(renderer.scene, count) != native_prefix
                or geometry_snapshot(renderer.scene, with_liquid) != liquid_prefix):
            raise RuntimeError("C14/liquid drawing replaced native geometry or truncated assembly")
        return cal, {"native_scene_geom_count": count, "liquid_geom_count": with_liquid - count,
                     "added_part_geom_count": 36, "native_geometry_prefix_checked": True,
                     "native_geometry_prefix_sha256": native_prefix,
                     "full_scene_geometry_sha256": geometry_snapshot(renderer.scene, renderer.scene.ngeom),
                     "configured_near_m": near}

    def _optical_probe(self, cal):
        # check_projection installs real isolated markers in the native scene.
        # _render_view rebuilds the full scene in finally, even on probe failure.
        renderer = self._renderer
        proof = self._projection["check_projection"](renderer, dict(cal, distance_m=.4))
        depth_z = max(2 * cal["near_m"], .12)
        points = depth_z * np.array([[0., 0., 1.], [.12, -.06, 1.], [-.09, .08, 1.]])
        transform = np.asarray(cal["world_from_camera_cv"])
        world = points @ transform[:3, :3].T + transform[:3, 3]
        uv, expected = self._projection["project"](world, cal)
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        if depth.shape != (HEIGHT, WIDTH) or not np.isfinite(depth).all():
            raise RuntimeError("Invalid native optical depth buffer")
        pixels = np.rint(uv).astype(int)
        if np.any(pixels < 0) or np.any(pixels[:, 0] >= WIDTH) or np.any(pixels[:, 1] >= HEIGHT):
            raise RuntimeError("Optical diagnostic marker lies outside the native image")
        measured = depth[pixels[:, 1], pixels[:, 0]]
        errors = np.abs(measured - expected)
        tolerance = .004 * expected + .0003
        if np.any(errors > tolerance):
            raise RuntimeError("Native optical depth/projection mismatch: " + repr(errors.tolist()))
        return dict(proof, depth_expected_m=expected.tolist(), depth_measured_m=measured.tolist(),
                    depth_error_m=errors.tolist(), depth_tolerance_m=tolerance.tolist(),
                    scope="isolated native optical markers; full scene restored before RGB")

    def _render_view(self, spec, tick, t_ns, *, probe):
        cal, evidence = self._build_scene(spec)
        proof = None
        if probe:
            full_scene = evidence["full_scene_geometry_sha256"]
            try:
                proof = self._optical_probe(cal)
            finally:
                cal, evidence = self._build_scene(spec)
                if evidence["full_scene_geometry_sha256"] != full_scene:
                    raise RuntimeError("Full live scene was not restored after optical diagnostics")
            proof["full_scene_restored"] = True
        rgb = self._renderer.render().copy()
        if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8:
            raise RuntimeError("Unexpected native RGB8 image contract")
        cal.update(tick=tick, t_ns=t_ns)
        evidence.update(camera=spec["id"], tick=tick, t_ns=t_ns, calibration=cal,
                        optical_probe=proof, rgb_sha256=array_sha(rgb))
        return rgb, cal, evidence

    def _capture(self, *, probe):
        self._assert_owner()
        before = self._live_evidence()
        tick, t_ns = before["tick"], before["tick"] * self._dt_ns
        try:
            payload_sha = self._prepare(before)
            images, calibration, views = {}, {}, {}
            for spec in self.specs:
                rgb, cal, evidence = self._render_view(spec, tick, t_ns, probe=probe)
                images[spec["id"]], calibration[spec["id"]], views[spec["id"]] = rgb, cal, evidence
        finally:
            guard = self._assert_live_unchanged(before)
        report = {"tick": tick, "t_ns": t_ns, "cameras": views,
                  "noninterference": guard, "live_visual_payload_sha256": payload_sha,
                  "input_lock_sha256": LOCK_SHA256}
        self.last_observation_evidence = report
        return {"tick": tick, "t_ns": t_ns, "images": images, "calibration": calibration}, report

    def observe(self):
        """Return three distinct current-tick 1280x720 RGB arrays and calibration."""
        packet, _ = self._capture(probe=False)
        return packet

    def optical_proof(self):
        """Private smoke API: actual marker pixels/depth, scene restoration and guards.

        Call on the physics owner, then observe/save/inspect the real images.
        This proves optical projection only, not task success or hand visibility.
        No optical pass is claimed before this method actually succeeds.
        """
        _, report = self._capture(probe=True)
        self.last_optical_proof = report
        self.metadata["optical_projection_verified"] = True
        return report

    def close(self):
        self._assert_owner()
        renderer, lists = self._renderer, self._mesh_lists
        if renderer is None:
            return
        try:
            if lists is not None:
                lists.close()
        finally:
            try:
                renderer.close()
            finally:
                self._renderer = self._mesh_lists = None
                self._gpu_thread = None
