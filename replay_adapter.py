"""Portable bindings for the original C14 saved-state renderer.

These helpers relocate inputs and check contracts; they do not implement a scene,
change camera settings, integrate state, or generate a substitute trajectory.
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

FPS = 30
NS = 1_000_000_000
DT_NS = 1_000_000
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
TIMING_LABEL = "native11 reference: original 2x/4x-timed motions; NOT a new 1x run"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    """Create an output exactly once, retaining failed partial files as evidence."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def contained_file(root, relative):
    root = Path(root).resolve(strict=True)
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError(f"Expected a contained relative file, got {relative!r}")
    path = (root / part).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Input escapes its declared root: {relative}")
    return path


def verify_files(root, files):
    if not isinstance(files, dict) or not files:
        raise ValueError("A nonempty path-to-SHA256 input binding is required")
    paths = {}
    for relative, digest in files.items():
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"Invalid SHA256 for {relative}")
        path = contained_file(root, relative)
        if sha256(path) != digest:
            raise ValueError(f"Immutable input changed: {relative}")
        paths[relative] = path
    return paths


def check_output(output, protected):
    """Never place new replay output inside reference/support/native inputs."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Replay output must be a new directory: {output}")
    for path in protected:
        path = Path(path).resolve(strict=True)
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(f"Replay output overlaps an immutable input: {path}")
    return output


def require_versions(packages):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The recorded replay runtime requires Python 3.11")
    required = {"mujoco", "numpy", "Pillow", "scipy", "trimesh", "PyOpenGL"}
    if not required.issubset(packages):
        raise ValueError("Replay lock must pin all six original rendering packages")
    actual = {name: importlib.metadata.version(name) for name in packages}
    if actual != packages:
        raise RuntimeError(f"Replay runtime differs from pinned packages: {actual}")
    return actual


class NoPhysics:
    """Guard both public and native Python mj_step entrypoints before imports.

    This is a fail-closed contract for the hash-bound Python renderer, not an OS
    sandbox for arbitrary malicious extensions. mj_forward is intentionally legal.
    """

    def __init__(self, mujoco):
        self.mujoco = mujoco
        self.attempts = 0
        self.originals = []
        self.blocked = self._forbid

    def _forbid(self, *args, **kwargs):
        self.attempts += 1
        raise RuntimeError("mj_step* is forbidden: this is saved-state replay only")

    def __enter__(self):
        modules = [self.mujoco]
        native = getattr(self.mujoco, "_functions", None)
        if native is not None:
            modules.append(native)
        for module in modules:
            for name in dir(module):
                if name.startswith("mj_step") and callable(getattr(module, name)):
                    self.originals.append((module, name, getattr(module, name)))
                    setattr(module, name, self.blocked)
        if not {"mj_step", "mj_step1", "mj_step2"}.issubset(
                {name for _, name, _ in self.originals}):
            self._restore()
            raise RuntimeError("Cannot install all required no-physics guards")
        return self

    def check(self):
        if self.attempts:
            raise RuntimeError(f"Forbidden physics calls attempted: {self.attempts}")
        if any(getattr(module, name) is not self.blocked for module, name, _ in self.originals):
            raise RuntimeError("A renderer dependency replaced a no-physics guard")

    def _restore(self):
        for module, name, original in reversed(self.originals):
            setattr(module, name, original)

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.check()
        finally:
            self._restore()


def import_file(name, path):
    """Import the bound copy, refusing an already-imported different module."""
    path = Path(path).resolve(strict=True)
    if name in sys.modules:
        existing = sys.modules[name]
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError(f"Unexpected previously imported module: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load bound module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


@contextmanager
def renderer_bindings(package_root, lock, files):
    """Inject the same original rendering leaves without running old launchers."""
    support = files[lock["renderer"]].parent
    native = (package_root / lock["native_directory"]).resolve(strict=True)
    if not native.is_relative_to(package_root) or not native.is_dir():
        raise ValueError("native_directory must be a package-local source directory")
    for path in native.glob("*.py"):
        if str(path.relative_to(package_root)) not in files:
            raise ValueError(f"Unbound native import candidate: {path.name}")
    old_path = sys.path[:]
    sys.path[:0] = [str(support), str(native)]
    try:
        import numpy as np
        import mujoco
        from PIL import Image, ImageDraw, ImageFont
        from scipy.interpolate import RegularGridInterpolator

        def from_support(name):
            relative = str((support / (name + ".py")).relative_to(package_root))
            return import_file(name, files[relative])

        def from_native(name):
            relative = str((native / (name + ".py")).relative_to(package_root))
            return import_file(name, files[relative])

        from_support("action_capture")
        exports = from_support("export_actions")
        timeline = from_support("v34_overlap_timeline")
        onboard = from_support("onboard_cameras")
        renderer = import_file("_portable_c14_renderer", files[lock["renderer"]])
        palette = from_native("t1_liquid_palette")
        gravity = from_native("t1_gravity_liquid_render_v2")
        # Only the original fluid drawing module is needed; do not instantiate
        # t1_spincoat_scene or follow its former private ROOT/ORIGINAL paths.
        fluid = import_file("_portable_recorded_fluid", files[lock["fluid_render"]])

        def fluid_module(name):
            if name != "fluid_render":
                raise ValueError(f"Unbound fluid module requested: {name}")
            return fluid

        for name, value in {
            "np": np, "mujoco": mujoco, "Image": Image, "ImageDraw": ImageDraw,
            "ImageFont": ImageFont, "RegularGridInterpolator": RegularGridInterpolator,
            "onboard": onboard, "GravityLiquidReplay": gravity.GravityLiquidReplay,
            "draw_fluid_with_palette": gravity.draw_fluid_with_palette,
            "apply_liquid_palette": palette.apply_liquid_palette,
            "LIQUID_RGBA": palette.LIQUID_RGBA, "fluid_module": fluid_module,
            "phase_at_tick": timeline.phase_at_tick,
        }.items():
            setattr(renderer, name, value)
        yield renderer, exports, timeline
    finally:
        sys.path[:] = old_path


def validate_stream(aligned, lock):
    import numpy as np
    manifest = aligned["manifest"]
    terminal = manifest["terminal_t_ns"]
    if (type(terminal) is not int or terminal <= 0 or terminal % DT_NS
            or manifest["source_physics_dt_ns"] != DT_NS
            or manifest["source_physics_steps"] * DT_NS != terminal):
        raise ValueError("Reference is not a complete original 1ms native clock")
    count = (terminal * FPS + NS - 1) // NS
    if count != lock["frame_count"] or terminal != lock["terminal_t_ns"]:
        raise ValueError("Reference frame/termination identity differs from the lock")
    indices = np.arange(count, dtype="<i8")
    presentation = indices * NS // FPS
    ticks = presentation // DT_NS
    expected = {
        "frame_index": indices, "t_ns": presentation, "source_tick": ticks,
        "source_t_ns": ticks * DT_NS, "source_lag_ns": presentation - ticks * DT_NS,
        "action_duration_ns": np.minimum((indices + 1) * NS // FPS, terminal) - presentation,
    }
    for name, value in expected.items():
        actual = aligned[name]
        if actual.dtype != np.dtype("<i8") or not np.array_equal(actual, value):
            raise ValueError(f"Reference source/presentation cadence changed: {name}")
    clock = aligned["source_time_s"]
    if (clock.shape != (count,) or not np.isfinite(clock).all()
            or np.max(np.abs(clock - ticks * .001)) > 1e-7):
        raise ValueError("Reference floating timestamps differ from source ticks")
    for name in ("qpos", "qvel", "action"):
        value = aligned[name]
        if (value.ndim != 2 or value.shape[0] != count
                or value.dtype != np.dtype("<f8") or not np.isfinite(value).all()):
            raise ValueError(f"Invalid original saved-state matrix: {name}")
    if manifest["action_dim"] != aligned["action"].shape[1]:
        raise ValueError("Recorded action dimension differs from its manifest")
    for value in aligned.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return count


def validate_overlap(reference, aligned, timeline, package_files, lock):
    """Validate portable same-episode evidence without opening historical paths."""
    import numpy as np
    source = read_json(reference / "source_manifest.json")
    labels = read_json(reference / "audited_display_labels.json")
    evidence = source["dual_arm_overlap"]
    manifest = aligned["manifest"]
    sidecar = read_json(reference / "replay/dual_arm_overlap.json")
    result = read_json(reference / "replay/result.json")
    if (source["version"] != lock["native_version_id"]
            or source["case_id"] != manifest["case_id"]
            or labels["dual_arm_overlap"] != evidence):
        raise ValueError("Reference overlap version/case identity differs")
    if result.get("success") is not True or result.get("stages", {}).get("finish", {}).get("success") is not True:
        raise ValueError("Reference does not contain its successful native11 finish")
    expected_hashes = {
        "sidecar_sha256": sha256(reference / "replay/dual_arm_overlap.json"),
        "trace_sha256": sha256(reference / "replay/dual_arm_overlap_trace.npz"),
        "source_result_sha256": sha256(reference / "replay/result.json"),
        "source_model_sha256": manifest["source_model_sha256"],
        "capture_manifest_sha256": manifest["source_capture_manifest_sha256"],
        "timeline_module_sha256": sha256(package_files[lock["timeline"]]),
    }
    if any(evidence[key] != value for key, value in expected_hashes.items()):
        raise ValueError("Native overlap hashes disagree with the immutable export")
    if (sidecar["schema_version"] != 1 or sidecar["status"] != "completed"
            or sidecar["physics_dt_ns"] != DT_NS
            or sidecar["source_case_id"] != manifest["case_id"]
            or evidence["physics_steps"] != manifest["source_physics_steps"]):
        raise ValueError("Reference overlap did not complete in the bound native episode")
    for key in ("source_case_id", "output_directory", "model_identity", "data_identity", "physics_dt_ns"):
        if sidecar[key] != evidence[key]:
            raise ValueError(f"Reference sidecar identity differs: {key}")
    if sidecar["output_directory"] != source["original_directory"]:
        raise ValueError("Reference sidecar provenance names a different source episode")
    handoffs = result["live_handoffs"]
    if not handoffs or any(row[key] != sidecar[key] for row in handoffs for key in ("model_identity", "data_identity")):
        raise ValueError("Reference live handoffs have mixed model/data identity")
    intervals = timeline.validate_intervals(sidecar["intervals"], terminal_tick=manifest["source_physics_steps"])
    if not intervals:
        raise ValueError("Missing recorded secondary activity; no fallback labels are allowed")
    for key in ("release_gate", "initial_hold_completed", "preshape_reference_completed",
                "arm_reference_completed", "original_bounds_passed", "command_ownership_passed",
                "precontact_passed", "right_environment_passed", "interarm_contact_passed",
                "native_staging_tracking_guard_passed"):
        if sidecar["qualification"][key] is not True:
            raise ValueError(f"Recorded native overlap gate failed: {key}")
    start, end = intervals[0]["start_tick"], intervals[-1]["end_tick"]
    preparation = [row for row in handoffs if row["stage"] == "reagent_preparation_end"]
    if (len(preparation) != 1 or not np.isfinite(preparation[0]["physical_time_s"])
            or abs(preparation[0]["physical_time_s"] - end / 1000) > 1e-7):
        raise ValueError("Overlap labels end at a different preparation handoff")
    with np.load(reference / "replay/dual_arm_overlap_trace.npz", allow_pickle=False) as trace:
        ticks = np.arange(start, end, dtype="<i8")
        for name, dtype in (("tick", "<i8"), ("state_tick", "<i8"), ("time_s", "<f8"),
                            ("primary_phase", "<U64"), ("secondary_phase", "<U32"),
                            ("ownership_violation", "<i8")):
            if trace[name].shape != ticks.shape or trace[name].dtype != np.dtype(dtype):
                raise ValueError(f"Invalid native overlap trace: {name}")
        if (not np.array_equal(trace["tick"], ticks)
                or not np.array_equal(trace["state_tick"], ticks + 1)
                or not np.isfinite(trace["time_s"]).all()
                or np.max(np.abs(trace["time_s"] - (ticks + 1) / 1000)) > 1e-7
                or np.any(trace["ownership_violation"] != 0)):
            raise ValueError("Native overlap trace changed its recorded tick/ownership grid")
        for interval in intervals:
            selected = slice(interval["start_tick"] - start, interval["end_tick"] - start)
            for field in ("primary_phase", "secondary_phase"):
                if not np.all(trace[field][selected] == interval[field]):
                    raise ValueError(f"Interval and native trace disagree: {field}")
    return intervals, evidence


def validate_video_probe(data, count, canvas):
    """Exact rational timestamps, not a rounded fps/duration approximation."""
    streams = data["streams"]
    if len(streams) != 1:
        raise ValueError("Expected exactly one encoded review video stream")
    stream = streams[0]
    if ([int(stream["width"]), int(stream["height"])] != list(canvas)
            or int(stream["nb_frames"]) != count):
        raise ValueError("Encoded video changed frame count or native panel dimensions")
    if Fraction(stream["r_frame_rate"]) != FPS or Fraction(stream["avg_frame_rate"]) != FPS:
        raise ValueError("Encoded video changed the original 30fps presentation rate")
    pts = sorted(int(packet["pts"]) for packet in data["packets"])
    base = Fraction(stream["time_base"])
    if base <= 0 or len(pts) != count or any(value * base != Fraction(i, FPS) for i, value in enumerate(pts)):
        raise ValueError("Encoded PTS does not equal the complete original CFR30 grid")
    if abs(Fraction(stream["duration"]) - Fraction(count, FPS)) > Fraction(1, 100_000):
        raise ValueError("Encoded duration differs from original frame coverage")
    return {"frames": count, "fps": FPS, "frame_wh": list(canvas),
            "exact_rational_pts_match": True, "stream": stream}


def verify_video(video, count, canvas):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "2",
                    "-i", str(video), "-f", "null", "-"], check=True)
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,time_base,nb_frames,duration:packet=pts",
        "-of", "json", str(video)], text=True))
    return dict(validate_video_probe(data, count, canvas), full_decode_passed=True)
