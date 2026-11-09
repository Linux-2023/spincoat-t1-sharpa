"""Pinned live-camera drawing inputs and fail-closed identity/evidence helpers."""

from __future__ import annotations

import ast
import hashlib
import json
import numbers
from pathlib import Path

import numpy as np

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
WIDTH, HEIGHT = 1280, 720
LOCK_SHA256 = "45a698f61961d8b2788659d8fef4f9e386c92dc8fcd9d65996ffda884e00fa4a"
REPOSITORY = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(repr(array.shape).encode())
    if array.size:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def state_sha(value):
    """Private evidence digest, including array dtypes and shapes."""
    def encode(item):
        if isinstance(item, np.ndarray):
            return {"array_sha256": array_sha(item)}
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError("Unsupported private snapshot value: " + type(item).__name__)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=encode, allow_nan=False).encode()).hexdigest()


def verified_inputs(root=REPOSITORY):
    """Validate existing public leaves/inputs before importing any drawing code.

    Only mounts and near-clip values are read from the reference manifest;
    its historical episode paths are never opened. The lock pins all 36 STLs.
    """
    lock = Path(__file__).with_name("inputs.lock.json")
    if sha256(lock) != LOCK_SHA256:
        raise ValueError("C14 input lock identity mismatch")
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["identity"] != "benchmark-live-c14-v1":
        raise ValueError("Unsupported C14 input identity")
    root = Path(root).resolve()
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or Path(relative).is_absolute():
            raise ValueError("C14 input must remain repository-relative: " + relative)
        if sha256(path) != expected:
            raise ValueError("C14 input hash mismatch: " + relative)
    return manifest


def visual_leaves(path, functions, constants=(), **bindings):
    """Reuse named hash-verified leaves, never historical import/launcher graphs."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    namespace = {"np": np, "json": json, **bindings}
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    namespace[target.id] = ast.literal_eval(node.value)
                    found.append(target.id)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
    found.extend(node.name for node in nodes)
    if sorted(found) != sorted([*functions, *constants]):
        raise ValueError("Missing or duplicate drawing leaves: " + str(path))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def validate_mounts(actual, expected):
    if [item["id"] for item in actual] != list(CAMERAS) or len(expected) != 3:
        raise ValueError("C14 requires the exact three-camera order")
    for mount, reference in zip(actual, expected):
        for key in ("id", "native_camera", "mount_body", "position_local_m",
                    "vertical_fov_deg", "fixed_relative_to_mount", "artificial_stereo_ipd_m"):
            if mount[key] != reference[key]:
                raise ValueError("Accepted C14 local mount changed: " + key)
        a = np.asarray(mount["quaternion_camera_gl_in_mount_wxyz"])
        b = np.asarray(reference["quaternion_camera_gl_in_mount_wxyz"])
        if (a.shape != (4,) or not np.isfinite(a).all()
                or min(np.max(np.abs(a - b)), np.max(np.abs(a + b))) > 1e-12):
            raise ValueError("Accepted C14 local optical rotation changed")
        # The head must stay on the original native head body and local optics.
        if mount["id"] == "cam_head" and mount["original"] != reference["original"]:
            raise ValueError("Native head binding/optics changed")


def geometry_snapshot(scene, count):
    """Hash the native prefix before/after additions without filtering occluders."""
    digest = hashlib.sha256()
    for geom in scene.geoms[:count]:
        digest.update(np.asarray([int(geom.type), int(geom.objtype), int(geom.objid),
                                  int(geom.dataid), int(geom.matid), int(geom.category)],
                                 dtype="<i8").tobytes())
        for name in ("size", "pos", "mat", "rgba"):
            digest.update(np.asarray(getattr(geom, name)).tobytes())
    return digest.hexdigest()


def native_numeric_evidence(data):
    """All Python-exposed native numeric fields, including derived/contact caches.

    This complements mjSTATE_INTEGRATION; it is not a raw C-ABI checkpoint.
    Unallocated buffers retain their descriptors, not detached NumPy heap bytes.
    No forward, state setter, control callback or simulation operation occurs.
    """
    result = {}
    for prefix, owner in (("data", data), ("contact", data.contact), ("solver", data.solver)):
        values = {}
        for name in dir(owner):
            if name.startswith("_"):
                continue
            value = getattr(owner, name)
            if isinstance(value, np.ndarray):
                # Unallocated pointer buffers have detached storage in MuJoCo 3.6.
                # The inline maxuse_threadstack array is a real-value copy instead.
                if value.flags.owndata and (prefix, name) != ("data", "maxuse_threadstack"):
                    values[name] = {"native_storage": False, "dtype": value.dtype.str,
                                    "shape": list(value.shape)}
                else:
                    values[name] = array_sha(value)
            elif isinstance(value, numbers.Real):
                # repr also faithfully compares any unused infinite sentinels.
                values[name] = repr(value)
        result[prefix] = values
    return state_sha(result)
