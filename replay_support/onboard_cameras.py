"""C14 native camera and STL drawing leaves, with no replay dependencies.

Ported from the accepted C14 full_experiment_20261027/executed_sources.zip:
configure_camera/calibration/isolate_mesh_display_lists and MeshLists retain
render_onboard_review.py, export_v31_three_cameras.py and
render_mechanical_redesign.py conventions. The caller owns a private native
model/data pair and GL context; none of these helpers advances physics.

Meshes are the manifest's original metre-scale palm-local STL triangles, not
CAD millimetres, convex hulls or replacements for original scene occluders.
The head's inherited near clip is a documented rendering approximation, not
physical clearance. These visuals certify neither installation nor dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ["MUJOCO_GL"] in ("egl", "osmesa"):
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


IMAGE_SIZE_WH = (1280, 720)
CAMERA_NAMES = {
    "cam_head": "head_stereo_left",
    "cam_left_wrist": "left_wrist_rgbd",
    "cam_right_wrist": "right_wrist_rgbd",
}
FRAMES = {"left": "sharpa_left/left_hand_C_MC", "right": "sharpa_right/right_hand_C_MC"}
KINDS = {"printed", "mount_base", "reference_interface", "camera_envelope", "fastener"}


def configure_camera(model, spec):
    """Bind one native camera once; local look-at vectors never track targets."""
    camera = model.camera(spec["native_camera"]).id
    original = dict(body=model.body(int(model.cam_bodyid[camera])).name,
                    position_local_m=model.cam_pos[camera].tolist(),
                    quaternion_local_wxyz=model.cam_quat[camera].tolist(),
                    vertical_fov_deg=float(model.cam_fovy[camera]))
    mount = spec.get("mount")
    if mount:
        position = np.asarray(mount["position_local_m"], dtype=float)
        forward = np.asarray(mount["lookat_local_m"], dtype=float) - position
        up = np.asarray(mount["up_local"], dtype=float)
        if (position.shape != (3,) or forward.shape != (3,) or up.shape != (3,)
                or not np.isfinite(np.r_[position, forward, up]).all()
                or np.linalg.norm(forward) < 1e-9):
            raise ValueError("Invalid fixed mount direction")
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-8:
            raise ValueError("Mount up axis is parallel to its optical axis")
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        gl_rotation = np.column_stack([right, up, -forward])
        model.cam_bodyid[camera] = model.body(mount["body"]).id
        model.cam_mode[camera] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        model.cam_targetbodyid[camera] = -1
        model.cam_pos[camera] = position
        model.cam_quat[camera] = Rotation.from_matrix(gl_rotation).as_quat()[[3, 0, 1, 2]]
    if "vertical_fov_deg" in spec:
        model.cam_fovy[camera] = float(spec["vertical_fov_deg"])
    if (model.cam_mode[camera] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            or model.cam_targetbodyid[camera] != -1):
        raise ValueError("Onboard cameras must be fixed relative to their native mount, not tracking")
    if np.any(model.cam_sensorsize[camera]):
        raise ValueError("C14 expects native fovy optics, not physical sensor intrinsics")
    model.cam_ipd[camera] = 0.0
    return dict(id=spec["id"], native_camera=spec["native_camera"], original=original,
                mount_body=model.body(int(model.cam_bodyid[camera])).name,
                position_local_m=model.cam_pos[camera].tolist(),
                quaternion_camera_gl_in_mount_wxyz=model.cam_quat[camera].tolist(),
                vertical_fov_deg=float(model.cam_fovy[camera]),
                fixed_relative_to_mount=True, artificial_stereo_ipd_m=0.0)


def set_camera_near_clip(model, spec):
    """Set every view explicitly: the left wrist intentionally omits this key."""
    near = float(spec.get("near_clip_m", 0.003))
    if not np.isfinite(near) or near <= 0:
        raise ValueError("Camera near clip must be finite and positive")
    model.vis.map.znear = near / model.stat.extent
    return near


def calibration(scene, spec, width, height):
    """Read native frusta and OpenCV extrinsics after the scene camera update."""
    a, b = scene.camera
    if not np.allclose(a.pos, b.pos, rtol=0, atol=1e-8):
        raise RuntimeError("Monocular rendering requires coincident stereo cameras")
    position = np.array(a.pos, dtype=float)
    forward = np.array(a.forward, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.array(a.up, dtype=float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    world_from_cv = np.eye(4)
    world_from_cv[:3, :3] = np.column_stack([right, -up, forward])
    world_from_cv[:3, 3] = position
    near = float(a.frustum_near)
    fy = height * near / float(a.frustum_top - a.frustum_bottom)
    if abs(float(a.frustum_center)) > 1e-7 or abs(float(a.frustum_top + a.frustum_bottom)) > 1e-7:
        raise RuntimeError("Unexpected asymmetric native-camera frustum")
    K = np.array([[fy, 0.0, (width - 1) / 2], [0.0, fy, (height - 1) / 2], [0.0, 0.0, 1.0]])
    q = Rotation.from_matrix(world_from_cv[:3, :3]).as_quat()
    return dict(**spec, position_world_m=position.tolist(), forward_world=forward.tolist(),
                up_world=up.tolist(), world_from_camera_cv=world_from_cv.tolist(),
                camera_cv_from_world=np.linalg.inv(world_from_cv).tolist(),
                camera_cv_quaternion_world_wxyz=q[[3, 0, 1, 2]].tolist(), K=K.tolist(),
                image_size_wh=[width, height], distortion_coefficients=[0.0] * 5,
                near_m=near, far_m=float(a.frustum_far),
                camera_frame="OpenCV: +X right, +Y down, +Z forward",
                pixel_coordinates="Top-left pixel centre is (0,0); principal point ((W-1)/2,(H-1)/2)")


def native_calibration(model, data, scene, spec, width, height):
    camera = model.camera(spec["native_camera"]).id
    cal = calibration(scene, dict(id=spec["id"], native_camera=spec["native_camera"],
                      vertical_fov_deg=float(model.cam_fovy[camera])), width, height)
    if not np.allclose(cal["position_world_m"], data.cam_xpos[camera], rtol=0, atol=2e-7):
        raise RuntimeError("Native renderer did not follow the camera body")
    expected = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    if not np.allclose(np.asarray(cal["world_from_camera_cv"])[:3, :3], expected, rtol=0, atol=2e-7):
        raise RuntimeError("Native camera orientation does not match its body")
    return cal


@dataclass
class Part:
    spec: dict
    path: Path
    triangles: np.ndarray
    normals: np.ndarray
    bounds: np.ndarray
    digest: str

    @property
    def id(self):
        return self.spec["id"]


def load_assembly(manifest_path):
    """Load every supplied rendering part once, validating its declared bytes."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("units") != "m":
        raise ValueError("Expected assembly manifest schema_version=1 and units=m")
    if manifest.get("frames") != FRAMES:
        raise ValueError("Manifest frames must be the two specified anatomical palm bodies")
    if manifest.get("manufacturing_ready") is not False or not isinstance(manifest.get("limitations"), list):
        raise ValueError("Assembly must retain its unapproved manufacturing status and limitations")
    parts = []
    for entry in manifest["parts"]:
        if (not isinstance(entry["id"], str) or not entry["id"]
                or entry["side"] not in FRAMES or entry["kind"] not in KINDS):
            raise ValueError("Invalid part ID, side or kind in assembly manifest")
        relative = Path(entry["mesh"])
        if relative.is_absolute() or relative.suffix.lower() != ".stl":
            raise ValueError("Part meshes must be manifest-relative STL paths")
        path = (manifest_path.parent / relative).resolve()
        rgba = np.asarray(entry["rgba"], dtype=float)
        if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)):
            raise ValueError("Invalid RGBA for " + entry["id"])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["mesh_sha256"]:
            raise ValueError("Assembly mesh differs from manifest: " + str(path))
        mesh = trimesh.load_mesh(io.BytesIO(raw), file_type="stl", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("Expected nonempty triangulated STL for " + entry["id"])
        triangles = np.asarray(mesh.triangles, dtype=np.float32)
        if not np.isfinite(triangles).all():
            raise ValueError("Nonfinite STL vertices for " + entry["id"])
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        if np.any(lengths <= 1e-18):
            raise ValueError("Degenerate STL triangles; fix upstream CAD export: " + entry["id"])
        parts.append(Part(entry, path, triangles, cross / lengths[:, None], np.asarray(mesh.bounds), digest))
    if not parts or len({p.id for p in parts}) != len(parts):
        raise ValueError("Manifest parts must be nonempty and have unique IDs")
    for side in FRAMES:
        if not any(p.spec["side"] == side and p.spec["kind"] in ("printed", "mount_base") for p in parts):
            raise ValueError("Missing real mechanical part geometry for " + side)
        if not any(p.spec["side"] == side and p.spec["kind"] == "camera_envelope" for p in parts):
            raise ValueError("Missing supplied camera-envelope geometry for " + side)
    return manifest, parts


def palm_transforms(model, data):
    return {side: (data.xmat[model.body(body).id].reshape(3, 3).copy(),
                   data.xpos[model.body(body).id].copy()) for side, body in FRAMES.items()}


def isolate_mesh_display_lists(model, renderer):
    """Relocate overlapping MuJoCo mesh/hull slots before dynamic uploads.

    MuJoCo 3.6 may allocate builtins in a free hull slot; a later mesh upload
    deletes both slots. This is the accepted native-list isolation mechanism.
    """
    from OpenGL import GL
    context = renderer._mjr_context
    base, count = int(context.baseMesh), int(context.rangeMesh)
    other_ranges = [(int(getattr(context, start)), int(getattr(context, length)))
                    for start, length in (
                        ("basePlane", "rangePlane"), ("baseHField", "rangeHField"),
                        ("baseBuiltin", "rangeBuiltin"), ("baseFontNormal", "rangeFont"),
                        ("baseFontShadow", "rangeFont"), ("baseFontBig", "rangeFont"))]
    overlaps = any(max(base, start) < min(base + count, start + length) for start, length in other_ranges)
    result = dict(relocated=overlaps, original_mesh_base=base,
                  mesh_list_count=count, builtin_base=int(context.baseBuiltin))
    if not overlaps:
        return result
    new_base = int(GL.glGenLists(count))
    if not new_base or any(max(new_base, start) < min(new_base + count, start + length)
                           for start, length in other_ranges):
        raise RuntimeError("Could not allocate non-overlapping native mesh display lists")
    context.baseMesh = new_base
    for mesh_id in range(model.nmesh):
        mujoco.mjr_uploadMesh(model, context, mesh_id)
    for list_id in range(base, base + count):
        if not any(start <= list_id < start + length for start, length in other_ranges):
            GL.glDeleteLists(list_id, 1)
    if not GL.glIsList(context.baseBuiltin):
        raise RuntimeError("Native sphere display list was lost during mesh relocation")
    result["new_mesh_base"] = new_base
    return result


class MeshLists:
    """Context-local original STL lists, following the entire native list range.

    Each native mesh uses TWO slots (surface/hull). Wrapper lists preserve
    their original contents; liquid uploads replace only the native prefix.
    Keep both ranges reserved, then restore/free ours before renderer.close().
    """
    def __init__(self, model, renderer, parts):
        from OpenGL import GL
        self.GL = GL
        self.renderer = renderer
        self.context = renderer._mjr_context
        self.model = model
        self.old_base = int(self.context.baseMesh)
        self.old_count = int(self.context.rangeMesh)
        self.base = 0
        self.count = self.old_count + len(parts)
        self.original_gaps = []
        self.closed = False
        self.offsets = {p.id: self.old_count + i for i, p in enumerate(parts)}
        self.other_ranges = [(int(getattr(self.context, base)), int(getattr(self.context, count)))
                             for base, count in (
                                 ("basePlane", "rangePlane"), ("baseHField", "rangeHField"),
                                 ("baseBuiltin", "rangeBuiltin"), ("baseFontNormal", "rangeFont"),
                                 ("baseFontShadow", "rangeFont"), ("baseFontBig", "rangeFont"))]
        self.required_native = [2 * i + hull for i in range(model.nmesh)
                                for hull in (0, 1) if not hull or model.mesh_graphadr[i] >= 0]
        self.report = dict(original_mesh_base=self.old_base, native_list_count=self.old_count,
                           native_mesh_count=int(model.nmesh), added_list_count=len(parts),
                           ownership="expanded range plus originally empty native gap reservations",
                           native_surface_and_hull_slots_preserved=True,
                           later_liquid_uploads_use_expanded_native_prefix=bool(model.nmesh),
                           released_before_context_destruction=False)
        try:
            self.make_current()
            if self.old_count != 2 * model.nmesh:
                raise RuntimeError("Unsupported native mesh-list layout; expected two slots per mesh")
            for start, length in self.other_ranges:
                if max(start, self.old_base) < min(start + length, self.old_base + self.old_count):
                    raise RuntimeError("Native mesh isolation left a list-range overlap")
            # mjr_uploadMesh can delete an unused hull slot. Reserve those holes
            # so glGenLists cannot reuse any address in the original mesh range.
            for offset in range(self.old_count):
                list_id = self.old_base + offset
                if not GL.glIsList(list_id):
                    if offset in self.required_native:
                        raise RuntimeError("A required original native mesh/hull list is missing")
                    self.empty_list(list_id)
                    self.original_gaps.append(list_id)
            self.other_defined = [i for start, length in self.other_ranges
                                  for i in range(start, start + length) if GL.glIsList(i)]
            self.base = int(GL.glGenLists(self.count))
            if not self.base:
                raise RuntimeError("OpenGL could not reserve the complete expanded mesh range")
            protected = self.other_ranges + [(self.old_base, self.old_count)]
            if any(max(self.base, start) < min(self.base + self.count, start + length)
                   for start, length in protected):
                raise RuntimeError("Allocated mesh range overlaps a protected native range")
            for offset in range(self.old_count):
                GL.glNewList(self.base + offset, GL.GL_COMPILE)
                GL.glCallList(self.old_base + offset)
                GL.glEndList()
            for part in parts:
                GL.glNewList(self.base + self.offsets[part.id], GL.GL_COMPILE)
                try:
                    GL.glBegin(GL.GL_TRIANGLES)
                    try:
                        for normal, triangle in zip(part.normals, part.triangles):
                            GL.glNormal3fv(normal)
                            for vertex in triangle:
                                GL.glVertex3fv(vertex)
                    finally:
                        GL.glEnd()
                finally:
                    GL.glEndList()
            self.context.baseMesh = self.base
            self.context.rangeMesh = self.count
            self.report.update(expanded_mesh_base=self.base, expanded_list_count=self.count,
                               reserved_original_gap_count=len(self.original_gaps),
                               part_list_offsets=self.offsets,
                               preserved_other_defined_list_count=len(self.other_defined))
            self.after_uploads()
        except BaseException:
            self.close()
            raise

    def make_current(self):
        if self.renderer._gl_context is not None:
            self.renderer._gl_context.make_current()

    def empty_list(self, list_id):
        self.GL.glNewList(list_id, self.GL.GL_COMPILE)
        self.GL.glEndList()

    def after_uploads(self):
        self.make_current()
        if int(self.context.baseMesh) != self.base or int(self.context.rangeMesh) != self.count:
            raise RuntimeError("Expanded mesh range changed unexpectedly")
        required = ([self.base + i for i in self.required_native]
                    + [self.base + i for i in self.offsets.values()] + self.other_defined)
        if any(not self.GL.glIsList(i) for i in required):
            raise RuntimeError("A native mesh/hull, builtin, font or added mesh list was lost")
        for offset in range(self.old_count):
            if not self.GL.glIsList(self.base + offset):
                self.empty_list(self.base + offset)

    def append(self, parts, transforms):
        scene = self.renderer.scene
        if scene.ngeom + len(parts) > scene.maxgeom:
            raise RuntimeError("Scene capacity would truncate original occluders or new parts")
        for part in parts:
            rotation, position = transforms[part.spec["side"]]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_MESH,
                               np.maximum((part.bounds[1] - part.bounds[0]) / 2, 1e-9),
                               position, rotation.ravel(), np.asarray(part.spec["rgba"], np.float32))
            geom.dataid = self.offsets[part.id]
            geom.category = mujoco.mjtCatBit.mjCAT_DYNAMIC
            geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            geom.objid = -1
            geom.matid = -1
            geom.texcoord = 0
            geom.segid = scene.ngeom
            geom.specular = 0.18
            geom.shininess = 0.25
            corners = np.asarray(list(itertools.product(*part.bounds.T)))
            geom.modelrbound = float(np.linalg.norm(corners, axis=1).max())
            scene.ngeom += 1

    def close(self):
        if self.closed:
            return
        self.make_current()
        self.context.baseMesh = self.old_base
        self.context.rangeMesh = self.old_count
        if self.base:
            self.GL.glDeleteLists(self.base, self.count)
        for list_id in self.original_gaps:
            self.GL.glDeleteLists(list_id, 1)
        self.closed = True
        self.report["released_before_context_destruction"] = True
