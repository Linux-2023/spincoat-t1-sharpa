

from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import REPOSITORY, visual_leaves


class LiveVisuals:
    """Current measured liquid/film, drawn only in the renderer-owned workspace.

    Drawing methods retain teacher_runtime/rendering.py:177-258 exactly.
    Source identity and scope are recorded in inputs.lock.json. The legacy
    archive-shaped argument is one live sample, never a saved episode.
    """

    def __init__(self, model, data):
        self.model, self.data = model, data
        palette = visual_leaves(REPOSITORY / 'native/t1_liquid_palette.py',
                                ('apply_liquid_palette',), ('LIQUID_RGBA',))
        self._liquid_rgba = np.asarray(palette['LIQUID_RGBA'], dtype=np.float32)
        palette['apply_liquid_palette'](model)
        gravity = visual_leaves(REPOSITORY / 'native/t1_gravity_liquid_render_v2.py',
                                ('topology', 'liquid_vertices'))
        self._liquid_vertices = gravity['liquid_vertices']
        self._draw_fluid = visual_leaves(REPOSITORY / 'native_fluid/fluid_render.py',
                                        ('_ellipsoid', 'draw_fluid_v2'), mujoco=mujoco)['draw_fluid_v2']
        self._gravity_geom = model.geom('gravity_liquid_v2_tube').id
        self._gravity_mesh = int(model.geom_dataid[self._gravity_geom])
        self._film_mesh = model.mesh('spincoater/liquid_surface').id
        self._film_geom = model.geom('spincoater/liquid_visual').id
        self._target_site = model.site('spincoater/dispense_target').id
        mesh = self._gravity_mesh
        vertices, faces = gravity['topology']()
        first, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        if (int(model.mesh_vertnum[mesh]) != len(vertices)
                or int(model.mesh_normalnum[mesh]) != len(vertices)
                or not np.array_equal(model.mesh_face[first:first + count], faces)):
            raise ValueError('Live gravity mesh topology differs from its pinned drawing leaf')
        if (model.geom_type[self._gravity_geom] != mujoco.mjtGeom.mjGEOM_MESH
                or model.geom_bodyid[self._gravity_geom] != 0):
            raise ValueError('Gravity liquid must be the original world-body visual mesh')
        first, count = int(model.mesh_vertadr[self._film_mesh]), int(model.mesh_vertnum[self._film_mesh])
        if count == 0:
            raise ValueError('Empty native film mesh')
        self._film_slice = slice(first, first + count)
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, model.mesh_quat[self._film_mesh])
        self._film_rotation = matrix.reshape(3, 3)
        self._film_local = (model.mesh_vert[self._film_slice] @ self._film_rotation.T
                            + model.mesh_pos[self._film_mesh])
        self._film_top = self._film_local[:, 2] > -0.00005
        if not self._film_top.any():
            raise ValueError('Native film mesh has no top surface')
        self._old_liquid_geoms = [i for i in range(model.ngeom) if any(
            token in (model.geom(i).name or '')
            for token in ('tube_liquid_proxy', 'tube_bottom_liquid_', 'pour_visual_'))]
        self._base_geoms = {key: getattr(model, key).copy() for key in (
            'geom_pos', 'geom_size', 'geom_quat', 'geom_sameframe', 'geom_rbound', 'geom_rgba')}

    def prepare(self, payload):
        for key, original in self._base_geoms.items():
            getattr(self.model, key)[:] = original
        cup = self.model.geom(payload['liquid']['config']['prefix'] + 'receiver_liquid_proxy_visual').id
        for geom in (self._gravity_geom, self._film_geom, cup, *self._old_liquid_geoms):
            if (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]
                    or np.any(self.model.pair_geom1 == geom) or np.any(self.model.pair_geom2 == geom)):
                raise ValueError('Liquid display must not replace native collision geometry')
        self._update_liquid(payload['liquid'])
        self.archive = self._update_film(payload['fluid'])
        self.gravity_parcels = payload['liquid']['state']['parcels']
        now = float(payload['fluid']['time_s'])
        self.flying = sum(p['birth_s'] <= now < p['hit_s'] for p in payload['fluid']['parcels'])
        releases = [payload['ballistic'], *payload.get('ballistic_parcels', [])]
        self.releases = [r for r in releases if r is not None and not r['landed']
                         and float(r['release_time_s']) <= self.data.time <= float(r['landing_time_s'])]

    def append(self, scene, reserved_parts):
        budget = 2 * self.flying + 32 + len(self.gravity_parcels) + len(self.releases) + reserved_parts
        if scene.ngeom + budget > scene.maxgeom:
            raise RuntimeError('Live liquid/assembly display exceeds scene capacity; nothing truncated')
        start = scene.ngeom
        target = self.data.site_xpos[self._target_site].copy()
        self._draw_fluid(scene, self.archive, 0, target, 0.0)
        for index in range(start, scene.ngeom):
            scene.geoms[index].rgba[:3] = self._liquid_rgba[:3]
        self._draw_gravity_parcels(scene, self.gravity_parcels)
        for release in self.releases:
            self._draw_release(scene, release)

    def _update_liquid(self, liquid: dict) -> None:
        m = self.model
        state, config = liquid["state"], SimpleNamespace(**liquid["config"])
        m.geom_rgba[self._old_liquid_geoms, 3] = 0.0
        gid, mesh = self._gravity_geom, self._gravity_mesh
        first, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        vertices, faces = self._liquid_vertices(state, config)
        if vertices.shape != (count, 3) or not np.isfinite(vertices).all():
            raise ValueError("Live gravity surface does not match the existing liquid mesh")
        m.geom_pos[gid] = 0.0
        m.geom_quat[gid] = [1.0, 0.0, 0.0, 0.0]
        m.geom_sameframe[gid] = 0
        m.geom_rbound[gid] = 3.0
        m.mesh_vert[first:first + count] = vertices
        face_vectors = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        normals = np.zeros_like(vertices)
        for column in range(3):
            np.add.at(normals, faces[:, column], face_vectors)
        normals /= np.maximum(1e-16, np.linalg.norm(normals, axis=1, keepdims=True))
        surface = -state["free_surface_plane_world"][3]
        normals[np.abs(vertices[:, 2] - surface) < 2e-7] = [0.0, 0.0, 1.0]
        normal_first, normal_count = int(m.mesh_normaladr[mesh]), int(m.mesh_normalnum[mesh])
        if normal_count != count:
            raise ValueError("Existing gravity mesh no longer has per-vertex normals")
        m.mesh_normal[normal_first:normal_first + count] = normals
        m.geom_rgba[gid] = self._liquid_rgba
        m.geom_rgba[gid, 3] = self._liquid_rgba[3] if state["tube_volume_m3"] > 1e-12 else 0.0
        cup = m.geom(config.prefix + "receiver_liquid_proxy_visual").id
        height = max(0.00001, state["receiver_volume_m3"] / (np.pi * config.receiver_inner_radius_m ** 2))
        m.geom_sameframe[cup] = 0
        m.geom_pos[cup, 2] = config.receiver_bottom_z_m + height / 2
        m.geom_size[cup, 1] = height / 2
        m.geom_rgba[cup] = self._liquid_rgba
        m.geom_rgba[cup, 3] = self._liquid_rgba[3] if state["receiver_volume_m3"] > 1e-12 else 0.0

    def _update_film(self, fluid: dict) -> dict:
        m = self.model
        h = np.asarray(fluid["h"], dtype=float)
        p = fluid["parameters"]
        resolution = int(p["resolution"])
        if h.shape != (resolution, resolution) or not np.isfinite(h).all() or np.any(h < 0):
            raise ValueError("Invalid live film grid")
        radius = float(p["radius_m"])
        axis = np.linspace(-radius + radius / resolution, radius - radius / resolution, resolution)
        heights = RegularGridInterpolator((axis, axis), h, bounds_error=False, fill_value=0)(self._film_local[self._film_top][:, [1, 0]])
        local = self._film_local.copy()
        local[self._film_top, 2] = heights - 0.00002 * np.clip(1 - heights / 1e-6, 0, 1)
        m.mesh_vert[self._film_slice] = (local - m.mesh_pos[self._film_mesh]) @ self._film_rotation
        m.geom_rgba[self._film_geom, 3] = 0.85 if h.max() > 1e-9 else 0.0
        # The existing draw leaf expects an archive-shaped view, but every value
        # here comes from this single live payload, with no interpolation/history.
        parcels = fluid["parcels"]
        archive = {"metadata_json": json.dumps({"film_parameters": p}, allow_nan=False), "drop_film_time_s": [float(fluid["time_s"])]}
        for key in ("birth_s", "hit_s", "volume_m3", "position_m", "velocity_m_s", "hit_position_m", "kind"):
            archive["fluid_parcel_" + key] = np.asarray([parcel[key] for parcel in parcels])
        return archive

    def _draw_gravity_parcels(self, scene, parcels: list) -> None:
        positions = np.asarray([p["position_m"] for p in parcels])
        for index, parcel in enumerate(parcels):
            radius = max(0.00035, (3 * parcel["volume_m3"] / (4 * np.pi)) ** (1 / 3))
            position = positions[index]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, [radius] * 3, position, np.eye(3).ravel(), self._liquid_rgba)
            if index + 1 < len(parcels) and np.linalg.norm(positions[index + 1] - position) < 0.008:
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, position, positions[index + 1])
            scene.ngeom += 1

    def _draw_release(self, scene, release: dict | None) -> None:
        if release is None or release["landed"]:
            return
        age = self.data.time - float(release["release_time_s"])
        if age < 0 or self.data.time > float(release["landing_time_s"]):
            return
        volume = float(release["volume_m3"])
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("A live released droplet must have a measured positive volume")
        position = np.asarray(release["release_position"], dtype=float) + age * np.asarray(release["release_velocity"], dtype=float)
        position[2] -= 0.5 * float(release["gravity_m_s2"]) * age * age
        radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [radius] * 3, position, np.eye(3).ravel(), self._liquid_rgba)
        scene.ngeom += 1
