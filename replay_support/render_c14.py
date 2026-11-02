"""Original C14 saved-state drawing definitions; no legacy launcher or physics loop."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 720
CANVAS = (3840, 800)
FPS = 30

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def definitions(path, names):
    """Reuse exact trusted drawing definitions without executing legacy launchers."""
    parsed = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    nodes = [node for node in parsed.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(nodes) != len(names) or {node.name for node in nodes} != set(names):
        raise RuntimeError('Expected unique recorded visual definitions: ' + str(names))
    namespace = dict(__name__='r08_c14_saved_visuals', np=np, mujoco=mujoco,
                     Path=Path, json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def state_hashes(data):
    return dict(qpos=array_sha(data.qpos), qvel=array_sha(data.qvel),
                ctrl=array_sha(data.ctrl), data_time_s=float(data.time))


def geometry_snapshot(scene, count):
    digest = hashlib.sha256()
    for geom in scene.geoms[:count]:
        digest.update(np.asarray([int(geom.type), int(geom.objtype), int(geom.objid),
                                  int(geom.dataid), int(geom.matid), int(geom.category)],
                                 dtype='<i8').tobytes())
        for name in ('size', 'pos', 'mat', 'rgba'):
            digest.update(np.asarray(getattr(geom, name)).tobytes())
    return digest.hexdigest()


class RecordedScene:
    def __init__(self, lock, aligned, *, model):
        self.lock, self.aligned = lock, aligned
        self.source = Path(lock['export_directory']) / 'replay'
        with np.load(self.source / 'episode.npz', allow_pickle=False) as saved:
            self.arrays = {key: saved[key] for key in saved.files}
        self.times = self.arrays['time']
        if not np.all(np.diff(self.times) > 0):
            raise RuntimeError('Recorded display timeline is not increasing')
        self.meta = json.loads(str(self.arrays['metadata_json']))
        self.model = model
        self.data = mujoco.MjData(self.model)
        model = self.model
        if model.nmocap or model.na:
            raise RuntimeError('This native frame contract lacks variable mocap/activation state')
        if (aligned['qpos'].shape != (lock['frame_count'], model.nq)
                or aligned['qvel'].shape != (lock['frame_count'], model.nv)
                or aligned['action'].shape != (lock['frame_count'], model.nu)):
            raise RuntimeError('Native model/frame dimensions disagree')
        if abs(float(model.opt.timestep) - .001) > 1e-12:
            raise RuntimeError('Expected recorded R08 1 ms physical model')
        self.physical = {name: getattr(model, name).copy() for name in (
            'body_mass', 'body_inertia', 'dof_damping', 'dof_frictionloss', 'jnt_range',
            'geom_friction', 'geom_contype', 'geom_conaffinity', 'actuator_gainprm',
            'actuator_biasprm', 'actuator_forcerange')}
        self.colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self.collision_geometry = {name: getattr(model, name)[self.colliding].copy()
                                   for name in ('geom_size', 'geom_pos', 'geom_quat')}
        self.spec_path = HERE / lock['camera_spec']
        self.specs = json.loads(self.spec_path.read_text())['cameras']
        if [s['id'] for s in self.specs] != ['cam_head', 'cam_left_wrist', 'cam_right_wrist']:
            raise RuntimeError('Camera order changed')
        self.mounts = [onboard.configure_camera(model, spec) for spec in self.specs]
        reference = json.loads((HERE / lock['mount_reference']).read_text())['mounts']
        for actual, expected in zip(self.mounts, reference):
            for key in ('id', 'native_camera', 'mount_body', 'position_local_m', 'vertical_fov_deg'):
                if actual[key] != expected[key]:
                    raise RuntimeError('Accepted C14 local mount changed: ' + key)
            a = np.asarray(actual['quaternion_camera_gl_in_mount_wxyz'])
            b = np.asarray(expected['quaternion_camera_gl_in_mount_wxyz'])
            if min(np.max(np.abs(a-b)), np.max(np.abs(a+b))) > 1e-12:
                raise RuntimeError('Accepted C14 local optical rotation changed')
        self.camera_arrays = {name: getattr(model, name).copy() for name in (
            'cam_bodyid', 'cam_mode', 'cam_targetbodyid', 'cam_pos', 'cam_quat',
            'cam_fovy', 'cam_ipd', 'cam_sensorsize')}
        assembly_path = HERE / lock['assembly_manifest']
        self.assembly, self.parts = onboard.load_assembly(assembly_path)
        if len(self.parts) != 36 or sha(assembly_path.parent / self.assembly['camera_spec']) != sha(self.spec_path):
            raise RuntimeError('C14 assembly/spec identity differs')
        prep = definitions(HERE / 'references/executed_renderer_v7.py', ('PreparationReplay',))
        self.prep = prep['PreparationReplay'](self.source, model)
        # The legacy zero-flow branch hides markers without resetting positions.
        # Restore only these noncolliding display markers before each saved frame.
        self.flow_marker_ids = np.asarray([
            model.geom(self.prep.config.prefix+'pour_visual_'+str(i)).id for i in range(12)])
        if (np.any(model.geom_contype[self.flow_marker_ids])
                or np.any(model.geom_conaffinity[self.flow_marker_ids])
                or np.any(np.isin(model.pair_geom1, self.flow_marker_ids))
                or np.any(np.isin(model.pair_geom2, self.flow_marker_ids))):
            raise RuntimeError('Recorded flow markers must remain display-only')
        self.flow_marker_positions = model.geom_pos[self.flow_marker_ids]
        self.gravity = GravityLiquidReplay(self.source / 'reagent_preparation/gravity_liquid_record.json',
                                          self.prep.config)
        apply_liquid_palette(model)
        self.lighting = {name: getattr(model, name).copy() for name in (
            'light_pos', 'light_dir', 'light_diffuse', 'light_ambient', 'light_specular',
            'mat_rgba', 'mat_shininess', 'mat_specular')}
        model.vis.global_.offwidth = max(WIDTH, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(HEIGHT, model.vis.global_.offheight)
        model.vis.global_.ipd = 0.
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.options.sitegroup[:] = 0
        self.options.frame = mujoco.mjtFrame.mjFRAME_NONE
        self.mesh = model.mesh('spincoater/liquid_surface').id
        self.film_geom = model.geom('spincoater/liquid_visual').id
        self.first = int(model.mesh_vertadr[self.mesh])
        self.count = int(model.mesh_vertnum[self.mesh])
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, model.mesh_quat[self.mesh])
        self.matrix = matrix.reshape(3, 3)
        self.mesh_local = model.mesh_vert[self.first:self.first+self.count] @ self.matrix.T + model.mesh_pos[self.mesh]
        self.top = self.mesh_local[:, 2] > -.00005
        shape = self.arrays['film'].shape[1:]
        if shape[0] != shape[1]:
            raise RuntimeError('Recorded film grid is not the expected square grid')
        self.film_axis = np.linspace(-.018+.018/shape[0], .018-.018/shape[0], shape[0])
        self.target_id = model.site('spincoater/dispense_target').id
        self.draw_fluid = draw_fluid_with_palette(fluid_module('fluid_render').draw_fluid_v2)
        self.released = next((float(e['time_s']) for e in self.meta.get('events', [])
                              if e['phase'] == 'retract'), float('inf'))
        self.projection = definitions(HERE / 'references/projection_reference.py',
                                      ('project', 'check_projection'))
        self.renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT, max_geom=22000)
        self.mesh_lists = None
        try:
            self.isolation = onboard.isolate_mesh_display_lists(model, self.renderer)
            self.mesh_lists = onboard.MeshLists(model, self.renderer, self.parts)
        except BaseException:
            self.renderer.close()
            raise

    def prepare(self, index):
        m, d = self.model, self.data
        self.index = int(index)
        self.time = float(self.aligned['source_time_s'][index])
        d.qpos[:] = self.aligned['qpos'][index]
        d.qvel[:] = self.aligned['qvel'][index]
        d.ctrl[:] = self.aligned['action'][index]
        d.time = self.time
        m.geom_pos[self.flow_marker_ids] = self.flow_marker_positions
        mujoco.mj_forward(m, d)
        self.expected_state = state_hashes(d)
        self.prep.update(m, d, self.time)
        self.gravity.update(m, d, self.time, self.renderer)
        self.j = int(np.clip(np.searchsorted(self.times, self.time, side='right')-1, 0, len(self.times)-1))
        k = min(self.j+1, len(self.times)-1)
        dt = float(self.times[k]-self.times[self.j])
        self.blend = float(np.clip((self.time-self.times[self.j])/dt, 0, 1)) if dt > 0 else 0.
        h = (1-self.blend)*self.arrays['film'][self.j] + self.blend*self.arrays['film'][k]
        heights = RegularGridInterpolator((self.film_axis, self.film_axis), h,
                    bounds_error=False, fill_value=0)(self.mesh_local[self.top][:, [1, 0]])
        local = self.mesh_local.copy()
        local[self.top, 2] = heights-.00002*np.clip(1-heights/1e-6, 0, 1)
        m.mesh_vert[self.first:self.first+self.count] = (local-m.mesh_pos[self.mesh]) @ self.matrix
        m.geom_rgba[self.film_geom, 3] = .85 if h.max() > 1e-9 else 0.
        mujoco.mjr_uploadMesh(m, self.renderer._mjr_context, self.mesh)
        self.mesh_lists.after_uploads()
        self.transforms = onboard.palm_transforms(m, d)
        self.check_state()

    def check_state(self):
        if state_hashes(self.data) != self.expected_state:
            raise RuntimeError('A visual operation changed native qpos/qvel/ctrl/time')
        for name, expected in self.camera_arrays.items():
            if not np.array_equal(getattr(self.model, name), expected):
                raise RuntimeError('C14 camera arrays changed during replay: ' + name)

    def optical_probe(self, cal):
        renderer = self.renderer
        proof = self.projection['check_projection'](renderer, dict(cal, distance_m=.4))
        depth_z = max(2*cal['near_m'], .12)
        camera_points = depth_z*np.array([[0.,0.,1.],[.12,-.06,1.],[-.09,.08,1.]])
        transform = np.asarray(cal['world_from_camera_cv'])
        world = camera_points @ transform[:3,:3].T + transform[:3,3]
        uv, expected = self.projection['project'](world, cal)
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        if not np.isfinite(depth).all():
            raise RuntimeError('Nonfinite native optical-probe depth')
        pixels = np.rint(uv).astype(int)
        measured = depth[pixels[:,1], pixels[:,0]]
        errors = np.abs(measured-expected)
        if np.any(errors > .004*expected+3e-4):
            raise RuntimeError('Native optical depth/projection mismatch: ' + repr(errors.tolist()))
        return dict(proof, depth_error_m=errors.tolist(),
                    depth_tolerance_m=(.004*expected+3e-4).tolist(),
                    scope='Isolated optical targets only; full scene rebuilt before delivered RGB')

    def render_view(self, spec, probe=False, preserve_check=False):
        m, d, renderer = self.model, self.data, self.renderer
        near = onboard.set_camera_near_clip(m, spec)
        renderer.update_scene(d, camera=spec['native_camera'], scene_option=self.options)
        cal = onboard.native_calibration(m, d, renderer.scene, spec, WIDTH, HEIGHT)
        proof = self.optical_probe(cal) if probe else None
        if probe:
            renderer.update_scene(d, camera=spec['native_camera'], scene_option=self.options)
            cal = onboard.native_calibration(m, d, renderer.scene, spec, WIDTH, HEIGHT)
        self.draw_fluid(renderer.scene, self.arrays, self.j,
                        d.site_xpos[self.target_id].copy(), self.blend)
        self.gravity.draw(renderer.scene)
        release = self.meta.get('release')
        age = self.time-self.released
        if release and 0 <= age <= release['flight_s']:
            position = np.asarray(release['position_m'])+age*np.asarray(release['velocity_m_s'])
            position[2] -= .5*9.81*age*age
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.ones(3)*(3*30e-9/(4*np.pi))**(1/3), position, np.eye(3).ravel(),
                np.asarray(LIQUID_RGBA, dtype=np.float32))
            renderer.scene.ngeom += 1
        count = int(renderer.scene.ngeom)
        before = geometry_snapshot(renderer.scene, count) if preserve_check else None
        self.mesh_lists.append(self.parts, self.transforms)
        if int(renderer.scene.ngeom) != count+len(self.parts):
            raise RuntimeError('C14 append changed the native scene count')
        if preserve_check and geometry_snapshot(renderer.scene, count) != before:
            raise RuntimeError('C14 append replaced native geometry or occluders')
        full_geometry = geometry_snapshot(renderer.scene, int(renderer.scene.ngeom)) if preserve_check else None
        render_flags = renderer.scene.flags.astype(bool).tolist()
        rgb = renderer.render().copy()
        if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8:
            raise RuntimeError('Unexpected native RGB buffer contract')
        self.check_state()
        return rgb, dict(camera=spec['id'], calibration=cal, configured_near_m=near,
                         native_scene_geom_count=count, added_part_geom_count=len(self.parts),
                         native_geometry_prefix_checked=preserve_check,
                         full_scene_geometry_sha256=full_geometry, render_flags=render_flags,
                         native_geometry_prefix_sha256=before, rgb_sha256=array_sha(rgb),
                         optical_probe=proof)

    def visual_signature(self):
        """Exact saved drawing inputs, not a live integration checkpoint."""
        return dict(state=state_hashes(self.data),
            model={name: array_sha(getattr(self.model, name)) for name in (
                'geom_rgba', 'geom_size', 'geom_pos', 'geom_quat', 'mesh_vert',
                'mesh_normal', 'mat_rgba', 'light_pos', 'light_dir', 'cam_pos', 'cam_quat')},
            derived={name: array_sha(getattr(self.data, name)) for name in (
                'xpos', 'xmat', 'cam_xpos', 'cam_xmat', 'geom_xpos', 'geom_xmat',
                'light_xpos', 'light_xdir')})

    def close(self):
        try:
            if self.mesh_lists is not None:
                self.mesh_lists.close()
        finally:
            self.renderer.close()
        for name, expected in self.physical.items():
            if not np.array_equal(getattr(self.model, name), expected):
                raise RuntimeError('Physical model array changed: ' + name)
        for name, expected in self.collision_geometry.items():
            if not np.array_equal(getattr(self.model, name)[self.colliding], expected):
                raise RuntimeError('Collision geometry changed: ' + name)
        for name, expected in self.lighting.items():
            if not np.array_equal(getattr(self.model, name), expected):
                raise RuntimeError('Recorded lighting/material arrays changed: ' + name)


def frame_record(scene, index, video_index, canvas, font, small, mode, save_root=None, probe=False):
    scene.prepare(index)
    canvas[:] = (18, 23, 29)
    records = []
    for col, spec in enumerate(scene.specs):
        rgb, cal = scene.render_view(spec, probe=probe, preserve_check=probe)
        canvas[40:760, col*WIDTH:(col+1)*WIDTH] = rgb
        if save_root is not None:
            folder = save_root/'onboard'/spec['id']
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/f'frame_{index:06d}.png'
            Image.fromarray(rgb).save(path)
            cal['png'] = str(path.relative_to(save_root))
            cal['png_sha256'] = sha(path)
        records.append(cal)
    frame = Image.fromarray(canvas)
    draw = ImageDraw.Draw(frame)
    for col, spec in enumerate(scene.specs):
        draw.text((col*WIDTH+12, 7), f"{spec['id']} | {spec['native_camera']} | native t={scene.time:.3f}s",
                  font=font, fill=(235,242,247))
    phase = str(scene.arrays['phase'][scene.j])
    source_stage = str(scene.arrays['source_stage'][scene.j])
    source_tick = int(scene.aligned['source_tick'][index])
    activity = phase_at_tick(scene.overlap_intervals, source_tick)
    secondary = {name: activity[name] if activity is not None else None for name in
                 ('secondary_stage', 'secondary_phase')}
    secondary.update(overlap_primary_stage=activity['primary_stage'] if activity is not None else None,
                     overlap_primary_phase=activity['primary_phase'] if activity is not None else None,
                     secondary_start_tick=activity['start_tick'] if activity is not None else None,
                     secondary_end_tick=activity['end_tick'] if activity is not None else None)
    lanes = f'PRIMARY {source_stage}/{phase}'
    if activity is not None:
        if source_stage != activity['primary_stage']:
            raise RuntimeError('Secondary activity is outside its original primary stage')
        lanes = (f"LEFT {activity['primary_stage']}/{activity['primary_phase']} | "
                 f"RIGHT {activity['secondary_stage']}/{activity['secondary_phase']}")
    prefix = 'SMOKE / SELECTED STATES | ' if mode == 'smoke' else ''
    draw.text((12, 762), prefix+scene.lock['display_label']+' | C14 | recorded-time / 30fps | '+lanes,
              font=small, fill=(238,201,151))
    draw.text((12, 781),
              'SAVED-STATE REPLAY; NO NEW PHYSICS | THIS EPISODE FUNCTIONAL COMPLETE; GEOMETRY / GRIP NOT CERTIFIED | C14 HARDWARE NOT CERTIFIED',
              font=small, fill=(238,201,151))
    if save_root is not None:
        image_path = save_root/f'three_views_{index:06d}.png'
        frame.save(image_path)
    row = dict(video_frame_index=video_index, source_frame_index=int(index),
               source_tick=int(scene.aligned['source_tick'][index]),
               source_t_ns=int(scene.aligned['source_t_ns'][index]),
               source_data_time_s=scene.time, original_presentation_t_ns=int(scene.aligned['t_ns'][index]),
               original_action_duration_ns=int(scene.aligned['action_duration_ns'][index]),
               phase=phase, source_stage=source_stage, **secondary,
               all_cameras_share_exact_native_state=True, **scene.expected_state, cameras=records)
    return frame, row
