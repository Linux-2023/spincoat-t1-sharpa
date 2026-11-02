"""Three fixed, synchronized RGB observations of the unchanged V31 episode.
Reuses the original V31 liquid visuals. No new dynamics, hidden hands, HUD,
phase-dependent cameras, state interpolation, or success claims are introduced.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
from render_t1_neutral_multiview import PreparationReplay
from render_t1_human_workflow_v7 import implementation
from t1_gravity_liquid_render_v2 import GravityLiquidReplay, draw_fluid_with_palette
from t1_liquid_palette import apply_liquid_palette, LIQUID_RGBA
from t1_spincoat_scene import fluid_module

PREVIEW_TIMES = [0., 12., 20., 24., 38., 39.5, 40.4, 60., 82., 106., 120., 129.9, 130.221, 130.241, 130.261, 130.701, 134., 137.881]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def isolate_mesh_display_lists(model, renderer):
    """Keep dynamic mesh uploads from deleting unrelated native primitives.

    MuJoCo 3.6 deletes both mesh/hull lists but may leave the hull slot free.
    Its later builtins allocation can reuse that slot. Move the mesh block
    once, after all native lists exist, without changing any model geometry.
    """
    from OpenGL import GL
    context = renderer._mjr_context
    base, count = context.baseMesh, context.rangeMesh
    other_ranges = [(context.basePlane, context.rangePlane),
                    (context.baseHField, context.rangeHField),
                    (context.baseBuiltin, context.rangeBuiltin),
                    (context.baseFontNormal, context.rangeFont),
                    (context.baseFontShadow, context.rangeFont),
                    (context.baseFontBig, context.rangeFont)]
    overlaps = any(max(base, start) < min(base + count, start + length)
                   for start, length in other_ranges)
    result = dict(relocated=overlaps, original_mesh_base=base,
                  mesh_list_count=count, builtin_base=context.baseBuiltin)
    if not overlaps:
        return result
    new_base = int(GL.glGenLists(count))
    if not new_base or any(max(new_base, start) < min(new_base + count, start + length)
                           for start, length in other_ranges):
        raise RuntimeError('Could not allocate non-overlapping native mesh display lists')
    context.baseMesh = new_base
    for mesh_id in range(model.nmesh):
        mujoco.mjr_uploadMesh(model, context, mesh_id)
    for list_id in range(base, base + count):
        if not any(start <= list_id < start + length for start, length in other_ranges):
            GL.glDeleteLists(list_id, 1)
    if not GL.glIsList(context.baseBuiltin):
        raise RuntimeError('Native sphere display list was lost during mesh relocation')
    result['new_mesh_base'] = new_base
    return result


def make_camera(spec):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = spec['lookat_world_m']
    camera.distance = spec['distance_m']
    camera.azimuth = spec['azimuth_mujoco_deg']
    camera.elevation = spec['elevation_mujoco_deg']
    return camera


def calibration(scene, spec, width, height):
    a, b = scene.camera
    if not np.allclose(a.pos, b.pos, atol=1e-8):
        raise RuntimeError('Monocular rendering requires coincident stereo cameras')
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
        raise RuntimeError('Unexpected asymmetric free-camera frustum')
    K = np.array([[fy, 0., (width - 1) / 2], [0., fy, (height - 1) / 2], [0., 0., 1.]])
    q = Rotation.from_matrix(world_from_cv[:3, :3]).as_quat()
    return dict(**spec, position_world_m=position.tolist(), forward_world=forward.tolist(),
                up_world=up.tolist(), world_from_camera_cv=world_from_cv.tolist(),
                camera_cv_from_world=np.linalg.inv(world_from_cv).tolist(),
                camera_cv_quaternion_world_wxyz=q[[3, 0, 1, 2]].tolist(), K=K.tolist(),
                image_size_wh=[width, height], distortion_coefficients=[0., 0., 0., 0., 0.],
                near_m=near, far_m=float(a.frustum_far),
                camera_frame='OpenCV: +X right, +Y down, +Z forward',
                pixel_coordinates='Top-left pixel centre is (0,0); principal point ((W-1)/2,(H-1)/2)')


def project(points, cal):
    points = np.atleast_2d(points)
    T = np.asarray(cal['camera_cv_from_world'])
    xyz = points @ T[:3, :3].T + T[:3, 3]
    uvw = xyz @ np.asarray(cal['K']).T
    return uvw[:, :2] / uvw[:, 2:3], xyz[:, 2]


def check_projection(renderer, cal):
    """Render isolated optical targets; caller rebuilds the full RGB scene.

    The native camera and uploaded primitive lists are unchanged. This probe
    validates projection, not object visibility or scene-segmentation labels.
    """
    renderer.scene.ngeom = 0
    T = np.asarray(cal['world_from_camera_cv'])
    depth = max(2. * cal['near_m'], min(.7, .3 * cal['distance_m']))
    camera_points = depth * np.array([[0., 0., 1.], [.12, -.06, 1.], [-.09, .08, 1.]])
    points = camera_points @ T[:3, :3].T + T[:3, 3]
    for i, point in enumerate(points):
        slot = renderer.scene.ngeom
        g = renderer.scene.geoms[slot]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, .004 * depth), point, np.eye(3).ravel(), np.ones(4, dtype=np.float32))
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.objtype = mujoco.mjtObj.mjOBJ_SITE
        g.objid = 1000000 + i
        g.segid = slot
        renderer.scene.ngeom += 1
    renderer.enable_segmentation_rendering()
    try:
        image = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    expected, _ = project(points, cal)
    measured = []
    for i in range(3):
        mask = (image[:, :, 0] == 1000000 + i) & (image[:, :, 1] == mujoco.mjtObj.mjOBJ_SITE)
        y, x = np.nonzero(mask)
        if not len(x):
            raise RuntimeError('Optical marker not observed')
        measured.append([float(x.mean()), float(y.mean())])
    errors = np.linalg.norm(np.asarray(measured) - expected, axis=1)
    if errors.max() > 1.:
        raise RuntimeError('OpenCV calibration disagrees with native pixels: ' + repr(errors.tolist()))
    return dict(scope='isolated_optical_targets_after_native_visual_uploads', native_marker_projection_error_px=errors.tolist(), maximum_error_px=float(errors.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--fps', type=int, default=25)
    args = parser.parse_args()
    source = args.source.resolve()
    spec = json.loads(args.spec.read_text())
    if len(spec['cameras']) != 3 or len({c['id'] for c in spec['cameras']}) != 3:
        raise ValueError('Exactly three distinct cameras are required')
    if args.fps <= 0 or args.fps > 50:
        raise ValueError('Sampling must be within the original 50 Hz recording')
    labels, assembly_hash = implementation('interior_workflow_v6_provenance.json').verified_inputs(source, source/'audited_labels_v31.json')
    source_paths = [source/'episode.npz', source/'model.mjb', source/'result.json', source/'reagent_preparation.npz', source/'reagent_preparation/gravity_liquid_record.json']
    input_hashes = {str(p): sha(p) for p in source_paths}
    with np.load(source/'episode.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    times = arrays['time']
    if np.any(np.diff(times) <= 0):
        raise RuntimeError('Source timestamps are not strictly increasing')
    targets = np.asarray(PREVIEW_TIMES) if args.preview_only else np.minimum(np.arange(int(np.ceil(times[-1] * args.fps)) + 1) / args.fps, times[-1])
    nearest = np.searchsorted(times, targets).clip(1, len(times) - 1)
    selected = np.unique(np.where(abs(times[nearest - 1] - targets) <= abs(times[nearest] - targets), nearest - 1, nearest))
    previews = {int(np.argmin(abs(times[selected] - t))) for t in PREVIEW_TIMES}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'review').mkdir()
    for c in spec['cameras']:
        (out/'rgb'/c['id']).mkdir(parents=True)
    (out/'camera_spec.json').write_bytes(args.spec.read_bytes())
    m = mujoco.MjModel.from_binary_path(str(source/'model.mjb'))
    d = mujoco.MjData(m)
    prep = PreparationReplay(source, m)
    gravity = GravityLiquidReplay(source_paths[-1], prep.config)
    apply_liquid_palette(m)
    width, height = map(int, spec['image_size_wh'])
    m.vis.global_.offwidth = max(width, m.vis.global_.offwidth)
    m.vis.global_.offheight = max(height, m.vis.global_.offheight)
    m.vis.global_.ipd = 0.
    options = mujoco.MjvOption()
    options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    options.sitegroup[:] = 0
    options.frame = mujoco.mjtFrame.mjFRAME_NONE
    preserved = {key: getattr(m, key).copy() for key in ['light_pos', 'light_dir', 'light_diffuse', 'light_ambient', 'light_specular', 'mat_rgba', 'mat_shininess', 'mat_specular']}
    cameras = [make_camera(c) for c in spec['cameras']]
    target_id = m.site('spincoater/dispense_target').id
    mesh = m.mesh('spincoater/liquid_surface').id
    film_geom = m.geom('spincoater/liquid_visual').id
    first, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, m.mesh_quat[mesh])
    matrix = matrix.reshape(3, 3)
    mesh_local = m.mesh_vert[first:first + count] @ matrix.T + m.mesh_pos[mesh]
    top = mesh_local[:, 2] > -.00005
    draw_fluid = draw_fluid_with_palette(fluid_module('fluid_render').draw_fluid_v2)
    meta = json.loads(str(arrays['metadata_json']))
    released = next((float(e['time_s']) for e in meta.get('events', []) if e['phase'] == 'retract'), float('inf'))
    streams = []
    calibration_rows = []
    optical_checks = []
    sample_rows = []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    if not args.preview_only:
        for c in spec['cameras']:
            streams.append(subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}', '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/(c['id']+'.mp4'))], stdin=subprocess.PIPE))
    try:
        with mujoco.Renderer(m, height=height, width=width, max_geom=22000) as renderer, (out/'frames.jsonl').open('w') as index_file:
            display_list_layout = isolate_mesh_display_lists(m, renderer)
            for frame_id, j in enumerate(selected):
                time = float(times[j])
                d.qpos[:] = arrays['qpos'][j]
                d.qvel[:] = arrays['qvel'][j]
                d.time = time
                mujoco.mj_forward(m, d)
                prep.update(m, d, time)
                gravity.update(m, d, time, renderer)
                target = d.site_xpos[target_id].copy()
                h = arrays['film'][j]
                axis = np.linspace(-.018 + .018/h.shape[0], .018 - .018/h.shape[0], h.shape[0])
                heights = RegularGridInterpolator((axis, axis), h, bounds_error=False, fill_value=0)(mesh_local[top][:, [1, 0]])
                local = mesh_local.copy()
                local[top, 2] = heights - .00002 * np.clip(1 - heights/1e-6, 0, 1)
                m.mesh_vert[first:first + count] = (local - m.mesh_pos[mesh]) @ matrix
                m.geom_rgba[film_geom, 3] = .85 if h.max() > 1e-9 else 0.
                mujoco.mjr_uploadMesh(m, renderer._mjr_context, mesh)
                row = dict(frame_id=frame_id, source_frame_index=int(j), physical_time_s=time, camera_order=[c['id'] for c in spec['cameras']])
                images = []
                for ci, (c, camera) in enumerate(zip(spec['cameras'], cameras)):
                    m.vis.global_.fovy = c['vertical_fov_deg']
                    renderer.update_scene(d, camera=camera, scene_option=options)
                    cal = calibration(renderer.scene, c, width, height)
                    if frame_id == 0:
                        calibration_rows.append(cal)
                    elif not np.allclose(cal['world_from_camera_cv'], calibration_rows[ci]['world_from_camera_cv'], atol=1e-8) or not np.allclose(cal['K'], calibration_rows[ci]['K'], atol=1e-8):
                        raise RuntimeError('A fixed camera changed during the episode')
                    if frame_id in previews:
                        optical_checks.append(dict(camera=c['id'], physical_time_s=time, **check_projection(renderer, cal)))
                        renderer.update_scene(d, camera=camera, scene_option=options)
                    draw_fluid(renderer.scene, arrays, int(j), target, 0.)
                    gravity.draw(renderer.scene)
                    release = meta.get('release')
                    age = time - released
                    if release and 0 <= age <= release['flight_s']:
                        position = np.asarray(release['position_m']) + age*np.asarray(release['velocity_m_s'])
                        position[2] -= .5*9.81*age*age
                        g = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.ones(3)*(3*30e-9/(4*np.pi))**(1/3), position, np.eye(3).ravel(), np.asarray(LIQUID_RGBA, dtype=np.float32))
                        renderer.scene.ngeom += 1
                    rgb = renderer.render().copy()
                    if streams:
                        streams[ci].stdin.write(rgb.tobytes())
                    if frame_id in previews:
                        image = Image.fromarray(rgb)
                        relative = Path('rgb')/c['id']/f'frame_{frame_id:06d}.png'
                        image.save(out/relative)
                        row.setdefault('rgb_png', {})[c['id']] = relative.as_posix()
                        images.append(image)
                if not np.array_equal(d.qpos, arrays['qpos'][j]) or not np.array_equal(d.qvel, arrays['qvel'][j]):
                    raise RuntimeError('Visual replay altered recorded robot/object state')
                index_file.write(json.dumps(row)+'\n')
                if frame_id in previews:
                    sheet = Image.new('RGB', (1920, 384), (20, 24, 29))
                    draw = ImageDraw.Draw(sheet)
                    for ci, (image, c) in enumerate(zip(images, spec['cameras'])):
                        sheet.paste(image.resize((640, 360)), (640*ci, 24))
                        draw.text((640*ci + 5, 2), c['id'] + f' | actual t={time:.3f}s', fill='white', font=font)
                    sheet_name = f'review/three_views_{frame_id:06d}.png'
                    sheet.save(out/sheet_name)
                    sample_rows.append(dict(**row, phase_for_audit=str(arrays['phase'][j]), review_only_contact_sheet=sheet_name))
                if frame_id % 100 == 0:
                    print('THREE_CAMERAS', frame_id, len(selected), flush=True)
    finally:
        for stream in streams:
            stream.stdin.close()
        for stream in streams:
            if stream.wait() != 0:
                raise RuntimeError('Camera video encoding failed')
    video_checks = []
    for c in spec['cameras'] if streams else []:
        video = out/(c['id']+'.mp4')
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], capture_output=True, text=True)
        if decoded.returncode:
            raise RuntimeError(decoded.stderr)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration', '-of', 'json', str(video)], text=True))['streams'][0]
        if int(probe['nb_frames']) != len(selected) or [probe['width'], probe['height']] != [width, height]:
            raise RuntimeError('Three-camera stream alignment failed')
        video_checks.append(dict(camera=c['id'], file=video.name, sha256=sha(video), full_decode_passed=True, **probe))
    if any(sha(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError('Original episode, model or liquid evidence changed')
    if not all(np.array_equal(getattr(m, key), value) for key, value in preserved.items()):
        raise RuntimeError('Lighting or material arrays changed')
    (out/'camera_calibration.json').write_text(json.dumps(dict(world_frame=spec['world_frame'], cameras=calibration_rows, optical_checks=optical_checks), indent=2))
    (out/'samples.json').write_text(json.dumps(sample_rows, indent=2))
    manifest = dict(source_episode=source.name, mode='selected_actual_states' if args.preview_only else 'full_episode_rgb_streams', source_hashes=input_hashes, assembly_manifest_sha256=assembly_hash, renderer_sha256=sha(__file__), camera_spec_sha256=sha(args.spec), source_strict_hand_status=labels['strict_hand_status'], frame_count=len(selected), image_size_wh=[width, height], physical_time_range_s=[float(times[selected[0]]), float(times[selected[-1]])], video_fps=args.fps if streams else None, actual_timestamp_index='frames.jsonl', camera_calibration='camera_calibration.json', sample_images='samples.json', videos=video_checks, cameras_fixed_whole_episode=True, phase_conditioned_cameras=False, rgb_has_hud_or_gt_labels=False, sites_and_debug_frames_hidden=True, hands_or_occluders_removed=False, source_files_unchanged=True, qpos_or_qvel_interpolation=False, all_three_views_share_same_source_index=True, optical_checks=optical_checks, limitations=['Offline rendering of actual V31 saved states; not a new dynamics run or deployed sensor stream', 'V31 is functionally complete but its grasp geometry/stability is not accepted; do not label this as certified expert grasp data', 'Original reduced liquid/film visualization is retained; not CFD or calibrated material appearance', 'MP4 is H264/yuv420p compressed; sample PNGs retain renderer RGB. Use frames.jsonl physical timestamps, not video playback time, for alignment', 'Virtual pinhole calibration has no lens distortion, exposure/readout delay or real-camera noise; hardware installation needs physical calibration'])
    manifest['native_mesh_list_isolation'] = display_list_layout
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(output=str(out), frame_count=len(selected), cameras=calibration_rows, optical_checks=optical_checks), indent=2), flush=True)


if __name__ == '__main__':
    main()
