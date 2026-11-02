"""Export complete recaptured controls and native states on one 30 Hz grid.

This is a derived delivery, not a DAG-VLA master or a changed execution policy.
Native states and controls share one past physics tick for every video frame.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import numpy as np

from action_capture import FPS, NS, frame_ticks, save_json, sha256


def export_actions(run_directory, output_directory):
    run = Path(run_directory).resolve()
    output = Path(output_directory).resolve()
    capture = run / 'action_capture'
    manifest_path = capture / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('capture_complete') is not True or manifest.get('output_fps') != FPS:
        raise ValueError('Incomplete capture or wrong derived output rate')
    metadata = json.loads((run / 'execution_metadata.json').read_text())
    raw = Path(metadata['paths']['raw_output']).resolve()
    if raw != Path(manifest['raw_directory']).resolve() or not raw.is_relative_to(run):
        raise ValueError('Control capture belongs to a different episode')
    result_path = raw / 'result.json'
    model_path = raw / 'model.mjb'
    if sha256(result_path) != manifest['source_result_sha256'] or sha256(model_path) != manifest['source_model_sha256']:
        raise ValueError('Native model/result changed after capture')
    if json.loads(result_path.read_text()).get('success') is not True:
        raise ValueError('This complete-workflow export requires a successful NEW native episode')
    samples_path = capture / manifest['frame_samples']
    if sha256(samples_path) != manifest['frame_samples_sha256']:
        raise ValueError('Native frame samples changed')
    with np.load(samples_path, allow_pickle=False) as archive:
        samples = {name: archive[name] for name in archive.files}
    count = int(manifest['physics_steps'])
    dt_ns = int(manifest['physics_dt_ns'])
    frames, ticks = frame_ticks(count, dt_ns)
    t_ns = frames * NS // FPS
    if not np.array_equal(samples['frame_index'], frames) or not np.array_equal(samples['source_tick'], ticks):
        raise ValueError('Captured frame sequence does not match the shared grid')
    if not np.array_equal(samples['t_ns'], t_ns) or not np.array_equal(samples['source_t_ns'], ticks * dt_ns):
        raise ValueError('Recorded frame timestamps disagree with their native ticks')
    if samples['data_time_s'].shape != (len(frames),) or not np.isfinite(samples['data_time_s']).all():
        raise ValueError('Invalid native frame clock')
    if np.max(np.abs(samples['data_time_s'] - ticks * dt_ns / NS)) > 1e-7:
        raise ValueError('Native frame clock differs from the bound physical-step clock')
    for name, width in [('ctrl', manifest['nu']), ('qpos', manifest['nq']), ('qvel', manifest['nv'])]:
        if samples[name].shape != (len(frames), width) or samples[name].dtype != np.dtype('<f8') or not np.isfinite(samples[name]).all():
            raise ValueError('Invalid actual frame field: ' + name)
    next_tick = 0
    verified_frames = 0
    for chunk in manifest['chunks']:
        path = capture / chunk['file']
        if not path.resolve().is_relative_to(capture) or sha256(path) != chunk['sha256']:
            raise ValueError('Native control chunk provenance failed')
        start, stop = int(chunk['start_tick']), int(chunk['stop_tick'])
        if start != next_tick or stop <= start or chunk['samples'] != stop - start:
            raise ValueError('Control chunks have a gap or overlap')
        with np.load(path, allow_pickle=False) as native:
            if not np.array_equal(native['tick'], np.arange(start, stop)):
                raise ValueError('Native control ticks are not contiguous')
            if not np.array_equal(native['t_ns'], native['tick'] * dt_ns):
                raise ValueError('Native control clock changed')
            if native['ctrl'].shape != (stop - start, manifest['nu']) or not np.isfinite(native['ctrl']).all():
                raise ValueError('Incomplete actual control matrix')
            selected = (ticks >= start) & (ticks < stop)
            if not np.array_equal(samples['ctrl'][selected], native['ctrl'][ticks[selected] - start]):
                raise ValueError('Derived actions differ from actual executed controls')
            verified_frames += int(selected.sum())
        next_tick = stop
    if next_tick != count or verified_frames != len(frames):
        raise ValueError('Native control coverage does not reach episode termination')
    terminal_ns = count * dt_ns
    duration_ns = np.minimum((frames + 1) * NS // FPS, terminal_ns) - t_ns
    lag_ns = t_ns - ticks * dt_ns
    if np.any(duration_ns <= 0) or np.any(lag_ns < 0) or np.any(lag_ns >= dt_ns):
        raise ValueError('Invalid final action interval or future control selection')
    output.mkdir(parents=True, exist_ok=True)
    actions_path = output / 'actions_30hz.npz'
    states_path = output / 'states_30hz.npz'
    with actions_path.open('xb') as stream:
        np.savez_compressed(stream, frame_index=frames, t_ns=t_ns,
                            source_tick=ticks, source_t_ns=ticks * dt_ns,
                            source_lag_ns=lag_ns, action=samples['ctrl'],
                            action_duration_ns=duration_ns,
                            actuator_names=np.asarray(manifest['actuator_names']))
    with states_path.open('xb') as stream:
        np.savez_compressed(stream, frame_index=frames, t_ns=t_ns,
                            source_tick=ticks, source_t_ns=ticks * dt_ns,
                            data_time_s=samples['data_time_s'],
                            qpos=samples['qpos'], qvel=samples['qvel'])
    report = {
        'schema_version': 1, 'format': 'dag-r03-derived-actions-30hz-v1',
        'case_id': metadata['case_id'], 'fps': FPS, 'frame_count': len(frames),
        'action_dim': manifest['nu'], 'action_dtype': '<f8',
        'actions': actions_path.name, 'states': states_path.name,
        'actions_sha256': sha256(actions_path), 'states_sha256': sha256(states_path),
        'source_capture_manifest': str(manifest_path),
        'source_capture_manifest_sha256': sha256(manifest_path),
        'source_model_sha256': manifest['source_model_sha256'],
        'source_result_sha256': manifest['source_result_sha256'],
        'source_execution_metadata_sha256': sha256(run / 'execution_metadata.json'),
        'source_physics_steps': count, 'source_physics_dt_ns': dt_ns,
        'terminal_t_ns': terminal_ns,
        'actuator_names': manifest['actuator_names'],
        'sample_time_definition': 'Frame k has presentation timestamp k/30; integer t_ns=floor(k*1e9/30). No terminal padding action.',
        'native_sample_definition': 'source_tick=floor(k*1e9/(30*physics_dt_ns)); actual pre-step state/control sampled together. No interpolation.',
        'maximum_native_sample_lag_ns': int(lag_ns.max()),
        'final_action_duration_ns': int(duration_ns[-1]),
        'native_controls_verified_bit_exact': True,
        'controls_complete_through_native_termination': True,
        'execution_rate_changed': False, 'physics_step_changed': False,
        'action_units': 'Original per-actuator native control units; not universally radians.',
        'action_projection': 'All native actuator columns in original model order; no learned action-space projection.',
        'video_scope': 'External display cameras, not robot policy input cameras.',
        'physical_acceptance': 'Functional/geometry conclusions must come from this new run; export validity does not certify stable grip or hardware safety.',
        'exporter_sha256': sha256(__file__),
    }
    save_json(output / 'action_manifest.json', report)
    return report


def load_render_stream(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'action_manifest.json').read_text())
    result = {}
    for field in ('actions', 'states'):
        path = directory / manifest[field]
        if sha256(path) != manifest[field + '_sha256']:
            raise ValueError('Aligned export file changed: ' + field)
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                if key in result and not np.array_equal(result[key], arrays[key]):
                    raise ValueError('Action/state frame alignment changed: ' + key)
                result[key] = arrays[key]
    if manifest['fps'] != FPS or len(result['frame_index']) != manifest['frame_count']:
        raise ValueError('Aligned render stream has the wrong rate/length')
    result['source_time_s'] = result['data_time_s']
    result['presentation_time_s'] = result['frame_index'].astype(float) / FPS
    result['terminal_time_s'] = manifest['terminal_t_ns'] / NS
    result['manifest'] = manifest
    return result


def verify_video(video_path, directory):
    stream = load_render_stream(directory)
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,time_base,nb_frames,duration,codec_name,pix_fmt:packet=pts',
        '-of', 'json', str(video_path)], text=True))
    video = data['streams'][0]
    count = len(stream['frame_index'])
    if Fraction(video['r_frame_rate']) != FPS or Fraction(video['avg_frame_rate']) != FPS or int(video['nb_frames']) != count:
        raise ValueError('Video rate/count differs from actions')
    pts = sorted(int(packet['pts']) for packet in data['packets'])
    base = Fraction(video['time_base'])
    if len(pts) != count or any(pts[k] * base.numerator * FPS != k * base.denominator for k in range(count)):
        raise ValueError('Video presentation timestamps differ from the action grid')
    return {'fps': FPS, 'frames': count, 'action_video_pts_match': True,
            'native_pose_and_control_share_source_tick': True,
            'stream': video, 'video_sha256': sha256(video_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_actions(args.run, args.out), ensure_ascii=False, indent=2))
