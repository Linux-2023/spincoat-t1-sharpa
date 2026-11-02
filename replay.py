#!/usr/bin/env python3
"""Replay the immutable successful reference using a locally rebuilt compatible model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from model_identity import restore_recorded_display, verify_model
from replay_adapter import (NoPhysics, check_output, read_json, renderer_bindings,
                            require_versions, save_json, sha256, validate_overlap,
                            validate_stream, verify_files, verify_video)

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True,
                        help='Locally generated model.mjb from run.py; never a downloaded embedded model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    args = parser.parse_args()
    reference, model_path = args.reference.resolve(), args.model.resolve()
    lock_path = ROOT / 'replay_manifest.json'
    lock = read_json(lock_path)
    versions = require_versions(lock['runtime_versions'])
    files = verify_files(ROOT, lock['files'])
    reference_lock = read_json(ROOT / 'reference/manifest.json')
    reference_files = {name: row['sha256'] for name, row in reference_lock['files'].items()}
    verify_files(reference, reference_files)
    protected = [reference, model_path] + [ROOT / name for name in
                 ('native', 'native_fluid', 'replay_support', 'reference', 'assets')]
    output = check_output(args.output, protected)
    output.mkdir(parents=True)
    started = time.monotonic()
    initial_model_sha = sha256(model_path)
    report = dict(schema_version=1, status='running', mode=args.mode,
                  source_episode=lock['source_episode'], timing_label=lock['timing_label'],
                  source_native_duration_s=lock['terminal_t_ns'] / 1e9,
                  input_manifest_sha256=sha256(lock_path), model_sha256=initial_model_sha,
                  runtime_versions=versions, physics_steps_executed=0,
                  physics_step_attempts=0, strict_grip_certified=False,
                  hardware_certified=False, started_at_utc=datetime.now(timezone.utc).isoformat())
    scene, encoder = None, None
    try:
        import mujoco
        import numpy as np
        from PIL import ImageFont
        with NoPhysics(mujoco) as guard, renderer_bindings(ROOT, lock, files) as (renderer, exports, timeline):
            aligned = exports.load_render_stream(reference)
            count = validate_stream(aligned, lock)
            intervals, evidence = validate_overlap(reference, aligned, timeline, files, lock)
            model = mujoco.MjModel.from_binary_path(str(model_path))
            restored = restore_recorded_display(model, read_json(files[lock['recorded_display_state']]))
            report['recorded_display_geometries_restored'] = restored
            verify_model(model, read_json(files[lock['model_fingerprint']]))
            scene_lock = dict(lock, export_directory=str(reference))
            scene = renderer.RecordedScene(scene_lock, aligned, model=model)
            scene.overlap_intervals = intervals
            font = ImageFont.truetype(str(files[lock['font']]), 20)
            small = ImageFont.truetype(str(files[lock['font']]), 15)
            canvas = np.zeros((renderer.CANVAS[1], renderer.CANVAS[0], 3), dtype=np.uint8)
            indices = list(range(count)) if args.mode == 'full' else sorted(set(
                [0, count - 1] + [min(count - 1, int(seconds * 30)) for seconds in lock['review_times_s']]))
            video = output / 'C14_reference.mp4'
            with (output / 'encoder.log').open('xb') as log:
                encoder = subprocess.Popen([
                    'ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', f'{renderer.CANVAS[0]}x{renderer.CANVAS[1]}', '-r', '30', '-i', '-',
                    '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p',
                    '-threads', '2', '-movflags', '+faststart', str(video)],
                    stdin=subprocess.PIPE, stderr=log)
                with (output / 'frames.jsonl').open('x', encoding='utf-8') as rows:
                    for position, index in enumerate(indices):
                        probe = position == 0
                        save_root = output if args.mode == 'smoke' else None
                        image, row = renderer.frame_record(scene, index, position, canvas, font,
                                                           small, args.mode, save_root, probe=probe)
                        encoder.stdin.write(image.tobytes())
                        rows.write(json.dumps(row, allow_nan=False) + '\n')
                    encoder.stdin.close()
                    code = encoder.wait()
                    if code:
                        raise RuntimeError(f'Video encoder failed with exit code {code}')
                encoder = None
            scene.close()
            scene = None
            guard.check()
            video_result = verify_video(video, len(indices), renderer.CANVAS)
            if args.mode == 'full':
                video_result['source_action_video_verifier'] = exports.verify_video(video, reference)
            verify_files(reference, reference_files)
            verify_files(ROOT, lock['files'])
            if sha256(model_path) != initial_model_sha:
                raise RuntimeError('Source model changed during read-only replay')
            report.update(status='completed', frames_rendered=len(indices),
                          available_native_frames=count, selected_frame_indices=indices,
                          video_sha256=sha256(video), video_validation=video_result,
                          original_timeline_complete=args.mode == 'full',
                          all_cameras_share_exact_native_state=True,
                          native_pose_and_control_clock_preserved=True,
                          robot_object_pose_interpolation=False,
                          physical_model_arrays_unchanged=True,
                          source_files_unchanged=True, optical_probe_passed=True,
                          physics_step_attempts=guard.attempts,
                          secondary_activity_evidence=evidence)
    except BaseException as error:
        report.update(status='failed', exception=repr(error))
        raise
    finally:
        if encoder is not None:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.wait()
        if scene is not None:
            scene.close()
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        save_json(output / 'manifest.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'source_episode', 'frames_rendered',
                                                  'physics_steps_executed', 'peak_rss_bytes')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
