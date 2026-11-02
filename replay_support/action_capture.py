"""Read-only, identity-bound capture at the final native MuJoCo step boundary.

Keep all actual ctrl samples. The 30 Hz snapshot grid is a derived output grid,
not a change to the controller or physics timestep. No desired/state-as-action
reconstruction is permitted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

NS = 1_000_000_000
FPS = 30


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def frame_ticks(step_count, dt_ns, fps=FPS):
    """Past native ticks for presentation times k/fps strictly before termination."""
    if step_count <= 0 or dt_ns <= 0 or fps <= 0 or dt_ns * fps > NS:
        raise ValueError('Invalid native duration or output rate')
    count = (int(step_count) * int(dt_ns) * int(fps) + NS - 1) // NS
    frames = np.arange(count, dtype='<i8')
    return frames, frames * NS // (int(fps) * int(dt_ns))


class NativeActionCapture:
    def __init__(self, directory, *, fps=FPS, chunk_size=1000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.chunks_directory = self.directory / 'controls'
        self.chunks_directory.mkdir()
        if fps != FPS or chunk_size < 1:
            raise ValueError('This derived delivery uses 30 Hz and positive chunks')
        self.fps = fps
        self.chunk_size = int(chunk_size)
        self.model = None
        self.data = None
        self.steps = 0
        self.used = 0
        self.chunks = []
        self.frames = []
        self.next_frame_tick = 0
        self.finished = False
        self.excluded_calls = 0

    def bind(self, model, data):
        if self.model is not None:
            if self.model is not model or self.data is not data:
                raise RuntimeError('Live model/data identity changed during capture')
            return
        import mujoco
        if mujoco.get_mjcb_control() is not None:
            raise RuntimeError('An in-step control callback prevents pre-native ctrl certification')
        dt = float(model.opt.timestep)
        self.dt_ns = int(round(dt * NS))
        if self.dt_ns <= 0 or not np.isclose(dt * NS, self.dt_ns, rtol=0, atol=1e-5):
            raise ValueError('Native timestep is not an integer nanosecond period')
        if self.dt_ns * self.fps > NS:
            raise ValueError('Output rate exceeds native stepping rate')
        self.model, self.data = model, data
        self.ctrl_buffer = np.empty((self.chunk_size, model.nu), dtype='<f8')
        self.time_buffer = np.empty(self.chunk_size, dtype='<f8')
        self.actuator_names = [model.actuator(i).name for i in range(model.nu)]
        self.dimensions = {'nq': model.nq, 'nv': model.nv, 'nu': model.nu}
        self.start_data_time_s = float(data.time)
        save_json(self.directory / 'binding.json', {
            'model_identity': id(model), 'data_identity': id(data),
            'physics_dt_ns': self.dt_ns, 'output_fps': self.fps,
            'start_data_time_s': self.start_data_time_s,
            'actuator_names': self.actuator_names, **self.dimensions,
            'scope': 'Actual data.ctrl immediately before the innermost native step; no C control callback.'})

    def step(self, native_step, model, data, *args, **kwargs):
        if self.finished:
            raise RuntimeError('Capture already finalized')
        if self.model is not model or self.data is not data:
            self.excluded_calls += 1
            return native_step(model, data, *args, **kwargs)
        nstep = kwargs.get('nstep', args[0] if args else 1)
        if nstep != 1:
            raise RuntimeError('Live multi-step calls need an explicit per-step capture boundary')
        if not np.isfinite(data.ctrl).all():
            raise RuntimeError('Nonfinite actual native control')
        before = float(data.time)
        self.ctrl_buffer[self.used] = data.ctrl
        self.time_buffer[self.used] = before
        snapshot = None
        if self.steps == self.next_frame_tick:
            snapshot = (self.steps, before, data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
        result = native_step(model, data, *args, **kwargs)
        if not np.isclose(float(data.time) - before, self.dt_ns / NS, rtol=0, atol=1e-9):
            raise RuntimeError('Native call did not advance exactly one physics step')
        if snapshot is not None:
            self.frames.append(snapshot)
            self.next_frame_tick = len(self.frames) * NS // (self.fps * self.dt_ns)
        self.steps += 1
        self.used += 1
        if self.used == self.chunk_size:
            self._flush()
        return result

    def _flush(self):
        if not self.used:
            return
        first = self.steps - self.used
        ticks = np.arange(first, self.steps, dtype='<i8')
        path = self.chunks_directory / f'{len(self.chunks):06d}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, tick=ticks, t_ns=ticks * self.dt_ns,
                                data_time_s=self.time_buffer[:self.used], ctrl=self.ctrl_buffer[:self.used])
        self.chunks.append({'file': path.relative_to(self.directory).as_posix(),
                            'start_tick': first, 'stop_tick': self.steps,
                            'samples': self.used, 'sha256': sha256(path)})
        self.used = 0

    def finalize(self, native_scope, *, raw_directory, native_returned):
        if self.finished:
            raise RuntimeError('Capture may only be finalized once')
        self.finished = True
        if self.model is None or not self.steps:
            report = {'capture_complete': False, 'physics_steps': self.steps,
                      'reason': 'No bound live native steps', 'native_returned': bool(native_returned)}
            save_json(self.directory / 'manifest.json', report)
            return report
        self._flush()
        frame_index, expected_ticks = frame_ticks(self.steps, self.dt_ns, self.fps)
        observed_ticks = np.asarray([row[0] for row in self.frames], dtype='<i8')
        if not np.array_equal(observed_ticks, expected_ticks):
            raise RuntimeError('Native 30 Hz snapshot coverage is incomplete')
        t_ns = frame_index * NS // self.fps
        raw = Path(raw_directory)
        samples = self.directory / 'frame_samples.npz'
        with samples.open('xb') as stream:
            np.savez_compressed(stream, frame_index=frame_index, t_ns=t_ns,
                                source_tick=observed_ticks,
                                source_t_ns=observed_ticks * self.dt_ns,
                                data_time_s=np.asarray([row[1] for row in self.frames], dtype='<f8'),
                                qpos=np.asarray([row[2] for row in self.frames], dtype='<f8'),
                                qvel=np.asarray([row[3] for row in self.frames], dtype='<f8'),
                                ctrl=np.asarray([row[4] for row in self.frames], dtype='<f8'))
        scope_count = native_scope.get('live_record_count')
        scope_identity = bool(native_scope.get('live_state_bound'))
        scope_clock = bool(native_scope.get('live_physical_clock_strictly_increasing'))
        complete = scope_count == self.steps and scope_identity and scope_clock
        model_path = raw / 'model.mjb'
        result_path = raw / 'result.json'
        report = {
            'schema_version': 1, 'format': 'dag-native-control-recapture-v1',
            'capture_complete': complete, 'native_returned': bool(native_returned),
            'physics_steps': self.steps, 'physics_dt_ns': self.dt_ns,
            'terminal_t_ns': self.steps * self.dt_ns, 'output_fps': self.fps,
            'frame_count': len(self.frames), 'start_data_time_s': self.start_data_time_s,
            'final_data_time_s': float(self.data.time),
            'model_identity': id(self.model), 'data_identity': id(self.data),
            **self.dimensions, 'actuator_names': self.actuator_names,
            'chunks': self.chunks, 'frame_samples': samples.name,
            'frame_samples_sha256': sha256(samples),
            'raw_directory': str(raw), 'native_observer_scope': native_scope,
            'source_model_sha256': sha256(model_path) if model_path.is_file() else None,
            'source_result_sha256': sha256(result_path) if result_path.is_file() else None,
            'independent_native_calls_excluded': self.excluded_calls,
            'recorder_sha256': sha256(__file__),
            'control_semantics': 'Final native data.ctrl, all actuator columns, original float64 values; no requested/desired/state substitution.',
            'sample_clock': 'presentation k/30; source_tick=floor(k*1e9/(30*physics_dt_ns)); source precedes presentation by less than one native tick.',
            'state_semantics': 'Same bound native pre-step qpos/qvel and ctrl sampled together; no state interpolation.',
            'native_step_or_controller_settings_changed': False,
            'observation_only': True,
        }
        save_json(self.directory / 'manifest.json', report)
        if not complete:
            raise RuntimeError('Captured native controls disagree with the original live observer clock')
        return report
