# T1 + Sharpa spin-coating reproduction

A self-contained reproduction package for the current scripted, dual-arm laboratory workflow: take and mix a test tube, pour into a shared beaker, acquire and fill a pipette, approach the spin-coater, visually center, dispense, spin-coat, and return the tube.

The default simulator retains the latest wall-approach / disk-edge-then-center planner, with **every motion-reference acceleration restored to 1× source time**. It does not slow a recorded video. Physics remains at its original 1 ms timestep, with the existing control, collision, liquid and acceptance logic.

This is scripted simulation, not a trained VLA policy or a hardware result. Functional completion does **not** imply accepted strict grasp/contact geometry, repeatability across layouts, calibrated liquid physics, or certified camera hardware.

## Interactive three-camera benchmark

`benchmark.py` serves a fresh, evaluator-owned world with current C14 head/left-wrist/right-wrist observations and a language-neutral actuator API. It does not prescribe or launch a participant agent workflow.

See [BENCHMARK.md](BENCHMARK.md) for installation, isolation, observation/action contracts, task criteria and complete-attempt scoring. [benchmark_verification.json](benchmark_verification.json) records the exercised native-step/HTTP/optical smoke and its limits. This benchmark has a separate measured-torque apparatus/judge identity, not historical all35 equivalence or a demonstrated full-task policy success.

The existing `run.py` and `replay.py` reproduction entrypoints and their reference evidence remain unchanged.

## Requirements

- Linux x86-64, Python 3.11, Git, FFmpeg/FFprobe.
- EGL/OpenGL support. An NVIDIA EGL driver or Mesa software EGL can be used; software rendering is slower.
- Allow 8 GiB RAM for one simulator process and several GiB of disk for prepared assets and independent run outputs. Do not run multiple copies without budgeting their combined memory.
- Internet access for the pinned upstream assets and the optional successful-reference archive.

On Debian/Ubuntu, the relevant system packages are `git`, `ffmpeg`, `libegl1`, `libopengl0`, and, for software rendering, `libgl1-mesa-dri`. Create an isolated Python environment; do not change an existing simulation environment.

```bash
git clone https://github.com/Linux-2023/spincoat-t1-sharpa.git
cd spincoat-t1-sharpa
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Rebuild the assets

```bash
MUJOCO_GL=egl .venv/bin/python prepare.py --assets .cache/runtime_inputs
```

`assets.lock.json` pins upstream commits, URLs, sizes and SHA256 digests. Robot and pipette models are acquired from their original repositories and the robot XML is rebuilt locally. No embedded third-party robot `model.mjb` is distributed. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and license boundaries.

An existing mismatched asset fails verification rather than being silently reused. Prepared inputs are read-only sources for each run; each run receives its own materialized workspace. `prepared.json` binds the completed preparation to the exact asset manifest.

## Run the original-time planner

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 \
  .venv/bin/python run.py --assets .cache/runtime_inputs \
  --output runs/original-time-01 --run-id layout_002
```

The output directory must be new. This command runs a fresh physical episode, not a saved-state playback. The fixed reference layout and initialization are in `inputs/`; changing them creates a different experiment and invalidates the supplied reference qualification.

Inspect:

- `runs/original-time-01/execution_metadata.json`: source/dependency identity, terminal status, native success, actual physics-step count.
- `runs/original-time-01/execution.log`: native stdout/stderr, including failure details.
- `runs/original-time-01/workspace/logs/spincoat/t1/layout_002/result.json`: the actual same-episode task result and original stage gates.
- `runs/original-time-01/action_capture/`: actual controls and synchronized native state samples on the derived 30 Hz grid.
- `runs/original-time-01/workspace/logs/spincoat/t1/layout_002/model.mjb`: locally generated model for the compatibility-checked reference replay below.

A zero process exit code alone is not a physics or strict-geometry certificate. Keep failed run directories as evidence; use a new output directory for another attempt. The Python input/output fence guards accidental accesses by the bound Python workflow; it is not an OS security sandbox for arbitrary extensions.

## Recover and replay the already-successful trajectory

The retained reference is exactly `walledge_native_11_layout_002`: 86.267 s of native simulation, 2,589 synchronized 30 Hz frames. It used the original selective 2×/4× motion references. **It is deliberately not relabeled as a new 1× episode.** The default simulator above is the separate original-time implementation.

```bash
.venv/bin/python fetch_reference.py --output .cache/native11-reference
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 \
  .venv/bin/python replay.py --reference .cache/native11-reference \
  --model runs/original-time-01/workspace/logs/spincoat/t1/layout_002/model.mjb \
  --output runs/reference-smoke --mode smoke
```

The downloader verifies the release archive and every member. An already-downloaded archive can be supplied with `fetch_reference.py --archive PATH --output NEW_DIRECTORY`.

For the complete video, replace the last line with `--output runs/reference-full --mode full`. The full output is `C14_reference.mp4` with three unscaled 1280×720 onboard panels, plus per-frame calibration/state hashes in `frames.jsonl` and checks in `manifest.json`. The smoke video contains explicitly labeled selected states, not the complete time sequence.

Replay first restores the reference's captured display state for the 26 animated, noncolliding liquid geometries in a disposable model. It then verifies **all 452 model-array fingerprints**, dimensions and physics options; only filesystem-path fields are excluded. Collision-enabled geometries cannot be restored, and physical-model mismatches remain errors. This is not a tolerance-based comparison or a physics-state repair. The replay uses saved robot/object state and actual controls without pose interpolation or new physics steps. Liquid display interpolation remains the original recorded-state renderer's behavior.

C14 preserves `cam_head`, `cam_left_wrist`, and `cam_right_wrist`, each wrist's own local extrinsics, per-camera near clipping, scene/hand occlusion, mesh display-list isolation and optical depth/projection checks. Rendering the C14 brackets does not certify their physical assembly, manufacturing or dynamics.

## Repository boundaries

- `native/`: only the source dependency closure of the current planner. Historical-looking module suffixes are active imports, not separate old experiment trees.
- `native_fluid/`: the task's current reduced liquid/film implementation.
- `bootstrap/`, `assets.lock.json`, `assets/`, `inputs/`: reproducible construction and the exact reference initialization.
- `replay_support/`: the accepted C14 saved-state drawing implementation and assembly inputs.
- `runtime_manifest.json`, `replay_manifest.json`, `reference/`: hash-bound execution and recovery contracts.
- `verification.json`, `source_provenance.json`: exercised evidence and source mapping.
- `benchmark.py`, `benchmark/`: workflow-independent live benchmark, C14 observations, native execution, public task definition and private evaluation.
- `BENCHMARK.md`, `benchmark_verification.json`: benchmark operation and separately scoped verification; generated attempts remain outside source control.

Generated runs, model binaries, videos, environments, credentials, private agent sessions and historical experiment directories are not tracked. The one preserved trajectory is a checksummed release asset, not a large Git history dump. The package is published as a subtree of the existing source repository; its source ancestry is recorded without publishing unrelated historical trees.

## Updating this repository

Preserve the current entrypoints and manifest-driven structure. Revalidate changed inputs in a new output directory, keep the old reference immutable, and update the corresponding hashes with evidence. Do not claim new success from replay, change reference timing labels, relax collision/acceptance rules, or replace a failed run with a successful historical trajectory.

The earlier 20-layout training-data collection is separate and is not represented as completed by this publication.
