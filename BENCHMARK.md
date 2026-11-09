# Workflow-independent three-camera benchmark

`benchmark.py` serves a fresh evaluator-owned robot world. Participants choose their own model, planner, language, skills, control batches and observation times. The server does not launch an agent or execute the expert workflow.

This is a separate apparatus/judge identity: `spincoat-three-camera-measured-torque-v1`. It is not the historical expert's all35 certification. `run.py`, `replay.py` and their existing reference evidence remain unchanged.

## Install and start

Use the benchmark source branch, Python 3.11 and the pinned dependencies from the repository. Reuse an existing prepared asset directory only when its receipt matches `assets.lock.json`; do not copy a running environment or overwrite another run.

```bash
git clone --branch benchmark/three-camera-agent --single-branch https://github.com/Linux-2023/spincoat-t1-sharpa.git
cd spincoat-t1-sharpa
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MUJOCO_GL=egl .venv/bin/python prepare.py --assets .cache/runtime_inputs
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 .venv/bin/python benchmark.py --assets .cache/runtime_inputs --output runs/benchmark-example --port 0
```

The startup JSON supplies the allocated loopback URL and a private `token_file`. Keep that file and the evaluator output directory away from participants. Deliver only the URL and bearer token through your evaluation harness. Do not commit credentials.

For explicit CPU rendering, install the system OSMesa library (`libosmesa6` on Debian/Ubuntu) and launch with `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` instead of EGL. The exercised CPU configuration and measured memory are recorded in [benchmark_verification.json](benchmark_verification.json). Budget memory and disk externally; concurrent worlds have additive costs.

The output path must be new. Each invocation constructs a new world and ends after the attempt is sealed. It is not a persistent shared simulator, an episode-resume command or an OS sandbox. Do not restart an uncertain physical request as a new attempt and call it a continuation.

## Participant interface

All routes require `Authorization: Bearer TOKEN`. JSON request bodies use `Content-Type: application/json`.

| Method | Route | Behavior |
|---|---|---|
| GET | `/v1/info` | Installed static task definition, budgets, actuator schema and camera contract |
| GET | `/v1/status` | Lifecycle and clocks; no private task state |
| POST | `/v1/observe` | Empty object; capture the three current-tick RGB images, calibration and robot proprioception |
| GET | image URL from an observation | Fetch that delivered PNG; arbitrary evaluator paths are inaccessible |
| POST | `/v1/act` | A request ID and a list of controls/step-count pairs |
| GET | `/v1/receipts/REQUEST_ID` | Reconcile a completed request without advancing physics |
| POST | `/v1/finish` | Empty object; irreversibly seal this attempt and return its measured result |

An action has `controls` and `steps`. Controls are either the full ordered vector from `action_schema` or a name-to-value map. A partial map retains the other requested controls. The schema declares each actuator's `range`, units, semantics and force limits. Inputs are original robot actuator commands, not object poses, named task stages or hidden trajectory waypoints.

Physics advances by one native 1 ms step per accepted step. Observation, networking and participant inference do not advance it. Batch lengths and observation frequency are not prescribed. Every row is validated before a batch starts; an illegal later row cannot cause an earlier row to execute. Budget or terminal-condition interruption records the executed prefix rather than an unexecuted suffix.

Retain the request ID and original payload before sending. Resending that payload with the same ID returns its original receipt; reusing the ID for different content is a conflict. The client does not retry actions automatically. After a transport failure, reconcile the original ID and evaluator journal; do not invent a new ID and blindly repeat a possibly executed action. There is no crash-recovery/reset API.

The optional `benchmark/client.py` uses only the Python standard library. Other HTTP clients are equally supported. This small example holds the initial command for 10 ms and submits an intentionally incomplete attempt; it is not a laboratory policy:

```python
import os
from uuid import uuid4
from benchmark.client import Client

client = Client(os.environ['BENCHMARK_URL'], os.environ['BENCHMARK_TOKEN'])
info = client.info()
observation = client.observe()
for camera in info['cameras']:
    png = client.image(observation, camera)
request_id = str(uuid4())
receipt = client.act(
    [{'controls': info['action_schema']['initial_controls'], 'steps': 10}],
    request_id=request_id,
)
result = client.finish()
```

For an untrusted participant, copy only the standalone client into its isolated environment or use raw HTTP. Do not expose the evaluator host's filesystem, private outputs, model/data objects or process memory. A bearer-protected loopback interface alone does not provide that isolation.

## Observations and physical execution

- `cam_head`, `cam_left_wrist`, `cam_right_wrist`: three distinct native 1280×720 RGB8 views from the same physical tick. No historical trajectory, external legacy camera, HUD or imposed inference rate is substituted.
- C14 head/body binding, independent left/right wrist transforms, per-camera near clipping, native scene occlusion and mesh-list isolation are retained. Rendering uses private native state and does not advance or repair the live world.
- Camera calibration includes absolute camera transforms. Do not interpret the absence of a separate world-base pose field as hiding absolute camera motion. Object poses, contacts, liquid state and private task metrics are not observation channels.
- Robot proprioception contains the declared actuated joints and current end-effector FK in `chassis_base`, not a scene-wide joint/object registry.
- The low-level execution transform is declared in `/v1/info`: bounded left-finger encoder correction, native wheel braking/hold and robot bias compensation. Requested controls and actual applied controls/forces are recorded separately. Original robot force caps, friction and collision rules are not relaxed.
- The rotor uses a separately identified bounded-torque motor. Measured native shaft velocity drives spin acceptance and film integration; no live rotor-velocity overwrite is used.

## Task definition and attempt accounting

[benchmark/task_contract.json](benchmark/task_contract.json) is the installed numerical definition and the static contract returned by `/v1/info`. It specifies acquisition/lift, supported mixing, pouring/return, aspiration, dispensing, spin/stop and configured-contact checks. These are measurements, not participant stage commands.

The default limits are 180,000 native physics steps and 1,800 seconds of session wall time. Waiting and rendering consume wall time but not physics steps. The declared reset includes a separate 15,000-step tube-only initialization settlement, not an expert rollout or continuation of a historical episode.

Only the supplied [reference case](benchmark/cases/reference.json) is accepted. `--task` can change budgets only; its remaining definition must match the installed evaluator. Different budgets have a different task SHA and must not be silently combined as the same evaluation.

Predeclare every attempt, then use its exact directory name:

```bash
.venv/bin/python -m benchmark.aggregate plan --task benchmark/task_contract.json --cases benchmark/cases/reference.json --repetitions 1 --output evaluation/suite.json
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 .venv/bin/python benchmark.py --assets .cache/runtime_inputs --output evaluation/attempts/case-000-rep-000 --port 0
```

Run the participant separately and submit the attempt. After all declared attempts finish:

```bash
.venv/bin/python -m benchmark.aggregate score --suite evaluation/suite.json --runs evaluation/attempts --output evaluation/score.json
```

A valid task failure stays in the denominator. Missing attempts, infrastructure errors and damaged evidence remain visible and prevent an official aggregate score. An official rate additionally requires matching source identities. Predeclaration does not establish that cases are held out, representative or physically feasible; do not claim those properties from a manifest alone.

## Retained evidence

Each attempt retains its task/case, frozen source identity, backend initialization/apparatus identity, delivered PNGs and calibration, request journal, private events, native transitions and final assessment. Initial and final native states are retained even for a zero-action submission. Transition chunks preserve native float64 state and requested/applied controls and forces; the terminal state is included. The result binds the source manifest digest, and aggregation checks sealed file/chunk hashes and tick continuity.

A failed native transition is quarantined with raw state and diagnostics rather than reset or presented as a successful step. Preserve failed outputs. Checksums provide consistency checks, not signatures or protection against a malicious evaluator that can rewrite all evidence. The recorder is not a general cross-version checkpoint/restore facility.

Functional completion and configured contact checks are separate outcomes. Strict-physics and hardware certification remain unknown; they are not inferred from either outcome.

## Exercised verification

[benchmark_verification.json](benchmark_verification.json) records the tested source hashes and measured verification scope:

- 57 regression tests, including native-state isolation, actuator ownership, task-measurement boundaries, HTTP privacy/parsing, exactly-once requests, budgets and damaged/missing-attempt handling. Existing model/reference compatibility tests also passed.
- A fresh CLI/HTTP attempt with 25 actual native steps, mixed action lengths, two synchronized three-camera observations and lossless PNG hash checks. Authentication, private-route rejection, atomic invalid-batch rejection and receipt reconciliation were exercised.
- Actual C14 marker projection/depth checks, scene restoration and live-state noninterference; the head and both wrist images at tick 25 were visually inspected. Hand/scene occlusion was retained.
- Explicit OSMesa/llvmpipe CPU rendering. The measured run used about 3.02 GiB peak RSS; this is not a bound for arbitrary participant workloads.

The smoke deliberately submitted an incomplete laboratory task and was correctly scored as a valid functional failure. It demonstrates the interface, native execution, rendering and evidence path, not a successful laboratory policy or trained-agent result. The unchanged full historical planner was not rerun for this feature; its existing evidence retains its original scope.

## Limits and operation

- No layout-perturbation, held-out-layout or repeated-policy robustness result is claimed.
- The reduced liquid/film models and kinematic pipette-volume proxy are uncalibrated. They do not establish chemical homogeneity, film quality, resolved droplet collisions or real liquid reaction forces.
- Contact data comes from the just-completed solver step; endpoint positions/FK are refreshed privately. Contact geometry can lag by one native millisecond.
- C14 optical agreement is not mounting dynamics, manufacturing or hardware certification.
- A single owner thread serializes physics and rendering. HTTP concurrency is not physics parallelism. Independent processes require independent state/output and resource admission; this package is not a whole-machine scheduler.
- Keep existing running episodes frozen. Source updates require a fresh process/output; changing files does not hot-update a live world. No shared service or default entrypoint is switched by this source feature.
