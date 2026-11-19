# PyRUA round 1 on the three-camera spin-coating benchmark (development results)

Coding agents following the PyRUA method worked on this benchmark's task (`spincoat-three-camera-measured-torque-v1`,
branch `benchmark/three-camera-agent`, commit `94a21b0`). Eight agents ("seeds") explored in parallel, each on its
own fresh layouts, and wrote Python programs that drive the robot from the three cameras and the joints; programs
were then run frozen, with no agent and no model, on more fresh layouts. This folder holds what that first round
came to and its best programs. The round was paused on 2026-11-19 at 12:25 (UTC+8) before it finished: seven of
the eight seeds had submitted, the eighth was still exploring.

**These are development numbers, not the benchmark's official evaluation.** They were measured on development
layouts with the benchmark's own world and task observer run in process (below), not through `benchmark.py`'s HTTP
judge on the reference case.

## How the episodes were run

- The benchmark's own world, reset, execution controller, rotor motor and task observer (`benchmark.backend.Backend`)
  in process; the HTTP layer and its per-step recording were not used.
- Development layouts: the reference `layout_002` with the rack and tube, the beaker, the pipette stand and the
  spin-coater each moved uniformly within 20 mm in x and in y (the judge's grasp dock follows the stand, as its own
  rule says). The reference case itself was not run.
- 25 Hz control: each row is 40 native 1 ms steps, the joint targets moved linearly across them. The torso was held
  at the reset's pose; the head was turned to (-0.389, 0.538) rad in the episode's first 80 control steps (counted),
  so that the head camera sees the tube, the beaker, the pipette and the coater from every base dock. An episode had
  4,420 control steps after that (180,000 native steps in all).
- The base was driven by its four wheel speeds; its pose in the room came from the cameras' absolute calibration.
- Agents saw the three RGB cameras with their calibration, the joints, the base pose and the public task contract
  restated in words. They did not see the scripted expert, its plans, the recorded reference trajectory or the
  judge's code.
- Ten stages, latched from the task observer's own measurements, were told to agents while they developed (never to
  a frozen program): `tube_lifted`, `mixed`, `poured`, `tube_returned`, `pipette_lifted`, `aspirated`, `at_coater`,
  `dispensed`, `spun`, `stopped`.

## Results

Frozen programs on fresh development layouts (no agent, no model; `results/ledger_stats.txt` has every program):

| program (seed) | runs | tube_lifted | mixed | poured and later | stage rate | complete |
| --- | --- | --- | --- | --- | --- | --- |
| `round_retained-95c9e50b91` (s01, submitted) | 10 | 9 | 3 | 0 | 0.12 | 0 |
| `spincoat_final-c50b2d5929` (s02, submitted) | 9 | 8 | 3 | 0 | 0.12 | 0 |
| `spincoat_direct-d7b044663a` (s03, submitted) | 10 | 10 | 0 | 0 | 0.10 | 0 |
| `spincoat_round-35249aadd7` (s07, submitted) | 10 | 8 | 0 | 0 | 0.08 | 0 |
| `spincoat_final-70bf54e7c5` (s08, submitted) | 10 | 8 | 0 | 0 | 0.08 | 0 |
| `spincoat_lift-92e4f7e831` (s05, submitted) | 9 | 5 | 0 | 0 | 0.06 | 0 |
| `spincoat_submit-ac2fd3d601` (s04, submitted) | 9 | 3 | 1 | 0 | 0.04 | 0 |
| `spincoat_final-a9278bc270` (s04) | 1 | 1 | 1 | 0 | 0.20 | 0 |

Exploring episodes (an agent with stage feedback and rewinds: what was reached at least once, not a rate):
`tube_lifted` by all eight seeds, `mixed` by s01, s02, s04 and s06, `poured` (at least 99 % of the 25 mL in the
beaker) by s04 and s06, and `tube_returned` by s06 (in an episode still open when the round was paused). No seed
reached the pipette, aspiration, dispensing or the spin.

## What is here

- `programs/`: four program bundles as their agents registered them: the two best by frozen measurement
  (`round_retained-95c9e50b91`, `spincoat_final-c50b2d5929`), s04's `spincoat_final-a9278bc270` (one frozen run,
  mixed), and s06's last program `spincoat_highlift-ddd2dae63a`, which carries its pouring and tube-return approach and
  was never measured frozen (the round was paused first). Each has `controller.py` with `STAGES`, `setup(cb, ctx)` and
  one function per stage, and its own helper modules. **They run on the PyRUA harness** (its session client and its
  `codebox` primitives: kinematics, IK, grasping, perception with SAM3, the base's `go_to`), which is not part of this
  folder; they do not run against `benchmark.py`'s HTTP interface as they are.
- `results/ledger_stats.txt`: every program's frozen and exploring records, stage by stage, with n.
- `results/seeds.json`: each seed's layouts used, programs registered, the program it submitted, and the stages its
  closed episodes reached.
- `videos/`: the three cameras side by side (head, left wrist, right wrist), one frame a simulated second, played at
  6 frames a second:
  - `s04_E006_pour.mp4`: s04's exploring episode that first poured (step 871), stitched across its rewinds;
  - `s06_E005_pour_then_station_contact.mp4`: s06 pours, then the robot touches the station and the judge ends the
    episode as a failure;
  - `s04_E005_rehearse_frozen_program.mp4`: one of s04's programs run frozen through a whole episode on a fresh layout.

## Notes from running it

- Some episodes' worlds stepped 10 to 20 times slower than others (10-17 native steps a second against 200-340), in
  states not identified yet: not pouring itself (an episode that poured stayed at 217).
- Holding an attempt to its step limit after a program has returned is not what the HTTP judge does (it measures an
  attempt when the participant calls `/v1/finish`); the harness is being changed to end development episodes the same
  way.
