# Third-party inputs and redistribution boundaries

This repository does not grant additional rights to upstream robot or simulator assets. Users must comply with each upstream source's terms. Public availability alone is not a redistribution license.

| Input | Pinned source | Packaging decision |
|---|---|---|
| Astribot T1 | [Astribot-Dev/astribot_descriptions](https://github.com/Astribot-Dev/astribot_descriptions/tree/5ca7235afa19180f7ee5d87bfb2dd11392853143), commit `5ca7235afa19180f7ee5d87bfb2dd11392853143` | Model files are acquired directly from upstream during preparation. No T1 model XML/mesh or embedded binary model is redistributed. The inspected revision did not establish a root license for these T1 assets; no extra permission is inferred. |
| Sharpa Wave hands | [sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e), commit `0d19cac602f46456b819e4b6a2c09a74982c9a3e` | Original files and their Apache-2.0 license/notice remain in the upstream checkout. Obtain them through `prepare.py`. |
| DexBioLab pipette/tip inputs | [CumulusAlpha/DexBioLab](https://github.com/CumulusAlpha/DexBioLab/tree/8f80711fad12f72aaf9c97f4d070bf6ead0d5ed5), commit `8f80711fad12f72aaf9c97f4d070bf6ead0d5ed5` | Acquire the exact required assets from upstream; no blanket license is inferred from unrelated vendored licenses. |
| Spin-coater geometry | [BirdbrainEngineer/Spin-coater-v1](https://github.com/BirdbrainEngineer/Spin-coater-v1/tree/3fbf81766061f0dafb3619357c7b72338ffb439c), commit `3fbf81766061f0dafb3619357c7b72338ffb439c` | Pinned original files are downloaded directly; upstream MIT license applies. |
| Laboratory bench | Konstantin Keller, [lab-bench source](https://sketchfab.com/3d-models/lab-bench-1ac2a62c52a848bbaf746146dc7253f8) | Modified geometry is included under CC BY 4.0. Original source, hash, scaling and removed/retained parts are recorded in `assets/assets/third_party/lab_bench/provenance.json` and `ATTRIBUTION.txt`. |
| DejaVu Sans | [DejaVu fonts](https://dejavu-fonts.github.io/) | The bundled font is accompanied by its complete upstream/distribution copyright and license notice in `FONT_LICENSE.txt`. |

The coat, task fixtures, C14 assembly, current planner additions and reduced liquid/film components are project-generated inputs. Their publication does not relicense underlying upstream work. No repository-wide license for third-party inputs is implied.

All downloaded assets have explicit URL/revision, byte count and SHA256 records in `assets.lock.json`. The robot is regenerated locally; only its path-normalized identity and compiled array fingerprints are published. Canonical XML normalization removes path differences and redundant STL MIME annotations only, not numeric parameters or geometry. A compiled comparison verified that this serialization difference did not change the model arrays or physics options.

The successful-reference release includes recorded state/action/liquid data but deliberately excludes `model.mjb`. Replay requires a locally built model and exact semantic compatibility checks.
