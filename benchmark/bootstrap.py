"""One fresh public leaf-built world, without entering the expert workflow.

Composition follows t1_anatomical_mount_scene_v7/build_scene, including its
separate 15 s tube reset settlement. The only wrapper reproduces the gravity
visual asset operation from run_t1_human_workflow_v2:30-36; no policy is installed.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from runtime import materialize, check_versions
from .native_sources import ROOT, file_hash, identity, robot_actuators
from .recording import save_json

BOOTSTRAP_ID = 'public_anatomical_ready_settled_tube_v1'


def case_inputs(case):
    reference = json.loads((Path(__file__).parent / 'cases/reference.json').read_text())
    if case != reference:
        raise ValueError('Only the exact evaluator-owned reference case is implemented')
    layout_path = ROOT / case['layout_source']
    config_path = ROOT / case['preparation_source']
    if file_hash(layout_path) != case['layout_sha256'] or file_hash(config_path) != case['preparation_sha256']:
        raise ValueError('Reference reset inputs changed')
    return json.loads(layout_path.read_text()), json.loads(config_path.read_text())


def station_regions(layout):
    """Public score regions, not navigation commands or an expert handoff."""
    c = layout['right_runtime_config']
    stand = np.asarray(layout['main_asset_targets_world_xy_m']['pipette_stand_tool'], dtype=float)
    delta = stand - np.array([.68, 0.])
    if not np.isfinite(delta).all() or np.any(np.abs(delta) > .02 + 1e-12):
        raise ValueError('Declared right dock rule only covers SAME20 layout offsets')
    return {'preparation': list(layout['v7_left_initial_coarse_base_xy_m']),
            'grasp': (np.asarray(c['grasp_base_xy_m']) + delta).tolist(),
            'aspiration': list(c['aspiration_base_xy_m']),
            'coating': list(c['coating_base_xy_m'])}


@contextmanager
def scene_imports(workspace):
    """Scope legacy top-level import names to reset construction, never a policy."""
    scripts = workspace / 'scripts'
    names = {path.stem for path in scripts.glob('*.py')}
    conflicts = names.intersection(sys.modules)
    if conflicts:
        raise RuntimeError('Native reset import namespace already in use: ' + ', '.join(sorted(conflicts)))
    previous_path, previous_cwd = sys.path[:], Path.cwd()
    env_keys = ('T1_LAB_CONTEXT', 'T1_CHEMIST_COAT', 'MUJOCO_GL', 'MPLCONFIGDIR', 'XDG_CACHE_HOME')
    previous_env = {key: os.environ.get(key) for key in env_keys}
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(scripts))
    os.environ['MPLCONFIGDIR'] = str(workspace / '_cache/matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(workspace / '_cache')
    os.chdir(workspace)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, module in list(sys.modules.items()):
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).resolve().is_relative_to(workspace):
                sys.modules.pop(name, None)


def initialize(assets, output, case):
    assets, output = Path(assets).resolve(), Path(output).resolve()
    if output == assets or output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError('Prepared input and output directories must be disjoint')
    layout, config = case_inputs(case)
    if layout['left_hand_mount_convention'] != 'RIGHT_anatomical_mirror_Rz180_provisional':
        raise ValueError('Unexpected anatomical mount identity')
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / 'runtime_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    versions = check_versions(manifest['runtime_versions'])
    workspace, unused_expert_argv, materialized = materialize(manifest, output, assets, case['id'])
    # materialize is input preparation only. Never execute its argv or plugins.
    setup = output / 'reset'
    with scene_imports(workspace):
        assets_module = importlib.import_module('t1_bimanual_glass_preparation')
        visuals = importlib.import_module('t1_gravity_liquid_render_v2')
        palette = importlib.import_module('t1_liquid_palette')
        original = assets_module.add_reagent_preparation
        def physical_assets(spec, native_config=None):
            handles = original(spec, native_config)
            visuals.attach_gravity_visuals(spec)
            for geom in spec.geoms:
                if any(token in (geom.name or '') for token in
                       ('liquid_visual', 'liquid_proxy', 'tube_bottom_liquid_', 'pour_visual_', 'fluid_', 'film_visual')):
                    geom.rgba[:3] = palette.LIQUID_RGBA[:3]
            return handles
        assets_module.add_reagent_preparation = physical_assets
        try:
            builder = importlib.import_module('t1_anatomical_mount_scene_v7')
            xy = layout['main_asset_targets_world_xy_m']
            scene, handles, report = builder.build_scene(
                setup, cup_xy=xy['shared_beaker'], tube_xy=xy['rack_and_tubes'],
                stand_xy=xy['pipette_stand_tool'], stand_yaw=layout['right_runtime_config']['stand_yaw_deg'],
                coater_xy=xy['spincoater'], decor_proposal=layout, appearance=True,
                settled_tube=True, anatomical_ready=True)
            spec, model, data, children = scene
            palette.apply_liquid_palette(model)
        finally:
            assets_module.add_reagent_preparation = original
        loaded = {str(Path(module.__file__).resolve().relative_to(workspace)): file_hash(module.__file__)
                  for module in list(sys.modules.values())
                  if getattr(module, '__file__', None) and Path(module.__file__).resolve().is_relative_to(workspace)}
    handles = {key: value for key, value in handles.items() if key != 'child_spec'}
    if handles['config'] != config:
        raise ValueError('Built public apparatus does not match the reference preparation input')
    if (model.nq, model.nv, model.nu, model.na) != (157, 154, 68, 0) or model.opt.timestep != .001:
        raise ValueError('Unexpected native robot dimensions or timestep')
    if data.time != 0. or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError('Initializer did not return a finite fresh reset')
    for i, sign in enumerate((-1., 1., -1., 1., -1., 1., -1.), 1):
        left = int(model.joint('astribot_arm_left_joint_' + str(i)).qposadr[0])
        right = int(model.joint('astribot_arm_right_joint_' + str(i)).qposadr[0])
        if data.qpos[left] != sign * data.qpos[right]:
            raise ValueError('Final anatomical ready state was overwritten')
    actuators = robot_actuators(model)
    wheels = [row['index'] for row in actuators if row['semantics'] == 'joint_velocity']
    # Original reset ctrl=qpos mapping, never used again during the episode.
    for row in actuators:
        if row['index'] not in wheels:
            data.ctrl[row['index']] = data.qpos[int(model.joint(row['joint']).qposadr[0])]
    route = {'wheel_ids': wheels, 'route_gain': model.actuator_gainprm[wheels].tolist(),
             'route_bias': model.actuator_biasprm[wheels].tolist(),
             'hold_gain_scale': 32., 'brake_duration_s': .4}
    mujoco.mj_forward(model, data)
    model_path = output / 'model.mjb'
    mujoco.mj_saveModel(model, str(model_path), None)
    state_path = output / 'initial_state.npz'
    state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    integration = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, integration, state_spec)
    with state_path.open('xb') as stream:
        np.savez_compressed(stream, qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl,
                            time=np.array(data.time), integration=integration, state_spec=state_spec)
    provenance = {'initializer_id': BOOTSTRAP_ID, 'case': case, 'case_identity': identity(case),
                  'public_parent_commit': 'a36cc83028390136831c84c75b7402bbd0783b50',
                  'runtime_manifest_sha256': file_hash(manifest_path),
                  'prepared_receipt_sha256': file_hash(assets / 'prepared.json'),
                  'assets_manifest_sha256': file_hash(ROOT / manifest['assets_manifest']),
                  'runtime_versions': versions, 'materialized_inputs': materialized,
                  'loaded_reset_sources': loaded, 'model_sha256': file_hash(model_path),
                  'initial_state_sha256': file_hash(state_path), 'scene_handles': handles,
                  'stations': station_regions(layout), 'controller_route': route,
                  'reset_report': report, 'episode_steps_during_setup': 0,
                  'separate_tube_reset_work_steps': 15000, 'expert_task_executed': False,
                  'historical_episode_required': False, 'layout_perturbations_implemented': False}
    save_json(output / 'initialization_manifest.json', provenance)
    return SimpleNamespace(model=model, data=data, handles=handles, config=config,
                           workspace=workspace, model_path=model_path, route=route,
                           stations=station_regions(layout), provenance=provenance)
