"""Hash-bound public physical leaves; no expert installation or task policy.

Loader/registry provenance: teacher_runtime/_sources.py; active leaf identities
are pinned by the existing runtime_manifest.json, not historical episode paths.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def leaf(name, path):
    path = Path(path).resolve()
    digest = file_hash(path)
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path or existing.__source_sha256__ != digest:
            raise RuntimeError('Loaded physical source identity changed: ' + name)
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module.__source_sha256__ = digest
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class PhysicalSources:
    """Load unchanged leaves under one real package for relative film imports."""
    def __init__(self, workspace=None):
        manifest = json.loads((ROOT / 'runtime_manifest.json').read_text())
        selected = ('native/t1_aspiration_stage.py', 'native/t1_liquid_palette.py',
                    'native/t1_gravity_pour_liquid_v2.py',
                    'native_fluid/thin_film.py', 'native_fluid/fluid_v2.py')
        rows = {row['source']: row for row in manifest['files']}
        self.files = {}
        for source in selected:
            row = rows[source]
            path = ROOT / source if workspace is None else Path(workspace) / row['target']
            if file_hash(path) != row['sha256']:
                raise ValueError('Physical leaf is not the pinned public input: ' + source)
            self.files[source] = path
        self.hashes = {name: file_hash(path) for name, path in self.files.items()}
        # Paths distinguish two independently materialized but identical worlds.
        package = '_benchmark_physics_' + identity({k: [str(v), self.hashes[k]]
                                                    for k, v in self.files.items()})[:20]
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = []
            sys.modules[package] = module
        self.aspiration = leaf(package + '.aspiration', self.files[selected[0]])
        self.palette = leaf(package + '.palette', self.files[selected[1]])
        leaf(package + '.thin_film', self.files[selected[3]])
        self.fluid = leaf(package + '.fluid_v2', self.files[selected[4]])
        previous = sys.modules.get('t1_liquid_palette')
        sys.modules['t1_liquid_palette'] = self.palette
        try:
            self.gravity = leaf(package + '.gravity', self.files[selected[2]])
        finally:
            if previous is None:
                sys.modules.pop('t1_liquid_palette', None)
            else:
                sys.modules['t1_liquid_palette'] = previous


def robot_actuators(model):
    """The only participant actuator descriptors, with no scene joint registry."""
    result = []
    chassis = int(model.body('chassis_base').id)
    for index in range(model.nu):
        joint = int(model.actuator_trnid[index, 0])
        if joint < 0 or int(model.actuator_trntype[index]) != 0:
            raise ValueError('Only direct native robot joint transmissions are supported')
        name = model.joint(joint).name
        body = int(model.jnt_bodyid[joint])
        while body and body != chassis:
            body = int(model.body_parentid[body])
        if body != chassis or int(model.jnt_type[joint]) not in (2, 3):
            raise ValueError('Non-robot or nonscalar participant actuator: ' + name)
        wheel = model.actuator(index).name.startswith('astribot_chassis_motor_')
        slide = int(model.jnt_type[joint]) == 2
        result.append({'name': model.actuator(index).name, 'index': index, 'joint': name,
                       'limited': bool(model.actuator_ctrllimited[index]),
                       'range': model.actuator_ctrlrange[index].tolist(),
                       'unit': ('m/s' if slide else 'rad/s') if wheel else ('m' if slide else 'rad'),
                       'semantics': 'joint_velocity' if wheel else 'joint_position',
                       'force_limited': bool(model.actuator_forcelimited[index]),
                       'force_range': model.actuator_forcerange[index].tolist(),
                       'force_unit': 'N' if slide else 'N*m'})
    return result
