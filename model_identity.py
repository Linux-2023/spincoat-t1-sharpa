"""Path-independent, fail-closed compatibility for recorded MuJoCo states."""
from __future__ import annotations

import hashlib
import numbers
import numpy as np
import mujoco

EXCLUDED_PATH_FIELDS = ('paths', 'mesh_pathadr', 'tex_pathadr', 'skin_pathadr', 'hfield_pathadr')
DIMENSIONS = ('nq', 'nv', 'nu', 'na', 'nmocap', 'nbody', 'njnt', 'ngeom', 'nmesh', 'ntex', 'ncam', 'nsensor')


def array_identity(value):
    contiguous = np.ascontiguousarray(value)
    payload = memoryview(contiguous).cast('B') if contiguous.size else b''
    return {'dtype': contiguous.dtype.str, 'shape': list(contiguous.shape),
            'sha256': hashlib.sha256(payload).hexdigest()}


def model_fingerprint(model):
    arrays = {}
    for name in sorted(dir(model)):
        if name.startswith('_') or name in EXCLUDED_PATH_FIELDS:
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            arrays[name] = array_identity(value)
    options = {}
    for name in sorted(dir(model.opt)):
        if name.startswith('_'):
            continue
        value = getattr(model.opt, name)
        if isinstance(value, np.ndarray):
            options[name] = array_identity(value)
        elif isinstance(value, numbers.Integral):
            options[name] = int(value)
        elif isinstance(value, numbers.Real):
            options[name] = float(value)
    return {'schema_version': 1, 'format': 'mujoco-model-semantic-v1',
            'mujoco_version': mujoco.__version__,
            'dimensions': {name: int(getattr(model, name)) for name in DIMENSIONS},
            'arrays': arrays, 'options': options,
            'excluded_fields': list(EXCLUDED_PATH_FIELDS)}


def verify_model(model, expected):
    actual = model_fingerprint(model)
    if actual == expected:
        return
    differences = []
    for section in ('dimensions', 'arrays', 'options'):
        wanted, observed = expected.get(section, {}), actual[section]
        differences.extend(section + '.' + name for name in sorted(set(wanted) | set(observed))
                           if wanted.get(name) != observed.get(name))
    for name in ('schema_version', 'format', 'mujoco_version', 'excluded_fields'):
        if actual[name] != expected.get(name):
            differences.append(name)
    raise ValueError('Recorded model compatibility failed: ' + ', '.join(differences[:40]))


def restore_recorded_display(model, snapshot):
    """Restore saved liquid-display geometry in a disposable replay model only.

    This is recorded visual state, not a model-identity exception. The caller
    must still verify every model array and option after restoration.
    """
    if snapshot.get('schema_version') != 1:
        raise ValueError('Unsupported recorded display-state format')
    changes, seen = [], set()
    for row in snapshot['geoms']:
        index = model.geom(row['name']).id
        if index in seen:
            raise ValueError('Duplicate recorded display geometry')
        seen.add(index)
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index]))
        if body != row['body']:
            raise ValueError('Recorded display geometry belongs to a different body')
        if (model.geom_contype[index] or model.geom_conaffinity[index]
                or np.any(model.pair_geom1 == index) or np.any(model.pair_geom2 == index)):
            raise ValueError('Recorded display restoration cannot change collision geometry')
        for field in ('geom_pos', 'geom_size'):
            value = np.asarray(row[field], dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError('Invalid recorded display vector: ' + field)
            changes.append((field, index, value))
    for field, index, value in changes:
        getattr(model, field)[index] = value
    return len(seen)
