"""Fixed-factor motion references; physical clocks and acceptance stay native.

Software reference ceilings are deliberately superseded. The original actuator
model, real feedback, geometry checks and physical qualification are not changed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
import linecache
import math
from pathlib import Path
import textwrap

import numpy as np
from scipy.interpolate import BPoly

FACTOR = 2.0
DT = 0.001
RECORDS = []
_PATCH_SERIAL = 0


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def record(kind, label, **fields):
    row = _plain(dict(kind=kind, label=label, **fields))
    json.dumps(row, allow_nan=False)
    RECORDS.append(row)
    return row


def replace_once(source, old_text, new_text, label, count=1):
    found = source.count(old_text)
    if found != count:
        raise RuntimeError(f'Uncapped {label}: expected {count} anchors, found {found}: {old_text!r}')
    return source.replace(old_text, new_text, count)


def patch(function, replacements, label, helpers=None):
    """Rewrite actual code while keeping aliases, mutable globals and closure cells.

    A temporary factory compiles the same free-variable layout. Only its code
    is installed: the original function retains its globals, defaults and cells.
    Never unwrap a generated function to an older disk implementation.
    """
    global _PATCH_SERIAL
    original_code = function.__code__
    original = textwrap.dedent(inspect.getsource(original_code))
    changed = original
    for old, new, count in replacements:
        changed = replace_once(changed, old, new, label, count)
    tree = ast.parse(changed)
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise RuntimeError(f'Uncapped {label}: expected one actual function definition')
    node = tree.body[0]
    # Preserve body spelling: downstream native factories use checked source
    # anchors inside nested functions. AST unparse would change those anchors.
    lines = changed.splitlines(True)
    start, end = node.lineno - 1, node.end_lineno
    freevars = original_code.co_freevars
    if freevars:
        first = node.body[0]
        if first.lineno == node.lineno:
            line = lines[start]
            header, body = line[:first.col_offset].rstrip(), line[first.col_offset:].rstrip('\n')
            lines[start] = (header + '\n    nonlocal ' + ','.join(freevars)
                            + '\n    ' + body + '\n')
        else:
            indentation = lines[first.lineno - 1][:first.col_offset]
            if indentation.strip():
                raise RuntimeError(f'Uncapped {label}: unsupported source suite indentation')
            insertion = first.lineno - 1
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                    and (len(node.body) == 1 or node.body[1].lineno > first.end_lineno)):
                insertion = first.end_lineno
            lines.insert(insertion, indentation + 'nonlocal ' + ','.join(freevars) + '\n')
            end += 1
    function_source = ''.join(lines[start:end])
    factory_name = '_uncapped_compile_factory'
    # A nested reference to the original function name must stay global unless
    # it was already an original closure cell. The compile-only wrapper must
    # not introduce a new self-reference cell for a formerly global name.
    binding = '' if node.name in freevars else '    global ' + node.name + '\n'
    source = ('def ' + factory_name + '(' + ','.join(freevars) + '):\n'
              + binding + textwrap.indent(function_source, '    ')
              + '\n    return ' + node.name + '\n')
    _PATCH_SERIAL += 1
    filename = f'<uncapped_{_PATCH_SERIAL}_{hashlib.sha256(source.encode()).hexdigest()[:12]}>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    # Compile only: do not evaluate decorators, annotations or default-value
    # expressions, and do not manufacture replacement globals or closure cells.
    compiled = compile(source, filename, 'exec')
    factory_code = next(value for value in compiled.co_consts
                        if isinstance(value, type(original_code)) and value.co_name == factory_name)
    new_code = next(value for value in factory_code.co_consts
                    if isinstance(value, type(original_code)) and value.co_name == node.name)
    signature = lambda code: (code.co_argcount, code.co_posonlyargcount,
                              code.co_kwonlyargcount, code.co_flags & 0x0c,
                              code.co_freevars)
    if signature(new_code) != signature(original_code):
        raise RuntimeError(f'Uncapped {label}: function signature or closure layout changed')
    for name, value in (helpers or {}).items():
        if name in function.__globals__ and function.__globals__[name] is not value:
            raise RuntimeError(f'Uncapped {label}: helper global collision: {name}')
    function.__globals__.update(helpers or {})
    function.__code__ = new_code
    record('callable_patch', label, module=function.__module__,
           original_file=original_code.co_filename,
           original_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
           compiled_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
           compiled_file=filename, replacement_count=sum(row[2] for row in replacements),
           original_globals_and_closure_cells_preserved=True)
    return function


def ceil_native_duration(seconds, dt=DT):
    seconds, dt = float(seconds), float(dt)
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite reference duration and native timestep required')
    return max(1, math.ceil(seconds / dt - 1e-10)) * dt


@dataclass
class Timing:
    path: BPoly
    clock: BPoly
    baseline_clock: BPoly
    duration: float
    source_duration: float
    blocks: list
    report: dict
    identity_source_clock: bool
    factor: float
    reference_duration: float

    def __call__(self, time, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('Only position, velocity and acceleration are supported')
        values = np.asarray(time, dtype=float)
        t = np.clip(values, 0.0, self.duration)
        source = self.clock(t)
        if derivative == 0:
            return self.path(source)
        rate = np.asarray(self.clock(t, 1))[..., None]
        first = self.path(source, 1)
        result = first * rate
        if derivative == 2:
            result = (self.path(source, 2) * rate**2
                      + first * np.asarray(self.clock(t, 2))[..., None])
        outside = (values < 0.0) | (values > self.reference_duration)
        return np.where(outside[..., None], 0.0, result)

    def source_time(self, time):
        value = np.clip(np.asarray(time, dtype=float) * self.factor,
                        0.0, self.source_duration)
        return float(value) if value.ndim == 0 else value

    def new_time_for_source(self, time):
        source = np.clip(np.asarray(time, dtype=float), 0.0, self.source_duration)
        value = np.where(source >= self.source_duration, self.duration, source / self.factor)
        return float(value) if value.ndim == 0 else value


def retime(path, *, duration, label, vlim=None, clock=None, alim=None,
           protected=(), maximum_rate=None, minimum_duration=None,
           accept_clock=None, factor=FACTOR):
    """Compose the supplied nominal clock with a fixed factor, without speed caps.

    A fractional final millisecond is an explicit endpoint hold, not a reduced
    factor or a shorter physics step. Physical waits must stay outside this API.
    """
    import v34_topp_core as core

    duration, factor = float(duration), float(factor)
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(factor) or factor <= 0:
        raise ValueError('Positive finite source duration and factor required')
    path = core.as_polynomial(path)
    identity = clock is None
    if clock is None:
        clock = BPoly(np.array([[0.0], [duration]]), [0.0, duration], extrapolate=False)
    if (not isinstance(clock, BPoly) or clock.c.ndim != 2
            or not np.isfinite(clock.c).all() or not np.isfinite(clock.x).all()
            or abs(float(clock.x[0])) > 1e-12
            or abs(float(clock.x[-1]) - duration) > 1e-10
            or not core._monotone(clock)):
        raise ValueError('A finite monotone scalar nominal clock over [0,duration] is required')
    if clock.c.min() < path.x[0] or clock.c.max() > path.x[-1]:
        raise ValueError('Nominal reference clock escapes the geometric path')
    callback = None
    if accept_clock is not None:
        callback = accept_clock.__module__ + '.' + accept_clock.__qualname__
        if callback != 'v34_topp_left_paths._native_rate_acceptance.<locals>.accept':
            raise RuntimeError('Cannot discard an unclassified clock predicate: ' + callback)
    reference_duration = duration / factor
    native_duration = ceil_native_duration(reference_duration)
    knots = clock.x / factor
    coefficients = clock.c.copy()
    terminal_hold = max(0.0, native_duration - reference_duration)
    if terminal_hold > 1e-12:
        coefficients = np.concatenate((coefficients,
                                       np.full((len(coefficients), 1), float(clock(duration)))), axis=1)
        knots = np.r_[knots, native_duration]
    else:
        knots = knots.copy()
        knots[-1] = native_duration
    direct_clock = BPoly(coefficients, knots, extrapolate=False)
    bounds = core._Certificate(path, direct_clock).partition()
    blocks = [dict(old_start=0.0, old_end=duration, new_start=0.0,
                   new_end=native_duration, decision='fixed_reference_factor')]
    report = record(
        'reference_timing', str(label), source_duration_s=duration,
        duration_s=native_duration, requested_reference_duration_s=reference_duration,
        requested_factor=factor, saved_s=duration-native_duration,
        terminal_grid_hold_s=terminal_hold, native_timestep_s=DT,
        method='fixed_reference_factor_without_software_speed_caps',
        source_clock_basis='Supplied native/R08 nominal clock; no cap-driven TOPP selection',
        path_preservation='Original geometric polynomial; only its reference clock is composed',
        velocity_limits=None, acceleration_limits=None,
        superseded_velocity_limits=vlim, superseded_acceleration_limits=alim,
        superseded_maximum_rate=maximum_rate,
        superseded_minimum_reference_duration=minimum_duration,
        superseded_protected_reference_intervals=list(protected),
        superseded_velocity_only_callback=callback,
        all_time_velocity_bound=bounds['velocity'],
        all_time_acceleration_bound=bounds['acceleration'],
        clock_knots=direct_clock.x, clock_coefficients=direct_clock.c,
        blocks=blocks, identity_source_clock=identity,
        software_speed_limits_enforced=False, old_topp_certificate_valid=False,
        physical_limits_unchanged=True, physical_acceptance_unchanged=True,
        physical_execution_passed=None)
    return Timing(path, direct_clock, clock, native_duration, duration,
                  blocks, report, identity, factor, reference_duration)


def save_report(directory):
    """Retain the established report locator, but never claim the old optimizer ran."""
    directory = Path(directory)
    report = dict(schema_version=1,
                  implementation='fixed_reference_factor_without_software_speed_caps',
                  requested_motion_factor=FACTOR,
                  native_timestep_s=DT,
                  records=RECORDS,
                  old_topp_and_speed_certificates_valid=False,
                  scope='Source adaptation and reference timing evidence, not actual-motion or physical-success certification')
    path = directory / 'topp_report.json'
    temporary = path.with_suffix('.json.writing')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return path
