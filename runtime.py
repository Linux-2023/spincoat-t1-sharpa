"""Manifest-bound execution of the current native planner; no historical host dependencies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from asset_identity import normalized_xml_sha

ROOT = Path(__file__).resolve().parent
TEXT_SUFFIXES = {'.py', '.json', '.xml', '.mjcf', '.urdf', '.mtl'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def contained(root, relative):
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError('Expected a contained relative path: ' + str(relative))
    result = (root / part).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Path escapes its declared root: ' + str(relative))
    return result


def check_versions(versions):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('This release requires Python 3.11')
    actual = {name: importlib.metadata.version(name) for name in versions}
    changed = {name: {'expected': versions[name], 'actual': actual[name]}
               for name in versions if versions[name] != actual[name]}
    if changed:
        raise RuntimeError('Runtime dependency mismatch: ' + json.dumps(changed))
    return actual


def copy_bound(source, target, expected, replacements):
    if sha(source) != expected:
        raise ValueError('Bound input changed: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError('Two inputs target the same file: ' + str(target))
    if target.suffix.lower() in TEXT_SUFFIXES:
        payload = source.read_bytes()
        for old, new in replacements:
            payload = payload.replace(old, new)
        with target.open('xb') as stream:
            stream.write(payload)
    else:
        shutil.copyfile(source, target)
    return sha(target)


def materialize(manifest, output, assets, run_id):
    workspace = output / 'workspace'
    workspace.mkdir()
    values = {'workspace': str(workspace), 'assets': str(assets), 'run_id': run_id}
    replacements = [(row['old'].format(**values).encode(), row['new'].format(**values).encode())
                    for row in manifest['path_prefixes']]
    replacements.append((str(assets).encode(), str(workspace).encode()))
    identity = {}
    for row in manifest['files']:
        source = contained(ROOT, row['source'])
        target = contained(workspace, row['target'].format(**values))
        identity[target.relative_to(workspace).as_posix()] = copy_bound(source, target, row['sha256'], replacements)
    lock_path = contained(ROOT, manifest['assets_manifest'])
    lock = json.loads(lock_path.read_text())
    receipt_path = assets / 'prepared.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('assets_manifest_sha256') != sha(lock_path) or receipt.get('complete') is not True:
        raise RuntimeError('Run prepare.py for this exact assets.lock.json first')
    for row in lock['files']:
        source = contained(assets, row['target'])
        target = contained(workspace, row['target'])
        identity[row['target']] = copy_bound(source, target, row['sha256'], replacements)
    for row in lock['generated']:
        source = contained(assets, row['target'])
        expected = receipt['generated'][row['target']]['sha256']
        if normalized_xml_sha(source, assets, row['canonical_prefix']) != row['normalized_sha256']:
            raise ValueError('Generated model is not the bound scene: ' + row['target'])
        target = contained(workspace, row['target'])
        identity[row['target']] = copy_bound(source, target, expected, replacements)
    argv = [value.format(**values) for value in manifest['argv']]
    return workspace, argv, identity


def install_io_fence(output, readable_roots):
    output = output.resolve()
    roots = tuple(Path(root).resolve() for root in readable_roots)
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def resolved(value):
        if value is None or isinstance(value, int):
            return None
        return Path(os.fsdecode(value)).resolve()

    def write_allowed(value):
        path = resolved(value)
        if path is not None and path != Path('/dev/null') and not path.is_relative_to(output):
            raise PermissionError('Native execution cannot write outside its output: ' + str(path))

    def hook(event, args):
        if event == 'open':
            value, mode, flags = args
            writing = isinstance(mode, str) and any(character in mode for character in 'wax+')
            writing = writing or (isinstance(flags, int) and bool(flags & write_flags))
            if writing:
                write_allowed(value)
            else:
                path = resolved(value)
                if path is not None and not any(path == root or path.is_relative_to(root) for root in roots):
                    raise PermissionError('Undeclared native input read: ' + str(path))
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime', 'os.truncate'):
            write_allowed(args[0])
        elif event in ('os.rename', 'os.link'):
            write_allowed(args[0])
            write_allowed(args[1])
        elif event in ('os.symlink', 'subprocess.Popen'):
            raise PermissionError('Native execution does not create links or launch subprocesses')
    sys.addaudithook(hook)


def load_plugin(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute_child(config_path):
    config = json.loads(config_path.read_text())
    output, workspace = Path(config['output']), Path(config['workspace'])
    raw = Path(config['raw'])
    os.chdir(workspace)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workspace / 'scripts'))
    sys.argv = config['argv']
    # Match the original order: activate the installed public TOPP dependency
    # before the native read/write fence and ordered planner installation.
    import toppra
    from action_capture import NativeActionCapture
    import recording_adapter
    capture = NativeActionCapture(output / 'action_capture')
    readable = [output, ROOT, Path(config['assets']), Path(sys.prefix), Path(sys.base_prefix),
                Path('/usr/lib'), Path('/lib'), Path('/usr/share'), Path('/etc'),
                Path('/proc'), Path('/sys'), Path('/dev')]
    install_io_fence(output, readable)
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'installed_plugins': [],
              'terminal_status': 'running', 'exception': None, 'capture_complete': False}
    code = 0
    try:
        for plugin in config['plugins']:
            name = plugin['module']
            if not name.isidentifier() or plugin['entrypoint'] != 'install':
                raise ValueError('Invalid declared planner entrypoint')
            module = load_plugin(name, workspace / 'scripts' / (name + '.py'))
            module.install()
            report['installed_plugins'].append(name)
        runner = importlib.import_module(config['entrypoint'])
        recording_adapter.install(capture)
        runner.main()
    except BaseException as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        report['exception'] = {'type': type(error).__name__, 'message': str(error)}
        report['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        try:
            scope_path = raw / 'fullrate_observer_scope.json'
            scope = json.loads(scope_path.read_text()) if scope_path.is_file() else {}
            captured = capture.finalize(scope, raw_directory=raw, native_returned=code == 0)
            report['capture'] = captured
            report['capture_complete'] = captured.get('capture_complete') is True
        except BaseException as error:
            report['capture_error'] = {'type': type(error).__name__, 'message': str(error)}
            code = code or 1
            traceback.print_exc()
        result_path = raw / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        report['native_success'] = result.get('success') is True
        report['terminal_status'] = 'completed' if code == 0 and report['native_success'] and report['capture_complete'] else 'failed'
        if code == 0 and report['terminal_status'] != 'completed':
            code = 2
        report.update(exit_code=code, finished_at_utc=datetime.now(timezone.utc).isoformat())
        save(output / 'child_execution.json', report)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--assets', type=Path, default=ROOT / '.cache/runtime_inputs')
    parser.add_argument('--run-id', default='reference')
    parser.add_argument('--child', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return execute_child(args.child.resolve())
    if args.output is None:
        parser.error('--output is required')
    if not args.run_id or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in args.run_id):
        parser.error('--run-id must contain only letters, digits, underscore or hyphen')
    manifest_path = ROOT / 'runtime_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    versions = check_versions(manifest['runtime_versions'])
    output, assets = args.output.resolve(), args.assets.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace, argv, materialized = materialize(manifest, output, assets, args.run_id)
    raw = workspace / 'logs/spincoat/t1' / args.run_id
    config = dict(output=str(output), workspace=str(workspace), raw=str(raw), assets=str(assets),
                  argv=argv, plugins=manifest['plugins'], entrypoint=manifest['entrypoint'])
    save(output / 'execution_config.json', config)
    metadata = dict(schema_version=1, case_id=args.run_id, baseline_tag='integrated_human_v31_08_preturn_margin',
                    terminal_status='running', exit_code=None, simulation_started=False,
                    physics_step_count=0, runtime_versions=versions, python_version=sys.version,
                    source_manifest_sha256=sha(manifest_path), source_identity=manifest['source_identity'],
                    materialized_sha256=materialized, strict_grip_certified=False,
                    paths={'raw_output': str(raw), 'result': str(raw / 'result.json'),
                           'actual_clock': str(raw / 'fullrate_observer_scope.json'),
                           'action_capture_manifest': str(output / 'action_capture/manifest.json')})
    save(output / 'execution_metadata.json', metadata)
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith('T1_') or key.startswith('PYTHON'):
            environment.pop(key)
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       MPLCONFIGDIR=str(output / 'cache/matplotlib'), XDG_CACHE_HOME=str(output / 'cache'),
                       TMPDIR=str(output / 'tmp'))
    environment.setdefault('MUJOCO_GL', 'egl')
    (output / 'tmp').mkdir()
    command = [sys.executable, '-B', str(ROOT / 'run.py'), '--child', str(output / 'execution_config.json')]
    lease_fds = (int(environment['V34_COMPUTE_SLOT_FD']),) if 'V34_COMPUTE_SLOT_FD' in environment else ()
    with (output / 'execution.log').open('xb') as log:
        code = subprocess.run(command, cwd=workspace, env=environment, stdout=log,
                              stderr=subprocess.STDOUT, pass_fds=lease_fds).returncode
    child_path = output / 'child_execution.json'
    child = json.loads(child_path.read_text()) if child_path.is_file() else {}
    capture = child.get('capture', {})
    metadata.update(exit_code=code, terminal_status=child.get('terminal_status', 'infrastructure_error'),
                    simulation_started=bool(capture.get('physics_steps', 0)),
                    physics_step_count=capture.get('physics_steps', 0),
                    native_success=child.get('native_success', False), exception=child.get('exception'),
                    capture_complete=child.get('capture_complete', False),
                    finished_at_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'execution_metadata.json', metadata)
    print(json.dumps({'output': str(output), 'terminal_status': metadata['terminal_status'],
                      'native_success': metadata['native_success'], 'physics_steps': metadata['physics_step_count'],
                      'strict_grip_certified': False}), flush=True)
    return code
