#!/usr/bin/env python3
"""Acquire pinned upstream assets and rebuild the robot locally; no redistributed robot model."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from asset_identity import normalized_xml_sha

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def child(root, relative):
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError('Expected a contained relative path: ' + str(relative))
    path = (root / part).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes its root')
    return path


def download(url, destination, digest, size):
    if not url.startswith('https://'):
        raise ValueError('Asset downloads require HTTPS')
    if destination.is_file():
        if destination.stat().st_size == size and sha(destination) == digest:
            return
        raise ValueError('Existing asset differs from the locked identity: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.downloading')
    hasher, count = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=90) as response, temporary.open('xb') as stream:
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > size:
                raise ValueError('Download exceeds its locked size')
            hasher.update(block)
            stream.write(block)
    if count != size or hasher.hexdigest() != digest:
        raise ValueError('Downloaded asset hash or size mismatch: ' + str(destination))
    os.replace(temporary, destination)


def checkout(record, assets):
    destination = child(assets, record['target'])
    environment = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1', 'GIT_TERMINAL_PROMPT': '0'}
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', '--depth', '1',
                        record['url'], str(destination)], check=True, env=environment)
        subprocess.run(['git', '-C', str(destination), 'fetch', '--depth', '1', 'origin', record['revision']],
                       check=True, env=environment)
        if record.get('sparse_paths'):
            subprocess.run(['git', '-C', str(destination), 'sparse-checkout', 'set', '--cone',
                            *record['sparse_paths']], check=True, env=environment)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', record['revision']],
                       check=True, env=environment)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    origin = subprocess.check_output(['git', '-C', str(destination), 'remote', 'get-url', 'origin'], text=True).strip()
    if actual != record['revision'] or origin.rstrip('/') != record['url'].rstrip('/'):
        raise ValueError('Existing upstream checkout is not the requested source: ' + str(destination))
    return destination


def recover_lfs(record, repository):
    requests = []
    for folder in record.get('lfs_paths', []):
        for path in child(repository, folder).rglob('*'):
            if not path.is_file() or path.stat().st_size >= 1024:
                continue
            if not path.resolve().is_relative_to(repository.resolve()):
                raise ValueError('Upstream LFS path escapes its checkout')
            value = path.read_bytes()
            if not value.startswith(b'version https://git-lfs.github.com/spec/v1'):
                continue
            lines = value.decode().splitlines()
            digest = next(line.split('sha256:', 1)[1] for line in lines if line.startswith('oid '))
            size = int(next(line.split()[1] for line in lines if line.startswith('size ')))
            relative = path.relative_to(repository).as_posix()
            url = 'https://media.githubusercontent.com/media/' + record['repository'] + '/' + record['revision'] + '/' + urllib.parse.quote(relative)
            requests.append((path, value, url, digest, size))

    def fetch(request):
        path, pointer, url, digest, size = request
        temporary = path.with_name(path.name + '.lfs-object')
        download(url, temporary, digest, size)
        if path.read_bytes() != pointer:
            raise RuntimeError('LFS pointer changed during acquisition')
        os.replace(temporary, path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, requests))


def normalized_generated(path, assets, record):
    return normalized_xml_sha(path, assets, record['canonical_prefix'])


def prepare(assets):
    lock_path = ROOT / 'assets.lock.json'
    lock = json.loads(lock_path.read_text())
    assets.mkdir(parents=True, exist_ok=True)
    for upstream in lock['upstreams']:
        repository = checkout(upstream, assets)
        recover_lfs(upstream, repository)
    for record in lock['files']:
        destination = child(assets, record['target'])
        if 'source' in record:
            source = child(ROOT, record['source'])
            if sha(source) != record['sha256']:
                raise ValueError('Bundled input changed: ' + record['source'])
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        elif 'url' in record:
            download(record['url'], destination, record['sha256'], record['bytes'])
        if not destination.is_file() or destination.stat().st_size != record['bytes'] or sha(destination) != record['sha256']:
            raise ValueError('Prepared asset does not match the lock: ' + record['target'])
    for record in lock.get('bootstrap_files', []):
        path = child(assets, record['target'])
        if sha(path) != record['sha256']:
            raise ValueError('Upstream model-generation input changed: ' + record['target'])
    builder = lock['builder']
    source = child(ROOT, builder['source'])
    if sha(source) != builder['sha256']:
        raise ValueError('Robot builder source changed')
    script = child(assets, 'scripts/build_t1_sharpa_preview.py')
    script.parent.mkdir(parents=True, exist_ok=True)
    if script.exists() and sha(script) != builder['sha256']:
        raise ValueError('Existing robot builder is not this release')
    if not script.exists():
        shutil.copyfile(source, script)
    missing = [record for record in lock['generated'] if not child(assets, record['target']).is_file()]
    if missing:
        (assets / 'research/hardware/generated').mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        environment.setdefault('MUJOCO_GL', 'egl')
        subprocess.run([sys.executable, '-B', str(script)], check=True, env=environment)
    generated = {}
    for record in lock['generated']:
        path = child(assets, record['target'])
        normalized = normalized_generated(path, assets, record)
        if normalized != record['normalized_sha256']:
            raise ValueError('Rebuilt robot differs from the validated source model: ' + record['target'])
        generated[record['target']] = {'sha256': sha(path), 'normalized_sha256': normalized}
    receipt = {'schema_version': 1, 'complete': True, 'assets_manifest_sha256': sha(lock_path),
               'generated': generated, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
               'upstream_models_reconstructed_locally': True, 'task_physics_executed': False}
    temporary = assets / 'prepared.json.writing'
    with temporary.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    os.replace(temporary, assets / 'prepared.json')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=ROOT / '.cache/runtime_inputs')
    args = parser.parse_args()
    print(json.dumps(prepare(args.assets.resolve()), indent=2))


if __name__ == '__main__':
    main()
