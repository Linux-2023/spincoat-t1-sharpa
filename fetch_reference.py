#!/usr/bin/env python3
"""Download and verify the single preserved successful trajectory; no model binaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from prepare import child, download

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path,
                        help='Optional already-downloaded release archive; its exact hash is required')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'reference/manifest.json').read_text())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Reference output must be a new directory')
    record = lock['archive']
    archive = args.archive.resolve() if args.archive else ROOT / '.cache' / record['name']
    if args.archive is None:
        download(record['url'], archive, record['sha256'], record['bytes'])
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if archive.stat().st_size != record['bytes'] or digest != record['sha256']:
        raise ValueError('Reference archive differs from the released identity')
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(lock['files']):
            raise ValueError('Archive members differ from the reference manifest')
        for member in members:
            expected = lock['files'][member.name]
            child(output, member.name)
            if not member.isfile() or member.size != expected['bytes']:
                raise ValueError('Unexpected reference entry type or size: ' + member.name)
        output.mkdir(parents=True)
        for member in members:
            target = child(output, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            hasher = hashlib.sha256()
            with bundle.extractfile(member) as source, target.open('xb') as destination:
                while block := source.read(1024 * 1024):
                    hasher.update(block)
                    destination.write(block)
            if hasher.hexdigest() != lock['files'][member.name]['sha256']:
                raise ValueError('Reference member hash mismatch: ' + member.name)
    print(json.dumps({'reference': str(output), 'files_verified': len(members),
                      'source_episode': lock['source_episode'], 'physics_steps_executed': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
