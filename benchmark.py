#!/usr/bin/env python3
"""Serve one fresh three-camera benchmark attempt; never launches an agent policy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / ".cache/runtime_inputs")
    parser.add_argument("--output", type=Path, required=True, help="New evaluator-owned directory; never reuse an old attempt")
    parser.add_argument("--case", type=Path, default=ROOT / "benchmark/cases/reference.json")
    parser.add_argument("--task", type=Path, default=ROOT / "benchmark/task_contract.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-file", type=Path, help="Existing bearer secret. Omit to generate a new private participant.token")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be in 0..65535")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from benchmark.recording import save_json
    from runtime import check_versions
    manifest = json.loads((ROOT / "runtime_manifest.json").read_text())
    runtime_versions = check_versions(manifest["runtime_versions"])
    task = json.loads(args.task.read_text())
    case = json.loads(args.case.read_text())
    token = args.token_file.read_text().strip() if args.token_file else secrets.token_urlsafe(32)
    if len(token) <= 20 or not token.isascii():
        parser.error("token must be ASCII and longer than 20 characters")
    canonical_task = json.loads((ROOT / "benchmark/task_contract.json").read_text())
    definition = {key: value for key, value in task.items() if key != "budgets"}
    expected = {key: value for key, value in canonical_task.items() if key != "budgets"}
    if definition != expected:
        parser.error("Task definition must match the installed judge; only budgets may differ")
    from replay_adapter import check_output
    assets = args.assets.resolve()
    protected = [ROOT / name for name in (
        "native", "native_fluid", "inputs", "assets", "bootstrap",
        "replay_support", "reference", "benchmark", "tests")]
    protected.extend((args.task, args.case, assets))
    if (ROOT / ".git").exists():
        protected.append(ROOT / ".git")
    output = check_output(args.output, protected)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, value in (("task.json", task), ("case.json", case)):
        save_json(output / name, value)
    token_path = output / "participant.token"
    with token_path.open("x") as stream:
        os.chmod(token_path, 0o600)
        stream.write(token + "\n")
    os.environ["MPLCONFIGDIR"] = str(output / "cache/matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(output / "cache")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    source_files = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (ROOT / "benchmark").rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts}
    for name in ("benchmark.py", "runtime.py", "model_identity.py", "asset_identity.py",
                 "replay_adapter.py", "requirements.txt", "assets.lock.json", "replay_manifest.json"):
        source_files[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    save_json(output / "source_manifest.json", {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": source_files, "runtime_manifest_sha256": hashlib.sha256((ROOT / "runtime_manifest.json").read_bytes()).hexdigest(),
        "runtime_versions": runtime_versions,
        "task_sha256": hashlib.sha256(args.task.read_bytes()).hexdigest(),
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "no_agent_workflow_installed": True})
    from benchmark.backend import Backend
    from benchmark.rendering import LiveC14
    from benchmark.session import Session
    from benchmark.server import BenchmarkServer
    backend = renderer = session = server = None
    try:
        backend = Backend(assets, output / "backend", case)
        renderer = LiveC14(backend, output / "rendering")
        session = Session(backend, renderer, output, task)
        server = BenchmarkServer((args.host, args.port), session, token)
        host, port = server.server_address[:2]
        connection = {"protocol": "spincoat-agent-benchmark/1", "url": f"http://{host}:{port}",
                      "token_file": str(token_path), "output": str(output),
                      "task_id": task["id"], "status": "ready"}
        save_json(output / "connection.json", connection)
        print(json.dumps(connection), flush=True)
        server.run_episode()
        return 0 if session.result and session.result["infrastructure_valid"] else 2
    except KeyboardInterrupt:
        if session is not None:
            session.finish(reason="interrupted")
        return 130
    except Exception as error:
        save_json(output / "infrastructure_error.json", {
            "type": type(error).__name__, "message": str(error),
            "task_success_claimed": False, "input_state_not_replayed": True})
        raise
    finally:
        if server is not None:
            server.server_close()
        if session is not None:
            session.close()
        else:
            if renderer is not None:
                renderer.close()
            if backend is not None:
                backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
