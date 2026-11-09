"""Predeclare attempts and aggregate every expected result without cherry-picking.

Run with ``python -m benchmark.aggregate plan`` before evaluation, then
``python -m benchmark.aggregate score`` against evaluator-owned run directories.
This command never launches an agent, changes a case, or advances physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .recording import file_sha256, json_bytes, save_json


def read_json(path):
    return json.loads(Path(path).read_text())


def plan(task_path, case_paths, *, repetitions):
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("Positive repetition count required")
    task = read_json(task_path)
    cases = []
    for path in case_paths:
        path = Path(path).resolve()
        cases.append({"case_sha256": file_sha256(path), "case_name": path.name})
    if not cases:
        raise ValueError("At least one declared case required")
    attempts = []
    for index, case in enumerate(cases):
        for repetition in range(repetitions):
            attempts.append({"attempt_id": f"case-{index:03d}-rep-{repetition:03d}", **case})
    return {"schema_version": 1, "task_id": task["id"],
            "task_sha256": hashlib.sha256(json_bytes(task)).hexdigest(),
            "attempts": attempts, "repetitions": repetitions,
            "policy": "Every declared attempt is required. Missing, invalid or infrastructure-failed attempts remain visible; no official score until all are valid.",
            "scope": "Case validity and held-out status must be established separately; this manifest alone does not prove either."}


def bounded(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Expected a contained relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Result path escaped its root")
    return path


def aggregate(suite, root):
    root = Path(root).resolve()
    identifiers = [entry["attempt_id"] for entry in suite["attempts"]]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("A nonempty suite with unique attempt IDs is required")
    rows = []
    source_identities = set()
    for entry in suite["attempts"]:
        directory = bounded(root, entry["attempt_id"])
        row = {"attempt_id": entry["attempt_id"], "status": "missing",
               "functional_success": False, "reason": None}
        try:
            if not (directory / "result.json").is_file():
                if (directory / "infrastructure_error.json").is_file():
                    row.update(status="infrastructure_error", reason="initialization_or_runtime_failure")
                rows.append(row)
                continue
            result = read_json(directory / "result.json")
            identity = read_json(directory / "source_manifest.json")
            manifest = read_json(directory / "evidence/manifest.json")
            if file_sha256(directory / "source_manifest.json") != result["source_manifest_sha256"]:
                raise ValueError("Source/case manifest changed after the attempt started")
            if result["task_sha256"] != suite["task_sha256"]:
                raise ValueError("Task identity differs from preregistration")
            if identity["case_sha256"] != entry["case_sha256"]:
                raise ValueError("Case identity differs from preregistration")
            evidence = directory / "evidence"
            for name, expected in manifest["files"].items():
                if file_sha256(bounded(evidence, name)) != expected:
                    raise ValueError("Evidence hash mismatch: " + name)
            if read_json(evidence / "result.json") != result:
                raise ValueError("Submitted result differs from sealed evidence")
            expected_tick = 0
            for chunk in manifest["chunks"]:
                if chunk["first_tick"] != expected_tick or chunk["last_tick"] - expected_tick != chunk["steps"]:
                    raise ValueError("Native transition chunks are not contiguous")
                if file_sha256(bounded(evidence, chunk["path"])) != chunk["sha256"]:
                    raise ValueError("Native transition chunk hash mismatch")
                expected_tick = chunk["last_tick"]
            if expected_tick != result["physics_steps"] or expected_tick != manifest["physics_steps"]:
                raise ValueError("Native physics step totals disagree")
            source_identity = hashlib.sha256(json_bytes({
                "files": identity["files"], "runtime_manifest": identity["runtime_manifest_sha256"]})).hexdigest()
            source_identities.add(source_identity)
            valid = result["infrastructure_valid"] is True
            success = result["assessment"]["outcome"]["functional"] is True
            row.update(status="valid" if valid else "infrastructure_error",
                       functional_success=bool(valid and success), reason=result["termination"],
                       physics_steps=result["physics_steps"],
                       simulation_seconds=result["simulation_seconds"],
                       wall_seconds=result["wall_seconds"],
                       action_requests=result["accepted_action_requests"],
                       observations=result["observations"], source_identity=source_identity)
        except (ValueError, KeyError, TypeError, OSError) as error:
            row.update(status="invalid_evidence", reason=str(error), functional_success=False)
        rows.append(row)
    complete = all(row["status"] == "valid" for row in rows)
    comparable = len(source_identities) == 1
    successes = sum(row["functional_success"] for row in rows)
    return {"schema_version": 1, "task_id": suite["task_id"],
            "planned_attempts": len(rows), "valid_attempts": sum(row["status"] == "valid" for row in rows),
            "missing_attempts": sum(row["status"] == "missing" for row in rows),
            "infrastructure_errors": sum(row["status"] == "infrastructure_error" for row in rows),
            "invalid_evidence": sum(row["status"] == "invalid_evidence" for row in rows),
            "functional_successes": successes, "complete": complete, "same_source": comparable,
            "official_functional_success_rate": successes / len(rows) if complete and comparable else None,
            "attempts": rows, "strict_grip_certified": False, "hardware_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    declare = commands.add_parser("plan")
    declare.add_argument("--task", type=Path, required=True)
    declare.add_argument("--cases", type=Path, nargs="+", required=True)
    declare.add_argument("--repetitions", type=int, default=1)
    declare.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--suite", type=Path, required=True)
    score.add_argument("--runs", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new; never replace a preregistration or past result")
    result = plan(args.task, args.cases, repetitions=args.repetitions) if args.command == "plan" else aggregate(read_json(args.suite), args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
