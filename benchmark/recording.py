"""Evaluator-owned, lossless native transitions and append-only episode evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def json_bytes(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("xb") as stream:
        stream.write(json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Recorder:
    def __init__(self, directory, *, chunk_steps=1000):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.chunk_steps = chunk_steps
        self.transitions = []
        self.chunk_index = 0
        self.total_steps = 0
        self.sealed = False
        self.journal = (self.directory / "interactions.jsonl").open("xb")
        self.events = (self.directory / "events.jsonl").open("xb")
        self.chunks = []

    def append(self, kind, **fields):
        if self.sealed:
            raise RuntimeError("Episode evidence is sealed")
        self.journal.write(json_bytes({"type": kind, **fields}) + b"\n")
        self.journal.flush()

    @staticmethod
    def state(data):
        return {name: np.array(getattr(data, name), dtype="float64", copy=True)
                for name in ("qpos", "qvel", "act", "qacc_warmstart", "ctrl",
                             "qfrc_applied", "xfrc_applied")} | {"time": float(data.time)}

    def boundary(self, name, tick, data):
        if name not in ("initial", "final") or self.sealed:
            raise ValueError("Boundary must be an unsealed initial/final native state")
        path = self.directory / (name + "_state.npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, tick=np.asarray([tick], dtype="int64"), **self.state(data))
        return file_sha256(path)

    def transition(self, tick, before, after, requested, events):
        if self.sealed:
            raise RuntimeError("Episode evidence is sealed")
        self.transitions.append((int(tick), before, after, np.array(requested, dtype="float64", copy=True)))
        self.total_steps += 1
        for item in events:
            event_tick = int(tick) if item.get("phase") in ("pre_control", "pre_step") else int(tick) + 1
            self.events.write(json_bytes({"tick": event_tick, "event": item}) + b"\n")
        if len(self.transitions) >= self.chunk_steps:
            self.flush()

    def flush(self):
        if not self.transitions:
            return
        rows = self.transitions
        payload = {
            "tick": np.asarray([row[0] for row in rows] + [rows[-1][0] + 1], dtype="int64"),
            "requested_ctrl": np.stack([row[3] for row in rows]),
            "applied_ctrl": np.stack([row[2]["ctrl"] for row in rows]),
            "qfrc_applied": np.stack([row[2]["qfrc_applied"] for row in rows]),
            "xfrc_applied": np.stack([row[2]["xfrc_applied"] for row in rows]),
            "time": np.asarray([row[1]["time"] for row in rows] + [rows[-1][2]["time"]], dtype="float64"),
        }
        for field in ("qpos", "qvel", "act", "qacc_warmstart"):
            payload[field] = np.stack([row[1][field] for row in rows] + [rows[-1][2][field]])
        name = f"native_{self.chunk_index:06d}.npz"
        path = self.directory / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        self.chunks.append({"path": name, "sha256": file_sha256(path),
                            "first_tick": int(payload["tick"][0]),
                            "last_tick": int(payload["tick"][-1]), "steps": len(rows)})
        self.chunk_index += 1
        self.transitions = []
        self.events.flush()

    def seal(self, result):
        if self.sealed:
            return
        self.flush()
        self.journal.flush()
        self.events.flush()
        os.fsync(self.journal.fileno())
        os.fsync(self.events.fileno())
        self.journal.close()
        self.events.close()
        save_json(self.directory / "result.json", result)
        save_json(self.directory / "manifest.json", {
            "schema_version": 1, "physics_steps": self.total_steps, "chunks": self.chunks,
            "state_semantics": "state[n] precedes applied_ctrl[n]; terminal state is retained",
            "precision": "Native float64, no policy projection; NPZ allow_pickle=False",
            "files": {name: file_sha256(self.directory / name)
                      for name in ("interactions.jsonl", "events.jsonl", "result.json",
                                   "initial_state.npz", "final_state.npz")},
        })
        self.sealed = True

    def close(self):
        if not self.sealed:
            self.flush()
            self.journal.close()
            self.events.close()
