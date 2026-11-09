"""One evaluator-owned attempt, independent of the participant's agent workflow."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from .recording import Recorder, file_sha256, json_bytes, save_json

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class ProtocolError(ValueError):
    def __init__(self, code, message, status=400):
        self.code, self.status = code, status
        super().__init__(message)


class Session:
    def __init__(self, backend, renderer, directory, task_spec, *, clock=time.monotonic):
        self.backend, self.renderer = backend, renderer
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.spec = task_spec
        self.clock, self.started_at = clock, clock()
        self.schema = backend.control_schema()
        self.actuators = self.schema["actuators"]
        self.name_to_index = {row["name"]: i for i, row in enumerate(self.actuators)}
        if len(self.name_to_index) != len(self.actuators):
            raise ValueError("Actuator names are not unique")
        self.requested = np.array(self.schema["initial_controls"], dtype="float64")
        self.max_steps = task_spec["budgets"]["max_physics_steps"]
        self.max_wall_seconds = task_spec["budgets"]["max_wall_seconds"]
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("Positive max_physics_steps required")
        if not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0:
            raise ValueError("Positive finite max_wall_seconds required")
        self.lifecycle = "running"
        self.stop_reason = None
        self.result = None
        self.receipts = {}
        self.image_paths = {}
        self.observation_count = 0
        self.accepted_requests = 0
        self.rejected_requests = 0
        self.source_manifest_sha256 = file_sha256(self.directory / "source_manifest.json")
        self.recorder = Recorder(self.directory / "evidence")
        self.recorder.boundary("initial", int(self.backend.tick), self.backend.data)
        self.recorder.append("episode_started", task_sha256=hashlib.sha256(json_bytes(task_spec)).hexdigest(),
                             action_schema=self.schema, task=task_spec,
                             source_manifest_sha256=self.source_manifest_sha256)

    def _budget(self):
        if self.lifecycle != "running":
            return
        if self.backend.tick >= self.max_steps:
            self.lifecycle, self.stop_reason = "stopped", "physics_budget"
        elif self.clock() - self.started_at >= self.max_wall_seconds:
            self.lifecycle, self.stop_reason = "stopped", "wall_time_budget"

    def status(self):
        self._budget()
        return {"lifecycle": self.lifecycle, "tick": int(self.backend.tick),
                "t_ns": int(self.backend.tick * self.backend.dt_ns),
                "elapsed_wall_seconds": self.clock() - self.started_at,
                "remaining_physics_steps": max(0, self.max_steps - self.backend.tick),
                "accepted_action_requests": self.accepted_requests,
                "rejected_action_requests": self.rejected_requests}

    def info(self):
        return {"protocol": "spincoat-agent-benchmark/1",
                "task": self.spec, "budgets": self.spec["budgets"],
                "action_schema": self.schema, "cameras": list(CAMERAS),
                "physics_dt_ns": int(self.backend.dt_ns),
                "task_sha256": hashlib.sha256(json_bytes(self.spec)).hexdigest(),
                "scheduling": "Physics advances only for accepted actions; observation and model waiting do not step.",
                "workflow_constraints": None}

    def _validate(self, actions):
        if not isinstance(actions, list) or not actions:
            raise ProtocolError("invalid_actions", "Nonempty actions list required")
        target = self.requested.copy()
        validated = []
        for row in actions:
            if not isinstance(row, dict) or set(row) != {"controls", "steps"}:
                raise ProtocolError("invalid_action", "Each action has exactly controls and steps")
            steps = row["steps"]
            if type(steps) is not int or steps <= 0 or steps > self.max_steps:
                raise ProtocolError("invalid_steps", "steps must be a positive integer within the episode physics budget")
            controls = row["controls"]
            if isinstance(controls, dict):
                unknown = set(controls) - self.name_to_index.keys()
                if unknown:
                    raise ProtocolError("unknown_actuator", "Unknown actuator names: " + ", ".join(sorted(unknown)))
                for name, value in controls.items():
                    if type(value) not in (int, float):
                        raise ProtocolError("invalid_control", "Controls must be finite numbers, not booleans or strings")
                    try:
                        target[self.name_to_index[name]] = value
                    except OverflowError as error:
                        raise ProtocolError("nonfinite_control", "Control exceeds float64 range") from error
            elif isinstance(controls, list) and len(controls) == len(self.actuators):
                if any(type(value) not in (int, float) for value in controls):
                    raise ProtocolError("invalid_control", "Controls must be finite numbers")
                try:
                    target = np.asarray(controls, dtype="float64")
                except OverflowError as error:
                    raise ProtocolError("nonfinite_control", "Control exceeds float64 range") from error
            else:
                raise ProtocolError("invalid_controls", "Use a named actuator map or the complete ordered control vector")
            if not np.isfinite(target).all():
                raise ProtocolError("nonfinite_control", "Every control must be finite")
            for index, actuator in enumerate(self.actuators):
                if actuator["limited"] and not actuator["range"][0] <= target[index] <= actuator["range"][1]:
                    raise ProtocolError("control_out_of_range", "Out-of-range control for " + actuator["name"])
            validated.append((target.copy(), steps))
        if sum(steps for _, steps in validated) > self.max_steps:
            raise ProtocolError("invalid_steps", "A request cannot exceed the entire episode physics budget")
        return validated

    def act(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"request_id", "actions"}:
            raise ProtocolError("invalid_request", "request_id and actions are required; no extra fields")
        request_id = payload["request_id"]
        if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
            raise ProtocolError("invalid_request_id", "Use 1-128 ASCII letters, digits, underscores or hyphens")
        try:
            digest = hashlib.sha256(json_bytes(payload)).hexdigest()
        except (ValueError, TypeError, OverflowError) as error:
            raise ProtocolError("invalid_json", "Request contains nonfinite or unsupported values") from error
        if request_id in self.receipts:
            previous = self.receipts[request_id]
            if previous["request_sha256"] != digest:
                raise ProtocolError("request_id_conflict", "This request_id already names different content", 409)
            return previous
        self._budget()
        if self.lifecycle != "running":
            raise ProtocolError("episode_not_running", "No new actions after the episode stops", 409)
        try:
            validated = self._validate(payload["actions"])
        except ProtocolError as error:
            self.rejected_requests += 1
            self.recorder.append("action_rejected", request_id=request_id, request_sha256=digest,
                                 tick=int(self.backend.tick), code=error.code, request=payload)
            raise
        start = int(self.backend.tick)
        self.accepted_requests += 1
        self.recorder.append("action_accepted", request=payload, request_sha256=digest, start_tick=start)
        failure = None
        try:
            for requested, steps in validated:
                for _ in range(steps):
                    self._budget()
                    if self.lifecycle != "running":
                        break
                    before = self.recorder.state(self.backend.data)
                    tick = int(self.backend.tick)
                    events = self.backend.step(requested)
                    if self.backend.tick != tick + 1:
                        raise RuntimeError("Backend violated the one-native-step contract")
                    self.requested = requested.copy()
                    after = self.recorder.state(self.backend.data)
                    self.recorder.transition(tick, before, after, requested, events)
                    if self.backend.task.terminal:
                        self.lifecycle, self.stop_reason = "stopped", "task_terminal"
                if self.lifecycle != "running":
                    break
        except Exception as error:
            self.lifecycle, self.stop_reason = "stopped", "infrastructure_error"
            failure = {"type": type(error).__name__, "message": str(error)}
            self.recorder.append("backend_error", tick=int(self.backend.tick), error=failure)
        self._budget()
        receipt = {"request_id": request_id, "request_sha256": digest,
                   "start_tick": start, "end_tick": int(self.backend.tick),
                   "planned_steps": sum(steps for _, steps in validated),
                   "advanced_steps": int(self.backend.tick) - start,
                   "lifecycle": self.lifecycle, "infrastructure_error": failure is not None}
        self.receipts[request_id] = receipt
        self.recorder.append("action_completed", **receipt)
        self.recorder.flush()
        return receipt

    def receipt(self, request_id):
        if request_id not in self.receipts:
            raise ProtocolError("unknown_request", "No completed request with this ID", 404)
        return self.receipts[request_id]

    def observe(self):
        if self.lifecycle == "finished":
            raise ProtocolError("episode_sealed", "No new observations after submission", 409)
        self._budget()
        tick = int(self.backend.tick)
        rendered = self.renderer.observe()
        if rendered["tick"] != tick or self.backend.tick != tick or rendered["t_ns"] != tick * self.backend.dt_ns:
            raise RuntimeError("Camera observation is not bound to the current physical tick")
        if set(rendered["images"]) != set(CAMERAS):
            raise RuntimeError("Current observation must contain exactly the three C14 views")
        number = self.observation_count
        folder = self.directory / "observations" / f"{number:06d}"
        folder.mkdir(parents=True, exist_ok=False)
        observation = {"observation_id": number, "tick": tick, "t_ns": rendered["t_ns"],
                       "images": {}, "calibration": rendered["calibration"],
                       "proprioception": self.backend.proprioception()}
        pending_images = {}
        for camera in CAMERAS:
            rgb = rendered["images"][camera]
            if rgb.shape != (720, 1280, 3) or rgb.dtype != np.uint8:
                raise RuntimeError("C14 observations must be native 1280x720 RGB8")
            path = folder / (camera + ".png")
            Image.fromarray(rgb).save(path)
            url = f"/v1/images/{number:06d}/{camera}.png"
            pending_images[url] = path
            observation["images"][camera] = {"url": url, "sha256": file_sha256(path),
                                              "width": 1280, "height": 720}
        save_json(folder / "observation.json", observation)
        self.observation_count += 1
        self.recorder.append("observation_delivered", observation=observation)
        self.image_paths.update(pending_images)
        return observation

    def image(self, path):
        if path not in self.image_paths:
            raise ProtocolError("unknown_image", "Only already delivered observation images are accessible", 404)
        return self.image_paths[path]

    def finish(self, *, reason="submitted"):
        if self.result is not None:
            return self.result
        self._budget()
        assessment = self.backend.outcome()
        result = {"schema_version": 1, "task_id": self.spec["id"],
                       "source_manifest_sha256": self.source_manifest_sha256,
                       "task_sha256": hashlib.sha256(json_bytes(self.spec)).hexdigest(),
                       "termination": self.stop_reason or reason,
                       "infrastructure_valid": self.stop_reason != "infrastructure_error",
                       "physics_steps": int(self.backend.tick),
                       "simulation_seconds": self.backend.tick * self.backend.dt_ns / 1e9,
                       "wall_seconds": self.clock() - self.started_at,
                       "accepted_action_requests": self.accepted_requests,
                       "rejected_action_requests": self.rejected_requests,
                       "observations": self.observation_count, "assessment": assessment,
                       "hardware_certified": False, "agent_workflow_prescribed": False}
        self.recorder.boundary("final", int(self.backend.tick), self.backend.data)
        self.recorder.append("episode_finished", result=result)
        self.recorder.seal(result)
        save_json(self.directory / "result.json", result)
        self.result, self.lifecycle = result, "finished"
        return result

    def close(self):
        if getattr(self, "_closed", False):
            return
        try:
            if self.result is None:
                self.finish(reason="server_closed")
        finally:
            self._closed = True
            try:
                self.renderer.close()
            finally:
                self.backend.close()
                self.recorder.close()
