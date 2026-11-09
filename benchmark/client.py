"""Optional standard-library client; any HTTP-capable agent workflow is supported.

There are no model, planning, inference-frequency or skill dependencies here.
Actions are never retried automatically: a lost response is reconciled using the
original request ID, rather than creating a second physical request.
"""
from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


class BenchmarkError(RuntimeError):
    def __init__(self, status, response):
        self.status = status
        self.response = response
        super().__init__(f"Benchmark HTTP {status}: {response}")


class Client:
    def __init__(self, url, token, *, timeout=300):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, path, payload=None, *, binary=False):
        headers = {"Authorization": "Bearer " + self.token}
        body = None
        if payload is not None:
            body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.url + path, data=body, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = response.read()
        except HTTPError as error:
            raw = error.read()
            try:
                detail = json.loads(raw)
            except (ValueError, UnicodeError):
                detail = {"error": "http_error", "message": raw.decode(errors="replace")}
            raise BenchmarkError(error.code, detail) from error
        return value if binary else json.loads(value)

    def info(self):
        """Read the public task, action schema and observation contract."""
        return self._request("/v1/info")

    def observe(self):
        """Capture synchronized current images; does not advance physics."""
        return self._request("/v1/observe", {})

    def image(self, observation, camera):
        """Fetch the exact PNG named by a delivered observation."""
        path = observation["images"][camera]["url"]
        if not path.startswith("/v1/images/") or ".." in path:
            raise ValueError("Not a benchmark observation image path")
        return self._request(path, binary=True)

    def act(self, actions, *, request_id=None):
        """Execute a list of {controls: vector-or-name-map, steps: integer}.

        The caller may choose arbitrary batch lengths and observation times.
        Retain the request ID before sending when implementing reconnects.
        """
        return self._request("/v1/act", {
            "request_id": request_id or str(uuid4()), "actions": actions})

    def receipt(self, request_id):
        """Reconcile an already submitted action without resubmitting it."""
        from urllib.parse import quote
        return self._request("/v1/receipts/" + quote(request_id, safe=""))

    def status(self):
        """Read clocks and lifecycle only, not private task state."""
        return self._request("/v1/status")

    def finish(self):
        """Irreversibly submit this attempt and receive its measured outcome."""
        return self._request("/v1/finish", {})
