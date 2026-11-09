"""Language-neutral agent HTTP API, with one physics/EGL owner thread.

This is an API isolation boundary, not a sandbox for code executed as the server's
OS user. Run untrusted participants under a different user/container or host.
"""
from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import unquote, urlsplit

from .recording import json_bytes
from .session import ProtocolError


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON value: " + value)


class BenchmarkServer(HTTPServer):
    def __init__(self, address, session, token, *, max_body_bytes=4 * 1024 * 1024):
        if not isinstance(token, str) or len(token) <= 20 or not token.isascii():
            raise ValueError("Use an ASCII bearer token longer than 20 characters")
        self.session = session
        self.token = token
        self.max_body_bytes = max_body_bytes
        super().__init__(address, Handler)
        self.timeout = 0.5

    def run_episode(self):
        """Run on the same thread that constructed backend and EGL renderer."""
        try:
            while self.session.lifecycle != "finished":
                self.handle_request()
                self.session.status()
                if self.session.clock() - self.session.started_at >= self.session.max_wall_seconds:
                    self.session.finish(reason="wall_time_budget")
        finally:
            self.server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "SpincoatBenchmark/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format, *args):
        # Do not print bearer tokens or arbitrary agent paths into evaluator logs.
        pass

    def _send(self, status, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        value = self.headers.get("Authorization", "")
        expected = "Bearer " + self.server.token
        if not hmac.compare_digest(value.encode(), expected.encode()):
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise ProtocolError("invalid_encoding", "Use Content-Length, not chunked transfer")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            raise ProtocolError("invalid_length", "One Content-Length header is required")
        size = int(lengths[0])
        if size > self.server.max_body_bytes:
            raise ProtocolError("request_too_large", "Request exceeds transport byte limit", 413)
        if self.headers.get_content_type() != "application/json":
            raise ProtocolError("invalid_content_type", "Use application/json", 415)
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ProtocolError("incomplete_request", "Request body was not completely received")
        try:
            return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=strict_object)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ProtocolError("invalid_json", "Strict finite JSON with unique keys is required") from error

    def _dispatch(self, method):
        if not self._authorized():
            return
        parts = urlsplit(self.path)
        if parts.query or parts.fragment:
            raise ProtocolError("invalid_path", "Query and fragment parameters are not supported")
        path = parts.path
        session = self.server.session
        if method == "GET":
            if path == "/v1/info":
                value = session.info()
            elif path == "/v1/status":
                value = session.status()
            elif path.startswith("/v1/receipts/"):
                value = session.receipt(unquote(path[len("/v1/receipts/"):]))
            elif path.startswith("/v1/images/"):
                image = session.image(path)
                self._send(200, image.read_bytes(), "image/png")
                return
            else:
                raise ProtocolError("unknown_endpoint", "No such participant endpoint", 404)
        else:
            payload = self._body()
            if path == "/v1/act":
                value = session.act(payload)
            elif path in ("/v1/observe", "/v1/finish"):
                if payload != {}:
                    raise ProtocolError("invalid_request", "This endpoint accepts only an empty object")
                value = session.observe() if path.endswith("observe") else session.finish()
            else:
                raise ProtocolError("unknown_endpoint", "No such participant endpoint", 404)
        self._send(200, value)

    def _handle(self, method):
        try:
            self._dispatch(method)
        except ProtocolError as error:
            try:
                self._send(error.status, {"error": error.code, "message": str(error)})
            except (BrokenPipeError, ConnectionResetError):
                pass
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # A disconnected client never replays or rolls back an accepted action.
            pass
        except Exception as error:
            session = self.server.session
            if session.lifecycle != "finished":
                session.lifecycle, session.stop_reason = "stopped", "infrastructure_error"
                session.recorder.append("server_error", tick=int(session.backend.tick),
                                        error={"type": type(error).__name__, "message": str(error)})
            try:
                self._send(500, {"error": "infrastructure_error",
                                 "message": "Evaluator retained diagnostics; reconcile the original request ID."})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")
