from __future__ import annotations

import atexit
import json
import logging
import mimetypes
import os
import subprocess
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


WEB_ROOT = Path(__file__).resolve().parent / "web"


def chunks_by_image_id(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for item in items:
        image_id = str(item["s3Id"])
        if image_id not in grouped:
            order.append(image_id)
        grouped[image_id].append(item)
    return [
        [item for image_id in order[index : index + chunk_size] for item in grouped[image_id]]
        for index in range(0, len(order), chunk_size)
    ]


def _start_server(port: int, model_id: str, sdk_key: str) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            return

        def do_GET(self) -> None:
            request_path = urlparse(self.path).path
            if request_path == "/config":
                body = json.dumps({"modelId": model_id, "sdkKey": sdk_key}).encode("utf-8")
                content_type = "application/json"
            else:
                target = (WEB_ROOT / ("capture.html" if request_path in {"", "/"} else request_path.lstrip("/"))).resolve()
                if not target.is_file() or WEB_ROOT not in target.parents:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = target.read_bytes()
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("localhost", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class CaptureSession:
    def __init__(self, worker_index: int):
        self.worker_index = worker_index
        self.model_id: str | None = None
        self.sdk_key: str | None = None
        self.server: ThreadingHTTPServer | None = None
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    def capture(self, items: list[dict[str, Any]], settle_ms: int, model_id: str, sdk_key: str, headed: bool) -> list[dict[str, Any]]:
        with self.lock:
            if self.model_id != model_id or self.sdk_key != sdk_key:
                self.close()
                self.model_id, self.sdk_key = model_id, sdk_key
            self._start(headed)
            assert self.process and self.process.stdin and self.process.stdout
            request = {"id": uuid.uuid4().hex, "items": items, "settleMs": settle_ms}
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"Matterport worker exited: {self.process.poll()}")
            response = json.loads(line)
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            return response.get("items") or []

    def _start(self, headed: bool) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.server = _start_server(0, str(self.model_id), str(self.sdk_key))
        port = int(self.server.server_address[1])
        env = {**os.environ, "MATTERPORT_CAPTURE_HEADED": "1" if headed else "0"}
        self.process = subprocess.Popen(
            ["node", str(WEB_ROOT / "capture.js"), f"http://localhost:{port}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._read_stderr, daemon=True).start()
        logging.info("Started Matterport worker=%s modelId=%s headed=%s", self.worker_index, self.model_id, headed)

    def _read_stderr(self) -> None:
        if self.process and self.process.stderr:
            for line in self.process.stderr:
                logging.info("Matterport worker=%s: %s", self.worker_index, line.rstrip())

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None and process.stdin:
                try:
                    process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream:
                    stream.close()
        server, self.server = self.server, None
        if server:
            server.shutdown()
            server.server_close()


class SessionPool:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions: dict[tuple[str, str, int], CaptureSession] = {}

    def get(self, count: int, scope: str, model_id: str) -> list[CaptureSession]:
        with self.lock:
            return [self.sessions.setdefault((scope, model_id, index), CaptureSession(index)) for index in range(count)]

    def close(self, scope: str | None = None, model_id: str | None = None) -> None:
        with self.lock:
            selected = [key for key in self.sessions if (scope is None or key[0] == scope) and (model_id is None or key[1] == model_id)]
            sessions = [self.sessions.pop(key) for key in selected]
        for session in sessions:
            session.close()


POOL = SessionPool()
atexit.register(POOL.close)


def close_persistent_sessions(session_scope: str, model_id: str) -> None:
    POOL.close(session_scope, model_id)


def _short_capture(items: list[dict[str, Any]], settle_ms: int, model_id: str, sdk_key: str, headed: bool) -> list[dict[str, Any]]:
    server = _start_server(0, model_id, sdk_key)
    port = int(server.server_address[1])
    env = {**os.environ, "MATTERPORT_CAPTURE_HEADED": "1" if headed else "0"}
    try:
        completed = subprocess.run(
            ["node", str(WEB_ROOT / "capture.js"), f"http://localhost:{port}"],
            input=json.dumps({"id": uuid.uuid4().hex, "items": items, "settleMs": settle_ms}) + "\n",
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        response = json.loads(lines[-1])
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        return response.get("items") or []
    finally:
        server.shutdown()
        server.server_close()


def capture_anchors(work_items: list[dict[str, Any]], model_id: str, sdk_key: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = chunks_by_image_id(work_items, max(1, int(config.get("chunkSize", 1))))
    workers = min(max(1, int(config.get("workers", 1))), len(chunks))
    settle_ms = int(config.get("settleMs", 120))
    headed = bool(config.get("headed", True))
    scope = str(config.get("sessionScope") or model_id)
    if not config.get("persistentSession", True):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(lambda chunk: _short_capture(chunk, settle_ms, model_id, sdk_key, headed), chunks))
        return [item for output in outputs for item in output]
    sessions = POOL.get(workers, scope, model_id)
    assignments = [[(index, chunk) for index, chunk in enumerate(chunks) if index % workers == worker] for worker in range(workers)]

    def run(session_and_chunks):
        session, assigned = session_and_chunks
        return [(index, session.capture(chunk, settle_ms, model_id, sdk_key, headed)) for index, chunk in assigned]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outputs = list(executor.map(run, zip(sessions, assignments)))
    indexed = sorted((item for output in outputs for item in output), key=lambda value: value[0])
    return [item for _index, output in indexed for item in output]
