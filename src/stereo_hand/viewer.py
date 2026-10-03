"""A browser 3-D view of the rover and its arm, fed live from Python.

The page (web/index.html, three.js) draws the whole rover and orbits, pans
and zooms with the mouse. Python pushes the arm's state to it as
Server-Sent Events, about 60 times a second, from a plain standard-library
HTTP server: no extra dependency, and any browser on the machine, or on the
network for a projector, can watch the same arm.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .arm import ARM, spec_dict

WEB = Path(__file__).parent / "web"


class _Hub:
    def __init__(self):
        self.cond = threading.Condition()
        self.seq = 0
        self.payload = b""

    def publish(self, state: dict) -> None:
        data = json.dumps(state, separators=(",", ":")).encode()
        with self.cond:
            self.payload = data
            self.seq += 1
            self.cond.notify_all()


class _Handler(SimpleHTTPRequestHandler):
    hub: _Hub = None  # set on the subclass

    def log_message(self, *args):  # keep the terminal for the tracking output
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/spec"):
            body = json.dumps(spec_dict(ARM)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with self.hub.cond:
                        self.hub.cond.wait_for(lambda: self.hub.seq != seen, timeout=1.0)
                        seen, data = self.hub.seq, self.hub.payload
                    if data:
                        self.wfile.write(b"data: " + data + b"\n\n")
                    else:
                        self.wfile.write(b": waiting\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        return super().do_GET()


class Viewer:
    def __init__(self, port: int = 8765, open_browser: bool = True, host: str = "0.0.0.0"):
        self.hub = _Hub()
        handler = type("Handler", (_Handler,), {"hub": self.hub})
        self.httpd = ThreadingHTTPServer((host, port), partial(handler, directory=str(WEB)))
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://localhost:{port}/"
        print(f"3-D view: {self.url}  (orbit: drag, pan: right-drag or shift-drag, zoom: scroll)")
        if open_browser:
            webbrowser.open(self.url)

    def publish(self, state: dict) -> None:
        self.hub.publish(state)

    def close(self) -> None:
        self.httpd.shutdown()
