#!/usr/bin/env python
"""A standalone port for the build history, for when the page is not running.

THE HISTORY NOW LIVES ON THE PAGE, at `/watch` -- same server, same port, no forwarding to
arrange. This file stays for the case where no page is up, and serves the same four routes on a
port of its own.

    python tools/watch.py --port 8802      # http://127.0.0.1:8802/

It reads `builder/{png,spec,why,mp4}/NNNN.*` and `log/gui_runs/journal.txt` off disk and changes
nothing, so watching cannot disturb a run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from plexus.gui import watch as W  # noqa: E402


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        if p in ("/", "/index.html", "/watch"):
            return self._send(200, "text/html; charset=utf-8", W.PAGE.encode())
        if p == "/api/watch/state":
            return self._send(200, "application/json",
                              json.dumps(W.g_watch_state(self, q)).encode())
        if p in ("/api/watch/shot", "/api/watch/mp4"):
            kind = "png" if p.endswith("shot") else "mp4"
            f = W._nth(q, kind)
            if not f or not os.path.exists(f):
                return self._send(404, "text/plain", b"nothing for this step")
            with open(f, "rb") as fh:
                return self._send(200, "image/png" if kind == "png" else "video/mp4", fh.read())
        self._send(404, "text/plain", b"no")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8802)
    # 0.0.0.0, or the editor's port forwarding cannot see it (a loopback bind is invisible to it).
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"watch on http://127.0.0.1:{a.port}/   (also at /watch on the page's own port)",
          flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
