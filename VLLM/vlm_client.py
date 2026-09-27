"""vlm_client.py -- reach the persistent VLM (`vlm_server.py`) if one is running. Stdlib only.

A caller that finds no server falls back to loading the model itself, so nothing REQUIRES the
server; it only removes the ~70 s load from every call after the first. That load was paid twice
per iteration of a simulation-branch search -- once by the caption pass, once by the judge -- and
was most of the wall time of an iteration whose engine run took 11 s.

    PLEXUS_VLM_URL   where the server listens (default http://127.0.0.1:8766); `off` disables it
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8766"


def server_url(timeout=1.0):
    """The server's URL if it answers /health within `timeout` seconds, else None."""
    url = os.environ.get("PLEXUS_VLM_URL", DEFAULT_URL).rstrip("/")
    if url.lower() == "off":
        return None
    try:
        with urllib.request.urlopen(url + "/health", timeout=timeout) as r:
            return url if r.status == 200 else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def post(url, route, payload, timeout=1800):
    """POST `payload` as JSON to `url/route`; the decoded JSON reply. Raises on transport errors
    and on a reply carrying `error`, so a failed request never reads as an empty answer."""
    req = urllib.request.Request(url + route, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    if out.get("error"):
        raise RuntimeError(f"vlm_server {route}: {out['error']}")
    return out
