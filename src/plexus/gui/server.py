"""A zero-dependency backend for the Plexus spec node-editor.

Stdlib `http.server` only. Three jobs:

  * serve the static frontend (`static/`),
  * expose the registry-driven operator catalog (`/api/catalog`),
  * load / validate / save `spec.yaml` files, plus a node-layout sidecar so the
    canvas remembers where you placed things.

Validation reuses `plexus.schema.load` verbatim -- the same gatekeeper the engine
trusts -- so "valid in the editor" == "runnable". Binds to localhost.
"""

from __future__ import annotations

import errno
import io
import json
import os
import posixpath
import re
import subprocess
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote, quote

import yaml

import plexus.schema as schema
from plexus.gui.catalog import build_catalog

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
# repo root: .../Plexus  (src/plexus/gui -> up 3)
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

_CATALOG = None   # built once, lazily


def catalog():
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = build_catalog()
    return _CATALOG


# --------------------------------------------------------------------------- #
#  path safety: only touch .yaml files inside allowed roots
# --------------------------------------------------------------------------- #
def _allowed_roots():
    roots = [REPO_ROOT]
    gd = os.path.join(REPO_ROOT, "graphs_data")   # symlink -> the dataset/output tree (mp4s live there)
    if os.path.exists(gd):
        roots.append(os.path.realpath(gd))
    extra = os.environ.get("PLEXUS_GUI_ROOTS", "")
    roots += [r for r in extra.split(os.pathsep) if r]
    return [os.path.realpath(r) for r in roots]


def _safe_spec_path(path: str) -> str:
    p = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    if not any(p == r or p.startswith(r + os.sep) for r in _allowed_roots()):
        raise PermissionError(f"path outside allowed roots: {path}")
    return p


def _layout_path(spec_path: str) -> str:
    base, _ = os.path.splitext(spec_path)
    return base + ".gui.json"


def _find_media_in(d: str):
    """Best mp4 (+ poster png) in directory `d`, or None."""
    if not os.path.isdir(d):
        return None
    files = os.listdir(d)
    mp4s = [f for f in files if f.lower().endswith(".mp4")]
    if not mp4s:
        return None
    # prefer movie.mp4, then a *particle* render, then any movie*, then first
    def vrank(f):
        fl = f.lower()
        return (fl != "movie.mp4", "particle" not in fl, not fl.startswith("movie"), fl)
    video = os.path.join(d, sorted(mp4s, key=vrank)[0])
    m = {"video": "/media?path=" + quote(video), "name": os.path.basename(video)}
    pngs = [f for f in files if f.lower().endswith(".png")]
    if pngs:
        def prank(f):
            fl = f.lower()
            return ("final" not in fl, "strip" not in fl, fl)   # a final/strip frame makes the best poster
        m["poster"] = "/media?path=" + quote(os.path.join(d, sorted(pngs, key=prank)[0]))
    return m


def _media_for(spec_path: str):
    """A rendered mp4 for this spec: next to it (prototype archive dirs) or under
    `graphs_data/<...>/` (the dataset/output tree for `config/*.yaml` runs)."""
    cands = [os.path.dirname(spec_path)]
    rel = os.path.relpath(spec_path, REPO_ROOT)
    parts = rel.split(os.sep)
    # config/<cat>/<name>.yaml -> graphs_data/<cat>/<name>/
    if len(parts) > 1 and parts[0] == "config":
        stem = os.path.splitext(os.sep.join(parts[1:]))[0]
        cands.append(os.path.join(REPO_ROOT, "graphs_data", stem))
    # generic fallback: graphs_data/<full-path-stem>/
    cands.append(os.path.join(REPO_ROOT, "graphs_data", os.path.splitext(rel)[0]))
    for d in cands:
        m = _find_media_in(d)
        if m:
            return m
    return None


# --------------------------------------------------------------------------- #
#  spec <-> ordered yaml
# --------------------------------------------------------------------------- #
_OP_KEY_ORDER = ("op", "at", "implementation", "to", "from")
_GEN_KEY_ORDER = ("name", "seed", "n_frames", "dt", "boundary", "dim", "world",
                  "record_cap", "field_record_cap", "obstacles")


def _order_op(o: dict) -> dict:
    d = {}
    for k in _OP_KEY_ORDER:
        if k in o and o[k] not in (None, ""):
            d[k] = o[k]
    for k, v in o.items():
        if k not in d and k not in _OP_KEY_ORDER:
            d[k] = v
    return d


def _order_general(g: dict) -> dict:
    d = {}
    for k in _GEN_KEY_ORDER:
        if k in g and g[k] is not None:
            d[k] = g[k]
    for k, v in g.items():
        if k not in d:
            d[k] = v
    return d


def _ordered_spec(spec: dict) -> dict:
    top = {}
    top["general"] = _order_general(spec.get("general", {}) or {})
    top["sets"] = spec.get("sets", {}) or {}
    top["fields"] = spec.get("fields", {}) or {}
    top["operators"] = [_order_op(o) for o in spec.get("operators", []) or []]
    top["schedule"] = spec.get("schedule", []) or []
    if spec.get("plotting"):
        top["plotting"] = spec["plotting"]
    for k, v in spec.items():
        if k not in top:
            top[k] = v
    return top


def _dump_yaml(spec: dict) -> str:
    return yaml.safe_dump(_ordered_spec(spec), sort_keys=False,
                          default_flow_style=False, allow_unicode=True)


def _studio_name(prompt: str) -> str:
    """A filesystem name from the prompt's first few words, uniquified against what is there."""
    import re as _re
    base = "_".join(_re.findall(r"[a-z0-9]+", prompt.lower())[:4]) or "scene"
    from plexus.gui import studio
    n, i = base, 2
    while os.path.exists(os.path.join(studio.CONFIG_DIR, n + ".yaml")):
        n, i = f"{base}_{i}", i + 1
    return n


def _validate(spec: dict):
    """Run the real schema validator on a temp copy. Returns (ok, error)."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(_dump_yaml(spec))
            tmp = f.name
        schema.load(tmp)
        return True, None
    except Exception as e:  # noqa: BLE001 -- surface any validation error verbatim
        return False, str(e)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- #
#  spec discovery
# --------------------------------------------------------------------------- #
def _list_specs():
    out = []
    scan = [
        os.path.join(REPO_ROOT, "prototype"),
        os.path.join(REPO_ROOT, "config"),
    ]
    for root in scan:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if fn.endswith((".yaml", ".yml")):
                    full = os.path.join(dirpath, fn)
                    out.append({
                        "path": full,
                        "rel": os.path.relpath(full, REPO_ROOT),
                    })
    out.sort(key=lambda d: d["rel"])
    return out


# --------------------------------------------------------------------------- #
#  HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "PlexusGUI/0.1"
    protocol_version = "HTTP/1.1"   # keep-alive -> smoother <video> range seeking

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- helpers ---------------------------------------------------------- #
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_media(self, path):
        """Serve an mp4/png next to a spec, with HTTP Range support so <video> can seek."""
        if not path:
            return self.send_error(400)
        try:
            sp = _safe_spec_path(path)
        except Exception:
            return self.send_error(403)
        if not os.path.isfile(sp):
            return self.send_error(404)
        ctype = _ctype(sp)
        size = os.path.getsize(sp)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng and rng.startswith("bytes="):
            partial = True
            try:
                s, e = rng[6:].split("-", 1)
                start = int(s) if s else 0
                end = int(e) if e else size - 1
                end = min(end, size - 1)
                if start > end or start < 0:
                    start, end, partial = 0, size - 1, False
            except Exception:
                start, end, partial = 0, size - 1, False
        length = end - start + 1
        try:
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(sp, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass   # client seeked/closed the stream — normal for <video>

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    # -- GET -------------------------------------------------------------- #
    def do_GET(self):
        u = urlparse(self.path)
        route = u.path
        q = parse_qs(u.query)

        if route == "/" or route == "/index.html":
            return self._send_file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")

        if route.startswith("/static/"):
            rel = posixpath.normpath(unquote(route[len("/static/"):]))
            if rel.startswith(".."):
                return self.send_error(403)
            full = os.path.join(STATIC, rel)
            return self._send_file(full, _ctype(full))

        if route == "/studio":
            from plexus.gui import studio
            body = studio.page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)

        if route == "/api/studio/session":
            from plexus.gui import studio
            S = studio.SESSION
            return self._send_json({"state": S.get("state"), "chars": S.get("chars"),
                                    "seconds": S.get("seconds"), "error": S.get("error"),
                                    "specs": len(studio.REFERENCES)})

        if route == "/api/studio/list":
            from plexus.gui import studio
            return self._send_json({"specs": studio.list_specs()})

        if route == "/api/studio/spec":
            from plexus.gui import studio
            name = (q.get("name") or [""])[0]
            sp = os.path.join(studio.CONFIG_DIR, name + ".yaml")
            if not name or not os.path.exists(sp):
                return self._send_json({"error": "no such spec"}, 404)
            raw = open(sp).read()
            try:
                ok, err = _validate(yaml.safe_load(raw) or {})
            except Exception as e:                                       # noqa: BLE001
                ok, err = False, str(e)
            return self._send_json({"name": name, "path": sp, "raw": raw,
                                    "valid": ok, "error": err, **studio.artefacts(name)})

        if route == "/api/studio/progress":
            from plexus.gui import studio
            j = studio.JOBS.get((q.get("name") or [""])[0])
            return self._send_json(j.status() if j else {"done": True, "rc": 0,
                                                         "frame": 0, "total": 0, "pct": 0})

        if route == "/api/catalog":
            return self._send_json(catalog())

        if route == "/api/specs":
            return self._send_json({"specs": _list_specs(), "repo_root": REPO_ROOT})

        if route == "/media":
            return self._serve_media((q.get("path") or [None])[0])

        if route == "/api/spec":
            path = (q.get("path") or [None])[0]
            if not path:
                return self._send_json({"error": "missing ?path"}, 400)
            try:
                sp = _safe_spec_path(path)
                with open(sp) as f:
                    raw = f.read()
                parsed = yaml.safe_load(raw) or {}
                ok, err = _validate(parsed)
                layout = None
                lp = _layout_path(sp)
                if os.path.exists(lp):
                    with open(lp) as f:
                        layout = json.load(f)
                return self._send_json({
                    "path": sp, "rel": os.path.relpath(sp, REPO_ROOT),
                    "raw": raw, "spec": parsed, "layout": layout,
                    "valid": ok, "error": err, "media": _media_for(sp),
                })
            except Exception as e:  # noqa: BLE001
                return self._send_json({"error": str(e)}, 400)

        return self.send_error(404)

    # -- POST ------------------------------------------------------------- #
    def do_POST(self):
        u = urlparse(self.path)
        route = u.path
        try:
            data = self._read_json()
        except Exception as e:  # noqa: BLE001
            return self._send_json({"error": f"bad json: {e}"}, 400)

        if route == "/api/studio/author":
            # THE SERVER OWNS THE FILE. Claude runs read-only and hands back text; nothing reaches
            # config/studio/ until `plexus.schema.load` -- the same validator the engine trusts --
            # has accepted it. An invalid spec is returned as an error with the schema's own
            # message, so what you see is what the engine would have said.
            from plexus.gui import studio
            prompt = (data.get("prompt") or "").strip()
            if not prompt:
                return self._send_json({"error": "empty prompt"}, 400)
            os.makedirs(studio.CONFIG_DIR, exist_ok=True)
            # AN EXISTING NAME MEANS EDIT, NOT REPLACE. "make the ball bigger" is only meaningful
            # against the spec on screen, so the current YAML goes with the request and the reply
            # is written back over the same file. A missing name starts a new one.
            name = data.get("name") or _studio_name(prompt)
            sp = os.path.join(studio.CONFIG_DIR, name + ".yaml")
            current = open(sp).read() if os.path.exists(sp) else ""
            err_ctx = (data.get("error") or "").strip()
            if err_ctx:
                _last = err_ctx.strip().splitlines()[-1][:160] if err_ctx else ""
                print(f"\n[studio] SECOND PASS on {name!r} -- feeding the failure back to Claude\n"
                      f"         {_last}", flush=True)
            res = studio.author_spec(prompt, name, current=current,
                                     model=data.get("model") or "sonnet",
                                     deep=bool(data.get("deep")),
                                     effort=data.get("effort") or "low",
                                     error=err_ctx)
            if not res["yaml"]:
                studio.fail(f"Claude returned no YAML for {prompt!r} "
                            f"(rc={res['rc']}, {res['seconds']}s)", res["log"] or res["raw"])
                return self._send_json({"error": f"Claude returned no YAML (rc={res['rc']}, "
                                                 f"{res['seconds']}s)",
                                        "detail": (res["log"] or res["raw"])[-600:],
                                        "seconds": res["seconds"]})
            try:
                spec = yaml.safe_load(res["yaml"]) or {}
            except Exception as e:                                       # noqa: BLE001
                studio.fail(f"the reply is not YAML: {e}", res["yaml"])
                return self._send_json({"error": f"not YAML: {e}", "seconds": res["seconds"],
                                        "detail": res["yaml"][:600]})
            # A COUNT NAMED IN THE PROMPT BEATS THE FIELD, and then updates it. The knobs own this
            # number so the model cannot decouple it from n_grid -- but "5m particles" in the prompt
            # is the person asking, not the model guessing, and running 100k instead would be the
            # interface overruling its user in silence.
            _kn = dict(data.get("knobs") or {})
            _pp = studio.particles_from_prompt(prompt or "")
            if _pp:
                _kn["particles"] = _pp
            try:
                spec = studio.apply_knobs(spec, _kn)
            except Exception as e:                                       # noqa: BLE001
                studio.fail(str(e), res["yaml"])
                return self._send_json({"error": str(e), "seconds": res["seconds"]})
            spec.setdefault("general", {})["name"] = name
            ok, err = _validate(spec)
            if not ok:
                # THE SCHEMA'S OWN WORDS, IN FULL, IN THE TERMINAL. The browser gets a truncated
                # copy; the reason a scene could not be built belongs where the pipeline speaks.
                studio.fail(f"the schema rejected the spec for {prompt!r}",
                            f"{err}\n\n--- the spec it rejected ---\n{res['yaml'][:3000]}")
                return self._send_json({"error": "schema rejected the spec", "detail": err,
                                        "seconds": res["seconds"]})
            with open(sp, "w") as f:
                f.write(_dump_yaml(spec))
            return self._send_json({"name": name, "seconds": res["seconds"], "valid": True,
                                    "particles": _kn.get("particles"),
                                    "report": studio.knob_report(spec, _kn)})

        if route == "/api/studio/quit":
            # SHUT THE SOCKET, NOT JUST THE PROCESS. A studio killed with the port still bound --
            # or suspended with Ctrl-Z, which is how this bit us -- leaves 8765 held and the next
            # launch dies on "Address already in use" with a traceback that looks like a bug in the
            # server. `shutdown()` must be called from ANOTHER thread than serve_forever, hence the
            # timer; `server_close()` is what actually releases the listening socket.
            from plexus.gui import studio
            import threading as _th

            def _bye():
                try:
                    studio.worker_stop()
                finally:
                    try:
                        self.server.shutdown()
                        self.server.server_close()
                    finally:
                        os._exit(0)
            self._send_json({"bye": True})
            _th.Timer(0.25, _bye).start()
            return

        if route == "/api/studio/save":
            from plexus.gui import studio
            name = data.get("name") or ""
            sp = os.path.join(studio.CONFIG_DIR, name + ".yaml")
            if not name or not os.path.exists(sp):
                return self._send_json({"saved": False, "error": "no such spec"}, 404)
            try:
                spec = yaml.safe_load(data.get("raw") or "") or {}
            except Exception as e:                                       # noqa: BLE001
                return self._send_json({"saved": False, "error": f"not YAML: {e}"})
            ok, err = _validate(spec)
            if not ok:
                return self._send_json({"saved": False, "error": err})
            # THE TEXT YOU TYPED IS WHAT IS WRITTEN, not a re-dump of the parse. A round-trip
            # through yaml.safe_dump silently reorders keys and drops the layout you were reading,
            # which makes SAVE feel like it edited your file behind you.
            with open(sp, "w") as f:
                f.write(data.get("raw") or "")
            return self._send_json({"saved": True, "valid": True})

        if route == "/api/studio/dev":
            from plexus.gui import studio
            p = (data.get("prompt") or "").strip()
            if not p:
                return self._send_json({"error": "empty prompt"}, 400)
            return self._send_json(studio.start_dev(p, model=data.get("model") or "sonnet"))

        if route == "/api/studio/devstatus":
            from plexus.gui import studio
            d = dict(studio.DEV.get("dev") or {"running": False})
            d["text"] = (d.get("text") or "")[:400]      # the full text went to the terminal
            return self._send_json(d)

        if route == "/api/studio/apply":
            # RE-SIZE WITHOUT ASKING CLAUDE. A particle count is not a question about the scene;
            # round-tripping it through the model would cost 20 s to be handed back the spec it
            # already wrote, and would risk it changing something else on the way.
            from plexus.gui import studio
            name = data.get("name") or ""
            sp = os.path.join(studio.CONFIG_DIR, name + ".yaml")
            if not os.path.exists(sp):
                return self._send_json({"error": "no such spec"}, 404)
            try:
                spec = studio.apply_knobs(yaml.safe_load(open(sp)) or {}, data.get("knobs") or {})
            except Exception as e:                                       # noqa: BLE001
                studio.fail(str(e))
                return self._send_json({"error": str(e)}, 400)
            spec.setdefault("general", {})["name"] = name
            ok, err = _validate(spec)
            if not ok:
                studio.fail(f"re-sizing {name!r} produced an invalid spec", err)
                return self._send_json({"error": err})
            with open(sp, "w") as f:
                f.write(_dump_yaml(spec))
            return self._send_json({"name": name, "valid": True,
                                    "report": studio.knob_report(spec, data.get("knobs") or {})})

        if route == "/api/studio/metrics":
            from plexus.gui import studio
            sp = os.path.join(studio.CONFIG_DIR, (data.get("name") or "") + ".yaml")
            if not os.path.exists(sp):
                return self._send_json({"error": "no such spec"}, 404)
            try:
                return self._send_json(studio.metrics(yaml.safe_load(open(sp)) or {},
                                                      data.get("knobs") or {}))
            except Exception as e:                                       # noqa: BLE001
                return self._send_json({"error": str(e)}, 400)

        if route == "/api/studio/run":
            from plexus.gui import studio
            return self._send_json(studio.start_run(data.get("name") or "",
                                                    data.get("device") or "cuda:1",
                                                    bool(data.get("preview"))))

        if route == "/api/studio/stop":
            from plexus.gui import studio
            j = studio.JOBS.get(data.get("name") or "")
            if j:
                j.kill()
            return self._send_json({"stopped": bool(j)})

        if route == "/api/validate":
            ok, err = _validate(data.get("spec", {}))
            yamltext = None
            try:
                yamltext = _dump_yaml(data.get("spec", {}))
            except Exception as e:  # noqa: BLE001
                ok, err = False, f"yaml dump failed: {e}"
            return self._send_json({"valid": ok, "error": err, "yaml": yamltext})

        if route == "/api/save":
            path = data.get("path")
            spec = data.get("spec", {})
            layout = data.get("layout")
            if not path:
                return self._send_json({"error": "missing path"}, 400)
            try:
                sp = _safe_spec_path(path)
            except Exception as e:  # noqa: BLE001
                return self._send_json({"error": str(e)}, 400)
            ok, err = _validate(spec)
            if not ok and not data.get("force"):
                return self._send_json({"saved": False, "valid": False, "error": err})
            try:
                with open(sp, "w") as f:
                    f.write(_dump_yaml(spec))
                if layout is not None:
                    with open(_layout_path(sp), "w") as f:
                        json.dump(layout, f, indent=1)
            except Exception as e:  # noqa: BLE001
                return self._send_json({"saved": False, "error": str(e)}, 500)
            return self._send_json({"saved": True, "valid": ok, "error": err,
                                    "path": sp, "rel": os.path.relpath(sp, REPO_ROOT)})

        if route == "/api/layout":
            # persist just the node layout without touching the spec
            path = data.get("path")
            layout = data.get("layout")
            if not path:
                return self._send_json({"error": "missing path"}, 400)
            try:
                sp = _safe_spec_path(path)
                with open(_layout_path(sp), "w") as f:
                    json.dump(layout, f, indent=1)
            except Exception as e:  # noqa: BLE001
                return self._send_json({"error": str(e)}, 400)
            return self._send_json({"saved": True})

        return self.send_error(404)


def _ctype(path):
    ext = os.path.splitext(path)[1].lower()
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json",
        ".svg": "image/svg+xml",
        ".woff2": "font/woff2",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")


def _port_holder(port: int) -> str:
    """Who is listening on `port`, in words -- or "" if it cannot be determined."""
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=3).stdout
    except Exception:                                                # noqa: BLE001
        return ""
    for line in out.splitlines():
        cols = line.split()
        if len(cols) < 5 or not cols[3].endswith(f":{port}"):        # the Local Address:Port column
            continue
        m = re.search(r"pid=(\d+)", line)
        if not m:
            return "another process (run `ss -ltnp` to see which)"
        pid = m.group(1)
        try:
            cmd = open(f"/proc/{pid}/cmdline").read().replace("\x00", " ").strip()
            st = open(f"/proc/{pid}/stat").read().split(") ", 1)[1].split()[0]
        except Exception:                                            # noqa: BLE001
            cmd, st = "?", "?"
        # STOPPED IS THE CASE THAT LOOKS LIKE A HANG. A studio suspended with Ctrl-Z keeps the
        # socket and accepts connections the kernel queues, but never answers one -- so the browser
        # spins and the port is unavailable, with nothing in either place saying why.
        stopped = "  [STOPPED by Ctrl-Z -- holds the port, answers nothing; kill -9 it]" if st == "T" else ""
        return f"pid {pid} ({cmd}){stopped}"
    return ""


def serve(host="127.0.0.1", port=8765, prime=True):
    # A BOUND PORT IS NOT A CRASH, AND A TRACEBACK SAYS IT IS. `Address already in use` almost
    # always means a studio is ALREADY RUNNING and doing its job -- open it. The other case is a
    # studio suspended with Ctrl-Z, which keeps the socket while answering nothing, and the two need
    # opposite responses. Neither is discoverable from a socketserver stack trace, so say which it
    # is, name the process, and give the three ways out.
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        who = _port_holder(port)
        print(f"\n[studio] port {port} is already in use"
              f"{' by ' + who if who else ''}.\n"
              f"  open it      http://{host}:{port}/studio\n"
              f"  stop it      the 'Quit studio' button in that page (releases the port cleanly)\n"
              f"  or elsewhere python Plexus_gui.py --port {port + 25}\n", flush=True)
        raise SystemExit(1)
    if prime:
        # PRIME WHILE THE BROWSER IS STILL OPENING. Loading the corpus into a session takes a few
        # seconds and happens once; doing it lazily would make the FIRST prompt -- the one someone
        # is watching -- the one that pays for it.
        try:
            from plexus.gui import studio
            studio.prime_async()
            studio.worker_ready_async()      # imports torch/warp/pyvista once, off the hot path
        except Exception as e:                                       # noqa: BLE001
            print(f"[studio] could not start priming: {e}", flush=True)
    return httpd
