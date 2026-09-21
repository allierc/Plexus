"""A zero-dependency backend for the Plexus page and the spec node-editor.

Stdlib `http.server` only. The routes are a TABLE (`GET_ROUTES`, `POST_ROUTES`: path -> handler),
not a ladder of ifs, so a route is one entry and one function:

  * `/`                the one page (`gui/app.py`), `?tab=` bio | material | neurons | metabolism
  * `/api/tab/<t>/build`  the tab's form -> a validated spec in config/studio/, seeded
  * `/api/scene/*`     the shared panel: state, spec, seed, render, pick, info, view, run, frames,
                       artefacts, ls, open, counts, save, refine, visible, claude, reset
  * `/editor`, `/static/*`, `/api/catalog`, `/api/specs`, `/api/spec`, `/api/validate`,
    `/api/save`, `/api/layout`, `/media`   the node editor, unchanged

Validation reuses `plexus.schema.load` verbatim -- the same gatekeeper the engine trusts -- so
"valid in the page" == "runnable". Binds to localhost.
"""

from __future__ import annotations

import errno
import io
import json
import os

import numpy as np
import posixpath
import re
import subprocess
import sys
import tempfile
import time
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
    # THE DATA ROOT ITSELF, wherever it is. The runs the page starts land in
    # `graphs_data_path()` (GraphData on the NFS share, from $GNN_OUTPUT_ROOT or the default), and a
    # worktree has no `graphs_data` symlink at all -- so the movie the page had just written came
    # back 403 from `/media`. The data root is the page's own output; it is always servable.
    try:
        from plexus.paths import graphs_data_path
        roots.append(os.path.realpath(graphs_data_path()))
    except Exception:                                                # noqa: BLE001
        pass
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


def _tidy(v, sig: int = 6):
    """Round every float to `sig` significant digits on the way out, and drop the ones that are
    integers. A centre computed as the midpoint of two rounded corners comes out
    0.33330000000000004, and a spec a person has to read should not carry the last bit of a
    float's arithmetic: it says nothing, and thirty of them hide the three numbers that matter."""
    if isinstance(v, dict):
        return {k: _tidy(x, sig) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_tidy(x, sig) for x in v]
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return v
        r = float(f"%.{sig}g" % v)
        return r
    return v


def _dump_yaml(spec: dict) -> str:
    return yaml.safe_dump(_tidy(_ordered_spec(spec)), sort_keys=False,
                          default_flow_style=False, allow_unicode=True)


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
def _seed_check(spec: dict, name: str):
    """Build and seed the spec on the CPU on a temp copy. Returns (ok, error)."""
    from plexus.gui import bio
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(_dump_yaml(spec)); tmp = f.name
        bio.seed_scene(tmp)
        return True, None
    except Exception as e:                               # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


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

# --------------------------------------------------------------------------- #
#  the scene routes: one function each, registered in the tables at the bottom
# --------------------------------------------------------------------------- #
def _q1(q, k, default=""):
    return (q.get(k) or [default])[0]


def _spec_path(name: str) -> str:
    from plexus.gui import studio
    return os.path.join(studio.CONFIG_DIR, name + ".yaml")


def _tab_for(name: str, tab_hint: str | None = None):
    """The tab whose form a spec refills: the one named by the page, else the one recorded when
    the spec was built, else the current tab in STATE."""
    from plexus.gui import bio, tabs
    for t in (tab_hint, bio.STATE.get("specs", {}).get(name), bio.STATE.get("tab")):
        if t in tabs.ORDER:
            return tabs.get(t)
    return None


def _form_of(name: str, spec: dict, tab_hint: str | None = None):
    t = _tab_for(name, tab_hint)
    if t is None:
        return None
    try:
        return t.form_from_spec(spec)
    except Exception as e:                                       # noqa: BLE001
        print(f"[{t.NAME}] form_from_spec: {e}", flush=True)
        return None


def g_page(h, q):
    from plexus.gui import app, bio, tabs
    tab = _q1(q, "tab") or bio.STATE.get("tab") or "material"
    if tab not in tabs.ORDER:
        return h._send_json({"error": f"no tab {tab!r}; tabs are {', '.join(tabs.ORDER)}"}, 404)
    if bio.STATE.get("tab") and bio.STATE["tab"] != tab:
        # A PAGE OPENED ON ANOTHER TAB THAN THE SERVER HOLDS IS A SWITCH: same re-initialisation.
        # ONLY WHEN THE SERVER ALREADY HELD ONE, though: at boot `tab` is unset, so the FIRST page
        # load counted as a switch and threw away the scene the server had just opened -- which the
        # page then replaced with its own default. That is the "I see 27 cubes" of this session.
        _reset_scene(tab)
    bio.STATE.setdefault("tab", tab)
    if not bio.STATE.get("tab"):
        bio.STATE["tab"] = tab
    return h._send_html(app.page(tab))


def g_editor(h, q):
    return h._send_file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")


def g_watch(h, q):
    """`/watch` -- the read-only build history: picture, spec, why and movie, step by step.

    It rides on THIS port rather than one of its own because an unforwarded port inside a
    container is indistinguishable from a dead server, and this one already works. It reads files
    and changes nothing; see `plexus/gui/watch.py`.
    """
    from plexus.gui import watch
    return h._send_html(watch.PAGE)


def g_watch_state(h, q):
    from plexus.gui import watch
    return h._send_json(watch.g_watch_state(h, q))


def _watch_file(h, q, kind, ctype):
    from plexus.gui import watch
    f = watch._nth(q, kind)
    if not f or not os.path.exists(f):
        return h.send_error(404)
    return h._send_file(f, ctype)


def g_watch_shot(h, q):
    return _watch_file(h, q, "png", "image/png")


def g_watch_mp4(h, q):
    return _watch_file(h, q, "mp4", "video/mp4")


def g_watch_open3d(h, q):
    """`/api/watch/open3d?i=N` -- open THAT step's own saved spec here and seed it.

    The record keeps every step's spec beside its picture, so a step can be re-opened and turned
    rather than only looked at from the one camera it was shot with. This is the only route in
    the watcher that changes anything, which is why the page puts it behind a button: seeding a
    scene takes this server's single VTK thread, and a scene of a few hundred thousand particles
    takes seconds to build.
    """
    import shutil
    from plexus.gui import bio, bio_view, studio, watch
    f = watch._nth(q, "spec")
    if not f or not os.path.exists(f):
        return h._send_json({"error": "that step saved no spec"}, 404)
    try:
        spec = yaml.safe_load(open(f))
        name = str(((spec.get("general") or {}).get("name")) or "opened").strip()
        dst = _spec_path(name)
        os.makedirs(studio.CONFIG_DIR, exist_ok=True)
        shutil.copyfile(f, dst)
        # AND SEEDED HERE, not left for the first render to pay for. `g_open` only imports the
        # file; the scene is built by `open_view`, which for a few hundred thousand particles
        # takes seconds. Leaving that to the render meant the browser's first frame hung, and a
        # drag begun while it hung fired more renders behind it -- which is what a black panel
        # looks like from the outside.
        bio_view.open_view(dst)
        bio.STATE["name"] = name
        bio.bump(name, f"opened {f} for the 3-D view")
        return h._send_json({"name": name, "spec": dst})
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": f"{type(e).__name__}: {e}"[:300]}, 400)


def g_watch_render(h, q):
    """The live scene as a picture, at the camera the 3-D view is asking for."""
    return g_picture(h, q, "/api/scene/render")


def g_state(h, q):
    from plexus.gui import bio
    return h._send_json(dict(bio.STATE))


def g_spec(h, q):
    name = _q1(q, "name")
    sp = _spec_path(name)
    if not name or not os.path.exists(sp):
        return h._send_json({"error": "no such spec"}, 404)
    raw = open(sp).read()
    try:
        ok, err = _validate(yaml.safe_load(raw) or {})
    except Exception as e:                                       # noqa: BLE001
        ok, err = False, str(e)
    form = None
    try:
        form = _form_of(name, yaml.safe_load(raw), _q1(q, "tab") or None)
    except Exception:                                            # noqa: BLE001
        pass
    return h._send_json({"name": name, "path": sp, "raw": raw, "form": form, "valid": ok, "error": err})


def g_counts(h, q):
    from plexus.gui import bio
    name = _q1(q, "name")
    sp = _spec_path(name)
    if not name or not os.path.exists(sp):
        return h._send_json({"error": f"no spec {name!r}"}, 404)
    return h._send_json(bio.counts(bio.seed_scene(sp)))


def g_claude(h, q):
    from plexus.gui import bio
    since = int(_q1(q, "since", "0"))
    C = bio.CLAUDE
    secs = round(time.time() - C["started"], 1) if C["running"] else C["seconds"]
    return h._send_json({"running": C["running"], "task": C["task"], "n": len(C["lines"]),
                         "lines": C["lines"][since:], "seconds": secs, "error": C["error"]})


def g_frames(h, q):
    from plexus.gui import bio_view
    v = bio_view.current()
    n = 0 if v is None else (len(v.panel.hist) if getattr(v, "panel", None) is not None else len(v.snaps))
    # WHAT IS ON DISK, when nothing is in memory: the recorded trajectory of this spec's run, as a
    # frame count read from the file's header. PLAY loads it (`/api/scene/loadrun`).
    stored, recorded = 0, 0
    if v is not None and not n and getattr(v, "panel", None) is None:
        try:
            recorded, stored = (int(x) for x in v.stored_frames())
        except Exception:                                        # noqa: BLE001
            stored, recorded = 0, 0
    return h._send_json({"n": n, "stored": stored, "recorded": recorded,
                         "every": getattr(v, "_keep_every", 1) if v is not None else 1})


def g_run(h, q):
    from plexus.gui import bio_view
    v = bio_view.current()
    if v is None:
        return h._send_json({"running": False, "error": "no scene is open"})
    r = dict(v.RUN); r.pop("stop", None); r.pop("started", None)
    return h._send_json(r)


def g_artefacts(h, q):
    """What the run wrote for this spec: the movie (a `/media` URL with a cache buster), the
    newest still, the folder."""
    from plexus.gui import bio_view, studio
    name = _q1(q, "name")
    if not name:
        return h._send_json({"error": "name?"}, 400)
    a = studio.artefacts(name)
    # THE OPEN SCENE KNOWS WHERE ITS RUN LANDED (`View.run_dir`, which follows the spec's own
    # folder); `studio.artefacts` only ever looks in the page's tree, so a spec opened from
    # config/si_material showed no movie over a complete run.
    _v = bio_view.current()
    if not a.get("mp4") and _v is not None:
        _d = _v.run_dir()
        if _d and os.path.exists(os.path.join(_d, "movie.mp4")):
            a = dict(a); a["dir"] = _d
            a["mp4"] = os.path.join(_d, "movie.mp4"); a["mp4_mtime"] = os.path.getmtime(a["mp4"])
            _png = os.path.join(_d, "3d.png")
            if os.path.exists(_png):
                a["png"], a["png_mtime"] = _png, os.path.getmtime(_png)
            try:
                import imageio.v3 as _iio
                a["mp4_frames"] = sum(1 for _ in _iio.imiter(a["mp4"], plugin="pyav"))
            except Exception:                                    # noqa: BLE001
                pass
    out = {"dir": a.get("dir")}
    for k, mk in (("mp4", "mp4_mtime"), ("png", "png_mtime")):
        out[k] = ("/media?path=" + quote(a[k]) + f"&t={int(a.get(mk) or 0)}") if a.get(k) else None
    out["still"] = ("/media?path=" + quote(a["still"]) + f"&t={int(os.path.getmtime(a['still']))}") if a.get("still") else None
    # the movie's own frame count and rate, probed from the file: the page's slider spans them
    out["mp4_frames"] = int(a.get("mp4_frames") or 0)
    out["mp4_fps"] = float(a.get("mp4_fps") or 0.0)
    return h._send_json(out)


def g_ls(h, q):
    from plexus.gui import studio
    from plexus.paths import graphs_data_path
    roots = {"config": os.path.join(studio.REPO, "config"), "studio": studio.CONFIG_DIR,
             "graphs_data": graphs_data_path()}
    path = os.path.abspath(_q1(q, "path") or roots["config"])
    if not os.path.isdir(path):
        return h._send_json({"error": f"not a folder: {path}"}, 404)
    dirs, files = [], []
    try:
        for e in sorted(os.listdir(path)):
            if e.startswith("."):
                continue
            fp = os.path.join(path, e)
            if os.path.isdir(fp):
                dirs.append({"name": e, "spec": os.path.exists(os.path.join(fp, "spec.yaml"))})
            elif e.endswith((".yaml", ".yml")):
                files.append(e)
    except PermissionError:
        return h._send_json({"error": f"no access to {path}"}, 403)
    return h._send_json({"path": path, "parent": os.path.dirname(path), "dirs": dirs, "files": files, "roots": roots})


def g_open(h, q):
    """Import a spec (or a run folder's spec.yaml) into config/studio and open it as is."""
    import shutil
    from plexus.gui import bio, studio
    path = _q1(q, "path")
    if os.path.isdir(path):
        path = os.path.join(path, "spec.yaml")
    if not path or not os.path.exists(path):
        return h._send_json({"error": f"no spec at {path!r}"}, 404)
    spec = yaml.safe_load(open(path))
    name = str(((spec.get("general") or {}).get("name")) or os.path.basename(os.path.dirname(path)) or "opened").strip()
    dst = _spec_path(name)
    os.makedirs(studio.CONFIG_DIR, exist_ok=True)
    shutil.copyfile(path, dst)
    bio.bump(name, f"opened {path}")
    _sets = ", ".join(f"{k}" + (f" (types: {', '.join((v or {}).get('types') or {})})" if (v or {}).get("types") else "")
                      for k, v in (spec.get("sets") or {}).items())
    bio.claude_note(f"spec '{name}' opened from {path}; sets: {_sets}")
    return h._send_json({"name": name, "spec": dst, "form": _form_of(name, spec)})


def g_view(h, q):
    """Drive the page's view from outside (Claude): camera, pick, message."""
    from plexus.gui import bio, bio_view
    g = lambda k: (q.get(k) or [None])[0]                        # noqa: E731
    st = bio.set_view(azim=g("azim"), elev=g("elev"), zoom=g("zoom"), pick=g("pick"), message=g("message"))
    v = bio_view.current()
    if v is not None:
        v.set_camera(st["azim"], st["elev"], st["zoom"])
        if g("pick") is not None:
            v.highlight(st["pick"])
    return h._send_json(st)


def g_picture(h, q, route):
    """render / pick / info / snapshot: all read the session's view. `?name=` opens it when none is
    open (or a different spec is named)."""
    from plexus.gui import bio, bio_view
    name = _q1(q, "name")
    v = bio_view.current()
    try:
        if name and (v is None or os.path.basename(v.spec_path) != name + ".yaml"):
            sp = _spec_path(name)
            if not os.path.exists(sp):
                return h._send_json({"error": "no such spec"}, 404)
            v = bio_view.open_view(sp)
            bio.STATE["name"] = name
        if v is None:
            return h._send_json({"error": "no scene is open; seed one first (BUILD + SEED, or ?name=)"}, 400)
        if route.endswith("/info"):
            return h._send_json(bio.resolve_pick(v.scene, _q1(q, "pick")) or {"error": "no such object"})
        if route.endswith("/pick"):
            pk = v.pick_at(float(_q1(q, "x", "0.5")), float(_q1(q, "y", "0.5")))
            # THE OBJECT, NOT THE POINT: a click lands on one particle of a body; what was meant is
            # the body (the parent up the hierarchy), which is what is reported and outlined.
            pk = bio.climb(v.scene, pk)
            v.highlight(pk)
            bio.STATE["pick"] = pk
            return h._send_json({"pick": pk, "info": bio.resolve_pick(v.scene, pk) if pk else None})
        if q.get("azim") or q.get("elev") or q.get("zoom"):
            v.set_camera(*(float((q.get(k) or [str(getattr(v, k, 0.0) or 0.0)])[0])
                           for k in ("azim", "elev", "zoom", "roll")))
        if q.get("frame"):                                       # a kept frame of the last run, at this camera
            v.show_frame(int(q["frame"][0]))
        if q.get("pick"):
            v.highlight(_q1(q, "pick") or None)
        png = v.png()
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": f"{type(e).__name__}: {e}"[:800]}, 400)
    h.send_response(200)
    h.send_header("Content-Type", "image/png")
    h.send_header("Content-Length", str(len(png)))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    return h.wfile.write(png)


def g_seed(h, q):
    from plexus.gui import bio, bio_view
    name = _q1(q, "name")
    sp = _spec_path(name)
    if not name or not os.path.exists(sp):
        return h._send_json({"error": "no such spec"}, 404)
    try:
        # A SEED AFTER A RENDER CHANGE KEEPS THE RUN'S FRAMES (the style route sets the flag, once).
        carry = bool(bio.STATE.pop("carry_frames", False))
        v = bio_view.open_view(sp, carry=carry)                  # seeded once: the scene and the picture share it
        bio.STATE["name"] = name
        bio.claude_note(f"spec '{name}' seeded: " + ", ".join(f"{k} {s_.get('n_live')}" for k, s_ in v.scene["sets"].items()))
        out = dict(v.scene); out["seconds"] = v.seconds
        return h._send_json(out)
    except Exception as e:                                       # noqa: BLE001 -- the page shows the cause
        return h._send_json({"error": f"seed failed: {type(e).__name__}: {e}"[:800]}, 400)


def g_catalog(h, q):
    return h._send_json(catalog())


def g_specs(h, q):
    return h._send_json({"specs": _list_specs(), "repo_root": REPO_ROOT})


def g_media(h, q):
    return h._serve_media(_q1(q, "path") or None)


def g_editor_spec(h, q):
    path = _q1(q, "path") or None
    if not path:
        return h._send_json({"error": "missing ?path"}, 400)
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
        return h._send_json({"path": sp, "rel": os.path.relpath(sp, REPO_ROOT), "raw": raw, "spec": parsed,
                             "layout": layout, "valid": ok, "error": err, "media": _media_for(sp)})
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": str(e)}, 400)


# ---- POST ------------------------------------------------------------------ #
def _reset_scene(tab: str) -> None:
    """Stop the run, drop the view, forget the spec, remember the tab -- a tab switch."""
    from plexus.gui import bio, bio_view
    v = bio_view.current()
    if v is not None:
        v.stop()
        for _ in range(100):                                     # the hook raises StopRun at its next frame
            if not v.RUN.get("running"):
                break
            time.sleep(0.1)
        try:
            v.close()
        except Exception:                                        # noqa: BLE001
            pass
        bio_view.CURRENT["view"] = None
    bio.STATE.update(name=None, pick=None, message="", tab=tab)
    bio.STATE["version"] += 1
    bio.claude_note(f"the page switched to the {tab} tab; the scene was reset")


def p_reset(h, data):
    from plexus.gui import bio, tabs
    tab = str(data.get("tab") or bio.STATE.get("tab") or "material")
    if tab not in tabs.ORDER:
        return h._send_json({"error": f"no tab {tab!r}"}, 400)
    _reset_scene(tab)
    return h._send_json(dict(bio.STATE))


def p_tab_build(h, data, tab_name: str, quiet: bool | None = None):
    from plexus.gui import bio, tabs
    try:
        tab = tabs.get(tab_name)
    except KeyError as e:
        return h._send_json({"error": str(e)}, 404)
    try:
        spec = tab.build_spec(data)
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": f"form: {e}"}, 400)
    ok, err = _validate(spec)
    if not ok:
        return h._send_json({"error": "schema rejected the spec", "detail": err}, 400)
    name = spec["general"]["name"]
    sp = _spec_path(name)
    # WRITTEN THE WAY THE PIPELINE WILL READ IT: the tab's writer runs the guards the pipeline runs
    # (the CFL on an MPM spec) on the file now, so the YAML panel shows the YAML that runs.
    raw = tabs.write_spec(tab, spec, sp)
    bio.STATE.setdefault("specs", {})[name] = tab_name
    bio.STATE["tab"] = tab_name
    bio.bump(name, f"built {name}")
    bio.claude_note(f"{tab_name} spec '{name}' built from the form: sets {', '.join((spec.get('sets') or {}).keys())}")
    if quiet if quiet is not None else bool(data.get("quiet")):
        # THE YAML IS FOR THE PAGE'S EDITOR, NOT FOR A DRIVER. 32 KB of spec came back to a session
        # whose tool truncates at 30 KB, so its own change was unreadable to it; `quiet` answers
        # with what it asked for instead.
        sets = {k: (len((v or {}).get("types") or {}) or (v or {}).get("n") or (v or {}).get("per_parent"))
                for k, v in (spec.get("sets") or {}).items()}
        return h._send_json({"name": name, "valid": True, "version": bio.STATE["version"], "sets": sets})
    return h._send_json({"name": name, "raw": raw, "valid": True, "version": bio.STATE["version"]})


def p_claude(h, data):
    from plexus.gui import bio, tabs
    if data.get("stop"):
        return h._send_json(bio.claude_stop())
    if data.get("new_session"):
        return h._send_json(bio.claude_new_session())
    task = str(data.get("task") or "").strip()
    if not task:
        return h._send_json({"error": "empty task"}, 400)
    mode = str(data.get("mode") or bio.STATE.get("tab") or "bio")
    brief = None
    try:
        brief = tabs.get(mode).BRIEF
    except Exception:                                            # noqa: BLE001
        mode = "bio"
    # WHAT IS ON SCREEN, IN THE PROMPT. Without it the agent's first move is always a GET to read
    # the scene back -- one model round trip before any work, on every task.
    if data.get("form") is not None:
        f = dict(data["form"])
        bodies = f.pop("bodies", []) or []
        kinds = {}
        for b in bodies:
            kinds[str(b.get("material", "?"))] = kinds.get(str(b.get("material", "?")), 0) + 1
        task = (f"The scene on screen is spec '{data.get('name')}': {json.dumps(f)}, with "
                f"{len(bodies)} bodies ({', '.join(f'{v} {k}' for k, v in kinds.items()) or 'none'}).\n"
                f"Change it with ONE patch call when one will do:\n"
                f"  curl -s -X POST http://127.0.0.1:{{port}}/api/scene/patch -H 'Content-Type: application/json' "
                f"-d '{{\"form\": {{...changed top-level fields...}}, \"bodies\": {{\"*\": {{...changed body fields...}}}}}}'\n"
                f"Send only the fields that change, and do not read the scene first -- it is quoted above.\n"
                f"EVERY field of that form is patchable, including `n_bodies` (how many bodies there are; "
                f"the scene is re-laid on a lattice) and `particles` (material points PER BODY -- millions "
                f"are fine). A `bodies` key may be `*`, a name, an index, or a range like \"0-9\".\n"
                f"NOTHING IS OFF LIMITS. If a patch cannot say what the task asks -- a different kind of "
                f"scene, bodies at hand-picked places, a spec the form has no field for -- then build the "
                f"whole form with POST /api/tab/<tab>/build, or edit the spec in English with "
                f"POST /api/scene/refine. Do the task; never stop to ask whether you may."
                f"\n\nTask: {task}")
    return h._send_json(bio.claude_start(task, int(h.server.server_address[1]),
                                         model=str(data.get("model") or "sonnet"), brief=brief, mode=mode))


def p_run(h, data):
    from plexus.gui import bio_view
    v = bio_view.current()
    if v is None:
        return h._send_json({"error": "no scene is open; seed one first"}, 400)
    if data.get("stop"):
        return h._send_json(v.stop())
    return h._send_json(v.run(device=data.get("device")))


def p_visible(h, data):
    from plexus.gui import bio_view
    v = bio_view.current()
    if v is None:
        return h._send_json({"error": "no scene is open"}, 400)
    ok = v.set_visible(str(data.get("species") or ""), bool(data.get("on", True)))
    return h._send_json({"ok": ok, "hidden": sorted(v.hidden)})


def p_save(h, data):
    """The YAML panel's SAVE: the text goes through the tab's `normalise` (when it has one) and
    the validator, then to disk, then the page re-seeds."""
    from plexus.gui import bio
    name = str(data.get("name") or "")
    try:
        spec = yaml.safe_load(data.get("raw") or "")
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": f"not YAML: {e}"}, 400)
    if not isinstance(spec, dict) or not name:
        return h._send_json({"error": "no spec or no name"}, 400)
    tab = _tab_for(name, data.get("tab"))
    if tab is not None and getattr(tab, "normalise", None):
        spec = tab.normalise(spec)
    ok, err = _validate(spec)
    if not ok:
        return h._send_json({"error": "schema rejected the spec", "detail": err}, 400)
    open(_spec_path(name), "w").write(_dump_yaml(spec))
    bio.bump(name, f"saved {name}")
    return h._send_json({"name": name, "valid": True, "form": _form_of(name, spec, data.get("tab"))})


def p_refine(h, data):
    """An English edit of the current spec, applied by Claude and accepted only if it loads AND
    seeds; a failure goes back to Claude once with the error."""
    from plexus.gui import bio, studio
    name = str(data.get("name") or ""); prompt = str(data.get("prompt") or "").strip()
    sp = _spec_path(name)
    if not prompt or not os.path.exists(sp):
        return h._send_json({"error": "no prompt or no spec"}, 400)
    tab = _tab_for(name, data.get("tab"))
    norm = (getattr(tab, "normalise", None) if tab is not None else None) or (lambda s: s)
    current = open(sp).read()
    model = str(data.get("model") or "sonnet")
    res = studio.author_spec(prompt, name, current=current, model=model)
    if not res["yaml"]:
        return h._send_json({"error": f"Claude returned no YAML (rc={res['rc']})", "detail": res["log"][-1200:]})
    try:
        spec = norm(yaml.safe_load(res["yaml"]))
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": f"the reply is not YAML: {e}", "seconds": res["seconds"]})
    ok, err = _validate(spec)
    if ok:
        ok, err = _seed_check(spec, name)
    if not ok:
        res2 = studio.author_spec(prompt, name, current=_dump_yaml(spec), error=str(err), model=model)
        try:
            spec2 = norm(yaml.safe_load(res2["yaml"])) if res2["yaml"] else None
        except Exception:                                        # noqa: BLE001
            spec2 = None
        ok2, err2 = (_validate(spec2) if spec2 else (False, "no YAML on the fix pass"))
        if ok2:
            ok2, err2 = _seed_check(spec2, name)
        if not ok2:
            return h._send_json({"error": "the edited spec does not load or seed", "detail": f"{err}\n-- fix pass: {err2}",
                                 "seconds": res["seconds"] + res2.get("seconds", 0)})
        spec = spec2; res["seconds"] += res2.get("seconds", 0)
    raw = _dump_yaml(spec)
    open(sp, "w").write(raw)
    bio.bump(name, f"applied: {prompt[:80]} ({res['seconds']:.0f}s)")
    bio.claude_note(f"spec '{name}' refined by a prompt: {prompt[:160]}")
    return h._send_json({"name": name, "raw": raw, "seconds": res["seconds"], "valid": True,
                         "form": _form_of(name, spec, data.get("tab"))})


def p_nearside(h, data):
    """Show only the pieces that FACE THE CAMERA (`plotting.near_side`), or all of them again.

    `{"on": true}` cuts at the body's centre; `{"on": 0.3}` keeps a shallower cap, `-0.2` a little
    past the equator; `{"on": false}` shows everything. Applied to the open view at once -- no
    reseed -- and written into the spec so the movie draws what the page shows."""
    from plexus.gui import bio, bio_view
    v = bio_view.current()
    on = data.get("on", True)
    if v is None or v.lm is None:
        return h._send_json({"error": "no scene is open"}, 400)
    v.lm.style["near_side"] = on
    name = str(data.get("name") or bio.STATE.get("name") or "")
    sp = _spec_path(name)
    if name and os.path.exists(sp):
        spec = yaml.safe_load(open(sp)) or {}
        pl = spec.setdefault("plotting", {}) or {}
        if on:
            pl["near_side"] = on
        else:
            pl.pop("near_side", None)
        open(sp, "w").write(_dump_yaml(spec))
    bio_view._vtk(v.lm.near_side_refresh)
    return h._send_json({"near_side": on})


def p_delete(h, data):
    """Delete the spec on screen (config/studio/<name>.yaml) and forget it: the page's DELETE YAML."""
    from plexus.gui import bio, bio_view
    name = str(data.get("name") or bio.STATE.get("name") or "")
    sp = _spec_path(name)
    if not name or not os.path.exists(sp):
        return h._send_json({"error": "no spec is open"}, 400)
    os.remove(sp)
    bio.STATE.get("specs", {}).pop(name, None)
    bio.STATE["name"] = None
    v = bio_view.current()
    if v is not None:
        bio_view._vtk(v.close)
        bio_view.CURRENT["view"] = None
    bio.claude_note(f"spec '{name}' deleted from config/studio")
    return h._send_json({"deleted": os.path.basename(sp)})


def p_style(h, data):
    """The render selector: replace the spec's render keys, keep everything else, bump so the
    page re-seeds through the renderer with the new style (which is also the movie's)."""
    from plexus.gui import bio
    from plexus.gui.tabs import material as M
    name = str(data.get("name") or bio.STATE.get("name") or "")
    sp = _spec_path(name)
    if not name or not os.path.exists(sp):
        return h._send_json({"error": "no spec is open"}, 400)
    spec = yaml.safe_load(open(sp)) or {}
    try:
        spec["plotting"] = M.apply_render(spec.get("plotting") or {}, str(data.get("render", "small_dots")),
                                          str(data.get("light", "default")), str(data.get("color", "particles")))
    except ValueError as e:
        return h._send_json({"error": str(e)}, 400)
    ok, err = _validate(spec)
    if not ok:
        return h._send_json({"error": "schema rejected the spec", "detail": err}, 400)
    raw = _dump_yaml(spec)
    open(sp, "w").write(raw)
    bio.STATE["carry_frames"] = True                             # the next seed keeps the run's frames
    bio.bump(name, f"render {data.get('render', 'small_dots')}")
    return h._send_json({"name": name, "raw": raw, "version": bio.STATE["version"]})


def p_saveas(h, data):
    """SAVE...: copy the spec on screen to a path under the repo's config/ (the studio copy stays)."""
    import shutil
    name = str(data.get("name") or "")
    src = _spec_path(name)
    if not name or not os.path.exists(src):
        return h._send_json({"error": "no spec is open"}, 400)
    dst = os.path.abspath(os.path.expanduser(str(data.get("path") or "")))
    root = os.path.realpath(os.path.join(REPO_ROOT, "config"))
    if not os.path.realpath(os.path.dirname(dst)).startswith(root) or not dst.endswith(".yaml"):
        return h._send_json({"error": f"the path must be a .yaml under {root}"}, 400)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return h._send_json({"path": dst})


# THE CURVE PANELS a spec may carry beside the picture (`plotting.curve`, at most three): a live
# plot per quantity, drawn by `live_movie._curves_setup` from `plexus.measures`.
CURVES = ("cells", "area", "volume", "radius", "phase", "cycle_progress")


def p_curves(h, data):
    from plexus.gui import bio
    name = str(data.get("name") or bio.STATE.get("name") or "")
    sp = _spec_path(name)
    if not name or not os.path.exists(sp):
        return h._send_json({"error": "no spec is open"}, 400)
    want = [q for q in (data.get("curves") or []) if q in CURVES][:3]
    spec = yaml.safe_load(open(sp)) or {}
    pl = spec.setdefault("plotting", {})
    if want:
        pl["curve"] = [{"quantity": q, "xlabel": "frame", "ylabel": q.replace("_", " "), "ticks": 4} for q in want]
    else:
        pl.pop("curve", None)
    ok, err = _validate(spec)
    if not ok:
        return h._send_json({"error": "schema rejected the spec", "detail": err}, 400)
    raw = _dump_yaml(spec)
    open(sp, "w").write(raw)
    bio.STATE["carry_frames"] = True
    bio.bump(name, f"curves {', '.join(want) or 'none'}")
    return h._send_json({"name": name, "raw": raw, "curves": want, "version": bio.STATE["version"]})


def p_loadrun(h, data):
    """Read this spec's recorded trajectory into the view, so PLAY replays the REAL frames through
    the current render at any camera (the movie file is a picture; this is the data)."""
    from plexus.gui import bio_view
    v = bio_view.current()
    if v is None:
        return h._send_json({"error": "no scene is open"}, 400)
    try:
        n = bio_view._vtk(v.load_run_frames)
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": f"{type(e).__name__}: {e}"[:400]}, 400)
    if not n:
        return h._send_json({"error": "this spec has no recorded trajectory"}, 404)
    return h._send_json({"n": n, "every": v._keep_every, "n_frames": v.RUN.get("n_frames")})


def _relay_bodies(bodies: list, n: int, world: float) -> list:
    """`n` bodies on a cubic lattice in the middle half of the box, properties tiled from the ones
    on screen, sizes shrunk so a denser lattice still leaves gaps; a shrink keeps the first `n`."""
    import matplotlib
    n = max(1, int(n))
    src = list(bodies) or [{"name": "b0", "shape": "ball", "material": "elastic"}]
    if n <= len(src):
        return src[:n]
    k = int(np.ceil(n ** (1.0 / 3.0)))
    while k ** 3 < n:
        k += 1
    pitch = 0.5 * world / k
    side = 0.55 * pitch                                          # a body fills about half its cell
    # a block's own size, if the scene has one, caps the new size so a patch never inflates bodies
    for b in src:
        if b.get("block") and len(b["block"]) == 6:
            side = min(side, max(1e-6, float(b["block"][3]) - float(b["block"][0])))
            break
    x0 = 0.5 * world - 0.5 * k * pitch + 0.5 * (pitch - side)
    cm_a, cm_b = matplotlib.colormaps["tab20"], matplotlib.colormaps["tab20b"]
    out, i = [], 0
    for iy in range(k):
        for iz in range(k):
            for ix in range(k):
                if i >= n:
                    break
                b = dict(src[i % len(src)])
                a = [round(x0 + ix * pitch, 4), round(0.40 * world + iy * pitch, 4), round(x0 + iz * pitch, 4)]
                b["name"] = f"b{i:03d}"
                b["color"] = [round(float(v), 3) for v in (cm_a(i % 20) if (i // 20) % 2 == 0 else cm_b(i % 20))[:3]]
                if str(b.get("shape", "ball")).lower() == "block":
                    b["block"] = [a[0], a[1], a[2], round(a[0] + side, 4), round(a[1] + side, 4), round(a[2] + side, 4)]
                    b.pop("centre", None)
                else:
                    b["centre"] = [float(f"%.6g" % (a[k] + 0.5 * side)) for k in range(3)]
                    b.pop("block", None)
                out.append(b); i += 1
    return out


def p_patch(h, data):
    """Change a FEW FIELDS of the scene on screen, and rebuild: `{"form": {...}, "bodies": {"*": {...}}}`.

    WHY A PATCH AND NOT THE WHOLE FORM. Handing the page's driver the form and asking it to post
    the edited copy back means, on a 27-body scene, 5,000 characters of JSON it must READ and
    then WRITE, and generating them is most of a 50-second turn. "Change the material to water" is
    two fields. `bodies` keys are body names, or `*` for every body.
    """
    from plexus.gui import bio, tabs
    name = str(data.get("name") or bio.STATE.get("name") or "")
    sp = _spec_path(name)
    tab_name = str(data.get("tab") or bio.STATE.get("specs", {}).get(name) or bio.STATE.get("tab") or "")
    if not name or not os.path.exists(sp):
        return h._send_json({"error": "no scene is open"}, 400)
    raw = yaml.safe_load(open(sp))
    form = None
    # A SPEC OPENED FROM A FILE IS PATCHABLE TOO. "form-built" was a bookkeeping flag, not a fact
    # about the spec: `form_from_spec` reads any spec the tab understands, which is how the page
    # fills its own form after OPEN. Try the named tab, then every other one, and remember which
    # answered -- a driver asked to change the scene should not be told the scene is off limits.
    for cand in ([tab_name] if tab_name in tabs.ORDER else []) + [x for x in tabs.ORDER if x != tab_name]:
        try:
            f = tabs.get(cand).form_from_spec(raw)
            if not f:
                continue
            # THE TAB THE PAGE NAMES IS THE TAB, no further argument: the rebuild test below is for
            # GUESSING among tabs when nothing said which, and a tab whose builder adds or renames a
            # set (the neurons tab's edge set) would fail its own spec.
            if cand == tab_name:
                form = f
                bio.STATE.setdefault("specs", {})[name] = cand
                break
            # A FORM IS ONLY THE RIGHT ONE IF IT REBUILDS THIS SPEC'S SETS. Every tab's reader
            # returns a dict of defaults for any spec, so "the first that answers" chose the bio
            # tab for a box of cubes and refused the patch for want of a protein species.
            got = set((tabs.get(cand).build_spec(f).get("sets") or {}).keys())
        except Exception:                                        # noqa: BLE001
            continue
        if got and got == set((raw.get("sets") or {}).keys()):
            form, tab_name = f, cand
            bio.STATE.setdefault("specs", {})[name] = cand
            break
    if form is None:
        return h._send_json({"error": "no tab reads this spec back into a form; edit it with /api/scene/refine"}, 400)
    tab = tabs.get(tab_name)
    patch_form = dict(data.get("form") or {})
    # HOW MANY BODIES IS A PATCHABLE FIELD. "100 boxes, 10 of water" is one sentence and was two
    # impossibilities: `{"form": {"bodies": 100}}` put an int where the list goes and the handler
    # died mid-response ("Empty reply from server"), and nothing else could change the COUNT, so
    # the driver had to rebuild the whole form. `n_bodies` (or `bodies` as a number) re-lays the
    # scene: the existing bodies' properties are tiled over a cubic lattice of N, in the same half
    # of the box the 27-cube default uses, with one colour each.
    n_want = patch_form.pop("n_bodies", None)
    if isinstance(patch_form.get("bodies"), (int, float)):
        n_want = patch_form.pop("bodies")
    form.update(patch_form)
    if n_want is not None:
        form["bodies"] = _relay_bodies(form.get("bodies") or [], int(n_want), float(form.get("world", 0.5)))
    # ANY LIST THE FORM HAS, NOT ONLY `bodies`. A tissue's form carries `organelles` and `species`,
    # and a patch that could name neither sent the driver back to a full rebuild to widen a nucleus.
    for _key in ("organelles", "species"):
        _p = data.get(_key) or {}
        _lst = form.get(_key) or []
        if not _p or not isinstance(_lst, list):
            continue
        for i, it in enumerate(_lst):
            for k2, patch in _p.items():
                if k2 == "*" or k2 == it.get("name") or k2 == str(i) \
                        or ("-" in str(k2) and str(k2).replace("-", "").isdigit()
                            and int(str(k2).split("-")[0]) <= i <= int(str(k2).split("-")[1])):
                    it.update(patch)
    bp = data.get("bodies") or {}
    if bp:
        blist = form.get("bodies") or []
        for i, b in enumerate(blist):
            # `*`, the body's name, its index, or a range "0-9" -- so "ten of them are water" is
            # one key rather than ten.
            for key, patch in bp.items():
                if key == "*" or key == b.get("name") or key == str(i) \
                        or ("-" in str(key) and str(key).replace("-", "").isdigit()
                            and int(str(key).split("-")[0]) <= i <= int(str(key).split("-")[1])):
                    b.update(patch)
        # a material change carries its own stiffness key: drop the one that no longer applies
        for b in form.get("bodies") or []:
            if str(b.get("material", "")).lower() == "liquid":
                b.pop("youngs", None)
            else:
                b.pop("bulk_modulus", None)
    try:
        return p_tab_build(h, form, tab_name, quiet=bool(data.get("quiet", True)))
    except Exception as e:                                       # noqa: BLE001 -- a bad patch is an answer, not a dead socket
        return h._send_json({"error": f"patch: {type(e).__name__}: {e}"[:400]}, 400)


def p_quit(h, data):
    """SHUT THE SOCKET, NOT JUST THE PROCESS. A server killed with the port still bound -- or
    suspended with Ctrl-Z -- leaves the port held and the next launch dies on "Address already in
    use". `shutdown()` must be called from ANOTHER thread than serve_forever, hence the timer;
    `server_close()` is what releases the listening socket."""
    import threading as _th

    def _bye():
        try:
            h.server.shutdown()
            h.server.server_close()
        finally:
            os._exit(0)
    h._send_json({"bye": True})
    _th.Timer(0.25, _bye).start()


def p_validate(h, data):
    ok, err = _validate(data.get("spec", {}))
    yamltext = None
    try:
        yamltext = _dump_yaml(data.get("spec", {}))
    except Exception as e:                                       # noqa: BLE001
        ok, err = False, f"yaml dump failed: {e}"
    return h._send_json({"valid": ok, "error": err, "yaml": yamltext})


def p_editor_save(h, data):
    path = data.get("path"); spec = data.get("spec", {}); layout = data.get("layout")
    if not path:
        return h._send_json({"error": "missing path"}, 400)
    try:
        sp = _safe_spec_path(path)
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": str(e)}, 400)
    ok, err = _validate(spec)
    if not ok and not data.get("force"):
        return h._send_json({"saved": False, "valid": False, "error": err})
    try:
        with open(sp, "w") as f:
            f.write(_dump_yaml(spec))
        if layout is not None:
            with open(_layout_path(sp), "w") as f:
                json.dump(layout, f, indent=1)
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"saved": False, "error": str(e)}, 500)
    return h._send_json({"saved": True, "valid": ok, "error": err, "path": sp, "rel": os.path.relpath(sp, REPO_ROOT)})


def p_layout(h, data):
    path = data.get("path"); layout = data.get("layout")
    if not path:
        return h._send_json({"error": "missing path"}, 400)
    try:
        sp = _safe_spec_path(path)
        with open(_layout_path(sp), "w") as f:
            json.dump(layout, f, indent=1)
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": str(e)}, 400)
    return h._send_json({"saved": True})


def g_regions(h, q):
    """The frozen neuprint regions on this host, for the neurons tab's source menu."""
    from plexus.gui.tabs import neurons as N
    try:
        return h._send_json({"regions": N.regions()})
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"regions": [], "error": f"{type(e).__name__}: {e}"[:200]})


def g_stacks(h, q):
    """EVERY THREAD'S PYTHON STACK, right now.

    A run that sits at frame 0 with the GPU idle is either waiting on a lock or blocked inside a
    C call, and from outside the two look identical: the status route says `running: true` and
    nothing else moves. `py-spy` cannot attach here (ptrace_scope is 1 and the process is not
    ours to trace), so the process has to be able to say where it is itself. This is that.

    Read-only and cheap: `sys._current_frames()` is a snapshot of frame objects already held.
    """
    import sys as _sys
    import threading as _th
    import traceback as _tb
    names = {t.ident: t.name for t in _th.enumerate()}
    out = {}
    for tid, frame in _sys._current_frames().items():
        out[f"{names.get(tid, '?')} ({tid})"] = [
            f"{fs.filename}:{fs.lineno} in {fs.name}" for fs in _tb.extract_stack(frame)][-25:]
    # THE TWO QUEUES THE RUN GOES THROUGH. A run that is "running" with no thread executing it is
    # a task sitting in a queue, and which queue says which thing is wrong: `exec_queue` is work
    # submitted to the one VTK thread and not yet picked up, `pending` is the page's camera and
    # screenshot requests waiting for that same thread. Both zero with a live `running` flag means
    # the task was taken and then lost, which is a different bug from a task never taken.
    q = {}
    try:
        from plexus.gui import bio_view as _bv
        q["exec_queue"] = _bv._EXEC._work_queue.qsize()
        q["pending"] = _bv._PENDING.qsize()
        q["vtk_thread_ident"] = _bv._VTK_THREAD.get("ident")
        v = _bv.CURRENT.get("view")
        q["run"] = {k: v.RUN.get(k) for k in ("running", "frame", "n_frames", "error", "stopped")} \
            if v is not None else None
        q["view_id"] = id(v) if v is not None else None
    except Exception as e:                                           # noqa: BLE001
        q["error"] = f"{type(e).__name__}: {e}"
    return h._send_json({"threads": out, "n": len(out), "queues": q})


def g_shot(h, q):
    """The picture AS A FILE, for an agent that can look at images but cannot hold a PNG body.

    `/api/scene/render` streams the bytes, which is what the browser wants and what a curl-only
    session cannot use. This writes the same picture under log/gui_shots/ and returns its path, so
    the session can read the image and SEE the scene it is building rather than infer it from counts.
    """
    import time as _t
    from plexus.gui import bio_view, studio
    v = bio_view.current()
    if v is None:
        return h._send_json({"error": "no scene is open; seed one first"}, 400)
    try:
        if q.get("azim") or q.get("elev") or q.get("zoom"):
            v.set_camera(*(float((q.get(k) or [str(getattr(v, k, 0.0) or 0.0)])[0])
                           for k in ("azim", "elev", "zoom", "roll")))
        if q.get("frame"):
            v.show_frame(int(q["frame"][0]))
        png = v.png()
    except Exception as e:                                       # noqa: BLE001
        return h._send_json({"error": f"{type(e).__name__}: {e}"[:800]}, 400)
    d = os.path.join(studio.REPO, "log", "gui_shots")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"shot_{int(_t.time() * 1000)}.png")
    with open(p, "wb") as f:
        f.write(png)
    # NOTHING IS DELETED HERE. This used to keep only the last forty and remove the rest, which
    # is right for a scratch preview and wrong for what this folder became: the record of every
    # scene this session built, walked back and forth in `tools/watch.py`. A history that quietly
    # drops its oldest forty-first entry is not a history. A shot is about a megabyte; if the
    # folder ever needs trimming that is a decision for a person, not a side effect of taking a
    # picture.
    return h._send_json({"path": p, "bytes": len(png), "azim": v.azim, "elev": v.elev, "zoom": v.zoom,
                         "frame": q.get("frame", [None])[0]})
POST_ROUTES = {
    "/api/scene/reset": p_reset, "/api/scene/claude": p_claude, "/api/scene/run": p_run,
    "/api/scene/visible": p_visible, "/api/scene/save": p_save, "/api/scene/refine": p_refine,
    "/api/scene/style": p_style, "/api/scene/saveas": p_saveas, "/api/scene/curves": p_curves,
    "/api/scene/loadrun": p_loadrun, "/api/scene/patch": p_patch, "/api/scene/nearside": p_nearside, "/api/scene/delete": p_delete,
    "/api/quit": p_quit, "/api/studio/quit": p_quit,
    "/api/validate": p_validate, "/api/save": p_editor_save, "/api/layout": p_layout,
}
for _x in ("reset", "claude", "run", "visible", "save", "refine"):
    POST_ROUTES[f"/api/bio/{_x}"] = POST_ROUTES[f"/api/scene/{_x}"]


# THE TABLES. `/api/scene/<x>` is canonical; `/api/bio/<x>` is the same handler under the name the
# older Claude sessions were primed with.
GET_ROUTES = {
    "/": g_page, "/index.html": g_page, "/editor": g_editor,
    "/api/scene/state": g_state, "/api/scene/spec": g_spec, "/api/scene/counts": g_counts,
    "/api/scene/claude": g_claude, "/api/scene/frames": g_frames, "/api/scene/run": g_run,
    "/api/scene/artefacts": g_artefacts, "/api/scene/ls": g_ls, "/api/scene/open": g_open,
    "/api/scene/view": g_view, "/api/scene/seed": g_seed,
    "/api/catalog": g_catalog, "/api/specs": g_specs, "/media": g_media, "/api/spec": g_editor_spec,
    "/api/scene/shot": g_shot, "/api/bio/shot": g_shot, "/api/neurons/regions": g_regions,
    "/api/debug/stacks": g_stacks,
    "/watch": g_watch, "/api/watch/state": g_watch_state,
    "/api/watch/shot": g_watch_shot, "/api/watch/mp4": g_watch_mp4,
    "/api/watch/open3d": g_watch_open3d,
}
for _x in ("state", "spec", "counts", "claude", "frames", "run", "artefacts", "ls", "open", "view", "seed"):
    GET_ROUTES[f"/api/bio/{_x}"] = GET_ROUTES[f"/api/scene/{_x}"]
GET_ROUTES["/api/studio/spec"] = g_spec
GET_ROUTES["/api/material/run"] = g_artefacts
PICTURE_ROUTES = {f"/api/{p}/{x}" for p in ("scene", "bio") for x in ("render", "pick", "info", "snapshot")}
# The watcher's 3-D view streams the same picture the page's own view does.
PICTURE_ROUTES.add("/api/watch/render")


class Handler(BaseHTTPRequestHandler):
    server_version = "PlexusGUI/0.2"
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

    def _send_html(self, text: str):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return self.wfile.write(body)

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
            pass   # client seeked/closed the stream -- normal for <video>

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    # -- dispatch --------------------------------------------------------- #
    def do_GET(self):
        u = urlparse(self.path)
        route, q = u.path, parse_qs(u.query)
        if route.startswith("/static/"):
            rel = posixpath.normpath(unquote(route[len("/static/"):]))
            if rel.startswith(".."):
                return self.send_error(403)
            full = os.path.join(STATIC, rel)
            return self._send_file(full, _ctype(full))
        if route in ("/bio", "/material", "/neurons", "/metabolism", "/studio"):
            tab = "material" if route == "/studio" else route[1:]
            self.send_response(302)
            self.send_header("Location", f"/?tab={tab}")
            self.send_header("Content-Length", "0")
            return self.end_headers()
        if route in PICTURE_ROUTES:
            return g_picture(self, q, route)
        fn = GET_ROUTES.get(route)
        if fn is None:
            return self.send_error(404)
        return fn(self, q)

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            data = self._read_json()
        except Exception as e:  # noqa: BLE001
            return self._send_json({"error": f"bad json: {e}"}, 400)
        m = re.match(r"^/api/tab/([a-z_]+)/build$", route)
        if m:
            return p_tab_build(self, data, m.group(1))
        if route == "/api/material/build":
            return p_tab_build(self, data, "material")
        if route == "/api/bio/build":
            return p_tab_build(self, data, "bio")
        fn = POST_ROUTES.get(route)
        if fn is None:
            return self.send_error(404)
        return fn(self, data)


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
        print(f"\n[gui] port {port} is already in use"
              f"{' by ' + who if who else ''}.\n"
              f"  open it      http://{host}:{port}/\n"
              f"  stop it      curl -X POST http://{host}:{port}/api/quit (releases the port cleanly)\n"
              f"  or elsewhere python Plexus_gui.py --port {port + 25}\n", flush=True)
        raise SystemExit(1)
    if prime:
        # PRIME WHILE THE BROWSER IS STILL OPENING. Loading the corpus into a session takes a few
        # seconds and happens once; doing it lazily would make the FIRST prompt -- the one someone
        # is watching -- the one that pays for it. The engine's imports (torch, warp, pyvista)
        # are paid once too, by the first seed, in this process: the run is in-process now.
        try:
            from plexus.gui import bio, studio
            studio.prime_async()
            bio.claude_warm()                    # the page's own session, corpus already in it
        except Exception as e:                                       # noqa: BLE001
            print(f"[studio] could not start priming: {e}", flush=True)
    return httpd
