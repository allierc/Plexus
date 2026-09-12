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
import posixpath
import re
import subprocess
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


def _dump_yaml(spec: dict) -> str:
    return yaml.safe_dump(_ordered_spec(spec), sort_keys=False,
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
    if bio.STATE.get("tab") != tab:
        # A PAGE OPENED ON ANOTHER TAB THAN THE SERVER HOLDS IS A SWITCH: same re-initialisation.
        _reset_scene(tab)
    return h._send_html(app.page(tab))


def g_editor(h, q):
    return h._send_file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")


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
    return h._send_json({"n": len(v.snaps) if v is not None else 0,
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
    from plexus.gui import studio
    name = _q1(q, "name")
    if not name:
        return h._send_json({"error": "name?"}, 400)
    a = studio.artefacts(name)
    out = {"dir": a.get("dir")}
    for k, mk in (("mp4", "mp4_mtime"), ("png", "png_mtime")):
        out[k] = ("/media?path=" + quote(a[k]) + f"&t={int(a.get(mk) or 0)}") if a.get(k) else None
    out["still"] = ("/media?path=" + quote(a["still"]) + f"&t={int(os.path.getmtime(a['still']))}") if a.get("still") else None
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
            v.highlight(pk)
            bio.STATE["pick"] = pk
            return h._send_json({"pick": pk, "info": bio.resolve_pick(v.scene, pk) if pk else None})
        if q.get("azim") or q.get("elev") or q.get("zoom"):
            v.set_camera(*(float((q.get(k) or [str(getattr(v, k))])[0]) for k in ("azim", "elev", "zoom")))
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
        v = bio_view.open_view(sp)                               # seeded once: the scene and the picture share it
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


def p_tab_build(h, data, tab_name: str):
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


# THE TABLES. `/api/scene/<x>` is canonical; `/api/bio/<x>` is the same handler under the name the
# older Claude sessions were primed with.
GET_ROUTES = {
    "/": g_page, "/index.html": g_page, "/editor": g_editor,
    "/api/scene/state": g_state, "/api/scene/spec": g_spec, "/api/scene/counts": g_counts,
    "/api/scene/claude": g_claude, "/api/scene/frames": g_frames, "/api/scene/run": g_run,
    "/api/scene/artefacts": g_artefacts, "/api/scene/ls": g_ls, "/api/scene/open": g_open,
    "/api/scene/view": g_view, "/api/scene/seed": g_seed,
    "/api/catalog": g_catalog, "/api/specs": g_specs, "/media": g_media, "/api/spec": g_editor_spec,
}
for _x in ("state", "spec", "counts", "claude", "frames", "run", "artefacts", "ls", "open", "view", "seed"):
    GET_ROUTES[f"/api/bio/{_x}"] = GET_ROUTES[f"/api/scene/{_x}"]
GET_ROUTES["/api/studio/spec"] = g_spec
GET_ROUTES["/api/material/run"] = g_artefacts
PICTURE_ROUTES = {f"/api/{p}/{x}" for p in ("scene", "bio") for x in ("render", "pick", "info", "snapshot")}
POST_ROUTES = {
    "/api/scene/reset": p_reset, "/api/scene/claude": p_claude, "/api/scene/run": p_run,
    "/api/scene/visible": p_visible, "/api/scene/save": p_save, "/api/scene/refine": p_refine,
    "/api/quit": p_quit, "/api/studio/quit": p_quit,
    "/api/validate": p_validate, "/api/save": p_editor_save, "/api/layout": p_layout,
}
for _x in ("reset", "claude", "run", "visible", "save", "refine"):
    POST_ROUTES[f"/api/bio/{_x}"] = POST_ROUTES[f"/api/scene/{_x}"]


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
            from plexus.gui import studio
            studio.prime_async()
        except Exception as e:                                       # noqa: BLE001
            print(f"[studio] could not start priming: {e}", flush=True)
    return httpd
