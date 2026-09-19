#!/usr/bin/env python
"""Drive the Plexus page over its own HTTP API, so a session can test it without a person.

The page is a browser UI, and everything it does it does through routes that anything can call.
This is those calls, wrapped so one command builds a scene, looks at it, runs it, waits, collects
what the run wrote and reads the engine's own output back. The point is a loop that closes: the
picture and the log come back here, so a fault is seen rather than reported.

    python tools/gui_drive.py shot   --tab platynereis --form '{"render":"somata"}'
    python tools/gui_drive.py run    --tab platynereis --form '{...}' --device cuda:0
    python tools/gui_drive.py status
    python tools/gui_drive.py log    [--tail 60]
    python tools/gui_drive.py caption --mp4 <path>

Every call prints JSON on stdout, and `shot` prints the PATH of a PNG this session can read.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

BASE = os.environ.get("PLEXUS_GUI", "http://127.0.0.1:8799")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL = os.path.join(REPO, "log", "gui_runs", "journal.txt")
# THE HISTORY. Every picture is kept with the SPEC that produced it and the REASON it was taken,
# because a picture alone does not say what was built and a spec alone does not say why. One
# SUBFOLDER PER KIND OF OUTPUT, and the step number is the filename, so a step is read across the
# folders and a kind is read down one:
#
#   builder/png/0007.png        what the scene looked like
#   builder/spec/0007.yaml      the spec that made it -- the whole thing, not a diff
#   builder/why/0007.txt        the time, and in plain words what this step was for
#   builder/mp4/0007.mp4        the movie, when the step ran one
#
# The mp4 is COPIED, not linked: the run folder is rewritten by the next run of the same spec, and
# a history whose oldest entries quietly change is not a history. Nothing here is ever deleted.
# `tools/watch.py` walks it.
HISTORY = os.path.join(REPO, "builder")
KINDS = {"png": ".png", "spec": ".yaml", "why": ".txt", "mp4": ".mp4"}


def _path(kind: str, i: int) -> str:
    """Where step `i` keeps its `kind` of output. Makes the folder, so a new kind costs one line."""
    d = os.path.join(HISTORY, kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{i:04d}{KINDS[kind]}")


def note(msg: str) -> None:
    """One line about what this session just did, for `tools/watch.py` to show.

    A person watching a read-only mirror can see the newest picture but not what produced it, and
    a picture without its cause is not much use -- "8 bodies, glassy" and "run started, 800
    frames" are what make the image legible. Appended, never rewritten, so the order is the order
    things happened.
    """
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")


def _get(path, **q):
    url = f"{BASE}{path}" + ("?" + urllib.parse.urlencode(q) if q else "")
    with urllib.request.urlopen(url, timeout=1800) as r:
        body = r.read()
    try:
        return json.loads(body)
    except Exception:                                                # noqa: BLE001
        return {"raw": body[:400].decode("utf8", "replace")}


def _post(path, payload):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def build(tab: str, form: dict, seed: bool = True) -> dict:
    """Write the tab's spec and seed the scene, which is BUILD + SEED on the page."""
    note(f"build  tab={tab}  name={form.get('name', '?')}  "
         + "  ".join(f"{k}={v}" for k, v in form.items()
                     if k not in ("name", "bodies") and not isinstance(v, (list, dict)))
         + (f"  bodies={len(form['bodies'])}" if isinstance(form.get("bodies"), list) else ""))
    t0 = time.perf_counter()
    r = _post(f"/api/tab/{tab}/build", form)
    if "error" in r:
        note(f"  build FAILED: {str(r.get('error'))[:120]}")
        return {"step": "build", **r}
    out = {"built": r.get("name"), "build_s": round(time.perf_counter() - t0, 2)}
    if seed:
        t1 = time.perf_counter()
        s = _get("/api/scene/seed", name=r.get("name"))
        out["seed_s"] = round(time.perf_counter() - t1, 2)
        if "error" in s:
            out["error"] = s["error"]
            note(f"  seed FAILED: {str(s['error'])[:120]}")
        else:
            out["sets"] = {k: v.get("n_live") for k, v in (s.get("sets") or {}).items()}
            note(f"  seeded in {out['seed_s']}s: "
                 + ", ".join(f"{k} {v:,}" for k, v in out["sets"].items() if v))
    return out


def _next_index() -> int:
    """One past the highest step number ANY folder holds -- a step that produced no movie must not
    let the next step reuse its number."""
    ns = [0]
    for k in KINDS:
        d = os.path.join(HISTORY, k)
        if os.path.isdir(d):
            ns += [int(f[:4]) for f in os.listdir(d) if f[:4].isdigit()]
    return max(ns) + 1


def shot(why: str = "", mp4: str | None = None, **cam) -> dict:
    """The scene as a FILE, kept beside the spec that made it and the reason it was taken.

    `/api/scene/render` streams bytes a curl-only caller cannot look at; `/api/scene/shot` writes
    the same picture and returns its path. This adds the other two thirds of the record.
    """
    t0 = time.perf_counter()
    r = _get("/api/scene/shot", **cam)
    r["shot_s"] = round(time.perf_counter() - t0, 2)

    i = _next_index()
    try:
        import shutil
        shutil.copy(r["path"], _path("png", i))
        st = _get("/api/scene/state")
        name = st.get("name", "")
        sp = _get("/api/scene/spec", name=name) if name else {}
        raw = sp.get("raw") or sp.get("yaml") or ""
        if raw:
            with open(_path("spec", i), "w") as f:
                f.write(raw)
        with open(_path("why", i), "w") as f:
            f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"spec   {name}\n")
            f.write(f"camera azim={cam.get('azim')} elev={cam.get('elev')} zoom={cam.get('zoom')}\n")
            if mp4 and os.path.exists(mp4):
                f.write(f"movie  mp4/{i:04d}.mp4  ({os.path.getsize(mp4) / 1e6:.1f} MB, "
                        f"copied from {mp4})\n")
            f.write(f"why    {why or '(not stated)'}\n")
        if mp4 and os.path.exists(mp4):
            shutil.copy(mp4, _path("mp4", i))
            r["history_mp4"] = _path("mp4", i)
        r["history"] = os.path.join(HISTORY, f"*/{i:04d}.*")
        r["index"] = i
    except Exception as e:                                           # noqa: BLE001
        r["history_error"] = f"{type(e).__name__}: {e}"

    note(f"  shot {i:04d}  azim={cam.get('azim')} elev={cam.get('elev')} zoom={cam.get('zoom')}"
         f"  {r.get('shot_s')}s" + (f"  -- {why}" if why else ""))
    return r


def _mp4_path(art: dict) -> str | None:
    """The movie the run wrote, as a PATH ON DISK.

    The artefacts route answers with browser URLs -- `/media?path=<abs>&t=<mtime>` -- under the
    key `mp4`. An earlier version of this read a key named `video`, which the route has never
    returned, so every cycle silently collected no movie and said nothing about it. Two ways in,
    and the run directory is tried first because it needs no URL to be parsed at all.
    """
    d = art.get("dir") or ""
    if d and os.path.exists(os.path.join(d, "movie.mp4")):
        return os.path.join(d, "movie.mp4")
    u = art.get("mp4") or ""
    if u.startswith("/media?path="):
        return urllib.parse.unquote(u[len("/media?path="):].split("&")[0])
    return None


def status() -> dict:
    return _get("/api/scene/run")


def wait(poll: float = 5.0, timeout: float = 3600.0) -> dict:
    """Block until the run stops, reporting progress. Returns the final status."""
    t0 = time.time()
    last = -1
    while time.time() - t0 < timeout:
        s = status()
        if not s.get("running"):
            return s
        f = s.get("frame", 0)
        if f != last:
            print(f"  frame {f}/{s.get('n_frames')}  {s.get('seconds')}s", file=sys.stderr, flush=True)
            note(f"  running  frame {f}/{s.get('n_frames')}")
            last = f
        time.sleep(poll)
    return {"error": f"still running after {timeout}s", **status()}


def artefacts(name: str) -> dict:
    return _get("/api/scene/artefacts", name=name)


def engine_log(name: str, tail: int = 80) -> dict:
    """The engine's own output for the last run of `name` -- the tee added in bio_view.run."""
    p = os.path.join(REPO, "log", "gui_runs", f"{name}.log")
    if not os.path.exists(p):
        return {"error": f"no log at {p}"}
    with open(p, errors="replace") as f:
        lines = f.read().splitlines()
    return {"log": p, "lines": len(lines), "tail": lines[-tail:]}


def caption(mp4: str, frames: int = 8) -> dict:
    """Ask the local Gemma VLLM what the movie shows. It is the only reader here that can say
    'the body never moves' without being told what to look for."""
    script = os.path.join(REPO, "VLLM", "describe_video.py")
    if not os.path.exists(script):
        return {"error": f"no {script}"}
    out = os.path.join(REPO, "log", "gui_runs", "captions.txt")
    r = subprocess.run([sys.executable, script, mp4, "--out", out, "--frames", str(frames)],
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        return {"error": r.stderr[-400:]}
    txt = open(out, errors="replace").read() if os.path.exists(out) else ""
    return {"out": out, "caption": txt[-1200:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "shot", "run", "status", "wait", "artefacts",
                                    "log", "caption", "cycle"])
    ap.add_argument("--tab", default="platynereis")
    ap.add_argument("--form", default="{}")
    # A FORM IS OFTEN TOO BIG FOR A COMMAND LINE. The material tab's form carries one entry per
    # body -- twenty-seven of them in the default scene, each with a block, a colour and a
    # material -- which is kilobytes of JSON and a shell-quoting hazard. `--form-file` takes the
    # same object from a file instead.
    ap.add_argument("--form-file", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mp4", default=None)
    ap.add_argument("--tail", type=int, default=80)
    ap.add_argument("--azim", type=float, default=35.0)
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--zoom", type=float, default=1.3)
    ap.add_argument("--no-caption", action="store_true")
    # WHY THIS STEP EXISTS, in the session's own words. It is the third file of the record and the
    # only one that cannot be reconstructed from the others.
    ap.add_argument("--why", default="")
    a = ap.parse_args()
    form = json.load(open(a.form_file)) if a.form_file else json.loads(a.form)

    if a.cmd == "build":
        print(json.dumps(build(a.tab, form), indent=1))
    elif a.cmd == "shot":
        print(json.dumps(shot(a.why, a.mp4, azim=a.azim, elev=a.elev, zoom=a.zoom), indent=1))
    elif a.cmd == "status":
        print(json.dumps(status(), indent=1))
    elif a.cmd == "wait":
        print(json.dumps(wait(), indent=1))
    elif a.cmd == "artefacts":
        print(json.dumps(artefacts(a.name or form.get("name", "")), indent=1))
    elif a.cmd == "log":
        print(json.dumps(engine_log(a.name or form.get("name", ""), a.tail), indent=1))
    elif a.cmd == "caption":
        print(json.dumps(caption(a.mp4, ), indent=1))
    elif a.cmd == "run":
        note(f"run  device={a.device}")
        print(json.dumps(_post("/api/scene/run", {"device": a.device}), indent=1))
    elif a.cmd == "cycle":
        # THE WHOLE LOOP, which is the reason this file exists: build, look, run, wait, collect,
        # read the engine's own words, and ask the VLLM what the movie shows.
        rep = {"build": build(a.tab, form)}
        if rep["build"].get("error"):
            print(json.dumps(rep, indent=1)); return
        rep["shot_before"] = shot(f"{a.why} (before the run)".strip(),
                                  azim=a.azim, elev=a.elev, zoom=a.zoom)
        note(f"run  device={a.device}")
        rep["started"] = _post("/api/scene/run", {"device": a.device})
        rep["final"] = wait()
        nm = rep["build"]["built"]
        rep["artefacts"] = artefacts(nm)
        rep["engine_log"] = engine_log(nm, a.tail)
        rep["mp4"] = mp4 = _mp4_path(rep["artefacts"])
        rep["shot_after"] = shot(f"{a.why} (after the run)".strip(), mp4,
                                 azim=a.azim, elev=a.elev, zoom=a.zoom)
        if mp4 and not a.no_caption and os.path.exists(mp4):
            rep["caption"] = caption(mp4)
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
