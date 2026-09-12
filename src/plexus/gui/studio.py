"""What Claude writes for the page, and where the page's specs and runs live.

`CONFIG_DIR` (config/studio/) is where every tab writes the spec it builds; `out_dir(name)` is
where the pipeline puts that spec's run; `artefacts(name)` is what the run wrote (movie, stills).
`author_spec` is the one Claude call the page makes on its own behalf -- an English EDIT of the
spec on screen (`/api/scene/refine`), primed once per server with the reference corpus.

The Studio page that used to live here -- prompt-to-scene, knobs, previews in a warm worker
process -- is retired: the one page (`gui/app.py`) builds scenes from forms and runs them
in-process through `plexus.pipeline.generate`.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import threading
import time

# --------------------------------------------------------------------------------------- paths
_HERE = os.path.dirname(os.path.abspath(__file__))          # src/plexus/gui
# THREE LEVELS, NOT TWO: gui -> plexus -> src -> the repo. Two put REPO at `src/`, and everything
# derived from it went quietly to the wrong place -- CONFIG_DIR became src/config/studio (so the
# spec list was always empty), MAIN became src/Plexus_Main.py (so no run could start), and the
# priming corpus lost both reference specs and the paper, shrinking to 1,586 chars of operator
# names with nothing to copy the shape of.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
CONFIG_DIR = os.path.join(REPO, "config", "studio")
MAIN = os.path.join(REPO, "Plexus_Main.py")
PY = os.environ.get("PLEXUS_PYTHON") or os.sys.executable

def fail(what: str, detail: str = "") -> None:
    """Print a failure to the server's own terminal, framed so it cannot be missed in a log."""
    print("\n" + "=" * 78, flush=True)
    print(f"[studio] CANNOT GENERATE: {what}", flush=True)
    if detail:
        for line in str(detail).rstrip().splitlines()[-30:]:
            print(f"         {line}", flush=True)
    print("=" * 78 + "\n", flush=True)


def _claude_bin() -> str:
    from shutil import which
    return which("claude") or "claude"


# THE REFERENCE SPEC IS FETCHED BY THE SERVER, NOT READ BY THE AGENT, and that is a measurement not
# a preference. With Read/Glob/Grep enabled over config/ and paper/ the call did not finish inside
# 120 s -- an agent with file access reads files. With one real spec pasted into the prompt and no
# tools it returns in 24 s, inside the 30 s budget. The grounding is the same either way (it is the
# same file, from the same folder); only who fetches it changes. Repo access is still available
# behind DEEP, at a longer cap, for the cases the reference does not cover.
# WHAT GETS PRIMED INTO THE SESSION, ONCE. Two real specs (one drop-scale and capillary, one with
# obstacles and a second material), the REGISTRY's own operator names -- authoritative, unlike a
# paper that can drift -- and the sections of plexus2.tex that decide a spec's shape. Not the whole
# paper: it is 111 KB and most of it is about inverse modelling and the atlas, which no spec needs.
# ONE SPEC PER DISTINCT SHAPE, not every spec. There are ~1,744 configs; priming with all of them
# is megabytes and mostly repetition -- the si_ families are parameter sweeps of the same handful
# of forms. What a spec-writer needs is COVERAGE of the forms: a capillary drop at zero g, an
# obstacle field with a second material, several types sharing one grid, a frame-gated two-stage
# run, and a viscous one. Six of those is ~14 KB and spans the vocabulary.
REFERENCES = ("si_laplace_r10",        # zero g, surface tension, csf_rho/csf_band
              "si_hourglass",          # obstacles as boxes, snow, wall_damp
              "si_crown_drop",         # two liquid types, capillary scale
              "si_split_merge",        # frame-gated operators: before_frame / after_frame
              "si_viscous_spread",     # mpm_viscosity
              "si_restitution",        # elastic solids, four blocks in ONE cell
              # N SEPARATE BODIES, WHICH NOTHING ELSE HERE SHOWED. "add 10 balls" failed because the
              # corpus had no example of the idiom that does it: `n: N` with N `start` positions and
              # `shape: ball`, one body per cell. si_restitution LOOKS like several bodies but is one
              # cell holding four blocks, and a block cannot be a sphere -- so a model reading only
              # that has no way to reach ten balls except by writing ten blocks, which is not what
              # was asked and does not scale.
              "si_two_drops3d_s144",   # n + start + shape: ball -- two bodies, generalises to N
              "si_balls_bouncy",       # n = 3, per-body types and colours
              # PER-TYPE eta, WHICH THE BRIEF ALONE DID NOT ACHIEVE. Told only in prose that `eta`
              # may sit on a type, the model kept writing three liquid types with no eta at all and
              # one operator scalar -- so a "gel" ball came out identical to the water one. A spec
              # that does it is worth more than a paragraph that describes it.
              "si_three_viscosities")  # eta per type: water 1e-3, gel 1.0, snow none
# THE SOURCE, NOT THE PAPER. plexus2.tex describes the language in prose; it does not say which
# SET carries `node_type`, and that omission cost two passes on a real request. "three balls with
# different viscosity" produced `at: mpm_particle[type=...]`, which is an AttributeError every time,
# because `_assign_types` registers `node_type` ONLY on a set that declares `types:` -- the parent
# `cell` does, `mpm_particle` does not. No amount of prose about the operator algebra prevents that;
# forty lines of the function that raises it do.
SOURCE_SLICES = (
    ("src/plexus/engine.py", 573, 630,
     "_assign_types: WHICH sets get node_type and type_names -- only those declaring `types:`"),
    ("src/plexus/engine.py", 1008, 1036,
     "_selector_mask: what `at: set[type=a]` resolves to, and when it cannot"),
    ("src/plexus/models/entities.py", 160, 235,
     "how a body is SIZED and PLACED: particle_mass -> volume -> shape, and block-fill"),
)

SYSTEM_BRIEF = """\
You are writing ONE Plexus2 simulation spec, as YAML, and nothing else.

Copy the SHAPE of the reference: the general/sets/fields/operators/schedule/plotting blocks, the
substep micro-loop, the operator names and their `at`/`to`/`from`. Change only what the request asks
for. The references you were given are the curated ones; never treat a spec under config/studio/ as
a model to copy, since those are this tool's own drafts and may be wrong.

Hard requirements:
  * SI units throughout -- metres, seconds, kilograms, pascals. Keep general.units as given.
  * dim: 3, general.save_data: false.
  * `bulk_modulus` in pascals for liquids, NOT `youngs` (the code zeroes mu for liquids, so a
    Young's modulus on a liquid sets nothing you meant). `youngs` in Pa for solids and snow.
  * Every type must declare `material:` explicitly -- liquid, snow, or elastic.
  * gravity `g: 9.81` unless the scene is explicitly weightless.
  * `surface_tension` in N/m, and only WITH `csf_rho` set to the liquid density; otherwise omit it.
  * Obstacles are axis-aligned boxes [x0,y0,z0,x1,y1,z1] or spheres [cx,cy,cz,r], in metres.
  * `at: <set>[type=<name>]` WORKS ONLY ON A SET THAT DECLARES `types:`. The parent `cell` does;
    `mpm_particle` does NOT, so `at: mpm_particle[type=water]` is always
    "AttributeError: 'Level' object has no attribute 'node_type'". See _assign_types below.
    An operator that acts on the particles acts on ALL of them: `at: mpm_particle`, no selector.

PER-BODY MATERIAL PROPERTIES GO ON THE TYPE, NOT ON THE OPERATOR. Anything that differs between
bodies is declared under `sets.cell.types.<name>`, and the engine expands it to a per-particle
buffer:
    material, bulk_modulus (liquids), youngs (solids/snow), density, and `eta` -- the DYNAMIC
    VISCOSITY in Pa s, per type.
So "one ball of water and one of gel" is `eta: 0.001` on the water type and `eta: 1.0` on the gel
type. `mpm_viscosity` still carries an `eta` of its own; that is only the FALLBACK, used for types
that do not declare one. Water is 1e-3 Pa s, olive oil ~0.08, honey ~10, and anything past ~50
stops flowing on a one-second run.

What genuinely CANNOT vary per body is a parameter the operator owns rather than the material:
mpm_scatter's `drag`, mpm_grid_update's `surface_tension`. Those are one value for the whole run.

N SEPARATE BODIES -- "ten balls", "three drops", "a row of cubes" -- is `sets.cell.n: N` with N
entries under `sets.cell.start` (one [x,y,z] per body, in metres) and a type carrying `shape: ball`
or `shape: cube`. ONE body per cell, placed at its start. Give each type a `fraction` summing to 1;
several types split the bodies between them and each can have its own colour under plotting.colors.
Do NOT write N `block:` entries to make N bodies -- a block is one box at fixed coordinates, it
cannot be a sphere, and it does not scale past a handful. See si_two_drops3d_s144 (n: 2) and
si_balls_bouncy (n: 3) in the material above.

HOW A BODY GETS ITS SIZE. Particle count, particle mass, density and body volume are ONE relation,
not four numbers:

    V_body  =  N * particle_mass / density          N = per_parent * sets.cell.n

so any three of them fix the fourth, and `particles-per-cell` -- the thing that decides whether the
grid can see the material at all -- follows:

    ppc  =  N * dx^3 / V_body                       dx = world / n_grid   (aim for 8)

There are exactly TWO ways to say how big a body is, and you must use one of them:

  * `block: [x0,y0,z0,x1,y1,z1]` in METRES, per type. The block states the volume outright and
    PLACES the particles; the mass then only says what each particle represents. This is the normal
    case -- prefer it.
  * No block, plus `shape: cube` or `shape: ball` and a `start`. Then the volume is DERIVED from
    `per_parent * particle_mass / density` and the shape only decides how that volume is arranged.
    This is the only way to get a sphere, because there is no radius key. If you use it you MUST
    write `per_parent` and `particle_mass` so the intended volume is unambiguous.

THE INTERFACE OWNS THE COUNT, AND PRESERVES YOUR VOLUME. It overwrites
  general.n_frames, general.world, fields.*.n_grid, sets.mpm_particle.per_parent,
  sets.mpm_particle.particle_mass
-- but it reads your V_body FIRST (from the block, or from your per_parent x particle_mass) and
re-derives the mass as `density * V_body / N_new`, so the body you described keeps its size at
whatever particle count the user asked for. Never set a mass and a block that disagree: the engine
warns and then believes the mass, so the picture and the physics come apart.

Place bodies as FRACTIONS of the world box you are given, so they stay put when the box is resized.

OUTPUT: the YAML document ONLY, in a single ```yaml fenced block. No preamble, no explanation, no
second block.
"""


def _extract_yaml(text: str) -> str:
    """The first fenced YAML block, or the whole reply if it is bare YAML."""
    m = re.search(r"```(?:yaml|yml)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    return text


SESSION: dict = {"id": None, "model": None, "primed": 0.0, "chars": 0,
                 "state": "cold", "seconds": 0.0, "error": None}


def prime_async(model: str = "sonnet") -> None:
    """Prime in the background at server start, so the first PREVIEW is not the one that pays."""
    if SESSION.get("state") == "priming":
        return
    SESSION.update(state="priming", error=None)

    def _go():
        r = prime_session(model)
        SESSION.update(state=("ready" if r.get("ok") else "cold"),
                       seconds=r.get("seconds", 0.0), error=r.get("error"))
    threading.Thread(target=_go, daemon=True).start()


def _corpus() -> str:
    """The material the session is primed with: operator names, two specs, the language sections."""
    parts = []
    try:
        import plexus.operators                                  # registers the whole atlas
        from plexus.models.registry import _OPERATOR_REGISTRY as REG   # name -> default impl class
        names = sorted(REG)
        parts.append("REGISTERED OPERATORS (these names, and no others):\n" + ", ".join(names))
    except Exception:                                            # noqa: BLE001
        pass
    for r in REFERENCES:
        # FROM si_material ONLY, never from config/studio: the corpus is the thing every generated
        # spec is modelled on, so anything unreviewed in it propagates into everything after it.
        f = os.path.join(REPO, "config", "si_material", r + ".yaml")
        if os.path.exists(f):
            parts.append(f"REFERENCE SPEC {r}.yaml -- copy this SHAPE:\n```yaml\n"
                         + open(f).read() + "```")
    for rel, a, b, why in SOURCE_SLICES:
        f = os.path.join(REPO, rel)
        if not os.path.exists(f):
            continue
        lines = open(f, errors="replace").read().splitlines()
        parts.append(f"FROM {rel}:{a}-{b} -- {why}\n```python\n"
                     + "\n".join(lines[a - 1:b]) + "\n```")
    return "\n\n".join(parts)


def _session_dir() -> str:
    """Where the CLI keeps this project's transcripts: ~/.claude/projects/<slugged cwd>/."""
    slug = REPO.replace("/", "-")
    return os.path.join(os.path.expanduser("~"), ".claude", "projects", slug)


def _session_id_for(corpus: str) -> str:
    """A session id DERIVED FROM THE CORPUS, so state can be reused and invalidated correctly.

    A random id would re-prime on every server start, paying 4-6 s and a corpus of tokens each
    time for a session the CLI has already written to disk. Hashing the corpus into the uuid gives
    both halves for free: the same corpus finds the same transcript and skips priming entirely,
    and ANY change to it -- another reference spec, an edited tex range -- yields a different id
    and re-primes automatically. No cache to invalidate by hand, which is the kind of cache that
    goes stale and lies.
    """
    import hashlib
    import uuid
    return str(uuid.UUID(hashlib.md5(corpus.encode()).hexdigest()))


def prime_session(model: str = "sonnet", timeout: int = 120) -> dict:
    """Load the corpus into ONE claude session; later requests fork from it.

    WHY THIS EXISTS. A cold `claude -p` per spec request carries the whole reference inline:
    24-30 s, and the same tens of thousands of tokens re-sent every time -- which is what
    pushed it past the 30 s cap. Priming once and forking per request measured 4.5 s to prime and
    14.8 s per generation, with a prompt that is now just the request.

    FORK, DO NOT CONTINUE. `--resume` alone would append every past request to one growing session,
    so the tenth prompt of an evening carries the nine before it and the cost climbs with use.
    `--fork-session` branches from the primed state each time, so the context is constant: corpus +
    the current spec + the instruction. Iteration still works because the current spec is passed
    explicitly, which it has to be anyway -- the file on disk is the truth, not the transcript.
    """
    corpus = _corpus()
    sid = _session_id_for(corpus)
    tr = os.path.join(_session_dir(), sid + ".jsonl")
    if os.path.exists(tr) and os.path.getsize(tr) > 0:
        SESSION.update(id=sid, model=model, primed=os.path.getmtime(tr), chars=len(corpus))
        print(f"[studio] reusing primed session {sid[:8]} from disk "
              f"({len(corpus):,} chars, {os.path.getsize(tr) / 1024:.0f} KB transcript) -- "
              f"no priming needed", flush=True)
        return {"ok": True, "id": sid, "seconds": 0.0, "chars": len(corpus), "reused": True}
    # `--effort low` FOR THE PRIMING TURN ONLY. It has nothing to decide -- it is loading a corpus
    # and saying "ready" -- so thinking about it is pure latency. The generation turns that fork
    # from it run at high effort, which is where the effort belongs.
    msg = (corpus + "\n\nYou will now be asked to write or modify Plexus2 specs against the above. "
           "Reply to THIS message with one short line of acknowledgement and no YAML.")
    t0 = time.time()
    try:
        r = subprocess.run([_claude_bin(), "-p", "--session-id", sid, msg,
                            "--allowedTools", "", "--model", model, "--effort", "low"],
                           cwd=REPO, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        ok = r.returncode == 0
    except Exception as e:                                       # noqa: BLE001
        fail(f"could not prime the Claude session: {e}")
        return {"ok": False, "error": str(e)}
    dt = round(time.time() - t0, 1)
    if not ok:
        fail("could not prime the Claude session", r.stderr[-1500:])
        return {"ok": False, "error": (r.stderr or "")[-300:]}
    SESSION.update(id=sid, model=model, primed=time.time(), chars=len(corpus))
    print(f"[studio] primed session {sid[:8]} with {len(corpus):,} chars "
          f"({len(REFERENCES)} specs + operator registry + plexus2.tex) in {dt}s", flush=True)
    return {"ok": True, "id": sid, "seconds": dt, "chars": len(corpus)}


def author_spec(prompt: str, name: str, current: str = "", timeout: int = 600,
                model: str = "sonnet", deep: bool = False, effort: str = "low",
                error: str = "") -> dict:
    """Write a spec, or MODIFY the one passed in. Returns {yaml, log, seconds}.

    ITERATION IS THE POINT. "a ball of water falling", then "make the ball bigger", then "increase
    the viscosity" -- the second and third only mean anything against the first. When `current` is
    given it is pasted in whole and the instruction is applied to it, so the model edits a spec that
    already validated rather than writing a fresh one that may not.
    """
    if error.strip():
        # THE SECOND PASS. The engine's own message is the best possible description of what went
        # wrong -- better than anything this tool could infer -- and pasting it back is exactly what
        # a person does when a run fails. Automating it removes the copy-and-paste, not the
        # judgement: the model still has to read the traceback and work out what it means.
        msg = (f"This spec was written for: {prompt}\n\n"
               f"```yaml\n{current}\n```\n\n"
               f"It FAILED. The error was:\n\n{error.strip()[-2500:]}\n\n"
               f"Fix the cause and output the COMPLETE corrected spec as one fenced yaml block. "
               f"Change only what the error requires.")
    elif current.strip():
        msg = (f"Here is the current spec:\n\n```yaml\n{current}\n```\n\n"
               f"Apply this change, and change nothing else:\n\n{prompt}\n\n"
               f"Output the COMPLETE modified spec as one fenced yaml block.")
    else:
        msg = f"Write a Plexus2 spec for this scene:\n\n{prompt}\n"
    # EFFECTIVELY UNCAPPED (600 s), because a cap here only ever destroys work. The 30 s budget was
    # set from a measured 24 s cold call, and then the sizing rule went into the brief and effort
    # went to high: the same request now takes 103 s and a 30 s cap returned nothing at all, having
    # paid for all of it. The elapsed time is REPORTED instead, which is the number worth having.
    cmd = [_claude_bin(), "-p", msg, "--append-system-prompt", SYSTEM_BRIEF,
           "--model", model, "--effort", effort]
    # FORK THE PRIMED SESSION when there is one: the corpus is already in it, so this prompt is
    # just the request. Falls back to a cold call with the reference inlined if priming never
    # happened or the session has gone -- slower, but it still answers.
    if not deep and SESSION.get("id") and SESSION.get("model") == model:
        cmd += ["--resume", SESSION["id"], "--fork-session"]
    elif not deep:
        msg_ref = _corpus()
        cmd[2] = f"{msg_ref}\n\n{msg}"
    if deep:
        cmd += ["--allowedTools", "Read", "Glob", "Grep",
                "--add-dir", os.path.join(REPO, "config"),
                "--add-dir", os.path.join(REPO, "paper")]
        timeout = max(timeout, 150)
    else:
        cmd += ["--allowedTools", ""]                     # no tools: the reference is inline
    # SAY IT IN THE TERMINAL, AT BOTH ENDS. The browser shows a spinner; the terminal showed
    # nothing at all between pressing PREVIEW and the spec landing, so a 25 s call and a hung one
    # looked identical from where the server was started -- the same reason the dev channel streams
    # its reads and the render job has a heartbeat.
    _kind = "second pass" if error.strip() else ("edit" if current.strip() else "new spec")
    print(f"\n[studio/claude] {_kind}, effort {effort}"
          f"{' [deep]' if deep else ''}: {prompt.strip()[:110]}", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        out, err, rc = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err, rc = f"timed out after {timeout}s", -1
    dt = time.time() - t0
    _y = _extract_yaml(out).strip()
    print(f"[studio/claude] {'done' if _y else 'NO YAML'} in {dt:.1f}s"
          f"{f' ({len(_y):,} chars)' if _y else f' (rc={rc})'}", flush=True)
    return {"yaml": _y, "log": (err or "")[-4000:], "rc": rc,
            "seconds": round(dt, 1), "raw": out[-8000:]}


# THE FOUR KNOBS THE INTERFACE OWNS, and their defaults.
def out_dir(name: str) -> str:
    from plexus.paths import get_data_root
    return os.path.join(get_data_root(), "graphs_data", "studio", name)


_FPS_CACHE: dict = {}


def _mp4_info(path: str) -> tuple:
    """(fps, n_frames) of a movie, probed ONCE per file and cached on its mtime.

    Stepping a <video> frame by frame needs the real frame interval, and these movies do not have a
    conventional one: LiveMovie DERIVES fps so the clip lasts as long as the world it shows, which
    lands anywhere from 1 to 200. Guessing 1/30 would skip four frames on one spec and repeat three
    on another. ffmpeg is asked instead, and only when the file changes.
    """
    if not path or not os.path.exists(path):
        return 0.0, 0
    key = (path, os.path.getmtime(path))
    if key in _FPS_CACHE:
        return _FPS_CACHE[key]
    fps, n = 0.0, 0
    try:
        import imageio_ffmpeg
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path],
                           capture_output=True, text=True, timeout=20)
        m = re.search(r"([\d.]+)\s+fps", r.stderr or "")
        fps = float(m.group(1)) if m else 0.0
        d = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr or "")
        if d and fps:
            secs = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
            n = int(round(secs * fps))
    except Exception:                                                    # noqa: BLE001
        pass
    _FPS_CACHE[key] = (fps, n)
    return fps, n


def artefacts(name: str) -> dict:
    """What this spec's run wrote: the folder, the poster (`3d.png`), the movie, the newest still.

    TWO PLACES, NEWEST WINS. A run started from the page lands in `graphs_data/studio/<name>`; a
    spec OPENED from a finished run has its own folder (`graphs_data/<type>/<name>`, found by
    `resolve_run`), and that one already holds a movie -- which is what makes PLAY work the moment
    a spec is opened, with no run and no trajectory read."""
    full = out_dir(name)
    if not os.path.exists(os.path.join(full, "movie.mp4")):
        try:
            from plexus.paths import resolve_run
            alt = resolve_run(name)
            if os.path.exists(os.path.join(alt, "movie.mp4")):
                full = alt
        except Exception:                                            # noqa: BLE001 -- no such run
            pass
    png = os.path.join(full, "3d.png")
    mp4 = os.path.join(full, "movie.mp4")
    stills = sorted(glob.glob(os.path.join(full, "still_*.png")))
    return {"dir": full,
            "png": png if os.path.exists(png) else None,
            "mp4": mp4 if os.path.exists(mp4) else None,
            "still": stills[-1] if stills else None,
            "png_mtime": os.path.getmtime(png) if os.path.exists(png) else 0,
            # THE VIDEO NEEDS A BUSTER TOO. `Cache-Control: no-store` covers the browser, but the
            # <video> element keeps whatever it already decoded for a src it has seen before, so a
            # second run at the same path replayed the first one's movie.
            "mp4_mtime": os.path.getmtime(mp4) if os.path.exists(mp4) else 0,
            "mp4_fps": _mp4_info(mp4 if os.path.exists(mp4) else "")[0],
            "mp4_frames": _mp4_info(mp4 if os.path.exists(mp4) else "")[1]}
