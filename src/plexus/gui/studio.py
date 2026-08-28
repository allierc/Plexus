#!/usr/bin/env python
"""`plexus.gui.studio` -- say what you want, look at frame 0, then render it.

    PYTHONPATH=src python -m plexus.gui --studio        # http://localhost:PORT/studio

WHAT IT IS. One page: a scene box, a prompt under it, and two buttons. You type a scene in English;
Claude reads the real `config/` specs and `paper/plexus2.tex` and writes a spec; the server VALIDATES
it, saves it under `config/studio/`, and renders frame 0 so you can see whether the geometry is what
you meant before spending an hour on it. GENERATE then runs the ordinary pipeline with a progress
bar; VIEW opens the spec text with a SAVE that runs the same validator.

WHAT IT DELIBERATELY DOES NOT DO: any rendering, any generation, any validation of its own. Frame 0
and the full run are the SAME command -- `Plexus_Main.py -o generate studio/<name>` -- differing only
in `n_frames`, so a preview cannot diverge from the thing it is previewing, which is the entire point
of a preview. Validation is `plexus.schema.load`, the gatekeeper the engine already trusts. This
module is a window onto the pipeline, not a second one.

THE CLAUDE CALL IS AGENTIC BUT READ-ONLY, and both halves of that matter.
  * Agentic, because a spec is not guessable from a prompt: the operator names, the schedule form
    and the substep block are conventions that live in `config/` and in the paper. Claude gets Read,
    Glob and Grep over `config/` and `paper/` so it can copy a working spec's shape rather than
    invent one.
  * Read-only, because nothing it writes should reach disk unchecked. It returns YAML on stdout; the
    SERVER parses it, runs `plexus.schema.load`, and only then writes. An invalid spec is reported
    in the status line and never lands.
  * Capped at 30 s. An agentic call with no budget will happily read forty files.
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

# THE PREVIEW IS THE SAME RUN, ONE FRAME LONG. Not a separate renderer: the generator already writes
# `3d.png` at the end of a run, so `n_frames: 0` leaves a picture of the seeded scene.
#
# THE ENGINE ITERATES `range(n_frames + 1)`, so this is one MORE than the frames rendered: 2 gave
# three (the bar read 0/3) and 1 gives two. It is not 0, which would be one frame and is the
# obvious thing to want: at `n_frames: 0` the run completes and writes a trajectory but the
# renderer produces NO movie, NO stills and NO 3d.png, so the preview has nothing to show. (It also
# used to crash outright inside LiveMovie on a division by the run's duration -- that guard is
# worth keeping either way, but it only got as far as producing nothing.)
#
# THE SAVING IS SMALL AND WORTH SAYING SO. Measured at 10k particles: 3 frames 16.0 s, 2 frames
# 14.1 s, and capture off vs on 15.2 s vs 16.8 s. Roughly 12 s of every preview is interpreter and
# warp import before a single particle moves, so the frame count is not where a fast preview would
# come from -- a warm worker process would be.
PREVIEW_FRAMES = 1


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
              "si_restitution")        # elastic solids, several types at once
TEX_SECTIONS = ((125, 258),      # The language: operators, sets, states, fields, hierarchy
                (934, 1045))     # Schedules, model specification, units

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
    tex = os.path.join(REPO, "paper", "plexus2.tex")
    if os.path.exists(tex):
        lines = open(tex, errors="replace").read().splitlines()
        for a, b in TEX_SECTIONS:
            parts.append("FROM plexus2.tex:\n" + "\n".join(lines[a - 1:b]))
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

    WHY THIS EXISTS. Every spec request used to be a cold `claude -p` carrying the whole reference
    inline: 24-30 s, and the same tens of thousands of tokens re-sent every time -- which is what
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
                model: str = "sonnet", deep: bool = False, effort: str = "low") -> dict:
    """Write a spec, or MODIFY the one passed in. Returns {yaml, log, seconds}.

    ITERATION IS THE POINT. "a ball of water falling", then "make the ball bigger", then "increase
    the viscosity" -- the second and third only mean anything against the first. When `current` is
    given it is pasted in whole and the instruction is applied to it, so the model edits a spec that
    already validated rather than writing a fresh one that may not.
    """
    if current.strip():
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
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        out, err, rc = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err, rc = f"timed out after {timeout}s", -1
    dt = time.time() - t0
    return {"yaml": _extract_yaml(out).strip(), "log": (err or "")[-4000:], "rc": rc,
            "seconds": round(dt, 1), "raw": out[-8000:]}


# THE FOUR KNOBS THE INTERFACE OWNS, and their defaults.
DEFAULTS = {"frames": 500, "particles": 100_000, "n_grid": 96, "width": 0.10}


def _body_volume(spec: dict) -> float:
    """The volume the particles occupy, in m^3, AS THE SPEC CURRENTLY STANDS.

    Two ways a spec can say how big its body is, and only these two. An explicit `block` per type
    states it directly. Otherwise the body is DERIVED from `per_parent * particle_mass / density`
    (models/entities.py), so the volume has to be read off those before they are overwritten --
    computing it after would be circular, since the new mass is what this volume is used to set.

    `radius` is NOT a third way. It is a fraction, not a length; a first version fell back to
    4/3 pi r^3 with radius 0.5 and reported a body of 523,598 cm^3 in a 1-litre box.
    """
    tot = 0.0
    sets = spec.get("sets") or {}
    types = ((sets.get("cell") or {}).get("types") or {})
    for t in types.values():
        b = t.get("block")
        if b and len(b) >= 6:
            v = [float(x) for x in b]
            tot += abs((v[3] - v[0]) * (v[4] - v[1]) * (v[5] - v[2]))
    if tot <= 0.0:
        mp = sets.get("mpm_particle") or {}
        n = int((mp.get("per_parent") or 0)) * max(1, int(((sets.get("cell") or {}).get("n", 1)) or 1))
        m = float(mp.get("particle_mass") or 0.0)
        rho = float(mp.get("density") or 1000.0)
        tot = n * m / rho if (n and m and rho) else 0.0
    return max(tot, 1e-12)


def apply_knobs(spec: dict, k: dict) -> dict:
    """Force frames / world / n_grid / particle count, and keep mass consistent with geometry.

    WHY THE INTERFACE OWNS THESE AND THE MODEL DOES NOT. They are the four numbers that decide
    whether a spec is cheap or ruinous, and three of them are not independent: particles-per-cell is
    N*dx^3/V, so a model choosing N and n_grid separately produces either a body the grid cannot see
    or one it over-samples by two orders of magnitude. Making them controls also takes the whole
    sizing calculation off the model's plate, which is what brings the call inside 30 s.

    `particle_mass` is RECOMPUTED, never taken from the reply: with an explicit `block`, the block
    places the particles and the mass says what each represents, and if N*mass/rho does not equal
    the block volume the engine warns that the two disagree and uses the mass. Deriving it here from
    the blocks the model actually wrote means they cannot disagree.
    """
    w = float(k.get("width", DEFAULTS["width"]))
    ng = int(k.get("n_grid", DEFAULTS["n_grid"]))
    n_tot = int(k.get("particles", DEFAULTS["particles"]))
    g = spec.setdefault("general", {})
    g["n_frames"] = int(k.get("frames", DEFAULTS["frames"]))
    old_w = float((g.get("world") or [w])[0] or w)
    g["world"] = [w, w, w]
    g["save_data"] = False
    g["dim"] = 3
    # RESCALE THE GEOMETRY WITH THE BOX. The model is told to place bodies as fractions of the world
    # it was given; if it used the reference's box instead, everything would sit outside a smaller
    # one. Scaling every declared length by the ratio keeps the scene the scene.
    if old_w > 0 and abs(old_w - w) > 1e-12:
        r = w / old_w
        for t in (((spec.get("sets") or {}).get("cell") or {}).get("types") or {}).values():
            if t.get("block"):
                t["block"] = [float(x) * r for x in t["block"]]
        if g.get("obstacles"):
            g["obstacles"] = [[float(x) * r for x in o] for o in g["obstacles"]]
        st = (spec.get("sets") or {}).get("cell") or {}
        if st.get("start"):
            st["start"] = [[float(x) * r for x in p] for p in st["start"]]
    for fc in (spec.get("fields") or {}).values():
        if isinstance(fc, dict) and "n_grid" in fc:
            fc["n_grid"] = ng
    mp = (spec.get("sets") or {}).get("mpm_particle") or {}
    n_bodies = max(1, int(((spec.get("sets") or {}).get("cell") or {}).get("n", 1) or 1))
    rho = float(mp.get("density", 1000.0) or 1000.0)
    V = _body_volume(spec)                       # BEFORE per_parent / particle_mass are touched
    if V <= 1e-11:
        # NEITHER WAY OF SAYING THE SIZE WAS USED. No `block`, and no per_parent x particle_mass to
        # derive one from -- so there is no volume to preserve and any mass chosen here would be
        # invented. Say so rather than shipping a body of 1e-12 m^3.
        raise ValueError(
            "the spec does not say how big its body is: give every type a `block:` in metres, or "
            "(for a sphere) `shape: ball` with `per_parent` and `particle_mass`, since "
            "V_body = N * particle_mass / density is the only way the size is expressed")
    mp["per_parent"] = max(1, n_tot // n_bodies)
    mp["particle_mass"] = float(f"{rho * V / max(n_bodies * mp['per_parent'], 1):.6g}")
    spec["_body_volume_m3"] = V                  # so knob_report does not recompute it circularly
    return spec


# THE TWO NUMBERS THAT DECIDE WHETHER A RUN IS WORTH STARTING, and their bands.
#
# ppc -- particles per cell, N*dx^3/V. 8 is the target (two per axis, what the quadratic B-spline
# needs to see a filled cell). Below 1 the grid cannot see the material at all; far above it the
# extra particles buy nothing the grid can represent and cost time linearly. Both failure modes are
# invisible in the picture until it is too late, which is why they get a light each.
#
# CFL -- the declared substep against the stability limit, micro_dt / dt_cfl. Over 1.0 the step is
# unstable AS WRITTEN; the CFL pass then shrinks it and the substep count rises to pay for it, so
# red here means "this will be slower than you think", not "this will explode".
def metrics(spec: dict, k: dict) -> dict:
    """CFL and particles-per-cell for this spec, BOTH from the engine's own reporters.

    Neither number is recomputed here. `particles_per_cell` already walks the declared bodies, sizes
    each from its block or radius, and divides by the CELL SIZE rather than n_grid^dim (the two agree
    only on a unit box -- on a 0.1 m box the naive form is wrong by 1000x). `Courant_..._condition`
    already knows the elastic and capillary limits and which one binds. Reimplementing either here
    would give the studio a second opinion, and a second opinion is exactly what a gauge must not be.

    THE WORST BODY IS THE ONE REPORTED. A spec with a well-sampled pool and an under-sampled drop is
    an under-sampled spec: the drop is what will fracture.
    """
    import contextlib
    import io
    import tempfile
    import yaml as _y
    from plexus.generators.mpm_cfl import (PPC_FLOOR, PPC_TARGET,
                                           Courant_Friedrichs_Lewy_condition as CFL,
                                           particles_per_cell)
    dim = int((spec.get("general") or {}).get("dim", 3))
    target = PPC_TARGET if dim == 3 else 4.0
    floor = target * PPC_FLOOR                       # the engine's own under-sampled line
    ng = int(k.get("n_grid", DEFAULTS["n_grid"]))
    dx = float(k.get("width", DEFAULTS["width"])) / max(ng, 1)

    ppc = {"color": "dim", "text": "--", "value": 0.0, "suggest": ""}
    cfl = {"color": "dim", "text": "not an MPM spec", "value": 0.0, "ok": False, "suggest": ""}
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            _y.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
            tmp = f.name
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rows = particles_per_cell(tmp)
            changed, info = CFL(tmp)
        if rows:
            label, v, cells, n = min(rows, key=lambda r: r[1])
            col = "red" if v < floor else ("green" if v <= target * 4 else "amber")
            # THE SAME TWO CURES THE ENGINE'S OWN WARNING OFFERS, with the numbers filled in: the
            # grid that would put this body at the target, and the particle count that would. Both
            # are given because they are not equivalent -- one changes what the grid can resolve,
            # the other only changes cost -- and choosing between them is not this tool's call.
            vol = cells * (dx ** dim)
            W = float(k.get("width", DEFAULTS["width"]))
            want_grid = max(1, int(round(W * (n / (target * vol)) ** (1.0 / dim))))
            want_n = int(target * cells)
            # "coarser"/"finer" is DERIVED, not asserted. Saying it the wrong way round is worse
            # than saying nothing: it sends you in the direction that makes the number worse.
            way = "coarser" if want_grid < ng else "finer"
            sug = ""
            if v < floor:
                sug = (f"UNDER-SAMPLED: the grid cannot see this body. "
                       f"n_grid {want_grid} ({way}), or {want_n:,} particles.")
            elif v > target * 4:
                sug = (f"oversampled {v / target:.0f}x: costs time, resolves nothing more. "
                       f"n_grid {want_grid} ({way}), or {want_n:,} particles.")
            ppc = {"value": round(v, 2), "color": col, "suggest": sug,
                   "text": f"{v:.1f} per cell   {label}   {n:,.0f} over {cells:,.0f} cells"}
        if info:
            # TWO SHAPES OF `info`, because the pass either accepted the step or replaced it: the
            # accepting branch reports `micro_dt`/`dt_cfl`, the correcting one `dt_old`/`over_by`.
            # Reading only the first raised KeyError on every spec that needed correcting -- which
            # is to say on exactly the specs the gauge exists for.
            r = float(info.get("over_by") or
                      (float(info["micro_dt"]) / max(float(info["dt_cfl"]), 1e-30)))
            after = _y.safe_load(open(tmp))
            blk = next((b for b in (after.get("schedule") or [])
                        if isinstance(b, dict) and "substep_dt" in b), None)
            subs = round(float(after["general"]["dt"]) / float(blk["substep_dt"])) if blk else 0
            col = "green" if r <= 0.8 else ("amber" if r <= 1.0 else "red")
            # SUBSTEPS SCALE WITH n_grid, which is the lever worth naming. The stable step is
            # cfl*dx/c and dx = world/n_grid, so substeps per frame go as n_grid: halving the grid
            # halves the cost of every frame. The other lever is the stiffness -- substeps go as
            # sqrt(K) -- and it changes the physics, so it is named second and last.
            sug = ""
            if changed or r > 1.0:
                half = max(16, int(ng / 2))
                sug = (f"the declared step was {r:.1f}x too big and has been corrected to "
                       f"{subs} substeps/frame. Substeps scale with n_grid: {half} would cost "
                       f"about {max(1, int(subs / 2))}. A softer bulk_modulus also helps "
                       f"(substeps go as sqrt(K)) but changes the physics.")
            elif subs > 200:
                sug = (f"{subs} substeps/frame is the run's real cost. n_grid "
                       f"{max(16, int(ng / 2))} would roughly halve it.")
            cfl = {"value": round(r, 3), "ok": r <= 1.0, "color": col, "suggest": sug,
                   "text": (f"{r:.2f}x of the limit   {subs} substeps/frame   "
                            f"c {float(info['cmax']):.0f} m/s   binds: {info.get('binds', '-')}"
                            + ("   CORRECTED" if changed else ""))}
    except Exception as e:                                              # noqa: BLE001
        cfl = {"value": 0.0, "ok": False, "color": "red", "suggest": "",
               "text": f"{type(e).__name__}: {e}"[:130]}
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    return {"ppc": ppc, "cfl": cfl}


def knob_report(spec: dict, k: dict) -> str:
    """What the knobs came out to, in the terms that decide whether the run is sane."""
    w = float(k.get("width", DEFAULTS["width"]))
    ng = int(k.get("n_grid", DEFAULTS["n_grid"]))
    n = int(k.get("particles", DEFAULTS["particles"]))
    dx = w / ng
    V = float(spec.pop("_body_volume_m3", 0.0)) or _body_volume(spec)
    return (f"{n:,} particles  |  {ng}^3 cells, dx {dx * 1000:.2f} mm  |  body {V * 1e6:.1f} cm^3"
            f"  |  ppc {n * dx ** 3 / V:.1f}")


# ------------------------------------------------------------------------------ the dev channel
DEV_BRIEF = """\
You are asked for something the current Plexus2 cannot express. Do NOT write a spec, and do NOT
edit any file. Produce an ANALYSIS of what the engine would need.

Never read config/studio/ -- those are generated drafts, unreviewed and sometimes wrong. The
curated specs are config/si_material/ and config/material/.

Read first, and say what you read: src/plexus/operators/ (the operator library -- registry-driven,
one class per operator with PARAM_ROLES, EMIT, SUPPORTED_DIMS), src/plexus/models/registry.py
(how an operator is registered and dispatched), src/plexus/schema.py (what a spec is allowed to
say), and paper/plexus2.tex for the vocabulary and the intent.

Answer these, in this order, and nothing else:

  1. WHAT IS MISSING. State precisely which part of the request the current operators cannot
     express, and name the operators that come closest and why they fall short. If it IS already
     expressible, say so and give the spec fragment that does it -- that is the most useful
     possible answer and it ends here.
  2. THE MECHANISM. The physics or rule to be added, as an equation or a clear procedure, with the
     quantity each symbol denotes and its units.
  3. THE OPERATOR. Its name, `family`, `kind`, which set or field it acts `at`, what it reads,
     what it EMITs, its parameters with roles, units and defaults. Say whether it is a NEW operator
     or a new `implementation:` of an existing one -- the latter is much cheaper and is right
     whenever the mechanism is unchanged and only the numerics differ.
  4. THE EDIT. The files to touch and what changes in each, concretely. Flag anything that would
     alter existing runs, because that is the expensive part.
  5. THE TEST. One closed form, invariant or limiting case that would show the new operator is
     right rather than merely running.

Be specific and short. Cite file:line where you can. If the request is under-determined, say which
decision you would need from a person rather than choosing for them.
"""

DEV: dict[str, dict] = {}


def report(title: str, body: str) -> None:
    """Print an analysis to the server's own terminal, framed."""
    print("\n" + "=" * 78, flush=True)
    print(f"[studio/dev] {title}", flush=True)
    print("=" * 78, flush=True)
    for line in str(body).rstrip().splitlines():
        print(line, flush=True)
    print("=" * 78 + "\n", flush=True)


def start_dev(prompt: str, model: str = "sonnet", timeout: int = 900) -> dict:
    """Ask what Plexus would need in order to do this. Runs in a thread; result goes to stdout.

    FULL READ ACCESS AND A LONG LEASH, unlike the spec path. Writing a spec is a 30 s job precisely
    because the interface hands over the reference and owns the sizing; working out which operator
    is missing is the opposite -- it needs the operator library, the registry and the schema, and
    there is no shortcut that keeps it honest. So this one reads the repo, takes minutes, and is
    polled rather than awaited.

    IT WRITES NOTHING, AND THAT IS ENFORCED BY DENY, NOT BY ALLOW. `--allowedTools Read Glob Grep`
    turned out to be a permission HINT, not a restriction: the first run happily made 26 tool calls
    of which several were Bash. `--disallowedTools` is the half that actually forbids, so the write
    and shell tools are named there explicitly. Measured before and after.

    FIFTEEN MINUTES, because the first attempt died at 150 s mid-read having called 26 tools and
    produced nothing at all -- a cap that stops an analysis before its conclusion buys nothing and
    throws away the work. It is a background task that prints its progress; it can take its time.
    """
    key = "dev"
    if DEV.get(key, {}).get("running"):
        return {"error": "an analysis is already running"}
    DEV[key] = {"running": True, "prompt": prompt, "started": time.time(),
                "text": "", "error": None, "seconds": 0}

    def _go():
        # STREAMED, NOT AWAITED IN SILENCE. A minutes-long background call that prints only at the
        # end is indistinguishable from a hung one, and the interesting part of an analysis is
        # WHICH FILES it went to -- that is how you tell an answer grounded in the operator library
        # from one grounded in a guess. `--output-format stream-json --verbose` emits an event per
        # step, so each tool call becomes one line here as it happens.
        cmd = [_claude_bin(), "-p", prompt,
               "--append-system-prompt", DEV_BRIEF,
               "--allowedTools", "Read", "Glob", "Grep",
               # NAMES THAT EXIST. "MultiEdit" is not a tool and the CLI warns "matches no known
               # tool" for it, which is the kind of line that trains you to ignore warnings.
               "--disallowedTools", "Write", "Edit", "NotebookEdit", "Bash",
               "WebFetch", "WebSearch", "Task",
               # CURATED FOLDERS ONLY. `--add-dir config` handed it config/studio/ as well, and
               # those are the studio's OWN generated drafts -- unreviewed, sometimes wrong, and
               # written by the same model. Feeding them back as reference is how a mistake becomes
               # a convention. si_material/ and material/ are the folders a person maintains.
               "--add-dir", os.path.join(REPO, "src"),
               "--add-dir", os.path.join(REPO, "config", "si_material"),
               "--add-dir", os.path.join(REPO, "config", "material"),
               "--add-dir", os.path.join(REPO, "paper"),
               "--model", model, "--effort", "high",
               "--output-format", "stream-json", "--verbose"]
        t0 = time.time()
        print(f"\n[studio/dev] {prompt}", flush=True)
        print(f"[studio/dev] reading src/plexus, config/, paper/ ... (cap {timeout}s)", flush=True)
        out, err, n_tool = "", "", 0
        try:
            pr = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1)
            for line in pr.stdout:                                   # type: ignore[union-attr]
                if time.time() - t0 > timeout:
                    pr.kill(); err = f"timed out after {timeout}s"; break
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except Exception:                                    # noqa: BLE001
                    continue
                t = ev.get("type")
                if t == "assistant":
                    for c in (ev.get("message") or {}).get("content") or []:
                        if c.get("type") == "tool_use":
                            n_tool += 1
                            inp = c.get("input") or {}
                            what = (inp.get("file_path") or inp.get("pattern")
                                    or inp.get("path") or "")
                            what = str(what).replace(REPO + "/", "")
                            print(f"[studio/dev]   {time.time() - t0:6.1f}s  "
                                  f"{c.get('name', '?'):<5} {what[:88]}", flush=True)
                        elif c.get("type") == "text" and c.get("text", "").strip():
                            DEV[key]["text"] = c["text"]
                elif t == "result":
                    out = (ev.get("result") or "").strip() or DEV[key].get("text", "")
            pr.wait(timeout=10)
            err = err or (pr.stderr.read().strip() if pr.stderr else "")
        except Exception as e:                                       # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        dt = round(time.time() - t0, 1)
        out = out or DEV[key].get("text", "")
        DEV[key].update(running=False, text=out, error=(err if not out else None), seconds=dt,
                        tools=n_tool)
        if out:
            report(f"{prompt}   [{dt}s, {n_tool} file reads]", out)
        else:
            fail(f"dev analysis produced nothing for {prompt!r}", err)

    threading.Thread(target=_go, daemon=True).start()
    return {"started": True}


# --------------------------------------------------------------------------------- run tracking
class Job:
    """One `Plexus_Main.py -o generate` subprocess, with its tqdm counter scraped for a bar.

    NOT a reimplementation of the generator: it starts the same command a person would type and
    reads its stdout. The frame counter tqdm already prints is the progress bar's only source, so
    the bar cannot claim progress the run has not made.
    """

    _RE = re.compile(r"(\d+)/(\d+)\s*\[")            # tqdm's "  184/801 [04:43<15:54, ...]"

    def __init__(self, name: str, device: str, frames: int | None = None,
                 render_n: int = 400_000, tag: str = ""):
        self.name, self.device, self.tag = name, device, tag
        self.frame, self.total = 0, frames or 0
        self.done, self.rc, self.error = False, None, None
        self.lines: list[str] = []
        self.started = time.time()
        cmd = [PY, "-u", MAIN, "-o", "generate", f"studio/{name}",
               "--device", device, "--no-describe", "--force",
               "--render-n", str(render_n)]
        self.cmd = cmd
        self.p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        # A HEARTBEAT IN THE TERMINAL, every 5 s. The browser has the bar; the terminal has nothing
        # unless it is told, and a background render that prints only on failure looks identical to
        # one that never started.
        last = 0.0
        print(f"\n[studio] {self.tag} {self.name!r} on {self.device} -> "
              f"{' '.join(self.cmd[-6:])}", flush=True)
        for raw in self.p.stdout:                                        # type: ignore[union-attr]
            for chunk in raw.replace("\r", "\n").split("\n"):
                if not chunk.strip():
                    continue
                m = self._RE.search(chunk)
                if m:
                    self.frame, self.total = int(m.group(1)), int(m.group(2))
                    now = time.time()
                    if now - last > 5.0:
                        last = now
                        pct = 100.0 * self.frame / max(self.total, 1)
                        print(f"[studio]   {self.tag} {self.name}  {self.frame}/{self.total}"
                              f"  {pct:5.1f}%  {now - self.started:6.1f}s", flush=True)
                elif len(self.lines) < 400:
                    self.lines.append(chunk[:400])
        self.rc = self.p.wait()
        self.done = True
        if self.rc != 0:
            tail = [l for l in self.lines if l.strip()][-6:]
            self.error = " / ".join(tail) or f"exited {self.rc}"
            # IN THE TERMINAL, NOT ONLY IN THE BROWSER. A run that cannot be generated is the one
            # thing you most need the full text of -- a schema message, a CUDA OOM, a missing
            # operator -- and a status line in a web page truncates it. This prints where you
            # started the server, next to everything else the pipeline says.
            fail(f"{self.tag or 'run'} {self.name!r} exited {self.rc}",
                 "\n".join(self.lines[-25:]) or "(no output)")
        else:
            print(f"[studio]   {self.tag} {self.name} DONE in "
                  f"{time.time() - self.started:.1f}s\n", flush=True)

    def kill(self):
        try:
            self.p.terminate()
        except Exception:                                                # noqa: BLE001
            pass

    def status(self) -> dict:
        return {"name": self.name, "tag": self.tag, "frame": self.frame, "total": self.total,
                "done": self.done, "rc": self.rc, "error": self.error,
                "elapsed": round(time.time() - self.started, 1),
                "pct": (100.0 * self.frame / self.total) if self.total else 0.0,
                "tail": self.lines[-8:]}


JOBS: dict[str, Job] = {}
PREVIEW_PREFIX = "__preview__"


def start_run(name: str, device: str = "cuda:1", preview: bool = False) -> dict:
    """Launch the ordinary generator on this spec. `preview` = the same run, PREVIEW_FRAMES long.

    THE PREVIEW IS A REAL SPEC ON DISK, not an in-memory tweak. Rewriting `n_frames` in the file,
    running, and putting it back would be a race with the editor and would leave the file wrong if
    anything threw; and `Plexus_Main.py` takes no `--n-frames`, so there is nowhere else to put it.
    A sibling `__preview__<name>.yaml` costs one file and keeps the real spec untouched -- it is
    filtered out of the spec list so it never looks like something you wrote.
    """
    import yaml as _y
    if name in JOBS and not JOBS[name].done:
        return {"error": f"{name} is already running"}
    src = os.path.join(CONFIG_DIR, name + ".yaml")
    if not os.path.exists(src):
        return {"error": f"no such spec: {name}"}
    run_name = name
    if preview:
        s = _y.safe_load(open(src))
        s["general"]["n_frames"] = PREVIEW_FRAMES
        s["general"]["name"] = run_name = PREVIEW_PREFIX + name
        s["general"]["save_data"] = False
        # NO COMPILE AND NO GRAPH CAPTURE FOR A TWO-FRAME RUN. Both are amortised optimisations:
        # torch.compile pays tens of seconds of tracing to make later substeps cheaper, and the
        # CUDA-graph capture runs the substep three times to warm up and once more to record. Over
        # 2400 frames that is free; over PREVIEW_FRAMES it is the entire cost, and the preview
        # exists to be fast. The full GENERATE keeps both -- it is the run that has frames to
        # amortise them over.
        for blk in s.get("schedule", []):
            if isinstance(blk, dict) and "substep_dt" in blk:
                blk["compile"] = False
                blk["capture"] = False
        _y.safe_dump(s, open(os.path.join(CONFIG_DIR, run_name + ".yaml"), "w"),
                     sort_keys=False, default_flow_style=False)
    j = Job(run_name, device, tag="preview" if preview else "generate",
            render_n=200_000 if preview else 400_000)
    j.for_spec = name                                                    # type: ignore[attr-defined]
    JOBS[name] = j
    return {"started": True, "name": name, "run_name": run_name, "tag": j.tag}


def out_dir(name: str) -> str:
    from plexus.paths import get_data_root
    return os.path.join(get_data_root(), "graphs_data", "studio", name)


def artefacts(name: str) -> dict:
    """What this spec has produced, showing the NEWEST frame of the two places one can land.

    Both a full run and a preview write a `3d.png`, into `studio/<name>/` and
    `studio/__preview__<name>/`. Preferring the full run's outright was wrong the moment the loop
    became iterative: edit a generated spec, press PREVIEW, and the fresh preview frame was hidden
    behind the stale picture of the run before the edit -- Claude answered, the render succeeded,
    and the scene did not change. Newest wins, which is the only rule that means "what you are
    looking at is what you last asked for".
    """
    full, prev = out_dir(name), out_dir(PREVIEW_PREFIX + name)
    cands = [p for p in (os.path.join(full, "3d.png"), os.path.join(prev, "3d.png"))
             if os.path.exists(p)]
    png = max(cands, key=os.path.getmtime) if cands else None
    mp4 = os.path.join(full, "movie.mp4")
    stills = sorted(glob.glob(os.path.join(full, "still_*.png")))
    return {"dir": full,
            "png": png,
            "mp4": mp4 if os.path.exists(mp4) else None,
            "still": stills[-1] if stills else None,
            "png_mtime": os.path.getmtime(png) if png else 0,
            # THE VIDEO NEEDS A BUSTER TOO. `Cache-Control: no-store` covers the browser, but the
            # <video> element keeps whatever it already decoded for a src it has seen before, so a
            # second GENERATE at the same path replayed the first one's movie.
            "mp4_mtime": os.path.getmtime(mp4) if os.path.exists(mp4) else 0}


def list_specs() -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))):
        n = os.path.splitext(os.path.basename(p))[0]
        if n.startswith(PREVIEW_PREFIX):
            continue                                   # a run artefact, not something you wrote
        a = artefacts(n)
        out.append({"name": n, "path": p, "has_png": bool(a["png"]), "has_mp4": bool(a["mp4"]),
                    "mtime": os.path.getmtime(p)})
    return sorted(out, key=lambda r: -r["mtime"])


# ------------------------------------------------------------------------------------ the page
# The visual language is ngp-demo's: black ground, white ink, 1 px borders, square controls, small
# uppercase letterspaced labels, tabular numerals. No framework, no build step -- the page is a
# string and the JS is inline, so the whole UI is greppable from one file.
CSS = """
  :root { --fg:#fff; --bg:#000; --dim:#9a9a9a; --line:#333; --red:#e5484d;
          --blue:#4da3ff; --amber:#e5a23c; --green:#3fb950; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45
         -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1500px; margin:0 auto; padding:22px 20px 44px; }
  h1 { font-size:15px; font-weight:600; letter-spacing:.14em; text-transform:uppercase;
       margin:0 0 5px; }
  .sub { font-size:12px; color:var(--dim); margin:0 0 20px; max-width:980px; }
  .label { font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:var(--dim); }
  .cols { display:grid; grid-template-columns:260px 1fr 150px; gap:18px; align-items:start; }
  button { background:var(--bg); color:var(--fg); border:1px solid var(--fg); padding:7px 13px;
           font:inherit; font-size:12px; cursor:pointer; letter-spacing:.06em; }
  button:hover:not(:disabled) { background:var(--fg); color:var(--bg); }
  button:disabled { opacity:.32; cursor:default; }
  button.wide { width:100%; }
  textarea, input, select { background:var(--bg); color:var(--fg); border:1px solid var(--line);
        font:inherit; padding:8px 9px; width:100%; }
  textarea { resize:vertical; }
  textarea:focus, input:focus { outline:none; border-color:var(--fg); }
  .scene { border:1px solid var(--line); background:#000; min-height:520px; display:flex;
           align-items:center; justify-content:center; position:relative; }
  .scene img, .scene video { max-width:100%; max-height:74vh; display:block; }
  .scene .empty { color:#555; font-size:12px; letter-spacing:.14em; text-transform:uppercase; }
  .bar { height:3px; background:#222; margin-top:9px; }
  .bar i { display:block; height:100%; background:var(--fg); width:0; transition:width .25s; }
  .stat { font-size:11px; color:var(--dim); margin-top:7px; min-height:32px;
          font-variant-numeric:tabular-nums; white-space:pre-wrap; }
  .side { display:flex; flex-direction:column; gap:9px; }
  .list { border:1px solid var(--line); max-height:300px; overflow:auto; }
  .list div { padding:6px 9px; border-bottom:1px solid #1a1a1a; cursor:pointer;
              font-size:12px; display:flex; justify-content:space-between; gap:8px; }
  .list div:hover { background:#111; }
  .list div[aria-selected="true"] { background:var(--fg); color:var(--bg); }
  .list .m { color:var(--dim); font-size:10px; letter-spacing:.1em; }
  .list div[aria-selected="true"] .m { color:var(--bg); }
  .ed { display:none; margin-top:16px; }
  .ed.on { display:block; }
  .ed textarea { height:46vh; font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  .row { display:flex; gap:9px; align-items:center; margin-top:9px; }
  /* The prompt reads as a shell line: the label sits ON the first text row rather than above the
     box, so a three-line prompt still looks like one thing being said to one thing. */
  .promptrow { display:flex; gap:10px; align-items:flex-start; }
  .plexus { font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
            color:var(--dim); padding-top:8px; white-space:nowrap; letter-spacing:.04em; }
  .plexus.dev { color:var(--amber); }
  .ok { color:var(--green); } .bad { color:var(--red); } .warn { color:var(--amber); }
  #ready { margin-top:2px; min-height:16px; }
  .knobs { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .knobs .k { display:flex; flex-direction:column; gap:3px; }
  .knobs input { padding:5px 7px; font-variant-numeric:tabular-nums; }
  .chk { display:flex; gap:7px; align-items:center; font-size:11px; color:var(--dim);
         margin-top:8px; cursor:pointer; }
  .chk input { width:auto; }
  /* Two lights. The left edge carries the colour so the state is readable at a glance without
     reading the number, and the number is there for when the glance is not enough. */
  .gauge { border:1px solid var(--line); border-left-width:4px; border-left-color:#444;
           padding:6px 9px; margin-top:7px; }
  .gauge b { display:block; font-size:9px; letter-spacing:.16em; color:var(--dim);
             font-weight:600; margin-bottom:2px; }
  .gauge span { font-size:11px; font-variant-numeric:tabular-nums; }
  .gauge i { display:block; font-style:normal; margin-top:3px; color:var(--dim);
             font-size:10px; line-height:1.4; }
  .gauge.green { border-left-color:var(--green); } .gauge.green span { color:var(--green); }
  .gauge.amber { border-left-color:var(--amber); } .gauge.amber span { color:var(--amber); }
  .gauge.red   { border-left-color:var(--red);   } .gauge.red   span { color:var(--red); }
  .gauge.dim   { border-left-color:#444; } .gauge.dim span { color:var(--dim); }
"""

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>plexus studio</title><style>__CSS__</style></head><body><div class="wrap">
<h1>Plexus Studio</h1>
<p class="sub">Describe a scene. Claude reads the real specs under <code>config/</code> and
<code>paper/plexus2.tex</code> and writes one; the server validates it with the engine's own
schema, saves it to <code>config/studio/</code>, and renders the opening frames. Nothing here
renders or validates on its own &mdash; the preview and the full run are the same command.</p>

<div class="cols">
  <div class="side">
    <div class="label">Specs</div>
    <div class="list" id="list"></div>
    <div class="knobs">
      <div class="k"><span class="label">Frames</span><input id="frames" value="500"></div>
      <div class="k"><span class="label">Particles</span><input id="particles" value="100000"></div>
      <div class="k"><span class="label">Grid</span><input id="ngrid" value="96"></div>
      <div class="k"><span class="label">Width (cm)</span><input id="width" value="10"></div>
    </div>
    <div class="stat" id="knobstat" style="min-height:16px"></div>
    <div class="gauge" id="g_ppc"><b>PARTICLES / CELL</b><span id="t_ppc">--</span></div>
    <div class="gauge" id="g_cfl"><b>CFL</b><span id="t_cfl">--</span></div>
    <div class="label" style="margin-top:2px">Device</div>
    <select id="dev"><option>cuda:1</option><option>cuda:0</option><option>cpu</option></select>
    <div class="label" style="margin-top:6px">Model</div>
    <select id="model"><option>sonnet</option><option>opus</option><option>haiku</option></select>
    <div class="label" style="margin-top:6px">Effort</div>
    <select id="effort"><option>low</option><option>medium</option><option>high</option></select>
    <label class="chk"><input type="checkbox" id="deep"> deep (read config/ + paper, slower)</label>
    <button class="wide" id="newspec" style="margin-top:6px">New scene</button>
  </div>

  <div>
    <div class="scene" id="scene"><span class="empty">no scene yet</span></div>
    <div class="bar"><i id="bar"></i></div>
    <div class="stat" id="stat">Type a scene below and press PREVIEW.</div>
    <div class="stat" id="ready">Claude: starting ...</div>

    <div style="margin-top:14px">
      <div class="promptrow">
        <span class="plexus">Plexus:</span>
        <textarea id="prompt" rows="3" placeholder="describe a scene">a 2 cm water drop falls down</textarea>
      </div>
      <div class="row">
        <button id="preview">Preview</button>
        <span class="stat" id="ptime" style="margin:0"></span>
      </div>
    </div>

    <div style="margin-top:12px">
      <div class="promptrow">
        <span class="plexus dev">dev:</span>
        <textarea id="devprompt" rows="2" placeholder="something Plexus cannot do yet -- e.g. two immiscible fluids with a real contact angle at the wall"></textarea>
      </div>
      <div class="row">
        <button id="devgo">Analyse</button>
        <span class="stat" id="devstat" style="margin:0">reads src/plexus/operators, the registry
          and the schema, then prints what would have to change. Writes nothing.</span>
      </div>
    </div>

    <div class="ed" id="ed">
      <div class="label">config/studio/<span id="edname"></span>.yaml</div>
      <textarea id="yaml" spellcheck="false"></textarea>
      <div class="row">
        <button id="save">Save</button>
        <button id="revert">Revert</button>
        <span class="stat" id="edstat" style="margin:0"></span>
      </div>
    </div>
  </div>

  <div class="side">
    <button class="wide" id="gen">Generate</button>
    <button class="wide" id="view">View</button>
    <button class="wide" id="stop" disabled>Stop</button>
    <div class="label" style="margin-top:8px">Show</div>
    <button class="wide" id="showpng">Frame</button>
    <button class="wide" id="showmp4">Movie</button>
  </div>
</div>
</div><script>__JS__</script></body></html>
"""

JS = r"""
const $ = id => document.getElementById(id);
let cur = null, poll = null, showing = "png";

const knobs = () => ({frames:+$("frames").value||500, particles:+$("particles").value||100000,
                      n_grid:+$("ngrid").value||96, width:(+$("width").value||10)/100});

const api = async (u, body) => {
  const r = await fetch(u, body ? {method:"POST", headers:{"Content-Type":"application/json"},
                                   body:JSON.stringify(body)} : {});
  return await r.json();
};
const say = (t, cls) => { const s=$("stat"); s.textContent=t; s.className="stat "+(cls||""); };

async function refresh(sel) {
  const d = await api("/api/studio/list");
  const L = $("list"); L.innerHTML = "";
  (d.specs||[]).forEach(s => {
    const e = document.createElement("div");
    e.innerHTML = `<span>${s.name}</span><span class="m">${s.has_mp4?"MP4":(s.has_png?"PNG":"")}</span>`;
    e.setAttribute("aria-selected", s.name === (sel||cur) ? "true" : "false");
    e.onclick = () => select(s.name);
    L.appendChild(e);
  });
}

async function select(name) {
  cur = name; await refresh(name);
  $("edname").textContent = name;
  const d = await api("/api/studio/spec?name=" + encodeURIComponent(name));
  $("yaml").value = d.raw || "";
  $("edstat").textContent = d.valid ? "valid" : ("INVALID: " + (d.error||""));
  $("edstat").className = "stat " + (d.valid ? "ok" : "bad");
  showing = "png"; paint(d); gauges();
}

async function gauges() {
  if (!cur) { ["ppc","cfl"].forEach(k => { $("g_"+k).className = "gauge dim";
                                           $("t_"+k).textContent = "--"; }); return; }
  const m = await api("/api/studio/metrics", {name: cur, knobs: knobs()});
  if (m.error) return;
  ["ppc","cfl"].forEach(k => {
    $("g_"+k).className = "gauge " + (m[k].color || "dim");
    $("t_"+k).innerHTML = m[k].text
      + (m[k].suggest ? `<i>${m[k].suggest.replace(/</g,"&lt;")}</i>` : "");
  });
}
["frames","particles","ngrid","width"].forEach(id =>
  document.addEventListener("DOMContentLoaded", () => {}));

function paint(d) {
  const S = $("scene");
  const png = d.png, mp4 = d.mp4;
  if (showing === "mp4" && mp4) {
    S.innerHTML = `<video controls autoplay loop `
      + `src="/media?path=${encodeURIComponent(mp4)}&t=${d.mp4_mtime||0}"></video>`;
  } else if (png) {
    S.innerHTML = `<img src="/media?path=${encodeURIComponent(png)}&t=${d.png_mtime||0}">`;
  } else {
    S.innerHTML = `<span class="empty">${cur ? "no frame yet -- press PREVIEW" : "no scene yet"}</span>`;
  }
}

async function repaint() {
  if (!cur) return;
  const d = await api("/api/studio/spec?name=" + encodeURIComponent(cur));
  paint(d);
}

// PREVIEW IS THE WHOLE CHAIN: prompt -> Claude -> schema.load -> config/studio/<name>.yaml ->
// a two-frame run -> the picture. One button, because the four steps have no useful stopping
// point between them: an unvalidated spec is not worth looking at and a validated one you cannot
// see is not worth having. An empty prompt re-previews whatever spec is selected, which is what
// you want after editing the YAML by hand.
// PREVIEW IS THE WHOLE LOOP. With a spec selected the prompt is an EDIT of it -- "make the ball
// bigger", "increase the viscosity" -- so the current YAML goes with the request and the reply
// replaces it in place. With nothing selected it writes a new one. Empty prompt just re-renders,
// which is what you want after hand-editing the YAML.
$("preview").onclick = async () => {
  const p = $("prompt").value.trim();
  $("preview").disabled = true; $("gen").disabled = true; $("bar").style.width = "0";
  if (p) {
    const editing = !!cur;
    say(editing ? `applying "${p}" to ${cur} ...` : "asking Claude for a new scene ...");
    const d = await api("/api/studio/author",
      {prompt: p, model: $("model").value, deep: $("deep").checked, effort: $("effort").value,
       name: cur || null, knobs: knobs()});
    $("ptime").textContent = `claude ${d.seconds||0}s (effort ${$("effort").value})`;
    if (d.error) { $("preview").disabled = false; $("gen").disabled = false;
      return say("FAILED: " + d.error + (d.detail ? "\n" + d.detail : ""), "bad"); }
    $("knobstat").textContent = d.report || "";
    lastPrompt = p; lastKnobs = knobKey();
    await select(d.name);
    $("prompt").value = "";
  } else if (!cur) {
    $("preview").disabled = false; $("gen").disabled = false;
    return say("write a prompt first", "warn");
  }
  say("rendering the opening frames of " + cur + " ...");
  api("/api/studio/run", {name: cur, device: $("dev").value, preview: true}).then(watch);
};
$("newspec").onclick = () => { cur = null; $("prompt").focus(); $("yaml").value = "";
  $("scene").innerHTML = '<span class="empty">no scene yet</span>';
  ["particles","ngrid","width"].forEach(id => $(id).addEventListener("change", gauges));
async function readyPoll() {
  const s = await api("/api/studio/session");
  const R = $("ready");
  if (s.state === "ready") {
    R.textContent = `Claude ready -- ${(s.chars||0).toLocaleString()} chars primed `
      + `(${s.specs} specs + operator registry + plexus2.tex) in ${s.seconds}s, effort high`;
    R.className = "stat ok";
  } else if (s.state === "priming") {
    R.textContent = "Claude: loading configs and plexus2.tex into a session ...";
    R.className = "stat warn"; setTimeout(readyPoll, 1200);
  } else {
    R.textContent = "Claude: NOT primed" + (s.error ? " -- " + s.error : "")
      + " (each prompt will send the corpus inline; slower)";
    R.className = "stat bad"; setTimeout(readyPoll, 4000);
  }
}
readyPoll();
refresh();
$("prompt").focus();
$("prompt").setSelectionRange($("prompt").value.length, $("prompt").value.length); say("new scene -- describe it and press PREVIEW"); };
// GENERATE DOES WHATEVER THE STATE ASKS FOR, so there is no wrong order of buttons:
//   * prompt changed since this spec was authored -> author it, then render. No preview step: you
//     already know what you want and a preview would only be a slower way to get there.
//   * only the knobs changed -> re-apply them server-side and render. Claude has nothing to say
//     about a particle count; calling it would cost 20 s to receive back the spec it already wrote.
//   * nothing changed -> render.
let lastPrompt = "", lastKnobs = "";
const knobKey = () => JSON.stringify(knobs());
$("gen").onclick = async () => {
  const p = $("prompt").value.trim();
  $("gen").disabled = true; $("preview").disabled = true;
  try {
    if (p && p !== lastPrompt) {
      say(cur ? `applying "${p}" to ${cur}, then rendering ...` : "asking Claude, then rendering ...");
      const d = await api("/api/studio/author",
        {prompt: p, model: $("model").value, deep: $("deep").checked, effort: $("effort").value,
         name: cur || null, knobs: knobs()});
      $("ptime").textContent = `claude ${d.seconds||0}s (effort ${$("effort").value})`;
      if (d.error) { $("gen").disabled = false; $("preview").disabled = false;
        return say("FAILED: " + d.error + (d.detail ? "\n" + d.detail : ""), "bad"); }
      lastPrompt = p; await select(d.name); $("prompt").value = "";
    } else if (cur && knobKey() !== lastKnobs) {
      say("knobs changed -- re-sizing the spec without asking Claude ...");
      const d = await api("/api/studio/apply", {name: cur, knobs: knobs()});
      if (d.error) { $("gen").disabled = false; $("preview").disabled = false;
        return say("FAILED: " + d.error, "bad"); }
      await select(cur);
    } else if (!cur) {
      $("gen").disabled = false; $("preview").disabled = false;
      return say("nothing to generate -- write a prompt first", "warn");
    }
    lastKnobs = knobKey();
    api("/api/studio/run", {name: cur, device: $("dev").value, preview: false}).then(watch);
  } catch (e) {
    $("gen").disabled = false; $("preview").disabled = false; say("FAILED: " + e, "bad");
  }
};
$("stop").onclick = () => api("/api/studio/stop", {name: cur}).then(() => say("stopped", "warn"));

function watch(j) {
  if (j.error) return say("FAILED: " + j.error, "bad");
  $("stop").disabled = false; $("gen").disabled = true; $("preview").disabled = true;
  if (poll) clearInterval(poll);
  poll = setInterval(async () => {
    const s = await api("/api/studio/progress?name=" + encodeURIComponent(cur));
    $("bar").style.width = (s.pct||0) + "%";
    say(`${s.tag||""} frame ${s.frame}/${s.total}  ${(s.pct||0).toFixed(1)}%   ${s.elapsed}s`
        + (s.tail && s.tail.length ? "\n" + s.tail[s.tail.length-1].slice(0,150) : ""));
    if (s.done) {
      clearInterval(poll); poll = null;
      $("stop").disabled = true; $("gen").disabled = false; $("preview").disabled = false;
      $("bar").style.width = s.rc === 0 ? "100%" : "0";
      say(s.rc === 0 ? `done in ${s.elapsed}s` : ("FAILED: " + (s.error||"")),
          s.rc === 0 ? "ok" : "bad");
      showing = (s.tag === "generate" && s.rc === 0) ? "mp4" : "png";
      await repaint(); await refresh(cur);
    }
  }, 900);
}

// THE DEV CHANNEL ANSWERS INTO THE TERMINAL, not into this page. What comes back is a file-by-file
// analysis with citations -- the shape of thing you read in an editor next to the code, not in a
// status line -- so the browser only reports that it arrived and how long it took.
$("devgo").onclick = async () => {
  const p = $("devprompt").value.trim();
  if (!p) return ($("devstat").textContent = "describe what Plexus cannot do yet");
  $("devgo").disabled = true;
  $("devstat").textContent = "reading src/plexus/operators, registry.py, schema.py, plexus2.tex ...";
  $("devstat").className = "stat";
  const r = await api("/api/studio/dev", {prompt: p, model: $("model").value});
  if (r.error) { $("devgo").disabled = false;
                 $("devstat").textContent = r.error; $("devstat").className = "stat bad"; return; }
  const t = setInterval(async () => {
    const s = await api("/api/studio/devstatus");
    if (s.running) { $("devstat").textContent =
      `analysing ... ${Math.round((Date.now()/1000) - s.started)}s`; return; }
    clearInterval(t); $("devgo").disabled = false;
    if (s.error) { $("devstat").textContent = "FAILED: " + s.error; $("devstat").className="stat bad"; }
    else { $("devstat").textContent =
      `analysis printed in the terminal (${s.seconds}s, ${(s.text||"").length} chars)`;
      $("devstat").className = "stat ok"; }
  }, 1500);
};

$("view").onclick = () => $("ed").classList.toggle("on");
$("showpng").onclick = () => { showing = "png"; repaint(); };
$("showmp4").onclick = () => { showing = "mp4"; repaint(); };
$("revert").onclick = () => cur && select(cur);
$("save").onclick = async () => {
  if (!cur) return;
  const d = await api("/api/studio/save", {name: cur, raw: $("yaml").value});
  $("edstat").textContent = d.saved ? "saved -- valid" : ("NOT SAVED: " + (d.error||""));
  $("edstat").className = "stat " + (d.saved ? "ok" : "bad");
};

["particles","ngrid","width"].forEach(id => $(id).addEventListener("change", gauges));
async function readyPoll() {
  const s = await api("/api/studio/session");
  const R = $("ready");
  if (s.state === "ready") {
    R.textContent = `Claude ready -- ${(s.chars||0).toLocaleString()} chars primed `
      + `(${s.specs} specs + operator registry + plexus2.tex) in ${s.seconds}s, effort high`;
    R.className = "stat ok";
  } else if (s.state === "priming") {
    R.textContent = "Claude: loading configs and plexus2.tex into a session ...";
    R.className = "stat warn"; setTimeout(readyPoll, 1200);
  } else {
    R.textContent = "Claude: NOT primed" + (s.error ? " -- " + s.error : "")
      + " (each prompt will send the corpus inline; slower)";
    R.className = "stat bad"; setTimeout(readyPoll, 4000);
  }
}
readyPoll();
refresh();
$("prompt").focus();
$("prompt").setSelectionRange($("prompt").value.length, $("prompt").value.length);
"""


def page() -> str:
    return PAGE.replace("__CSS__", CSS).replace("__JS__", JS)
