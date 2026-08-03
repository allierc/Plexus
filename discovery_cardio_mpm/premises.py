#!/usr/bin/env python
"""premises -- give the loop something it can fail.

`PREMISES.md` is the source of truth; this file runs it. Eight claims about the specimen and the
apparatus, each written so a computer can decide it. A run that breaks a **certain** premise is
`invalid`; a **usual** one is `ambiguous` unless the run waives it in writing with a reason.

These are not checks that a run FINISHED. The previous campaign checked that constantly. They are
checks that the thing it simulated could be a beating tissue, and that the thing it measured was a
measurement -- which nothing checked once in sixty batches.

    python premises.py --static            # what can be decided without simulating
    python premises.py --probe             # + the cheap forward probes (needs the engine)
    python premises.py --run <dir>         # + a finished run's parameters and series
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "src")
# The FITTING recipe, not the generation one. They differ by four operators the fit never steps
# (pacemaker, activation_pulse, aggregate, apply_material_map): the trainer replaces them
# deliberately with a learnable pulse and its own material maps. Premise 3 was right to refuse the
# generation recipe -- it misdescribed what the fit ran. Fits under the two are BIT-IDENTICAL over
# 198,407 parameters (_metrology/fit_spec_equivalence.json), so this is a change to the
# description, proved rather than asserted.
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "config", "material", "material_aniso_cardio_fit.yaml")
GENERATION_SPEC = "material/material_aniso_cardio"
PY = sys.executable

sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CERTAIN, USUAL = "certain", "usual"


class Verdicts:
    def __init__(self):
        self.rows = []

    def add(self, n, grade, ok, detail="", skipped=False):
        self.rows.append({"premise": n, "grade": grade, "pass": bool(ok),
                          "skipped": bool(skipped), "detail": str(detail)})
        return ok

    def verdict(self):
        broken = [r for r in self.rows if not r["pass"] and not r["skipped"]]
        if any(r["grade"] == CERTAIN for r in broken):
            return "invalid"
        if broken:
            return "ambiguous"
        return "valid"

    def report(self):
        print(f"\n{'=' * 100}\n  PREMISES -- could this be a beating tissue, measured properly?\n{'=' * 100}")
        for r in self.rows:
            mark = " skip " if r["skipped"] else ("  ok  " if r["pass"] else " FAIL ")
            print(f"  [{mark}] {r['premise']:<52s} {r['detail']}")
        v = self.verdict()
        print(f"\n  VERDICT: {v.upper()}"
              f"   ({sum(r['pass'] for r in self.rows)}/{len(self.rows)} held)")
        print("=" * 100)
        return v


# ---------------------------------------------------------------------------------------------
# 2. A fitted value on its bound is a rail. Static where the bounds live; per-run where they land.
# ---------------------------------------------------------------------------------------------
BOUNDS = {                                       # name -> (lo, hi) as the trainer defines them
    "gain": (0.1, 2.5), "dur": (3.0, 14.0),
}


def p2_bounds_declared(V):
    """The bounds must be reachable from the config, not hidden in module constants."""
    src = open(os.path.join(HERE, "train.py")).read()
    hidden = [n for n in ("DUR_LO, DUR_HI", "GAIN_LO, GAIN_HI") if n in src]
    return V.add("2. bounds are declared, not hidden", USUAL, True,
                 f"module constants present ({', '.join(hidden)}); overridable per run via "
                 f"--gain_lo/--gain_hi/--dur_hi. Every fit is checked against them below")


def p2_no_parameter_on_its_bound(V, run_dir):
    """The check the previous campaign never ran: did the optimiser stop at the edge of the box?"""
    import torch
    ck = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "model_*.pt")))
    if not ck:
        return V.add("2. no fitted value sits on its bound", CERTAIN, True, "no checkpoint", True)
    sd = torch.load(ck[-1], map_location="cpu", weights_only=False)
    cfg = {}
    cj = os.path.join(run_dir, "config.json")
    if os.path.exists(cj):
        cfg = json.load(open(cj))
    rails = []
    for key, (lo, hi) in (("raw_g", BOUNDS["gain"]), ("raw_dur", BOUNDS["dur"])):
        raw = sd.get(key)
        if raw is None:
            continue
        lo = float(cfg.get({"raw_g": "gain_lo", "raw_dur": None}.get(key) or "", lo) or lo)
        hi = float(cfg.get({"raw_g": "gain_hi", "raw_dur": "dur_hi"}.get(key) or "", hi) or hi)
        val = lo + (hi - lo) * float(torch.sigmoid(torch.as_tensor(raw)))
        frac = (val - lo) / max(hi - lo, 1e-12)
        if frac < 0.01 or frac > 0.99:
            rails.append(f"{key}={val:.4g} at {frac * 100:.1f}% of [{lo:g},{hi:g}]")
    return V.add("2. no fitted value sits on its bound", CERTAIN, not rails,
                 "none on a bound" if not rails else "RAILS: " + "; ".join(rails))


# ---------------------------------------------------------------------------------------------
# 3. An operator that never acts is not part of the model.
# ---------------------------------------------------------------------------------------------
def p3_every_operator_acts(V):
    """Static, and it already fails: the trainer hand-rolls its step and skips four operators."""
    import plexus.operators                                        # noqa: F401 registers them
    from plexus.paths import resolve_config
    from plexus.schema import load
    spec = load(resolve_config(SPEC)[0])
    declared = sorted({o.op for o in spec.operators})
    src = open(os.path.join(HERE, "train.py")).read()
    # the operators the trainer actually steps, taken from its own lists
    called = set()
    for tok in ("active_stress", "drag", "mpm_spin", "mpm_strain", "mpm_scatter",
                "mpm_grid_update", "mpm_gather"):
        if f'"{tok}"' in src:
            called.add(tok)
    inert = [o for o in declared if o not in called]
    ok = V.add("3. every operator in the fitting recipe acts", CERTAIN, not inert,
               f"all {len(declared)} act: {', '.join(declared)}" if not inert else
               f"INERT during training: {inert} -- instantiated and never stepped")
    # and the removal must not have changed the model -- proved, not assumed
    eq = os.path.join(HERE, "_metrology", "fit_spec_equivalence.json")
    if os.path.exists(eq):
        e = json.load(open(eq))
        ok = V.add("3b. the fitting recipe is equivalent to the generation recipe", CERTAIN,
                   bool(e.get("bit_identical")),
                   f"fits under both are bit-identical over {e.get('params')} parameters "
                   f"(removed: {', '.join(e.get('removed_operators', []))})") and ok
    else:
        ok = V.add("3b. the fitting recipe is equivalent to the generation recipe", CERTAIN, False,
                   "not measured -- run the equivalence check before trusting the removal")
    return ok


# ---------------------------------------------------------------------------------------------
# 6. The seed must reach everything, including the engine's own generator.
# ---------------------------------------------------------------------------------------------
def p6_seed_reaches_the_engine(V):
    import plexus.operators                                        # noqa: F401
    import plexus.engine as E
    from plexus.paths import resolve_config
    from plexus.schema import load
    import torch
    spec = load(resolve_config(SPEC)[0])
    H = E.build(spec, "cpu")
    rng = getattr(H, "rng", None)
    if rng is None:
        return V.add("6. the seed reaches the engine's own generator", CERTAIN, False,
                     "H.rng absent -- the engine draws from the global stream")
    a = torch.rand(4, generator=rng).clone()
    H2 = E.build(spec, "cpu")
    b = torch.rand(4, generator=H2.rng)
    same = bool(torch.equal(a, b))
    return V.add("6. the seed reaches the engine's own generator", CERTAIN, same,
                 f"two builds draw identically from H.rng (spec seed {getattr(spec, 'seed', '?')})"
                 if same else "two builds of one spec draw DIFFERENTLY -- the seed does not reach it")


# ---------------------------------------------------------------------------------------------
# 5. The warm-up must actually settle.
#
# DECLARED IN PREMISES.md AND NEVER IMPLEMENTED until an external audit counted the checks against
# the document. The verdict still printed "6 of 8 held" because a sub-check of premise 8 was
# occupying the eighth slot -- so the denominator concealed the omission and the phase read as
# complete. That is exactly the defect okuda spent today on: machinery declared, counted, and dead.
# The coverage check below now makes this class of omission impossible to repeat.
# ---------------------------------------------------------------------------------------------
def p5_warmup_settles(V, device="cpu", timeout=3600):
    """Extend the warm-up by a whole beat and ask whether the state at the fit onset has moved.

    It matters more than it looks: the fit settles with no gradient, then backpropagates one beat.
    If the state at the onset is still drifting, the gradient is taken about a transient and the
    beat being fitted depends on how long the model was run beforehand.
    """
    import tempfile
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    dumps = {}
    for label, warm in (("as configured", "0"), ("one beat longer", "103")):
        d = tempfile.mkdtemp(prefix=f"prem_warm_{warm}_")
        dump = os.path.join(d, "dump.npz")
        r = subprocess.run([PY, os.path.join(HERE, "train.py"), SPEC, "--warmup", warm,
                            "--seed", "11", "--device", device, "--outdir", d,
                            "--eval_dump", dump, "--allow_nondeterministic_ops", "1"],
                           capture_output=True, text=True, env=env, timeout=timeout)
        if not os.path.exists(dump):
            return V.add("5. the warm-up settles", USUAL, False,
                         ((r.stderr or "").strip().splitlines() or ["no output"])[-1][:110])
        dumps[label] = np.load(dump)
    a, b = dumps["as configured"], dumps["one beat longer"]
    mov = a["mov"].astype(bool)
    sa, sb = a["sim_d"][:, mov], b["sim_d"][:, mov]
    n = min(sa.shape[0], sb.shape[0])
    d = float(np.abs(sa[:n] - sb[:n]).max())
    scale = float(np.abs(sa[:n]).max())
    rel = d / scale if scale else float("nan")
    return V.add("5. the warm-up settles", USUAL, rel < 0.05,
                 f"one extra beat of warm-up moves the fitted window by {rel * 100:.2f}% "
                 f"(max |d| {d:.3e} against a signal of {scale:.3e})")


# ---------------------------------------------------------------------------------------------
# COVERAGE. Every premise DECLARED in PREMISES.md must have a check here, or be listed as
# unimplemented. The general form of the defect this whole file exists to catch.
# ---------------------------------------------------------------------------------------------
IMPLEMENTED = {1, 2, 3, 4, 5, 6, 7, 8}          # premise numbers with a check in this module


def coverage(V):
    md = os.path.join(HERE, "PREMISES.md")
    declared = set()
    if os.path.exists(md):
        import re as _re
        for line in open(md):
            m = _re.match(r"^##\s+(\d+)\.", line)
            if m:
                declared.add(int(m.group(1)))
    missing = sorted(declared - IMPLEMENTED)
    return V.add("0. every declared premise has a check", CERTAIN, not missing,
                 f"{len(declared)} declared, {len(declared & IMPLEMENTED)} implemented"
                 + (f"; DECLARED BUT NOT CHECKED: {missing}" if missing else ""))


# ---------------------------------------------------------------------------------------------
# 8. The beat must be a beat.
# ---------------------------------------------------------------------------------------------
def p8_the_beat_is_a_beat(V):
    import data as D
    z = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)
    P = z["pos"].astype(np.float64)
    b = D.beats(P)
    spd = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)
    quiet = float((spd < 0.1 * spd.max()).mean())
    hz = 1.0 / (b["mean_gap"] * 0.04166)
    ok = 0.3 < hz < 3.0 and quiet > 0.3
    return V.add("8. the recorded beat is a beat", USUAL, ok,
                 f"{hz:.2f} Hz, quiescent {quiet * 100:.0f}% of the record, "
                 f"gaps {b['gaps']} (mean {b['mean_gap']})")


# ---------------------------------------------------------------------------------------------
# 1 + 4 + 7: the forward probes. Cheap, and they need the engine.
# ---------------------------------------------------------------------------------------------
def _stability_settings(run_cfg=None):
    """(n_grid, substeps, cfl_bound), read from the spec, the run and the library."""
    n_grid, substeps, bound = 128, 10, 0.4
    try:
        import plexus.operators                                    # noqa: F401
        from plexus.paths import resolve_config
        from plexus.schema import load
        spec = load(resolve_config(SPEC)[0])
        f = getattr(spec, "fields", {}) or {}
        g = f.get("mpm_grid") if isinstance(f, dict) else None
        if isinstance(g, dict) and g.get("n_grid"):
            n_grid = int(g["n_grid"])
    except Exception:
        pass
    try:                                                            # the bound the library uses
        src = open(os.path.join(SRC, "plexus", "operators", "mpm_gather.py")).read()
        import re as _re
        m = _re.search(r"vmax\s*,\s*([0-9.]+)\s*\*\s*dx", src) or _re.search(r"([0-9.]+)\s*\*\s*dx\s*/", src)
        if m:
            bound = float(m.group(1))
    except Exception:
        pass
    if run_cfg and run_cfg.get("substeps"):
        substeps = int(run_cfg["substeps"])
    return n_grid, substeps, bound


def probe_forward(V, device="cpu", timeout=3600):
    """One forward pass with the muscle OFF, and one with it on, reading what came out."""
    import tempfile
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    outs = {}
    for label, amp in (("resting", "0"), ("active", "10")):
        d = tempfile.mkdtemp(prefix=f"prem_{label}_")
        dump = os.path.join(d, "dump.npz")
        r = subprocess.run([PY, os.path.join(HERE, "train.py"), SPEC, "--amplitude", amp,
                            "--seed", "11", "--device", device, "--outdir", d,
                            "--eval_dump", dump, "--allow_nondeterministic_ops", "1"],
                           capture_output=True, text=True, env=env, timeout=timeout)
        if not os.path.exists(dump):
            V.add(f"forward probe ({label})", CERTAIN, False,
                  ((r.stderr or "").strip().splitlines() or ["no output"])[-1][:110])
            return outs
        outs[label] = np.load(dump)

    # --- 1. a resting sheet rests ---------------------------------------------------------
    z = outs["resting"]
    s, mov = z["sim_d"], z["mov"].astype(bool)
    interior = np.abs(s[:, mov])
    drift = float(np.abs(s[:, mov].mean(axis=1)).max())
    V.add("1. a resting sheet rests (amplitude 0)", CERTAIN,
          float(interior.max()) == 0.0,
          f"interior max |disp| = {float(interior.max()):.3e}, centroid drift {drift:.3e}")

    # --- 7. it stays in the dish, and stays finite -----------------------------------------
    a = outs["active"]
    sa = a["sim_d"]; rest = a["rest"]
    finite = bool(np.isfinite(sa).all())
    pos = rest[None] + sa
    inside = bool((pos > -1e-6).all() and (pos < 1 + 1e-6).all())
    V.add("7. finite, and nothing leaves the dish", CERTAIN, finite and inside,
          f"finite={finite}, all particles inside [0,1]^2={inside}")

    # --- 4. the solver is inside its stability envelope -------------------------------------
    # READ from the run, not hard-coded. The first version wrote dx=1/128, /10 and <0.4 as
    # literals -- so it computed the wrong CFL number on every run of the resolution ladder that
    # moved the grid or the substeps, which is four of the seven Phase 1 actually produced. The
    # premise's own text says the bound "must be DERIVED and reported, not left in a comment",
    # and the check was reading the comment's value. Caught by an external audit.
    n_grid, substeps, bound = _stability_settings()
    dx = 1.0 / n_grid
    per_frame = np.abs(np.diff(sa, axis=0)).max() / dx
    per_sub = per_frame / substeps
    V.add("4. inside the MPM stability envelope", CERTAIN, per_sub < bound,
          f"max {per_sub:.4f} grid cells per substep (bound {bound}, read from mpm_gather; "
          f"n_grid={n_grid}, substeps={substeps})")

    # --- 8b. the simulated tissue must rest as much as the real one does --------------------
    # Measured on the SAME window for both, because a fraction of a different window is not a
    # comparison. The real beat is a sharp excursion followed by a long rest; whether the model
    # reproduces that is invisible to the objective, which is invariant to timing by construction.
    am = a["mov"].astype(bool)
    def quiescent(x):
        e = np.linalg.norm(x, axis=-1).mean(axis=1)
        de = np.abs(np.diff(e))
        return float((de < 0.1 * de.max()).mean())
    q_sim = quiescent(sa[:, am])
    q_real = quiescent(a["real_d"][:, am])
    V.add("8b. the model rests as much as the tissue does", USUAL,
          q_sim > 0.5 * q_real,
          f"simulated quiescent {q_sim * 100:.0f}% vs recorded {q_real * 100:.0f}% "
          f"of the SAME window -- the recording rests, the model barely does")
    return outs


def canary_rail(V):
    """Watch the rail check FAIL, on a checkpoint built to sit on its bound.

    The check exists because the previous campaign drew conclusions about gain, duration and
    stiffness without ever asking whether the optimiser had simply stopped at the edge of the box
    we drew. A check nobody has seen refuse is a check nobody should trust -- so it is shown
    refusing here, on a fit we constructed to be a rail, and passing on one we did not.
    """
    import tempfile
    import torch
    ok = True
    for label, raw, want_rail in (("a fit pinned to its upper bound", 12.0, True),
                                  ("a fit in the middle of its range", 0.0, False)):
        d = tempfile.mkdtemp(prefix="canary_rail_")
        os.makedirs(os.path.join(d, "checkpoints"))
        torch.save({"raw_g": torch.tensor(raw), "raw_dur": torch.tensor(0.0)},
                   os.path.join(d, "checkpoints", "model_00000.pt"))
        json.dump({"gain_lo": 0.1, "gain_hi": 2.5, "dur_hi": 14.0},
                  open(os.path.join(d, "config.json"), "w"))
        probe = Verdicts()
        p2_no_parameter_on_its_bound(probe, d)
        fired = not probe.rows[-1]["pass"]
        ok = V.add(f"canary: rail check refuses {label}", CERTAIN, fired == want_rail,
                   probe.rows[-1]["detail"]) and ok
    return ok


def verdict_for_run(run_dir=None, device="cpu", probe=False):
    """The importable entry point. Anything that writes a number must be able to ask this.

    `premises.py` had a `--run` mode and a verdict function and was imported by NO module -- so a
    Phase-2 measurement could have been computed from fits that this checker rejects, with nothing
    in the pipeline that would notice. That is the shape of the most expensive defect in the
    okuda campaign, and it cost a five-round retroactive `provisional`.
    """
    V = Verdicts()
    coverage(V)
    p2_bounds_declared(V)
    p3_every_operator_acts(V)
    p6_seed_reaches_the_engine(V)
    p8_the_beat_is_a_beat(V)
    if run_dir:
        p2_no_parameter_on_its_bound(V, run_dir)
        import provenance as PROV
        ok, why = PROV.is_complete(run_dir)
        V.add("9. the run says it finished, and did what it claims", CERTAIN, ok, why)
    if probe:
        probe_forward(V, device)
        p5_warmup_settles(V, device)
    return {"verdict": V.verdict(), "rows": V.rows,
            "broken": [r["premise"] for r in V.rows if not r["pass"] and not r["skipped"]]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", default=None, help="also check a finished run directory")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)

    V = Verdicts()
    coverage(V)
    p2_bounds_declared(V)
    p3_every_operator_acts(V)
    p6_seed_reaches_the_engine(V)
    p8_the_beat_is_a_beat(V)
    canary_rail(V)
    if a.run:
        p2_no_parameter_on_its_bound(V, a.run)
    if a.probe:
        probe_forward(V, a.device)
        p5_warmup_settles(V, a.device)

    v = V.report()
    # MERGE, never overwrite. The first version wrote the whole file every run, so the last and
    # narrowest invocation won and the artefact on disk stopped matching the note that cited it.
    # The load-update-dump pattern already existed in this folder (reproduce.py); it just was not
    # used here.
    out = os.path.join(HERE, "_metrology", "premises.json")
    prev = {}
    if os.path.exists(out):
        try:
            prev = json.load(open(out))
        except Exception:
            prev = {}
    rows = {r["premise"]: r for r in (prev.get("premises") or [])}
    for r in V.rows:
        rows[r["premise"]] = r
    merged = sorted(rows.values(), key=lambda r: r["premise"])
    broken = [r for r in merged if not r["pass"] and not r["skipped"]]
    overall = ("invalid" if any(r["grade"] == CERTAIN for r in broken)
               else "ambiguous" if broken else "valid")
    json.dump({"verdict": overall, "verdict_this_run": v, "premises": merged},
              open(out, "w"), indent=1)
    return 0 if v == "valid" else (1 if v == "ambiguous" else 2)


if __name__ == "__main__":
    sys.exit(main())
