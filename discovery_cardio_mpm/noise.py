#!/usr/bin/env python
"""noise -- how big a difference has to be before it is a difference.

THE THIRD THING A METRIC NEEDS
================================================================================================
A measurement is admissible when three things are known about it: what a model that knows nothing
scores (`floors.py`), that it moves on the axis it claims and holds still on the others
(`metrics.certify`), and **how much it wobbles when nothing has changed**. This is the third.

Without it every ranking is a guess. The previous campaign settled orderings on differences of
0.003 and the word "noise floor" appears nowhere in nine thousand lines of its record, while
"replicate" appears about a hundred and fifty times -- it knew the problem existed and never
measured it.

THREE FLOORS, AND THEY ANSWER DIFFERENT QUESTIONS
------------------------------------------------------------------------------------------------
  beat to beat   compare one real beat with another real beat. The tissue is not a metronome, so
                 this is the irreducible floor for anything that compares a model to a recording:
                 **no model may be scored better than the recording agrees with itself.** Costs
                 seconds and needs no fitting, which is why it comes first.
  same seed      run the identical command twice. Zero on the processor; NOT zero on the graphics
                 card, because the scatter uses atomics and `grid_sample` has no deterministic
                 backward. Pure arithmetic wobble.
  seed to seed   the same configuration from different initialisations. The optimiser lottery, and
                 usually the largest of the three.

The unit the campaign then works in is the largest of them. A difference below it is reported
`indistinguishable` and may not be called a finding.

    python noise.py --beats            # the floor that needs no fitting
    python noise.py --fits --seeds 4   # + the two that do (slow)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
PY = sys.executable
sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import metrics as M                                                 # noqa: E402

FIT_SPEC = os.path.join(HERE, "config", "material", "material_aniso_cardio_fit.yaml")


# ---------------------------------------------------------------------------------------------
# 1. BEAT TO BEAT -- the recording against itself. No fitting.
# ---------------------------------------------------------------------------------------------
def beat_to_beat(device="cuda:0", verbose=True):
    import floors as F
    rest, real_disp, bnd, mask, sp = F.geometry(device)
    G = sp["beats"]["common_length"]
    spans = sp["beats"]["complete_spans"]
    beats = [real_disp[s[0]:s[0] + G] - real_disp[s[0]] for s in spans]

    pairs = [(i, j) for i in range(len(beats)) for j in range(i + 1, len(beats))]
    out = {"n_beats": len(beats), "n_pairs": len(pairs), "beat_frames": int(G),
           "nodes": int(mask.sum()), "metrics": {}}
    for name, m in M.live().items():
        vals = []
        for i, j in pairs:
            try:
                vals.append(m(beats[i], beats[j], mask))
            except Exception:
                vals = None
                break
        if not vals:
            continue
        v = np.array(vals, float)
        out["metrics"][name] = {"median": float(np.median(v)), "min": float(v.min()),
                                "max": float(v.max()), "sd": float(v.std()),
                                "spread": float(v.max() - v.min()),
                                "higher_is_better": m.higher_is_better, "null": m.null}
    if verbose:
        print(f"\n{'=' * 106}\n  BEAT TO BEAT -- the recording against itself, "
              f"{out['n_pairs']} pairs of {out['n_beats']} complete beats, {out['nodes']} nodes"
              f"\n{'=' * 106}")
        print(f"  {'metric':<26s} {'median':>10s} {'min':>10s} {'max':>10s} {'sd':>9s} "
              f"{'null':>9s}   what it means")
        for name, r in out["metrics"].items():
            nul = f"{r['null']:+.3f}" if r["null"] is not None else "--"
            note = ("THE CEILING: no model may score above this"
                    if name == "loopscore" else
                    "the tissue's own variation" if r["sd"] > 0 else "")
            print(f"  {name:<26s} {r['median']:>10.4f} {r['min']:>10.4f} {r['max']:>10.4f} "
                  f"{r['sd']:>9.4f} {nul:>9s}   {note}")
        print("=" * 106)
    return out


# ---------------------------------------------------------------------------------------------
# 2 and 3. THE TWO THAT NEED FITS
# ---------------------------------------------------------------------------------------------
# the fields the fit is allowed to move; identical in both invocations, because the model has to
# be built the same way before its state can be loaded back into it
MODEL_ARGS = ["--stiff_src", "siren", "--siren_fibre", "1", "--siren_omega", "5",
              "--learn", "fibre,gain,dur,stiff", "--allow_nondeterministic_ops", "1"]


def _fit(seed, n_iter, device, tag):
    """Train, then evaluate the checkpoint it left. TWO invocations, and it must be two.

    `--eval_dump` short-circuits: it runs one forward from a loaded checkpoint and exits without
    training (train.py:851). Passing it alongside `--n_iter` therefore measures the spread of the
    INITIALISATION and not of the fit -- which would have answered a question nobody asked, and
    answered it with a number small enough to look reassuring.
    """
    d = tempfile.mkdtemp(prefix=f"noise_{tag}_")
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    base = [PY, os.path.join(HERE, "train.py"), FIT_SPEC, "--seed", str(seed),
            "--device", device, "--outdir", d, *MODEL_ARGS]

    r = subprocess.run(base + ["--n_iter", str(n_iter), "--ckpt_every",
                               str(max(1, n_iter // 2))],
                       capture_output=True, text=True, env=env, timeout=14400)
    ck = sorted(glob.glob(os.path.join(d, "checkpoints", "model_*.pt")))
    if not ck:
        return None, "train: " + ((r.stderr or "").strip().splitlines() or ["no output"])[-1][:140]

    dump = os.path.join(d, "dump.npz")
    r2 = subprocess.run(base + ["--resume", ck[-1], "--eval_dump", dump],
                        capture_output=True, text=True, env=env, timeout=3600)
    if not os.path.exists(dump):
        return None, "eval: " + ((r2.stderr or "").strip().splitlines() or ["no output"])[-1][:140]
    return dump, None


def _score_dump(path, mask):
    z = np.load(path)
    sim, real = z["sim_d"].astype(np.float64), z["real_d"].astype(np.float64)
    out = {}
    for name, m in M.live().items():
        try:
            out[name] = m(sim, real, mask)
        except Exception:
            pass
    return out


def same_material(devices, tol=1e-6, verbose=True):
    """Do these devices start from the SAME particle layout? Answered, never assumed.

    Splitting the fits across two cards halves the wall clock and is only admissible if the two
    cards build the same material. They might not: the layout is already known to differ between
    the processor and the card by the width of the whole sheet at the same seed, which once turned
    into a confident, false finding about the anchor. So the fits go on one device unless this
    returns True.
    """
    import floors as F
    ref, out = None, {}
    for d in devices:
        rest = np.load(F.reference_dump(d))["rest"].astype(np.float64)
        if ref is None:
            ref, out[d] = rest, 0.0
        else:
            out[d] = float(np.abs(rest - ref).max()) if rest.shape == ref.shape else float("inf")
    ok = all(v <= tol for v in out.values())
    if verbose:
        print(f"  device check: {' '.join(f'{d} max|dx|={v:.2e}' for d, v in out.items())}"
              f"  ->  {'the same material, splitting is safe' if ok else 'DIFFERENT MATERIAL'}")
    return ok, out


def fit_spreads(seeds=4, n_iter=60, device="cuda:0", devices=None, verbose=True):
    """Same seed twice, and several different seeds, at a STATED depth."""
    import floors as F
    from concurrent.futures import ThreadPoolExecutor
    _, _, _, mask, _ = F.geometry(device)

    pool = [device]
    same = None
    if devices and len(devices) > 1:
        same, _dx = same_material(devices, verbose=verbose)
        pool = list(devices) if same else [device]

    # (label, seed) -- the repeat pair asks about ONE device, the seeds about the optimiser
    jobs = [("repeat_a", 7), ("repeat_b", 7)] + [(f"seed{s}", s) for s in range(11, 11 + seeds)]
    # the repeat pair must share a device or it stops being a determinism question; every job here
    # is on a device from the same verified-identical pool, so any assignment is admissible
    plan = [(k, s, pool[i % len(pool)]) for i, (k, s) in enumerate(jobs)]
    if len(pool) > 1:
        plan[0] = (jobs[0][0], jobs[0][1], pool[0])
        plan[1] = (jobs[1][0], jobs[1][1], pool[0])
    if verbose:
        print(f"  plan: " + "  ".join(f"{k}(seed {s}) on {d}" for k, s, d in plan), flush=True)

    runs, errs = {}, {}
    with ThreadPoolExecutor(max_workers=len(pool)) as ex:
        futs = {ex.submit(_fit, s, n_iter, d, k): k for k, s, d in plan}
        for f, k in futs.items():
            p, e = f.result()
            if p:
                runs[k] = _score_dump(p, mask)
                if verbose:
                    print(f"  [done] {k}: loopscore {runs[k].get('loopscore', float('nan')):+.4f}",
                          flush=True)
            else:
                errs[k] = e
                if verbose:
                    print(f"  [FAIL] {k}: {e}", flush=True)

    out = {"n_iter": n_iter, "device": device, "devices": pool, "devices_identical": same,
           "errors": errs, "metrics": {}}
    seed_keys = [k for k in runs if k.startswith("seed")]
    for name in M.live():
        rep = [runs[k][name] for k in ("repeat_a", "repeat_b") if k in runs and name in runs[k]]
        sds = [runs[k][name] for k in seed_keys if name in runs[k]]
        rec = {}
        if len(rep) == 2:
            rec["same_seed_difference"] = float(abs(rep[0] - rep[1]))
        if len(sds) >= 2:
            v = np.array(sds, float)
            rec.update({"seed_sd": float(v.std(ddof=1)), "seed_min": float(v.min()),
                        "seed_max": float(v.max()), "seed_n": int(v.size),
                        "seed_mean": float(v.mean())})
        if rec:
            out["metrics"][name] = rec
    if verbose:
        print(f"\n{'=' * 100}\n  THE TWO FLOORS THAT NEED FITS -- at {n_iter} iterations, "
              f"{len(seed_keys)} seeds, on {device}\n{'=' * 100}")
        print(f"  {'metric':<26s} {'same seed':>12s} {'seed sd':>10s} {'seed range':>22s}")
        for name, r in out["metrics"].items():
            rng = (f"{r['seed_min']:+.4f} .. {r['seed_max']:+.4f}" if "seed_min" in r else "--")
            print(f"  {name:<26s} {r.get('same_seed_difference', float('nan')):>12.2e} "
                  f"{r.get('seed_sd', float('nan')):>10.4f} {rng:>22s}")
        if errs:
            print(f"\n  failed: {errs}")
        print(f"\n  DEPTH IS PART OF THE ANSWER: this is {n_iter} iterations, not the 2400 the "
              f"campaign used.\n  A converged spread may be larger or smaller and must be "
              f"re-measured before the loop ranks on it.")
        print("=" * 100)
    return out


# ---------------------------------------------------------------------------------------------
# PROMOTION -- mechanical, not editorial
# ---------------------------------------------------------------------------------------------
def promotion_report(verbose=True):
    """What each metric still lacks before it may be cited. Read from the artefacts on disk."""
    p_floor = os.path.join(HERE, "_metrology", "floors.json")
    p_cert = os.path.join(HERE, "_metrology", "metrics_certify.json")
    p_beat = os.path.join(HERE, "_metrology", "noise_beats.json")
    p_fit = os.path.join(HERE, "_metrology", "noise_fits.json")
    cert = json.load(open(p_cert)) if os.path.exists(p_cert) else {}
    beat = json.load(open(p_beat)) if os.path.exists(p_beat) else {}
    fit = json.load(open(p_fit)) if os.path.exists(p_fit) else {}
    bad = {n for _, n in (cert.get("disagreements") or [])}

    rows = []
    for name, m in M.live().items():
        has_null = m.null is not None
        passed = bool(cert) and name not in bad
        has_beat = name in (beat.get("metrics") or {})
        has_fit = name in (fit.get("metrics") or {})
        missing = []
        if not has_null:
            missing.append("a measured null")
        if not passed:
            missing.append("the battery")
        if not has_beat:
            missing.append("a beat-to-beat floor")
        if not has_fit:
            missing.append("a fitted-noise floor")
        rows.append({"metric": name, "null": has_null, "battery": passed,
                     "beat_floor": has_beat, "fit_floor": has_fit,
                     "eligible": not missing, "missing": missing})
    if verbose:
        print(f"\n{'=' * 100}\n  PROMOTION -- what each metric still lacks before it may be cited"
              f"\n{'=' * 100}")
        print(f"  {'metric':<26s} {'null':>6s} {'battery':>8s} {'beats':>7s} {'fits':>6s}   still needs")
        for r in rows:
            tick = lambda b: "yes" if b else "-"
            print(f"  {r['metric']:<26s} {tick(r['null']):>6s} {tick(r['battery']):>8s} "
                  f"{tick(r['beat_floor']):>7s} {tick(r['fit_floor']):>6s}   "
                  f"{', '.join(r['missing']) if r['missing'] else 'NOTHING -- eligible'}")
        n = sum(r["eligible"] for r in rows)
        print(f"\n  {n} of {len(rows)} are eligible for certification. Nothing is promoted "
              f"automatically:\n  a tier is a judgement recorded in the class, and this only says "
              f"what the evidence supports.")
        print("=" * 100)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", action="store_true")
    ap.add_argument("--fits", action="store_true")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n_iter", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--devices", default="", help="comma-separated; the fits are split across them "
                                                 "ONLY if they build an identical material")
    ap.add_argument("--promotion", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)

    if a.beats or not (a.fits or a.promotion):
        out = beat_to_beat(a.device)
        json.dump(out, open(os.path.join(HERE, "_metrology", "noise_beats.json"), "w"),
                  indent=1, default=float)
    if a.fits:
        out = fit_spreads(a.seeds, a.n_iter, a.device,
                          [d for d in a.devices.split(",") if d.strip()] or None)
        json.dump(out, open(os.path.join(HERE, "_metrology", "noise_fits.json"), "w"),
                  indent=1, default=float)
    promotion_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
