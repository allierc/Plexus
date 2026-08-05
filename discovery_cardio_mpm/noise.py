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


def _load(name):
    """One of the metrology artefacts, or {} if it has not been measured yet."""
    p = os.path.join(HERE, "_metrology", name)
    return json.load(open(p)) if os.path.exists(p) else {}


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

    def _go(args, phase, timeout):
        """Run, streaming to a file. NOT capture_output=True.

        The first version captured both streams into memory, which on a job measured in hours means
        the only evidence of progress is invisible until the process exits. It cost an afternoon:
        two fits sat on a card another session had filled, made no progress for 105 minutes, and
        looked identical from outside to two fits working normally. `progress.txt` was no help
        either, because it is written only at checkpoints -- hence the fifth of the run below.
        """
        log = os.path.join(d, f"{phase}.log")
        with open(log, "w") as fh:
            r = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env,
                               timeout=timeout)
        tail = (open(log, errors="replace").read().strip().splitlines() or ["no output"])[-1]
        return r, tail[:140]

    _, tail = _go(base + ["--n_iter", str(n_iter),
                          # a checkpoint every fifth, so progress.txt is a progress file
                          "--ckpt_every", str(max(1, n_iter // 5))], "train", 14400)
    ck = sorted(glob.glob(os.path.join(d, "checkpoints", "model_*.pt")))
    if not ck:
        return None, f"train: {tail}"

    dump = os.path.join(d, "dump.npz")
    _, tail = _go(base + ["--resume", ck[-1], "--eval_dump", dump], "eval", 3600)
    if not os.path.exists(dump):
        return None, f"eval: {tail}"
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

    # THE REPEAT PAIR SHARES A DEVICE, and it must not share it AT THE SAME TIME.
    # "Does the same command twice give the same answer" is a question about one card, so both halves
    # go on pool[0]. But the executor runs len(pool) jobs at once in submission order, so listing
    # them adjacently put both on one card while the other sat idle -- 15.7 s/it and an 85-minute ETA
    # for the first pair alone. Interleaving by device means every concurrent slice spans different
    # cards, and the repeat pair still shares one.
    per_dev = {d: [] for d in pool}
    for i, (k, sd) in enumerate(jobs):
        per_dev[pool[0] if k.startswith("repeat") else pool[(i - 2) % len(pool)]].append((k, sd))
    plan = []
    for r in range(max(len(v) for v in per_dev.values())):
        for d in pool:
            if r < len(per_dev[d]):
                plan.append((per_dev[d][r][0], per_dev[d][r][1], d))
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
# RESOLVING POWER -- precision is only meaningful next to a range
# ---------------------------------------------------------------------------------------------
def resolving_power(verbose=True):
    """How many distinguishable steps each metric offers across its usable range.

    A small spread is not the same as a useful metric. `peak_excursion` wobbles by 0.0000 between
    real beats, which sounds superb until you notice its values are around 0.001 and it has no
    measured null, so there is nothing to divide by: **precision without a zero buys nothing, and a
    zero without precision buys nothing.** A metric needs both ends of a scale before any of its
    digits mean anything.

    Where both ends exist the question has an answer:

        levels = |what the tissue scores against itself  -  what knowing nothing scores|
                 -------------------------------------------------------------------
                                       3 x the working unit

    and the working unit is the LARGEST of the noise floors that have been measured, never the
    cheapest. Below `metrics.MIN_LEVELS` a metric may be reported and may not carry a claim.
    """
    beat = _load("noise_beats.json").get("metrics", {})
    fit = _load("noise_fits.json").get("metrics", {})
    rows = []
    for name, m in M.live().items():
        b = beat.get(name, {})
        f = fit.get(name, {})
        floors = {k: float(v) for k, v in (("beat_to_beat", b.get("sd")),
                                           ("same_seed", f.get("same_seed_difference")),
                                           ("seed_to_seed", f.get("seed_sd")))
                  if v is not None and np.isfinite(v)}
        unit = max(floors.values()) if floors else None
        which = max(floors, key=floors.get) if floors else None
        ceiling = b.get("median")
        rng = (abs(ceiling - m.null) if (ceiling is not None and m.null is not None) else None)
        lev = (rng / (3.0 * unit) if (rng is not None and unit and unit > 0) else None)
        rows.append({"metric": name, "role": m.role, "null": m.null, "ceiling": ceiling,
                     "range": rng, "unit": unit, "unit_from": which,
                     "floors_measured": sorted(floors), "levels": lev,
                     "verdict": ("no null -- its range is undeclared, so its precision cannot be "
                                 "interpreted" if m.null is None else
                                 "no floor measured" if unit is None else
                                 "objective, never an instrument" if m.role != M.EVIDENCE else
                                 "enough to rank on" if lev >= M.MIN_LEVELS else
                                 "TOO COARSE to carry a claim")})
    rows.sort(key=lambda r: (-1e9 if r["levels"] is None else -r["levels"]))
    if verbose:
        print(f"\n{'=' * 112}\n  RESOLVING POWER -- how many steps between knowing nothing and "
              f"matching the tissue (threshold {M.MIN_LEVELS:g})\n{'=' * 112}")
        print(f"  {'metric':<26s} {'null':>8s} {'ceiling':>8s} {'range':>8s} {'unit':>9s} "
              f"{'levels':>7s}   verdict")
        for r in rows:
            g = lambda v, f="{:>8.3f}": ("      --" if v is None else f.format(v))
            print(f"  {r['metric']:<26s} {g(r['null'])} {g(r['ceiling'])} {g(r['range'])} "
                  f"{g(r['unit'], '{:>9.4f}')} {g(r['levels'], '{:>7.1f}')}   {r['verdict']}")
        miss = sorted(r["metric"] for r in rows if r["null"] is None)
        if miss:
            print(f"\n  NO MEASURED NULL, so precision is uninterpretable for: {', '.join(miss)}")
        part = [r for r in rows if r["unit"] and set(r["floors_measured"]) != {
            "beat_to_beat", "same_seed", "seed_to_seed"}]
        if part:
            print(f"  PROVISIONAL: {len(part)} of these use only "
                  f"{'+'.join(sorted(set(f for r in part for f in r['floors_measured'])))}. The "
                  f"working unit is the LARGEST floor, so every `levels` here can only fall when "
                  f"the fitted floors land.")
        print("=" * 112)
    return rows


# ---------------------------------------------------------------------------------------------
# PROMOTION -- mechanical, not editorial
# ---------------------------------------------------------------------------------------------
def promotion_report(verbose=True):
    """What each metric still lacks before it may be cited. Read from the artefacts on disk."""
    cert, beat, fit = (_load("metrics_certify.json"), _load("noise_beats.json"),
                       _load("noise_fits.json"))
    bad = {n for _, n in (cert.get("disagreements") or [])}
    power = {r["metric"]: r for r in resolving_power(verbose=False)}

    rows = []
    for name, m in M.live().items():
        has_null = m.null is not None
        passed = bool(cert) and name not in bad
        has_beat = name in (beat.get("metrics") or {})
        # THREE floors, not two. A metric's row appearing in noise_fits.json only means SOME fitted
        # floor was measured -- when repeat_b died of an out-of-memory error every `same seed` cell
        # read nan and this still reported `fits: yes`, which is the same silence the caption
        # failure had. Each floor is now checked for itself.
        frow = (fit.get("metrics") or {}).get(name, {})
        has_same = np.isfinite(frow.get("same_seed_difference", float("nan")))
        has_seeds = np.isfinite(frow.get("seed_sd", float("nan")))
        has_fit = bool(has_same and has_seeds)
        lev = power.get(name, {}).get("levels")
        missing = []
        if m.role != M.EVIDENCE:
            missing.append(f"nothing -- it is the {m.role}, and no evidence it ever gathers can "
                           f"make it one")
        if not has_null:
            missing.append("a measured null")
        if not passed:
            missing.append("the battery")
        if not has_beat:
            missing.append("a beat-to-beat floor")
        if not has_same:
            missing.append("the same-seed floor")
        if not has_seeds:
            missing.append("the seed-to-seed floor")
        if lev is not None and lev < M.MIN_LEVELS and m.role == M.EVIDENCE:
            missing.append(f"resolving power ({lev:.1f} steps, {M.MIN_LEVELS:g} required)")
        rows.append({"metric": name, "null": has_null, "battery": passed,
                     "beat_floor": has_beat, "fit_floor": has_fit,
                     "same_seed_floor": bool(has_same), "seed_floor": bool(has_seeds),
                     "levels": lev,
                     "eligible": not missing, "missing": missing})
    if verbose:
        print(f"\n{'=' * 100}\n  PROMOTION -- what each metric still lacks before it may be cited"
              f"\n{'=' * 100}")
        print(f"  {'metric':<26s} {'null':>6s} {'battery':>8s} {'beats':>7s} {'fits':>6s} "
              f"{'levels':>7s}   still needs")
        for r in rows:
            tick = lambda b: "yes" if b else "-"
            print(f"  {r['metric']:<26s} {tick(r['null']):>6s} {tick(r['battery']):>8s} "
                  f"{tick(r['beat_floor']):>7s} {tick(r['fit_floor']):>6s} "
                  f"{'     --' if r['levels'] is None else format(r['levels'], '>7.1f')}   "
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
    ap.add_argument("--power", action="store_true", help="the resolving-power table alone")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)

    if a.power:
        resolving_power()
        return 0
    if a.beats or not (a.fits or a.promotion):
        out = beat_to_beat(a.device)
        json.dump(out, open(os.path.join(HERE, "_metrology", "noise_beats.json"), "w"),
                  indent=1, default=float)
    if a.fits:
        out = fit_spreads(a.seeds, a.n_iter, a.device,
                          [d for d in a.devices.split(",") if d.strip()] or None)
        json.dump(out, open(os.path.join(HERE, "_metrology", "noise_fits.json"), "w"),
                  indent=1, default=float)
    rows = resolving_power()
    json.dump(rows, open(os.path.join(HERE, "_metrology", "resolving_power.json"), "w"),
              indent=1, default=float)
    promotion_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
