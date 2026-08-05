#!/usr/bin/env python
"""add_field -- does giving the model a field it never had actually buy anything?

THE QUESTION ABLATION CANNOT ANSWER
================================================================================================
`ablate.py` removes a field from a trained model and measures the loss. That is the right test for
a field the model HAS. It is the wrong test for one it does not: the other fields were fitted around
the missing one's absence, so there is nothing to remove and nothing to conclude. The only way to
find out what prestress is worth is to train with it and train without it, and compare.

AND THE COMPARISON NEEDS A FLOOR, WHICH IS WHY THIS COMES AFTER THE NOISE WORK
------------------------------------------------------------------------------------------------
Two fits differ for two reasons: the field, and the optimiser lottery. The seed-to-seed spread is
measured -- coordination 0.0384, orientation_error 0.0116, chirality_match 0.0171 -- so a difference
smaller than three of those is the lottery and may not be attributed to the field. This is exactly
the comparison the previous campaign made hundreds of times without a floor to make it against.

So: N seeds with the field, the SAME N seeds without, at the same depth, and the verdict is
mechanical.

    python add_field.py --field prestress --seeds 11 12 13 14
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import metrics as M                                                 # noqa: E402
import noise as N                                                   # noqa: E402

REPORT = ["loopscore", "orientation_error", "chirality_match", "peak_excursion",
          "coordination", "path_length"]

# the extra arguments each addable field needs, on top of noise.MODEL_ARGS
FIELD_ARGS = {
    "prestress": ["--residual_stress", "1", "--residual_amp", "0.2",
                  "--residual_omega", "5", "--learn", "fibre,gain,dur,stiff,residual"],
}


def existing_baseline(mask):
    """Score the without-field fits already on disk from the noise run, if they survive."""
    out = {}
    for d in sorted(glob.glob("/tmp/noise_seed*_*")):
        dump = os.path.join(d, "dump.npz")
        if not os.path.exists(dump):
            continue
        seed = int(os.path.basename(d).split("_")[1].replace("seed", ""))
        out[seed] = N._score_dump(dump, mask)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="prestress", choices=sorted(FIELD_ARGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14])
    ap.add_argument("--n_iter", type=int, default=300)
    ap.add_argument("--devices", default="cuda:0,cuda:1")
    a = ap.parse_args(argv)
    devs = [d for d in a.devices.split(",") if d.strip()]

    import floors as F
    _, _, _, mask, _ = F.geometry(devs[0])

    base = existing_baseline(mask)
    have = [s for s in a.seeds if s in base]
    print(f"\n  WITHOUT {a.field}: reusing {len(have)} fits already on disk, seeds {have}")
    if len(have) < 2:
        raise SystemExit("need at least two without-field fits; re-run noise.py --fits first")

    # with the field: same seeds, same depth, same everything else
    saved = list(N.MODEL_ARGS)
    extra = FIELD_ARGS[a.field]
    # --learn is replaced, not appended, so strip the original
    merged = [x for x in saved]
    if "--learn" in extra:
        i = merged.index("--learn")
        del merged[i:i + 2]
    N.MODEL_ARGS = merged + extra
    print(f"  WITH {a.field}: {' '.join(extra)}")
    print(f"  {len(have)} fits at {a.n_iter} iterations across {devs}\n", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    plan = [(s, devs[i % len(devs)]) for i, s in enumerate(have)]
    with_field, errs = {}, {}
    with ThreadPoolExecutor(max_workers=len(devs)) as ex:
        futs = {ex.submit(N._fit, s, a.n_iter, d, f"{a.field}{s}"): s for s, d in plan}
        for f, s in futs.items():
            dump, err = f.result()
            if dump:
                with_field[s] = N._score_dump(dump, mask)
                print(f"  [done] seed {s}: loopscore "
                      f"{with_field[s].get('loopscore', float('nan')):+.4f}", flush=True)
            else:
                errs[s] = err
                print(f"  [FAIL] seed {s}: {err}", flush=True)
    N.MODEL_ARGS = saved

    if len(with_field) < 2:
        raise SystemExit(f"not enough with-field fits succeeded: {errs}")

    fit = N._load("noise_fits.json").get("metrics", {})
    beat = N._load("noise_beats.json").get("metrics", {})
    print(f"\n{'=' * 112}\n  DOES ADDING `{a.field}` BUY ANYTHING? "
          f"{len(with_field)} seeds with, {len(have)} without, {a.n_iter} iterations"
          f"\n{'=' * 112}")
    print(f"  {'metric':<22s} {'without':>12s} {'with':>12s} {'change':>12s} {'3x noise':>11s}"
          f"   verdict")
    rows = {}
    for n in REPORT:
        w0 = np.array([base[s][n] for s in with_field if n in base.get(s, {})], float)
        w1 = np.array([with_field[s][n] for s in with_field if n in with_field[s]], float)
        if w0.size < 2 or w1.size < 2:
            continue
        f_, b_ = fit.get(n, {}), beat.get(n, {})
        cand = [v for v in (b_.get("sd"), f_.get("same_seed_difference"), f_.get("seed_sd"))
                if v is not None and np.isfinite(v)]
        unit = max(cand) if cand else float("nan")
        d = w1.mean() - w0.mean()
        better = (d < 0) if not M.REGISTRY[n].higher_is_better else (d > 0)
        real = np.isfinite(unit) and abs(d) > 3 * unit
        verdict = ("indistinguishable" if not real else
                   ("BETTER with it" if better else "WORSE with it"))
        if M.REGISTRY[n].role != M.EVIDENCE:
            verdict += "  (the objective -- reported, not evidence)"
        rows[n] = {"without": w0.mean(), "with": w1.mean(), "change": d, "unit": unit,
                   "distinguishable": bool(real), "verdict": verdict}
        print(f"  {n:<22s} {w0.mean():>+12.4f} {w1.mean():>+12.4f} {d:>+12.4f} "
              f"{3 * unit:>11.4f}   {verdict}")
    print(f"\n  A change smaller than three times the noise is the optimiser lottery, not the "
          f"field.\n  Depth is {a.n_iter} iterations, not the 2400 the campaign used, and this "
          f"verdict is only about that depth.")
    print("=" * 112)
    json.dump({"field": a.field, "n_iter": a.n_iter, "seeds": sorted(with_field), "rows": rows,
               "errors": errs},
              open(os.path.join(HERE, "_metrology", f"add_{a.field}.json"), "w"),
              indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
