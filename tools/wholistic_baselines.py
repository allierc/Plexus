#!/usr/bin/env python
"""exp19's non-learned baselines on the frozen f338 recordings, per channel -> exp19 data/baselines.json.

    PYTHONPATH=src python tools/wholistic_baselines.py [--name wholistic_f338] [--hmax 10]

Every number is `plexus.tasks.field_recording.score` on the held-out TEST volumes (the last 20 %), the
call the trainer's test makes, so a model and a baseline differ in nothing but the forecast:

    persistence           x(t - h)
    window mean, W        each voxel's mean over its last W = 1..6 volumes (ZAPBench's mean baseline,
                          Lueckmann 2025); the BASELINE of the target is the best W per horizon, W chosen
                          on the VALIDATION volumes (the 10 % before the test), never on the test
    space-time (K, sigma) the window mean over K volumes blurred in (y, x) by sigma voxels (exp16's best
                          non-learned forecaster), (K, sigma) chosen on validation per horizon
    noise floor           sigma from the structure function's intercept (exp16's method), and the skill
                          ceiling it allows: 1 - sigma^2 / MSE_baseline(h)

Short range is h 1-3, long range h 5-10 (exp19 target). Written 2026-10-02 for exp19.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from plexus.paths import graphs_data_path                              # noqa: E402
from plexus.tasks import field_recording as FR                         # noqa: E402

OUT = os.path.join(ROOT, "experiments", "exp19_wholistic_graphcast", "data", "baselines.json")
WS = (1, 2, 3, 4, 5, 6)
KNOBS = [(k, s) for k in (1, 3, 6) for s in (0.5, 1.0, 2.0)]
SHORT, LONG = (1, 2, 3), (5, 6, 7, 8, 9, 10)


def channel(name, hmax):
    rec = FR.load(name)
    prov = json.load(open(graphs_data_path("redox", f"{name}_recording.provenance.json")))
    T = rec["ratio"].shape[0]
    n_test, n_val = prov["n_test"], prov["n_val"]
    test = np.arange(T - n_test, T)
    val = np.arange(T - n_test - n_val, T - n_test)
    D = FR.structure_function(rec, test, hs=(1, 2, 3, 4, 6))
    nf = FR.noise_floor(D)
    out = {"recording": name, "T": int(T), "test_volumes": [int(test[0]) + 1, int(test[-1]) + 1],
           "val_volumes": [int(val[0]) + 1, int(val[-1]) + 1], "noise": nf, "structure_function": D, "h": {}}
    wm = {W: FR.window_mean(rec, W) for W in WS}
    st = {k: FR.space_time(rec, k) for k in KNOBS}
    for h in range(1, hmax + 1):
        t0 = time.time()
        r = {"persistence_mse": FR.score(FR.persistence(rec), rec, test, h)["mse"]}
        wv = {W: FR.score(wm[W], rec, val, h)["mse"] for W in WS}
        Wb = min(wv, key=wv.get)
        r["window_val_mse"] = {str(W): v for W, v in wv.items()}
        r["best_W"] = int(Wb)
        s = FR.score(wm[Wb], rec, test, h)
        r["baseline_mse"], r["n_vox"], r["n_pairs"] = s["mse"], s["n_vox"], s["n_pairs"]
        sv = {k: FR.score(st[k], rec, val, h)["mse"] for k in KNOBS}
        kb = min(sv, key=sv.get)
        r["best_space_time"] = list(kb)
        r["space_time_mse"] = FR.score(st[kb], rec, test, h)["mse"]
        r["space_time_skill"] = 1 - r["space_time_mse"] / r["baseline_mse"]
        r["persistence_skill"] = 1 - r["persistence_mse"] / r["baseline_mse"]
        r["ceiling_skill"] = 1 - nf["sigma"] ** 2 / r["baseline_mse"]
        out["h"][str(h)] = r
        print(f"  {name} h {h:2d}: baseline (W {Wb}) {r['baseline_mse']:.5f}  persistence skill "
              f"{r['persistence_skill']:+.3f}  space-time {kb} skill {r['space_time_skill']:+.3f}  "
              f"ceiling {r['ceiling_skill']:+.3f}  ({time.time() - t0:.0f} s)", flush=True)
    for tag, hs in (("short", SHORT), ("long", LONG)):
        hs = [h for h in hs if h <= hmax]
        for k in ("persistence_skill", "space_time_skill", "ceiling_skill"):
            out[f"{k}_{tag}"] = float(np.mean([out["h"][str(h)][k] for h in hs]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="wholistic_f338")
    ap.add_argument("--hmax", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    res = {"written": time.strftime("%Y-%m-%d %H:%M"), "short_h": list(SHORT), "long_h": list(LONG)}
    for ch in ("gcamp", "mcherry"):
        res[ch] = channel(f"{a.name}_{ch}", a.hmax)
    res["ceiling_gap_long"] = res["gcamp"]["ceiling_skill_long"] - res["mcherry"]["ceiling_skill_long"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
