#!/usr/bin/env python
"""The baselines every redox model must beat, measured on the held-out LATER volumes.

    PYTHONPATH=src python tools/redox_baselines.py [--name hlo_washout] [--n-test 15]

Writes experiments/exp16_redox_graphcast/data/baselines.json and figs/baselines.png. Every number
comes from `plexus.tasks.field_recording`, the scorer the trainer's `test` uses, so a model and its
baselines are scored by the same arithmetic.

A BASELINE'S OWN KNOB IS CHOSEN ON THE TRAINING VOLUMES ONLY. The window of `window_mean` and
`linear_trend` (K volumes) and the blur of `smoothed` (sigma voxels) are each picked at each
horizon by their RMSE on train targets, then frozen and scored on the test targets. Choosing them
on the test volumes would make the baseline a fit to what it is meant to be judged on.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from plexus.tasks import field_recording as FR                          # noqa: E402

EXP = os.path.join(ROOT, "experiments", "exp16_redox_graphcast")
HS = (1, 3, 6)
KNOBS = {"window_mean": (FR.window_mean, (2, 3, 6, 12)),
         "linear_trend": (FR.linear_trend, (3, 6, 12, 24)),
         "smoothed": (FR.smoothed, (0.5, 1.0, 2.0, 4.0)),
         "space_time": (FR.space_time, tuple((K, s) for K in (2, 3, 6, 12) for s in (1.0, 2.0, 3.0)))}


def measure(rec, n_test, label):
    tr, te = FR.split(rec, n_test)
    tr_targets = tr[12:]                   # every train target has 12 volumes behind it
    out = {"resolution": label, "grid": list(rec["ratio"].shape[1:]), "dx_um": rec["dx_um"],
           "dz_um": rec["dz_um"], "test_volumes": [int(te[0]) + 1, int(te[-1]) + 1], "horizons": {}}
    for h in HS:
        row = {"minutes": float(np.median(np.diff(rec["t_s"])) * h / 60)}
        p = FR.score(FR.persistence(rec), rec, te, h)
        row["persistence"] = p
        for name, (make, grid) in KNOBS.items():
            tr_rmse = {str(k): FR.score(make(rec, k), rec, tr_targets, h)["rmse"] for k in grid}
            best = min(grid, key=lambda k: tr_rmse[str(k)])
            s = FR.score(make(rec, best), rec, te, h)
            s.update(knob=best, train_rmse_by_knob=tr_rmse, skill=1 - s["mse"] / p["mse"])
            row[name] = s
        out["horizons"][str(h)] = row
        print(f"  [{label}] h={h} ({row['minutes']:.0f} min): persistence {p['rmse']:.4f}  "
              + "  ".join(f"{n} {row[n]['rmse']:.4f} (knob {row[n]['knob']}, skill {row[n]['skill']:+.3f})"
                          for n in KNOBS), flush=True)
    D = FR.structure_function(rec, tr, hs=(1, 2, 3, 4, 6, 9, 12))
    out["structure_function_train"] = D
    out["noise_floor"] = FR.noise_floor(D)
    out["mask_drift"] = {str(h): FR.mask_drift(rec, te, h) for h in HS}
    print(f"  [{label}] noise floor sigma {out['noise_floor']['sigma']:.4f} (nugget {out['noise_floor']['nugget']:.5f}); "
          f"D(h) {', '.join(f'{k}:{v:.5f}' for k, v in D.items())}", flush=True)
    return out


def organoid_series(rec, n_test):
    """The G20 quantity: the organoid-mean ratio per volume, and its own forecast baselines."""
    tr, te = FR.split(rec, n_test)
    y = FR.organoid_mean(rec)
    ts = rec["t_s"]
    res = {"mean_ratio": y.tolist(), "t_h": (ts / 3600).tolist(), "trend_ratio_xlsx": rec["trend_ratio"].tolist(),
           "corr_with_xlsx": float(np.corrcoef(y, rec["trend_ratio"])[0, 1]), "horizons": {}}
    for h in HS:
        e_p = [y[t] - y[t - h] for t in te]
        e_l = []
        for t in te:
            o = t - h
            K = 6
            b, a = np.polyfit(ts[o - K + 1:o + 1], y[o - K + 1:o + 1], 1)
            e_l.append(a + b * ts[t] - y[t])
        res["horizons"][str(h)] = {"persistence_rmse": float(np.sqrt(np.mean(np.square(e_p)))),
                                   "linear6_rmse": float(np.sqrt(np.mean(np.square(e_l))))}
    d1 = np.diff(y[tr])
    res["step_jitter"] = float(np.std(np.diff(d1)) / np.sqrt(6))  # white noise sigma from 2nd differences
    return res


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    names = ["persistence"] + list(KNOBS)
    cols = {"persistence": "0.3", "window_mean": "tab:red", "linear_trend": "tab:blue", "smoothed": "tab:orange",
            "space_time": "tab:purple"}
    for i, key in enumerate(("coarse", "full")):
        r = res[key]
        for n in names:
            v = [r["horizons"][str(h)][n]["rmse"] for h in HS]
            ax[i].plot([r["horizons"][str(h)]["minutes"] for h in HS], v, "o-", color=cols[n], label=n)
        ax[i].axhline(r["noise_floor"]["sigma"], color="0.6", ls="--", label="noise floor")
        ax[i].set_xlabel("forecast horizon (min)")
        ax[i].set_ylabel("RMSE of the redox ratio on the organoid")
        ax[i].set_title(f"{key}: grid {' x '.join(map(str, r['grid']))}, test volumes "
                        f"{r['test_volumes'][0]}-{r['test_volumes'][1]}", fontsize=10, loc="left")
        ax[i].legend(frameon=False, fontsize=8)
    o = res["organoid"]
    ax[2].plot(o["t_h"], o["mean_ratio"], "o-", ms=3, color="tab:red", label="mean ratio on the mask (model grid)")
    ax2 = ax[2].twinx()
    ax2.plot(o["t_h"], o["trend_ratio_xlsx"], "s-", ms=3, color="tab:blue", label="xlsx: mean NADH / mean FAD")
    ax2.spines[["top"]].set_visible(False)
    ax[2].axvspan(o["t_h"][res["coarse"]["test_volumes"][0] - 1], o["t_h"][-1], color="0.92")
    ax[2].set_xlabel("time since the first volume (h)")
    ax[2].set_ylabel("mean ratio on the mask", color="tab:red")
    ax2.set_ylabel("Development_Time_Trend.xlsx", color="tab:blue")
    ax[2].set_title(f"the washout response (corr {o['corr_with_xlsx']:.3f}); grey = test", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hlo_washout")
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--factor", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    full = FR.load(a.name)
    coarse = FR.coarsen(full, a.factor)
    res = {"name": a.name, "n_test": a.n_test, "coarsen": a.factor}
    res["coarse"] = measure(coarse, a.n_test, "coarse")
    res["organoid"] = organoid_series(coarse, a.n_test)
    res["full"] = measure(full, a.n_test, "full")
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(EXP, "data", "figs"), exist_ok=True)
    json.dump(res, open(os.path.join(EXP, "data", "baselines.json"), "w"), indent=1)
    figure(res, os.path.join(EXP, "data", "figs", "baselines.png"))
    print(f"[done] {res['seconds']:.0f} s -> {EXP}/data/baselines.json")


if __name__ == "__main__":
    main()
