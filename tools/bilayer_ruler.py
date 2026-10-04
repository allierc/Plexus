#!/usr/bin/env python
"""exp04 Phase G, B2: what a protein-free bilayer patch IS -- read off its runs, one per pipette tension.

    PYTHONPATH=src python tools/bilayer_ruler.py B2a B2b B2c B2d [--label B2r] [--discard 60000]

Per run, over the hold (the ramp's end + `--discard` frames):
    tension          the pipette's reading, mN/m (mean, sd)
    area per lipid   the frame's enclosed area pi (R_patch (1 + strain))^2 over the lipids per leaflet, nm^2
    thickness        head plane to head plane (medians of the two leaflets' head heights), nm
    tails home       fraction of lipids whose tails sit on their head's side of the mid-plane (1 = no flip-flop)
    D_lateral        heads' in-plane mean-square displacement / (4 t) over the hold, um^2/s (a fluid moves)
Across runs: K_A = d tension / d (area strain), the area strain measured from the zero-tension run's area per
lipid; and the thinning d(thickness)/d(tension).
Rawicz et al. 2000 (papers/): K_A 230-265 mN/m for twelve PC bilayers; lateral diffusion of lipids ~1-10 um^2/s.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def one(label, discard):
    from plexus.paths import graphs_data_path
    d = os.path.join(graphs_data_path(), "channel", f"exp04_v{label}")
    spec = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    pred = json.load(open(os.path.join(ROOT, "config", "channel", f"exp04_v{label}.pred.json")))
    u = spec["general"]["units"]
    L = float(u["length_um"]) * 1e3; k = float(u["force_nN"]) / float(u["length_um"]); tau_s = float(u["time_s"])
    dt = float(spec["general"]["dt"]); N = int(spec["general"]["n_frames"])
    rd = next(o for o in spec["operators"] if o["op"] == "radial_drive")
    t = np.load(os.path.join(d, "trajectory.npz"))
    X = t["lipid__pos"].astype(np.float64) * L
    n = X.shape[1] // 3
    fr = np.linspace(0, N, len(X))
    m = fr >= float(rd["protocol"][2][0]) + discard
    T = t["cell__tension"][:, 0, 0] * k
    S = t["cell__strain"][:, 0, 0]
    H = X[:, :n]
    zc = np.median(X[:, :, 2], axis=1)
    thick, home = [], []
    for f in np.nonzero(m)[0]:
        hz = H[f, :, 2] - zc[f]; up = hz > 0
        thick.append(np.median(hz[up]) - np.median(hz[~up]))
        tz = X[f, n:, 2].reshape(n, 2) - zc[f]
        home.append(np.mean(np.sign(tz.mean(1)) == np.sign(hz)))
    R0 = float(pred.get("R_patch_nm", 9.0))                            # B2a-h were built with 9.0 (BILAYER)
    apl = np.pi * (R0 * (1 + S[m])) ** 2 / (n / 2)
    idx = np.nonzero(m)[0]
    lag = max(1, len(idx) // 4)
    dxy = H[idx[lag:], :, :2] - H[idx[:-lag], :, :2]
    dxy = dxy - dxy.mean(1, keepdims=True)                           # the patch's own drift out
    t_lag = (fr[idx[lag]] - fr[idx[0]]) * dt * tau_s
    D = float((dxy ** 2).sum(-1).mean() / (4 * t_lag)) * 1e-18 / 1e-12   # nm^2/s -> um^2/s
    return {"label": label, "tension_mN_m": float(T[m].mean()), "tension_sd": float(T[m].std()),
            "area_per_lipid_nm2": float(apl.mean()), "thickness_nm": float(np.mean(thick)),
            "tails_home": float(np.mean(home)), "D_um2_s": D, "eps_tail_kT": pred.get("eps_tail_kT")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+"); ap.add_argument("--label"); ap.add_argument("--discard", type=float, default=60000.0)
    a = ap.parse_args()
    rows = [one(l, a.discard) for l in a.labels]
    groups = {}
    for r in rows:
        groups.setdefault(r["eps_tail_kT"], []).append(r)
    out = {"runs": rows, "fits": {}}
    print(f"{'run':>5} {'eps':>5} {'tension':>14} {'area/lipid':>10} {'thick':>6} {'home':>6} {'D um2/s':>8}")
    for r in rows:
        print(f"{r['label']:>5} {r['eps_tail_kT']:5.2f} {r['tension_mN_m']:6.2f} +- {r['tension_sd']:4.2f} "
              f"{r['area_per_lipid_nm2']:10.3f} {r['thickness_nm']:6.2f} {r['tails_home']:6.3f} {r['D_um2_s']:8.2f}")
    for e, g in groups.items():
        g = sorted(g, key=lambda r: r["tension_mN_m"])
        tau = np.array([r["tension_mN_m"] for r in g]); A = np.array([r["area_per_lipid_nm2"] for r in g])
        th = np.array([r["thickness_nm"] for r in g])
        A0 = float(np.polyval(np.polyfit(tau, A, 1), 0.0))
        KA = float(np.polyfit((A - A0) / A0, tau, 1)[0]) if len(g) > 1 else float("nan")
        thin = float(np.polyfit(tau, th, 1)[0]) if len(g) > 1 else float("nan")
        out["fits"][str(e)] = {"A0_nm2": A0, "K_A_mN_m": KA, "thinning_nm_per_mN_m": thin,
                               "thickness_rest_nm": float(np.polyval(np.polyfit(tau, th, 1), 0.0))}
        print(f"tail depth {e} kT: rest area/lipid {A0:.3f} nm^2, K_A {KA:.0f} mN/m (Rawicz 230-265), thickness at rest "
              f"{out['fits'][str(e)]['thickness_rest_nm']:.2f} nm, thinning {thin * 1e3:.1f} pm per mN/m")
    if a.label:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plexus.paths import graphs_data_path
        d = os.path.join(graphs_data_path(), "channel", f"exp04_v{a.label}"); os.makedirs(d, exist_ok=True)
        json.dump(out, open(os.path.join(d, "bilayer_ruler.json"), "w"), indent=1)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        cols = ["tab:blue", "tab:red", "0.3", "tab:green"]
        for c, (e, g) in zip(cols, groups.items()):
            g = sorted(g, key=lambda r: r["tension_mN_m"])
            tau = [r["tension_mN_m"] for r in g]
            f_ = out["fits"][str(e)]
            ax[0].plot([(r["area_per_lipid_nm2"] - f_["A0_nm2"]) / f_["A0_nm2"] for r in g], tau, "o-", color=c,
                       label=f"tail depth {e} kT: K_A {f_['K_A_mN_m']:.0f} mN/m")
            ax[1].plot(tau, [r["thickness_nm"] for r in g], "o-", color=c, label=f"tail depth {e} kT")
            ax[2].plot(tau, [r["D_um2_s"] for r in g], "o-", color=c, label=f"tail depth {e} kT")
        xs = np.linspace(0, 0.06, 10)
        ax[0].fill_between(xs, 230 * xs, 265 * xs, color="0.85", label="Rawicz 2000: 230-265 mN/m")
        ax[0].set_xlabel("area strain", fontsize=16); ax[0].set_ylabel("tension (mN/m)", fontsize=16)
        ax[1].set_xlabel("tension (mN/m)", fontsize=16); ax[1].set_ylabel("head-to-head thickness (nm)", fontsize=16)
        ax[2].set_xlabel("tension (mN/m)", fontsize=16); ax[2].set_ylabel("lateral diffusion (um2/s)", fontsize=16)
        for k_, x in enumerate(ax):
            x.tick_params(labelsize=13); x.legend(fontsize=10, frameon=False)
            for s_ in ("top", "right"):
                x.spines[s_].set_visible(False)
            x.text(0.0, 1.04, "abc"[k_], transform=x.transAxes, fontsize=18)
        fig.tight_layout(); fig.savefig(os.path.join(d, "3d.png"), dpi=120, facecolor="white")
        print(f"wrote {d}/3d.png")


if __name__ == "__main__":
    main()
