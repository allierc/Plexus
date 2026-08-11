#!/usr/bin/env python
"""test_05p -- G30: is the sheet's tracking a property of the sheet, or of the spring?

    python test_05p_stiffness.py [--device cuda:0] [--frames 399]
        ->  log/okuda_ECM/05k_kn{...}/

THE GATE, AND WHY IT IS THE FIRST ONE. A sheet can always be made to follow a growing tissue by
stiffening whatever holds it on, and then lambda_geo matches the tissue BY CONSTRUCTION and is a
measurement of the spring rather than of the membrane. If the sheet is adhered, lambda_geo follows
the surface kinematically and must be INSENSITIVE to kappa_n. So: sweep it over 4x and require the
tracking to move by less than 5%. A sweep that moves it says the tracking is being forced, and every
number downstream of it -- the stretch, the standoff, the stress -- is a number about the adhesion.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "discovery_okuda")):
    if p not in sys.path: sys.path.insert(0, p)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import test_04_spheroid_ecm as T4, test_05j_real_surface as J, tissue as TIS

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
def arg(f, d, c=str): return c(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d

def main():
    dev = arg("--device", "cuda:0"); 
    npz = TIS.load_or_build(frames=401, device=dev, buffer_x=4, myosin=1.0, myo_tau=20.0,
                            myo_new=1.0, myo_model="two_pool", myo_k_on=0.219, myo_tau_med=20.0,
                            myo_k_ex=0.05, myo_beta_T=0.0, myo_ring=1.0, myo_new_rel=True)
    z = np.load(npz); nmesh = len(z["mesh_frames"])
    scale = T4.R_FINAL_BOX / float(z["r_apical"][-1])
    frames = arg("--frames", 2*nmesh-1, int); stride = max(1, round((frames+1)/nmesh))
    R = np.linalg.norm(z[f"m{nmesh-1}_pos"], axis=1).mean() / np.linalg.norm(z["m0_pos"], axis=1).mean()
    out = {}
    for kn in (0.5e4, 1.0e4, 2.0e4, 4.0e4):
        o = J.run("tether_real", dev, frames, npz, scale, stride, kn=kn)
        d = os.path.join(LOG, f"05p_kn{kn:g}"); os.makedirs(d, exist_ok=True)
        json.dump(o, open(os.path.join(d, "metrics.json"), "w"), indent=1)
        out[kn] = o
        print(f"[05p] kn={kn:8.0f}  lam_geo {o['series']['lam_geo'][-1]:6.3f}  "
              f"q_min {o['q_min_last']:.4f}  {'ran' if o['failed_at'] is None else 'FAILED %d' % o['failed_at']}",
              flush=True)
    lam = np.array([out[k]["series"]["lam_geo"][-1] for k in out])
    spread = float((lam.max()-lam.min())/max(lam.mean(), 1e-12))
    print(f"[gate] G30 lambda_geo spread over a 8x kn sweep: {spread:.4f} "
          f"(threshold < 0.05); the tissue's own ratio is {R:.3f}", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4), facecolor="white")
    kk = np.array(sorted(out))
    ax[0].semilogx(kk, lam, "o-", color="#2b6cb0")
    ax[0].axhline(R, color="#999", ls="--", lw=0.9)
    ax[0].set_xlabel(r"$\kappa_n$"); ax[0].set_ylabel(r"$\lambda^{\rm geo}$ at the last frame")
    ax[1].semilogx(kk, [out[k]["q_min_last"] for k in kk], "o-", color="#B03A2E")
    ax[1].set_xlabel(r"$\kappa_n$"); ax[1].set_ylabel("worst triangle quality")
    for i, a in enumerate(ax):
        a.text(0.0, 1.03, "ab"[i], transform=a.transAxes, fontweight="bold", fontsize=11)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for k in kk: fig.savefig(os.path.join(LOG, f"05p_kn{k:g}", "gate.png"), dpi=150, facecolor="white")
    json.dump(dict(spread=spread, tissue_ratio=float(R),
                   lam={str(k): float(out[k]["series"]["lam_geo"][-1]) for k in out}),
              open(os.path.join(LOG, f"05p_kn{kk[0]:g}", "G30.json"), "w"), indent=1)
    print(f"[05p] -> {LOG}/05p_*", flush=True)

if __name__ == "__main__":
    main()
