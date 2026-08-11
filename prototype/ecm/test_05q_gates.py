"""
05q_gates -- G40 to G45 re-measured on the driver that actually works.

WHY THESE ARE RE-RUN AND NOT JUST RE-READ. G40 to G45 were written against `test_06_three_bodies`'
vertex-plaque law -- one bond per tissue VERTEX, a per-plaque rest length, turnover retargeting --
which I invented and which fails. 06c showed 05b's own law (bind to a FACE with barycentric weights,
one shared rest length) tracks the real tissue on the first try. A gate certified against a driver
that has been discarded certifies nothing, so every one of them is measured again here.

WHAT EACH GATE READS, AND THE UNIT ITS THRESHOLD IS IN.
  G40  lam_geo must not depend on the adhesion stiffness. Swept kappa_n over 8x. If the sheet's
       stretch moves with kappa_n then the number is reporting the BOND's compliance, not the
       tissue's growth, and no threshold on it means anything.
  G41  lam_geo within 5% of sqrt(A_ep(T)/A_ep(0)) -- the APICAL AREA ratio, measured on the tissue's
       own triangles, not the radius ratio. A dividing epithelium is not a sphere and the two differ.
  G42  worst triangle quality q = 4*sqrt(3)*A / sum(edge^2) > 0.2, the mesh's own failure road.
  G44  mean edge in [0.8, 1.7] x seeded. With no refinement this is EXPECTED TO FAIL at a 3.7x
       dilation -- it is the measurement that says how much refinement the real driver needs, and
       reporting it as a pass would require bm_refine, which lives in the sheet session.
  G45  all of them on the same frame axis, which is the only way the four can be read against
       each other.
Also reported, because 06c's standoff came out negative and G28 exists: the signed radial gap.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_05b_plaque as B                                              # noqa: E402
from test_06c_real_driver import Rig06c                                  # noqa: E402


def tri_area(X, F):
    a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    return 0.5 * torch.cross(b - a, c - a, dim=1).norm(dim=1)


def quality(X, F):
    """4*sqrt(3)*A / sum(edge^2): 1 for equilateral, 0 for a sliver."""
    a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    A = 0.5 * torch.cross(b - a, c - a, dim=1).norm(dim=1)
    s = (b - a).pow(2).sum(1) + (c - b).pow(2).sum(1) + (a - c).pow(2).sum(1)
    return 4.0 * math.sqrt(3.0) * A / s.clamp(min=1e-30)


def series(rig, frames, label):
    """Run and record the four gate readouts plus the signed gap, every frame."""
    e0 = float((rig.sheet.x[rig.sheet.Ed[:, 1]] - rig.sheet.x[rig.sheet.Ed[:, 0]]).norm(dim=1).mean())
    A_ep0 = float(tri_area(rig.x_epi, rig.F_epi).sum())
    out = {k: [] for k in ("t", "lam_geo", "lam_el", "area_ratio", "q_worst", "edge", "gap")}
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[{label}] DIVERGED at {t}", flush=True)
            break
        X, F = rig.sheet.x, rig.sheet.Fc
        l1, _ = rig.sheet.stretch_geo()
        A_ep = float(tri_area(rig.x_epi, rig.F_epi).sum())
        rb = (X - rig.c).norm(dim=1).mean()
        re = (rig.x_epi[:rig.nv0] - rig.c).norm(dim=1).mean()
        out["t"].append(t)
        out["lam_geo"].append(float(l1.mean()))
        out["lam_el"].append(float(rig.res["lam_el"][-1]) if rig.res["lam_el"] else float("nan"))
        out["area_ratio"].append(math.sqrt(A_ep / A_ep0))
        out["q_worst"].append(float(quality(X, F).min()))
        out["edge"].append(float((X[rig.sheet.Ed[:, 1]] - X[rig.sheet.Ed[:, 0]]).norm(dim=1).mean()) / e0)
        out["gap"].append(float(rb - re))
    return out


def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)
    name = arg("--name", str, "05q_gates")
    d = os.path.join(B.LOG, name)
    os.makedirs(d, exist_ok=True)

    T = 2.0e-3
    P = dict(subdiv=4, subdiv_epi=3, E=400.0, thickness=T, nu=0.3, kn=5.0, xi=0.0,
             l0=0.3 * T, zeta=20.0, s_target=1.0, k_drive=50.0, dev=dev)

    nom = series(Rig06c(**P), frames, "nominal")

    # G40: the same run at four adhesion stiffnesses over 8x
    sweep = {}
    for kn in (2.5, 5.0, 10.0, 20.0):
        r = series(Rig06c(**{**P, "kn": kn}), frames, f"kn {kn:g}")
        sweep[kn] = r["lam_geo"][-1] if r["lam_geo"] else float("nan")
        print(f"[05q] kappa_n {kn:5g}  lam_geo {sweep[kn]:.4f}", flush=True)

    lam = nom["lam_geo"][-1]
    ar = nom["area_ratio"][-1]
    vals = [v for v in sweep.values() if np.isfinite(v)]
    g40 = (max(vals) - min(vals)) / max(np.mean(vals), 1e-12) if vals else float("nan")
    g41 = abs(lam - ar) / ar
    g42 = min(nom["q_worst"])
    g44 = nom["edge"][-1]
    gates = {
        "G40 lam_geo independent of kappa_n (<5% over 8x)": {
            "value": g40, "threshold": 0.05, "pass": bool(g40 < 0.05), "per_kn": sweep},
        "G41 lam_geo within 5% of sqrt(A_ep(T)/A_ep(0))": {
            "value": g41, "lam_geo": lam, "area_ratio": ar,
            "threshold": 0.05, "pass": bool(g41 < 0.05)},
        "G42 worst triangle quality > 0.2": {
            "value": g42, "threshold": 0.2, "pass": bool(g42 > 0.2)},
        "G44 mean edge in [0.8, 1.7] x seeded": {
            "value": g44, "threshold": [0.8, 1.7], "pass": bool(0.8 <= g44 <= 1.7),
            "note": "no bm_refine in this rig; the number is the demand for it"},
        "G45 all four on one frame axis": {
            "value": len(nom["t"]), "threshold": frames,
            "pass": bool(len(nom["t"]) == frames)},
        "G28 signed gap stays positive (sheet outside)": {
            "value": min(nom["gap"]), "threshold": 0.0, "pass": bool(min(nom["gap"]) > 0.0)},
    }
    for k, v in gates.items():
        print(f"[05q] {'PASS' if v['pass'] else 'FAIL'}  {k}: {v['value']}", flush=True)
    json.dump({"run": name, "frames": frames, "gates": gates, "series": nom},
              open(os.path.join(d, "metrics.json"), "w"), indent=1)

    # ---- figures: white background, no box, no titles
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.4), facecolor="white")
    t = nom["t"]
    for a in ax:
        a.set_facecolor("white")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].plot(t, nom["lam_geo"], color="black", lw=1.6)
    ax[0].plot(t, nom["area_ratio"], color="green", lw=1.2, ls="--")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel(r"$\lambda^{\rm geo}$ (black), $\sqrt{A/A_0}$ (green)")
    for kn, v in sweep.items():
        ax[1].plot([kn], [v], "o", color="black")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\kappa_n$")
    ax[1].set_ylabel(r"$\lambda^{\rm geo}$ at the last frame")
    ax[2].plot(t, nom["q_worst"], color="black", lw=1.6)
    ax[2].axhline(0.2, color="red", lw=0.8, ls=":")
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("worst triangle quality")
    ax[3].plot(t, nom["edge"], color="black", lw=1.6)
    ax[3].axhspan(0.8, 1.7, color="green", alpha=0.12)
    ax[3].set_xlabel("frame"); ax[3].set_ylabel("mean edge / seeded")
    for i, a in enumerate(ax):
        a.text(-0.16, 1.04, "abcd"[i], transform=a.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "gates.png"), dpi=150, facecolor="white")
    print(f"[05q] gates.png -> {d}", flush=True)


if __name__ == "__main__":
    main()
