"""06 with the sheet that GROWS: `bm_refine` and `bm_secrete` on, on the same replayed tissue.

    python test_06_refine.py --device cuda:1

WHY THIS FOLDER EXISTS. In `06_spheroid_bm_ecm` the panel prints 5,120 triangles and 2,562 plaques on
every frame of a run in which the tissue quadruples, and that is not a rendering bug: 06's sheet is
`Rig06c`, which is 05b's rig -- no refinement, no secretion, plaques seeded once. The counts are fixed
BY CONSTRUCTION there, and the price of it is G44's number: the mean edge reaches 3.63x its seeded
length, which is what the demand for `bm_refine` measures.

THIS IS THE SAME 06, ONE RIG HIGHER. `Rig05l` = 05f (refinement, mass balance, tear) under the same
`RealDriver` swap, at 05f's published values -- the configuration G43 and G44 finally passed on:
rho held to 0.70% with `bm_secrete` ON, mean edge 0.929x with `bm_refine` ON, the sheet refining twice
from 5,120 to 81,920 faces. So here the triangle count and the plaque count DO climb, because one
adhesion patch is held per live sheet node and refinement makes nodes.

Everything else is 06's: the same tissue replay, the same matrix trajectory (re-drawn, never re-run,
because nothing here reaches it), the same camera, boxes and frame map.
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_05b_plaque as B                                              # noqa: E402
from test_05l_supply import Rig05l                                       # noqa: E402

NAME = "06_refine"
SRC = "06_spheroid_ecm"


def solve(d, dev, frames, keep_n=201):
    """05f's published values, unchanged, and a RAGGED store -- the whole point of this run is that the
    node and face counts change, so nothing about the sheet can be stacked."""
    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, sigma_T=7.0, zeta=20.0,
             s_target=1.0, k_drive=50.0, dev=dev)
    Q = dict(max_refine=2, edge_trigger=1.45, reseed=True, tau_bm=40.0, rho_crit=0.0)
    rig = Rig05l(**P, **Q)
    e0 = float((rig.sheet.x[rig.sheet.Ed[:, 1]] - rig.sheet.x[rig.sheet.Ed[:, 0]]).norm(dim=1).mean())
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, keep_n))).astype(int).tolist())

    store, S = {}, {k: [] for k in ("t", "rho", "rho_p05", "rho_p95", "edge", "n_face", "n_node",
                                    "n_plaque", "lam")}
    t0, i = time.time(), 0
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[{NAME}] DIVERGED at frame {t} -- the store keeps what it had", flush=True)
            break
        rho = (rig.sheet.areal_density() / rig.sheet.rho0)
        l1, _ = rig.sheet.stretch_geo()
        X = rig.sheet.x
        S["t"].append(t)
        S["rho"].append(float(rho.mean()))
        S["rho_p05"].append(float(torch.quantile(rho, 0.05)))
        S["rho_p95"].append(float(torch.quantile(rho, 0.95)))
        S["edge"].append(float((X[rig.sheet.Ed[:, 1]] - X[rig.sheet.Ed[:, 0]]).norm(dim=1).mean()) / e0)
        S["n_face"].append(int(rig.sheet.Fc.shape[0]))
        S["n_node"].append(int(X.shape[0]))
        S["n_plaque"].append(int(rig.ct_node.shape[0]))
        S["lam"].append(float(l1.mean()))
        if t in keep:
            store[f"t{i}"] = np.int32(t)
            store[f"x{i}"] = X.float().cpu().numpy()
            store[f"f{i}"] = rig.sheet.Fc.cpu().numpy().astype(np.int32)
            store[f"v{i}"] = l1.float().cpu().numpy()
            store[f"r{i}"] = rho.float().cpu().numpy()
            store[f"e{i}"] = rig.x_epi.float().cpu().numpy()
            store[f"n{i}"] = rig.ct_node.cpu().numpy().astype(np.int32)
            store[f"p{i}"] = ((rig.x_epi[rig.F_epi[rig.ct_face]] * rig.ct_w[:, :, None]).sum(1)
                              .float().cpu().numpy())
            i += 1

    np.savez_compressed(os.path.join(d, "bm_frames.npz"), n_kept=np.int32(i),
                        FE=rig.F_epi.cpu().numpy().astype(np.int32),
                        centre=rig.c.float().cpu().numpy(), scale=np.float64(rig.scale), **store)
    g43, g44 = abs(S["rho"][-1] - 1.0), S["edge"][-1]
    json.dump(dict(run=NAME, frames=len(S["t"]),
                   G43=dict(value=g43, threshold=0.10, passed=bool(g43 < 0.10),
                            rho_p05_p95_spread=S["rho_p95"][-1] - S["rho_p05"][-1]),
                   G44=dict(value=g44, threshold=[0.8, 1.7], passed=bool(0.8 <= g44 <= 1.7)),
                   faces=[S["n_face"][0], S["n_face"][-1]],
                   nodes=[S["n_node"][0], S["n_node"][-1]],
                   plaques=[S["n_plaque"][0], S["n_plaque"][-1]],
                   lam_geo=[S["lam"][0], S["lam"][-1]], series=S),
              open(os.path.join(d, "metrics.json"), "w"), indent=1)
    gate_png(S, g43, g44, os.path.join(d, "gate.png"))
    print(f"[{NAME}] {len(S['t'])} frames in {time.time()-t0:.0f}s -- faces {S['n_face'][0]} -> "
          f"{S['n_face'][-1]}, plaques {S['n_plaque'][0]} -> {S['n_plaque'][-1]}, lam_geo "
          f"{S['lam'][-1]:.4f}; G43 rho {g43*100:.2f}% "
          f"({'PASS' if g43 < 0.10 else 'FAIL'}), G44 edge {g44:.3f}x "
          f"({'PASS' if 0.8 <= g44 <= 1.7 else 'FAIL'})", flush=True)


def gate_png(S, g43, g44, path):
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.6), facecolor="white")
    for a in ax:
        a.set_facecolor("white")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_xlabel("frame")
    ax[0].plot(S["t"], S["n_face"], color="black", lw=1.6)
    ax[0].plot(S["t"], S["n_plaque"], color="#888888", lw=1.4)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("faces (black), plaques (grey)")
    ax[0].text(0.03, 0.93, f"{S['n_face'][0]} -> {S['n_face'][-1]} faces", transform=ax[0].transAxes,
               color="green", fontsize=10, va="top")
    ax[1].plot(S["t"], S["rho"], color="black", lw=1.6)
    ax[1].fill_between(S["t"], S["rho_p05"], S["rho_p95"], color="black", alpha=0.15, linewidth=0)
    ax[1].axhline(1.0, color="red", lw=0.9, ls=":")
    ax[1].set_ylabel(r"$\rho/\rho_0$, p05--p95 shaded")
    ax[1].text(0.03, 0.93, f"G43 {g43*100:.2f}%", transform=ax[1].transAxes,
               color="green" if g43 < 0.10 else "red", fontsize=10, va="top")
    ax[2].plot(S["t"], S["edge"], color="black", lw=1.6)
    for y in (0.8, 1.7):
        ax[2].axhline(y, color="red", lw=0.9, ls=":")
    ax[2].set_ylabel("mean edge / seeded")
    ax[2].text(0.03, 0.93, f"G44 {g44:.3f}x", transform=ax[2].transAxes,
               color="green" if 0.8 <= g44 <= 1.7 else "red", fontsize=10, va="top")
    for i, a in enumerate(ax):
        a.text(-0.16, 1.05, "abc"[i], transform=a.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    def arg(f, c, dflt):
        return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else dflt

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    d = os.path.join(B.LOG, arg("--name", str, NAME))
    os.makedirs(d, exist_ok=True)
    if "--reuse" in sys.argv and os.path.exists(os.path.join(d, "bm_frames.npz")):
        print(f"[{NAME}] reusing the solved sheet in {d}", flush=True)
    else:
        solve(d, dev, arg("--frames", int, 401))

    import yaml
    import run_ecm
    from test_06_panels import BMPanel
    src = os.path.join(B.LOG, SRC)
    spec = yaml.safe_load(open(os.path.join(src, "spec_run.yaml")))
    op = next(o for o in spec["operators"] if o["op"] == "mesh_contact")
    mf = np.asarray(np.load(op["tissue"].replace("/groups/saalfeld/home/allierc/Graph", "/workspace"),
                            mmap_mode="r")["mesh_frames"])
    panel = BMPanel(os.path.join(d, "bm_frames.npz"), mf, int(op.get("mesh_stride", 1)), mode="lam",
                    name=os.path.basename(d))
    run_ecm.rerender(src, dest=d, movie_frames=arg("--movie-frames", int, 200),
                     fps=arg("--fps", int, 20), bm_draw=panel, movie="--no-movie" not in sys.argv)
    print(f"[{NAME}] -> {d}", flush=True)


if __name__ == "__main__":
    main()
