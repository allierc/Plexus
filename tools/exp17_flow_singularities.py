"""exp17: DO THE FLOW'S SINKS SIT ON THE GRAPH'S OWN GEOMETRY? (Cedric, 2026-10-03: "how much do the singularities of
the flow, the flow concentration, align with the nodes of the coarse mesh of the graph?"). Local analysis.

The flow's concentrations are its CONVERGENCE zones -- where the wind of tools/exp17_wind.py points inward and the
particles pile up: C = -div(w) of the time-averaged wind w (excitatory and inhibitory apart). The neuron graph has
no mesh, but its middle and long edges are a scaffold of their own (each neuron's partners are the neurons nearest
fixed points at +-32 / +-128 um along the axes), so the scaffold has a geometry before any learning: the GRAPH WIND,
every edge deposited with weight 1, sender to receiver, on the same grid, at the same three points along the edge and
smoothed the same way (25 um), and its convergence C_graph. Also the neuron density (neurons per cell, smoothed).

For each graph: the Pearson correlation, over the cells inside the brain, of the learned flow's convergence with
  own      its own graph's C_graph            (the sinks are the scaffold's)
  other    the base graph's C_graph (for the base: the random graph's)   (a control: any scaffold)
  density  the neuron density
and the overlap of the top 5 % convergence cells (Jaccard). Writes data/flow_singularities.json and
presentation/figs/flow_singularities.png (+ png/).

    PYTHONPATH=src:tools python tools/exp17_flow_singularities.py
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
G = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "training", "zapbench")
RUNS = [("base: axes, 32 / 128 um", "zap_e15_cur", {}),
        ("axes turned 45 deg", "zap_g17_rot45", {"reach_dirs": "rotated", "reach_rotation_deg": 45.0}),
        ("random directions", "zap_g17_randdir", {"reach_dirs": "random", "graph_seed": 0}),
        ("18 nearest only", "zap_g17_knn18", {"short_k": 18, "mid_um": 0.0, "long_um": 0.0}),
        ("no highways", "zap_g17_nolong", {"long_um": 0.0}),
        ("reaches 16 / 64 um", "zap_g17_r16_64", {"mid_um": 16.0, "long_um": 64.0}),
        ("reaches 64 / 256 um", "zap_g17_r64_256", {"mid_um": 64.0, "long_um": 256.0}),
        ("random graph (the null)", "zap_g17_random", {"graph": "random", "graph_seed": 0})]
SIGMA_UM = 25.0


def conv(wx, wy, h):
    """-div of a vector field [ny, nx] (cells of h um): positive where the flow converges."""
    return -(np.gradient(wx, h, axis=1) + np.gradient(wy, h, axis=0))


def main():
    from scipy.ndimage import gaussian_filter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp17_graph_examples import build
    from exp17_ablation import _brain_view
    from plexus.paths import graphs_data_path
    pos = np.load(graphs_data_path("zebrafish/zapbench_destripe_recording.npz"))["pos_um"].astype(np.float64)
    P = _brain_view(pos)[:, :2]
    z0 = np.load(os.path.join(G, "zap_e15_cur", "results", "zap_e15_cur_wind_fields.npz"))
    x0, y0, h = (float(v) for v in z0["grid"])
    ins = z0["inside"]
    ny, nx = ins.shape
    sg = SIGMA_UM / h

    def cell(xy):
        return (np.clip(((xy[:, 1] - y0) / h).astype(int), 0, ny - 1), np.clip(((xy[:, 0] - x0) / h).astype(int), 0, nx - 1))

    def graph_conv(kw):
        E, _ = build(kw)
        wx, wy = np.zeros((ny, nx)), np.zeros((ny, nx))
        for k, (s, r) in E.items():
            if not len(s):
                continue
            d = P[r] - P[s]
            u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
            for f in (0.25, 0.5, 0.75):
                iy, ix = cell(P[s] + f * d)
                np.add.at(wx, (iy, ix), u[:, 0])
                np.add.at(wy, (iy, ix), u[:, 1])
        return conv(gaussian_filter(wx, sg), gaussian_filter(wy, sg), h)
    dens = np.zeros((ny, nx))
    np.add.at(dens, cell(P), 1.0)
    dens = gaussian_filter(dens, sg)
    Cg = {n: graph_conv(kw) for _, n, kw in RUNS}
    pr = lambda a, b: float(np.corrcoef(a[ins], b[ins])[0, 1])

    def jac(a, b, q=95):
        ta, tb = a > np.percentile(a[ins], q), b > np.percentile(b[ins], q)
        ta &= ins
        tb &= ins
        return float((ta & tb).sum() / max((ta | tb).sum(), 1))
    doc, Cf = {}, {}
    for lab, n, _ in RUNS:
        z = np.load(os.path.join(G, n, "results", f"{n}_wind_fields.npz"))
        ctrl = "zap_g17_random" if n == "zap_e15_cur" else "zap_e15_cur"
        doc[n] = {"label": lab, "control_graph": ctrl}
        for sgn in ("ex", "inh"):
            c = conv(z[sgn][0], z[sgn][1], h)
            Cf[(n, sgn)] = c
            doc[n][sgn] = {"r_own_graph": pr(c, Cg[n]), "r_other_graph": pr(c, Cg[ctrl]), "r_density": pr(c, dens),
                           "jaccard_top5_own": jac(c, Cg[n]), "jaccard_top5_other": jac(c, Cg[ctrl])}
        print(f"{n:18s} ex: own {doc[n]['ex']['r_own_graph']:+.2f} other {doc[n]['ex']['r_other_graph']:+.2f} "
              f"density {doc[n]['ex']['r_density']:+.2f} | in: own {doc[n]['inh']['r_own_graph']:+.2f} "
              f"other {doc[n]['inh']['r_other_graph']:+.2f} density {doc[n]['inh']['r_density']:+.2f}")
    json.dump(doc, open(os.path.join(EXP, "data", "flow_singularities.json"), "w"), indent=1)
    plt.style.use("dark_background")
    fig, axs = plt.subplots(len(RUNS), 3, figsize=(12, 2.0 * len(RUNS)), facecolor="black")
    for i, (lab, n, _) in enumerate(RUNS):
        for j, (img, t) in enumerate(((Cf[(n, "ex")], "flow convergence, excitatory"), (Cf[(n, "inh")], "flow convergence, inhibitory"),
                                      (Cg[n], "the graph's own convergence (edges, weight 1)"))):
            a = axs[i, j]
            v = np.percentile(np.abs(img[ins]), 98)
            a.imshow(np.where(ins, img, np.nan), origin="lower", cmap="PuOr_r", vmin=-v, vmax=v)
            a.axis("off")
            if i == 0:
                a.set_title(t, fontsize=9, loc="left")
            if j == 0:
                a.text(-0.02, 0.5, lab, transform=a.transAxes, ha="right", va="center", fontsize=9)
            if j < 2:
                d_ = doc[n]["ex" if j == 0 else "inh"]
                a.text(0.0, -0.08, f"r own graph {d_['r_own_graph']:+.2f}, density {d_['r_density']:+.2f}",
                       transform=a.transAxes, fontsize=7.5, color="0.8", va="top")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.96, bottom=0.02, hspace=0.35, wspace=0.03)
    path = os.path.join(EXP, "presentation", "figs", "flow_singularities.png")
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    print("[sing]", path)


if __name__ == "__main__":
    main()
