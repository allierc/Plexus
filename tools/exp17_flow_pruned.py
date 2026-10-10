"""exp17: THE FLOW OF THE PRUNED LAW (Cedric, 2026-10-08: "flow movies like the first deck's slides 21 and 23, for 19.25
after pruning the edges, at two resolutions, coarse and middle; then for 20.3, also after pruning"). Local, no training.

The first deck's flow (tools/exp17_wind.py, exp17_wind_views.py): every message deposited as an arrow from sender to
receiver at 1/4, 1/2 and 3/4 of the edge, gridded, smoothed over `sigma` um, excitatory (m > 0) and inhibitory (m < 0)
apart; long-lived particles drift through the two fields, red and cyan, over the RECORDED dF/F in grey; three views
(from above, oblique from 45 deg above, from the side) and the brain-mean dF/F. HERE (Cedric, 2026-10-09: "remove the
oblique view, keep top and side below one another, same size, same position as in other slides"): from above over
from the side in one column, the baseline-per-block movies' geometry (exp17_vrest_blocks.movie), the strip below.
THE DOMAIN (Cedric, 2026-10-09: "too many flows that come from outside the zebrafish brain"; "the side view biased by
neurons at the bottom"): only the neurons registered INSIDE the Z-Brain brain (data/atlas_destripe.npz `inside`,
95,248 of 100,759) place edges, corners, the grey and the mask; the 5,511 others sit around the brain after
registration (top view y 39 .. 492 um against 97 .. 428 for the inside ones, side view z down to 55 um against 77) and
widened the old mask (occupancy blurred over 25 um, > 0.2). The mask now: the neuron count per grid cell blurred over
MASK_UM = 8 um, >= MASK_MIN = 2, holes filled, the largest connected piece; particles die when they leave it. The two resolutions are its two
smoothings: COARSE 25 um (the first deck's slide 21) and MIDDLE 10 um (its slide 23).
Here, the PRUNED law (tools/exp17_prune.py, data/prune_<run>.json): only the kept edges carry messages, and the
activity is the pruned law's own free rollout (data/prune/<run>/results/<run>_joint_movie.npz, 800 frames over the 2 h);
Omega is the run's (results/<run>_omega.npz, the same frames; the SIREN is not pruned). Positions in the ATLAS frame
(head left, x' = atlas y, y' = 620 x 0.798 um - atlas x, dorsal up), as the deck's other fish.
  19.25 (neuron graph, mesh)  m_ji = W_ji tanh(z_j) Omega_i on every kept edge, levels 0-2, |W| >= the level's threshold
  20.3 (lattice grid)         the hop between corners: m_kl = w_kl c_k Omega_l g_l, c_k = (1/n_k) sum_{j -> k} a_j tanh z_j
                              (the encoded corner), Omega_l and g_l = G_recv^2 / 8 the means over the neurons decoding
                              from corner l; a corner at the mean atlas position of those neurons; the kept edges only
                              (not inert, W_grid^2 >= the level's threshold), the self edges left out (no direction)

    PYTHONPATH=src:tools python tools/exp17_flow_pruned.py zap_n19_nom --sigma 25
-> presentation/Movies/flow_views_<run>_pruned_combined[_sigma10].mp4 (+ .png)
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
MASK_UM = 8.0            # the brain mask's smoothing, um (the old 25 um mask reached ~40 um past the outermost neurons)
MASK_MIN = 2.0           # ... and its density: >= 2 neurons per grid cell after that smoothing (Cedric, 2026-10-09: "there
                         # are still flows coming from outside neurons in the top view" -- the sparse anterior fringe,
                         # ~1 neuron per 6-um cell, held the 1-neuron mask open); then the largest connected piece


def atlas_frame(A):
    """Atlas um -> the deck's fish frame: head left, x' = atlas y, y' = 620 x 0.798 - atlas x, z dorsal."""
    return np.stack([A[:, 1], (621 - 1) * 0.798 - A[:, 0], A[:, 2]], 1)


def project(P3, view):
    if view == "top":
        return P3[:, :2]
    if view == "side":
        return np.stack([P3[:, 0], P3[:, 2]], 1)
    return np.stack([P3[:, 0], np.cos(np.pi / 4) * P3[:, 1] + np.sin(np.pi / 4) * P3[:, 2]], 1)   # oblique


def edges(run, device):
    """The pruned law's messages as (positions [M, 3], sender, receiver, weight, S [F, M], O [F, M]): the message on an
    edge at frame f is weight * S[f, sender] * O[f, receiver]; plus the run's pieces."""
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    spec = T.load(run)
    out = T.out_dir(spec, None)
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    th = json.load(open(os.path.join(EXP, "data", f"prune_{run}.json")))["thresholds"]
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    mu, sd = float(rec["dff"].mean()), float(rec["dff"].std())
    mv = np.load(os.path.join(EXP, "data", "prune", run, "results", f"{run}_joint_movie.npz"))
    fr, pred = mv["frames"], np.nan_to_num(mv["pred"].astype(np.float32))
    om_ = np.load(os.path.join(out, "results", f"{run}_omega.npz"))
    assert np.array_equal(om_["frames"], fr), "Omega and the pruned rollout on different frames"
    om = torch.as_tensor(om_["omega"].astype(np.float32), device=device)
    phi = torch.tanh(torch.as_tensor((pred - mu) / sd, device=device))
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    P3, ins = atlas_frame(za["atlas_um"].astype(np.float64)), za["inside"].astype(bool)
    info = {"spec": spec, "rec": rec, "fr": fr, "pred": pred, "P3": P3, "sel": np.flatnonzero(ins)}
    if "state_diffuse.W_grid" not in fit:                                 # the neuron graph: neuron-to-neuron edges
        from exp17_ablation import neuron_graph_op
        op = neuron_graph_op(spec, "cpu")
        snd, rcv, w = [], [], []
        kept = {}
        for s in ("short", "mid", "long"):
            if f"state_diffuse.W_{s}" not in fit:
                continue
            w_ = fit[f"state_diffuse.W_{s}"].float().reshape(-1)
            k_ = w_.abs() >= th.get(s, 0.0)
            k_ &= torch.as_tensor(ins[op._E[s][0].cpu().numpy()] & ins[op._E[s][1].cpu().numpy()])   # both ends inside
            snd.append(op._E[s][0].cpu()[k_]); rcv.append(op._E[s][1].cpu()[k_]); w.append(w_[k_])
            kept[s] = (int(k_.sum()), int(len(w_)))
        info["kept"] = kept
        return P3, torch.cat(snd), torch.cat(rcv), torch.cat(w), phi, om, info
    # the lattice grid: corner-to-corner hops
    from exp17_prune import grid_edges
    E = grid_edges(spec)
    op = E["op"]
    nm = op._grid["n_mesh"]
    gs, gr = (t.numpy() for t in op._grid["g2m"])
    cs, cr = (t.numpy() for t in op._grid["m2g"])
    wg = fit["state_diffuse.W_grid"].float().numpy()
    wq = wg ** 2 if op.sign == "neuron" else wg
    thv = np.array([th.get(s, 0.0) for s in E["set"]])
    keep = ~E["inert"] & (np.abs(wq) >= thv) & (E["set"] != "self")
    info["kept"] = {s: (int((keep & (E["set"] == s)).sum()), int((E["set"] == s).sum())) for s in ("fine", "middle", "coarse")}
    a = (fit["state_diffuse.A_send"].float().numpy() if op.sign == "neuron" else np.ones(op.n_elements))
    g = fit["state_diffuse.G_recv"].float().numpy() ** 2 / 8.0
    nread = np.maximum(np.bincount(cs, minlength=nm), 1)
    ni = np.bincount(cs[ins[cr]], minlength=nm)                         # a corner placed by its INSIDE neurons only
    C3 = np.stack([np.bincount(cs[ins[cr]], weights=P3[cr[ins[cr]], d], minlength=nm) / np.maximum(ni, 1)
                   for d in range(3)], 1)
    keep &= (ni[E["ms"]] > 0) & (ni[E["mr"]] > 0)
    nfed = np.maximum(np.bincount(gr, minlength=nm), 1)
    gs_t, gr_t = torch.as_tensor(gs, device=device), torch.as_tensor(gr, device=device)
    cs_t, cr_t = torch.as_tensor(cs, device=device), torch.as_tensor(cr, device=device)
    a_t, g_t = torch.as_tensor(a, device=device), torch.as_tensor(g, device=device)
    F = phi.shape[0]
    S = torch.zeros(F, nm, device=device).index_add_(1, gr_t, a_t[gs_t] * phi[:, gs_t]) / torch.as_tensor(nfed, device=device)
    O = torch.zeros(F, nm, device=device).index_add_(1, cs_t, om[:, cr_t] * g_t[cr_t]) / torch.as_tensor(nread, device=device)
    return (C3, torch.as_tensor(E["ms"][keep]), torch.as_tensor(E["mr"][keep]), torch.as_tensor(wq[keep]), S, O, info)


def fields(run, view, sigma, pieces, cells=160, device="cuda:0"):
    """exp17_wind.fields' output (wind, speed, act, inside, grid, fr, rec, pred, P) for the pruned law in `view`."""
    import exp17_wind as W
    X3, snd, rcv, w, S, O, info = pieces
    sel = info["sel"]
    Pn = project(info["P3"][sel], view)                                 # the INSIDE neurons: the grid, the grey, the mask
    pad = 0.03 * np.ptp(Pn[:, 0])
    x0, y0 = Pn[:, 0].min() - pad, Pn[:, 1].min() - pad
    h = (np.ptp(Pn[:, 0]) + 2 * pad) / cells
    nx, ny = cells, int(np.ceil((np.ptp(Pn[:, 1]) + 2 * pad) / h))
    cell_of = lambda xy: (((xy[:, 1] - y0) / h).long().clamp(0, ny - 1) * nx + ((xy[:, 0] - x0) / h).long().clamp(0, nx - 1))  # noqa: E731
    Pt = torch.as_tensor(Pn, device=device, dtype=torch.float32)
    inside = torch.zeros(ny * nx, device=device).index_add_(0, cell_of(Pt), torch.ones(len(Pn), device=device))
    X = torch.as_tensor(project(X3, view), device=device, dtype=torch.float32)
    snd, rcv, w = snd.to(device), rcv.to(device), w.to(device).float()
    d = X[rcv] - X[snd]
    u = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cells_ = [cell_of(X[snd] + f_ * d) for f_ in (0.25, 0.5, 0.75)]

    def blur(a, s_um):
        if s_um <= 0:
            return a
        sw = s_um / h
        Rw = int(np.ceil(3 * sw))
        gw = torch.exp(-0.5 * (torch.arange(-Rw, Rw + 1, device=device, dtype=torch.float32) / sw) ** 2)
        gw = gw / gw.sum()
        return torch.nn.functional.conv2d(a[:, None], (gw[:, None] * gw[None, :])[None, None], padding=Rw)[:, 0]
    fr, pred = info["fr"], info["pred"]
    wind = {k: np.zeros((len(fr), 2, ny, nx), np.float32) for k in ("ex", "in")}
    act = np.zeros((len(fr), ny, nx), np.float32)
    cnt = inside.clamp(min=1)
    for f in range(len(fr)):
        m = w * S[f, snd] * O[f, rcv]
        ae, ai = m.clamp(min=0), (-m).clamp(min=0)
        V = torch.zeros(4, ny * nx, device=device)
        for c in cells_:
            V[0].index_add_(0, c, ae * u[:, 0]); V[1].index_add_(0, c, ae * u[:, 1])
            V[2].index_add_(0, c, ai * u[:, 0]); V[3].index_add_(0, c, ai * u[:, 1])
        Sm = blur(V.reshape(4, ny, nx), sigma).cpu().numpy()
        wind["ex"][f], wind["in"][f] = Sm[:2], Sm[2:]
        A_ = torch.zeros(ny * nx, device=device).index_add_(0, cell_of(Pt), torch.as_tensor(pred[f][sel], device=device)) / cnt
        act[f] = blur(A_.reshape(1, ny, nx), W.SIGMA_UM)[0].cpu().numpy()
    from scipy.ndimage import binary_fill_holes, label
    ins = binary_fill_holes((blur(inside.reshape(1, ny, nx), MASK_UM)[0] >= MASK_MIN).cpu().numpy())
    lab_, n_ = label(ins)
    if n_ > 1:                                                          # the brain: the largest connected piece
        sz_ = np.bincount(lab_.ravel())
        sz_[0] = 0
        ins = lab_ == sz_.argmax()
    sp = {k: np.linalg.norm(v, axis=1) for k, v in wind.items()}
    print(f"[flow] {run} {view}: {len(snd):,} edges, grid {nx} x {ny} ({h:.1f} um cells), sigma {sigma} um; 99th pct "
          f"speed ex {np.percentile(sp['ex'][:, ins], 99):.3g}, in {np.percentile(sp['in'][:, ins], 99):.3g}", flush=True)
    return dict(wind=wind, speed=sp, act=act, inside=ins, grid=(x0, y0, h), fr=fr, rec=info["rec"], pred=pred, P=Pn,
                sigma=sigma, sel=sel)


def main(run, sigma, n_part=5000, device="cuda:0"):
    import exp17_wind_views as WV
    pieces = edges(run, device)
    print(f"[flow] {run}: kept edges per level {pieces[-1]['kept']}", flush=True)
    V = {v: fields(run, v, sigma, pieces, device=device) for v in ("top", "side")}
    bg = {}
    for v, D in V.items():
        a = WV.recorded_act(D)
        bg[v] = np.clip(a / np.percentile(a[:, D["inside"]], 99), 0, 1)
    WV.render(f"{run}_pruned", V, bg, n_part, combined=True, suffix="" if sigma == 25 else f"_sigma{sigma:g}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sigma", type=float, default=25.0, help="the flow's smoothing, um: 25 coarse, 10 middle")
    ap.add_argument("--device", default="cuda:0")
    a_ = ap.parse_args()
    main(a_.run, a_.sigma, device=a_.device)
