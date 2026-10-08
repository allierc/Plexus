"""The HD twin against its reference, by TRANSPLANT (Cedric, 2026-10-08: "do we have a reference ARTR training code that
we can use as reference to test ours?"). Local, no training.

The reference's trained network (connectome-gnn-cx run zebrafish_hd_si_ipn_917_v1_selfmotion_rotation, its last
checkpoint; its own test: 2.08 +- 1.15 deg heading RMSE over 512 trials) is copied into the Plexus model
config/neural/zf_hd_917.yaml, parameter for parameter, and both are rolled out on the same held-out trials of the
Plexus corpus t8_hd_heading. If the twin is the same law, the two decoded headings agree to float precision and both
score ~2 deg against the corpus's teacher. (The first run, 2026-10-08, found trainer.rollout feeding the drive two
frames late: 2.09 deg for the twin against 1.50 deg for the reference; fixed in trainer.rollout the same day, and the
two now agree to float precision -- the tool stops if they differ by more than 1e-4.)

THE REFERENCE LAW (connectome-gnn-cx models/zebrafish_hd_task_rnn.py, `forward`), written out here, not imported:
    h <- h + (dt / tau) (-h + W_rec sigma(h) + W_in u_t + b),   y_t = W_out sigma(h)[ring] + b_out,   h_0 = 0
    W_rec = |S| sign(W_con), diagonal zeroed;  W_in[:, 0] = the ARTR gate, -softplus(v_artr_l) on ARTR_L and
    +softplus(v_artr_r) on ARTR_R, 0 elsewhere (rotation mode); tau = 0.1 s, dt = 0.01 s.
THE PLEXUS LAW (the spec's six operators):
    dx/dt = -a x + g sum_j w_ij sigma(x_j) + gain (sum w_omega omega + sum w_cue cue) + bias,   a = g = gain = 10 / s
so w = |S| (Dale signs from the sender's type), w_omega = the gate scalar, w_cue = W_in[:, 1:3], bias = b / tau
(10 b), readout w = W_out, heading bias = b_out.

CELL ORDER. The reference orders its rows ring-first, then by soma X; the twin by (group, side, ring angle). The
checkpoint stores its W_con, so the map is found from the wiring itself: weights back to integer contact areas, then
colour refinement on the weighted graph (each cell coloured by its own colour and the multiset of (area, colour) of
its in- and out-edges, 8 rounds). It must come out a bijection with W_con equal to the twin's wiring entry for entry,
or the tool stops.

    PYTHONPATH=src python tools/hd_transplant_check.py [--n 512] [--device cuda:0]
-> printed table; <zf_hd_heading run>/results/hd_transplant.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src")]
REF = ("/groups/saalfeld/home/allierc/GraphData/log/zebrafish/archive_2/"
       "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation")
CKPT = os.path.join(REF, "models", "best_model_with_0_graphs_9.pt")
TAU, N_RING, WARMUP = 0.1, 700, 10       # the reference's tau (s), its readout's ring rows, its RMSE warm-up (frames)


def _edges(name):
    from plexus.paths import graphs_data_path
    d = np.load(graphs_data_path("neural", f"zf_hd_917_{name}.npz"))
    return d["edge_index"][0], d["edge_index"][1], d["weights"].astype(np.float64)


def cell_map(W_con):
    """perm[r] = the twin's row of the reference's row r, from the wiring alone (see the docstring)."""
    from plexus.paths import graphs_data_path
    pre, post, w = _edges("edges")
    raw = max(float(r["weight"]) for r in csv.DictReader(open(graphs_data_path("neural", "zf_hd_ipn917",
                                                                                 "connections.csv"))))
    s = w.max() / raw                                                     # the twin's scale: weight per unit of area
    n = W_con.shape[0]
    A = np.zeros((n, n), np.int64)
    A[post, pre] = np.round(w / s)
    B = np.round(np.abs(W_con) / s).astype(np.int64)
    np.fill_diagonal(B, 0)

    def colours(M):
        c = [0] * n
        for _ in range(8):
            c = [hash((c[i], tuple(sorted((M[i, j], c[j]) for j in np.flatnonzero(M[i]))),
                       tuple(sorted((M[j, i], c[j]) for j in np.flatnonzero(M[:, i]))))) for i in range(n)]
        return c

    ours = defaultdict(list)
    for i, k in enumerate(colours(A)):
        ours[k].append(i)
    perm = np.array([ours[k][0] if len(ours.get(k, [])) == 1 else -1 for k in colours(B)])
    if (perm < 0).any() or len(set(perm.tolist())) != n:
        raise SystemExit(f"cell map: {(perm < 0).sum()} reference rows without a unique twin")
    if not np.array_equal(A[np.ix_(perm, perm)], B):
        raise SystemExit("cell map: the permuted twin wiring differs from the reference's W_con")
    return perm


def transplant(sd, perm):
    """The reference's trained parameters as the twin's learnables {key: tensor}, in the twin's edge orders."""
    sp = torch.nn.functional.softplus
    inv = np.argsort(perm)                                                # twin row -> reference row
    S = sd["S"].abs().double().numpy()
    W_in, W_out = sd["W_in"].double().numpy(), sd["W_out"].double().numpy()
    pre, post, _ = _edges("edges")
    syn = S[inv[post], inv[pre]]
    _, oL, _ = _edges("omegaL")
    _, oR, _ = _edges("omegaR")
    cp, cq, _ = _edges("cue")
    rp, rq, _ = _edges("out")
    if (inv[rp] >= N_RING).any():
        raise SystemExit("the twin's readout reads a cell outside the reference's ring rows")
    col = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32)[:, None]
    gl, gr = -float(sp(sd["v_artr_l"])), float(sp(sd["v_artr_r"]))
    return {"synapse.w": col(syn),
            "omega_L.w": col(np.full(len(oL), gl)),
            "omega_R.w": col(np.full(len(oR), gr)),
            "cue.w": col(W_in[inv[cq], cp]),                              # cue element 1 = cos, 2 = sin = W_in col
            "neuron.bias": col(sd["b"].double().numpy()[inv] / TAU),
            "readout.w": col(W_out[rq, inv[rp]]),
            "heading.bias": col(sd["b_out"].double().numpy())}, (gl, gr)


def rmse_deg(y, theta):
    """Per-trial RMSE of the wrapped heading error, after the reference's 10-frame warm-up -> [N] in deg."""
    th_hat = np.arctan2(y[..., 1], y[..., 0])
    err = np.angle(np.exp(1j * (th_hat[:, WARMUP:] - theta[:, WARMUP:])))
    return np.degrees(np.sqrt((err ** 2).mean(1)))


def main(n=512, device="cuda:0"):
    from plexus import engine
    from plexus import trainer as T
    engine.quiet(True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    perm = cell_map(sd["W_con"].double().numpy())
    fit, (gl, gr) = transplant(sd, perm)
    print(f"[cell map] 917 / 917 reference rows matched; wiring identical on {len(_edges('edges')[0])} edges")
    print(f"[gate] omega -> ARTR_L {gl:+.5f}, ARTR_R {gr:+.5f} per deg/s")

    spec = T.load("zf_hd_heading")
    U, Y, _ = T._data(spec, "test", 1, device)
    U, Y = U[:n].float(), Y[:n].float()
    theta = np.arctan2(Y[..., 1].cpu().numpy(), Y[..., 0].cpu().numpy())

    # --- the reference, in its own row order: the ARTR rows of the gate are the twin's ARTR rows mapped back
    inv = torch.as_tensor(np.argsort(perm), device=device)
    dev = torch.device(device)
    W_con = sd["W_con"].to(dev)
    N = W_con.shape[0]
    W_rec = sd["S"].to(dev).abs() * torch.sign(W_con) * (1.0 - torch.eye(N, device=dev))
    W_in = sd["W_in"].to(dev).clone()
    _, oL, _ = _edges("omegaL")
    _, oR, _ = _edges("omegaR")
    gate = torch.zeros(N, device=dev)
    gate[inv[torch.as_tensor(oL, device=dev)]] = gl
    gate[inv[torch.as_tensor(oR, device=dev)]] = gr
    W_in[:, 0] = gate
    b, W_out, b_out = sd["b"].to(dev), sd["W_out"].to(dev), sd["b_out"].to(dev)

    h = torch.zeros(n, N, device=dev)
    y_ref = []
    with torch.no_grad():                               # the reference's clock: h_{t+1} = step(h_t, u_t), y_t = g(h_{t+1})
        for t in range(U.shape[1]):
            h = h + (0.01 / TAU) * (-h + torch.sigmoid(h) @ W_rec.T + U[:, t] @ W_in.T + b)
            y_ref.append(torch.sigmoid(h[:, :N_RING]) @ W_out.T + b_out)
    y_ref = torch.stack(y_ref, 1).cpu().numpy()

    # --- the twin with the transplanted parameters, through the trainer's own rollout
    sim = T._model(spec, train=False)
    learn = T.Learnables(spec["learnable"], device)
    learn.restore({k: v.to(device) for k, v in fit.items()})
    with torch.no_grad():
        _, Yp = T.rollout(sim, learn, U, spec["task"], device, grad=False)
    y_pl = Yp.cpu().numpy()                                                 # [n, T(+1), 2]

    T0 = theta.shape[1]
    d = float(np.abs(y_pl[:, :T0] - y_ref).max())
    r_ref, r_pl = rmse_deg(y_ref, theta), rmse_deg(y_pl[:, :T0], theta)
    print(f"[the law] max |y_twin - y_ref| over (cos, sin), all {n} trials x {T0} frames = {d:.2e}")
    print(f"[heading RMSE, {n} held-out trials of t8_hd_heading, wrapped, after {WARMUP} frames]")
    print(f"  reference law                         {r_ref.mean():.2f} +- {r_ref.std():.2f} deg   "
          f"(its own test: 2.08 +- 1.15 deg over 512 trials of its corpus)")
    print(f"  twin, transplanted (trainer.rollout)  {r_pl.mean():.2f} +- {r_pl.std():.2f} deg")
    if d > 1e-4:
        raise SystemExit(f"the twin is not the reference law: max |y_twin - y_ref| = {d:.2e}")
    out = {"n_trials": n, "max_abs_diff_twin_vs_ref": d,
           "rmse_ref_deg": [float(r_ref.mean()), float(r_ref.std())],
           "rmse_twin_deg": [float(r_pl.mean()), float(r_pl.std())], "gate_per_deg_s": [gl, gr]}
    res = os.path.join(T.out_dir(spec), "results")
    os.makedirs(res, exist_ok=True)
    json.dump(out, open(os.path.join(res, "hd_transplant.json"), "w"), indent=1)
    print(f"[written] {os.path.join(res, 'hd_transplant.json')}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    main(a.n, a.device)
