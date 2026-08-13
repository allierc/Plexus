#!/usr/bin/env python
"""test_05n -- can the sheet track a REAL epithelium? Four variants, one cause each.

    python test_05n_real_surface.py [--device cuda:0] [--frames 399]
        ->  log/okuda_ECM/05n_{tether_smooth,tether_real,plaque_real,plaque_refine}/

WHY THIS EXISTS. 05's sheet has only ever tracked a smooth, analytically expanding icosphere, and
G5's stretch fidelity of 0.9935 is a measurement against that. The real epithelium is different in
two ways at once: it is BUMPY at the scale of a cell, and it GAINS VERTICES by division. Asked to
track one -- with the matrix removed entirely, so nothing else can be blamed -- the sheet loses the
positive-definiteness of its metric partway through. The gate that does not exist is the one that
failed.

WHAT FAILS BEFORE WHY IT FAILS. A Cholesky that will not factor says a triangle's metric has stopped
being positive-definite, and there are two quite different roads to that:

    a STABILITY violation   the step is too large for the current stiffness. The sheet's own bound
                            is s = dt*M*lambda_max < 2 and lambda_max grows as it stretches, so `s`
                            is a number that can be watched crossing.
    a GEOMETRIC degeneracy  a triangle collapses or inverts because the surface it is tracking has
                            features finer than its own edges. Watched as the minimum triangle
                            QUALITY (4*sqrt(3)*A / sum of squared edges: 1 equilateral, 0 degenerate)
                            and the signed area.

The two want opposite fixes -- smaller substeps against refinement -- so a run that reports only
"it failed" cannot be acted on. Every variant here logs both, every frame, and the report says which
crossed first.

THE FOUR VARIANTS, each adding exactly one thing to the one before:

    tether_smooth   a tether to the SMOOTHED R(theta,phi) map. The closest thing to 05a's driver
                    that still uses the real run's growth: smooth in space, real in time. If this
                    fails, the problem is the growth, not the surface.
    tether_real     the same tether, to each node's own nearest VERTEX. Adds the bumpiness and
                    nothing else.
    plaque_real     plaques at the measured density instead of a tether on every node. Adds the
                    sparseness of the anchoring: a tether pulls every node, a plaque pulls one in
                    six and leaves the rest to the sheet's own elasticity.
    plaque_refine   the same with `max_refine` on. The tissue's radius triples, so the sheet's mean
                    edge would grow 3.4x if it never refined; this asks whether the failure is that.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "discovery_okuda", "ops"),
          os.path.join(_ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

import bm_ops as BM                                                  # noqa: E402
import test_04_spheroid_ecm as T4                                    # noqa: E402
import test_06_three_bodies as T6                                    # noqa: E402
import tissue as TIS                                                 # noqa: E402

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
BOX_UM = 1172.33


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _panel(ax, letter):
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def quality(x, F):
    """4*sqrt(3)*A / (sum of squared edges): 1 for equilateral, 0 for degenerate.

    The scale-free way to ask whether a triangle is still a triangle. Area alone shrinks as the mesh
    refines and would report a healthy refinement as a degeneracy.
    """
    a, b, c = x[F[:, 0]], x[F[:, 1]], x[F[:, 2]]
    e1, e2 = b - a, c - a
    A = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    s2 = (e1 * e1).sum(1) + (e2 * e2).sum(1) + ((c - b) ** 2).sum(1)
    return (4.0 * np.sqrt(3.0) * A / s2.clamp_min(1e-30)), A


def run(kind, dev, frames, npz, scale, stride, subdiv=4, n_plq=1000, kn=2.0e4, zeta=20.0,
        refresh=10, tau_p=5.0, E=400.0, tau_r=25.0):
    z = np.load(npz)
    c = torch.tensor(T4.CENTRE, device=dev, dtype=torch.float64)
    tis = T6.Tissue(npz, scale, T4.CENTRE, stride, dev)
    x_ep0, _ = tis.at(0)
    ub, Fb, _ = BM.icosphere(subdiv, device=dev, dtype=torch.float64)
    ub = torch.nn.functional.normalize(ub, dim=1)
    l0 = 0.7 / BOX_UM

    # the smoothed radius map, which is `tether_smooth`'s driver and every variant's fallback field
    smap = torch.as_tensor(z["smap"], device=dev, dtype=torch.float64) * scale
    nth, nph = smap.shape[1], smap.shape[2]
    th = torch.acos(ub[:, 2].clamp(-1, 1)); ph = torch.atan2(ub[:, 1], ub[:, 0]) % (2 * np.pi)
    it = (th / np.pi * nth).long().clamp(0, nth - 1)
    ip = (ph / (2 * np.pi) * nph).long().clamp(0, nph - 1)

    def R_smooth(f):
        return smap[min(f // stride, smap.shape[0] - 1)][it, ip]

    node = vert = None
    if kind.startswith("plaque"):
        node, vert, n_max = T6.seed_plaques(ub, x_ep0, c, target=n_plq)
    R0 = R_smooth(0).clone()
    x0 = c + ub * (R0 + l0)[:, None]        # every node on the SMOOTH field; see VertexPlaques
    sheet = BM.Sheet(subdiv=subdiv, R0=1.0, E=E, thickness=0.1 / BOX_UM, nu=0.3, tau_r=tau_r,
                     max_refine=(2 if kind == "plaque_refine" else 0), dev=dev, dtype=torch.float64)
    sheet.reseed(x0)
    if node is not None:
        node = sheet.live_nodes[node]
        plq = T6.VertexPlaques(node, vert,
                               (sheet.x[node] - x_ep0[vert]).norm(dim=1).clone(), kn=kn, xi=0.0)
    with torch.enable_grad():
        lam, pv = sheet.spectral_rate(return_vec=True)
    sheet.M = zeta / (lam + (kn if node is not None else kn))
    n_sub = max(1, int(np.ceil(sheet.M * (lam + kn))))
    gen = torch.Generator().manual_seed(0)
    rec = {k: [] for k in ("frame", "lam_geo", "q_min", "A_min", "s_group", "edge", "n_sub",
                           "n_plaque", "n_face")}
    fail, why = None, None
    t0 = time.time()
    for t in range(frames):
        try:
            if t % refresh == 0 and bool(torch.isfinite(sheet.x).all()):
                with torch.enable_grad():
                    lam, pv = sheet.spectral_rate(iters=25, v0=pv, return_vec=True)
                n_sub = max(1, int(np.ceil(sheet.M * (lam + kn))))
            x_ep, v_ep = tis.at(t)
            if node is not None:
                n_now, _ = plq.retarget(sheet.x, x_ep, c, n_plq, 1.0 / tau_p, gen)
            else:
                n_now = 0
                # the tether: every node pulled toward its own target, which is what 05a does
                if kind == "tether_smooth":
                    tgt = c + ub * (R_smooth(t) + l0)[:, None]
                else:
                    ue = torch.nn.functional.normalize(x_ep - c, dim=1)
                    k = torch.empty(ub.shape[0], dtype=torch.long, device=dev)
                    for i in range(0, ub.shape[0], 2048):
                        k[i:i + 2048] = (ub[i:i + 2048] @ ue.T).argmax(dim=1)
                    tgt = x_ep[k] + l0 * ue[k]
            dt = 1.0 / n_sub
            for _ in range(n_sub):
                with torch.enable_grad():
                    f = sheet.elastic_force(sheet.x)
                if node is not None:
                    _, _, _, _, f_n = plq.geometry(sheet.x, x_ep, v_ep)
                    fb = torch.zeros_like(sheet.x); fb.index_add_(0, plq.node, f_n)
                    f = f + fb
                else:
                    f = f + kn * (tgt - sheet.x)
                sheet.advance(dt * sheet.M * f, dt)
            if kind == "plaque_refine" and sheet.n_refinements < sheet.max_refine:
                if float((sheet.x[sheet.Ed[:, 1]] - sheet.x[sheet.Ed[:, 0]]).norm(dim=1).mean()) \
                        > 1.45 * sheet.mean_edge_seed:
                    sheet.refine()
                    with torch.enable_grad():
                        lam, pv = sheet.spectral_rate(iters=25, return_vec=True)
            q, A = quality(sheet.x, sheet.Fc)
            l1, _ = sheet.stretch_geo()
            rec["frame"].append(t); rec["lam_geo"].append(float(l1.mean()))
            rec["q_min"].append(float(q.min())); rec["A_min"].append(float(A.min()))
            rec["s_group"].append(float(sheet.M * lam / n_sub))
            rec["edge"].append(float((sheet.x[sheet.Ed[:, 1]]
                                      - sheet.x[sheet.Ed[:, 0]]).norm(dim=1).mean()
                                     / sheet.mean_edge_seed))
            rec["n_sub"].append(int(n_sub)); rec["n_plaque"].append(int(n_now))
            rec["n_face"].append(int(sheet.m))
            if not np.isfinite(rec["lam_geo"][-1]):
                fail, why = t, "lam_geo not finite"
                break
        except Exception as e:                          # the Cholesky, and anything like it
            fail, why = t, f"{type(e).__name__}: {str(e)[:80]}"
            break
    out = dict(kind=kind, frames=int(frames), failed_at=fail, why=why,
               wall_s=float(time.time() - t0), n_sub_final=int(n_sub),
               subdiv=int(subdiv), n_plaque=(0 if node is None else int(node.shape[0])),
               series=rec)
    # WHICH CROSSED FIRST: the stability group or the geometry. This is the whole point of the run.
    if rec["frame"]:
        q0 = rec["q_min"][0]
        i_q = next((i for i, v in enumerate(rec["q_min"]) if v < 0.25 * q0), None)
        i_s = next((i for i, v in enumerate(rec["s_group"]) if v > 2.0), None)
        out["first_crossing"] = ("geometry" if (i_q is not None and (i_s is None or i_q < i_s))
                                 else ("stability" if i_s is not None else "neither"))
        out["q_min_first"], out["q_min_last"] = q0, rec["q_min"][-1]
    return out


def main():
    dev = arg("--device", "cuda:0", str)
    npz = TIS.load_or_build(frames=401, device=dev, buffer_x=4, myosin=1.0, myo_tau=20.0,
                            myo_new=1.0, myo_model="two_pool", myo_k_on=0.219, myo_tau_med=20.0,
                            myo_k_ex=0.05, myo_beta_T=0.0, myo_ring=1.0, myo_new_rel=True)
    z = np.load(npz)
    nmesh = len(z["mesh_frames"])
    scale = T4.R_FINAL_BOX / float(z["r_apical"][-1])
    frames = arg("--frames", 2 * nmesh - 1, int)
    stride = max(1, round((frames + 1) / nmesh))
    kinds = ("tether_smooth", "tether_real", "plaque_real", "plaque_refine")
    allout = {}
    for k in kinds:
        o = run(k, dev, frames, npz, scale, stride)
        allout[k] = o
        d = os.path.join(LOG, f"05n_{k}")
        os.makedirs(d, exist_ok=True)
        json.dump(o, open(os.path.join(d, "metrics.json"), "w"), indent=1)
        print(f"[05n] {k:14s} {'ran to the end' if o['failed_at'] is None else 'FAILED at frame %d' % o['failed_at']}"
              f" | lam_geo {o['series']['lam_geo'][-1]:.3f} | q_min {o['q_min_first']:.3f} -> "
              f"{o['q_min_last']:.3f} | first crossing: {o['first_crossing']} | {o['wall_s']:.0f} s",
              flush=True)
        if o["why"]:
            print(f"              {o['why']}", flush=True)

    fig, ax = plt.subplots(1, 4, figsize=(16.4, 3.4), facecolor="white")
    col = dict(zip(kinds, ("#999", "#2b6cb0", "#B03A2E", "#1B7F4B")))
    for k in kinds:
        r = allout[k]["series"]
        f = np.asarray(r["frame"])
        ax[0].plot(f, r["lam_geo"], color=col[k], label=k)
        ax[1].semilogy(f, np.maximum(r["q_min"], 1e-6), color=col[k])
        ax[2].plot(f, r["s_group"], color=col[k])
        ax[3].plot(f, r["edge"], color=col[k])
    ax[1].axhline(0.05, color="#ccc", ls="--", lw=0.8)
    ax[2].axhline(2.0, color="#ccc", ls="--", lw=0.8)
    for a, lab in zip(ax, (r"$\lambda^{\rm geo}$", "worst triangle quality",
                           r"stability group $\Delta t M\lambda_{\max}$", "mean edge / seeded")):
        a.set_xlabel("frame"); a.set_ylabel(lab)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7, frameon=False)
    for i, a in enumerate(ax):
        _panel(a, "abcd"[i])
    fig.tight_layout()
    out0 = os.path.join(LOG, "05n_tether_smooth", "gate.png")
    fig.savefig(out0, dpi=150, facecolor="white")
    for k in kinds[1:]:
        fig.savefig(os.path.join(LOG, f"05n_{k}", "gate.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[05n] -> {LOG}/05j_*", flush=True)


if __name__ == "__main__":
    main()
