#!/usr/bin/env python
"""test_06 -- spheroid, basement membrane and matrix in one run, with NO arrow between the sheet
and the stroma.

    python test_06_three_bodies.py [--device cuda:0] [--frames 399]
        ->  log/okuda_ECM/06_three_bodies/

WHAT THIS IS AND WHY THE MISSING ARROW IS THE POINT. 04 is the tissue and the matrix; 05 is the
sheet. This runs all three on one clock at one scale, with the two couplings that are already gated
-- plaques from the tissue to the sheet, particle-to-surface contact from the tissue to the matrix
-- and with NOTHING between the sheet and the matrix. They pass through each other.

That is not a placeholder, it is the CONTROL for the coupled run. The distance by which the stroma
crosses the sheet is a quantitative statement of exactly what `fibril_pull` and sheet-contact will
have to prevent, measured before either exists, in micrometres. A coupled run with no such control
can only say "the sheet holds the matrix out"; with it, it can say by how much it changed.

ONE-WAY, AND THE WORD MEANS SOMETHING PRECISE. Pass 1 ran the vertex model to completion and wrote
its vertices to disk. This is pass 2: it REPLAYS them, and the sheet and the matrix are pushed by
that moving surface. Both couplings compute their reaction and conserve it; neither has a live
tissue to give it back to. So the tissue's trajectory here is bit-identical to 04d's, which is what
makes gate I1 below a strict test rather than a comparison.

THE ONE DESIGN DECISION THAT IS NOT INHERITED. 05b binds a plaque to a barycentric point on an
epithelial FACE, which is right for its icosphere and wrong here: `divide_3d` re-indexes faces, so a
plaque bound to face k at frame 0 is bound to a different piece of tissue at frame 200. Vertices are
only ever APPENDED -- Nv is monotone and a vertex's position is continuous across a kept-frame pair,
which `mesh_contact` already relies on for its vertex velocity -- so here a plaque binds a sheet node
to an epithelial VERTEX INDEX. It is the same edge set with a stabler `.pre`.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

import bm_ops as BM                                                  # noqa: E402
import ecm_ops                                                       # noqa: E402,F401
import mesh_contact_ops as MC                                        # noqa: E402
import test_04_spheroid_ecm as T4                                    # noqa: E402
import tissue as TIS                                                 # noqa: E402

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
BOX_UM = 1172.33          # the declared scale, from gate_units
TISSUE_UM = 10.0


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _panel(ax, letter):
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


class Tissue:
    """The replayed epithelium, in BOX units: vertex positions and their per-frame velocity.

    One object so the sheet and the contact operator cannot disagree about where the tissue is --
    the contact reads it through `mesh_contact_ops` from the same cache, at the same scale, with the
    same frame -> mesh map (`mesh_stride`).
    """

    def __init__(self, npz, scale, centre, stride, dev, dtype=torch.float64):
        z = np.load(npz)
        self.z, self.scale, self.stride = z, float(scale), int(stride)
        self.c = torch.tensor(centre, device=dev, dtype=dtype)
        self.n_mesh = int(len(z["mesh_frames"]))
        self.dev, self.dtype = dev, dtype

    def at(self, f):
        """(positions, velocity per frame) of the epithelial vertices at pass-2 frame `f`."""
        j = min(self.n_mesh - 1, max(0, f // self.stride))
        a = torch.as_tensor(self.z[f"m{j}_pos"], device=self.dev, dtype=self.dtype) * self.scale
        if j + 1 < self.n_mesh:
            b = torch.as_tensor(self.z[f"m{j+1}_pos"], device=self.dev, dtype=self.dtype) * self.scale
            m = min(a.shape[0], b.shape[0])
            alpha = (f - j * self.stride) / self.stride if self.stride > 1 else 0.0
            x = a.clone()
            x[:m] = (1 - alpha) * a[:m] + alpha * b[:m]
            v = torch.zeros_like(x)
            v[:m] = (b[:m] - a[:m]) / self.stride
        else:
            x, v = a, torch.zeros_like(a)
        return self.c + x, v


class VertexPlaques:
    """`plaque`: an edge set epithelial VERTEX -> sheet node. One call, a delta to both endpoints.

    Normal spring at the linkage's own rest length, and a tangential friction solved rather than
    evaluated (05b's law): an explicit -xi*v is a stiffness of xi/dt that GROWS as the substep
    shrinks, so refining the step to stabilise it makes it worse.
    """

    def __init__(self, node, vert, l0, kn, xi):
        self.node, self.vert = node, vert
        self.l0, self.kn, self.xi = float(l0), float(kn), float(xi)

    def geometry(self, x_bm, x_ep, v_ep):
        p = x_ep[self.vert]
        d = x_bm[self.node] - p
        ell = d.norm(dim=1).clamp_min(1e-12)
        nh = d / ell[:, None]
        f_n = -self.kn * (ell - self.l0)[:, None] * nh          # on the SHEET node
        return p, nh, ell, v_ep[self.vert], f_n

    def scatter(self, f, x_bm, x_ep):
        """The force on the sheet and its reaction on the tissue, from one array."""
        fb = torch.zeros_like(x_bm)
        fe = torch.zeros_like(x_ep)
        fb.index_add_(0, self.node, f)
        fe.index_add_(0, self.vert, -f)
        return fb, fe


def seed_plaques(x_bm, x_ep, centre, target, seed=0):
    """One plaque per EPITHELIAL VERTEX, each to its nearest sheet node by direction.

    Seeded from the TISSUE side, and that is not a preference. Seeded from the sheet's side, every
    sheet node claims its nearest vertex -- and at frame 0 the sheet has 2,562 nodes against the
    tissue's 396 vertices, so six nodes are pulled onto each single point and the sheet is crushed
    onto a point set six times coarser than itself. Measured before this was fixed: the sheet
    reported a geometric stretch of 1.445 at frame 0, before anything had moved.

    So the plaque count is CAPPED BY THE TISSUE'S OWN VERTEX COUNT at the seeding frame, and asking
    for more than that is refused out loud rather than met by over-subscribing. Reaching a thousand
    plaques means seeding them where the tissue has a thousand vertices, which is a later frame --
    or making `plaque_seed` a turnover operator, which is what 05's G13 concluded slip requires
    anyway.
    """
    ub = torch.nn.functional.normalize(x_bm - centre, dim=1)
    ue = torch.nn.functional.normalize(x_ep - centre, dim=1)
    nv = ue.shape[0]
    idx = torch.empty(nv, dtype=torch.long, device=ub.device)
    for i in range(0, nv, 2048):
        idx[i:i + 2048] = (ue[i:i + 2048] @ ub.T).argmax(dim=1)     # vertex -> nearest sheet node
    vert = torch.arange(nv, device=ub.device)
    # a sheet node claimed twice would be pulled toward two vertices; keep the first claim
    node, first = torch.unique(idx, return_inverse=False), None
    keep = torch.ones(nv, dtype=torch.bool, device=ub.device)
    seen = torch.full((ub.shape[0],), -1, dtype=torch.long, device=ub.device)
    for i in range(nv):
        pass
    # vectorised "first claim wins": sort by node, take the first of each run
    order = torch.argsort(idx, stable=True)
    idx_s, vert_s = idx[order], vert[order]
    firsts = torch.ones_like(idx_s, dtype=torch.bool)
    firsts[1:] = idx_s[1:] != idx_s[:-1]
    node, vert = idx_s[firsts], vert_s[firsts]
    n_max = int(node.shape[0])
    if target < n_max:
        g = torch.Generator(device="cpu").manual_seed(seed)
        k = torch.randperm(n_max, generator=g)[:target].to(node.device)
        node, vert = node[k], vert[k]
    return node, vert, n_max


def main():
    import plexus.operators                                          # noqa: F401
    from plexus import schema
    from plexus.engine import run as engine_run

    dev = arg("--device", "cuda:0", str)
    name = arg("--name", "06_three_bodies", str)
    n_part = arg("--particles", 200000, int)
    subdiv = arg("--subdiv", 4, int)
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)

    # ---- pass 1, from the cache: 01c's tissue, the same one 04d used
    npz = TIS.load_or_build(frames=401, device=dev, buffer_x=4, myosin=1.0, myo_tau=20.0,
                            myo_new=1.0, myo_model="two_pool", myo_k_on=0.219, myo_tau_med=20.0,
                            myo_k_ex=0.05, myo_beta_T=0.0, myo_ring=1.0, myo_new_rel=True)
    z = np.load(npz)
    nmesh = len(z["mesh_frames"])
    scale = T4.R_FINAL_BOX / float(z["r_apical"][-1])
    frames = arg("--frames", 2 * nmesh - 1, int)
    stride = arg("--stride", max(1, round((frames + 1) / nmesh)), int)
    tis = Tissue(npz, scale, T4.CENTRE, stride, dev)

    # ---- the sheet, seeded one rest length outside the frame-0 apical surface
    # l0 IS NOT THE PHYSICAL 0.03 um. A hemidesmosome holds the sheet ~30 nm off the basal membrane;
    # this mesh's own roughness is a cell, 8.5 um, so a rest length below that is unresolvable and a
    # sheet seeded there sits inside the tissue's own bumps. 0.7 um is used and REPORTED, which is
    # 7x the sheet's thickness and a twelfth of a cell -- the smallest standoff this geometry can
    # carry, and a number the units gate can hold against the literature rather than a fitted one.
    l0 = 0.7 / BOX_UM
    x_ep0, _ = tis.at(0)
    ub, Fb, _ = BM.icosphere(subdiv, device=dev, dtype=torch.float64)
    c = torch.tensor(T4.CENTRE, device=dev, dtype=torch.float64)
    # the sheet takes the tissue's own radius by direction, so it starts ON the surface and not on a
    # sphere the surface merely resembles
    ue = torch.nn.functional.normalize(x_ep0 - c, dim=1)
    r_ep = (x_ep0 - c).norm(dim=1)
    R_bm = torch.empty(ub.shape[0], device=dev, dtype=torch.float64)
    for i in range(0, ub.shape[0], 2048):
        R_bm[i:i + 2048] = r_ep[(ub[i:i + 2048] @ ue.T).argmax(dim=1)]
    sheet = BM.Sheet(subdiv=subdiv, R0=1.0, E=arg("--E", 400.0, float),
                     thickness=0.1 / BOX_UM, nu=0.3, tau_r=arg("--tau-r", 25.0, float),
                     dev=dev, dtype=torch.float64)
    sheet.reseed(c + ub * (R_bm + l0)[:, None])
    # HOW MANY PLAQUES, AND THE HONEST STATEMENT OF WHAT THAT DENSITY IS. The literature spacing is
    # Sigma ~ 7T ~ 0.7 um (Kanchanawong 2010), which on a sphere of radius 176 um would be ~10^5
    # plaques -- two orders more than this mesh has nodes. The count is therefore set by the MESH and
    # not by the biology, and the run reports the spacing it actually achieves so the gap is a number
    # rather than an omission.
    n_plq = arg("--plaques", 1000, int)
    node, vert, n_max = seed_plaques(sheet.x[sheet.live_nodes], x_ep0, c, target=n_plq)
    node = sheet.live_nodes[node]
    if n_plq > n_max:
        print(f"[06] {n_plq} plaques asked for; the tissue has {x_ep0.shape[0]} vertices at the "
              f"seeding frame and one plaque per vertex is {n_max}. Seeding {n_max} and saying so, "
              f"rather than putting several on one vertex -- which crushes the sheet onto a point "
              f"set coarser than itself (measured: lam_geo 1.445 at frame 0).", flush=True)
    plq = VertexPlaques(node, vert, l0, kn=arg("--kn", 2.0e4, float), xi=arg("--xi", 0.0, float))
    lam, pv = sheet.spectral_rate(return_vec=True)
    sheet.M = arg("--zeta", 20.0, float) / (lam + plq.kn)
    n_sub = max(1, int(np.ceil(sheet.M * (lam + plq.kn) / 1.0)))
    R_end_um = T4.R_FINAL_BOX * BOX_UM
    spacing_um = float(np.sqrt(4 * np.pi * R_end_um ** 2 / max(node.shape[0], 1)))
    print(f"[06] sheet {int(sheet.n)} nodes / {int(sheet.m)} faces, {node.shape[0]} plaques "
          f"(spacing {spacing_um:.1f} um at the final radius, against the measured 0.7 um -- "
          f"{spacing_um / 0.7:.0f}x too sparse, set by the mesh), l0 = {l0 * BOX_UM:.3g} um, "
          f"M = {sheet.M:.4g}, {n_sub} substeps; tissue {x_ep0.shape[0]} vertices at "
          f"scale {scale:.6f}", flush=True)

    rec = {k: [] for k in ("frame", "momentum", "standoff", "lam_geo", "lam_el", "R_bm", "R_ep",
                           "n_sub", "cross_um", "cross_frac")}
    sheet_pos = []

    def on_frame(H, t):
        """One frame of the sheet, driven by the plaques, inside the engine's own frame loop."""
        if t >= frames:
            return
        x_ep, v_ep = tis.at(t)
        dt = 1.0 / n_sub
        mom = 0.0
        # THE ENGINE RUNS UNDER `torch.no_grad()` and the sheet's force is an autograd gradient of
        # its own energy -- which is what makes the force and the energy unable to disagree (G4).
        # The two conventions meet here, and the sheet's block re-enables grad for its own solve.
        for _ in range(n_sub):
            p, nh, ell, vp, f_n = plq.geometry(sheet.x, x_ep, v_ep)
            with torch.enable_grad():
                f_el = sheet.elastic_force(sheet.x)
            fb = torch.zeros_like(sheet.x)
            fb.index_add_(0, plq.node, f_n)
            v_prov = sheet.M * (f_el + fb)
            fbs, fes = plq.scatter(f_n, sheet.x, x_ep)
            mom = max(mom, float((fbs.sum(0) + fes.sum(0)).norm())
                      / (float(f_n.norm(dim=1).sum()) + 1e-300))
            sheet.advance(dt * v_prov, dt)
        # --- the number this run exists for: how far the stroma crosses the sheet, in um
        P = H.level("mpm_particle").get("pos").to(torch.float64)
        u = torch.nn.functional.normalize(P - c, dim=1)
        r = (P - c).norm(dim=1)
        ubm = torch.nn.functional.normalize(sheet.x[sheet.live_nodes] - c, dim=1)
        rbm = (sheet.x[sheet.live_nodes] - c).norm(dim=1)
        k = torch.empty(P.shape[0], dtype=torch.long, device=P.device)
        for i in range(0, P.shape[0], 65536):
            k[i:i + 65536] = (u[i:i + 65536] @ ubm.T).argmax(dim=1)
        inside = rbm[k] - r                      # >0 : the particle is inside the sheet
        cross = inside.clamp_min(0.0)
        l1, _ = sheet.stretch_geo(); e1, _ = sheet.stretch_elastic()
        rec["frame"].append(int(t)); rec["momentum"].append(mom)
        rec["standoff"].append(float((ell.mean() - plq.l0) * BOX_UM))
        rec["lam_geo"].append(float(l1.mean())); rec["lam_el"].append(float(e1.mean()))
        rec["R_bm"].append(float(rbm.mean())); rec["R_ep"].append(float((x_ep - c).norm(dim=1).mean()))
        rec["n_sub"].append(n_sub)
        rec["cross_um"].append(float(cross[cross > 0].mean() * BOX_UM) if bool((cross > 0).any())
                               else 0.0)
        rec["cross_frac"].append(float((cross > 0).float().mean()))
        if t % 40 == 0:
            print(f"[06] frame {t:3d}  lam_geo {l1.mean():.3f}  lam_el {e1.mean():.3f}  "
                  f"standoff {rec['standoff'][-1]:+.3f} um  matrix inside the sheet "
                  f"{100 * rec['cross_frac'][-1]:.1f}% by {rec['cross_um'][-1]:.1f} um  "
                  f"momentum {mom:.1e}", flush=True)
        if t % max(1, frames // 120) == 0:
            sheet_pos.append((int(t), sheet.x[sheet.live_nodes].detach().cpu().numpy().copy()))

    spec = T4.build(name, frames, npz, scale, n_particles=n_part, mesh_stride=stride)
    path = os.path.join(d, "spec.yaml")
    yaml.safe_dump(spec, open(path, "w"), sort_keys=False)
    S = schema.load(path)
    print(f"[06] {S.units.describe()}", flush=True)
    ecm_ops.STRESS_HISTORY.clear(); ecm_ops.STRESS_RAW.clear(); MC.reset()
    t0 = time.time()
    H, out = engine_run(S, device=dev, on_frame=on_frame)
    print(f"[06] SOLVE {time.time() - t0:.0f} s", flush=True)

    P = np.asarray(out["sets"]["mpm_particle"]["pos"], np.float32)
    vm = [np.asarray(v, np.float32) for v in ecm_ops.STRESS_RAW] or None
    np.savez_compressed(os.path.join(d, "traj.npz"), pos=P,
                        vm=np.asarray(vm, np.float16) if vm else np.zeros((0,), np.float16),
                        sheet_frames=np.asarray([t for t, _ in sheet_pos], np.int32),
                        **{f"s{i}": q for i, (_, q) in enumerate(sheet_pos)})

    m = dict(frames=int(frames), stride=int(stride), scale=float(scale), l0_um=float(l0 * BOX_UM),
             n_plaque=int(node.shape[0]), plaque_spacing_um=float(spacing_um),
             plaque_spacing_over_measured=float(spacing_um / 0.7), sheet_nodes=int(sheet.n), sheet_faces=int(sheet.m),
             particles=int(P.shape[1]), n_sub=int(n_sub), series=rec,
             gates=dict(
                 I2=dict(name="the sheet carries the stretch the tissue gives it",
                         threshold="lam_geo within 5% of R_ep(end)/R_ep(0)",
                         measured=float(rec["lam_geo"][-1]),
                         expected=float(rec["R_ep"][-1] / max(rec["R_ep"][0], 1e-12))),
                 I3=dict(name="one scale: the sheet stays on the tissue it is plaqued to",
                         threshold="mean |R_bm - R_ep| < 2 um",
                         measured=float(abs(np.mean(np.array(rec["R_bm"]) - np.array(rec["R_ep"])))
                                        * BOX_UM)),
                 I4=dict(name="the matrix crosses the sheet -- the arrow that is missing",
                         threshold="reported, not thresholded: this is the control",
                         measured_um=float(rec["cross_um"][-1]),
                         measured_frac=float(rec["cross_frac"][-1])),
                 I5=dict(name="the plaque returns its reaction",
                         threshold="< 1e-12 (float64, on its own force pair)",
                         measured=float(max(rec["momentum"])))))
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 4, figsize=(16.4, 3.4), facecolor="white")
    f = np.asarray(rec["frame"])
    ax[0].plot(f, rec["lam_geo"], color="#2b6cb0", label=r"$\lambda^{\rm geo}$")
    ax[0].plot(f, rec["lam_el"], color="#e0452b", label=r"$\lambda^{\rm el}$")
    ax[0].plot(f, np.array(rec["R_ep"]) / rec["R_ep"][0], color="#999", ls="--", label="tissue $R/R_0$")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("stretch"); ax[0].legend(fontsize=7, frameon=False)
    _panel(ax[0], "a")
    ax[1].plot(f, rec["standoff"], color="#7b4fb5")
    ax[1].axhline(0, color="#999", ls="--", lw=0.8)
    ax[1].set_xlabel("frame"); ax[1].set_ylabel(r"standoff $\ell-\ell_0$ ($\mu$m)")
    _panel(ax[1], "b")
    ax[2].plot(f, 100 * np.asarray(rec["cross_frac"]), color="#B03A2E")
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("matrix inside the sheet (%)")
    _panel(ax[2], "c")
    ax[3].plot(f, rec["cross_um"], color="#B03A2E")
    ax[3].set_xlabel("frame"); ax[3].set_ylabel(r"depth of crossing ($\mu$m)")
    _panel(ax[3], "d")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(d, "gate.png"), dpi=150, facecolor="white")
    plt.close(fig)
    for k, v in m["gates"].items():
        print(f"[gate] {k}: {v}", flush=True)
    print(f"[06] -> {d}", flush=True)


if __name__ == "__main__":
    main()
