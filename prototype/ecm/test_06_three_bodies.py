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

    def retarget(self, x_bm, x_ep, centre, target, frac, gen):
        """`plaque_seed` AS TURNOVER: a fraction of the bonds let go and re-form where their node is
        NOW, and the set grows as the tissue makes new vertices.

        This is the operator 05's G13 concluded slip requires. A plaque anchored to a FIXED point is
        a pin in the tangential direction however large its friction is -- xi can only set how fast
        the sheet equilibrates to the pin, never whether it slides. A bond that lets go and re-forms
        at the tissue under it now lets the sheet move over the epithelium at a rate set by the
        turnover, which is what a hemidesmosome does.

        RE-FORMING IS NOT EXCLUSIVE, and the first version made it so. Requiring a free vertex meant
        that at frame 0, with 395 of the tissue's 396 vertices already claimed, all 79 released bonds
        re-formed on the ONE free vertex -- 79 sheet nodes pulled onto a single point, lam_geo 2.89
        before anything had moved, and the sheet's Cholesky failing three frames later. A vertex is
        8.5 um of tissue and can carry several adhesions; what cannot happen is many nodes on one
        point, and re-forming each bond at its OWN nearest vertex cannot produce that.
        """
        ub = torch.nn.functional.normalize(x_bm - centre, dim=1)
        ue = torch.nn.functional.normalize(x_ep - centre, dim=1)
        n_rel = int(round(frac * self.node.shape[0]))
        if n_rel:
            k = torch.randperm(self.node.shape[0], generator=gen)[:n_rel].to(x_ep.device)
            self.vert[k] = (ub[self.node[k]] @ ue.T).argmax(dim=1)
        # grow: vertices no bond has claimed can take one, up to the target
        room = int(target) - int(self.node.shape[0])
        if room > 0:
            claimed = torch.zeros(x_ep.shape[0], dtype=torch.bool, device=x_ep.device)
            claimed[self.vert] = True
            free_v = torch.nonzero(~claimed, as_tuple=False).flatten()
            if free_v.numel():
                add = free_v[: min(room, free_v.numel())]
                self.node = torch.cat([self.node, (ue[add] @ ub.T).argmax(dim=1)])
                self.vert = torch.cat([self.vert, add])
        return int(self.node.shape[0]), int(n_rel)

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


def movie(d, name, npz, scale, stride, sheet_pos, P, band, cmap_colors, slab=0.022):
    """The section: matrix strands, the epithelium's own cross-section, and the sheet on top of it.

    A section and not a 3D view, because the question this run asks -- is the sheet where the tissue
    is, and is it stretched more than the tissue stretched it -- is a question about a rim, and a rim
    is what a cut shows.
    """
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import LineCollection
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    z = np.load(npz)
    nmesh = len(z["mesh_frames"])
    c = np.asarray(T4.CENTRE, np.float32)
    per = 20
    nf = P.shape[1] // per
    mid0 = P[0][: nf * per].reshape(nf, per, 3).mean(1)
    keep2d = np.nonzero(np.abs(mid0[:, 2] - c[2]) < slab)[0]
    lim = 0.30
    fig = plt.figure(figsize=(6.0, 6.0), facecolor="black")
    wri = FFMpegWriter(fps=20, metadata={"title": name})
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=110):
        for (t, q) in sheet_pos:
            ti = min(P.shape[0] - 1, t)
            fig.clf()
            a = fig.add_subplot(1, 1, 1, facecolor="black")
            # the matrix, as strands, coloured by its stress band
            S = P[ti][: nf * per].reshape(nf, per, 3)[keep2d]
            B = np.median(band[ti][: nf * per].reshape(nf, per), axis=1).astype(int)[keep2d]
            a.add_collection(LineCollection([q_[:, :2] for q_ in S], linewidths=0.7,
                                            colors=[cmap_colors[int(b) % len(cmap_colors)]
                                                    for b in B], alpha=0.85))
            # the epithelium, from its own mesh
            j = min(nmesh - 1, t // stride)
            V = z[f"m{j}_pos"] * scale + c
            es, et = z[f"m{j}_E_srce"], z[f"m{j}_E_trgt"]
            sl = np.abs(0.5 * (V[es] + V[et])[:, 2] - c[2]) < slab
            a.add_collection(LineCollection(
                [np.stack([V[x][:2], V[y][:2]]) for x, y in zip(es[sl], et[sl])],
                linewidths=0.5, colors="#e8dcc0", alpha=0.85))
            # the sheet, as the dots it is, in the same slab
            m = np.abs(q[:, 2] - c[2]) < slab
            a.scatter(q[m][:, 0], q[m][:, 1], s=5.0, c="#2ecc71", marker=".", linewidths=0)
            a.set_xlim(0.5 - lim, 0.5 + lim); a.set_ylim(0.5 - lim, 0.5 + lim)
            a.set_aspect("equal"); a.axis("off")
            a.text(0.02, 0.98, f"{name}   frame {t}", transform=a.transAxes, color="white",
                   fontsize=10, va="top")
            a.text(0.02, 0.94, "matrix (stress) / epithelium (cream) / sheet (green)",
                   transform=a.transAxes, color="#aaa", fontsize=8, va="top")
            wri.grab_frame()
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)


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
    # ONE PAIRING, USED TWICE. The sheet's seed radius and its plaque partner were chosen by two
    # different maps -- the radius from each NODE's nearest vertex, the plaque from each VERTEX's
    # nearest node -- so a node was seeded at the radius of one vertex and then pulled toward a
    # different one. That is an inconsistency of order the tissue's own bumpiness applied at frame 0,
    # before anything moves, and it is why raising the anchored fraction from 15% to 62% made the
    # frame-0 stretch WORSE (1.049 -> 1.175) instead of better. Here the pairing is computed ONCE,
    # from the tissue side, and the seed radius is read through it: an anchored node is placed at
    # exactly l0 from its own partner, so every plaque starts at its rest length and exerts nothing.
    ub_unit = torch.nn.functional.normalize(
        BM.icosphere(subdiv, device=dev, dtype=torch.float64)[0], dim=1)
    n_plq = arg("--plaques", 1000, int)
    node, vert, n_max = seed_plaques(ub_unit, x_ep0, c, target=n_plq)
    if n_plq > n_max:
        print(f"[06] {n_plq} plaques asked for; the tissue has {x_ep0.shape[0]} vertices at the "
              f"seeding frame and one plaque per vertex is {n_max}. Seeding {n_max}, and the "
              f"turnover operator grows the set as `divide_3d` makes more.", flush=True)
    # the radius field: the smoothed map everywhere, overridden by the partner's own radius where a
    # plaque lands, so the anchored nodes are exact and the rest are smooth rather than bumpy
    smap = torch.as_tensor(z["smap"][0], device=dev, dtype=torch.float64) * scale
    nth, nph = smap.shape
    th = torch.acos(ub[:, 2].clamp(-1, 1)); ph = torch.atan2(ub[:, 1], ub[:, 0]) % (2 * np.pi)
    R_bm = smap[(th / np.pi * nth).long().clamp(0, nth - 1),
                (ph / (2 * np.pi) * nph).long().clamp(0, nph - 1)].clone()
    # AND AN ANCHORED NODE SITS ON ITS PARTNER'S OWN RAY, not on its own at the partner's radius.
    # The two directions differ by up to the vertex spacing -- 396 vertices is ~10 deg, and 10 deg at
    # the seeding radius is 8 um against an l0 of 0.7 um -- so placing the node along its own ray
    # leaves every plaque stretched by ten rest lengths before the clock starts. Measured that way:
    # lam_geo 3.34 and slip 11.3 um at frame 0.
    x0 = c + ub * (R_bm + l0)[:, None]
    u_v = torch.nn.functional.normalize(x_ep0[vert] - c, dim=1)
    x0[node] = x_ep0[vert] + l0 * u_v
    # the sheet, on that radius field, and the plaque indices moved into the sheet's own numbering
    sheet = BM.Sheet(subdiv=subdiv, R0=1.0, E=arg("--E", 400.0, float),
                     thickness=0.1 / BOX_UM, nu=0.3, tau_r=arg("--tau-r", 25.0, float),
                     max_refine=arg("--max-refine", 0, int),
                     dev=dev, dtype=torch.float64)
    sheet.reseed(x0)
    node = sheet.live_nodes[node]
    plq = VertexPlaques(node, vert, l0, kn=arg("--kn", 2.0e4, float), xi=arg("--xi", 0.0, float))
    lam, pv = sheet.spectral_rate(return_vec=True)
    zeta = arg("--zeta", 20.0, float)
    s_target = arg("--s-target", 1.0, float)
    refresh = arg("--refresh", 10, int)
    sheet.M = zeta / (lam + plq.kn)
    n_sub = max(1, int(np.ceil(sheet.M * (lam + plq.kn) / s_target)))
    R_end_um = T4.R_FINAL_BOX * BOX_UM
    spacing_um = float(np.sqrt(4 * np.pi * R_end_um ** 2 / max(node.shape[0], 1)))
    print(f"[06] sheet {int(sheet.n)} nodes / {int(sheet.m)} faces, {node.shape[0]} plaques "
          f"(spacing {spacing_um:.1f} um at the final radius, against the measured 0.7 um -- "
          f"{spacing_um / 0.7:.0f}x too sparse, set by the mesh), l0 = {l0 * BOX_UM:.3g} um, "
          f"M = {sheet.M:.4g}, {n_sub} substeps; tissue {x_ep0.shape[0]} vertices at "
          f"scale {scale:.6f}", flush=True)

    tau_p = arg("--plaque-tau", 5.0, float)          # frames; 5 x 600 s = 50 min, focal-adhesion-ish
    gen = torch.Generator().manual_seed(0)
    ub0 = torch.nn.functional.normalize(sheet.x[sheet.live_nodes] - c, dim=1).clone()
    rec = {k: [] for k in ("frame", "momentum", "standoff", "lam_geo", "lam_el", "R_bm", "R_ep",
                           "n_sub", "cross_um", "cross_frac", "n_plaque", "slip_um")}
    sheet_pos = []

    def on_frame(H, t):
        """One frame of the sheet, driven by the plaques, inside the engine's own frame loop."""
        if t >= frames:
            return
        nonlocal lam, pv, n_sub
        # THE SUBSTEP COUNT IS RE-MEASURED, NOT FIXED AT SEEDING. A StVK membrane's tangent stiffens
        # as it stretches: 05 measured its largest Hessian eigenvalue growing 19.1x over a run, and
        # the stability bound is dt*M*lambda_max < 2 -- so a count fixed from the seeded value is a
        # count that is 19x too small by the end. Fixing it here was not a guess: this rig NaN'd on
        # the sheet's Cholesky partway through, which is that bound being crossed.
        if t % refresh == 0 and bool(torch.isfinite(sheet.x).all()):
            with torch.enable_grad():          # power iteration on Hessian-vector products
                lam, pv = sheet.spectral_rate(iters=25, v0=pv, return_vec=True)
            n_sub = max(1, int(np.ceil(sheet.M * (lam + plq.kn) / s_target)))
        x_ep, v_ep = tis.at(t)
        n_now, n_rel = plq.retarget(sheet.x, x_ep, c, n_plq, 1.0 / tau_p, gen)
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
        rec["n_plaque"].append(n_now)
        # SLIP: how far the sheet has moved TANGENTIALLY over the epithelium since seeding. With a
        # fixed anchor this is identically zero by construction, which is G13; with turnover it is
        # the quantity a friction law can act on.
        ub_now = torch.nn.functional.normalize(sheet.x[sheet.live_nodes] - c, dim=1)
        rec["slip_um"].append(float((1.0 - (ub_now * ub0).sum(1)).clamp_min(0).mean().sqrt()
                                    * rbm.mean() * BOX_UM * 1.4142))
        rec["cross_um"].append(float(cross[cross > 0].mean() * BOX_UM) if bool((cross > 0).any())
                               else 0.0)
        rec["cross_frac"].append(float((cross > 0).float().mean()))
        if t % 40 == 0:
            print(f"[06] frame {t:3d}  lam_geo {l1.mean():.3f}  lam_el {e1.mean():.3f}  "
                  f"standoff {rec['standoff'][-1]:+.3f} um  matrix inside the sheet "
                  f"{100 * rec['cross_frac'][-1]:.1f}% by {rec['cross_um'][-1]:.1f} um  "
                  f"plaques {n_now}  slip {rec['slip_um'][-1]:.2f} um  "
                  f"momentum {mom:.1e}", flush=True)
        if t % max(1, frames // 120) == 0:
            sheet_pos.append((int(t), sheet.x[sheet.live_nodes].detach().cpu().numpy().copy()))

    # --no-matrix: the sheet against the REAL tissue and nothing else. This exists to attribute a
    # failure rather than to model anything. 05's sheet has only ever tracked a smooth, analytically
    # expanding icosphere; here it tracks a surface that is bumpy and that GAINS VERTICES by
    # division. If it fails at the same frame with the matrix removed, the failure is the sheet
    # meeting that driver and belongs in 05; if it survives, the failure is in this assembly.
    if "--no-matrix" in sys.argv:
        class _L:                                     # the one thing on_frame reads off the engine
            def get(self, _k):
                return torch.zeros(1, 3, device=dev)
        class _H:
            def level(self, _n):
                return _L()
        t0 = time.time()
        for t in range(frames):
            on_frame(_H(), t)
        print(f"[06] sheet-only: {time.time() - t0:.0f} s for {frames} frames", flush=True)
        json.dump(dict(mode="no-matrix", frames=int(frames), series=rec),
                  open(os.path.join(d, "metrics.json"), "w"), indent=1)
        print(f"[06] -> {d}", flush=True)
        return

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
                 I6=dict(name="turnover grows the plaque set with the tissue",
                         threshold=f"reaches the target of {n_plq}",
                         measured=int(rec["n_plaque"][-1]), target=int(n_plq)),
                 I7=dict(name="the sheet slides: slip is not identically zero (05's G13)",
                         threshold="> 0 um; a fixed anchor gives exactly 0",
                         measured_um=float(rec["slip_um"][-1])),
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
    import ecm_spec as ES
    import test_02_ecm_block as T2
    band, _ = T2.bands_from_vm(vm) if vm else (np.zeros((P.shape[0], P.shape[1]), np.uint8), 1.0)
    if sheet_pos:
        movie(d, name, npz, scale, stride, sheet_pos, P, np.asarray(band), ES.STRESS_COLORS)
    for k, v in m["gates"].items():
        print(f"[gate] {k}: {v}", flush=True)
    print(f"[06] -> {d}", flush=True)


if __name__ == "__main__":
    main()
