#!/usr/bin/env python
"""test_03_mesh_contact -- the vertex/MPM interface, on the smallest rig that can falsify it.

    python test_03_mesh_contact.py [--device cuda:0]   ->  log/okuda_ECM/03_mesh_contact/

WHY A RIG AND NOT THE SPHEROID. The question is whether a triangulated surface and an MPM continuum can
exchange force at OUR resolution, and the spheroid answers it with sixty thousand cells, a growth law
and a replay in the way. A flat patch dropped on a block of matrix answers the same question with three
numbers, and each of them kills the method if it fails.

WHICH SCHEME, AND WHY NOT THE OBVIOUS ONE. Grid-based coupling (CFEMP, Lian et al. 2011 CMAME 200:3482)
resolves contact by comparing the two bodies' velocities at shared grid nodes, and needs the mesh and
the grid to be comparable in size. Ours are not: a cell is 0.73 dx and the basement membrane 0.1 dx, so
both bodies live inside one cell and the grid hands them one velocity -- the weld measured in runs
142-151. ICFEMP (Chen et al. 2015 CMAME 293:1) removes that restriction by making contact
PARTICLE-TO-SURFACE: a particle is tested against a mesh FACE, not against a node it shares. That is
the scheme implemented here.

THE THREE MEASUREMENTS, each falsifying:
  1. momentum        sum of the contact forces on the particles + on the vertices must be zero to
                     machine precision. A one-way coupling -- what the spheroid has now -- fails this
                     by construction, and it is the cheapest way to catch a missing reaction.
  2. interpenetration how many particles end up behind the surface, and how deep, against the penalty
                     stiffness. A penalty method permits penetration; the question is how much, and
                     whether it stays bounded as the load rises (BFEMP, Li et al. 2022, forbids it
                     outright at the cost of an implicit solve).
  3. slip            the tangential velocity difference across the interface with friction on and off.
                     MPM's shared grid gives no-slip for free; if the two runs agree, the contact is
                     the grid's weld wearing a friction coefficient.

WHAT IS SIMPLIFIED, SO IT IS NOT MISTAKEN FOR GENERAL: the patch is flat and axis-aligned, so "which
face is this particle under" is an O(1) lookup rather than a BVH query. Everything else -- barycentric
distribution of the reaction, the regularised Coulomb law, the penalty in the normal direction -- is
the general form.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

OFF = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)])
LOG = os.path.join(_ROOT, "log", "okuda_ECM")
MESH_C = "#e8dcc0"
MAT_C = "#5b5b9c"
HOT_C = "#e0452b"
from matplotlib.colors import ListedColormap          # noqa: E402
sys.path.insert(0, _HERE)
import ecm_spec as _ES                                # noqa: E402
CMAP = ListedColormap(_ES.STRESS_COLORS)


def _bspline(x, inv_dx):
    base = (x * inv_dx - 0.5).floor()
    fx = x * inv_dx - base
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 0)
    return base.long(), fx, w


class MeshOnMatrix:
    """A triangulated patch with mass, resting on an MPM block, coupled particle-to-surface."""

    def __init__(self, nx=17, patch=0.42, z_mesh=0.585, block=(0.28, 0.28, 0.18, 0.72, 0.72, 0.58),
                 n_grid=64, ppc=4, E=400.0, nu=0.2, rho=1.0, m_vert=2.0e-4,
                 k_frac=0.15, mu=0.4, g=0.0, dt=1.0e-4, floor=None, floor_stick=False,
                 hole_r=0.0, walls=None, track_vm=False, dev="cuda:0"):
        self.dev, self.n_grid, self.dt, self.mu = dev, n_grid, dt, mu
        # THE RIM'S PRESCRIBED VELOCITY, as a VECTOR rather than a descent rate. The press is
        # (0, 0, press_v) and stays the default; a tangential drive -- the shear demo -- is the same
        # boundary condition pointed sideways, and writing it as a vector is what keeps the two runs
        # the same experiment with one thing changed rather than two rigs.
        self.drive = None
        # LATERAL WALLS, the same grid-velocity boundary condition as the floor, on x and y. Without
        # them a confined press is not confined: the material squeezes out from under the patch's rim
        # and the hole carries almost nothing. `walls` is (lo, hi) in box units.
        self.walls = walls
        self.track_vm = track_vm      # also colour by von Mises, for the loadings that are shear
        self.dx = 1.0 / n_grid; self.inv_dx = float(n_grid)
        self.g = g
        # --- the mesh: a flat lattice of vertices, two triangles per quad
        xs = torch.linspace(0.5 - patch / 2, 0.5 + patch / 2, nx)
        X, Y = torch.meshgrid(xs, xs, indexing="ij")
        self.V = torch.stack([X.reshape(-1), Y.reshape(-1),
                              torch.full((nx * nx,), float(z_mesh))], -1).to(dev)
        self.nx, self.x0, self.h = nx, float(xs[0]), float(xs[1] - xs[0])
        self.Vv = torch.zeros_like(self.V)
        self.m_v = m_vert
        # THE PATCH IS A MESH, so it has edges. Without them it is a cloud of independent points: the
        # first version pressed the rim and the interior did not follow, and gravity alone never
        # brought the two bodies together because the BLOCK settles faster than the patch descends
        # (measured: patch 0.5850 -> 0.5778 while the block's top went 0.5800 -> 0.5728). Four-
        # neighbour springs on the lattice, at rest at the seeded spacing.
        ii = torch.arange(nx * nx)
        e = []
        for step_, cond in ((nx, ii // nx < nx - 1), (1, ii % nx < nx - 1)):
            a_ = ii[cond]
            e.append(torch.stack([a_, a_ + step_], 1))
        self.E = torch.cat(e).to(dev)
        # A HOLE IN THE SURFACE. Vertices inside `hole_r` of the patch centre are removed from the
        # mesh entirely: the springs that touched them go, and every quad with a missing corner stops
        # being a contact face. The material under it then has nowhere to be pushed but through --
        # which is the geometry a proteolytic breach makes, and the one case where the epithelium and
        # the matrix touch each other directly rather than through the sheet.
        rr = (self.V[:, :2] - torch.tensor([0.5, 0.5], device=dev)).norm(dim=1)
        self.hole = rr < hole_r
        self.hole_r = hole_r
        if hole_r > 0:
            keep = ~(self.hole[self.E[:, 0]] | self.hole[self.E[:, 1]])
            self.E = self.E[keep]
            print(f"[mesh] hole of radius {hole_r:g} -- {int(self.hole.sum())} vertices and "
                  f"{int((~keep).sum())} springs removed", flush=True)
        self.l0 = (self.V[self.E[:, 1]] - self.V[self.E[:, 0]]).norm(dim=1)
        self.k_mesh = 0.02 * m_vert / (dt * dt)      # the same ceiling discipline as the penalty
        # PRESS DEEP, THEN LIFT. At 0.25 over 600 steps the patch descended 0.015 box units -- one
        # grid cell -- so the movie was a static picture with a number changing on it. The indentation
        # has to be several cells before it reads as an indentation, and the lift is what makes it a
        # dynamic test rather than a static one: the block has to push back and recover.
        self.v_press = 1.0
        self.press_v = -self.v_press                  # set per frame by the caller
        self.cell_size = self.h                      # one mesh cell, for the resolution ratio
        # --- the matrix: a block of MPM material under it
        g_ = torch.Generator().manual_seed(0)
        lo = torch.tensor(block[:3]); hi = torch.tensor(block[3:])
        vol = float((hi - lo).prod())
        n = max(1, int(round(ppc * vol / self.dx ** 3)))
        self.x = (lo + torch.rand(n, 3, generator=g_) * (hi - lo)).to(dev)
        self.N = self.x.shape[0]
        self.v = torch.zeros_like(self.x)
        self.C = torch.zeros(self.N, 3, 3, device=dev)
        self.F = torch.eye(3, device=dev).expand(self.N, 3, 3).contiguous()
        self.vol = self.dx ** 3 / ppc
        self.m = rho * self.vol
        # THE PENALTY AS A FRACTION OF THE EXPLICIT CEILING, not as an absolute number. The contact
        # force enters the particle as a body force, so it is integrated at `dt` and is stable only
        # while dt*sqrt(k/m) < 1, i.e. k < m/dt^2. With m = rho*dx^3/ppc = 9.5e-7 and dt = 1e-4 that
        # ceiling is 95 -- and the first version of this file used k = 3e4, three hundred times past
        # it. The same mistake NaN'd the flat sheet in `flat_mpm` and is recorded in HANDOVER.md; it
        # is written as a fraction here so it cannot be made a fourth time.
        self.k_ceiling = self.m / (dt * dt)
        self.k_pen = k_frac ** 2 * self.k_ceiling
        # NO GRAVITY. With it on, nothing holds the block up -- its floor is twenty grid cells below
        # its underside -- so it fell at exactly the rate the patch descended and the two never met
        # (measured: block top 0.5800 -> 0.5728 while the patch went 0.5850 -> 0.5778). The experiment
        # is about the interface, and the load it needs is the prescribed press, not a body force.
        self.d_expect = self.k_pen and (self.m * 0.0)
        self.mu_l = E / (2 * (1 + nu)); self.la = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.off = OFF.to(dev)
        # A RIGID PLANE UNDER THE BLOCK, as a GRID boundary condition rather than a particle clamp:
        # zeroing the downward component of the grid velocity is how MPM states an obstacle, and it is
        # the same mechanism the stock "material pours around a sphere" demo uses. Without it the block
        # simply translates away from the press and nothing is compressed -- with it, the material has
        # nowhere to go but sideways, which is what makes the compression visible.
        self.floor = floor
        self.floor_stick = floor_stick
        self.res = dict(momentum=[], n_pen=[], depth=[], slip=[], z_mesh=[], f_norm=[],
                        height=[], vmax=[])

    # ---- the interface -------------------------------------------------------------------------
    def _contact(self):
        """Particle-to-surface penalty + regularised Coulomb friction, with the reaction distributed
        to the face's vertices by barycentric weight. Returns (force on particles, force on vertices).
        """
        dev = self.dev
        # which quad each particle sits under: O(1) because the patch is a regular lattice, and that
        # is the ONLY thing here that is not general.
        gi = ((self.x[:, 0] - self.x0) / self.h).floor().long()
        gj = ((self.x[:, 1] - self.x0) / self.h).floor().long()
        inside = (gi >= 0) & (gj >= 0) & (gi < self.nx - 1) & (gj < self.nx - 1)
        if self.hole_r > 0:
            gi_c = gi.clamp(0, self.nx - 2); gj_c = gj.clamp(0, self.nx - 2)
            c0 = gi_c * self.nx + gj_c
            gone = (self.hole[c0] | self.hole[c0 + 1] | self.hole[c0 + self.nx]
                    | self.hole[c0 + self.nx + 1])
            inside = inside & ~gone
        fp = torch.zeros_like(self.x)
        fv = torch.zeros_like(self.V)
        if not bool(inside.any()):
            return fp, fv, 0, 0.0, 0.0
        idx = torch.nonzero(inside).squeeze(1)
        gi, gj = gi[idx], gj[idx]
        c00 = gi * self.nx + gj
        corners = torch.stack([c00, c00 + self.nx, c00 + 1, c00 + self.nx + 1], 1)
        # bilinear weights inside the quad -- the barycentric distribution, on a quad
        u = (self.x[idx, 0] - (self.x0 + gi * self.h)) / self.h
        w_ = (self.x[idx, 1] - (self.x0 + gj * self.h)) / self.h
        W = torch.stack([(1 - u) * (1 - w_), u * (1 - w_), (1 - u) * w_, u * w_], 1)
        z_face = (self.V[corners, 2] * W).sum(1)
        n_hat = torch.tensor([0.0, 0.0, 1.0], device=dev)          # the flat patch's normal
        depth = (self.x[idx, 2] - z_face).clamp_min(0.0)           # >0 = the particle is inside
        hit = depth > 0
        if not bool(hit.any()):
            return fp, fv, 0, 0.0, 0.0
        idx, W, corners, depth = idx[hit], W[hit], corners[hit], depth[hit]
        f_n = self.k_pen * depth
        # relative tangential velocity, mesh minus particle, at the contact point
        v_mesh = (self.Vv[corners] * W[:, :, None]).sum(1)
        dv = self.v[idx] - v_mesh
        dv_t = dv - (dv @ n_hat)[:, None] * n_hat
        speed = dv_t.norm(dim=1)
        # REGULARISED COULOMB: the force saturates at mu*f_n and is linear below it, so a static
        # contact does not chatter. `eps` is a velocity, not a fudge: it is the slip below which the
        # contact is treated as stuck.
        eps = 1.0e-3
        f_t_mag = torch.minimum(self.mu * f_n, self.mu * f_n * speed / eps)
        f_t = -f_t_mag[:, None] * dv_t / speed.clamp_min(1e-12)[:, None]
        f_par = -f_n[:, None] * n_hat + f_t                        # on the particle
        fp.index_add_(0, idx, f_par)
        # THE REACTION, distributed by the same weights. This is the line whose absence the momentum
        # measurement exists to catch.
        fv.index_add_(0, corners.reshape(-1), (-f_par[:, None, :] * W[:, :, None]).reshape(-1, 3))
        return fp, fv, int(hit.sum()), float(depth.max()), float(speed.mean())

    def step(self):
        dev, ng, dt, inv_dx = self.dev, self.n_grid, self.dt, self.inv_dx
        fp, fv, npen, dmax, slip = self._contact()
        resid = float((fp.sum(0) + fv.sum(0)).norm())
        scale = float(fp.abs().sum()) + 1e-30
        self.res["momentum"].append(resid / scale)
        self.res["n_pen"].append(npen); self.res["depth"].append(dmax)
        self.res["slip"].append(slip); self.res["z_mesh"].append(float(self.V[:, 2].mean()))
        self.res["f_norm"].append(float(fp.norm(dim=1).sum()))
        # --- the mesh: explicit, with mass, gravity and the reaction; its rim is pinned so the patch
        # presses rather than falls away
        dvec = self.V[self.E[:, 1]] - self.V[self.E[:, 0]]
        L = dvec.norm(dim=1).clamp_min(1e-12)
        fe = (self.k_mesh * (L - self.l0) / L)[:, None] * dvec
        f_mesh = torch.zeros_like(self.V)
        f_mesh.index_add_(0, self.E[:, 0], fe)
        f_mesh.index_add_(0, self.E[:, 1], -fe)
        a_v = (fv + f_mesh) / self.m_v + torch.tensor([0.0, 0.0, -self.g], device=dev)
        self.Vv = self.Vv + dt * a_v
        # THE RIM IS DRIVEN, not pinned. Gravity alone cannot bring the two together -- the block
        # settles out from under the patch -- so the press is prescribed and the springs carry it to
        # the interior, which is the loading an epithelium applies to a matrix.
        ii = torch.arange(self.nx * self.nx, device=dev)
        rim = (ii % self.nx == 0) | (ii % self.nx == self.nx - 1) | (ii < self.nx) \
            | (ii >= self.nx * (self.nx - 1))
        self.Vv[rim] = (torch.as_tensor(self.drive, dtype=torch.float32, device=dev)
                        if self.drive is not None
                        else torch.tensor([0.0, 0.0, float(self.press_v)], device=dev))
        self.V = self.V + dt * self.Vv
        # --- the matrix: one MLS-MPM cycle, with the contact force as a body force
        gm = torch.zeros(ng ** 3, device=dev); gmv = torch.zeros(ng ** 3, 3, device=dev)
        U, S, Vh = torch.linalg.svd(self.F)
        R = U @ Vh
        J = torch.linalg.det(self.F).clamp_min(1e-6)
        P = (2 * self.mu_l * (self.F - R) @ self.F.transpose(1, 2)
             + self.la * ((J - 1) * J)[:, None, None] * torch.eye(3, device=dev))
        if self.track_vm:
            # VON MISES OF THE KIRCHHOFF STRESS, sigma_vm = sqrt(3/2 |dev tau|) / J. `P` above is
            # tau for the fixed-corotated model, so this costs an outer product and no second SVD.
            # |J-1| is the right colour for a press -- the material is squeezed and cannot leave --
            # but a DRAG is deviatoric: it changes shape at constant volume, so a volumetric colour
            # shows nothing where the shear band is.
            tr = P.diagonal(dim1=1, dim2=2).sum(1) / 3.0
            dvi = P - tr[:, None, None] * torch.eye(3, device=dev)
            self.vm = torch.sqrt(1.5 * (dvi * dvi).sum((1, 2))) / J
        affine = self.m * self.C - (4 * inv_dx * inv_dx * dt * self.vol) * P
        fb = fp + torch.tensor([0.0, 0.0, -self.g * self.m], device=dev)
        base, fx, w = _bspline(self.x, inv_dx)
        for o in self.off:
            ww = w[o[0], :, 0] * w[o[1], :, 1] * w[o[2], :, 2]
            dpos = (o.float() - fx) * self.dx
            idx = ((base[:, 0] + o[0]).clamp(0, ng - 1) * ng * ng
                   + (base[:, 1] + o[1]).clamp(0, ng - 1) * ng
                   + (base[:, 2] + o[2]).clamp(0, ng - 1))
            gm.index_add_(0, idx, ww * self.m)
            gmv.index_add_(0, idx, ww[:, None] * (self.m * self.v
                                                  + torch.einsum('nij,nj->ni', affine, dpos)
                                                  + dt * fb))
        gv = gmv / gm.clamp_min(1e-12)[:, None]
        if self.floor is not None:
            kz = torch.arange(ng, device=dev).repeat(ng * ng)
            below = (kz.float() * self.dx) < self.floor
            if self.floor_stick:
                # NO-SLIP, and the difference is the whole shear experiment. A floor that only stops
                # the DOWNWARD component lets the block translate bodily under a tangential load --
                # measured: the whole column moved with the patch and the strain never appeared --
                # so what looked like "the drag does nothing" was the boundary condition, not the
                # contact. Zeroing all three components pins the bottom, and the block then has to
                # shear between a pinned base and a dragged top.
                gv[below] = 0.0
            else:
                gv[below, 2] = gv[below, 2].clamp_min(0.0)
        if self.walls is not None:
            lo, hi = self.walls
            ii_ = torch.arange(ng, device=dev)
            ix = ii_.repeat_interleave(ng * ng).float() * self.dx
            iy = ii_.repeat_interleave(ng).repeat(ng).float() * self.dx
            gv[ix < lo, 0] = gv[ix < lo, 0].clamp_min(0.0)
            gv[ix > hi, 0] = gv[ix > hi, 0].clamp_max(0.0)
            gv[iy < lo, 1] = gv[iy < lo, 1].clamp_min(0.0)
            gv[iy > hi, 1] = gv[iy > hi, 1].clamp_max(0.0)
        newv = torch.zeros_like(self.v); newC = torch.zeros_like(self.C)
        for o in self.off:
            ww = w[o[0], :, 0] * w[o[1], :, 1] * w[o[2], :, 2]
            dpos = (o.float() - fx) * self.dx
            idx = ((base[:, 0] + o[0]).clamp(0, ng - 1) * ng * ng
                   + (base[:, 1] + o[1]).clamp(0, ng - 1) * ng
                   + (base[:, 2] + o[2]).clamp(0, ng - 1))
            gg = gv[idx]
            newv += ww[:, None] * gg
            newC += (4 * inv_dx) * ww[:, None, None] * torch.einsum('ni,nj->nij', gg, dpos * inv_dx)
        self.v, self.C = newv, newC
        self.x = self.x + dt * self.v
        self.F = (torch.eye(3, device=dev) + dt * self.C) @ self.F
        # THE STRESS THE COMPRESSION PRODUCES, per particle, in the measure the rest of the prototype
        # colours by: |J - 1|, the volumetric strain (`ecm_stress`, measure="vol"). Under a press
        # against a rigid floor this is the quantity the experiment is about -- the material cannot
        # leave, so it is squeezed, and J < 1 is that squeeze.
        self.J = torch.linalg.det(self.F)
        zlo = 2 * self.dx if self.floor is None else self.floor
        below = self.x[:, 2] < zlo
        self.v[below, 2] = self.v[below, 2].clamp_min(0.0)
        self.x[below, 2] = zlo
        z = self.x[:, 2]
        self.res["height"].append(float(torch.quantile(z, 0.98) - torch.quantile(z, 0.02)))
        self.res["vmax"].append(float(self.v.norm(dim=1).max()))


def render(rig, frames, d, every=8, fps=15, n_draw=6000):
    # TWO PASSES, because the colour scale cannot be known before the run. The physics is stepped
    # once and the drawn frames are kept; the stress scale is then one p99 over all of them, fixed for
    # the whole movie -- the same convention as every other artefact here, and the reason 02's first
    # movie was a saturated block.
    st = max(1, rig.N // n_draw)
    strip_at = set(np.round(np.linspace(0, frames - 1, 8)).astype(int).tolist())
    kept, strip = [], []
    for t in range(frames):
        rig.press_v = -rig.v_press if t < 0.6 * frames else +0.7 * rig.v_press
        rig.step()
        if t % every and t not in strip_at:
            continue
        X = rig.x.detach().cpu().numpy(); V = rig.V.detach().cpu().numpy()
        S = np.abs(rig.J.detach().cpu().numpy() - 1.0)
        kept.append((t, X, V, S))
    allS = np.concatenate([k[3][::7] for k in kept])
    s_hi = float(np.percentile(allS[np.isfinite(allS)], 99)) or 1.0
    print(f"[{os.path.basename(d)}] stress colour full-scale |J-1| = {s_hi:.4g} (p99 over the run)",
          flush=True)
    fig = plt.figure(figsize=(11.2, 5.6), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": "03_mesh_contact"})
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for (t, X, V, S) in kept:
            col = np.clip(S / s_hi, 0, 1)
            zf = V[:, 2].reshape(rig.nx, rig.nx)
            pen = X[:, 2] > np.interp(X[:, 0], np.linspace(V[:, 0].min(), V[:, 0].max(), rig.nx),
                                      zf.mean(axis=1))
            fig.clf()
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            ax.scatter(X[::st, 0], X[::st, 1], X[::st, 2], s=3.5, c=col[::st], cmap=CMAP,
                       vmin=0, vmax=1, marker=".", linewidths=0, alpha=0.85, depthshade=False)
            if rig.hole_r > 0:                       # a hole is drawn by not drawing it
                zf = zf.copy()
                zf.reshape(-1)[rig.hole.cpu().numpy()] = np.nan
            ax.plot_wireframe(V[:, 0].reshape(rig.nx, rig.nx), V[:, 1].reshape(rig.nx, rig.nx), zf,
                              color=MESH_C, lw=0.7)
            ax.set_xlim(0.25, 0.75); ax.set_ylim(0.25, 0.75)
            ax.set_zlim(0.13 if rig.floor is not None else 0.30, 0.66)
            ax.set_box_aspect((1, 1, 0.9)); ax.view_init(elev=14, azim=-62)
            if "--nolabel" not in sys.argv:
                ax.text2D(0.02, 0.96, f"03_mesh_contact   frame {t}\n"
                                      f"mesh cell {rig.cell_size/rig.dx:.2f} dx   "
                                      f"{rig.res['n_pen'][-1]} particles in contact",
                           transform=ax.transAxes, color="white", fontsize=10, va="top")
            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            sl = np.abs(X[:, 1] - 0.5) < 0.02
            a2.scatter(X[sl][:, 0], X[sl][:, 2], s=7, c=col[sl], cmap=CMAP, vmin=0, vmax=1,
                       marker=".", linewidths=0)
            mid = V[:, 1].reshape(rig.nx, rig.nx)[0].argmin() * 0 + rig.nx // 2
            a2.plot(V[:, 0].reshape(rig.nx, rig.nx)[:, mid], zf[:, mid], "-", color=MESH_C, lw=1.6)
            if rig.floor is not None:
                a2.plot([0.25, 0.75], [rig.floor, rig.floor], "-", color="#9aa0a6", lw=2.5)
            a2.set_xlim(0.25, 0.75); a2.set_ylim(0.13, 0.66); a2.set_aspect("equal"); a2.axis("off")
            if "--nolabel" not in sys.argv:
                a2.text(0.02, 0.98, f"section, coloured by $|J-1|$ up to {s_hi:.3g}",
                        transform=a2.transAxes, color="white", fontsize=10, va="top")
            wri.grab_frame()
            if t in strip_at:
                strip.append((t, X.copy(), V.copy(), col.copy()))
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)
    # strip.png, always
    figs = plt.figure(figsize=(3.2 * len(strip), 3.4), facecolor="black")
    for i, (t, X, V, col) in enumerate(strip):
        a = figs.add_subplot(1, len(strip), i + 1, facecolor="black")
        sl = np.abs(X[:, 1] - 0.5) < 0.02
        a.scatter(X[sl][:, 0], X[sl][:, 2], s=5, c=col[sl], cmap=CMAP, vmin=0, vmax=1, marker=".",
                  linewidths=0)
        zf = V[:, 2].reshape(rig.nx, rig.nx)
        a.plot(V[:, 0].reshape(rig.nx, rig.nx)[:, rig.nx // 2], zf[:, rig.nx // 2], "-",
               color=MESH_C, lw=1.6)
        a.set_xlim(0.25, 0.75); a.set_ylim(0.35, 0.66); a.set_aspect("equal"); a.axis("off")
        a.text(0.03, 0.96, f"frame {t}", transform=a.transAxes, color="white", fontsize=11, va="top")
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=110, facecolor="black")
    plt.close(figs)


def metrics_png(res, res_free, d, k_pen):
    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.2), facecolor="white")
    t = np.arange(len(res["momentum"]))
    ax[0].semilogy(t, np.maximum(res["momentum"], 1e-18), color="#2b6cb0", lw=1.2)
    ax[0].axhline(1e-15, color="#999", ls="--", lw=0.8)
    ax[0].set_ylabel(r"$|\sum f_{\rm part} + \sum f_{\rm vert}| / \sum|f|$")
    ax[0].set_title(f"momentum: max {max(res['momentum']):.1e}", fontsize=9)
    ax[1].plot(t, res["n_pen"], color="#e08a2e", lw=1.3, label="particles in contact")
    a1 = ax[1].twinx()
    a1.plot(t, np.asarray(res["depth"]) / (1 / 64), color="#e0452b", lw=1.2, label="max depth")
    a1.set_ylabel("max penetration (grid cells)", color="#e0452b")
    ax[1].set_ylabel("particles behind the surface")
    ax[1].set_title(f"penetration at $k$ = {k_pen:.0e}: max "
                    f"{max(res['depth'])/(1/64):.2f} cells", fontsize=9)
    ax[2].plot(t, res["slip"], color="#1f8a5c", lw=1.4, label=r"friction $\mu$ = 0.4")
    ax[2].plot(np.arange(len(res_free["slip"])), res_free["slip"], color="#7bbf6a", lw=1.4, ls="--",
               label=r"$\mu$ = 0")
    ax[2].set_ylabel("tangential slip at the interface")
    ax[2].set_title("if these coincide, the contact is the grid's weld", fontsize=9)
    ax[2].legend(fontsize=7.5, frameon=False)
    for a in ax:
        a.set_xlabel("frame"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(d, "metrics.png"), dpi=150, facecolor="white")
    plt.close(fig)


def main():
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 600
    floor = 0.18 if ("--floor" in sys.argv or "--hole" in sys.argv) else None
    hole = 0.06 if "--hole" in sys.argv else 0.0
    d = os.path.join(LOG, "03c_mesh_contact_hole" if hole else
                     ("03b_mesh_contact_floor" if floor else "03_mesh_contact"))
    os.makedirs(d, exist_ok=True)
    rig = MeshOnMatrix(dev=dev, floor=floor, hole_r=hole)
    print(f"[03_mesh_contact] penalty k = {rig.k_pen:.3g} = 5% of the explicit ceiling "
          f"{rig.k_ceiling:.3g}; press {rig.v_press} box units per unit time", flush=True)
    render(rig, frames, d)
    # THE FRICTIONLESS CONTROL, same rig, same seed: without it "the contact resists sliding" is a
    # claim about a number nobody compared with anything.
    free = MeshOnMatrix(dev=dev, mu=0.0, floor=floor, hole_r=hole)
    for t in range(frames):
        free.press_v = -free.v_press if t < 0.6 * frames else +0.7 * free.v_press
        free.step()
    metrics_png(rig.res, free.res, d, rig.k_pen)
    out = dict(frames=frames, mesh_cell_over_dx=rig.cell_size / rig.dx,
               particles=rig.N, k_penalty=rig.k_pen, mu=rig.mu,
               momentum_residual_max=float(max(rig.res["momentum"])),
               penetration_max_cells=float(max(rig.res["depth"]) * 64),
               contacts_max=int(max(rig.res["n_pen"])),
               # OVER THE FRAMES THAT HAVE A CONTACT, not the last hundred. With the patch lifting
               # at 60% of the run the last hundred frames have no contact at all, so both numbers
               # came back 0.000 and the comparison said nothing -- a metric averaged over the wrong
               # window reads as a null.
               slip_friction=float(np.mean([v for v, n in zip(rig.res["slip"], rig.res["n_pen"])
                                            if n > 0] or [0.0])),
               slip_frictionless=float(np.mean([v for v, n in zip(free.res["slip"], free.res["n_pen"])
                                                if n > 0] or [0.0])),
               contact_frames=int(sum(1 for n in rig.res["n_pen"] if n > 0)),
               floor=floor, hole_r=hole,
               height_start=float(rig.res["height"][0]), height_min=float(min(rig.res["height"])),
               compression=float(1 - min(rig.res["height"]) / max(rig.res["height"][0], 1e-9)),
               series={k: [float(x) for x in v[::4]] for k, v in rig.res.items()})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    import yaml
    yaml.safe_dump(dict(
        what="particle-to-surface contact between a triangulated patch and an MPM block",
        scheme="ICFEMP-style (Chen et al. 2015 CMAME 293:1); grid-node coupling (CFEMP, Lian et al. "
               "2011) needs mesh/grid size consistency, which this prototype violates",
        simplification="flat axis-aligned patch, so face lookup is O(1); a general mesh needs a BVH",
        measures=["momentum residual", "penetration depth in grid cells", "slip with and without "
                  "friction"],
        rig=dict(mesh_cell_over_dx=rig.cell_size / rig.dx, n_grid=rig.n_grid, particles=rig.N,
                 k_penalty=rig.k_pen, mu=rig.mu, dt=rig.dt)),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{os.path.basename(d)}] momentum residual max {out['momentum_residual_max']:.2e}, "
          f"penetration max {out['penetration_max_cells']:.2f} cells, "
          f"slip {out['slip_friction']:.3e} with friction against {out['slip_frictionless']:.3e} "
          f"without, block compressed {100*out['compression']:.1f}% -> {d}", flush=True)


if __name__ == "__main__":
    main()
