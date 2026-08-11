#!/usr/bin/env python
"""test_05b_plaque -- the hemidesmosome as an EDGE SET, with its reaction and with slip.

    python test_05b_plaque.py [--device cuda:0] [--frames 401]  ->  log/okuda_ECM/05b_plaque/

WHY THIS RUN EXISTS. In the archived line the adhesion was first a field on the membrane and then a
column of MPM particles, and both lost the reaction: whatever force held the sheet out was applied to
the sheet and to nothing else, so the epithelium never felt what it was carrying. `note_spheroid_bm_ecm`
S10 says why that is a modelling error rather than an omission -- an integrin plaque is not a body, it
is a RELATION between two entities, and in Plexus2 a relation is an edge set carrying `.pre` and
`.post`. One operator on `plaque` returns a delta to both endpoints, so the force the sheet feels and
the force the cell feels cannot get out of step. This rig is that operator on the smallest geometry that
can falsify it.

IT WAITED FOR 05a ON PURPOSE. A coupling test is worthless on a sheet that launders strain: if the
membrane cannot report the deformation it is given, no number about what the adhesion did to it means
anything. 05a certified the strain measure (rigid, dilation and affine maps to 1e-8, 1e-8, 1e-15) and
measured fidelity 1.00 against the drive. Only then is this run interpretable.

WHAT IS DIFFERENT FROM THE ARCHIVED TETHER, in three lines of physics:
  * the anchor is a POINT ON A DEFORMING MESH, tracked barycentrically, not a frozen direction in
    space. A node tied to a direction can never slide; a real sheet does.
  * the tangential law is FRICTION AGAINST RELATIVE VELOCITY, xi (v_bm - v_epi)_par, so the sheet can
    slip and the slip rate is a material property rather than zero by construction.
  * the normal law is a spring at rest length l0 ~ 0.3 T, and l0 -- not a force balance -- is what sets
    the standoff. The archived line proved the standoff was a TRACKING LAG to 3e-5 box units, i.e. a
    number that fell out of the integrator; here it is a length that was measured in a cell.

THE SIX MEASUREMENTS, each able to come back wrong:
  1. MOMENTUM, as bookkeeping   |sum f_bm + sum f_epi| / sum|f|, every frame. A one-way coupling fails
                                this by construction and passes everything else, which is why it is
                                first.
  2. MOMENTUM, as motion        the same claim with the drive switched OFF: two free bodies joined by
                                pre-stretched plaques must leave their mobility-weighted centroid where
                                it was. Bookkeeping can be right while a sign is wrong somewhere that
                                only shows up as drift.
  3. THE STANDOFF               <l> over the plaques against l0, and against PLAQUE DENSITY. The
                                archived one-sided version was monotone and bad -- 100% anchored gave
                                -0.0082, 20% gave -0.0449, 5% gave -0.1190, the sheet sagging between
                                anchors. A two-sided plaque at the measured spacing Sigma ~ 7T has to
                                do better than that or the relation bought nothing.
  4. SLIP                       with the epithelium ROTATING under the sheet: xi = 0 against xi finite.
                                If the two agree, the friction law is decorative.
  5. RUPTURE                    the threshold is taken from the load distribution the nominal run
                                MEASURES, not from a round number. Run 127 was reported as a null for
                                the mechanism when in fact its threshold (0.02) sat above the largest
                                load in the whole run (0.017): a bond that never breaks is not evidence
                                about breaking.
  6. NO-PLAQUE CONTROL          with the edge set empty the sheet must not be carried at all.

WHAT IS NOT HERE. The epithelium is a driven icosphere, not the vertex model -- 05b is about the
relation, and `mesh_contact_ops.py` already certified contact against the real re-meshed tissue. No
matrix, no fibril, no secretion. And the plaques never re-seed: `plaque_seed` as a turnover operator
is 05c, so a bond that ruptures here is gone for good.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.animation import FFMpegWriter                           # noqa: E402
from matplotlib.colors import ListedColormap                           # noqa: E402
from matplotlib.collections import LineCollection                       # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection   # noqa: E402

import bm_ops as BM                                                     # noqa: E402
import ecm_spec as ES                                                   # noqa: E402
from test_05_sheet import SurfaceReplay, LOG, TISSUE, SCALE, UNITS, T_REAL_UM             # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

CMAP = ListedColormap(ES.STRESS_COLORS)
EPI_C = "#e8dcc0"
PLQ_C = "#e0452b"


# =============================================================================================
#  the edge set
# =============================================================================================
def seed_plaques(u_bm, V_epi, F_epi, fraction=1.0, seed=0):
    """Attach each (selected) bm node to the epithelial FACE its own direction points into, and record
    the barycentric weights of that intersection.

    THE LOOKUP IS BRUTE FORCE ON PURPOSE. `mesh_contact_ops` needed a certified accelerator because it
    runs 200,000 particles against 19,000 faces every substep; here it is 2,562 nodes against 1,280
    faces ONCE, at seeding, so the version that cannot be wrong is also the affordable one. Barycentric
    coordinates of the ray u in the basis of the face's three (unit) vertices: all three non-negative
    is the containment test, exactly, with no tolerance to tune.
    """
    n = u_bm.shape[0]
    keep = torch.arange(n, device=u_bm.device)
    if fraction < 1.0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        m = torch.randperm(n, generator=g)[:max(1, int(round(fraction * n)))]
        keep = m.to(u_bm.device).sort().values
    tri = V_epi[F_epi]                                     # (nf,3,3), rows are the three vertices
    Minv = torch.linalg.inv(tri.transpose(1, 2))           # solves  u = a*V0 + b*V1 + c*V2
    bc = torch.einsum("fij,nj->nfi", Minv, u_bm[keep])     # (n,nf,3)
    ok = (bc >= -1e-12).all(-1)
    face = torch.argmax(ok.to(torch.int8), dim=1)
    hit = ok.any(1)
    face, keep = face[hit], keep[hit]
    w = torch.gather(bc[hit], 1, face[:, None, None].expand(-1, 1, 3)).squeeze(1)
    w = w / w.sum(1, keepdim=True)
    return keep, face, w, int((~hit).sum())


class Plaques:
    """`plaque`: an edge set cell -> bm. One call returns a delta to BOTH endpoints.

    The normal law is a spring at rest length l0 -- the linkage's own length, a material property
    (Kanchanawong 2010 puts the integrin layer ~30 nm under a plaque ~350 nm across) -- and the
    tangential law is friction against the relative velocity, so the sheet slides at a rate rather than
    being pinned to a direction.
    """

    def __init__(self, node, face, w, l0, kn, xi, break_load=None):
        self.node, self.face, self.w = node, face, w
        self.l0, self.kn, self.xi = float(l0), float(kn), float(xi)
        self.break_load = break_load
        self.bound = torch.ones(node.shape[0], dtype=torch.bool, device=node.device)
        self.load = torch.zeros(node.shape[0], dtype=w.dtype, device=node.device)

    def geometry(self, x_bm, x_epi, v_epi, F_epi, centre=None):
        """The attachment point and the normal force -- everything that does not need the sheet's
        velocity, which the tangential law has to solve for rather than read.

        THE NORMAL DIRECTION IS THE EPITHELIAL FACE'S, NOT THE LINE BETWEEN THE TWO POINTS, and that is
        a correctness fix rather than a refinement. With `n_hat` taken along `x_bm - p`, the spring's
        equilibrium is |x_bm - p| = l0 with NO PREFERRED SIDE: a sheet that has sunk through the
        epithelium sits at the same energy as one resting on it, and nothing pushes it back out. 05e
        measured the consequence -- the sheet started 5.96e-4 OUTSIDE (= l0, correct) and ended
        7.7e-3 INSIDE, thirteen l0 through the surface -- and the unsigned standoff metric reported it
        as +4.0e-3, i.e. as though the sheet were being held out. Projecting onto the outward face
        normal gives the offset a SIGN, so `d < l0` (including `d < 0`) is restored outward.
        """
        vf = F_epi[self.face]                                     # (p,3) vertex ids of each face
        tri = x_epi[vf]
        p = (tri * self.w[:, :, None]).sum(1)                     # the attachment point, tracked
        vp = (v_epi[vf] * self.w[:, :, None]).sum(1)
        nh = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
        nh = nh / nh.norm(dim=1, keepdim=True).clamp_min(1e-30)
        if centre is not None:                                    # orient outward, once per call
            nh = nh * torch.sign(((p - centre) * nh).sum(1, keepdim=True)).clamp(min=-1.0, max=1.0)
        dvec = x_bm[self.node] - p
        d = (dvec * nh).sum(1)                                    # SIGNED offset along the normal
        f_n = -self.kn * (d - self.l0)[:, None] * nh * self.bound[:, None]
        self.load = (self.kn * (d - self.l0)).abs() * self.bound
        self.ell = d                                              # signed: negative = inside the epi
        return vf, nh, vp, f_n

    def scatter(self, f, vf, x_bm, x_epi):
        """The reaction, and the ONLY place it is written. The same vector is added to the bm node and
        subtracted, barycentrically, from the epithelial face's three vertices -- which is what writing
        the adhesion as one operator on an edge set buys, and what two operators on two sets lose."""
        fb = torch.zeros_like(x_bm)
        fe = torch.zeros_like(x_epi)
        fb.index_add_(0, self.node, f)
        fe.index_add_(0, vf.reshape(-1), (-f[:, None, :] * self.w[:, :, None]).reshape(-1, 3))
        return fb, fe

    def rupture(self):
        if self.break_load is None:
            return 0
        gone = self.bound & (self.load > self.break_load)
        self.bound = self.bound & ~gone
        return int(gone.sum())


# =============================================================================================
#  the rig
# =============================================================================================
class Rig05b:
    def __init__(self, subdiv=4, subdiv_epi=3, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, xi=0.0,
                 l0=6.0e-4, fraction=1.0, zeta=20.0, s_target=1.0, refresh=10, tau_r=0.0,
                 k_drive=50.0, omega=0.0, driven=True, break_load=None, dev="cuda:0",
                 dtype=torch.float64):
        self.dev, self.dtype, self.driven, self.omega = dev, dtype, driven, float(omega)
        Vb, Fb, Eb = BM.icosphere(subdiv, device=dev, dtype=dtype)
        Ve, Fe, Ee = BM.icosphere(subdiv_epi, device=dev, dtype=dtype)
        self.rep = SurfaceReplay(Vb, dev=dev, dtype=dtype)
        self.rep_e = SurfaceReplay(Ve, dev=dev, dtype=dtype)
        self.c = torch.tensor([0.5, 0.5, 0.5], device=dev, dtype=dtype)
        # -- the epithelium: a mesh with a mobility, driven toward the recorded surface. It is not
        # pinned, because a pinned body absorbs any reaction silently and measurement 2 exists to stop
        # that being invisible.
        self.u_epi, self.F_epi = Ve, Fe
        self.x_epi = self.c + Ve * self.rep_e.R(0)[:, None]
        self.v_epi = torch.zeros_like(self.x_epi)
        self.k_drive, self.M_epi = float(k_drive), 1.0
        # -- the sheet, seeded l0 outside the epithelium's own surface
        self.sheet = BM.Sheet(subdiv=subdiv, R0=1.0, E=E, thickness=thickness, nu=nu, tau_r=tau_r,
                              dev=dev, dtype=dtype)
        self.sheet.reseed(self.c + Vb * (self.rep.R(0) + l0)[:, None])
        self.u_bm = Vb
        node, face, w, missed = seed_plaques(Vb, Ve, Fe, fraction=fraction)
        self.plq = Plaques(node, face, w, l0, kn, xi, break_load=break_load)
        self.n_plaque, self.missed = node.shape[0], missed
        self.lam_el, self._pv = self.sheet.spectral_rate(return_vec=True)
        self.lam_el_seed = self.lam_el
        self.sheet.M = float(zeta) / (self.lam_el + kn)
        self.s_target, self.refresh = float(s_target), int(refresh)
        self.n_sub = self._nsub()
        self.res = {k: [] for k in ("momentum", "standoff", "ell_mean", "ell_p99", "lam_geo",
                                    "lam_el", "R_epi", "R_bm", "bound", "slip", "load_p50",
                                    "load_p99", "drift", "n_sub", "epi_track", "twist")}

    def _nsub(self):
        # xi enters the EPITHELIUM's bound but not the sheet's: the sheet solves its tangential law
        # implicitly (see `frame`), the epithelium receives the resulting force explicitly.
        a = self.sheet.M * (self.lam_el + self.plq.kn)
        b = self.M_epi * (self.k_drive + self.plq.kn + self.plq.xi)
        return max(1, int(math.ceil(max(a, b) / self.s_target)))

    def _epi_anchor(self, t):
        R = self.rep_e.R(t)
        u = self.u_epi
        if self.omega:
            # THE TANGENTIAL DRIVE. The epithelium turns under the sheet, which is the only loading
            # that can tell a friction law from a weld: a purely radial drive has no tangential
            # component for xi to act on, so a slip measurement under it is a measurement of zero.
            th = self.omega * t
            Rz = torch.tensor([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0],
                               [0.0, 0.0, 1.0]], device=self.dev, dtype=self.dtype)
            u = u @ Rz.T
        return self.c + u * R[:, None]

    def frame(self, t):
        if t % self.refresh == 0 and torch.isfinite(self.sheet.x).all():
            self.lam_el, self._pv = self.sheet.spectral_rate(iters=25, v0=self._pv, return_vec=True)
            self.n_sub = self._nsub()
        dt = 1.0 / self.n_sub
        a_epi = self._epi_anchor(t) if self.driven else None
        mom = 0.0
        M = self.sheet.M
        for _ in range(self.n_sub):
            vf, nh, vp, f_n = self.plq.geometry(self.sheet.x, self.x_epi, self.v_epi,
                                                      self.F_epi, centre=self.c)
            # the sheet's velocity if the plaques exerted their NORMAL force only
            f_el = self.sheet.elastic_force(self.sheet.x)
            fb_n = torch.zeros_like(self.sheet.x)
            fb_n.index_add_(0, self.plq.node, f_n)
            v_prov = M * (f_el + fb_n)
            # THE TANGENTIAL LAW, SOLVED RATHER THAN EVALUATED. xdot_par = M(f_par + xi vp_par) implies
            # xdot_par (1 + M xi) = xdot_prov_par + M xi vp_par, which is exact, unconditionally stable
            # in xi, and tends to the no-slip limit xdot -> vp as xi -> infinity. Evaluating -xi*v
            # explicitly instead is a stiffness of xi/dt that RISES as the substep shrinks, and it took
            # the first version of this rig to NaN on frame 0.
            vpr = v_prov[self.plq.node]
            par = lambda a: a - (a * nh).sum(1, keepdim=True) * nh                      # noqa: E731
            xi_eff = self.plq.xi * self.plq.bound.to(nh.dtype)
            v_par = (par(vpr) + (M * xi_eff)[:, None] * par(vp)) / (1.0 + M * xi_eff)[:, None]
            f_t = xi_eff[:, None] * (par(vp) - v_par)
            fb, fe = self.plq.scatter(f_n + f_t, vf, self.sheet.x, self.x_epi)
            # MEASUREMENT 1, INSIDE the substep and on the adhesion's OWN force pair, not on the
            # total: with the drive's force included the number stops meaning "the reaction is there".
            mom = max(mom, float((fb.sum(0) + fe.sum(0)).norm()) /
                      (float((f_n + f_t).norm(dim=1).sum()) + 1e-300))
            dxb = dt * v_prov
            dxb.index_add_(0, self.plq.node, dt * (v_par - par(vpr)))
            self.sheet.advance(dxb, dt)
            f_epi = fe + (self.k_drive * (a_epi - self.x_epi) if self.driven else 0.0)
            dxe = dt * self.M_epi * f_epi
            self.v_epi = dxe / dt
            self.x_epi = self.x_epi + dxe
            self.plq.rupture()
        self._record(t, mom)

    def _record(self, t, mom):
        l1, _ = self.sheet.stretch_geo()
        e1, _ = self.sheet.stretch_elastic()
        r_bm = (self.sheet.x - self.c).norm(dim=1)
        r_ep = (self.x_epi - self.c).norm(dim=1)
        b = self.plq.bound
        ell = self.plq.ell[b] if bool(b.any()) else self.plq.ell[:1] * 0
        self.res["momentum"].append(mom)
        self.res["ell_mean"].append(float(ell.mean()))
        self.res["ell_p99"].append(float(torch.quantile(ell, 0.99)) if ell.numel() > 1 else 0.0)
        self.res["standoff"].append(float(ell.mean() - self.plq.l0))
        self.res["lam_geo"].append(float(l1.mean()))
        self.res["lam_el"].append(float(e1.mean()))
        self.res["R_bm"].append(float(r_bm.mean()))
        self.res["R_epi"].append(float(r_ep.mean()))
        self.res["bound"].append(float(b.float().mean()))
        self.res["load_p50"].append(float(torch.quantile(self.plq.load[b], 0.5))
                                    if bool(b.any()) else 0.0)
        self.res["load_p99"].append(float(torch.quantile(self.plq.load[b], 0.99))
                                    if bool(b.any()) else 0.0)
        self.res["n_sub"].append(int(self.n_sub))
        # the twist of the sheet about z, against the twist the epithelium was driven through: the
        # slip measurement, as an angle rather than as a velocity, so it accumulates and can be seen.
        u_now = (self.sheet.x - self.c)
        u_now = u_now / u_now.norm(dim=1, keepdim=True).clamp_min(1e-30)
        ang = torch.atan2(u_now[:, 1], u_now[:, 0]) - torch.atan2(self.u_bm[:, 1], self.u_bm[:, 0])
        ang = torch.atan2(torch.sin(ang), torch.cos(ang))
        w = (1.0 - self.u_bm[:, 2] ** 2)              # weight by distance from the rotation axis
        self.res["twist"].append(float((ang * w).sum() / w.sum()))
        self.res["slip"].append(float(self.omega * t - self.res["twist"][-1]))
        if self.driven:
            self.res["epi_track"].append(float((r_ep.mean() - self.rep_e.R(t).mean())))
            self.res["drift"].append(0.0)
        else:
            # MEASUREMENT 2: with no drive, sum f = 0 means the MOBILITY-WEIGHTED centroid cannot
            # move. Written as sum(x_i / M_i) so it stays the invariant even when the two bodies do
            # not share a mobility.
            self.res["epi_track"].append(0.0)
            g = (self.sheet.x.sum(0) / self.sheet.M + self.x_epi.sum(0) / self.M_epi)
            if not hasattr(self, "_g0"):
                self._g0 = g.clone()
                self._gscale = float((self.sheet.x - self.c).norm(dim=1).sum() / self.sheet.M
                                     + (self.x_epi - self.c).norm(dim=1).sum() / self.M_epi)
            self.res["drift"].append(float((g - self._g0).norm()) / max(self._gscale, 1e-30))

    def alive(self):
        return bool(torch.isfinite(self.sheet.x).all() and torch.isfinite(self.x_epi).all())


def run(rig, frames, keep=None, label=""):
    keep, kept, t0 = (set() if keep is None else keep), [], time.time()
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[{label}] DIVERGED at frame {t}", flush=True)
            return kept, t
        if t in keep:
            l1, _ = rig.sheet.stretch_geo()
            kept.append((t, rig.sheet.x.float().cpu().numpy(), l1.float().cpu().numpy(),
                         rig.x_epi.float().cpu().numpy(),
                         rig.plq.node[rig.plq.bound].cpu().numpy(),
                         (rig.x_epi[rig.F_epi[rig.plq.face]] * rig.plq.w[:, :, None]).sum(1)
                         [rig.plq.bound].float().cpu().numpy()))
    if label:
        print(f"[{label}] {frames} frames in {time.time()-t0:.1f}s -- momentum "
              f"{max(rig.res['momentum']):.2e}, standoff {rig.res['standoff'][-1]:+.3e}, "
              f"bound {rig.res['bound'][-1]*100:.1f}%, lambda_geo {rig.res['lam_geo'][-1]:.4f}",
              flush=True)
    return kept, frames


# =============================================================================================
def render(kept, faces_bm, faces_epi, d, name, s_hi, fps=20):
    fig = plt.figure(figsize=(11.6, 5.8), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    strip, strip_at = [], set(np.round(np.linspace(0, len(kept) - 1, 8)).astype(int).tolist())
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for i, (t, X, L, XE, nod, PP) in enumerate(kept):
            fig.clf()
            c, lim = np.array([0.5, 0.5, 0.5]), 0.165
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            kf = X[faces_bm][:, :, 1].mean(1) > c[1]
            tri = Poly3DCollection(X[faces_bm][kf], linewidths=0.0)
            tri.set_facecolor(CMAP(np.clip((L[kf] - 1.0) / max(s_hi - 1.0, 1e-9), 0, 1)))
            ax.add_collection3d(tri)
            ke = XE[faces_epi][:, :, 1].mean(1) > c[1]
            wf = Line3DCollection(XE[faces_epi][ke][:, [0, 1, 2, 0]], colors=EPI_C, linewidths=0.35,
                                  alpha=0.55)
            ax.add_collection3d(wf)
            ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[1] - lim, c[1] + lim)
            ax.set_zlim(c[2] - lim, c[2] + lim)
            try:
                ax.set_box_aspect((1, 1, 1), zoom=1.55)
            except TypeError:
                ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=16, azim=-58)
            ax.text2D(0.02, 0.97, f"{name}   frame {t}\nsheet coloured by $\\lambda_{{geo}}$ to "
                                  f"{s_hi:.2f}; wireframe = the epithelium",
                      transform=ax.transAxes, color="white", fontsize=10.5, va="top")
            # the section, with the plaques drawn as what they are: segments between two bodies
            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            sl = np.abs(X[:, 1] - c[1]) < 0.004
            nl = np.zeros(X.shape[0]); cnt = np.zeros(X.shape[0])
            np.add.at(nl, faces_bm.reshape(-1), np.repeat(L, 3)); np.add.at(cnt, faces_bm.reshape(-1), 1)
            nl = nl / np.maximum(cnt, 1)
            seg_sel = np.abs(X[nod][:, 1] - c[1]) < 0.004
            a2.add_collection(LineCollection(
                np.stack([PP[seg_sel][:, [0, 2]], X[nod][seg_sel][:, [0, 2]]], 1),
                colors=PLQ_C, linewidths=0.8))
            a2.scatter(X[sl][:, 0], X[sl][:, 2], s=11, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP,
                       vmin=1.0, vmax=s_hi, marker=".", linewidths=0)
            se = np.abs(XE[:, 1] - c[1]) < 0.012
            a2.scatter(XE[se][:, 0], XE[se][:, 2], s=9, c=EPI_C, marker=".", linewidths=0, alpha=0.8)
            a2.set_xlim(c[0] - lim, c[0] + lim); a2.set_ylim(c[2] - lim, c[2] + lim)
            a2.set_aspect("equal"); a2.axis("off")
            a2.text(0.02, 0.98, "section: sheet (coloured), epithelium (cream), plaques (red)",
                    transform=a2.transAxes, color="white", fontsize=9.5, va="top")
            wri.grab_frame()
            if i in strip_at:
                strip.append((t, X.copy(), nl.copy(), XE.copy()))
    fig.savefig(os.path.join(d, "3d.png"), dpi=115, facecolor="black")
    plt.close(fig)
    figs = plt.figure(figsize=(3.0 * len(strip), 3.3), facecolor="black")
    for i, (t, X, nl, XE) in enumerate(strip):
        a = figs.add_subplot(1, len(strip), i + 1, facecolor="black")
        sl = np.abs(X[:, 1] - 0.5) < 0.004
        se = np.abs(XE[:, 1] - 0.5) < 0.012
        a.scatter(XE[se][:, 0], XE[se][:, 2], s=6, c=EPI_C, marker=".", linewidths=0, alpha=0.7)
        a.scatter(X[sl][:, 0], X[sl][:, 2], s=7, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP, vmin=1.0,
                  vmax=s_hi, marker=".", linewidths=0)
        a.set_xlim(0.335, 0.665); a.set_ylim(0.335, 0.665); a.set_aspect("equal"); a.axis("off")
        a.text(0.03, 0.97, f"frame {t}", transform=a.transAxes, color="white", fontsize=11, va="top")
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=115, facecolor="black")
    plt.close(figs)


def metrics_png(nom, free, dens, slip, rup, noplq, l0, d):
    fig, ax = plt.subplots(2, 3, figsize=(13.4, 6.8), facecolor="white")
    t = np.arange(len(nom["momentum"]))
    ax[0, 0].semilogy(t, np.maximum(nom["momentum"], 1e-18), color="#2b6cb0", lw=1.3)
    ax[0, 0].axhline(2.2e-16, color="#999", ls="--", lw=0.9)
    ax[0, 0].set_ylabel(r"$|\sum f_{bm} + \sum f_{epi}| / \sum |f|$")
    ax[0, 0].set_title(f"1. the reaction, as bookkeeping: max {max(nom['momentum']):.1e}\n"
                       f"(double precision eps dashed)", fontsize=8.5)
    ax[0, 1].semilogy(np.arange(len(free["drift"])), np.maximum(free["drift"], 1e-18),
                      color="#7a3b9a", lw=1.4)
    ax[0, 1].set_ylabel("relative drift of the mobility-weighted centroid")
    ax[0, 1].set_title(f"2. the reaction, as motion: two FREE bodies,\nmax "
                       f"{max(free['drift']):.1e} of the system's own size", fontsize=8.5)
    for lab, r in dens.items():
        ax[0, 2].plot(np.arange(len(r["standoff"])), np.asarray(r["standoff"]) / l0, lw=1.4,
                      label=lab)
    ax[0, 2].axhline(0.0, color="#1a1a1a", ls="--", lw=1.0)
    ax[0, 2].set_ylabel(r"$(\langle \ell\rangle - \ell_0)\,/\,\ell_0$")
    ax[0, 2].set_title(r"3. the standoff IS $\ell_0$, and how far density moves it" "\n"
                       r"(one-sided archive: $-0.008$ / $-0.045$ / $-0.119$ at 100/20/5%)",
                       fontsize=8.5)
    ax[0, 2].legend(fontsize=7, frameon=False)
    for lab, r in slip.items():
        ax[1, 0].plot(np.arange(len(r["twist"])), np.asarray(r["twist"]) * 180 / np.pi, lw=1.5,
                      label=lab)
    ax[1, 0].set_ylabel("twist of the sheet about the drive axis (deg)")
    ax[1, 0].set_title("4. slip: if these coincide, the friction law is decorative", fontsize=8.5)
    ax[1, 0].legend(fontsize=7, frameon=False)
    a1 = ax[1, 1]
    a1.plot(np.arange(len(rup["bound"])), np.asarray(rup["bound"]) * 100, color="#b03030", lw=1.6,
            label="fraction still bound")
    a2 = a1.twinx()
    a2.plot(np.arange(len(rup["load_p99"])), rup["load_p99"], color="#2b6cb0", lw=1.2, ls="--")
    a2.plot(np.arange(len(nom["load_p99"])), nom["load_p99"], color="#9ab", lw=1.0, ls=":")
    a2.set_ylabel("plaque load p99 (dashed: rupture run, dotted: nominal)", color="#2b6cb0",
                  fontsize=7.5)
    a1.set_ylabel("% of plaques still bound")
    a1.set_title("5. rupture at a threshold taken from the MEASURED load\n"
                 "(run 127 set it above the largest load in the run)", fontsize=8.5)
    ax[1, 2].plot(t, nom["lam_geo"], color="#1a1a1a", lw=1.7, label="with plaques")
    ax[1, 2].plot(np.arange(len(noplq["lam_geo"])), noplq["lam_geo"], color="#b03030", lw=1.4,
                  label="control: no plaques")
    ax[1, 2].plot(t, np.asarray(nom["R_epi"]) / nom["R_epi"][0], color="#1f8a5c", lw=1.2, ls="--",
                  label=r"$R_{epi}(t)/R_{epi}(0)$")
    ax[1, 2].set_ylabel(r"$\lambda_{geo}$ of the sheet")
    ax[1, 2].set_title("6. with the edge set empty the sheet is not carried", fontsize=8.5)
    ax[1, 2].legend(fontsize=7, frameon=False)
    for a in ax.reshape(-1):
        a.set_xlabel("frame"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "metrics.png"), dpi=150, facecolor="white")
    plt.close(fig)


# =============================================================================================
def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)
    name = arg("--name", str, "05b_plaque")
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)

    cert = BM.selftest(dev=dev, subdiv=4)
    assert cert["affine_max_err"] < 1e-10, cert
    T = 2.0e-3
    P = dict(subdiv=4, subdiv_epi=3, E=400.0, thickness=T, nu=0.3, kn=5.0, xi=0.0,
             l0=0.3 * T, zeta=20.0, s_target=1.0, k_drive=50.0, dev=dev)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 160))).astype(int).tolist())

    nom = Rig05b(**P)
    sp = float((nom.sheet.x[nom.sheet.Ed[:, 1]] - nom.sheet.x[nom.sheet.Ed[:, 0]]).norm(dim=1).mean())
    print(f"[{name}] {nom.sheet.n} bm nodes / {nom.x_epi.shape[0]} epi vertices / "
          f"{nom.n_plaque} plaques ({nom.missed} directions hit no face); plaque spacing "
          f"{sp:.4g} box units = {sp/T:.1f} T against the measured Sigma ~ 7 T; "
          f"l0 = {P['l0']:.3g} = 0.3 T", flush=True)
    kept, _ = run(nom, frames, keep=keep, label=name)

    # 2. the same claim as motion, with nothing driven
    freer = Rig05b(**{**P, "driven": False})
    # PRE-STRETCH EVERY PLAQUE, or there is no force whose conservation could be tested: the sheet is
    # pushed out by 4*l0, which loads every bond and nothing else.
    _R0 = float((freer.sheet.x - freer.c).norm(dim=1).mean())
    freer.sheet.x = freer.c + (freer.sheet.x - freer.c) * (1.0 + 4.0 * P["l0"] / _R0)
    run(freer, min(120, frames), label="free pair (no drive)")

    # 3. plaque density
    dens = {}
    # 0.046 IS THE MEASURED DENSITY, not a round number: at 100% anchoring this mesh puts a plaque
    # every 1.5 T, and Kanchanawong 2010's spacing is Sigma ~ 7 T, so the biological density is
    # (1.5/7)^2 = 4.6% of the nodes. The archived one-sided tether at 5% sagged to -0.119.
    for frac in (1.0, 0.2, 0.046):
        r = Rig05b(**{**P, "fraction": frac}) if frac < 1.0 else nom
        if frac < 1.0:
            run(r, frames, label=f"density {frac*100:g}%")
        dens[f"{frac*100:g}% of nodes anchored"] = r.res

    # 4. slip, under a rotating epithelium
    slip = {}
    for xi, lab in ((0.0, r"$\xi = 0$"), (5.0, r"$\xi = 5$")):
        r = Rig05b(**{**P, "xi": xi, "omega": 0.0025})
        run(r, min(200, frames), label=f"slip xi = {xi:g}")
        slip[lab] = r.res

    # 5. rupture, at a threshold taken from what the nominal MEASURED
    # THE THRESHOLD COMES FROM THE LOAD THE NOMINAL ACTUALLY CARRIED. Run 127's `detach: 0.02` sat
    # above the largest load in its whole run (0.017), so it was reported as a mechanism that did
    # nothing when it was a threshold that was never reached. The p75 over frames of the nominal's own
    # p99 load is inside the distribution by construction: the most-loaded plaques go and the rest
    # hold, which is the only regime in which "fraction bound vs load" is a measurement.
    thr = float(np.percentile(nom.res["load_p99"], 75))
    rup = Rig05b(**{**P, "break_load": thr})
    run(rup, frames, label=f"rupture at load {thr:.3g} (p50 of the nominal's own p50)")

    # 6. the control
    noplq = Rig05b(**{**P, "fraction": 1e-6})
    run(noplq, frames, label="control: no plaques")

    s_hi = float(np.percentile(np.concatenate([k[2] for k in kept[::4]]), 99))
    render(kept, nom.sheet.Fc.cpu().numpy(), nom.F_epi.cpu().numpy(), d, name, s_hi)
    metrics_png(nom.res, freer.res, dens, slip, rup.res, noplq.res, P["l0"], d)

    out = dict(
        run=name, frames=frames, certification=cert,
        rig=dict(**{k: v for k, v in P.items() if k != "dev"},
                 bm_nodes=nom.sheet.n, epi_vertices=int(nom.x_epi.shape[0]),
                 plaques=nom.n_plaque, directions_missing_a_face=nom.missed,
                 plaque_spacing_box=sp, plaque_spacing_in_T=sp / T,
                 l0_in_T=P["l0"] / T, mobility_bm=nom.sheet.M,
                 substeps_first=nom.res["n_sub"][0], substeps_final=nom.res["n_sub"][-1]),
        momentum=dict(max=max(nom.res["momentum"]),
                      median=float(np.median(nom.res["momentum"])),
                      eps_float64=float(np.finfo(np.float64).eps)),
        free_pair=dict(max_relative_drift=max(freer.res["drift"]),
                       frames=len(freer.res["drift"])),
        standoff=dict(l0=P["l0"],
                      **{f"frac_{k.split('%')[0]}": dict(final=v["standoff"][-1],
                                                         relative_to_l0=v["standoff"][-1] / P["l0"],
                                                         ell_final=v["ell_mean"][-1])
                         for k, v in dens.items()},
                      archived_one_sided={"100%": -0.0082, "20%": -0.0449, "5%": -0.1190}),
        slip={k.replace("$", "").replace("\\", ""):
              dict(twist_final_deg=v["twist"][-1] * 180 / math.pi,
                   drive_deg=0.0025 * (len(v["twist"]) - 1) * 180 / math.pi)
              for k, v in slip.items()},
        rupture=dict(threshold=thr, threshold_source="p50 over frames of the nominal's own per-frame "
                                                     "median plaque load",
                     nominal_load_p99_max=max(nom.res["load_p99"]),
                     bound_final=rup.res["bound"][-1], bound_min=min(rup.res["bound"]),
                     standoff_final=rup.res["standoff"][-1]),
        control_no_plaque=dict(lambda_geo_final=noplq.res["lam_geo"][-1],
                               lambda_geo_with_plaques=nom.res["lam_geo"][-1],
                               R_bm_final=noplq.res["R_bm"][-1],
                               R_epi_final=noplq.res["R_epi"][-1]),
        stretch=dict(lambda_geo_final=nom.res["lam_geo"][-1],
                     applied=nom.res["R_epi"][-1] / nom.res["R_epi"][0]),
        series={k: [float(x) for x in v] for k, v in nom.res.items()})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    np.savez_compressed(
        os.path.join(d, "traj.npz"),
        frames=np.asarray([k[0] for k in kept]),
        pos=np.stack([k[1] for k in kept]).astype(np.float32),
        lam=np.stack([k[2] for k in kept]).astype(np.float32),
        pos_epi=np.stack([k[3] for k in kept]).astype(np.float32),
        faces=nom.sheet.Fc.cpu().numpy().astype(np.int32),
        faces_epi=nom.F_epi.cpu().numpy().astype(np.int32),
        plaque_node=nom.plq.node.cpu().numpy().astype(np.int32),
        plaque_face=nom.plq.face.cpu().numpy().astype(np.int32),
        plaque_w=nom.plq.w.float().cpu().numpy(),
        **{f"series_{k}": np.asarray(v, dtype=np.float32) for k, v in nom.res.items()})
    yaml.safe_dump(dict(
        units=dict(**UNITS, force_nN=None,
                   note="declared per plexus/units.py: length MEASURED from the cache (a cell is "
                        "8.54e-3 box across; assuming 10 um gives 1 box = 1171 um and a 318 um "
                        "spheroid), time from 3.99 doublings over 401 frames at a 12-24 h cycle. "
                        "Force is NOT declared: nothing here fixes a force scale, so only ratios "
                        "are meaningful.",
                   exposes=f"thickness T = 2e-3 box = {2e-3*UNITS['length_um']:.2f} um against a real "
                           f"basement membrane of {T_REAL_UM} um -- {2e-3*UNITS['length_um']/T_REAL_UM:.0f}x "
                           f"too thick; l0 and Sigma are defined in T and are too large by the same "
                           f"factor. The proportions are right, the absolute scale is not."),
        what="05b -- the hemidesmosome as an edge set cell->bm: one operator, a delta to both ends",
        question="does an adhesion written as a RELATION conserve momentum, set the standoff to its "
                 "own rest length, and permit slip -- none of which the archived one-sided tether did",
        plaque=dict(normal="spring at rest length l0 = 0.3 T (Kanchanawong 2010 puts the integrin "
                           "layer ~30 nm under a ~350 nm plaque)",
                    tangential="friction xi against RELATIVE velocity, so the sheet slides",
                    reaction="barycentric, to the three vertices of the epithelial face",
                    seeding="brute force ray-in-triangle over every face, once",
                    rupture="load threshold taken from the load distribution the nominal measured"),
        epithelium=dict(kind="driven icosphere with a finite mobility, NOT pinned",
                        why="a pinned body absorbs a reaction silently",
                        subdiv=P["subdiv_epi"], surface=os.path.basename(TISSUE), scale=SCALE),
        sheet=dict(model="the 05a codim-1 StVK membrane", subdiv=P["subdiv"], E=P["E"],
                   thickness=T),
        measures=["momentum residual on the adhesion's own force pair",
                  "centroid drift of a free pair", "standoff vs l0 and vs plaque density",
                  "slip with and without friction under a rotating epithelium",
                  "fraction bound vs a measured load threshold", "no-plaque control"],
        not_modelled=["plaque re-seeding / turnover", "the matrix and the fibril", "secretion",
                      "the vertex-model epithelium (mesh_contact_ops already certifies that "
                      "interface)"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{name}] momentum max {out['momentum']['max']:.2e}, free-pair drift "
          f"{out['free_pair']['max_relative_drift']:.2e}, standoff/l0 "
          f"{out['standoff']['frac_100']['relative_to_l0']:+.4f}, bound after rupture "
          f"{100*out['rupture']['bound_final']:.1f}%, control lambda "
          f"{out['control_no_plaque']['lambda_geo_final']:.4f} -> {d}", flush=True)


if __name__ == "__main__":
    main()
