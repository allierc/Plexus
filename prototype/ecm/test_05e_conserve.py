#!/usr/bin/env python
"""test_05e_conserve -- 05e: what a remesh must carry, and the two things it was dropping.

    python test_05e_conserve.py [--device cuda:0] [--frames 401]  ->  log/okuda_ECM/05e_conserve/

WHY THIS RUN EXISTS. An external review of `note_sheet` made one correction that matters more than it
looks: **a remesh is a NUMERICAL operation.** It changes the triangulation and nothing else -- not the
physical surface, not the material state, not the mass. Secretion, remodelling and proteolysis are the
biological operators; they share the `occ` reservoir with refinement as an implementation substrate and
they share nothing else. Calling all three "the same mechanism" -- which an earlier draft did -- is the
confusion that lets a discretisation choice be read as a result, and it is the same error as the MPM
sheet whose tearing was set by dx.

That correction turned up two real defects, and this run is both fixes plus their gates.

  1. RHO WAS DERIVED, NOT CONSERVED. The sheet reported areal density as `rho0 / J`, a formula, so
     nothing conserved anything: a split produced four faces each still claiming the parent's density
     and the total material in the sheet was whatever the arithmetic happened to give. Fixed by making
     MASS the state and rho the derived quantity, `rho = mass_f / A_f(now)`. Three consequences:
       * `sum_f mass_f` is invariant across a split by construction (each child takes a quarter);
       * the dilution term `-rho (Adot/A)` of the mass balance is no longer something to integrate --
         it is what m/A does on its own, and integrating it as well would count it twice;
       * `bm_secrete` (05f) now has something to add TO.
  2. PLAQUES WERE SEEDED AS A FRACTION OF NODES. Correct on a fixed mesh, wrong on a refining one: a
     refinement quadruples the nodes and leaves every new one unanchored, so the plaque AREAL density
     -- the quantity Kanchanawong et al. actually measured, Sigma ~ 7T between plaques -- fell 4x at
     every refinement while the fraction stayed put. Fixed by making `plaque_seed` take Sigma in units
     of the thickness and top the count up to A(t)/Sigma^2 every frame, which is also the biology: a
     growing epithelium makes more hemidesmosomes, it does not stretch the ones it has.

THE GATES (thresholds fixed before the run; `bm_ops.selftest()` runs G18b/G18d before any physics):

  G18b  sum_f mass_f is invariant across a split                        exact
  G18c  plaque areal density stays at Sigma^-2 across two refinements   within 10%
  G18d  the physical surface is unchanged by a split:
        enclosed volume, total area, area-weighted centroid             < 1e-12 relative
  G18e  rho measured as mass/A equals the rho0/J the fixed mesh predicts within 1%
        -- i.e. making mass the state did not change the physics, only what is conserved

WHAT IS NOT HERE. No secretion and no proteolysis: `mass` is conserved in this run because nothing is
allowed to change it, which is exactly what makes it a control for 05f. No turnover -- plaques are added
and never removed, so this is `plaque_seed` as growth and not yet as exchange (05h). The epithelium is
the driven icosphere of 05b, not the vertex model.
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
from test_05b_plaque import seed_plaques, Plaques                       # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

CMAP = ListedColormap(ES.STRESS_COLORS)
EPI_C = "#e8dcc0"
PLQ_C = "#e0452b"
TARGET_C = "#8fb8de"


class Rig05e:
    """The 05c reservoir and the 05b plaque in one rig, so that a refinement has something to break.

    `sigma_T` is the plaque spacing in units of the sheet thickness -- Kanchanawong et al. 2010 measure
    plaques ~350 nm across on ~700 nm centres against a ~100 nm sheet, so Sigma ~ 7T. The plaque COUNT
    is then a consequence of the sheet's area rather than a parameter, which is the whole point.
    """

    def __init__(self, subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, xi=0.0, sigma_T=7.0,
                 zeta=20.0, s_target=1.0, refresh=10, tau_r=0.0, max_refine=2, edge_trigger=1.45,
                 k_drive=50.0, reseed=True, k_c=20.0, edge_target_sigma=1.0 / 3.0,
                 dev="cuda:0", dtype=torch.float64):
        self.dev, self.dtype = dev, dtype
        Vb, Fb, Eb = BM.icosphere(subdiv, device=dev, dtype=dtype)
        Ve, Fe, Ee = BM.icosphere(3, device=dev, dtype=dtype)
        self.rep_e = SurfaceReplay(Ve, dev=dev, dtype=dtype)
        self.c = torch.tensor([0.5, 0.5, 0.5], device=dev, dtype=dtype)
        self.u_epi, self.F_epi = Ve, Fe
        self.x_epi = self.c + Ve * self.rep_e.R(0)[:, None]
        self.v_epi = torch.zeros_like(self.x_epi)
        self.k_drive, self.M_epi = float(k_drive), 1.0
        self.T, self.l0 = float(thickness), 0.3 * float(thickness)
        self.sigma = float(sigma_T) * self.T
        self.reseed_on = bool(reseed)
        self.sheet = BM.Sheet(subdiv=subdiv, R0=1.0, E=E, thickness=thickness, nu=nu, tau_r=tau_r,
                              max_refine=max_refine, dev=dev, dtype=dtype)
        rep0 = SurfaceReplay(Vb, dev=dev, dtype=dtype)
        self.sheet.reseed(self.c + Vb * (rep0.R(0) + self.l0)[:, None])
        self.kn, self.xi = float(kn), float(xi)
        # THE TRIGGER IS TIED TO Sigma, NOT TO THE SEEDED EDGE, and that is a correction. Refining
        # whenever the edge passes 1.45x its own seeded value holds the mesh at whatever resolution it
        # happened to start with -- which produced 30 triangles and 35 nodes per plaque spacing by the
        # last frame, resolving nothing biological in between and costing 81,920 faces. The finest
        # length this sheet has to represent is the plaque spacing Sigma: below it there is no
        # structure. The target is therefore a fraction of Sigma, and the seeded resolution becomes an
        # initial condition rather than a standard.
        self.edge_target = float(edge_target_sigma) * self.sigma
        self.edge_trigger, self.max_refine = float(edge_trigger), int(max_refine)
        # the plaque set starts EMPTY and is filled by `plaque_seed` to the measured areal density,
        # so the count is never a number anybody chose
        self._n_pen, self._pen_max = 0, 0.0
        self.events = []            # one row per refinement: the before/after of every invariant
        self.plq = Plaques(torch.zeros(0, dtype=torch.long, device=dev),
                           torch.zeros(0, dtype=torch.long, device=dev),
                           torch.zeros(0, 3, device=dev, dtype=dtype), self.l0, kn, xi)
        self.k_c = float(k_c)
        self.build_contact()
        self.plaque_seed(force=True)      # the initial set is at the measured density either way
        self.lam_el, self._pv = self.sheet.spectral_rate(return_vec=True)
        self.sheet.M = float(zeta) / (self.lam_el + self.kn)
        self.s_target, self.refresh = float(s_target), int(refresh)
        self.n_sub = self._nsub()
        self.res = {k: [] for k in ("mass", "rho", "rho_pred", "area", "volume", "centroid",
                                    "n_plaque", "plq_density", "lam_geo", "lam_el", "mean_edge",
                                    "n_faces", "n_nodes", "standoff", "momentum", "n_sub",
                                    "R_target", "refined", "radial_gap", "inside_frac",
                                    "n_penetrating", "penetration_max")}

    # -- plaque_seed as an operator: top up to the MEASURED areal density -------------------------
    def plaque_seed(self, force=False):
        """Add plaques until there is one per Sigma^2 of sheet. Never removes any -- unbinding is
        turnover and belongs to 05h. Returns how many were added.

        `force` seeds the initial set even in the control run, so that the control differs from the
        nominal in ONE thing: whether the count is topped up as the sheet grows and refines."""
        area = float(self.sheet.area().sum())
        target = max(1, int(round(area / self.sigma ** 2)))
        have = int(self.plq.node.numel())
        if (not force and not self.reseed_on) or target <= have:
            return 0
        free = torch.ones(self.sheet.x.shape[0], dtype=torch.bool, device=self.dev)
        free[self.plq.node] = False
        cand = self.sheet.live_nodes[free[self.sheet.live_nodes]]
        if cand.numel() == 0:
            return 0
        n_add = min(target - have, int(cand.numel()))
        g = torch.Generator(device="cpu").manual_seed(len(self.events) * 7919 + have)
        pick = cand[torch.randperm(int(cand.numel()), generator=g)[:n_add].to(self.dev)]
        u = self.sheet.x[pick] - self.c
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-30)
        keep, face, w, _ = seed_plaques(u, self.u_epi, self.F_epi)
        if keep.numel() == 0:
            return 0
        self.plq.node = torch.cat([self.plq.node, pick[keep]])
        self.plq.face = torch.cat([self.plq.face, face])
        self.plq.w = torch.cat([self.plq.w, w])
        self.plq.bound = torch.cat([self.plq.bound,
                                    torch.ones(keep.numel(), dtype=torch.bool, device=self.dev)])
        self.plq.load = torch.cat([self.plq.load,
                                   torch.zeros(keep.numel(), device=self.dev, dtype=self.dtype)])
        return int(keep.numel())

    # -- bm_contact: the epithelium excludes the sheet ------------------------------------------
    def build_contact(self):
        """Map every live sheet node to the epithelial face its own direction points into.

        Rebuilt only when the sheet's topology changes, so a 40,962-node sheet pays for the lookup
        twice in a run rather than 401 times. Chunked over nodes because the brute-force test is
        (nodes x faces x 3) and the finest level would otherwise ask for a gigabyte.
        """
        idx = self.sheet.live_nodes
        u = self.sheet.x[idx] - self.c
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-30)
        tri = self.u_epi[self.F_epi]
        Minv = torch.linalg.inv(tri.transpose(1, 2))
        faces, ws, keep = [], [], []
        for i in range(0, u.shape[0], 4096):
            bc = torch.einsum("fij,nj->nfi", Minv, u[i:i + 4096])
            ok = (bc >= -1e-12).all(-1)
            f = torch.argmax(ok.to(torch.int8), dim=1)
            hit = ok.any(1)
            w = torch.gather(bc, 1, f[:, None, None].expand(-1, 1, 3)).squeeze(1)
            faces.append(f[hit]); ws.append((w / w.sum(1, keepdim=True))[hit])
            keep.append(idx[i:i + 4096][hit])
        self.ct_node = torch.cat(keep); self.ct_face = torch.cat(faces); self.ct_w = torch.cat(ws)

    def contact(self):
        """`bm_contact`: a ONE-SIDED penalty holding every sheet node at or outside the epithelium.

        WHY THIS OPERATOR EXISTS, AND IT IS NOT A NUMERICAL PATCH. 05e's first run put the sheet
        5.96e-4 outside at seeding (= l0, correct) and 7.7e-3 INSIDE by the last frame -- thirteen l0
        through the epithelium -- because at the measured plaque spacing Sigma = 7T only ~4% of the
        sheet's nodes are anchored at all, and the sheet's own hoop stress pulls the unanchored 96%
        inward, dragging the anchors after it. Giving the plaque's normal a SIGN (its face's outward
        normal instead of the line between the two points) was necessary and not sufficient: the
        sheet still ended 89.6% inside.

        The missing statement is the one no adhesion model contains: a basement membrane cannot be
        inside the cells. That is non-penetration, it applies at EVERY node and not only where a
        hemidesmosome happens to be, and it is one-sided -- it pushes out and never pulls in, so a
        sheet held out by its plaques feels nothing from it. Its reaction goes back to the three
        vertices of the epithelial face, so the momentum gate still covers it.

        This is also what the archived line did with `cell_exclude_3d` and got wrong: that operator
        PROJECTED particles out, which repositions without touching F and launders deformation --
        run 88's sheet was carried outward as a decal at a strain of 7e-4. A force cannot do that.
        """
        if self.k_c <= 0 or self.ct_node.numel() == 0:
            z = torch.zeros_like(self.sheet.x)
            return z, torch.zeros_like(self.x_epi), 0, 0.0
        vf = self.F_epi[self.ct_face]
        tri = self.x_epi[vf]
        p = (tri * self.ct_w[:, :, None]).sum(1)
        n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-30)
        n = n * torch.sign(((p - self.c) * n).sum(1, keepdim=True)).clamp(min=-1.0, max=1.0)
        d = ((self.sheet.x[self.ct_node] - p) * n).sum(1)
        pen = (d - self.l0).clamp(max=0.0)              # <0 only where the sheet is too close
        f = -self.k_c * pen[:, None] * n                # outward
        fb = torch.zeros_like(self.sheet.x)
        fe = torch.zeros_like(self.x_epi)
        fb.index_add_(0, self.ct_node, f)
        fe.index_add_(0, vf.reshape(-1), (-f[:, None, :] * self.ct_w[:, :, None]).reshape(-1, 3))
        return fb, fe, int((pen < 0).sum()), float(-pen.min())

    def _nsub(self):
        a = self.sheet.M * (self.lam_el + self.kn + self.k_c)
        b = self.M_epi * (self.k_drive + self.kn + self.xi)
        return max(1, int(math.ceil(max(a, b) / self.s_target)))

    def _invariants(self):
        return dict(mass=self.sheet.total_mass(), volume=self.sheet.enclosed_volume(),
                    area=float(self.sheet.area().sum()),
                    centroid=self.sheet.area_centroid().clone(),
                    lam=float(self.sheet.stretch_geo()[0].mean()),
                    energy=float(self.sheet.energy(self.sheet.x)),
                    n_plaque=int(self.plq.bound.sum()))

    def frame(self, t):
        refined = 0
        if (self.max_refine and self.sheet.n_refinements < self.max_refine
                and self.sheet.mean_edge() > self.edge_target):
            # EVERY INVARIANT, IMMEDIATELY EITHER SIDE OF THE SPLIT AND NOTHING ELSE IN BETWEEN.
            # Measured across a frame instead, the dynamics would move all of them and the gate would
            # be a statement about the timestep.
            before = self._invariants()
            ne, nf = self.sheet.refine()
            self.build_contact()          # new nodes need a face to be excluded by
            after = self._invariants()
            added = self.plaque_seed()
            self.events.append(dict(
                frame=t, faces=self.sheet.m, nodes=self.sheet.n, new_nodes=int(ne),
                plaques_added_after=int(added),
                **{f"{k}_rel": (float((after[k] - before[k]).norm()) / max(float(before[k].norm()),
                                                                          1e-30)
                                if torch.is_tensor(before[k])
                                else abs(after[k] - before[k]) / max(abs(before[k]), 1e-30))
                   for k in ("mass", "volume", "area", "centroid", "lam", "energy")},
                plaque_density_before=before["n_plaque"] / max(before["area"], 1e-30),
                plaque_density_after=after["n_plaque"] / max(after["area"], 1e-30)))
            self.lam_el, self._pv = self.sheet.spectral_rate(iters=40, return_vec=True)
            self.n_sub = self._nsub()
            refined = nf
            e = self.events[-1]
            print(f"    [refine] frame {t}: {self.sheet.m} faces, {self.sheet.n} nodes, "
                  f"+{added} plaques; mass {e['mass_rel']:.1e}, volume {e['volume_rel']:.1e}, "
                  f"centroid {e['centroid_rel']:.1e}, lambda {e['lam_rel']:.1e}", flush=True)
        self.plaque_seed()
        if t % self.refresh == 0 and torch.isfinite(self.sheet.x).all():
            self.lam_el, self._pv = self.sheet.spectral_rate(iters=25, v0=self._pv, return_vec=True)
            self.n_sub = self._nsub()
        a_epi = self.c + self.u_epi * self.rep_e.R(t)[:, None]
        dt, M, mom = 1.0 / self.n_sub, self.sheet.M, 0.0
        for _ in range(self.n_sub):
            vf, nh, vp, f_n = self.plq.geometry(self.sheet.x, self.x_epi, self.v_epi,
                                                      self.F_epi, centre=self.c)
            f_el = self.sheet.elastic_force(self.sheet.x)
            fb_n = torch.zeros_like(self.sheet.x)
            fb_n.index_add_(0, self.plq.node, f_n)
            fb_c, fe_c, n_pen, pen_max = self.contact()
            fb_n = fb_n + fb_c
            v_prov = M * (f_el + fb_n)
            vpr = v_prov[self.plq.node]
            par = lambda a: a - (a * nh).sum(1, keepdim=True) * nh                      # noqa: E731
            xi_eff = self.plq.xi * self.plq.bound.to(nh.dtype)
            v_par = (par(vpr) + (M * xi_eff)[:, None] * par(vp)) / (1.0 + M * xi_eff)[:, None]
            f_t = xi_eff[:, None] * (par(vp) - v_par)
            fb, fe = self.plq.scatter(f_n + f_t, vf, self.sheet.x, self.x_epi)
            fe = fe + fe_c
            # the momentum residual now covers BOTH couplings -- the adhesion and the contact -- so a
            # contact that forgot its reaction would show up here rather than in a quiet drift
            mom = max(mom, float((fb.sum(0) + fb_c.sum(0) + fe.sum(0)).norm()) /
                      (float((f_n + f_t).norm(dim=1).sum())
                       + float(fb_c.norm(dim=1).sum()) + 1e-300))
            dxb = dt * v_prov
            dxb.index_add_(0, self.plq.node, dt * (v_par - par(vpr)))
            self.sheet.advance(dxb, dt)
            dxe = dt * self.M_epi * (fe + self.k_drive * (a_epi - self.x_epi))
            self.v_epi = dxe / dt
            self.x_epi = self.x_epi + dxe
            self._n_pen, self._pen_max = n_pen, pen_max
        self._record(t, mom, refined)

    def _record(self, t, mom, refined):
        l1, l2 = self.sheet.stretch_geo()
        e1, _ = self.sheet.stretch_elastic()
        area = float(self.sheet.area().sum())
        self.res["mass"].append(self.sheet.total_mass())
        self.res["rho"].append(float(self.sheet.areal_density().mean()))
        self.res["rho_pred"].append(float((1.0 / (l1 * l2).clamp_min(1e-12)).mean()))
        self.res["area"].append(area)
        self.res["volume"].append(self.sheet.enclosed_volume())
        self.res["centroid"].append(float((self.sheet.area_centroid() - self.c).norm()))
        self.res["n_plaque"].append(int(self.plq.bound.sum()))
        self.res["plq_density"].append(int(self.plq.bound.sum()) / max(area, 1e-30) * self.sigma ** 2)
        self.res["lam_geo"].append(float(l1.mean()))
        self.res["lam_el"].append(float(e1.mean()))
        self.res["mean_edge"].append(self.sheet.mean_edge() / self.sheet.mean_edge_seed)
        self.res["n_faces"].append(self.sheet.m)
        self.res["n_nodes"].append(self.sheet.n)
        # SIGNED, along the epithelium's outward normal, and cross-checked radially. The first
        # version reported |x_bm - p| - l0, an unsigned distance, which gives the SAME positive number
        # for a sheet held l0 outside and a sheet that has sunk l0 inside -- and 05e's first run did
        # sink, 7.7e-3 through the surface, while the metric read +4.0e-3.
        # `ell` exists only once the plaque law has run; a subclass that replaces the plaque with a
        # bond density (05d) overwrites this entry in its own _record and has no `ell` at frame 0.
        self.res["standoff"].append(float(getattr(self.plq, "ell", self.sheet.x.new_zeros(1)).mean())
                                    - self.l0)
        r_bm = (self.sheet.x[self.sheet.live_nodes] - self.c).norm(dim=1).mean()
        r_ep = (self.x_epi - self.c).norm(dim=1).mean()
        self.res["radial_gap"].append(float(r_bm - r_ep))
        _e = getattr(self.plq, "ell", None)
        self.res["inside_frac"].append(float((_e < 0).to(self.dtype).mean())
                                       if _e is not None and _e.numel() else 0.0)
        self.res["momentum"].append(mom)
        self.res["n_penetrating"].append(int(self._n_pen))
        self.res["penetration_max"].append(float(self._pen_max))
        self.res["n_sub"].append(int(self.n_sub))
        self.res["R_target"].append(float(self.rep_e.R(t).mean()))
        self.res["refined"].append(int(refined))

    def alive(self):
        return bool(torch.isfinite(self.sheet.x[self.sheet.live_nodes]).all())


def run(rig, frames, keep=None, label=""):
    keep, kept, t0 = (set() if keep is None else keep), [], time.time()
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[{label}] DIVERGED at frame {t}", flush=True)
            return kept, t
        if t in keep:
            l1, _ = rig.sheet.stretch_geo()
            pp = (rig.x_epi[rig.F_epi[rig.plq.face]] * rig.plq.w[:, :, None]).sum(1)
            kept.append((t, rig.sheet.x.float().cpu().numpy(), l1.float().cpu().numpy(),
                         rig.sheet.Fc.cpu().numpy(), rig.x_epi.float().cpu().numpy(),
                         rig.plq.node.cpu().numpy(), pp.float().cpu().numpy()))
    if label:
        print(f"[{label}] {frames} frames in {time.time()-t0:.1f}s -- {rig.sheet.m} faces, "
              f"{rig.res['n_plaque'][-1]} plaques at {rig.res['plq_density'][-1]:.3f} of "
              f"Sigma^-2, mass {rig.res['mass'][-1]:.6e}, rho {rig.res['rho'][-1]:.4f}", flush=True)
    return kept, frames


# =============================================================================================
def model_png(rig, d, sigma_T):
    """`remesh_model.png` -- the operator, its equations, and the four things it must not change.

    Same register as `01_junction/junction_model.png`: the operator names itself on the left, states
    the equations it IS, and every panel on the right is a measurement that could have come back wrong.
    """
    fig = plt.figure(figsize=(14.6, 6.0), facecolor="white")
    axE = fig.add_axes([0.005, 0.05, 0.235, 0.90]); axE.axis("off")
    ax = [fig.add_axes([0.315, 0.575, 0.29, 0.345]), fig.add_axes([0.695, 0.575, 0.29, 0.345]),
          fig.add_axes([0.315, 0.095, 0.29, 0.375]), fig.add_axes([0.695, 0.095, 0.29, 0.375])]
    axE.text(0.0, 1.00, "bm_refine", fontsize=13, fontweight="bold", va="top", family="monospace")
    axE.text(0.0, 0.935, "a Divide operator on the bm_face hyperedge set\n"
                         "NUMERICAL MAINTENANCE ONLY: it changes the\n"
                         "triangulation and nothing else", fontsize=8.2, va="top", color="#444")
    axE.text(0.0, 0.795, r"$\mathbf{D}_m^{\rm child}=\mathbf{D}_m^{\rm parent}\,\mathbf{S}_k$,"
                         r"$\quad\det\mathbf{S}_k=1/4$", fontsize=12.5, va="top")
    axE.text(0.0, 0.715, r"$A^0\!\to\!A^0/4$,$\quad m_f\!\to\!m_f/4$,"
                         r"$\quad \mathbf{C}_0,\,Y_2$ inherited", fontsize=12.5, va="top")
    axE.text(0.0, 0.615, r"$\rho_f=\dfrac{m_f}{A_f(\mathrm{now})}$", fontsize=14, va="top")
    axE.text(0.0, 0.505, r"$N_{\rm plaque}=A(t)\,/\,\Sigma^{2}$", fontsize=13, va="top")
    axE.text(0.0, 0.415,
             "a split is invisible to the material: the child\n"
             "inherits its parent's reference frame, so $\\lambda$ is\n"
             "unchanged, and takes a quarter of its MASS, so\n"
             "$\\sum_f m_f$ is unchanged. $\\rho$ is derived, never stored,\n"
             "so the dilution term of the mass balance is not\n"
             "integrated -- it is what $m/A$ does on its own.",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.195,
             f"$\\Sigma$ = {sigma_T:g}$\\,T$   $\\ell_0$ = 0.3$\\,T$   $T$ = {rig.T:g} box units\n"
             f"refine when $\\langle\\ell\\rangle > \\Sigma/3$ = {rig.edge_target:.4g} box\n"
             f"reservoir: {rig.sheet.F_all.shape[0]} face slots, "
             f"{rig.sheet.x.shape[0]} node slots",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.03, "Kanchanawong et al. 2010 Nature 468:580\n"
                        "Rivara 1984 Int J Numer Methods Eng 20:745\n"
                        "engine.py:453 -- `grow_reserve`, the same pattern for MPM",
             fontsize=7.3, va="bottom", color="#666")

    t = np.arange(len(rig.res["mass"]))
    rf = [i for i, v in enumerate(rig.res["refined"]) if v]

    def marks(a):
        for f in rf:
            a.axvline(f, color="#c33", lw=1.0, ls="-.")

    m0 = rig.res["mass"][0]
    ax[0].plot(t, np.asarray(rig.res["mass"]) / m0, color="#1a1a1a", lw=1.8, label=r"$\sum_f m_f$")
    ax[0].plot(t, np.asarray(rig.res["volume"]) / rig.res["volume"][0], color="#2b6cb0", lw=1.3,
               label="enclosed volume")
    ax[0].plot(t, np.asarray(rig.res["area"]) / rig.res["area"][0], color="#1f8a5c", lw=1.3,
               ls="--", label="area")
    marks(ax[0]); ax[0].set_yscale("log")
    ax[0].set_ylabel("relative to frame 0")
    ax[0].set_title(f"G18b/G18d: mass is untouched by a split (and by growth);\n"
                    f"area and volume are not, because the tissue grew", fontsize=8.5)
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot(t, rig.res["plq_density"], color="#b03030", lw=1.8)
    ax[1].axhline(1.0, color="#1a1a1a", ls="--", lw=1.0)
    a1 = ax[1].twinx()
    a1.plot(t, rig.res["n_plaque"], color="#e08a2e", lw=1.2, ls=":")
    a1.set_ylabel("plaque count", color="#e08a2e")
    marks(ax[1])
    ax[1].set_ylabel(r"plaque areal density $\times\,\Sigma^{2}$")
    ax[1].set_title(f"G18c: held at $\\Sigma^{{-2}}$ across two refinements\n"
                    f"(a fixed FRACTION of nodes would drop 4x at each line)", fontsize=8.5)
    ax[2].plot(t, rig.res["rho"], color="#1a1a1a", lw=1.8, label=r"$\rho = m_f/A_f$, measured")
    ax[2].plot(t, rig.res["rho_pred"], color="#7a3b9a", lw=1.2, ls="--", label=r"$1/J$, predicted")
    marks(ax[2]); ax[2].set_yscale("log")
    ax[2].set_ylabel(r"areal density $\rho/\rho_0$")
    # G18e is an IDENTITY once mass is the conserved state, not a measurement: m/A = rho0 A0/A = rho0/J
    # exactly. It is drawn because it is the baseline `bm_secrete` has to beat, and it is labelled as an
    # identity so that nobody reads a tautology as a validation.
    ax[2].set_title("with nothing secreting, mass/area IS $1/J$ identically ---\n"
                    "the baseline `bm_secrete` has to hold flat [G18e]", fontsize=8.5)
    ax[2].legend(fontsize=7, frameon=False)
    ax[3].plot(t, rig.res["lam_geo"], color="#1a1a1a", lw=1.8, label=r"$\lambda_{geo}$")
    ax[3].plot(t, np.asarray(rig.res["R_target"]) / rig.res["R_target"][0], color="#1f8a5c",
               lw=1.2, ls=":", label="applied $R(t)/R(0)$")
    a3 = ax[3].twinx()
    a3.plot(t, rig.res["mean_edge"], color="#2b6cb0", lw=1.2, ls="--")
    a3.axhline(rig.edge_target / rig.sheet.mean_edge_seed, color="#2b6cb0", lw=0.7, ls=":")
    a3.set_ylabel("mean edge / seeded", color="#2b6cb0")
    marks(ax[3])
    ax[3].set_ylabel(r"$\lambda_{geo}$")
    ax[3].set_title("the split is invisible in $\\lambda$: no step at the red lines,\n"
                    "while the edge length is halved at each", fontsize=8.5)
    ax[3].legend(fontsize=7, frameon=False)
    for a in ax[2:]:
        a.set_xlabel("frame")
    for a in ax[:2]:
        a.tick_params(labelbottom=False)          # the top row's labels collided with the lower titles
    for a in ax:
        a.spines[["top"]].set_visible(False)
    fig.savefig(os.path.join(d, "remesh_model.png"), dpi=150, facecolor="white")
    plt.close(fig)


def render(kept, d, name, s_hi, fps=20):
    fig = plt.figure(figsize=(11.6, 5.8), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    strip, strip_at = [], set(np.round(np.linspace(0, len(kept) - 1, 8)).astype(int).tolist())
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for i, (t, X, L, F, XE, nod, PP) in enumerate(kept):
            fig.clf()
            c, lim = np.array([0.5, 0.5, 0.5]), 0.165
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            kf = X[F][:, :, 1].mean(1) > c[1]
            tri = Poly3DCollection(X[F][kf], linewidths=0.1, edgecolors=(1, 1, 1, 0.16))
            tri.set_facecolor(CMAP(np.clip((L[kf] - 1.0) / max(s_hi - 1.0, 1e-9), 0, 1)))
            ax.add_collection3d(tri)
            ke = XE[F_EPI_CACHE][:, :, 1].mean(1) > c[1]
            ax.add_collection3d(Line3DCollection(XE[F_EPI_CACHE][ke][:, [0, 1, 2, 0]], colors=EPI_C,
                                                 linewidths=0.3, alpha=0.5))
            ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[1] - lim, c[1] + lim)
            ax.set_zlim(c[2] - lim, c[2] + lim)
            try:
                ax.set_box_aspect((1, 1, 1), zoom=1.55)
            except TypeError:
                ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=16, azim=-58)
            ax.text2D(0.02, 0.97, f"{name}   frame {t}\n{F.shape[0]} live faces, {len(nod)} plaques"
                                  f"   $\\lambda_{{geo}}$ to {s_hi:.2f}",
                      transform=ax.transAxes, color="white", fontsize=10.5, va="top")
            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            # A ZOOM ON THE INTERFACE, not the whole ring. At full scale the standoff is l0 = 6e-4 box
            # units against a radius of 0.13 -- under one pixel -- so the question the section exists
            # to answer (is the sheet OUTSIDE the epithelium?) cannot be read off it. The window is
            # centred on the equator at +x and is a few l0 across.
            Rc = float(np.linalg.norm(XE - c, axis=1).mean())
            half = max(88.0 * L0_CACHE, 0.52 * Rc)      # x4 the previous window
            cx, cz = c[0] + Rc, c[2]
            band = 0.55 * half
            nl = np.zeros(X.shape[0]); cnt = np.zeros(X.shape[0])
            np.add.at(nl, F.reshape(-1), np.repeat(L, 3)); np.add.at(cnt, F.reshape(-1), 1)
            live = cnt > 0
            nl = nl / np.maximum(cnt, 1)
            # the epithelium as a LINE through the section, so "which side" is unambiguous
            # EVERY POINT IS DRAWN AT ITS TRUE 3D RADIUS, not at its projected one. A node sitting
            # off the section plane by dy projects to sqrt(r^2 - dy^2), which is SMALLER than r by up
            # to 0.26 l0 for this slab -- comparable to the standoff itself -- and the zigzag that
            # produced looked like roughness of the sheet. Re-plotting at (r, theta) removes the
            # artefact and leaves only whatever roughness is real.
            def unroll(P):
                rr = np.linalg.norm(P - c, axis=1)
                th = np.arctan2(P[:, 2] - c[2], P[:, 0] - c[0])
                return c[0] + rr * np.cos(th), c[2] + rr * np.sin(th)
            se = (np.abs(XE[:, 1] - c[1]) < band) & (XE[:, 0] > c[0])
            if se.sum() > 2:
                ex, ez = unroll(XE[se])
                o = np.argsort(np.arctan2(ez - c[2], ex - c[0]))
                a2.plot(ex[o], ez[o], "-", color=EPI_C, lw=1.6, zorder=2)
                a2.scatter(ex, ez, s=22, c=EPI_C, marker="o", linewidths=0, zorder=3)
            sel = (np.abs(X[nod][:, 1] - c[1]) < band) & (X[nod][:, 0] > c[0])
            if sel.any():
                px, pz = unroll(PP[sel]); qx, qz = unroll(X[nod][sel])
                a2.add_collection(LineCollection(
                    np.stack([np.stack([px, pz], 1), np.stack([qx, qz], 1)], 1), colors=PLQ_C,
                    linewidths=1.4, zorder=4))
            sl = live & (np.abs(X[:, 1] - c[1]) < band) & (X[:, 0] > c[0])
            if sl.sum() > 2:   # the sheet as a LINE too, so "which side" is read and not inferred
                sx, sz = unroll(X[sl])
                o2 = np.argsort(np.arctan2(sz - c[2], sx - c[0]))
                a2.plot(sx[o2], sz[o2], "-", color="#9ad2ff", lw=1.0, alpha=0.7, zorder=5)
                a2.scatter(sx, sz, s=14, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP, vmin=1.0,
                           vmax=s_hi, marker="o", linewidths=0, zorder=6)
            # a scale bar of one l0, so the standoff is read against a length and not against the axes
            a2.plot([cx - 0.85 * half, cx - 0.85 * half], [cz - half * 0.85,
                                                           cz - half * 0.85 + L0_CACHE],
                    "-", color="white", lw=2.5, zorder=7)
            a2.text(cx - 0.80 * half, cz - half * 0.85, r"$\ell_0$", color="white", fontsize=9,
                    va="bottom", zorder=7)
            a2.set_xlim(cx - half, cx + half); a2.set_ylim(cz - half, cz + half)
            a2.set_aspect("equal"); a2.axis("off")
            gap = float(np.linalg.norm(X[live] - c, axis=1).mean() - Rc)
            a2.text(0.02, 0.98, f"zoom on the interface, {2*half:.4f} box units across\n"
                                f"cream line = epithelium, red = plaques ($\\ell_0$ = {L0_CACHE:.1e})\n"
                                f"sheet is {'OUTSIDE' if gap > 0 else 'INSIDE'} by "
                                f"{abs(gap):.2e} = {abs(gap)/L0_CACHE:.1f} $\\ell_0$",
                    transform=a2.transAxes, color="white" if gap > 0 else "#ff8080", fontsize=9,
                    va="top")
            wri.grab_frame()
            if i in strip_at:
                strip.append((t, X.copy(), nl.copy(), live.copy(), XE.copy(), F.shape[0], len(nod)))
    fig.savefig(os.path.join(d, "3d.png"), dpi=115, facecolor="black")
    plt.close(fig)
    figs = plt.figure(figsize=(3.0 * len(strip), 3.5), facecolor="black")
    for i, (t, X, nl, live, XE, nf, npq) in enumerate(strip):
        a = figs.add_subplot(1, len(strip), i + 1, facecolor="black")
        se = np.abs(XE[:, 1] - 0.5) < 0.012
        a.scatter(XE[se][:, 0], XE[se][:, 2], s=6, c=EPI_C, marker=".", linewidths=0, alpha=0.7)
        sl = live & (np.abs(X[:, 1] - 0.5) < 0.004)
        a.scatter(X[sl][:, 0], X[sl][:, 2], s=7, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP, vmin=1.0,
                  vmax=s_hi, marker=".", linewidths=0)
        a.set_xlim(0.335, 0.665); a.set_ylim(0.335, 0.665); a.set_aspect("equal"); a.axis("off")
        a.text(0.03, 0.97, f"frame {t}\n{nf} faces, {npq} plaques", transform=a.transAxes,
               color="white", fontsize=10, va="top")
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=115, facecolor="black")
    plt.close(figs)


F_EPI_CACHE = None
L0_CACHE = 6.0e-4


# =============================================================================================
def main():
    global F_EPI_CACHE, L0_CACHE

    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)
    name = arg("--name", str, "05e_conserve")
    sigma_T = arg("--sigma", float, 7.0)
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)

    cert = BM.selftest(dev=dev, subdiv=4)
    for k, lim in (("remesh_mass_rel", 1e-12), ("remesh_volume_rel", 1e-12),
                   ("remesh_area_rel", 1e-12), ("remesh_centroid_rel", 1e-12),
                   ("refine_loaded_dlambda", 1e-12)):
        assert cert[k] < lim, f"{k} = {cert[k]:.3e}, gate is {lim:.0e}"
    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, sigma_T=sigma_T, zeta=20.0,
             s_target=1.0, k_drive=50.0, dev=dev)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 160))).astype(int).tolist())

    nom = Rig05e(**P, max_refine=2, edge_trigger=1.45, reseed=True)
    F_EPI_CACHE = nom.F_epi.cpu().numpy(); L0_CACHE = nom.l0
    print(f"[{name}] Sigma = {sigma_T:g} T = {nom.sigma:.4g} box units; {nom.res and ''}"
          f"{int(nom.plq.node.numel())} plaques seeded on {nom.sheet.m} faces "
          f"(one per {float(nom.sheet.area().sum())/max(int(nom.plq.node.numel()),1):.3g} of area, "
          f"target {nom.sigma**2:.3g})", flush=True)
    kept, _ = run(nom, frames, keep=keep, label=name)

    # THE CONTROL FOR G18c: the same run with re-seeding OFF, which is what 05b did. If the plaque
    # density does not fall 4x at each refinement here, the fix was not fixing anything.
    noseed = Rig05e(**P, max_refine=2, edge_trigger=1.45, reseed=False)
    run(noseed, frames, label="control: plaques seeded as a FRACTION of nodes (05b's rule)")

    s_hi = float(np.percentile(np.concatenate([k[2] for k in kept[::4]]), 99))
    render(kept, d, name, s_hi)
    model_png(nom, d, sigma_T)

    ev = nom.events
    out = dict(
        run=name, frames=frames, certification=cert,
        rig=dict(**{k: v for k, v in P.items() if k != "dev"}, l0=nom.l0, sigma_box=nom.sigma,
                 face_slots=int(nom.sheet.F_all.shape[0]), node_slots=int(nom.sheet.x.shape[0]),
                 faces_first=nom.res["n_faces"][0], faces_final=nom.res["n_faces"][-1]),
        refinements=ev,
        G18b=dict(mass_first=nom.res["mass"][0], mass_final=nom.res["mass"][-1],
                  relative_change=abs(nom.res["mass"][-1] - nom.res["mass"][0])
                  / max(abs(nom.res["mass"][0]), 1e-30),
                  worst_across_a_split=max([e["mass_rel"] for e in ev] or [0.0])),
        G18c=dict(density_over_sigma2_first=nom.res["plq_density"][0],
                  density_over_sigma2_final=nom.res["plq_density"][-1],
                  density_min=min(nom.res["plq_density"]), density_max=max(nom.res["plq_density"]),
                  plaques_first=nom.res["n_plaque"][0], plaques_final=nom.res["n_plaque"][-1],
                  control_no_reseed_final=noseed.res["plq_density"][-1],
                  control_plaques_final=noseed.res["n_plaque"][-1]),
        standoff=dict(signed_first=nom.res["standoff"][0], signed_final=nom.res["standoff"][-1],
                      radial_gap_first=nom.res["radial_gap"][0],
                      radial_gap_final=nom.res["radial_gap"][-1],
                      fraction_inside_final=nom.res["inside_frac"][-1], l0=nom.l0,
                      note="SIGNED along the epithelial outward normal. The unsigned version reported "
                           "+4.0e-3 for a sheet that had sunk 7.7e-3 INSIDE the epithelium."),
        G18d=dict(volume_worst=max([e["volume_rel"] for e in ev] or [0.0]),
                  area_worst=max([e["area_rel"] for e in ev] or [0.0]),
                  centroid_worst=max([e["centroid_rel"] for e in ev] or [0.0]),
                  lambda_worst=max([e["lam_rel"] for e in ev] or [0.0]),
                  energy_worst=max([e["energy_rel"] for e in ev] or [0.0])),
        G18e=dict(rho_final=nom.res["rho"][-1], rho_predicted_1_over_J=nom.res["rho_pred"][-1],
                  relative_difference=abs(nom.res["rho"][-1] - nom.res["rho_pred"][-1])
                  / max(nom.res["rho_pred"][-1], 1e-30)),
        momentum=dict(max=max(nom.res["momentum"]), median=float(np.median(nom.res["momentum"]))),
        series={k: [float(x) for x in v] for k, v in nom.res.items()})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    np.savez_compressed(
        os.path.join(d, "traj.npz"), frames=np.asarray([k[0] for k in kept]),
        **{f"pos_{k[0]}": k[1].astype(np.float32) for k in kept},
        **{f"lam_{k[0]}": k[2].astype(np.float32) for k in kept},
        **{f"faces_{k[0]}": k[3].astype(np.int32) for k in kept},
        **{f"plaque_{k[0]}": k[5].astype(np.int32) for k in kept},
        # the epithelium too, so the section can be re-rendered without re-running the physics
        **{f"epi_{k[0]}": k[4].astype(np.float32) for k in kept},
        faces_epi=nom.F_epi.cpu().numpy().astype(np.int32),
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
        what="05e -- what a remesh must carry: mass, the physical surface, and the plaque density",
        correction="an external audit: a remesh is NUMERICAL ONLY. It changes the triangulation and "
                   "nothing else. Secretion, remodelling and proteolysis are the biological operators; "
                   "they share the occ reservoir with refinement and nothing else.",
        fixes=["mass is now the per-face STATE and rho = mass/area is derived, so a split conserves "
               "material exactly and the dilution term is not integrated twice",
               "plaque_seed takes Sigma in units of T and tops the count up to A(t)/Sigma^2 every "
               "frame, instead of anchoring a fixed fraction of nodes"],
        gates=dict(G18b="sum_f mass_f invariant across a split", G18c="plaque areal density at "
                   "Sigma^-2 across two refinements", G18d="enclosed volume, area and centroid "
                   "unchanged by a split", G18e="rho = mass/area equals the 1/J a fixed mesh predicts"),
        rig=dict(sigma_in_T=sigma_T, l0_in_T=0.3, thickness=P["thickness"], subdiv=P["subdiv"],
                 max_refine=2, edge_trigger=1.45, epithelium="driven icosphere, subdiv 3",
                 surface=os.path.basename(TISSUE), scale=SCALE),
        not_modelled=["secretion and proteolysis (mass is conserved here BECAUSE nothing may change "
                      "it, which is what makes this the control for 05f)",
                      "plaque turnover -- plaques are added and never removed (05h)",
                      "the vertex-model epithelium", "bending", "the matrix"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{name}] G18b mass {out['G18b']['worst_across_a_split']:.2e} across a split, "
          f"{out['G18b']['relative_change']:.2e} over the run; G18c density "
          f"{out['G18c']['density_over_sigma2_final']:.4f} of Sigma^-2 "
          f"({out['G18c']['plaques_first']} -> {out['G18c']['plaques_final']} plaques) against the "
          f"no-reseed control's {out['G18c']['control_no_reseed_final']:.4f}; G18d volume "
          f"{out['G18d']['volume_worst']:.2e}, centroid {out['G18d']['centroid_worst']:.2e}; "
          f"G18e rho {out['G18e']['relative_difference']:.2e}; standoff "
          f"{out['standoff']['signed_final']:+.3e} signed (radial gap "
          f"{out['standoff']['radial_gap_final']:+.3e}, "
          f"{100*out['standoff']['fraction_inside_final']:.1f}% of plaques inside) -> {d}",
          flush=True)


if __name__ == "__main__":
    main()
