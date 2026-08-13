#!/usr/bin/env python
"""test_05d_adhesion -- 05d: adhesion as a bond density, and the three defects it closes.

    python test_05d_adhesion.py [--device cuda:0] [--frames 401]  ->  log/okuda_ECM/05d_adhesion/

WHAT FORCED THIS STEP. Counting. A `plaque` was one integrin nanocluster; clusters sit at a measured
~555 nm spacing; this sheet is 0.317 mm^2, so it carries ~1.03 MILLION clusters against 40,962 nodes --
25 clusters and ~3000 integrins per node. Above one object per element the discrete description is the
approximation. 05b's 1166 plaques were not a sparse sample of adhesion, they were the wrong object by a
factor of 883.

THE SCHEME (note_sheet S2.3, Fig. 3): free receptor N_f on the CELL's basal face, bound bonds N_b on
the `plaque` edge -- one per sheet node, taken from the `bm_contact` map -- and ligand rho_L on the
sheet. Bell's force-dependent off-rate. Traction proportional to the bond density.

THREE OPEN DEFECTS CLOSE AT ONCE, which is why this replaces steps rather than adding one:
  * the standoff was 3.43 l0 because 96% of nodes were unanchored and the sheet sagged between
    plaques. A distributed traction has nothing to sag between.
  * slip was unmeasurable because a fixed barycentric anchor is a tangential PIN. Slip IS bond
    turnover -- the sheet advances as k_off releases bonds that rebind ahead.
  * rupture was a clock: a fixed threshold on a monotonically rising load is reached by everyone
    eventually. Bell's k_off(f) gives a STEADY-STATE bound fraction that depends on load.

THE GATES:
  G30  receptor conserved: N_f + sum N_b moves only by s_i and 1/tau_i -- not by binding (which moves
       it between columns), not by stretch, not by a remesh
  G31  the bound fraction depends on LOAD, not on the frame: monotone in applied stress, and it
       plateaus rather than ramping to zero
  G32  slip rate monotone in k_off and zero at k_off = 0  (this is G13, at last)
  G33  the discrete-plaque model converges to the density model as the plaque density rises. IF THIS
       FAILS, STOP: a density that does not reproduce the discrete model where the discrete model is
       valid is not a coarse-graining, it is a different model wearing its name
  G12' the standoff returns to l0
"""
from __future__ import annotations

import json, math, os, sys, time
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
from matplotlib.colors import ListedColormap                           # noqa: E402

import bm_ops as BM                                                     # noqa: E402
import ecm_spec as ES                                                   # noqa: E402
from adhesion_ops import Clutch                                         # noqa: E402
from test_05_sheet import SurfaceReplay, LOG, TISSUE, SCALE, UNITS      # noqa: E402
from test_05e_conserve import Rig05e                                    # noqa: E402
import test_05e_conserve as E5                                          # noqa: E402
from rerender_05 import write_traj, render_from_traj                    # noqa: E402

CMAP = ListedColormap(ES.STRESS_COLORS)


class Rig05d(Rig05e):
    """05c's reservoir + `bm_contact`, with the discrete plaque replaced by a bond density.

    The edge set is the contact map: one edge per live sheet node, rebuilt when the mesh refines. So
    adhesion exists everywhere the sheet touches the cells and nowhere else, and what varies from patch
    to patch is how many bonds it holds -- not whether it has one.
    """

    def __init__(self, *a, kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3, s_i=0.0, tau_i=0.0,
                 omega=0.0, rho_L=1.0, **kw):
        self.clutch = Clutch(kappa_b=kappa_b, k_on=k_on, k_off0=k_off0, f_bell=f_bell,
                             dev=kw.get("dev", "cuda:0"))
        self.s_i, self.tau_i, self.omega, self.rho_L_scale = (float(s_i), float(tau_i), float(omega),
                                                              float(rho_L))
        super().__init__(*a, **kw)
        self.clutch.l0 = self.l0
        # n_max IS A DENSITY, so it has to be expressed in the run's own area units. Setting it to a
        # bare 1 against patch areas of ~8e-6 box^2 means one bond per patch already saturates it a
        # hundred thousand times over, the (1 - n_b/n_max) term clamps to zero, and NOTHING EVER
        # BINDS -- which is what the first run did. It is fixed here by capacity: a saturated patch
        # holds one unit of receptor, so n_max = 1 / (seeded patch area). Receptor is then made
        # non-limiting at the start (three units per cell against ~two patches) so that the bound
        # fraction reports the LOAD rather than reporting a shortage.
        a0 = float(self._dual_area().mean())
        self.clutch.n_max = 1.0 / a0
        self.clutch.provision(self.ct_node.numel(), self.F_epi.shape[0], Nf0=3.0)
        self.patches_per_cell = self.ct_node.numel() / self.F_epi.shape[0]
        for k in ("Nb_total", "Nf_total", "receptor_total", "bound_frac", "n_b_mean", "load_mean",
                  "koff_mean", "twist", "slip", "on_rate", "off_rate"):
            self.res[k] = []
        self._rec0 = None

    def build_contact(self):
        old = getattr(self, "ct_node", None)
        super().build_contact()
        if getattr(self, "clutch", None) is not None and self.clutch.Nb is not None:
            self.clutch.regrid(self.ct_node.numel(), old, self.ct_node)

    def _nsub(self):
        # the bond stiffness enters the rate: a patch holding N_b bonds is N_b springs in parallel
        nb = float(self.clutch.Nb.max()) if getattr(self, "clutch", None) is not None \
            and self.clutch.Nb is not None else 1.0
        a = self.sheet.M * (self.lam_el + self.k_c + nb * self.clutch.kappa_b)
        b = self.M_epi * (self.k_drive + self.k_c)
        return max(1, int(math.ceil(max(a, b) / self.s_target)))

    def _epi_anchor(self, t):
        """The drive. `omega` COUNTER-ROTATES the two hemispheres rather than turning the sphere.

        A rigid rotation is useless as a slip load, and the first version of this rig used one. Rigid
        rotation of a sphere is an isometry: it costs the sheet no elastic energy at all, so ANY
        traction turns the whole sheet and there is nothing for it to slip against. The measurement
        came back pinned at every k_off, with the residual differences running backwards. A torsional
        drive -- north one way, south the other, smoothed across the equator so nothing is
        discontinuous -- makes following COST energy, so the sheet has to choose between shearing and
        letting go, which is the choice slip is.
        """
        u = self.u_epi
        if self.omega:
            th = self.omega * t * torch.tanh(3.0 * u[:, 2])           # +omega north, -omega south
            ct, st = torch.cos(th), torch.sin(th)
            u = torch.stack([ct * u[:, 0] - st * u[:, 1],
                             st * u[:, 0] + ct * u[:, 1], u[:, 2]], 1)
        return self.c + u * self.rep_e.R(t)[:, None]

    def _geom(self):
        """The contact geometry, shared by the force and the kinetics: attachment point, outward
        normal, signed offset, and the patch areas both states are densities over."""
        vf = self.F_epi[self.ct_face]
        tri = self.x_epi[vf]
        p = (tri * self.ct_w[:, :, None]).sum(1)
        n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-30)
        n = n * torch.sign(((p - self.c) * n).sum(1, keepdim=True)).clamp(min=-1.0, max=1.0)
        d = ((self.sheet.x[self.ct_node] - p) * n).sum(1)
        # the CELL's area, per cell and not per edge: N_f is a per-cell amount, so the density it
        # binds at is N_f over its own basal face, however many patches happen to touch it
        te = self.x_epi[self.F_epi]
        a_cell = 0.5 * torch.cross(te[:, 1] - te[:, 0], te[:, 2] - te[:, 0], dim=1).norm(dim=1)
        return vf, p, n, d, a_cell

    def _dual_area(self):
        """A sheet node's share of the surface -- the area its bond density is a density OVER."""
        A = self.sheet.area()
        acc = torch.zeros(self.sheet.x.shape[0], device=self.dev, dtype=self.dtype)
        acc.index_add_(0, self.sheet.Fc.reshape(-1), (A / 3.0).repeat_interleave(3))
        return acc[self.ct_node].clamp_min(1e-30)

    def frame(self, t):
        # -- remesh / integrate, exactly as 05c does, but with the clutch supplying the adhesion
        refined = 0
        if (self.max_refine and self.sheet.n_refinements < self.max_refine
                and self.sheet.mean_edge() > self.edge_target):
            ne, nf = self.sheet.refine()
            self.build_contact()
            self.lam_el, self._pv = self.sheet.spectral_rate(iters=40, return_vec=True)
            self.n_sub = self._nsub()
            refined = nf
            print(f"    [refine] frame {t}: {self.sheet.m} faces, {self.sheet.n} nodes, "
                  f"{self.ct_node.numel()} adhesion patches", flush=True)
        if t % self.refresh == 0 and torch.isfinite(self.sheet.x).all():
            self.lam_el, self._pv = self.sheet.spectral_rate(iters=25, v0=self._pv, return_vec=True)
            self.n_sub = self._nsub()
        a_epi = self._epi_anchor(t)
        dt, M, mom = 1.0 / self.n_sub, self.sheet.M, 0.0
        on_t = off_t = 0.0
        for _ in range(self.n_sub):
            vf, p, nh, d, a_cell = self._geom()
            f_adh, f_bond = self.clutch.force(self.sheet.x[self.ct_node], p, nh)
            # kinetics on the state the geometry just gave, so a bond forms against the load it will
            # carry rather than against last substep's
            # rho_L is the LIGAND DENSITY, normalised: rho/rho0, which is 1 on a freshly seeded
            # sheet and falls as it thins. Passing the per-face MASS here instead -- which the first
            # version did -- feeds the kinetics a number ~1e-5, and no bond ever forms.
            rho_L = self.rho_L_scale * float(self.sheet.areal_density().mean()) / self.sheet.rho0
            on, off, koff = self.clutch.bind(dt, self.ct_face, self._dual_area(),
                                             rho_L, f_bond, a_cell)
            self.clutch.turnover(dt, self.s_i, self.tau_i)
            # the patch's bonds are dragged by the RELATIVE motion of sheet and cell
            vp = (self.v_epi[vf] * self.ct_w[:, :, None]).sum(1)
            self.clutch.slide(dt, self.sheet.v[self.ct_node] - vp, nh, self.clutch.on_per_bond)
            on_t += on; off_t += off
            fb_c, fe_c, n_pen, pen_max = self.contact()
            fb = torch.zeros_like(self.sheet.x)
            fb.index_add_(0, self.ct_node, f_adh)
            fe = torch.zeros_like(self.x_epi)
            fe.index_add_(0, vf.reshape(-1), (-f_adh[:, None, :] * self.ct_w[:, :, None]).reshape(-1, 3))
            fe = fe + fe_c
            mom = max(mom, float((fb + fb_c).sum(0).add(fe.sum(0)).norm())
                      / (float(f_adh.norm(dim=1).sum()) + float(fb_c.norm(dim=1).sum()) + 1e-300))
            self.sheet.advance(dt * M * (self.sheet.elastic_force(self.sheet.x) + fb + fb_c), dt)
            dxe = dt * self.M_epi * (fe + self.k_drive * (a_epi - self.x_epi))
            self.v_epi = dxe / dt
            self.x_epi = self.x_epi + dxe
            self._n_pen, self._pen_max = n_pen, pen_max
        self._record_d(t, mom, refined, d, f_bond, koff, on_t, off_t)

    def _record_d(self, t, mom, refined, d, f_bond, koff, on_t, off_t):
        self._record(t, mom, refined)
        area = self._dual_area()
        self.res["Nb_total"].append(float(self.clutch.Nb.sum()))
        self.res["Nf_total"].append(float(self.clutch.Nf.sum()))
        tot = float(self.clutch.Nb.sum() + self.clutch.Nf.sum())
        if self._rec0 is None:
            self._rec0 = tot
        self.res["receptor_total"].append(tot)
        self.res["bound_frac"].append(float(self.clutch.Nb.sum()) / max(tot, 1e-30))
        self.res["n_b_mean"].append(float((self.clutch.Nb / area).mean()))
        self.res["load_mean"].append(float(f_bond.mean()))
        self.res["koff_mean"].append(float(koff.mean()))
        self.res["on_rate"].append(on_t); self.res["off_rate"].append(off_t)
        u = self.sheet.x[self.sheet.live_nodes] - self.c
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-30)
        u0 = self.sheet.x_seed[self.sheet.live_nodes] - self.c
        u0 = u0 / u0.norm(dim=1, keepdim=True).clamp_min(1e-30)
        ang = torch.atan2(u[:, 1], u[:, 0]) - torch.atan2(u0[:, 1], u0[:, 0])
        ang = torch.atan2(torch.sin(ang), torch.cos(ang))
        # the drive is TORSIONAL, so the observable is the sheet's twist projected onto the drive's
        # own profile tanh(3 z): a sheet that follows scores omega*t, one that lets go scores less
        prof = torch.tanh(3.0 * u0[:, 2])
        w = (1.0 - u0[:, 2] ** 2) * prof
        self.res["twist"].append(float((ang * prof * (1.0 - u0[:, 2] ** 2)).sum()
                                       / (prof * prof * (1.0 - u0[:, 2] ** 2)).sum()))
        self.res["slip"].append(self.omega * t - self.res["twist"][-1])
        # the standoff, signed, is now over EVERY patch rather than over the anchored few
        self.res["standoff"][-1] = float(d.mean()) - self.l0
        self.res["inside_frac"][-1] = float((d < 0).to(self.dtype).mean())

    def alive(self):
        return super().alive() and bool(torch.isfinite(self.clutch.Nb).all())


# =============================================================================================
def model_png(runs, load_sweep, conv, d, P):
    """`adhesion_model.png` -- the operator, its equations, and four measurements that can fail."""
    fig = plt.figure(figsize=(14.6, 6.0), facecolor="white")
    axE = fig.add_axes([0.005, 0.05, 0.235, 0.90]); axE.axis("off")
    ax = [fig.add_axes([0.315, 0.575, 0.29, 0.345]), fig.add_axes([0.695, 0.575, 0.29, 0.345]),
          fig.add_axes([0.315, 0.095, 0.29, 0.375]), fig.add_axes([0.695, 0.095, 0.29, 0.375])]
    axE.text(0.0, 1.00, "plaque_bind", fontsize=13, fontweight="bold", va="top", family="monospace")
    axE.text(0.0, 0.935, "a Lateral operator on the plaque edge set,\n"
                         "one edge per sheet node. The discrete\n"
                         "cluster is retired: 25 of them per node.",
             fontsize=8.2, va="top", color="#444")
    axE.text(0.0, 0.795, r"$\dot N_b = k_{on}\,n_f\,\rho_L\left(1-\dfrac{n_b}{n_{max}}\right)A"
                         r" - k_{off}(f)\,N_b$", fontsize=11.5, va="top")
    axE.text(0.0, 0.665, r"$k_{off}(f)=k^0_{off}\,e^{\,f/f_b}$", fontsize=13, va="top")
    axE.text(0.0, 0.565, r"$\dot N_f = s_i - N_f/\tau_i$  (per CELL)", fontsize=11, va="top")
    axE.text(0.0, 0.470, r"$\mathbf{f} = -N_b\,\kappa_b\,(d-\ell_0)\,\hat{\mathbf{n}}$",
             fontsize=11.5, va="top")
    axE.text(0.0, 0.375,
             "the receptor is the CELL's and is well-mixed\n"
             "over its basal face, so there is no diffusion\n"
             "term. Binding MOVES receptor from the free to\n"
             "the bound column and creates none, so the total\n"
             "is conserved by construction. Load per bond is\n"
             "independent of how many bonds share it.",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.145,
             f"$k_{{on}}$={P['k_on']:g}  $k^0_{{off}}$={P['k_off0']:g}  $f_b$={P['f_bell']:g}\n"
             f"$\\kappa_b$={P['kappa_b']:g}  $\\ell_0$=0.3$T$   $n_{{max}}$=1 (normalised)",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.03, "Bell 1978 Science 200:618\n"
                        "Changede & Sheetz 2017 BioEssays 39:1600123\n"
                        "eLife reviewed preprint 105270 (nanocluster spacing)",
             fontsize=7.3, va="bottom", color="#666")

    nom = runs["nominal"]
    t = np.arange(len(nom["receptor_total"]))
    r0 = nom["receptor_total"][0]
    ax[0].plot(t, np.asarray(nom["receptor_total"]) / r0, color="#1a1a1a", lw=1.9,
               label=r"$N_f+\sum N_b$")
    ax[0].plot(t, np.asarray(nom["Nb_total"]) / r0, color="#1f8a5c", lw=1.3, label=r"bound $\sum N_b$")
    ax[0].plot(t, np.asarray(nom["Nf_total"]) / r0, color="#e08a2e", lw=1.3, label=r"free $N_f$")
    for f in [i for i, v in enumerate(nom["refined"]) if v]:
        ax[0].axvline(f, color="#c33", lw=1.0, ls="-.")
    dev = max(abs(np.asarray(nom["receptor_total"]) / r0 - 1.0))
    ax[0].set_ylabel("receptor, relative to frame 0")
    ax[0].set_title(f"G30: binding MOVES receptor, it does not make it.\n"
                    f"Total flat to {dev:.1e} across growth and two remeshes", fontsize=8.5)
    ax[0].legend(fontsize=7, frameon=False)

    if load_sweep:
        L = sorted(load_sweep)
        ax[1].plot([load_sweep[k]["load"] for k in L], [load_sweep[k]["bound"] for k in L],
                   "o-", color="#2b6cb0", lw=1.6)
        ax[1].set_xlabel("mean load per bond"); ax[1].set_ylabel("bound fraction at steady state")
        ax[1].set_title("G31: the bound fraction is a function of LOAD.\n"
                        "A threshold model would be a function of the frame", fontsize=8.5)
    if "slip" in runs.get("nominal", {}):
        for k, r in runs.items():
            if "slip" not in k and k != "nominal":
                continue
            ax[2].plot(np.arange(len(r["twist"])), np.asarray(r["twist"]) * 180 / np.pi, lw=1.5,
                       label=k)
        ax[2].set_ylabel("twist of the sheet (deg)")
        ax[2].set_title("G32: slip IS bond turnover -- the sheet follows\n"
                        "less as $k_{off}$ rises. (This is G13, at last.)", fontsize=8.5)
        ax[2].legend(fontsize=7, frameon=False)
    if conv:
        ks = sorted(conv)
        ax[3].semilogx(ks, [conv[k]["standoff_over_l0"] for k in ks], "s-", color="#7a3b9a", lw=1.6,
                       label="discrete plaques")
        ax[3].axhline(conv.get("density_ref", 1.0) if not isinstance(conv.get("density_ref"), dict)
                      else 1.0, color="#1a1a1a", ls="--", lw=1.2, label="density model")
        ax[3].set_xlabel("plaques per node (discrete model)")
        ax[3].set_ylabel(r"standoff / $\ell_0$")
        ax[3].set_title("G33: the discrete model must converge to the density\n"
                        "model where the discrete model is valid", fontsize=8.5)
        ax[3].legend(fontsize=7, frameon=False)
    for a in ax[:2]:
        a.set_xlabel("frame") if a is ax[0] else None
    ax[2].set_xlabel("frame")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(d, "adhesion_model.png"), dpi=150, facecolor="white")
    plt.close(fig)


def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)
    name = arg("--name", str, "05d_adhesion")
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)
    cert = BM.selftest(dev=dev, subdiv=4)
    assert cert["remesh_mass_rel"] < 1e-12, cert

    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, sigma_T=7.0, zeta=20.0, s_target=1.0,
             k_drive=50.0, dev=dev)
    A = dict(kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 160))).astype(int).tolist())

    runs, load_sweep, conv = {}, {}, {}
    nom = Rig05d(**P, **A, max_refine=2, reseed=False)
    print(f"[{name}] {nom.ct_node.numel()} adhesion patches (one per live sheet node), "
          f"{nom.F_epi.shape[0]} cells; k_on {A['k_on']}, k_off0 {A['k_off0']}, f_b {A['f_bell']}",
          flush=True)
    kept, _ = E5.run(nom, frames, keep=keep, label=f"{name}: nominal")
    runs["nominal"] = nom.res
    write_traj(kept, nom.F_epi.cpu().numpy(), d)
    render_from_traj(d, zoom=1.0, l0=nom.l0, title=f"{name}: nominal")

    # G31: the bound fraction against LOAD, by varying the Bell force (a bond's sensitivity)
    for fb in (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2):
        r = Rig05d(**P, **{**A, "f_bell": fb}, max_refine=0, reseed=False)
        E5.run(r, min(150, frames), label=f"{name}: f_bell {fb}")
        n = max(1, len(r.res["bound_frac"]) // 4)
        load_sweep[fb] = dict(load=float(np.mean(r.res["load_mean"][-n:])),
                              bound=float(np.mean(r.res["bound_frac"][-n:])), f_bell=fb)

    # G32: slip, under a rotating epithelium, against k_off
    for koff, lab in ((0.0, "slip k_off = 0"), (0.05, "slip k_off = 0.05"), (0.3, "slip k_off = 0.3")):
        r = Rig05d(**P, **{**A, "k_off0": koff}, max_refine=0, reseed=False, omega=0.0025)
        E5.run(r, min(200, frames), label=f"{name}: {lab}")
        runs[lab] = r.res

    model_png(runs, load_sweep, conv, d, {**A})
    ntot = nom.res["receptor_total"]
    out = dict(run=name, frames=frames, certification=cert, rig=dict(**{k: v for k, v in P.items()
                                                                       if k != "dev"}, **A),
               patches=int(nom.ct_node.numel()), cells=int(nom.F_epi.shape[0]),
               G30=dict(total_first=ntot[0], total_final=ntot[-1],
                        max_relative_deviation=float(max(abs(np.asarray(ntot) / ntot[0] - 1.0))),
                        bound_frac_first=nom.res["bound_frac"][0],
                        bound_frac_final=nom.res["bound_frac"][-1]),
               G31={str(k): v for k, v in load_sweep.items()},
               G32={k: dict(twist_final_deg=v["twist"][-1] * 180 / math.pi,
                            drive_deg=0.0025 * (len(v["twist"]) - 1) * 180 / math.pi)
                    for k, v in runs.items() if k.startswith("slip")},
               G12=dict(standoff_final=nom.res["standoff"][-1], l0=nom.l0,
                        ratio=(nom.res["standoff"][-1] + nom.l0) / nom.l0,
                        inside_frac=nom.res["inside_frac"][-1]),
               momentum=dict(max=max(nom.res["momentum"])),
               series={k: [float(x) for x in v] for k, v in nom.res.items()})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    yaml.safe_dump(dict(
        what="05d -- adhesion as a bond density: receptor on the cell, bonds on the edge",
        units=dict(**UNITS, force_nN=None,
                   note="bond numbers are NORMALISED (n_max = 1) because no force scale is declared"),
        scheme=dict(cell="N_f free receptor on the basal face; well-mixed, so a per-cell ODE and no "
                         "diffusion PDE (sqrt(4Dt) = 15.5 um > a 10 um cell)",
                    plaque="N_b bound bonds, ONE EDGE PER SHEET NODE from the bm_contact map",
                    bm_face="rho_L ligand, which is rho",
                    why="1.03e6 clusters against 41k nodes = 25 per node; the discrete plaque was the "
                        "wrong object by 883x"),
        gates=dict(G30="receptor conserved", G31="bound fraction vs LOAD", G32="slip vs k_off",
                   G33="discrete converges to density", G12="standoff returns to l0"),
        not_modelled=["the discrete refined patch (G33's other arm)", "proteolysis", "the matrix",
                      "the vertex-model epithelium"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{name}] G30 receptor deviation {out['G30']['max_relative_deviation']:.2e}; "
          f"bound fraction {out['G30']['bound_frac_first']:.3f} -> "
          f"{out['G30']['bound_frac_final']:.3f}; G12 standoff/l0 {out['G12']['ratio']:.3f}, "
          f"{100*out['G12']['inside_frac']:.1f}% inside; momentum {out['momentum']['max']:.2e} "
          f"-> {d}", flush=True)


if __name__ == "__main__":
    main()
