#!/usr/bin/env python
"""test_05h1_hetero -- the ternary mechanism with SPATIALLY VARYING rates, per cell.

    python test_05h1_hetero.py [--device cuda:0] [--frames 300]  ->  log/okuda_ECM/05h_1_hetero/

WHY THE RATES HAVE TO BECOME STATES. In 05g and 05h every rate was a global scalar -- one `s_timp`,
one `s_pro`, and MT1 a binary flag on 3% of cells -- so the only spatial information in the whole
model was the MT1 stencil, and every hole came out as a copy of it. That is why G46 (the breach must be
LARGER than its source) could not be met: with a uniform inhibitor field there was nothing for the
breach to inherit a shape from except the source itself.

WHAT CHANGES WHEN THE RATES ARE PER-CELL STATES. The bell of 05h says activation peaks at c_T = K, so a
TIMP field that VARIES puts different regions at different points on the same curve. A cell expressing
MT1 in a TIMP-rich region is inhibited; the same cell where TIMP has fallen to K is maximally active.
The prediction is therefore not a patch but a BAND -- activation is highest along the contour where
c_T ~ K, which sits between the inhibitor's sources and its sinks and need not touch any MT1 cell at
its centre. That is a pattern the source map does not contain, which is what makes it a mechanism
rather than a stencil.

THREE NEW PER-CELL STATES (`cell` gains rate fields, exactly as it gained N_f in 05d):

    cell.mt1     graded expression of the tethered activator      was a binary flag
    cell.s_timp  this cell's inhibitor secretion                  was one global number
    cell.s_pro   this cell's zymogen secretion                    was one global number

each seeded as a smooth random field on the epithelium so that the heterogeneity is spatial rather
than salt-and-pepper: a per-cell rate drawn independently would be white noise, and diffusion would
average it away within one frame, leaving exactly the uniform field 05g had.

AND THAT IS NOT ENOUGH, WHICH THE FIRST VERSION OF THIS RUN MEASURED. With per-cell sources but a
SOLUBLE inhibitor, activation still came back at corr = 0.970 with the source map and 100% of faces
inside the bell's peak band: the field was uniform anyway. The arithmetic says it must be. A soluble
species holds structure only over sqrt(D*tau), and at D = 30 um^2/s a 50 um pattern needs a clearance
time of 83 s -- 0.14 of a frame. At a 600 s frame NO SOLUBLE SPECIES can hold a pattern finer than the
whole spheroid.

THE ESCAPE IS BIOLOGICAL AND SPECIFIC. TIMP-3 is the one TIMP that is not soluble: it is sequestered in
the matrix by binding sulfated glycosaminoglycans through its N-terminal domain (Yu et al. 2000, JBC
275:31226), and sulfated GAGs also block its LRP-1-mediated clearance (Troeberg et al. 2014). An
ECM-bound inhibitor does not diffuse, so it keeps whatever pattern it was deposited with -- it is a
per-face STATE with D = 0, not a field. That is what lets the bell vary in space, and it is why this
run carries TIMP-2 (soluble, a field) and TIMP-3 (bound, a state) as different objects rather than one
"inhibitor".
"""
from __future__ import annotations

import json, math, os, sys
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection                 # noqa: E402

import bm_ops as BM                                                     # noqa: E402
import ecm_spec as ES                                                   # noqa: E402
from protease_ops import diffuse_implicit                               # noqa: E402
from test_05_sheet import LOG, UNITS                                    # noqa: E402
from test_05h_ternary import Rig05h                                     # noqa: E402
from test_05g_degrade import run_g                                      # noqa: E402
from rerender_05 import unroll                                          # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

CMAP = ListedColormap(ES.STRESS_COLORS)
UM = 1171.0


def smooth_field(u, n_modes=6, seed=0, dev="cuda:0", dtype=torch.float64):
    """A smooth random scalar on the sphere: a few low-order modes rather than per-cell noise.

    Drawing a rate independently per cell would be white noise, and a field with D = 30 um^2/s
    averages that away inside one frame -- which is exactly how 05g ended up with a uniform field it
    could not learn anything from. Low-order modes give heterogeneity at a length the diffusion cannot
    erase, which is the only kind that can matter.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.zeros(u.shape[0], device=dev, dtype=dtype)
    for _ in range(n_modes):
        k = torch.randn(3, generator=g).to(device=dev, dtype=dtype)
        k = k / k.norm().clamp_min(1e-30)
        phase = float(torch.rand(1, generator=g))
        out = out + torch.cos(3.0 * (u @ k) + 6.2831 * phase)
    out = out - out.min()
    return out / out.max().clamp_min(1e-30)


class Rig05h1(Rig05h):
    """05h with the rates promoted from global scalars to per-cell states."""

    def __init__(self, *a, hetero=1.0, seed_mt1=3, seed_timp=7, seed_pro=11,
                 s_timp3=1.5e-5, tau_timp3=40.0, **kw):
        self.hetero = float(hetero)
        self.s_timp3, self.tau_timp3 = float(s_timp3), float(tau_timp3)
        self._seeds = (seed_mt1, seed_timp, seed_pro)
        super().__init__(*a, **kw)
        u = self.u_epi[self.F_epi].mean(1)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-30)
        h = self.hetero
        # each rate becomes a per-cell state: mean preserved, modulated by its own smooth field, so
        # `hetero = 0` reproduces the uniform runs exactly and the comparison is one parameter
        self.mt1 = (1.0 - h + h * 2.0 * smooth_field(u, seed=self._seeds[0], dev=self.dev,
                                                     dtype=self.dtype)) * self.mt1.mean().clamp_min(
            1e-30) / max(1.0 - h + h, 1e-30) if h > 0 else self.mt1
        self.mt1 = (self.mt1 / self.mt1.mean().clamp_min(1e-30)) * self.mt1_frac
        self.s_timp_cell = self.s_timp * (1.0 - h + h * 2.0 * smooth_field(
            u, seed=self._seeds[1], dev=self.dev, dtype=self.dtype))
        self.s_pro_cell = self.s_pro * (1.0 - h + h * 2.0 * smooth_field(
            u, seed=self._seeds[2], dev=self.dev, dtype=self.dtype))
        # TIMP-3: BOUND, so it is a per-face state and never diffuses. Deposited where the cells that
        # make it are, and it stays there -- which is the only way an inhibitor can hold a pattern at
        # this frame length.
        self.timp3 = torch.zeros(self.sheet.F_all.shape[0], device=self.dev, dtype=self.dtype)
        self.s_timp3_cell = self.s_timp3 * (1.0 - h + h * 2.0 * smooth_field(
            u, seed=self._seeds[1] + 101, dev=self.dev, dtype=self.dtype))
        for k in ("act_max", "act_cv", "band_frac", "timp3_mean", "timp3_cv"):
            self.res[k] = []

    def _cell_to_face(self, per_cell):
        """Carry a per-cell rate onto the sheet faces it touches, through the contact map."""
        v = torch.zeros(self.sheet.x.shape[0], device=self.dev, dtype=self.dtype)
        cnt = torch.zeros_like(v)
        v.index_add_(0, self.ct_node, per_cell[self.ct_face])
        cnt.index_add_(0, self.ct_node, torch.ones_like(self.ct_node, dtype=self.dtype))
        return (v / cnt.clamp_min(1.0))[self.sheet.Fc].mean(1)

    def fields(self, dt=1.0):
        i, j, w, a, _ = self._lapl()
        li = self.sheet.live
        mmp, timp, pro = self.mmp[li], self.timp[li], self.pro[li]
        mt1 = self._mt1_on_faces()

        pro = pro + dt * self._cell_to_face(self.s_pro_cell)
        timp = timp + dt * self._cell_to_face(self.s_timp_cell)

        pro = diffuse_implicit(pro, self.D_mmp, dt, i, j, w, a)
        mmp_d = diffuse_implicit(mmp, self.D_mmp, dt, i, j, w, a)
        timp_d = diffuse_implicit(timp, self.D_timp, dt, i, j, w, a)

        # TIMP-3 accumulates where it is deposited and is cleared slowly; NO diffusion term, which
        # is the entire point of it
        t3 = self.timp3[li] + dt * self._cell_to_face(self.s_timp3_cell)
        t3 = t3 * math.exp(-dt / self.tau_timp3)
        self.timp3[li] = t3.clamp_min(0.0)
        # both inhibitors occupy the same site, so the bell reads their SUM -- but only one of them
        # carries spatial information
        x = (timp_d + t3) / self.K_timp
        occ, free = x / (1.0 + x), 1.0 / (1.0 + x)
        act = self.k_act * (mt1 * occ) * (mt1 * free) * pro * dt
        act = torch.minimum(act, pro)
        pro, mmp_d = pro - act, mmp_d + act

        r = (self.k_inhib * mmp_d * (timp_d + t3) * dt).clamp(max=mmp_d)
        mmp_d, timp_d = mmp_d - r, timp_d - r
        self.pro[li] = (pro * math.exp(-dt / self.tau_pro)).clamp_min(0.0)
        self.mmp[li] = (mmp_d * math.exp(-dt / self.tau_mmp)).clamp_min(0.0)
        self.timp[li] = (timp_d * math.exp(-dt / self.tau_timp)).clamp_min(0.0)
        self._act = act / max(dt, 1e-30)
        self._act_rate = float(act.sum())
        self._free_frac = float(free.mean())
        self.res_cons = 0.0
        return 0.0

    def frame(self, t):
        super().frame(t)
        A = getattr(self, "_act", None)
        if A is not None and A.numel():
            self.res["act_max"].append(float(A.max()))
            self.res["act_cv"].append(float(A.std() / A.mean().clamp_min(1e-30)))
            # the BAND: faces sitting within a factor of 2 of the bell's peak, c_T = K
            ct = self.timp[self.sheet.live] + self.timp3[self.sheet.live]
            self.res["band_frac"].append(float(((ct > 0.5 * self.K_timp)
                                                & (ct < 2.0 * self.K_timp)).to(self.dtype).mean()))
            t3 = self.timp3[self.sheet.live]
            self.res["timp3_mean"].append(float(t3.mean()))
            self.res["timp3_cv"].append(float(t3.std() / t3.mean().clamp_min(1e-30)))
        else:
            for k in ("act_max", "act_cv", "band_frac", "timp3_mean", "timp3_cv"):
                self.res[k].append(0.0)


# =============================================================================================
def render_hetero(kept, d, name, fps=24):
    """The video: the source, the inhibitor, where the bell is satisfied, and the sheet being eaten.

    Panel 3 is the one this run exists for -- the ACTIVATION map. If it is a copy of panel 1 the
    mechanism is a stencil; if it is a band that panel 1 does not contain, the field is doing work.
    """
    fig = plt.figure(figsize=(12.4, 8.8), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    C, lim = np.array([0.5, 0.5, 0.5]), 0.165
    keys = ("mt1", "timp3", "act", "rho")
    hi = {k: float(np.percentile(np.concatenate([f[k][::5] for f in [x[4] for x in kept[::3]]]), 99))
          for k in keys}
    for k in hi:
        hi[k] = hi[k] if hi[k] > 0 else 1.0
    titles = {"mt1": "MT1-MMP expression (per cell) -- the source",
              "timp3": "TIMP-3 -- ECM-bound, so it keeps its pattern",
              "act": "activation rate -- where the bell is satisfied",
              "rho": r"$\rho/\rho_0$ -- gaps are breached"}
    import ecm_render as RD
    # THE CAMERA IS AIMED AT THE HOLE, AND HELD THERE. The breach is the thing this run is about and
    # it forms where the activation is highest, which is a FIXED patch on the sphere -- so a camera
    # that turns 0.25 deg a frame carries it out of view by the end, which is what "the hole is on
    # the backside" was. The direction is taken from the LAST frame's rho: the faces that have lost
    # the most material are the hole, and their centroid is where to look from. No turn.
    _t, _X, _F, _XE, _fl = kept[-1]
    _cen = _X[_F].mean(1) - C
    _lo = np.argsort(_fl["rho"])[: max(8, _F.shape[0] // 50)]
    _dir = _cen[_lo].mean(0)
    _dir = _dir / (np.linalg.norm(_dir) + 1e-12)
    elev0 = float(np.degrees(np.arcsin(np.clip(_dir[2], -1, 1))))
    azim0 = float(np.degrees(np.arctan2(_dir[1], _dir[0])))
    print(f"[{name}] camera aimed at the breach: elev {elev0:.0f}, azim {azim0:.0f} "
          f"(from the {_lo.size} faces with the least material at the last frame)", flush=True)
    strip, strip_at = [], set(np.round(np.linspace(0, len(kept) - 1, 8)).astype(int).tolist())
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for idx, (t, X, F, XE, flds) in enumerate(kept):
            fig.clf()
            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02,
                                hspace=0.04)
            for n, k in enumerate(keys):
                ax = fig.add_subplot(2, 2, n + 1, projection="3d", facecolor="black",
                                     computed_zorder=False)
                ax.set_facecolor("black"); ax.axis("off")
                # THE CUT FOLLOWS THE CAMERA. It was fixed in world space -- keep the faces with
                # y above the centre -- while the camera turns 0.25 deg a frame, 75 deg over the
                # run, so the opening swung into view and the sphere read as a half-disc with a
                # black void beside it. Rotating the camera only mirrored that. Here the half
                # removed is the one BEHIND the viewer, recomputed from (elev, azim) every frame, so
                # what is drawn is always the near, convex face and there is no gap to be on the
                # wrong side. `screen_basis` is `ecm_render`'s, so the two renderers cannot disagree
                # about which way the camera points.
                az = azim0
                dvec, _, _ = RD.screen_basis(elev0, az)
                kf = ((X[F].mean(1) - C) @ dvec) < 0
                tri = Poly3DCollection(X[F][kf], linewidths=0.0)
                tri.set_facecolor(CMAP(np.clip(flds[k][kf] / hi[k], 0, 1)))
                ax.add_collection3d(tri)
                ax.set_xlim(C[0] - lim, C[0] + lim); ax.set_ylim(C[1] - lim, C[1] + lim)
                ax.set_zlim(C[2] - lim, C[2] + lim)
                try:
                    ax.set_box_aspect((1, 1, 1), zoom=2.15)
                except TypeError:
                    ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=elev0, azim=az)
                ax.text2D(0.02, 0.99, f"{titles[k]}\n0 to {hi[k]:.3g}", transform=ax.transAxes,
                          color="white", fontsize=9.5, va="top")
                if n == 0:
                    ax.text2D(0.02, 0.06, f"{name}   frame {t}   {F.shape[0]} live faces",
                              transform=ax.transAxes, color="#888", fontsize=9)
            wri.grab_frame()
            if idx in strip_at:
                strip.append((t, X.copy(), F.copy(), flds["act"].copy(), hi["act"]))
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)
    figs = plt.figure(figsize=(3.0 * len(strip), 3.2), facecolor="black")
    for i, (t, X, F, act, ah) in enumerate(strip):
        a = figs.add_subplot(1, len(strip), i + 1, projection="3d", facecolor="black")
        a.set_facecolor("black"); a.axis("off")
        dvec, _, _ = RD.screen_basis(elev0, azim0)
        kf = ((X[F].mean(1) - np.asarray([0.5, 0.5, 0.5])) @ dvec) < 0
        tri = Poly3DCollection(X[F][kf], linewidths=0.0)
        tri.set_facecolor(CMAP(np.clip(act[kf] / ah, 0, 1)))
        a.add_collection3d(tri)
        a.set_xlim(0.335, 0.665); a.set_ylim(0.335, 0.665); a.set_zlim(0.335, 0.665)
        try:
            a.set_box_aspect((1, 1, 1), zoom=2.0)
        except TypeError:
            pass
        a.view_init(elev=elev0, azim=azim0)
        a.text2D(0.03, 0.97, f"frame {t}", transform=a.transAxes, color="white", fontsize=10,
                 va="top")
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=110, facecolor="black")
    plt.close(figs)


def run_h(rig, frames, keep, label, stop_at=0.02):
    """`stop_at` ends the run once the sheet is essentially gone: past that point every frame is the
    same picture of an absence, and a film of it reads as a rendering failure rather than a result."""
    kept, n0 = [], None
    for t in range(frames):
        rig.frame(t)
        n0 = n0 or rig.sheet.m
        if rig.sheet.m < stop_at * n0:
            print(f"[{label}] the sheet is gone at frame {t} ({rig.sheet.m} of {n0} faces)",
                  flush=True)
            return kept, t
        if not rig.alive():
            print(f"[{label}] stopped at frame {t}", flush=True); return kept, t
        if t in keep:
            li = rig.sheet.live
            flds = dict(mt1=rig._mt1_on_faces().float().cpu().numpy(),
                        timp3=rig.timp3[li].float().cpu().numpy(),
                        act=getattr(rig, "_act", rig.timp[li] * 0).float().cpu().numpy(),
                        rho=(rig.sheet.areal_density() / rig.sheet.rho0).float().cpu().numpy())
            kept.append((t, rig.sheet.x.float().cpu().numpy(), rig.sheet.Fc.cpu().numpy(),
                         rig.x_epi.float().cpu().numpy(), flds))
    print(f"[{label}] {frames} frames -- {rig.sheet.m} faces, {rig._dead_cum} dead, "
          f"hole {100*rig.res['hole_frac'][-1]:.1f}%, act CV {rig.res['act_cv'][-1]:.2f}", flush=True)
    return kept, frames


def main():
    def arg(f, c, dflt):
        return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else dflt
    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 300)
    name = arg("--name", str, "05h_1_hetero")
    K = arg("--K", float, 1.0e-3)
    # THE TWO AXES OF THE 05i PHASE DIAGRAM, so a point identified there can be run here in 3D.
    # 05i's map says localisation needs BOTH a high inhibitor (most of the sheet pushed off the
    # bell's peak) AND a BOUND one (so what is left can hold a pattern): corr(activation, MT1) falls
    # from ~0.9 to 0.57 only in that corner, and the single cell with both breach and pattern is
    # inhibitor/K = 10 with bound = 1.
    inhib = arg("--inhib", float, 1.0)          # total inhibitor / K
    bound = arg("--bound", float, 0.6)          # fraction of it that is TIMP-3 (immobile)
    # k_deg SETS HOW LONG THE FILM IS, and it has to be set per phase point: activation falls ~10x
    # from the bell's peak to the high-inhibitor corner, so a rate that breaches in 250 frames at one
    # point does nothing at another. At 250 the sheet was gone by frame 50 and the film went black;
    # at 30 nothing died at all in 300.
    kdeg = arg("--kdeg", float, 100.0)
    d = os.path.join(LOG, name); os.makedirs(d, exist_ok=True)
    cert = BM.selftest(dev=dev, subdiv=4)
    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, sigma_T=7.0, zeta=20.0, s_target=1.0,
             k_drive=50.0, dev=dev)
    A = dict(kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3)
    S = dict(s_mode="homeostatic", tau_bm=40.0, rho_crit=0.35, max_refine=0, reseed=False)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 180))).astype(int).tolist())

    # THREE ARMS, because the bell is BROAD and that has to be shown rather than hidden. x/(1+x)^2
    # falls only 4x over two decades of inhibitor, so wherever MT1 also varies the activation map is
    # dominated by MT1^2 and reads as a copy of the source (corr 0.96, measured). The middle arm holds
    # MT1 UNIFORM so that whatever pattern survives is the bell's alone.
    # The inhibitor is set to STRADDLE K: TIMP-2 (5e-5 x 8 = 4e-4) plus TIMP-3 (1.5e-5 x 40 = 6e-4)
    # averages K = 1e-3, and TIMP-3's spatial modulation carries the sheet across the peak. At ten
    # times K the whole sheet sat on the far shoulder and 1.3% of faces were in the band.
    runs, corr = {}, {}
    for lab, h, mt1_uniform in (("realistic: all rates vary", 1.0, False),
                                ("bell isolated: uniform MT1", 1.0, True),
                                ("uniform control", 0.0, True)):
        # k_deg SETS HOW LONG THE FILM IS. At 250 the sheet went from 5120 faces to 661 by frame 50
        # and the remaining 250 frames rendered a hole that had stopped changing -- the movie was
        # "nice at the beginning and then black". The rate is set so the breach develops over the
        # whole run instead of in the first sixth of it.
        s_t2 = inhib * K * (1.0 - bound) / 8.0        # tau_timp2  = 8 frames
        s_t3 = inhib * K * bound / 40.0               # tau_timp3 = 40 frames
        r = Rig05h1(**P, **A, **S, K_timp=K, hetero=h, s_timp=s_t2, s_timp3=s_t3, s_mmp=0.0,
                    s_mt1=0.0, k_deg=kdeg, mt1_frac=0.25)
        if mt1_uniform:
            r.mt1 = torch.full_like(r.mt1, float(r.mt1.mean()))
        kept, _ = run_h(r, frames, keep if lab.startswith("realistic") else set(), f"{name}: {lab}")
        runs[lab] = r.res
        src_ = r._mt1_on_faces().float().cpu().numpy()
        act_ = r._act.float().cpu().numpy()
        corr[lab] = (float(np.corrcoef(src_, act_)[0, 1])
                     if act_.std() > 0 and src_.std() > 0 else float("nan"))
        if lab.startswith("realistic"):
            render_hetero(kept, d, name)

    out = dict(run=name, frames=frames, K_timp=K, inhib_over_K=inhib, bound_frac=bound,
               phase_point=f"05i coordinates: inhibitor/K = {inhib:g}, bound = {bound:g}",
               certification=cert,
               reference="Karagiannis & Popel 2004 JBC 279:39105",
               G46=dict(corr_activation_with_source=corr.get('realistic: all rates vary'),
                        corr_bell_isolated=corr.get('bell isolated: uniform MT1'),
                        bell_dynamic_range="x/(1+x)^2 falls only 4x over two decades, so the bell "
                                           "cannot carve a sharp band; where MT1 also varies the map "
                                           "is dominated by MT1^2",
                        note="if the activation map is a copy of the source this is ~1 and the "
                             "mechanism is a stencil; the bell should decorrelate them"),
               heterogeneity={k: dict(act_cv=v["act_cv"][-1], band_frac=v["band_frac"][-1],
                                      timp3_cv=v["timp3_cv"][-1], hole=v["hole_frac"][-1],
                                      corr_with_source=corr[k]) for k, v in runs.items()},
               series={k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                       for k, v in runs.items()})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    yaml.safe_dump(dict(
        what="05h_1 -- the ternary mechanism with per-cell rate states, so the bell has spatial "
             "structure to act on",
        units=dict(**UNITS, force_nN=None),
        new_states=dict(mt1="graded expression per cell (was a binary flag)",
                        s_timp="per-cell inhibitor secretion (was one global number)",
                        s_pro="per-cell zymogen secretion (was one global number)"),
        why="the bell peaks at c_T = K, so a VARYING TIMP field puts different regions at different "
            "points on the same curve -- the prediction is a band along c_T ~ K rather than a copy "
            "of the source",
        seeding="smooth low-order modes, not per-cell noise: white noise at this D is averaged away "
                "inside one frame, which is how the uniform runs got a featureless field",
        reference="Karagiannis & Popel (2004) J. Biol. Chem. 279(37):39105"),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    for k, v in out["heterogeneity"].items():
        print(f"[{name}] {k:28s} corr(act,source) {v['corr_with_source']:+.3f}  act CV "
              f"{v['act_cv']:.2f}  TIMP3 CV {v['timp3_cv']:.2f}  band "
              f"{100*v['band_frac']:5.1f}%  hole {100*v['hole']:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
