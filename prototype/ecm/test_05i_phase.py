#!/usr/bin/env python
"""test_05i_phase -- the three-species reaction on a flat 2D sheet of cells, as a phase diagram.

    python test_05i_phase.py [--device cuda:0] [--n 48] [--frames 400]
                                                     ->  log/okuda_ECM/05i_phase/

WHY 2D AND WHY NOW. Everything asked of the protease system in 05g/05h was a question about the
CHEMISTRY -- can a hole have a location, does the bell give it a size, does an inhibitor hold a pattern
-- and every one of those was being answered on a 3D spheroid carrying a mesh, a mechanics, an adhesion
and a mass balance, at ~9 minutes per condition. Three or four conditions is not a phase diagram. On a
flat sheet of cells the same reaction network runs in milliseconds per frame, so a 12x12 grid of
conditions is affordable and the regimes can be MAPPED rather than sampled.

The mechanics is dropped on purpose and the loss is stated: this rig cannot say anything about stretch,
standoff, tearing under load, or the coupling to growth. It is a preliminary for the chemistry only.

THE TWO AXES, chosen because they are the two things the 3D runs could not separate:

  x   total inhibitor / K       where the sheet sits on the bell x/(1+x)^2, which peaks at x = 1
  y   BOUND fraction of it      TIMP-3 is sequestered on sulfated GAGs (Yu et al. 2000, JBC
                                275:31226) and does not diffuse; TIMP-2 is soluble and does. At y = 0
                                the inhibitor cannot hold a pattern at all -- sqrt(4Dt) = 268 um per
                                frame against a 10 um cell -- and at y = 1 it keeps whatever pattern
                                it was deposited with.

so the diagram asks: WHEN does this network produce localised proteolysis rather than uniform
dissolution or silence? The 3D runs answered it at one point of that plane and could not see the shape.

THE SPECIES, as in 05h, with per-cell rates:
  MT1-MMP  tethered, per cell, no diffusion      the activator; sets where
  proMMP-2 diffusing field                        the zymogen
  MMP-2    diffusing field                        active, cuts the sheet
  TIMP-2   diffusing field         }             one inhibitor, split by whether it can move
  TIMP-3   BOUND field, D = 0      }
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

import ecm_spec as ES                                                   # noqa: E402
from test_05_sheet import LOG, UNITS                                    # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

CMAP = ListedColormap(ES.STRESS_COLORS)
UM = 1171.0                     # box units -> um, the declared scale
CELL_UM = 10.0                  # one lattice cell is one epithelial cell


def smooth2d(n, modes=5, seed=0, dev="cuda:0", dtype=torch.float64):
    """A smooth random field on the periodic lattice: a few low wavenumbers, not white noise.

    Per-cell noise is erased by diffusion inside one frame -- measured in 05h_1 -- so heterogeneity
    has to be laid down at a length the transport cannot reach."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    y, x = torch.meshgrid(torch.arange(n, dtype=dtype), torch.arange(n, dtype=dtype), indexing="ij")
    out = torch.zeros(n, n, dtype=dtype)
    for _ in range(modes):
        kx, ky = [int(torch.randint(1, 4, (1,), generator=g)) for _ in range(2)]
        ph = float(torch.rand(1, generator=g)) * 6.2831
        out = out + torch.cos(2 * math.pi * (kx * x + ky * y) / n + ph)
    out = out - out.min()
    return (out / out.max().clamp_min(1e-30)).to(device=dev, dtype=dtype)


def lap2(c):
    """Periodic 5-point Laplacian, in units of 1/h^2."""
    return (torch.roll(c, 1, 0) + torch.roll(c, -1, 0) + torch.roll(c, 1, 1)
            + torch.roll(c, -1, 1) - 4.0 * c)


def diffuse_fft(c, Dt, k2):
    """Exact periodic diffusion in one FFT round trip: (I - Dt*lap) c_new = c  =>  chat/(1 + Dt k^2).

    The explicit version is what the first run of this rig tried, and it made the same point the 3D
    argument did, in wall clock: D = 30 um^2/s over a 10 um cell and a 600 s frame is 180 cells^2 per
    frame, so the CFL asks for ~720 substeps EVERY frame -- 35 conditions then cost more than the 3D
    spheroid it was built to be cheaper than. On a periodic lattice the solve is exact and free.
    """
    return torch.fft.irfft2(torch.fft.rfft2(c) / (1.0 + Dt * k2), s=c.shape).clamp_min(0.0)


class Sheet2D:
    """A flat sheet of cells carrying the three-species network. No mechanics.

    Rates are PER CELL (that is the whole point of the rig): MT1 expression, the two inhibitor
    secretions and the zymogen secretion each have their own smooth spatial field.
    """

    def __init__(self, n=48, K=1.0e-3, k_act=0.5, k_inhib=200.0, k_deg=250.0,
                 D=30.0, tau=8.0, tau3=40.0, s_pro=2.0e-3, inhib_total=1.0, bound_frac=0.5,
                 mt1_mean=0.25, hetero=1.0, dt_frame=1.0, seed=0, dev="cuda:0",
                 dtype=torch.float64):
        self.n, self.dev, self.dtype = n, dev, dtype
        self.K, self.k_act, self.k_inhib, self.k_deg = K, k_act, k_inhib, k_deg
        self.tau, self.tau3, self.s_pro = tau, tau3, s_pro
        # D in box units: a lattice cell is CELL_UM across, and a frame is UNITS['time_s'] seconds
        self.D_cells2 = D * UNITS["time_s"] / (CELL_UM ** 2)     # cells^2 per frame
        self.dt = dt_frame
        z = torch.zeros(n, n, device=dev, dtype=dtype)
        self.pro, self.mmp, self.t2, self.t3 = z.clone(), z.clone(), z.clone(), z.clone()
        self.mass = torch.ones(n, n, device=dev, dtype=dtype)
        h = hetero
        def field(sd):
            return (1.0 - h) + h * 2.0 * smooth2d(n, seed=sd, dev=dev, dtype=dtype)
        self.mt1 = mt1_mean * field(seed + 3)
        # the inhibitor is ONE budget split by whether it can move: total sets where on the bell,
        # bound_frac sets how much of it can hold a pattern
        s_t2 = inhib_total * K * (1.0 - bound_frac) / max(tau, 1e-30)
        s_t3 = inhib_total * K * bound_frac / max(tau3, 1e-30)
        self.s_t2 = s_t2 * field(seed + 7)
        self.s_t3 = s_t3 * field(seed + 11)
        self.s_pro_f = s_pro * field(seed + 13)
        ky = torch.fft.fftfreq(n, d=1.0 / n, device=dev, dtype=dtype) * (2 * math.pi / n)
        kx = torch.fft.rfftfreq(n, d=1.0 / n, device=dev, dtype=dtype) * (2 * math.pi / n)
        self.k2 = (ky[:, None] ** 2 + kx[None, :] ** 2)
        self.hist = {k: [] for k in ("act_mean", "act_cv", "mass_mean", "dead_frac", "t_inhib_mean",
                                     "corr_act_mt1", "mmp_mean")}

    def step(self):
        dt = self.dt
        # transport: explicit, sub-stepped to its own CFL. In 2D on a lattice this is cheap, and
        # keeping it explicit here (rather than the CG solve the 3D rig needs) makes the substep count
        # visible as the cost it is.
        Dt = self.D_cells2 * dt
        self.pro = diffuse_fft(self.pro, Dt, self.k2)
        self.mmp = diffuse_fft(self.mmp, Dt, self.k2)
        self.t2 = diffuse_fft(self.t2, Dt, self.k2)
        self.pro = self.pro + dt * self.s_pro_f
        self.t2 = self.t2 + dt * self.s_t2
        self.t3 = self.t3 + dt * self.s_t3            # BOUND: no transport term at all
        # the bell: one MT1 must hold the zymogen and another must cut it
        x = (self.t2 + self.t3) / self.K
        act = self.k_act * (self.mt1 * x / (1 + x)) * (self.mt1 / (1 + x)) * self.pro * dt
        act = torch.minimum(act, self.pro)
        self.pro, self.mmp = self.pro - act, self.mmp + act
        # inhibition of the active enzyme, then clearance of everything that can be cleared
        r = (self.k_inhib * self.mmp * (self.t2 + self.t3) * dt).clamp(max=self.mmp)
        self.mmp = (self.mmp - r) * math.exp(-dt / self.tau)
        self.t2 = (self.t2 - r * (self.t2 / (self.t2 + self.t3).clamp_min(1e-30))
                   ) * math.exp(-dt / self.tau)
        self.t3 = (self.t3 - r * (self.t3 / (self.t2 + self.t3).clamp_min(1e-30))
                   ).clamp_min(0.0) * math.exp(-dt / self.tau3)
        self.pro = self.pro * math.exp(-dt / self.tau)
        self.mmp, self.t2, self.t3 = (self.mmp.clamp_min(0), self.t2.clamp_min(0),
                                      self.t3.clamp_min(0))
        self.mass = (self.mass * torch.exp(-self.k_deg * self.mmp * dt)).clamp_min(0.0)
        self._act = act / dt
        a, m = self._act, self.mt1
        self.hist["act_mean"].append(float(a.mean()))
        self.hist["act_cv"].append(float(a.std() / a.mean().clamp_min(1e-30)))
        self.hist["mass_mean"].append(float(self.mass.mean()))
        self.hist["dead_frac"].append(float((self.mass < 0.35).to(self.dtype).mean()))
        self.hist["t_inhib_mean"].append(float((self.t2 + self.t3).mean()))
        self.hist["mmp_mean"].append(float(self.mmp.mean()))
        av, mv = a.flatten(), m.flatten()
        if float(av.std()) > 0 and float(mv.std()) > 0:
            self.hist["corr_act_mt1"].append(float(((av - av.mean()) * (mv - mv.mean())).mean()
                                                   / (av.std() * mv.std())))
        else:
            self.hist["corr_act_mt1"].append(float("nan"))


# =============================================================================================
def phase_png(grid, xs, ys, d, K, frames):
    """The phase diagram: four maps over the same plane, because one number cannot name a regime."""
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 9.0), facecolor="white")
    keys = [("dead", "fraction of the sheet breached", "magma"),
            ("act_cv", "spatial CV of activation", "viridis"),
            ("corr", r"corr(activation, MT1) -- 1 means a stencil", "coolwarm"),
            ("act", "mean activation rate", "cividis")]
    X, Y = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)))
    for a, (k, lab, cm) in zip(ax.reshape(-1), keys):
        Z = np.array([[grid[(xi, yi)][k] for xi in range(len(xs))] for yi in range(len(ys))])
        kw = dict(vmin=-1, vmax=1) if k == "corr" else {}
        im = a.pcolormesh(X, Y, Z, cmap=cm, shading="nearest", **kw)
        fig.colorbar(im, ax=a, fraction=0.046)
        a.set_xticks(range(len(xs)))
        a.set_xticklabels([f"{v:g}" for v in xs], fontsize=7, rotation=45)
        a.set_yticks(range(len(ys)))
        a.set_yticklabels([f"{v:.2f}" for v in ys], fontsize=7)
        a.set_xlabel("total inhibitor / $K$   (the bell peaks at 1)")
        a.set_ylabel("bound fraction (TIMP-3)")
        a.set_title(lab, fontsize=9)
        # the bell's peak, which is a prediction and not a fitted line
        if 1.0 in list(xs):
            a.axvline(list(xs).index(1.0), color="w", ls="--", lw=1.0)
    fig.suptitle(f"05i: the three-species network on a flat sheet, {frames} frames per point, "
                 f"{len(xs)}x{len(ys)} conditions", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "phase.png"), dpi=150, facecolor="white")
    plt.close(fig)


def movie(sheets, labels, d, frames, fps=24):
    """Four representative points of the plane, side by side, each showing where it is being eaten."""
    fig = plt.figure(figsize=(13.0, 7.2), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": "05i"})
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for t in range(frames):
            for s in sheets:
                s.step()
            if t % 2:
                continue
            fig.clf()
            fig.subplots_adjust(0.02, 0.02, 0.98, 0.92, wspace=0.06, hspace=0.16)
            for i, (s, lab) in enumerate(zip(sheets, labels)):
                for j, (arr, nm) in enumerate(((s._act, "activation"), (s.mass, r"$\rho/\rho_0$"))):
                    a = fig.add_subplot(2, len(sheets), j * len(sheets) + i + 1)
                    v = arr.float().cpu().numpy()
                    a.imshow(v, cmap="magma" if j == 0 else "bone",
                             vmin=0, vmax=max(float(v.max()), 1e-12) if j == 0 else 1.0)
                    a.set_xticks([]); a.set_yticks([])
                    a.set_title(f"{lab}\n{nm}" if j == 0 else nm, fontsize=8.5, color="white")
            fig.suptitle(f"05i   frame {t}   (a lattice cell is {CELL_UM:g} um)", color="white",
                         fontsize=10)
            wri.grab_frame()
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)


def main():
    def arg(f, c, dflt):
        return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else dflt
    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    n = arg("--n", int, 48)
    frames = arg("--frames", int, 400)
    name = arg("--name", str, "05i_phase")
    K = arg("--K", float, 1.0e-3)
    d = os.path.join(LOG, name); os.makedirs(d, exist_ok=True)

    xs = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]          # total inhibitor / K
    ys = [0.0, 0.25, 0.5, 0.75, 1.0]                      # bound fraction
    grid, t0 = {}, __import__("time").time()
    for yi, bf in enumerate(ys):
        for xi, it in enumerate(xs):
            s = Sheet2D(n=n, K=K, inhib_total=it, bound_frac=bf, dev=dev)
            for _ in range(frames):
                s.step()
            q = max(4, frames // 4)
            grid[(xi, yi)] = dict(dead=s.hist["dead_frac"][-1],
                                  act=float(np.mean(s.hist["act_mean"][-q:])),
                                  act_cv=float(np.mean(s.hist["act_cv"][-q:])),
                                  corr=float(np.nanmean(s.hist["corr_act_mt1"][-q:])),
                                  mass=s.hist["mass_mean"][-1],
                                  inhib=float(np.mean(s.hist["t_inhib_mean"][-q:])))
        print(f"[{name}] bound fraction {bf:.2f} done "
              f"({__import__('time').time()-t0:.0f}s elapsed)", flush=True)
    phase_png(grid, xs, ys, d, K, frames)

    # four representative points, as a movie
    picks = [(0.3, 1.0, "under-inhibited, bound"), (1.0, 1.0, "at the peak, bound"),
             (1.0, 0.0, "at the peak, all soluble"), (10.0, 1.0, "over-inhibited, bound")]
    sheets = [Sheet2D(n=n, K=K, inhib_total=a, bound_frac=b, dev=dev) for a, b, _ in picks]
    movie(sheets, [p[2] for p in picks], d, min(frames, 240))

    out = dict(run=name, n=n, frames=frames, K=K, cell_um=CELL_UM,
               axes=dict(x="total inhibitor / K", y="bound fraction (TIMP-3)", xs=xs, ys=ys),
               reference="Karagiannis & Popel 2004 JBC 279:39105 (network); "
                         "Yu et al. 2000 JBC 275:31226 (TIMP-3 is ECM-bound)",
               grid={f"{xs[xi]:g}|{ys[yi]:g}": v for (xi, yi), v in grid.items()},
               note="2D, no mechanics: this rig says nothing about stretch, standoff or tearing "
                    "under load. It is a preliminary for the chemistry only.")
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    yaml.safe_dump(dict(
        what="05i -- the three-species network on a flat sheet of cells, as a phase diagram",
        units=dict(**UNITS, force_nN=None, cell_um=CELL_UM),
        why="every question asked of the protease system is about the CHEMISTRY, and answering it on "
            "a 3D spheroid costs ~9 min per condition. A flat sheet runs the same network in ms per "
            "frame, so the regimes can be MAPPED rather than sampled.",
        dropped="the mechanics: no stretch, no standoff, no tearing under load, no growth coupling",
        axes=dict(x="total inhibitor / K -- where the sheet sits on the bell",
                  y="bound fraction -- TIMP-3 does not diffuse and so can hold a pattern; TIMP-2 "
                    "cannot, since sqrt(4Dt) = 268 um per frame against a 10 um cell"),
        species=dict(MT1_MMP="per cell, tethered", proMMP_2="diffusing", MMP_2="diffusing",
                     TIMP_2="diffusing", TIMP_3="BOUND, D = 0")),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    best = max(grid.items(), key=lambda kv: kv[1]["act_cv"])
    print(f"[{name}] {len(xs)}x{len(ys)} conditions in {__import__('time').time()-t0:.0f}s; "
          f"most patterned at inhibitor/K = {xs[best[0][0]]:g}, bound = {ys[best[0][1]]:g} "
          f"(act CV {best[1]['act_cv']:.2f}, corr with MT1 {best[1]['corr']:+.2f}) -> {d}",
          flush=True)


if __name__ == "__main__":
    main()
