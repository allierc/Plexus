#!/usr/bin/env python
"""pillar_rig -- a sheet held off a substrate by MPM PILLARS, at the real proportions, fully resolved.

    python pillar_rig.py                      -> log/okuda_ECM/161_pillar_*.mp4

WHY THIS EXISTS. `FibreRig` asked whether a fibre of two or three particles could hold a gap, and the
answer on the spheroid was no at every setting: a column spanning 0.19 of a grid cell is not a material
the grid can take a stress gradient across, and 142-151 measured that ten times over. But that failure
was set up by a shape nobody checked against the biology. A hemidesmosome is not a thin thread:

    lamina densa / basement membrane      ~100 nm thick        T
    integrin a6b4 linkage, lamina lucida  ~30 nm long          L = 0.3 T
    hemidesmosome plaque                  ~400 nm across       D = 4 T
    plaque spacing on the basal surface   ~800 nm              S = 8 T

The pillar is SQUAT -- four times wider than the sheet is thick and a third as long -- and that changes
what the grid has to resolve. The binding dimension is the pillar's LENGTH, so `dx <= L/2` puts the
sheet at ~7 cells thick and the plaque at ~27 cells across, which is a body MPM handles without
argument. This rig builds that at 4 particles per cell and asks the three questions the spheroid needs:
does the pillar hold the gap in compression, hold the sheet in tension, and DRAG it when the substrate
moves -- the last being the one the spheroid actually asks, and the one `FibreRig` could not pose
because its substrate is static.

Every particle is MPM material. There is no bond, no spring and no prescribed target anywhere in here:
the only thing that is imposed is the substrate's bottom row, and everything else is scatter -> grid ->
gather. If the sheet follows, it follows because a pillar carried it.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")):
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
OUT = "/workspace/Plexus/log/okuda_ECM"

SHEET_C = "#4aa3ff"      # the sheet, blue
PILLAR_C = "#f0913a"     # the integrin plaques, orange
SUB_C = "#9aa0a6"        # the substrate the plaques stand on


def _bspline(x, inv_dx):
    base = (x * inv_dx - 0.5).floor()
    fx = x * inv_dx - base
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 0)
    return base.long(), fx, w


class PillarRig:
    """Substrate + pillars + sheet, all MPM, at the proportions above.

    `T` is the sheet's thickness and everything else is a multiple of it, so the geometry is stated as
    the biology states it and the grid is chosen to resolve the smallest of them.
    """

    def __init__(self, T=0.035, L_over_T=0.3, D_over_T=3.0, S_over_T=6.0, nx=2, ppc=4,
                 E_sheet=2.0e3, E_pillar=2.0e3, E_sub=2.0e3, nu=0.2, rho=1.0,
                 sub_kind="vertex", n_grid=192, dt=5.0e-5, dev="cuda:0"):
        # THE PROPORTIONS ARE THE BIOLOGY'S, AT THE LOW END OF EACH RANGE so the whole array fits a
        # unit box at a grid that resolves the shortest of them: plaque 300 nm on a 100 nm membrane is
        # D = 3 T (the range is 3-4), spacing 600 nm is S = 6 T (the range is 6-8), and the linkage
        # across the lamina lucida is L = 0.3 T either way. Widening D or S costs grid, not honesty.
        self.dev, self.n_grid, self.dt = dev, n_grid, dt
        self.dx = 1.0 / n_grid; self.inv_dx = float(n_grid)
        self.T = T; self.L = L_over_T * T; self.D = D_over_T * T; self.S = S_over_T * T
        self.z_sub = 0.35                      # top of the substrate
        h_sub = 4 * self.dx
        # THE CHECK THAT DECIDES WHETHER ANY OF THIS IS RESOLVED, printed rather than assumed.
        self.cells = dict(sheet=T / self.dx, pillar_len=self.L / self.dx, pillar_dia=self.D / self.dx)

        g = torch.Generator().manual_seed(0)

        def fill_box(lo, hi, n):
            u = torch.rand(n, 3, generator=g)
            return torch.tensor(lo) + u * (torch.tensor(hi) - torch.tensor(lo))

        def n_for(vol):
            return max(1, int(round(ppc * vol / self.dx ** 3)))

        span = nx * self.S
        x0 = 0.5 - span / 2
        # substrate: a slab under everything
        pad = self.D / 2
        sub = fill_box([x0 - pad, x0 - pad, self.z_sub - h_sub],
                       [x0 + span + pad, x0 + span + pad, self.z_sub],
                       n_for((span + 2 * pad) ** 2 * h_sub))
        # pillars: nx x nx cylinders of diameter D, length L, standing on the substrate
        pil = []
        centres = [(x0 + (i + 0.5) * self.S, x0 + (j + 0.5) * self.S)
                   for i in range(nx) for j in range(nx)]
        npil = n_for(math.pi * (self.D / 2) ** 2 * self.L)
        for (cx, cy) in centres:
            q = fill_box([cx - self.D / 2, cy - self.D / 2, self.z_sub],
                         [cx + self.D / 2, cy + self.D / 2, self.z_sub + self.L],
                         int(npil * 4 / math.pi))
            keep = ((q[:, 0] - cx) ** 2 + (q[:, 1] - cy) ** 2) < (self.D / 2) ** 2
            pil.append(q[keep])
        pil = torch.cat(pil)
        # sheet: a slab resting on the pillar tops
        z0 = self.z_sub + self.L
        sh = fill_box([x0 - pad, x0 - pad, z0], [x0 + span + pad, x0 + span + pad, z0 + T],
                      n_for((span + 2 * pad) ** 2 * T))

        self.n_sub, self.n_pil, self.n_sh = len(sub), len(pil), len(sh)
        self.x = torch.cat([sub, pil, sh]).to(dev)
        self.N = self.x.shape[0]
        self.grp = torch.cat([torch.zeros(self.n_sub), torch.ones(self.n_pil),
                              2 * torch.ones(self.n_sh)]).to(dev).long()
        # WHAT THE SUBSTRATE IS, AND IT IS NOT A FREE CHOICE. On the spheroid the epithelium is a
        # REPLAY -- pass 1's vertex model, recorded and played back -- so it cannot be an MPM body and
        # can only be prescribed, which is exactly what `integrin_track` does to the fibres' cell ends.
        # A rig whose substrate is solved MPM material answers an easier question than the spheroid
        # asks: a solved body transmits through its own stress, a prescribed one transmits only the
        # momentum it scatters. `vertex` (the default) prescribes EVERY substrate particle; `mpm` keeps
        # one prescribed row under a solved slab and is the control that says how much of the result
        # came from the substrate being a real material.
        self.sub_kind = sub_kind
        self.kin = ((self.grp == 0) if sub_kind == "vertex" else
                    ((self.grp == 0) & (self.x[:, 2] < self.z_sub - h_sub + 1.5 * self.dx)))
        self.kin_x = self.x[self.kin].clone()
        self.v = torch.zeros_like(self.x)
        self.C = torch.zeros(self.N, 3, 3, device=dev)
        self.F = torch.eye(3, device=dev).expand(self.N, 3, 3).contiguous()
        self.vol = self.dx ** 3 / ppc
        self.m = rho * self.vol
        E = torch.tensor([E_sub, E_pillar, E_sheet], device=dev)[self.grp]
        self.mu = (E / (2 * (1 + nu)))[:, None, None]
        self.la = (E * nu / ((1 + nu) * (1 - 2 * nu)))[:, None, None]
        self.off = OFF.to(dev)
        # what the sheet started at, for the gap measurement
        self.z_sheet0 = float(self.x[self.grp == 2, 2].min())

    def step(self, v_kin=(0.0, 0.0, 0.0), g_sheet=0.0):
        dev, ng, dt, inv_dx = self.dev, self.n_grid, self.dt, self.inv_dx
        gm = torch.zeros(ng ** 3, device=dev)
        gmv = torch.zeros(ng ** 3, 3, device=dev)
        U, S, Vh = torch.linalg.svd(self.F)
        R = U @ Vh
        J = torch.linalg.det(self.F).clamp_min(1e-6)
        P = (2 * self.mu * (self.F - R) @ self.F.transpose(1, 2)
             + self.la * ((J - 1) * J)[:, None, None] * torch.eye(3, device=dev))
        affine = self.m * self.C - (4 * inv_dx * inv_dx * dt * self.vol) * P
        fb = torch.zeros_like(self.x)
        fb[self.grp == 2, 2] = -g_sheet * self.m
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
        # the driven row: position AND velocity, because a particle told it is at rest scatters no
        # momentum and the constraint becomes invisible (the bug that made 142/143 exact nulls).
        vk = torch.tensor(v_kin, device=dev, dtype=self.x.dtype)
        self.kin_x = self.kin_x + dt * vk
        self.x[self.kin] = self.kin_x
        self.v[self.kin] = vk

    # --- measurements -------------------------------------------------------------------------
    def gap(self):
        """Sheet underside minus substrate top, in units of the pillar's own length."""
        zs = float(torch.quantile(self.x[self.grp == 2, 2], 0.02))
        zt = float(torch.quantile(self.x[self.grp == 0, 2], 0.98))
        return (zs - zt) / self.L

    def sheet_x(self):
        return float(self.x[self.grp == 2, 0].mean())

    def sub_x(self):
        return float(self.x[self.grp == 0, 0].mean())


def render(rig, frames, drive, out, title, every=3, dpi=100, fps=15, n_draw=9000):
    """Two panels: the 3D body cut open, and the true cross-section through the middle."""
    fig = plt.figure(figsize=(11.4, 5.6), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": title})
    g = rig.grp.cpu().numpy()
    st = max(1, rig.N // n_draw)
    cols = np.array([SUB_C, PILLAR_C, SHEET_C])[g]
    hist = []
    with wri.saving(fig, out, dpi=dpi):
        for t in range(frames + 1):
            v_kin, g_sheet = drive(t, frames)
            rig.step(v_kin=v_kin, g_sheet=g_sheet)
            hist.append((t, rig.gap(), rig.sheet_x() - rig.sub_x()))
            if t % every:
                continue
            X = rig.x.detach().cpu().numpy()
            fig.clf()
            # ---- 3D, cut at y > 0.5 so the pillars are not hidden inside the slab
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            cut = X[:, 1] < 0.5
            xs, cs = X[cut][::st], cols[cut][::st]
            ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], s=5, c=cs, marker="o", linewidths=0,
                       alpha=0.95, depthshade=False)
            lo = 0.5 - 0.6 * (rig.S * 2 + rig.D); hi = 0.5 + 0.6 * (rig.S * 2 + rig.D)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_zlim(rig.z_sub - 6 * rig.dx, rig.z_sub + rig.L + rig.T + 6 * rig.dx)
            ax.set_box_aspect((1, 1, 0.55)); ax.view_init(elev=16, azim=-60)
            # ALL THE TEXT ON THE 3D PANEL, which has empty corners. On the section it sat ON the
            # drawing: with equal aspect the layers occupy a band a tenth of the panel's height and
            # every label lands inside it, so the one number the panel exists to show -- where the
            # sheet is relative to the substrate -- was written over by the words describing it.
            ax.text2D(0.02, 0.98,
                      f"{title}\n"
                      f"frame {t}\n"
                      f"gap {hist[-1][1]:.2f} of the pillar's length\n"
                      f"sheet lag {1e3*hist[-1][2]:+.2f}e-3 box units\n"
                      f"sheet {rig.cells['sheet']:.1f} cells thick, "
                      f"pillar {rig.cells['pillar_len']:.1f} long x {rig.cells['pillar_dia']:.0f} across",
                      transform=ax.transAxes, color="white", fontsize=10, va="top", linespacing=1.6)
            # ---- the cross-section: a slab one cell thick through the middle
            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            sl = np.abs(X[:, 1] - 0.5) < rig.dx
            a2.scatter(X[sl][:, 0], X[sl][:, 2], s=14, c=cols[sl], marker="o", linewidths=0,
                       alpha=0.95)
            # TWO PILLARS WIDE, NOT THE WHOLE ARRAY. At equal aspect -- and it has to be equal, the
            # panel's whole job is a length ratio -- the full span against a 0.066 stack is 8:1 and the
            # layers collapse into a line. A window of 2.2 pillar spacings makes them legible without
            # scaling one axis against the other.
            w = 1.1 * rig.S
            a2.set_xlim(0.5 - w, 0.5 + w)
            a2.set_ylim(rig.z_sub - 6 * rig.dx, rig.z_sub + rig.L + rig.T + 6 * rig.dx)
            a2.set_aspect("equal"); a2.axis("off")
            wri.grab_frame()
    plt.close(fig)
    return hist


# --- the drives ---------------------------------------------------------------------------------
# CALIBRATED TO THE PILLAR, not chosen. Over the run (800 steps x 5e-5 = 0.04 time units) an
# UNRESISTED sheet should move about two pillar lengths, so the load is g = 2*(2L)/t^2 ~ 25, and the
# substrate's speed is three pillar lengths over the run, 0.75 box units per unit time. A load that
# swamps the mechanism measures the load -- the first `FibreRig` sweep used one 1.4e4 times too big and
# every row read the same because the sheet was in free fall.
def drive_press(t, n):
    return (0.0, 0.0, 0.0), 25.0 * min(1.0, t / (0.3 * n))           # push the sheet DOWN


def drive_pull(t, n):
    return (0.0, 0.0, 0.0), -25.0 * min(1.0, t / (0.3 * n))          # pull the sheet UP


def drive_shear(t, n):
    return (0.75 * min(1.0, t / (0.3 * n)), 0.0, 0.0), 0.0           # slide the substrate sideways


def drive_grow(t, n):
    return (0.0, 0.0, 0.75 * min(1.0, t / (0.3 * n))), 0.0           # the substrate advances UPWARD


CASES = [
    ("161_pillar_press", drive_press, "press: does the pillar hold the gap?", dict()),
    ("162_pillar_pull", drive_pull, "pull: does the pillar hold the sheet on?", dict()),
    ("163_pillar_shear", drive_shear, "shear: does the pillar resist sliding?", dict()),
    ("164_pillar_grow", drive_grow, "grow: does the substrate DRAG the sheet?", dict()),
    # the control every one of them needs: the same load with no pillars at all
    ("165_nopillar_grow", drive_grow, "grow, NO pillars: the control", dict(D_over_T=0.0)),
]


def main(frames=800, dev="cuda:0", only=None, sub_kind="vertex"):
    """One FOLDER per case, with the spec that made it -- the convention every numbered run here uses.

    A movie without the spec that produced it is an anecdote, and these cases differ only in the drive,
    so the folder is the only place that difference is written down.
    """
    import json

    import yaml
    print(f"  {'case':24s}{'gap0':>7}{'gap end':>9}{'d(gap)':>9}   geometry")
    for name, drive, title, kw in CASES:
        if only and name not in only:
            continue
        k2 = dict(kw)
        if k2.get("D_over_T") == 0.0:
            k2["D_over_T"] = 1e-6                      # no plaque at all: the control
        rig = PillarRig(dev=dev, sub_kind=sub_kind, **k2)
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        spec = dict(
            rig=dict(T=rig.T, L=rig.L, D=rig.D, S=rig.S, n_grid=rig.n_grid, dt=rig.dt,
                     sub_kind=sub_kind, frames=frames,
                     particles=dict(substrate=rig.n_sub, pillars=rig.n_pil, sheet=rig.n_sh,
                                    total=rig.N, prescribed=int(rig.kin.sum()))),
            proportions=dict(L_over_T=round(rig.L / rig.T, 3), D_over_T=round(rig.D / rig.T, 3),
                             S_over_T=round(rig.S / rig.T, 3),
                             biology="BM ~100 nm thick; integrin a6b4 linkage ~30 nm; hemidesmosome "
                                     "plaque ~300-400 nm across; plaque spacing ~600-800 nm"),
            resolution_in_cells={k: round(v, 2) for k, v in rig.cells.items()},
            drive=drive.__name__, title=title)
        yaml.safe_dump(spec, open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
        h = render(rig, frames, drive, os.path.join(d, "movie.mp4"), title)
        json.dump(dict(gap_start=h[0][1], gap_end=h[-1][1], gap_change=h[-1][1] - h[0][1],
                       lag_end=h[-1][2],
                       history=[(int(a), float(b), float(c)) for a, b, c in h[::10]]),
                  open(os.path.join(d, "metrics.json"), "w"), indent=1)
        print(f"  {name:24s}{h[0][1]:>7.2f}{h[-1][1]:>9.2f}{h[-1][1]-h[0][1]:>+9.2f}   "
              f"sheet {rig.cells['sheet']:.1f} cells, pillar {rig.cells['pillar_len']:.1f} x "
              f"{rig.cells['pillar_dia']:.0f}, N={rig.N}", flush=True)


if __name__ == "__main__":
    main(dev=(sys.argv[1] if len(sys.argv) > 1 else "cuda:0"),
         sub_kind=(sys.argv[2] if len(sys.argv) > 2 else "vertex"))
