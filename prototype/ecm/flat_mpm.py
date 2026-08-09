"""The flat integrin tests again, with a REAL MPM sheet this time.

The earlier rig (`flat_test.py`) had no MPM in it: every membrane particle was pulled by its own fibre
and coupled to nothing, `P += dt*(k/gamma)*(target - P)`. Its results were clean -- standoff exactly the
fibre length, roughness decaying as exp(-t k/gamma) to 98% -- because each particle was an isolated
one-line ODE, not because the sheet behaved. Neighbours never spoke.

Here the sheet is genuine MLS-MPM: particles carry a deformation gradient F, scatter mass and stress
onto a background grid, the grid is solved, and velocity and velocity-gradient are gathered back. That
round trip is what makes the sheet a SHEET. The integrin enters as an external body force on the
particle, which is how any body force enters MPM.
"""
from __future__ import annotations

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
matplotlib.rcParams["animation.ffmpeg_path"] = os.path.join(os.path.dirname(sys.executable), "ffmpeg")

OFF = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)])


def _bspline(x, inv_dx, dev):
    base = (x * inv_dx - 0.5).floor()
    fx = x * inv_dx - base
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 0)
    return base.long(), fx, w


class Sheet:
    """A flat elastic sheet as MLS-MPM particles, held by integrin fibres to a prescribed plane."""

    def __init__(self, n=40, n_grid=32, E=8.0e3, nu=0.2, rho=1.0, L_fib=None,
                 k_fib=None, dt=2.0e-4, dev="cuda:0"):
        self.dev, self.n_grid, self.dt = dev, n_grid, dt
        self.dx = 1.0 / n_grid; self.inv_dx = float(n_grid)
        self.L_fib = L_fib if L_fib is not None else 1.5 * self.dx
        # THE STIFFNESS IS SET BY THE PARTICLE MASS, not chosen. In the direct-force rig the fibre had
        # no mass to accelerate (x += dt*F/gamma) so k could be anything; here the explicit limit is
        # dt*sqrt(k/m) < 1, and with m = rho*vol ~ 5e-7 a stiffness of 6e3 gives 700 and NaN on the
        # first steps. `k_fib` is now a FRACTION of that ceiling.
        self.k_frac = 0.25 if k_fib is None else float(k_fib)
        self.mu = E / (2 * (1 + nu)); self.la = E * nu / ((1 + nu) * (1 - 2 * nu))
        xs = torch.linspace(0.35, 0.65, n)
        Xg, Yg = torch.meshgrid(xs, xs, indexing="ij")
        # THREE LAYERS THROUGH THE THICKNESS. One layer of particles is not a continuum: it has no
        # volume to carry stress with, and MPM needs several particles per cell to integrate at all.
        zs = 0.5 + self.L_fib + torch.linspace(-self.dx * 0.4, self.dx * 0.4, 3)
        pts = [torch.stack([Xg.reshape(-1), Yg.reshape(-1), torch.full((n * n,), float(z))], -1)
               for z in zs]
        self.x = torch.cat(pts).to(dev)
        self.N = self.x.shape[0]
        self.v = torch.zeros_like(self.x)
        self.C = torch.zeros(self.N, 3, 3, device=dev)
        self.F = torch.eye(3, device=dev).expand(self.N, 3, 3).contiguous()
        self.vol = (0.3 * 0.3 * (0.8 * self.dx)) / self.N
        self.m = rho * self.vol
        self.k_fib = self.k_frac ** 2 * self.m / (dt * dt)      # dt*sqrt(k/m) = k_frac < 1
        # the anchored layer: the lowest sheet particles, each with a fibre to the plane below
        self.anch = torch.arange(n * n, device=dev)
        self.base = self.x[self.anch].clone(); self.base[:, 2] = 0.5
        self.off = OFF.to(dev)

    def body_force(self, tgt_z=None):
        f = torch.zeros_like(self.x)
        tgt = self.base.clone()
        tgt[:, 2] = 0.5 + (self.L_fib if tgt_z is None else tgt_z)
        f[self.anch] = self.k_fib * (tgt - self.x[self.anch])
        return f

    def step(self, tgt_z=None):
        dev, ng, dt, inv_dx = self.dev, self.n_grid, self.dt, self.inv_dx
        gm = torch.zeros(ng ** 3, device=dev)
        gmv = torch.zeros(ng ** 3, 3, device=dev)
        U, S, Vh = torch.linalg.svd(self.F)
        R = U @ Vh
        J = torch.linalg.det(self.F).clamp_min(1e-6)
        P = (2 * self.mu * (self.F - R) @ self.F.transpose(1, 2)
             + self.la * ((J - 1) * J)[:, None, None] * torch.eye(3, device=dev))
        affine = self.m * self.C - (4 * inv_dx * inv_dx * dt * self.vol) * P
        fb = self.body_force(tgt_z)
        base, fx, w = _bspline(self.x, inv_dx, dev)
        for o in self.off:
            ww = w[o[0], :, 0] * w[o[1], :, 1] * w[o[2], :, 2]
            dpos = (o.float() - fx) * self.dx
            idx = ((base[:, 0] + o[0]).clamp(0, ng - 1) * ng * ng
                   + (base[:, 1] + o[1]).clamp(0, ng - 1) * ng
                   + (base[:, 2] + o[2]).clamp(0, ng - 1))
            gm.index_add_(0, idx, ww * self.m)
            contrib = ww[:, None] * (self.m * self.v + torch.einsum('nij,nj->ni', affine, dpos)
                                     + dt * fb)
            gmv.index_add_(0, idx, contrib)
        gv = gmv / gm.clamp_min(1e-12)[:, None]
        newv = torch.zeros_like(self.v); newC = torch.zeros_like(self.C)
        for o in self.off:
            ww = w[o[0], :, 0] * w[o[1], :, 1] * w[o[2], :, 2]
            dpos = (o.float() - fx) * self.dx
            idx = ((base[:, 0] + o[0]).clamp(0, ng - 1) * ng * ng
                   + (base[:, 1] + o[1]).clamp(0, ng - 1) * ng
                   + (base[:, 2] + o[2]).clamp(0, ng - 1))
            g = gv[idx]
            newv += ww[:, None] * g
            newC += (4 * inv_dx) * ww[:, None, None] * torch.einsum('ni,nj->nij', g, dpos * inv_dx)
        self.v, self.C = newv, newC
        self.x = self.x + dt * self.v
        self.F = (torch.eye(3, device=dev) + dt * self.C) @ self.F


def wave(n=40, E=8.0e3, amp=1.2, frames=400, every=4, dev="cuda:0",
         out="/workspace/Plexus/log/okuda_ECM/_mpm_wave.mp4"):
    """Push the sheet UP in one half and DOWN in the other, through the fibres, and let MPM answer.

    Positive and negative together is the test the single dimple could not be: a sheet resists CURVATURE,
    so the interesting place is the line where up meets down, where the continuum has to bridge a sign
    change. Independent springs would reproduce the imposed pattern exactly, each particle sitting
    wherever its own fibre asks. A real continuum cannot: it rounds the crossing and undershoots the
    extremes, and how much it does so IS the sheet's stiffness.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    s = Sheet(n=n, E=E, dev=dev)
    b = s.base
    # what each fibre is told to hold: +amp on one side, -amp on the other, in units of fibre length
    patt = amp * torch.tanh((b[:, 0] - 0.5) / 0.035) * torch.cos((b[:, 1] - 0.5) / 0.30 * np.pi)
    fig = plt.figure(figsize=(11.4, 5.6), facecolor="black")
    wri = FFMpegWriter(fps=15)
    hist = []
    with wri.saving(fig, out, dpi=105):
        for t in range(frames + 1):
            a = min(1.0, t / (0.5 * frames))
            tgt_z = s.L_fib * (1.0 + a * patt)
            s.step(tgt_z=tgt_z)
            if t % every:
                continue
            x = s.x.detach().cpu().numpy()
            zt = (0.5 + tgt_z).detach().cpu().numpy()
            za = x[s.anch.cpu().numpy(), 2]
            hist.append((t, float(np.abs(za - zt).mean())))
            fig.clf()
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            ax.set_xlim(0.33, 0.67); ax.set_ylim(0.33, 0.67); ax.set_zlim(0.46, 0.60)
            ax.set_box_aspect((1, 1, 0.7)); ax.view_init(elev=20, azim=-62)
            A = b.detach().cpu().numpy(); Q = x[s.anch.cpu().numpy()]
            st = max(1, len(Q) // 120)
            fr = np.linspace(0.0, 1.0, 4)[:, None, None]
            bd = (A[::st][None] * (1 - fr) + Q[::st][None] * fr).reshape(-1, 3)
            ax.scatter(bd[:, 0], bd[:, 1], bd[:, 2], s=12, c="#f0913a", marker="o",
                       linewidths=0, alpha=0.9, depthshade=False)
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=16, c="#4aa3ff", marker="o",
                       linewidths=0, alpha=0.9)
            ax.text2D(0.02, 0.95, "MPM sheet (blue)  +  integrin (orange), pushed up and down",
                      transform=ax.transAxes, color="white", fontsize=10)
            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            m = np.abs(A[:, 1] - 0.5) < 0.01                       # a cut along x at mid-y
            o = np.argsort(A[m, 0])
            a2.plot(A[m, 0][o], (zt[m] - 0.5)[o], "--", color="#e0452b", lw=1.4,
                    label="what the fibres ask for")
            a2.plot(A[m, 0][o], (Q[m, 2] - 0.5)[o], "-", color="#4aa3ff", lw=2,
                    label="what the sheet does")
            a2.set_xlim(0.35, 0.65); a2.set_ylim(-0.02, 0.12)
            a2.set_xlabel("x", color="#bbb"); a2.set_ylabel("height above the plane", color="#bbb")
            a2.tick_params(colors="#bbb"); a2.legend(facecolor="black", labelcolor="white", fontsize=8)
            for sp_ in a2.spines.values(): sp_.set_color("#666")
            wri.grab_frame()
    plt.close(fig)
    x = s.x.detach().cpu().numpy(); za = x[s.anch.cpu().numpy(), 2]
    zt = (0.5 + s.L_fib * (1.0 + patt)).detach().cpu().numpy()
    return float(np.abs(za - zt).mean()), float((za - 0.5).min()), float((za - 0.5).max()), out


def flatten(n=40, E=8.0e3, amp=1.1, frames=500, every=5, dev="cuda:0",
            out="/workspace/Plexus/log/okuda_ECM/_mpm_flatten.mp4"):
    """The sheet starts WAVY and the integrins hold it flat -- passive, not actuating.

    This is the right way round biologically. An integrin is not a piston: it binds laminin and holds
    the membrane at one standoff. So every fibre here has the SAME rest length, the sheet is launched
    with a wave in it, and flattening is something the adhesion and the sheet do together rather than a
    shape being dictated.

    TWO WAVELENGTHS ON PURPOSE. With a real continuum the short wave should die first: the fibres pull
    every patch equally, but the sheet's own elasticity also resists curvature, and curvature goes as
    1/lambda^2. Independent springs would flatten both at exactly the same rate -- that is the signature
    that separates a sheet from a field of dots, and it is measured below rather than asserted.
    """
    s = Sheet(n=n, E=E, dev=dev)
    b0 = s.x.clone()
    kx_lo, kx_hi = 2.0 * np.pi / 0.30, 2.0 * np.pi / 0.075     # long and short wave
    wav = amp * s.L_fib * (0.6 * torch.sin(kx_lo * (s.x[:, 0] - 0.35))
                           + 0.4 * torch.sin(kx_hi * (s.x[:, 1] - 0.35)))
    s.x = s.x + torch.stack([torch.zeros_like(wav), torch.zeros_like(wav), wav], -1)
    A = s.base.detach().cpu().numpy()
    fig = plt.figure(figsize=(11.4, 5.6), facecolor="black")
    wri = FFMpegWriter(fps=15); hist = []
    with wri.saving(fig, out, dpi=105):
        for t in range(frames + 1):
            s.step()                                            # uniform fibres: all want L_fib
            if t % every:
                continue
            x = s.x.detach().cpu().numpy(); Q = x[s.anch.cpu().numpy()]
            z = Q[:, 2] - 0.5 - s.L_fib
            # amplitude of each wave, by projection onto its own mode
            a_lo = float(np.abs((z * np.sin(kx_lo * (A[:, 0] - 0.35))).mean()) * 2)
            a_hi = float(np.abs((z * np.sin(kx_hi * (A[:, 1] - 0.35))).mean()) * 2)
            hist.append((t, a_lo, a_hi, float(z.std())))
            fig.clf()
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            ax.set_xlim(0.33, 0.67); ax.set_ylim(0.33, 0.67); ax.set_zlim(0.47, 0.58)
            ax.set_box_aspect((1, 1, 0.7)); ax.view_init(elev=20, azim=-62)
            st = max(1, len(Q) // 120)
            fr = np.linspace(0.0, 1.0, 4)[:, None, None]
            bd = (A[::st][None] * (1 - fr) + Q[::st][None] * fr).reshape(-1, 3)
            ax.scatter(bd[:, 0], bd[:, 1], bd[:, 2], s=12, c="#f0913a", marker="o",
                       linewidths=0, alpha=0.9, depthshade=False)
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=16, c="#4aa3ff", marker="o",
                       linewidths=0, alpha=0.9)
            ax.text2D(0.02, 0.95, "wavy sheet, uniform integrins -- they hold, they do not push",
                      transform=ax.transAxes, color="white", fontsize=10)
            h = np.array(hist)
            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            a2.semilogy(h[:, 0], np.maximum(h[:, 1], 1e-9), "-", color="#8cc04f", lw=2,
                        label="long wave (0.30)")
            a2.semilogy(h[:, 0], np.maximum(h[:, 2], 1e-9), "-", color="#e0452b", lw=2,
                        label="short wave (0.075)")
            a2.set_xlim(0, frames); a2.set_xlabel("step", color="#bbb")
            a2.set_ylabel("wave amplitude", color="#bbb"); a2.tick_params(colors="#bbb")
            a2.legend(facecolor="black", labelcolor="white", fontsize=8)
            for sp_ in a2.spines.values(): sp_.set_color("#666")
            wri.grab_frame()
    plt.close(fig)
    h = np.array(hist)
    return h[0, 1], h[-1, 1], h[0, 2], h[-1, 2], out


# ---------------------------------------------------------------------------------------------------
# THE MPM INTEGRIN, AND THE ONE QUESTION THAT DECIDES WHETHER IT CAN EXIST AT THIS GRID
# ---------------------------------------------------------------------------------------------------
class FibreRig:
    """Sheet + integrin, both as MPM MATERIAL, coupled only through the shared grid.

    Everything above holds the sheet with a BODY FORCE: `k*(target - x)` applied to the particle, which
    reaches the grid as momentum but is not a thing with a length. `INTEGRIN_DESIGN.md` proposes the
    other construction -- the fibre is its own MPM material, anchored at the surface, and the gap it
    holds is its REST LENGTH rather than a balance of forces. That is the only version in which the
    standoff is a material property and detachment is material failure.

    It has one prerequisite nobody has measured, and it is the design's own check #1: MPM transmits
    force only through the grid, and the grid's smallest feature is `dx`. A fibre spanning less than a
    cell puts both of its ends inside one cell, where they share a B-spline stencil and the grid cannot
    tell them apart -- so the fibre's own stress cannot separate them and the length it is supposed to
    hold is invisible. On the spheroid the fibre is 0.004 box units against `dx` = 1/48 = 0.0208, i.e.
    L/dx = 0.19. This rig measures what fraction of its rest length such a fibre actually holds when the
    sheet is pressed onto it, against the same fibre resolved at L/dx = 0.5, 1 and 2.

    The kinematic end is the inner one: an epithelium that is a REPLAY cannot be an MPM body, so the
    fibre's cell end is prescribed. That is one row of particles, not a grid boundary condition -- the
    distinction `mpm_tissue_boundary` got wrong.
    """

    def __init__(self, n=20, n_grid=32, L_over_dx=0.2, E_sheet=8.0e3, E_fib=8.0e3, nu=0.2,
                 rho=1.0, n_fib=3, dt=2.0e-4, dev="cuda:0"):
        self.dev, self.n_grid, self.dt = dev, n_grid, dt
        self.dx = 1.0 / n_grid; self.inv_dx = float(n_grid)
        self.L = L_over_dx * self.dx
        xs = torch.linspace(0.4, 0.6, n)
        Xg, Yg = torch.meshgrid(xs, xs, indexing="ij")
        base_xy = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], -1)
        ns = base_xy.shape[0]
        # the sheet: three layers, sitting one fibre length above the plane
        sheet = torch.cat([torch.cat([base_xy, torch.full((ns, 1), 0.5 + self.L + z)], -1)
                           for z in torch.linspace(0.0, 0.3 * self.dx, 3)])
        # the fibre: `n_fib` particles per site, from the plane up to the sheet's underside
        fib = torch.cat([torch.cat([base_xy, torch.full((ns, 1), 0.5 + float(z))], -1)
                         for z in torch.linspace(0.0, self.L, n_fib)])
        self.x = torch.cat([sheet, fib]).to(dev)
        self.N = self.x.shape[0]
        self.n_sheet = sheet.shape[0]
        self.is_fib = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.is_fib[self.n_sheet:] = True
        # KINEMATIC: the lowest fibre row only. Prescribed by position after every step, so the epithelium
        # enters as ~n*n particles rather than as a condition on every grid node.
        self.kin = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.kin[self.n_sheet:self.n_sheet + ns] = True
        self.kin_x = self.x[self.kin].clone()
        self.low = torch.arange(ns, device=dev)          # the sheet's bottom layer, for the gap
        self.v = torch.zeros_like(self.x)
        self.C = torch.zeros(self.N, 3, 3, device=dev)
        self.F = torch.eye(3, device=dev).expand(self.N, 3, 3).contiguous()
        self.vol = (0.2 * 0.2 * (self.L + 0.3 * self.dx)) / self.N
        self.m = rho * self.vol
        E = torch.where(self.is_fib, torch.tensor(float(E_fib), device=dev),
                        torch.tensor(float(E_sheet), device=dev))
        self.mu = (E / (2 * (1 + nu)))[:, None, None]
        self.la = (E * nu / ((1 + nu) * (1 - 2 * nu)))[:, None, None]
        self.off = OFF.to(dev)

    def step(self, g=0.0):
        dev, ng, dt, inv_dx = self.dev, self.n_grid, self.dt, self.inv_dx
        gm = torch.zeros(ng ** 3, device=dev)
        gmv = torch.zeros(ng ** 3, 3, device=dev)
        U, S, Vh = torch.linalg.svd(self.F)
        R = U @ Vh
        J = torch.linalg.det(self.F).clamp_min(1e-6)
        P = (2 * self.mu * (self.F - R) @ self.F.transpose(1, 2)
             + self.la * ((J - 1) * J)[:, None, None] * torch.eye(3, device=dev))
        affine = self.m * self.C - (4 * inv_dx * inv_dx * dt * self.vol) * P
        # THE LOAD PRESSES THE SHEET ONTO THE FIBRE, and acts on the sheet alone: the question is what
        # the fibre does when it is squashed, which is the case the spheroid is in (121 ends with every
        # integrin compressed to a third of its length).
        fb = torch.zeros_like(self.x)
        fb[~self.is_fib, 2] = -g * self.m
        base, fx, w = _bspline(self.x, inv_dx, dev)
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
            g_ = gv[idx]
            newv += ww[:, None] * g_
            newC += (4 * inv_dx) * ww[:, None, None] * torch.einsum('ni,nj->nij', g_, dpos * inv_dx)
        self.v, self.C = newv, newC
        self.x = self.x + dt * self.v
        self.F = (torch.eye(3, device=dev) + dt * self.C) @ self.F
        # the prescribed inner end, applied after the update: position held, velocity zeroed
        self.x[self.kin] = self.kin_x
        self.v[self.kin] = 0.0

    def gap(self):
        """What the sheet's underside actually keeps, in units of the fibre's rest length."""
        return float((self.x[self.low, 2] - 0.5).mean() / self.L)


def fibre_transmit(L_over_dx=(0.2, 0.5, 1.0, 2.0), steps=1500, g=2.0e3, n=20, n_grid=32,
                   E_fib=8.0e3, dev="cuda:0"):
    """Sweep the fibre's length against the grid cell and report how much of it survives a load.

    The load is the same for every row, so the only thing that changes is whether the grid can SEE the
    fibre. A fibre that holds its length reports ~1.0; one the grid cannot resolve reports ~0.
    """
    print(f"  {'L/dx':>6}{'L (box)':>10}{'gap/L @0':>11}{'gap/L end':>11}{'min over run':>14}")
    out = []
    for r in L_over_dx:
        s = FibreRig(n=n, n_grid=n_grid, L_over_dx=r, E_fib=E_fib, dev=dev)
        g0, lo = s.gap(), 1e9
        for t in range(steps):
            s.step(g=g)
            if t % 25 == 0:
                lo = min(lo, s.gap())
            if not torch.isfinite(s.x).all():
                print(f"  {r:>6}{s.L:>10.5f}   NaN at step {t}"); lo = float("nan"); break
        print(f"  {r:>6}{s.L:>10.5f}{g0:>11.3f}{s.gap():>11.3f}{lo:>14.3f}")
        out.append((r, s.gap()))
    return out


if __name__ == "__main__" and "--fibre" in sys.argv:
    print("\nCan an MPM fibre hold a gap the grid cannot resolve?  (dx = 1/32 = 0.03125)\n")
    fibre_transmit()
