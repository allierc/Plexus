#!/usr/bin/env python
"""test_02i -- compress the fibre block between plates and release it. The first of the three
loadings `note_fibre` section 8 says would falsify a fibre model, and the cheapest.

    python test_02i_compress.py [--device cuda:0]  ->  log/okuda_ECM/02i_compress/

WHY THIS LOADING AND NOT ANOTHER. Nothing measured in 02b--02h needs fibres to explain it, so
nothing measured there can falsify a fibre model either: a drop test loads a block once, in one
direction, and reports the constitutive law it was given. A network of straightening fibres differs
from an isotropic solid in three ways that a compression-and-release makes into three numbers, and
every one of them is a PREDICTION this substrate should fail to meet -- because `mpm_scatter`
computes a fixed-corotated stress from F alone and reads the strand direction nowhere.

  stiffening   the tangent modulus at 20% strain over the tangent at 5%. A fibre network goes from
               bending-dominated to stretch-dominated near 10--20% and raises this by one to two
               orders of magnitude (Storm 2005, Licup 2015). An isotropic hyperelastic solid raises
               it slightly -- fixed-corotated has a J-dependent volumetric term -- so the honest
               prediction here is "of order 1, not of order 10".
  plasticity   the residual strain after unloading. A network rearranges and keeps some; an elastic
               solid returns. Prediction: ~0.
  Poisson      the transverse expansion over the axial compression. A solid bulges at the nu of its
               own Lame pair (0.2 here, so ~0.2); a network densifies instead and gives 0 to 0.1.
               This is the one that separates "material" from "architecture" most sharply, and it
               is measured on the same frames as the other two.

WHAT MAKES IT A GATE RATHER THAN A PICTURE. Each number has its prediction written above, before
the run; a result near the network value would mean the fibres are doing something the constitutive
law cannot do, which would be a defect in the solver rather than a discovery.

THE PLATES ARE GRID BOUNDARY CONDITIONS, which is how MPM states an obstacle -- the same device
`test_03`'s floor uses and the stock "material pours around a sphere" demo uses. A moving plate is
a moving velocity constraint on the grid: above the top plate the grid's velocity is set to the
plate's, below the bottom one it is zeroed. Loading through particles instead would be loading
through the very thing under test.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
OFF = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)])


def _bspline(x, inv_dx):
    base = (x * inv_dx - 0.5).floor()
    fx = x * inv_dx - base
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 0)
    return base.long(), fx, w


class Press:
    """A fibre block between two plates, compressed at a constant rate and then released.

    The material, the seeding and the damping are 02h's: `seed_ecm`'s straight strands of 20
    particles, fixed-corotated at the stroma's modulus, `drag` on the particle velocity. Written
    here as one loop rather than through the engine for the same reason `test_03` is: the loading
    is a moving grid boundary condition on a schedule, which is a rig and not an operator.
    """

    def __init__(self, dev="cuda:0", n_grid=64, ppc=4, E=15.0, nu=0.2, rho=1.0, drag=8.0,
                 dt=4.0e-4, block=(0.30, 0.30, 0.30, 0.70, 0.70, 0.70), n_fibres=3000,
                 fibre_len=0.09, jitter=0.004, strain=0.20, rate=2.0e-4, seed=0):
        self.dev, self.n_grid, self.dt, self.drag = dev, n_grid, dt, drag
        self.dx = 1.0 / n_grid; self.inv_dx = float(n_grid)
        g = torch.Generator().manual_seed(seed)
        lo = torch.tensor(block[:3]); hi = torch.tensor(block[3:])
        # STRANDS, laid the way `seed_ecm` lays them: a centre, a direction, `per` points along it,
        # and a little across-strand jitter. Isotropic directions -- an aligned block would answer a
        # different question and `note_fibre` already measured that alignment buys 1.5x.
        d = torch.randn(n_fibres, 3, generator=g)
        d = d / d.norm(dim=1, keepdim=True)
        inset = 0.5 * fibre_len
        c = lo + inset + torch.rand(n_fibres, 3, generator=g) * ((hi - inset) - (lo + inset))
        per = 20
        t = (torch.arange(per, dtype=torch.float32) / (per - 1) - 0.5) * fibre_len
        x = (c[:, None, :] + t[None, :, None] * d[:, None, :]).reshape(-1, 3)
        x = x + torch.randn(x.shape, generator=g) * jitter
        self.x = x.to(dev)
        self.x0 = self.x.clone()
        self.N = self.x.shape[0]
        self.per, self.n_fibres = per, n_fibres
        self.v = torch.zeros_like(self.x)
        self.C = torch.zeros(self.N, 3, 3, device=dev)
        self.F = torch.eye(3, device=dev).expand(self.N, 3, 3).contiguous()
        self.vol = self.dx ** 3 / ppc
        self.m = rho * self.vol
        self.mu_l = E / (2 * (1 + nu)); self.la = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.off = OFF.to(dev)
        # THE PLATES SIT ON THE MATERIAL, NOT ON ITS SPARSEST TAIL. A random strand fill has long
        # thin tails -- the block's full z extent is 0.405 while its 1--99 percentile spread is
        # 0.323 -- so a plate placed at max(z) spends more than half its travel crossing a region
        # holding a per cent of the particles, and the run reports zero strain while the plate moves
        # 12% of the height. Both plates and BOTH ends of the strain measure now use the same
        # percentiles, so "the plate has moved x% of the height" and "the block is x% strained" are
        # statements about the same block.
        self.QL, self.QH = 0.005, 0.995
        self.z_lo = float(torch.quantile(self.x[:, 2], self.QL))
        self.z_hi = float(torch.quantile(self.x[:, 2], self.QH))
        # THE REFERENCE HEIGHT IS THE ONE THE STRAIN IS MEASURED WITH, and not the plate gap. `h`
        # below is a 1--99 percentile spread, so taking h0 as the full extent would make frame 0
        # report a strain of 0.2 before anything had moved -- a definition mismatch that reads
        # exactly like a block collapsing under its own seeding.
        self.h0 = self.z_hi - self.z_lo
        self.strain, self.rate = strain, rate
        self.plate_v = 0.0
        self.res = {k: [] for k in ("h", "strain", "stress", "w_xy", "vmax", "top")}

    def plate(self, t, n_load):
        """Down at a constant rate to `strain`, then up at the same rate. The release matters: the
        residual strain after unloading is the plasticity, and a run that only compresses cannot
        report it."""
        # A DISPLACEMENT PER STEP AND A VELOCITY ARE NOT THE SAME NUMBER, and using one as the other
        # is a factor of 1/dt = 2500 here. `rate` is how far the plate moves per step; the GRID needs
        # the speed that produces it, rate/dt. Handed `rate` directly, the constraint moved the
        # material at 8e-8 per step while the plate moved 2e-4 -- so the plate swept THROUGH the
        # block, leaving 2,748 particles above it by the end, and the only thing bringing the top of
        # the block down was the safety clamp two cells behind the plate. That clamp is a positional
        # update, so it carries no deformation: J stayed at 0.99998 through the whole press while the
        # height statistic fell 10.7%, which is the laundering of run 130 wearing a percentile.
        sgn = -1.0 if t < n_load else (+1.0 if t < 2 * n_load else 0.0)
        self.plate_v = sgn * self.rate / self.dt                    # velocity, box units per second
        self.z_hi += sgn * self.rate                                # displacement, per step
        return self.z_hi

    def step(self):
        dev, ng, dt, inv_dx = self.dev, self.n_grid, self.dt, self.inv_dx
        gm = torch.zeros(ng ** 3, device=dev); gmv = torch.zeros(ng ** 3, 3, device=dev)
        U, S, Vh = torch.linalg.svd(self.F)
        R = U @ Vh
        J = torch.linalg.det(self.F).clamp_min(1e-6)
        P = (2 * self.mu_l * (self.F - R) @ self.F.transpose(1, 2)
             + self.la * ((J - 1) * J)[:, None, None] * torch.eye(3, device=dev))
        # THE STRESS THE PLATE FEELS is the one the solver already computed: sigma = tau / J, and
        # the axial component averaged over the particles in the top layer is the traction. Taken
        # from P rather than from a separate force sum, so the reported modulus is the material's.
        sig = P @ self.F.transpose(1, 2) / J[:, None, None]
        affine = self.m * self.C - (4 * inv_dx * inv_dx * dt * self.vol) * P
        base, fx, w = _bspline(self.x, inv_dx)
        for o in self.off:
            ww = w[o[0], :, 0] * w[o[1], :, 1] * w[o[2], :, 2]
            dpos = (o.float() - fx) * self.dx
            idx = ((base[:, 0] + o[0]).clamp(0, ng - 1) * ng * ng
                   + (base[:, 1] + o[1]).clamp(0, ng - 1) * ng
                   + (base[:, 2] + o[2]).clamp(0, ng - 1))
            gm.index_add_(0, idx, ww * self.m)
            gmv.index_add_(0, idx, ww[:, None] * (self.m * self.v
                                                  + torch.einsum('nij,nj->ni', affine, dpos)))
        gv = gmv / gm.clamp_min(1e-12)[:, None]
        gv = gv * (1.0 - self.drag * dt)                       # Stokes drag, 02h's `drag`
        # THE PLATES, as grid velocity constraints
        kz = torch.arange(ng, device=dev).repeat(ng * ng).float() * self.dx
        gv[kz < self.z_lo, 2] = gv[kz < self.z_lo, 2].clamp_min(0.0)
        top = kz > self.z_hi
        gv[top, 2] = torch.minimum(gv[top, 2], torch.full_like(gv[top, 2], self.plate_v))
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
        # NO POSITIONAL CLAMP, AND THIS RIG COMMITTED THAT DEFECT ONCE. Clamping the particles
        # between the plates moved them without the grid seeing it, so `F` never registered the
        # squeeze: the height followed the plate exactly, the transverse width did not move by one
        # part in ten thousand, and the reported axial stress at 20% strain was 5e-4 where the
        # material's own modulus says 3. That is precisely the laundering `note_spheroid_bm_ecm`
        # documents for run 130 -- a position imposed outside the solver carries no deformation --
        # reproduced here by the rig meant to measure the material. The plate acts ONLY as a grid
        # velocity constraint; the clamp below is a safety net two cells outside the plates, which
        # nothing should ever reach.
        self.x[:, 2] = self.x[:, 2].clamp(self.z_lo - 2 * self.dx, self.z_hi + 2 * self.dx)
        z = self.x[:, 2]
        h = float(torch.quantile(z, self.QH) - torch.quantile(z, self.QL))
        self.res["h"].append(h)
        self.res["strain"].append(1.0 - h / self.h0)
        # THE LOAD IS READ AT THE FIXED BOUNDARY, NOT UNDER THE MOVING ONE. Sampling a slab three
        # cells under the plate means sampling a window whose population changes as the plate
        # descends and as material extrudes past it, so the reported stress fell while the strain
        # rose and the tangent at high strain came back NEGATIVE. The floor does not move and
        # nothing leaves through it, so the traction there is the load the block is carrying.
        near = z < (self.z_lo + 3 * self.dx)
        self.res["stress"].append(float(-sig[near, 2, 2].mean()) if bool(near.any()) else 0.0)
        w_xy = 0.5 * (float(torch.quantile(self.x[:, 0], self.QH)
                            - torch.quantile(self.x[:, 0], self.QL))
                      + float(torch.quantile(self.x[:, 1], self.QH)
                              - torch.quantile(self.x[:, 1], self.QL)))
        self.res["w_xy"].append(w_xy)
        self.res["vmax"].append(float(self.v.norm(dim=1).max()))
        self.res["top"].append(float(self.z_hi))



def _panel(ax, letter):
    """A bold letter top-left and no title. The numbers a title used to carry go into the note's
    caption, where they can be read against the gate they belong to; a title repeats them in a place
    the figure cannot explain them."""
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def main():
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    d = os.path.join(LOG, "02i_compress")
    os.makedirs(d, exist_ok=True)
    rig = Press(dev=dev)
    # the plate travels `strain` * h0 at `rate` per frame, then back
    n_load = int(round(rig.strain * rig.h0 / rig.rate))
    frames = int(2.4 * n_load)
    print(f"[02i] {rig.N} particles on {rig.n_fibres} strands, block height {rig.h0:.4f}, "
          f"compressing to {100 * rig.strain:.0f}% over {n_load} frames, releasing over the next "
          f"{n_load}, {frames} total", flush=True)
    t0 = time.time()
    for t in range(frames):
        rig.plate(t, n_load)
        rig.step()
    print(f"[02i] {time.time() - t0:.0f} s", flush=True)

    e = np.asarray(rig.res["strain"]); s = np.asarray(rig.res["stress"])
    w = np.asarray(rig.res["w_xy"]); h = np.asarray(rig.res["h"])
    load = slice(0, n_load)

    def tangent(target, half=0.02):
        """d(stress)/d(strain) over a window of the LOADING branch centred on `target`."""
        m = (e[load] > target - half) & (e[load] < target + half)
        if m.sum() < 8:
            return float("nan")
        return float(np.polyfit(e[load][m], s[load][m], 1)[0])

    # THE HIGH-STRAIN WINDOW IS RELATIVE TO THE STRAIN REACHED, not to the strain asked for. The
    # plate travels 20% but the block compresses 17.6% -- the rest is material extending past the
    # plate -- so a window pinned at 0.18 sat beyond the data and fitted its last few points.
    e_hi = float(e[load].max())
    # THE TOE IS NOT THE MATERIAL, so neither tangent may be taken inside it. A random strand fill
    # is porous: the first several per cent of "strain" is the loose arrangement consolidating, and
    # the block carries no load at all until it is closed. Measured here, the stress is under 1% of
    # its maximum up to 7.4% strain. A tangent at 5% is therefore a tangent to nothing, which is
    # what returned a stiffening ratio of -562. The low window is placed at the END of the toe --
    # the first strain at which the stress passes 5% of its maximum -- and the high window near the
    # strain actually reached, so both are on the material.
    smax = float(s[load].max())
    toe = float(e[load][np.argmax(s[load] > 0.05 * smax)]) if smax > 0 else float("nan")
    t5, t20 = tangent(toe + 0.02, half=0.015), tangent(e_hi - 0.02, half=0.015)
    # Poisson: transverse strain over axial strain, over the loading branch
    ok = (e[load] > 0.03) & (e[load] < e[load].max() - 0.01)
    # SIGN. `e` is a COMPRESSIVE strain and is positive, so the axial strain is -e; the transverse
    # strain is +(w/w0 - 1) when the block bulges. nu = -eps_trans/eps_axial is then +d(w/w0)/de, and
    # the extra minus sign reported a healthy bulging solid as auxetic at -0.216.
    nu_eff = float(np.polyfit(e[load][ok], (w[load][ok] / w[0] - 1.0), 1)[0]) if ok.sum() > 8 \
        else float("nan")
    resid = float(1.0 - h[-1] / rig.h0)
    m = dict(
        particles=rig.N, n_fibres=rig.n_fibres, frames=frames, n_load=n_load,
        h0=rig.h0, strain_max=float(e.max()), drag=rig.drag, dt=rig.dt,
        tangent_low=t5, tangent_high=t20, toe_strain=toe, stress_max=float(s[load].max()),
        gates=dict(
            G_stiffening_ratio=dict(
                threshold="order 1 for this law; a fibre network gives 10-100",
                measured=float(t20 / t5) if t5 and np.isfinite(t5) else float("nan"),
                at_strain=[float(toe + 0.02), float(e_hi - 0.02)],
                why="the constitutive law reads the strand direction nowhere, so it cannot stiffen "
                    "by straightening"),
            G_residual_strain=dict(
                threshold="< 0.02 after release",
                measured=resid,
                why="an elastic solid returns; a rearranging network keeps some"),
            G_poisson=dict(
                threshold="0.15-0.30 for this Lame pair; a network densifies at 0-0.1",
                measured=nu_eff,
                why="bulging against densifying is the sharpest separation of material from "
                    "architecture")),
        series={k: [float(x) for x in v] for k, v in rig.res.items()})
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.4), facecolor="white")
    ax[0].plot(e[load], s[load], color="#e0452b", label="loading")
    ax[0].plot(e[n_load:2 * n_load], s[n_load:2 * n_load], color="#2b6cb0", label="release")
    ax[0].set_xlabel("axial strain"); ax[0].set_ylabel(r"axial stress $-\sigma_{zz}$")
    _panel(ax[0], "a")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot(e, w / w[0], color="#7b4fb5")
    ax[1].set_xlabel("axial strain"); ax[1].set_ylabel("transverse width / initial")
    _panel(ax[1], "b")
    ax[2].plot(np.arange(len(h)), h / rig.h0, color="#1f8a5c")
    ax[2].axhline(1.0, color="#999", ls="--", lw=0.8)
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("height / initial")
    _panel(ax[2], "c")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(d, "compress.png"), dpi=150, facecolor="white")
    plt.close(fig)
    for k, v in m["gates"].items():
        print(f"[gate] {k}: {v['measured']:.4g}  (threshold: {v['threshold']})", flush=True)
    print(f"[02i] -> {d}", flush=True)


if __name__ == "__main__":
    main()
