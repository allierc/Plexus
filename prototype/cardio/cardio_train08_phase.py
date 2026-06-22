#!/usr/bin/env python
"""train_08_phase -- add a local activation-time (phase-lag) map phi(x,y).

Diagnosis (E19): amplitude correct, ORIENTATION wrong. Under synchronous activation
A_ij is inert (uniform u -> zero Laplacian), so the tissue has no mechanism for
directional organization. The missing degree of freedom is a small per-node activation
LAG phi(x,y): nodes fire slightly offset -> the contraction gets a phase gradient ->
directional/coherent flow -> loop orientation.

phi is ONE interpretable map, smooth by construction (a learnable coarse grid upsampled,
~10-node wavelength), fitted jointly with the UNet maps (stiffness, gain, fibre) and
scalars, under the sweep-winning amplitude+length loss. Reads specs/train_phase.yaml.

Outputs (archive/train_08_phase/): phase.png (recovered phi map), u_traces.png (activation
over the 10x10 nodes, showing the phase spread), trajectories png/mp4 (real blue vs fit
green), nodes mp4, properties, unet, run.log, and the resolved spec.

Run:  PYTHONPATH=src .../python prototype/cardio/cardio_train08_phase.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import cardio_stage2 as C            # noqa: E402
import cardio_real_fit as R          # noqa: E402
import cardio_sweep as S             # noqa: E402
from cardio_unet import UNet, maps_from_unet, load_image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "specs", "train_phase.yaml")
DT = 0.1


# --------------------------------------------------------------------------- #
class PhaseMap(nn.Module):
    """Smooth per-node activation lag phi(x,y) in ticks: learnable coarse grid -> upsample -> tanh."""
    def __init__(self, shape, coarse=6, max_lag=18.0):
        super().__init__()
        self.coarse = nn.Parameter(0.05 * torch.randn(1, 1, coarse, coarse))
        self.shape, self.max_lag = shape, float(max_lag)

    def forward(self):
        up = F.interpolate(self.coarse, size=self.shape, mode="bilinear", align_corners=True)
        return (self.max_lag * torch.tanh(up[0, 0])).reshape(-1)        # [N] ticks


def phase_shift(base_1d, phi):
    """base_1d [T] synchronous activation profile, phi [N] lag -> phased [T,N] (differentiable)."""
    T = base_1d.shape[0]
    t = torch.arange(T, dtype=torch.float32, device=base_1d.device)[:, None]
    q = (t - phi[None, :]).clamp(0, T - 1)
    lo = q.floor().long(); hi = (lo + 1).clamp(max=T - 1); f = q - lo.float()
    return (1 - f) * base_1d[lo] + f * base_1d[hi]


def full_forward_phase(sk, gn, fb, phi, sc, grid, spec):
    """Multi-cycle render with phased activation + boundary anchored to real."""
    m = spec["model"]
    H = C.build_tissue(grid, C._patterns({"pattern_wavelength": 6}), bwidth=m["boundary_width"])
    C.GridGraph({"neighbours": 8}).forward(H); lvl = H.level("tissue_particle")
    lvl.stiff = sk.detach(); lvl.gain = gn.detach(); lvl.fiber = fb.detach()
    Tt = spec["render"]["cycles"] * m["pulse"]["period"]
    pulse = C.PulseField({**m["pulse"], "t_start": 1})
    nag = C.ExcitableNagumo({**m["nagumo"], "dt": DT})
    lvl.state[:, 4] = -1.1994; lvl.state[:, 5] = (-1.1994 + m["nagumo"]["a"]) / m["nagumo"]["b"]
    base_1d, u_raw = [], []
    for t in range(Tt):
        H.tick = t; pulse.forward(H); nag.forward(H)
        u = lvl.get("u").squeeze(1).mean()
        base_1d.append(torch.sigmoid((u - m["signal"]["theta"]) / m["signal"]["eta"]))
        u_raw.append(u)                                                 # raw FHN u (with undershoot)
    base_1d = torch.stack(base_1d); u_raw = torch.stack(u_raw)          # [Tt] synchronous profiles
    phased = phase_shift(base_1d, phi.detach())                          # [Tt, N]
    real_ticks = torch.tensor(R.real_full(grid, Tt))
    mech = C.MpmMechanics({"beta": sc.beta.item(), "k_anchor": sc.k_anchor.item(),
                           "gamma": sc.gamma.item(), "aniso": sc.aniso.item(),
                           "dt": DT, "substeps": m["mechanics"]["substeps"]})
    lvl.state[:, 0:2] = lvl.X.clone(); lvl.state[:, 2:4] = 0
    rec = []
    with torch.no_grad():
        for t in range(Tt):
            lvl.act = lvl.gain * phased[t]
            mech.forward(H)
            lvl.state[lvl.boundary, 0:2] = real_ticks[t][lvl.boundary]
            if t % spec["render"].get("rec", 2) == 0:
                rec.append(lvl.get("pos").clone())
    return torch.stack(rec).numpy(), lvl, base_1d.detach(), u_raw.detach()


# --------------------------------------------------------------------------- #
def render_phase(phi, shape, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(phi.reshape(shape), cmap="coolwarm")
    cb = fig.colorbar(im, ax=ax, fraction=0.046); ax.set_xticks([]); ax.set_yticks([])
    cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.get_yticklabels(), color="white")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="black"); plt.close(fig)
    print(f"saved {out}")


def render_utraces(base_1d, phi, shape, out, grid_n=10, nticks=160):
    """10×10 GRID of green u-traces on black, laid out spatially — each node's phased
    activation in its own cell, so the phase-lag organization is visible across the field."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Hy, Wx = shape
    ii = np.linspace(0, Hy - 1, grid_n).round().astype(int)
    jj = np.linspace(0, Wx - 1, grid_n).round().astype(int)
    phased = phase_shift(base_1d, phi).detach().numpy()[:nticks]        # [nticks, N]  (raw FHN u)
    ylo, yhi = float(phased.min()) - 0.1, float(phased.max()) + 0.1
    fig, axs = plt.subplots(grid_n, grid_n, figsize=(grid_n, grid_n), facecolor="black")
    for r in range(grid_n):
        for c in range(grid_n):
            node = ii[grid_n - 1 - r] * Wx + jj[c]                      # row 0 at top = high-y
            ax = axs[r, c]; ax.set_facecolor("black")
            ax.axhline(0, color="#333", lw=0.5)                         # rest reference (see undershoot)
            ax.plot(phased[:, node], color="lime", lw=0.9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(ylo, yhi)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig(out, dpi=110, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
def main():
    spec_path = sys.argv[1] if len(sys.argv) > 1 else SPEC
    spec = yaml.safe_load(open(spec_path))
    GRID = tuple(spec["grid"]); m = spec["model"]; W = spec["loss"]
    print(f"=== {spec['name']}: + phase map (coarse {spec['learn']['phase']['coarse']}, "
          f"max_lag {spec['learn']['phase']['max_lag']}) ===")
    R.GRID = GRID; R.BWIDTH = m["boundary_width"]; R.PERIOD = m["pulse"]["period"]
    R.N_CYCLES = spec["render"]["cycles"]; R.AMP = spec["render"]["amp"]; R.FPS = spec["render"]["fps"]
    img = load_image(GRID)
    geo, base = R.setup_fit(GRID)
    base_1d = base[:, 0].clone()                                        # synchronous activation profile
    real = R.real_single_beat(GRID); rstats = S.traj_stats(real)

    net = UNet(); pm = PhaseMap(GRID, spec["learn"]["phase"]["coarse"], spec["learn"]["phase"]["max_lag"])
    sc = R.Scalars()
    k_learn = bool(m["mechanics"].get("k_anchor_learnable", False))   # train_08_02: global learnable spring
    sc.k_anchor.requires_grad_(k_learn)
    with torch.no_grad():
        sc.k_anchor.fill_(m["mechanics"]["k_anchor"])
    params = [p for p in list(net.parameters()) + list(pm.parameters()) + list(sc.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=spec["lr"])
    x = img[None, None]
    for it in range(spec["n_iter"]):
        opt.zero_grad()
        sk, gn, fb = maps_from_unet(net(x))
        phased = phase_shift(base_1d, pm())
        pred = R.resample(R.forward_beat(sk, gn, fb, sc, geo, phased))
        e = S.loss_terms(pred, real, rstats)
        loss = sum(W.get(k, 0.0) * e[k] for k in e)
        loss.backward(); opt.step()
        with torch.no_grad():
            sc.beta.clamp_(0.05, 2.0); sc.gamma.clamp_(0.05, 1.0); sc.aniso.clamp_(0.0, 1.2)
            if k_learn:
                sc.k_anchor.clamp_(0.02, 0.30)        # global learnable spring, bounded
        if it % 50 == 0 or it == spec["n_iter"] - 1:
            print(f"  it {it:4d}  " + " ".join(f"{k}={float(e[k]):.3f}" for k in
                  ("mse", "amp", "length", "area", "shape")) +
                  f"  φ±{pm().abs().max().item():.1f}t  beta={sc.beta.item():.3f}")
    with torch.no_grad():
        sk, gn, fb = maps_from_unet(net(x)); phi = pm()
        ef = {k: round(float(v), 4) for k, v in
              S.loss_terms(R.resample(R.forward_beat(sk, gn, fb, sc, geo, phase_shift(base_1d, phi))), real, rstats).items()}
    fit_pos, lvl, base_full, u_raw = full_forward_phase(sk, gn, fb, phi, sc, GRID, spec)
    real_rec = R.real_full(GRID, R.N_CYCLES * R.PERIOD)[::spec["render"].get("rec", 2)]
    rec_sc = {k: round(getattr(sc, k).item(), 3) for k in ("beta", "k_anchor", "gamma", "aniso")}
    print(f"  rel-errors {ef}  scalars {rec_sc}  phi range ±{phi.abs().max().item():.1f} ticks")

    d = os.path.join(HERE, "archive", spec["name"]); os.makedirs(d, exist_ok=True)
    render_phase(phi.detach().numpy(), GRID, os.path.join(d, "phase.png"))
    render_utraces(u_raw, phi, GRID, os.path.join(d, "u_traces.png"))
    R._unet_png(img, sk, gn, fb, GRID, os.path.join(d, "phase_unet.png"))
    C.render_properties(lvl, os.path.join(d, "phase_properties.png"))
    C.render_traj_png(fit_pos, os.path.join(d, "phase_trajectories.png"), GRID, amp=R.AMP, real=real_rec)
    C.render_trajectories(fit_pos, os.path.join(d, "phase_trajectories.mp4"), GRID, amp=R.AMP, fps=R.FPS, real=real_rec)
    C.render_nodes(fit_pos, os.path.join(d, "phase_nodes.mp4"), fps=R.FPS, stride=1, real=real_rec)
    shutil.copy(spec_path, os.path.join(d, "spec.yaml"))
    with open(os.path.join(d, "run.log"), "w") as f:
        f.write(f"experiment=phase-lag map phi(x,y) added (the missing directional DOF)\n")
        f.write(f"spec={spec}\nfinal_rel_errors={ef}\nscalars={rec_sc}\nphi_max_ticks={phi.abs().max().item():.2f}\n")
    print(f"  archived -> {d}")


if __name__ == "__main__":
    main()
