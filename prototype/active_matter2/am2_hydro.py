#!/usr/bin/env python
"""am2_hydro -- the HYDRODYNAMIC field theory of communicating active matter.

Integrates the coupled continuum equations of Ziepke, Maryshev, Aranson & Frey,
Nat. Commun. 13:6727 (2022), Eqs 6-9, for the density rho(r,t), polarization
p(r,t)=(px,py), internal state s(r,t) and chemical signal c(r,t):

    d_t rho = -v0 div(p) + Drho lap(rho)                                   (6)
    d_t p   = sigma (rho-1) p - delta |p|^2 p + Dp lap(p)
              - chi (p.grad) p - Q(rho) grad(rho) + rho omega grad(c)      (7)
    d_t c   = Dc lap(c) - alpha c + rho beta Theta(c-c_th)(1-s)            (8)
    d_t s   = Drho lap(s) + eps (c - s) - v0 (p.grad) s                    (9)

Periodic domain, explicit Euler, central-difference operators (torch.roll). This is
the field-theory sibling of the agent-based `am2_*` specs: the same aggregation +
excitable chemical waves emerge as a continuum. Renders the paper's hydrodynamic
panels: polarization orientation (HSV, brightness = rho) over the chemical field c.

Usage (repo root; conda env):
    python prototype/active_matter2/am2_hydro.py                  # nominal
    python prototype/active_matter2/am2_hydro.py --preset vortex  # spiral-wave regime
    DEVICE=cuda:0 python prototype/active_matter2/am2_hydro.py
Outputs -> $GNN_OUTPUT_ROOT/graphs_data/active_matter2/hydro_<preset>/.
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.animation import FFMpegWriter

try:                                                   # use imageio's bundled ffmpeg (no system ffmpeg needed)
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass


# --------------------------------------------------------------------------- #
#  periodic finite-difference operators (torch.roll)
# --------------------------------------------------------------------------- #
def ddx(f, dx):  return (torch.roll(f, -1, 0) - torch.roll(f, 1, 0)) / (2 * dx)
def ddy(f, dx):  return (torch.roll(f, -1, 1) - torch.roll(f, 1, 1)) / (2 * dx)
def lap(f, dx):
    return (torch.roll(f, -1, 0) + torch.roll(f, 1, 0)
            + torch.roll(f, -1, 1) + torch.roll(f, 1, 1) - 4 * f) / dx ** 2


PRESETS = {
    # EXCITABLE balance: strong decay+adaptation so c returns below threshold after
    # firing -> travelling wave fronts (not a saturated uniform field) + rho aggregation
    # c_th < 0: always-on refractory emission (gated by 1-s) -> an OSCILLATORY/excitable
    # medium that sustains travelling waves, as in the agent specs (c never reaches a flat
    # steady state because the slow refractory s + diffusion keep re-igniting fronts).
    "nominal": dict(sigma=0.6, delta=0.6, chi=0.5, Drho=0.5, Dp=0.5, Dc=0.9,
                    v0=0.6, Q=0.6, alpha=0.45, beta=0.6, eps=0.06, omega=1.2, c_th=-1.0),
    # strong chemotaxis + slow adaptation -> spiral / vortex waves
    "vortex":  dict(sigma=0.7, delta=0.6, chi=0.5, Drho=0.5, Dp=0.6, Dc=1.1,
                    v0=0.6, Q=0.5, alpha=0.42, beta=0.6, eps=0.045, omega=1.8, c_th=-1.0),
    # alignment-dominated, weak chemotaxis -> polar bands
    "bands":   dict(sigma=0.9, delta=0.7, chi=0.8, Drho=0.4, Dp=0.5, Dc=1.0,
                    v0=0.9, Q=0.4, alpha=0.40, beta=0.4, eps=0.06, omega=0.3, c_th=-1.0),
    # BASE for the Fig.2 / Fig.3 (v0, omega) sweeps: the aggregating vortex regime with
    # v0 & omega as the swept axes. NB our nondimensionalization puts the aggregation
    # threshold near omega~1 (not the paper's 0.05), so the omega axis is rescaled.
    "fig":     dict(sigma=0.7, delta=0.6, chi=0.5, Drho=0.5, Dp=0.6, Dc=1.1,
                    v0=0.6, Q=0.5, alpha=0.42, beta=0.6, eps=0.045, omega=1.8, c_th=-1.0),
}


def run(preset="nominal", N=220, L=110.0, dt=0.02, nsteps=48000, rec_every=240,
        seed=0, device="cpu", overrides=None):
    P = dict(PRESETS[preset])
    if overrides:
        P.update(overrides)                            # e.g. sweep {"v0":..,"omega":..,"sigma":0.02}
    dx = L / N
    g = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, device=device)

    rho0 = P.get("rho0", 1.2)                          # mean density; rho_c=1 is the flocking onset.
    # rho0 >> 1 -> stable homogeneous flock; rho0 ~ 1.0-1.05 sits in the Toner-Tu banding window
    # (ordered dense band coexists with disordered dilute gas) -> travelling polar bands.
    rho = rho0 + 0.05 * rand(N, N)                      # above rho_c=1 -> polar order grows
    px = 0.05 * rand(N, N)
    py = 0.05 * rand(N, N)
    s = torch.zeros(N, N, device=device)
    c = torch.zeros(N, N, device=device)
    # seed the excitable medium: a handful of Gaussian bumps nucleate waves
    xs = torch.linspace(0, L, N, device=device)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    for _ in range(14):
        cx, cy = float(torch.rand(1, generator=g, device=device) * L), float(torch.rand(1, generator=g, device=device) * L)
        c = c + 1.0 * torch.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * (2.5 * dx) ** 2))

    frames = []
    for it in range(nsteps + 1):
        p2 = px * px + py * py
        gcx, gcy = ddx(c, dx), ddy(c, dx)
        grx, gry = ddx(rho, dx), ddy(rho, dx)
        divp = ddx(px, dx) + ddy(py, dx)
        # (p.grad) advection
        adv_px = px * ddx(px, dx) + py * ddy(px, dx)
        adv_py = px * ddx(py, dx) + py * ddy(py, dx)
        adv_s = px * ddx(s, dx) + py * ddy(s, dx)

        drho = -P["v0"] * divp + P["Drho"] * lap(rho, dx)
        dpx = (P["sigma"] * (rho - 1) * px - P["delta"] * p2 * px + P["Dp"] * lap(px, dx)
               - P["chi"] * adv_px - P["Q"] * grx + rho * P["omega"] * gcx)
        dpy = (P["sigma"] * (rho - 1) * py - P["delta"] * p2 * py + P["Dp"] * lap(py, dx)
               - P["chi"] * adv_py - P["Q"] * gry + rho * P["omega"] * gcy)
        emit = rho * P["beta"] * (c > P["c_th"]).float() * (1 - s).clamp(min=0)
        dc = P["Dc"] * lap(c, dx) - P["alpha"] * c + emit
        ds = P["Drho"] * lap(s, dx) + P["eps"] * (c - s) - P["v0"] * adv_s

        rho = (rho + dt * drho).clamp(min=0.0)
        px = px + dt * dpx
        py = py + dt * dpy
        c = (c + dt * dc).clamp(0.0, 3.0)
        s = (s + dt * ds).clamp(0.0, 1.0)

        if it % rec_every == 0:
            frames.append((rho.detach().cpu().numpy().copy(),
                           px.detach().cpu().numpy().copy(),
                           py.detach().cpu().numpy().copy(),
                           c.detach().cpu().numpy().copy(),
                           s.detach().cpu().numpy().copy()))   # s -> real processing-rate R
            if not torch.isfinite(rho).all():
                print(f"[hydro] blew up at step {it} -- lower dt"); break
    return frames


def count_clusters(rho, rel=0.55, abs_frac=0.0):
    """Number of density aggregates: connected components of rho above a threshold.
    Threshold = max(mean + rel*std, mean*(1+abs_frac)). Periodic wrap is ignored.

    abs_frac is an ABSOLUTE contrast floor (peak must be >= (1+abs_frac)*mean) that
    rejects the near-uniform IC: rho starts as 1.2+0.05*noise, whose std-based threshold
    (mean+0.55*std) sits ~3% above mean, so ~29% of noise pixels cross it -> hundreds of
    speck 'clusters' at frame 0 BEFORE any droplet nucleates. That noise-inflated Nc_max
    (~765) is why Nc(t) only ever decayed and the paper's nucleation PLATEAU never showed.
    A modest abs_frac (~0.15) keeps the IC below threshold (Nc~0) so Nc RISES as real
    droplets condense -> plateau -> ~t^-1 merge -> faster, matching Fig.3a. abs_frac=0
    (default) preserves the agent-path / snapshot callers unchanged."""
    from scipy import ndimage
    m = rho.mean()
    thr = max(m + rel * rho.std(), m * (1.0 + abs_frac))
    lbl, n = ndimage.label(rho > thr)
    if n == 0:
        return 0
    sizes = ndimage.sum(np.ones_like(rho), lbl, range(1, n + 1))
    return int((sizes >= 4).sum())                     # drop specks (<4 px)


def field_info_bytes(f):
    """Information content proxy (paper's LZW/PNG file size): losslessly PNG-compress
    the normalized field and return the byte count in kB."""
    import io
    from PIL import Image
    a = f - f.min()
    a = (255 * a / (a.max() + 1e-12)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=False)
    return len(buf.getvalue()) / 1024.0


def emission_rate(rho, c, s, P):
    """Total signalling activity R ~ integral of the relay source rho*beta*(1-s)Theta
    over the domain -- the fraction of the medium actively processing/emitting."""
    fire = (c > P["c_th"]).astype(np.float64) * np.clip(1 - s, 0, None)
    return float((rho * P["beta"] * fire).mean())


def _orient_rgb(rho, px, py):
    """Polarization orientation as HSV: hue = angle(p), value = normalized rho."""
    ang = (np.arctan2(py, px) + np.pi) / (2 * np.pi)         # [0,1] hue
    v = np.clip((rho - rho.min()) / (np.ptp(rho) + 1e-9), 0, 1)
    hsv = np.stack([ang, np.ones_like(ang), v ** 0.7], -1)
    return hsv_to_rgb(hsv)


def render(frames, outdir, preset):
    os.makedirs(outdir, exist_ok=True)
    cmax = max(f[3].max() for f in frames) + 1e-9

    def panel(ax_top, ax_bot, fr):
        rho, px, py, c = fr[:4]
        ax_top.imshow(np.transpose(_orient_rgb(rho, px, py), (1, 0, 2)), origin="lower")
        ax_bot.imshow(c.T, origin="lower", cmap="magma", vmin=0, vmax=cmax)
        for ax in (ax_top, ax_bot):
            ax.set_xticks([]); ax.set_yticks([])

    # final-frame figure (paper panel: orientation over chemical)
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 8.4)); fig.patch.set_facecolor("black")
    panel(axes[0], axes[1], frames[-1])
    axes[0].text(0.03, 0.95, "a", transform=axes[0].transAxes, color="white",
                 fontsize=20, fontweight="bold", va="top")
    axes[1].text(0.03, 0.95, "b", transform=axes[1].transAxes, color="white",
                 fontsize=20, fontweight="bold", va="top")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.02)
    fig.savefig(os.path.join(outdir, "fig_hydro_final.png"), dpi=130, facecolor="black")
    plt.close(fig)

    # movie
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 8.4)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.02)
    w = FFMpegWriter(fps=20, metadata={"title": f"hydro_{preset}"})
    mp4 = os.path.join(outdir, "movie_hydro.mp4")
    with w.saving(fig, mp4, dpi=110):
        for fr in frames:
            axes[0].clear(); axes[1].clear()
            panel(axes[0], axes[1], fr)
            w.grab_frame()
    plt.close(fig)
    print(f"[hydro] {preset}: {len(frames)} frames -> {mp4}")
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nominal", choices=list(PRESETS))
    ap.add_argument("--all", action="store_true", help="run every preset")
    ap.add_argument("--N", type=int, default=160)
    ap.add_argument("--nsteps", type=int, default=40000)
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"
    root = os.environ.get("AM2_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    presets = list(PRESETS) if args.all else [args.preset]
    for pr in presets:
        print(f"\n===== hydro {pr} (device={dev}) =====", flush=True)
        frames = run(pr, N=args.N, nsteps=args.nsteps, device=dev)
        outdir = os.path.join(root, "graphs_data", "active_matter2", f"hydro_{pr}")
        render(frames, outdir, pr)


if __name__ == "__main__":
    main()
