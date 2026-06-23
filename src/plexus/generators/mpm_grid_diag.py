"""The MLS-MPM 6-panel grid-diagnostic movie -- ported from MPM_pytorch's grid plotter
(`MPM_pytorch/generators/graph_data_generator.py`, the `if 'grid' in style:` block).

Panels per frame: objects (particles by type) | C (||affine velocity||) |
F (||deformation gradient||) | Jp (plastic volume) | stress (||fixed-corotated||) |
grid momentum (||grid velocity||). None of F/C/Jp/stress/grid live in the generic
Plexus trajectory (it records only positions + recorded fields), so this RE-RUNS the
sim with the engine's per-frame hook (`run(..., on_frame=...)`) to snapshot those live
buffers, recomputes the stress exactly as `p2g` does, then stitches `grid.mp4`.

Self-contained MPM diagnostic: it does not touch the generic engine/trajectory format.
"""
from __future__ import annotations

import os
import glob
import subprocess

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

_BG, _FG = "black", "white"      # dark theme to match the MPM_pytorch grid plots

from plexus.engine import run
from plexus.plot import _ffmpeg, _draw_obstacles


_TYPE_COLORS = [(0.20, 0.55, 0.95), (0.95, 0.30, 0.30), (0.25, 0.75, 0.35),
                (0.85, 0.65, 0.15), (0.60, 0.35, 0.80)]


def _stress_norm(F, mu, la):
    """Per-particle ||fixed-corotated stress||, the same material law p2g scatters
    (2*mu*(F-R)F^T + la*J(J-1)I), R = analytic 2x2 polar rotation."""
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    J = a * d - b * c
    cs, sn = (a + d), (c - b)
    r = torch.sqrt(cs * cs + sn * sn) + 1e-9
    cs, sn = cs / r, sn / r
    R = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)
    eye = torch.eye(2, device=F.device).expand_as(F)
    stress = 2 * mu[:, None, None] * ((F - R) @ F.transpose(-2, -1)) \
        + eye * (la * J * (J - 1))[:, None, None]
    return stress.reshape(F.shape[0], -1).norm(dim=1)


def _capture(H, particle_set="mpm_particle", grid_field="mpm_grid"):
    """Snapshot the live MPM diagnostic arrays from the hierarchy at one frame."""
    p = H.levels[particle_set]
    X = p.get("pos").detach().cpu().numpy()
    nt = (p.node_type.detach().cpu().numpy() if hasattr(p, "node_type")
          else np.zeros(p.n, dtype=int))
    F, C = p.F.detach(), p.C.detach()
    rec = {
        "X": X, "nt": nt,
        "fnorm": F.reshape(p.n, -1).norm(dim=1).cpu().numpy(),
        "cnorm": C.reshape(p.n, -1).norm(dim=1).cpu().numpy(),
        "Jp": (p.Jp.detach().cpu().numpy() if hasattr(p, "Jp") else np.ones(p.n)),
        "stress": _stress_norm(F, p.mu, p.la).cpu().numpy(),
    }
    g = H.fields[grid_field] if (hasattr(H, "fields") and grid_field in H.fields) else None
    if g is not None and hasattr(g, "v"):
        nx, ny, dx = g.nx, g.ny, g.dx
        gv = g.v.detach().reshape(nx, ny, 2).norm(dim=2).cpu().numpy()
        xs = (np.arange(nx) + 0.5) * dx
        ys = (np.arange(ny) + 0.5) * dx
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        rec["grid"] = (gx[::2, ::2].ravel(), gy[::2, ::2].ravel(), gv[::2, ::2].ravel())
    return rec


def _rng(vals, lo=1.0, hi=99.0, pad=1e-9):
    """Robust (percentile) colour range across all captured frames, so the movie does
    not flicker and is scaled to THIS run (not MPM_pytorch's hard-coded tissue ranges)."""
    a = np.concatenate([v.ravel() for v in vals]) if vals else np.array([0.0, 1.0])
    vmin, vmax = np.percentile(a, lo), np.percentile(a, hi)
    if vmax - vmin < pad:
        vmax = vmin + pad
    return float(vmin), float(vmax)


def generate_grid_movie(sim, data_dir: str, device: str = "cpu", stride: int = 3,
                        fps: int = 25) -> str | None:
    """Re-run `sim`, capture the MPM diagnostic buffers every `stride` frames, render the
    6-panel figure per frame into `<data_dir>/Grid/`, and stitch `<data_dir>/grid.mp4`."""
    if "mpm_particle" not in sim.sets:
        print("[grid] no mpm_particle set -- skipping grid diagnostic movie", flush=True)
        return None

    frames: list[dict] = []

    def hook(H, frame):
        if frame % stride == 0:
            frames.append(_capture(H))

    print(f"[grid] re-running {sim.n_frames} frames to capture MPM diagnostics...", flush=True)
    run(sim, out_path=None, device=device, on_frame=hook)
    if not frames:
        return None

    # per-run colour ranges (Jp kept on the physical band, like the reference)
    c_lo, c_hi = _rng([f["cnorm"] for f in frames])
    f_lo, f_hi = _rng([f["fnorm"] for f in frames])
    s_lo, s_hi = _rng([f["stress"] for f in frames])
    has_grid = "grid" in frames[0]
    g_lo, g_hi = _rng([f["grid"][2] for f in frames]) if has_grid else (0.0, 1.0)

    grid_dir = os.path.join(data_dir, "Grid")
    os.makedirs(grid_dir, exist_ok=True)
    for old in glob.glob(os.path.join(grid_dir, "*.png")):
        os.remove(old)

    obstacles = list(getattr(sim, "obstacles", []) or [])   # wall geometry to overlay (grey)

    def _style(ax, title):
        ax.set_title(title, color=_FG, fontsize=8)
        _draw_obstacles(ax, obstacles, color="0.55")       # same grey walls as the particle/cell movies
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_facecolor(_BG); ax.axis("off")              # no boundary box / ticks

    def _cbar(ax, sc):
        # a small (thin + short) colorbar inset just outside the panel's right edge. It
        # steals no axis width, so all six main axes stay exactly the same size.
        if sc is None:
            return
        cax = inset_axes(ax, width="100%", height="100%", borderpad=0,
                         bbox_to_anchor=(1.012, 0.30, 0.016, 0.40), bbox_transform=ax.transAxes)
        cb = plt.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=5, colors=_FG, length=2, width=0.4, pad=1.0)
        cb.ax.locator_params(nbins=4)
        cb.outline.set_edgecolor(_FG); cb.outline.set_linewidth(0.3)

    def panel(sp, title, X, c, cmap, vmin, vmax):
        ax = plt.subplot(2, 3, sp)
        sc = ax.scatter(X[:, 0], X[:, 1], c=c, s=1, cmap=cmap, vmin=vmin, vmax=vmax)
        _style(ax, title); _cbar(ax, sc)

    for i, fr in enumerate(frames):
        X = fr["X"]
        plt.figure(figsize=(15, 10), facecolor=_BG)
        ax = plt.subplot(2, 3, 1)
        for t in np.unique(fr["nt"]):
            m = fr["nt"] == t
            ax.scatter(X[m, 0], X[m, 1], s=1, color=_TYPE_COLORS[int(t) % len(_TYPE_COLORS)])
        _style(ax, "objects"); _cbar(ax, None)             # spacer keeps panel 1 the same size
        panel(2, "C (Jacobian of velocity)", X, fr["cnorm"], "viridis", c_lo, c_hi)
        panel(3, "F (deformation)", X, fr["fnorm"], "coolwarm", f_lo, f_hi)
        panel(4, "Jp (volume deformation)", X, fr["Jp"], "viridis", 0.75, 1.25)
        panel(5, "stress", X, fr["stress"], "hot", s_lo, s_hi)
        ax6 = plt.subplot(2, 3, 6)
        if has_grid:
            gx, gy, gv = fr["grid"]
            sc = ax6.scatter(gx, gy, c=gv, s=4, cmap="viridis", vmin=g_lo, vmax=g_hi)
            _style(ax6, "grid momentum"); _cbar(ax6, sc)
        else:
            _style(ax6, "grid momentum"); _cbar(ax6, None)
        plt.tight_layout()
        plt.savefig(os.path.join(grid_dir, f"Fig_{i:06d}.png"), dpi=100, facecolor=_BG)
        plt.close()

    out = os.path.join(data_dir, "grid.mp4")
    ff = _ffmpeg()
    if ff is None:
        print(f"[grid] rendered {len(frames)} PNGs -> {grid_dir} (ffmpeg not found; no mp4)", flush=True)
        return grid_dir
    subprocess.run([ff, "-y", "-framerate", str(fps), "-i", os.path.join(grid_dir, "Fig_%06d.png"),
                    "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[grid] grid diagnostic movie -> {out} ({len(frames)} frames)", flush=True)
    return out
