"""External, generic plotting

`plexus.plot` is the ONLY place matplotlib lives. It reads a generated trajectory
(graphs_data/<type>/<name>/trajectory.npz) plus the spec's `plotting:` style block
and renders **generically over the sets present in the data** -- it never branches
on an operator or model name. A new simulation gets figures for free as long as it
writes the standard trajectory layout (`<set>__pos`, `<set>__occ`, optional
`<set>__node_type`).

Dispatch = data structure (which sets exist, do they carry a type) + declared
style (colormap, point size). New entity semantics (orientation arrows, fields as
heatmaps) plug in here as additional generic branches, not per-model code.

Figures land next to the data, in the dataset directory.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plexus.simulation import Simulation
from plexus.paths import graphs_data_path
from plexus.models.registry import get_entity
import plexus.models.entities  # noqa: F401  register entity render hints
from plexus.models.entities import DEFAULT_RENDER


def _sets_in(npz) -> list[str]:
    """Set names present in a trajectory file (keys '<set>__pos')."""
    return sorted({k[:-len("__pos")] for k in npz.files if k.endswith("__pos")})


def _render_meta(sname: str) -> dict:
    """Render hints for a set, from the entity registry (how to color/draw it)."""
    try:
        return getattr(get_entity(sname), "RENDER", None) or DEFAULT_RENDER
    except KeyError:
        return DEFAULT_RENDER


def _point_size(style: dict, n: int) -> float:
    if "point_size" in style:
        return float(style["point_size"])
    return 8.0 if n <= 4000 else (4.0 if n <= 12000 else 2.0)


def plot_dataset(sim: Simulation, pre_folder: str, movie: bool = False) -> str:
    """Render a generated simulation's trajectory into its dataset directory.
    Returns the directory written to."""
    folder = pre_folder.rstrip("/")
    data_dir = graphs_data_path(folder, sim.name)
    npz_path = os.path.join(data_dir, "trajectory.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"no trajectory at {npz_path} (run `-o generate` first)")
    d = np.load(npz_path)

    style = sim.plotting or {}
    cmap = plt.get_cmap(style.get("colormap", "tab10"))
    W = float(d["world"]) if "world" in d.files else sim.world

    for sname in _sets_in(d):
        pos = d[f"{sname}__pos"]                      # [T, N, 2]
        occ = d[f"{sname}__occ"]                      # [T, N]
        cby = _render_meta(sname).get("color_by")     # which per-node field colors the points
        nt = d[f"{sname}__{cby}"] if (cby and f"{sname}__{cby}" in d.files) else None
        T, N = pos.shape[0], pos.shape[1]
        s = _point_size(style, N)
        color = (cmap(nt % cmap.N) if nt is not None else None)

        def _draw(ax, i):
            live = occ[i]
            c = (color[live] if color is not None else "#1f77b4")
            ax.scatter(pos[i, live, 0], pos[i, live, 1], s=s, c=c, linewidths=0)
            ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

        # evolution montage
        idx = [0, T // 5, 2 * T // 5, 3 * T // 5, T - 1]
        fig, axes = plt.subplots(1, len(idx), figsize=(len(idx) * W * 3.2, 3.4))
        for ax, i in zip(np.atleast_1d(axes), idx):
            _draw(ax, i); ax.set_title(f"frame {i * sim.record_every}", fontsize=9)
        fig.suptitle(f"{folder}/{sim.name} — {sname}", fontsize=11)
        plt.tight_layout()
        evo = os.path.join(data_dir, f"fig_{sname}_evolution.png")
        plt.savefig(evo, dpi=100); plt.close(fig)

        # final frame
        fig, ax = plt.subplots(figsize=(6 * W, 6))
        _draw(ax, T - 1); ax.set_title(f"{sim.name} — {sname} (final)", fontsize=11)
        plt.tight_layout()
        fin = os.path.join(data_dir, f"fig_{sname}_final.png")
        plt.savefig(fin, dpi=110); plt.close(fig)
        print(f"[plot] {sname}: {os.path.basename(evo)}, {os.path.basename(fin)}", flush=True)

        if movie:
            _movie(pos, occ, color, s, W, T, os.path.join(data_dir, f"movie_{sname}.gif"))

    print(f"[plot] figures -> {data_dir}", flush=True)
    return data_dir


def _movie(pos, occ, color, s, W, T, out_path, max_frames: int = 80) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter
    stride = max(1, T // max_frames)
    frames = list(range(0, T, stride))
    fig, ax = plt.subplots(figsize=(5 * W, 5)); ax.set_xlim(0, W); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    sc = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=s, linewidths=0,
                    c=(color if color is not None else "#1f77b4"))

    def upd(i):
        live = occ[i]
        sc.set_offsets(pos[i, live])
        if color is not None:
            sc.set_color(color[live])
        return sc,

    FuncAnimation(fig, upd, frames=frames, interval=50).save(out_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"[plot] movie -> {os.path.basename(out_path)}", flush=True)
