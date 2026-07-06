#!/usr/bin/env python
"""run_vertex_mpm -- item 3: LINK the cell-tissue operators to the MLS-MPM continuum.

Embeds a differential-adhesion cell tissue (the `differential_adhesion` operator) inside an MPM
ELASTIC medium (a soft deformable disc with a liquid surface-tension skin). The cells sort into
domains AND push on the elastic continuum (`agent_to_mpm`), which drags + confines them back
(`mpm_to_agent`) -- a two-way coupling between the cell layer and the bulk material. Renders the
MPM medium (faint) with the sorting cells (red/blue) on top.

    python run_vertex_mpm.py --device cuda:0
"""
from __future__ import annotations
import os, sys, tempfile, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "embryo_cell_sorting"))   # differential_adhesion

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators                # noqa: F401  MPM ops + radius_graph
import embryo_cell_sorting_ops         # noqa: F401  differential_adhesion
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive_mpm")


def make_sim():
    cfg = {
        "general": {"name": "vertex_mpm", "seed": 0, "n_frames": 400, "dt": 0.002, "boundary": "wall"},
        "sets": {
            "agent": {"n": 500, "spawn": "disc", "spawn_radius": 0.22,
                      "types": {"A": {"fraction": 0.5, "move_speed": 0.12},
                                "B": {"fraction": 0.5, "move_speed": 0.12}}},
            "cell": {"n": 1, "start": [[0.5, 0.5]],
                     "types": {"body": {"fraction": 1.0, "youngs": 120, "layers": [
                         {"frac": 0.88, "youngs": 120, "material": "elastic"},
                         {"frac": 1.0, "youngs": 40, "material": "liquid"}]}}},
            "mpm_particle": {"parent": "cell", "per_parent": 12000, "radius": 0.34, "density": 1.0},
        },
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 64}},
        "operators": [
            {"op": "radius_graph", "at": "agent", "radius": 0.035},
            {"op": "heading_align", "at": "agent", "gain": 4.0, "noise": 0.5},   # heading (self-propulsion into MPM)
            {"op": "glide", "at": "agent"},                                      # move along heading
            {"op": "mpm_spin", "at": "mpm_particle", "omega": 0.4, "spin_k": 20.0},
            {"op": "mpm_strain", "at": "mpm_particle"},
            {"op": "p2g", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.3, "a_max": 200},
            {"op": "agent_to_mpm", "at": "agent", "to": "mpm_grid", "agent_mass": 4.0e-5, "k": 1.0},
            {"op": "mpm_grid_update", "at": "mpm_grid", "surface_tension": 120.0, "wall_damp": 0.7},
            {"op": "g2p", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 0.7,
             "wall_contact": 0.04, "vmax": 1.0e9},
            {"op": "mpm_to_agent", "at": "agent", "from": "mpm_grid", "k": 1.0, "confine": 260.0},
        ],
        "schedule": [
            "radius_graph", "heading_align", "glide", "mpm_spin",
            {"substep_dt": 0.0002, "steps": ["mpm_strain", "p2g", "agent_to_mpm",
                                             "mpm_grid_update", "g2p"]},
            "mpm_to_agent",
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def render(agent_pos, agent_type, mpm_pos, outdir, seconds=14.0, max_frames=200):
    os.makedirs(outdir, exist_ok=True)
    T = agent_pos.shape[0]
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    fps = max(1, round(len(idx) / seconds))
    RED = np.array([0.95, 0.3, 0.25]); BLUE = np.array([0.3, 0.5, 0.95])
    col = np.where(agent_type[:, None] == 0, RED[None], BLUE[None])

    def draw(ax, t):
        ax.clear(); ax.set_facecolor("black")
        if mpm_pos is not None:
            m = mpm_pos[t]
            ax.scatter(m[::6, 0], m[::6, 1], s=1.0, c="#3a3a55", alpha=0.5, linewidths=0)  # elastic medium
        a = agent_pos[t]
        ax.scatter(a[:, 0], a[:, 1], s=10, c=col, linewidths=0)
        ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 0.9); ax.set_aspect("equal"); ax.axis("off")

    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
    sfig, sax = plt.subplots(1, 4, figsize=(4 * 2.3, 2.4)); sfig.patch.set_facecolor("black")
    for a_, t in zip(sax, picks):
        draw(a_, t); a_.set_title(f"{int(100*t/max(T-1,1))}%", color="white", fontsize=9)
    sfig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.05)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black"); plt.close(sfig)
    w = FFMpegWriter(fps=fps, metadata={"title": "vertex_mpm"})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=110):
        for t in idx:
            draw(ax, t); w.grab_frame()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sim, cfg = make_sim()
    yaml.safe_dump(cfg, open(os.path.join(OUT, "spec.yaml"), "w"), sort_keys=False)
    _, out = engine_run(sim, device=a.device)
    ag = out["sets"]["agent"]; agent_pos = ag["pos"]; agent_type = ag["node_type"]
    mpm = out["sets"].get("mpm_particle", {}); mpm_pos = mpm.get("pos")
    print("agent pos:", agent_pos.shape, " mpm pos:", None if mpm_pos is None else mpm_pos.shape)
    render(agent_pos, agent_type, mpm_pos, OUT)
    print(f"done -> {OUT}/movie.mp4 + strip.png")


if __name__ == "__main__":
    main()
