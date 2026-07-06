#!/usr/bin/env python
"""galaxy3d -- the strict-Plexus N-body galaxy in 3D (thick rotating disk + black hole).

Same operators as the 2D sweep (galaxy_ops: nbody_gravity + disk_ic, both dimension-generic)
but with a 3D world and a disk THICKNESS, so the disk has real out-of-plane structure. Renders
with an inclined, slowly-rotating camera (perspective-free orthographic projection; depth ->
brightness) so the spiral arms + disk thickness read as a 3D object on black.

Sweeps a curated grid around the 2D winner (small BH + heavy cold disk) x disk thickness.
10000 frames at N=25000 (compiled force). Splits across GPUs like galaxy_sweep:

    python galaxy3d.py --rank 0 --nproc 2 --device cuda:0 &
    python galaxy3d.py --rank 1 --nproc 2 --device cuda:1 &
    wait; python galaxy3d.py --montage
"""
from __future__ import annotations
import os, sys, argparse, glob, json, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

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

import plexus.operators           # noqa: F401
import galaxy_ops                 # noqa: F401  nbody_gravity + disk_ic (3D-capable)
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive", "galaxy_3d")
WORLD = 8.0                                        # cubic world; galaxy centred at (4,4,4)


def configs():
    """Zoom the 2D winner corner (small BH + heavy cold disk) into 3D, sweeping disk THICKNESS
    (thin vs puffy disk) and the cold/warm + spin balance that sets the spiral."""
    C = []
    N, disc_R, dt, nf = 25000, 1.5, 0.004, 10000
    for m_bh in (0.25, 0.5):
        for jitter in (0.04, 0.09):
            for spin in (0.88, 0.92):
                for thick in (0.05, 0.12):
                    name = f"bh{m_bh:g}_j{jitter:g}_s{spin:g}_t{thick:g}"
                    C.append((name, dict(N=N, disc_R=disc_R, dt=dt, nframes=nf, spin=spin,
                                         m_bh=m_bh, mass=2.0 / N, jitter=jitter, soft=0.08,
                                         thick=thick)))
    return C


def make_sim(p):
    cfg = {
        "general": {"name": "g3", "seed": 2, "n_frames": p["nframes"], "dt": p["dt"],
                    "boundary": "free", "dim": 3, "world": [WORLD, WORLD, WORLD]},
        "sets": {"star": {"n": p["N"], "types": {"s": {"fraction": 1.0, "mass": p["mass"]}}}},
        "fields": {},
        "operators": [
            {"op": "disk_ic", "at": "star", "G": 1.0, "softening": p["soft"],
             "disc_radius": p["disc_R"], "spin": p["spin"], "m_bh": p["m_bh"],
             "vel_jitter": p["jitter"], "thickness": p["thick"], "before_frame": 1},
            {"op": "nbody_gravity", "at": "star", "G": 1.0, "softening": p["soft"], "compile": True},
        ],
        "schedule": ["disk_ic", "nbody_gravity"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f); path = f.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def project(xyz, incl_deg, az_deg):
    """Orthographic camera: rotate about z (azimuth) then tilt about x (inclination).
    Returns screen (X, Y) and depth Z (larger = nearer the camera)."""
    p = xyz - WORLD / 2.0
    az, i = np.radians(az_deg), np.radians(incl_deg)
    x1 = p[:, 0] * np.cos(az) - p[:, 1] * np.sin(az)
    y1 = p[:, 0] * np.sin(az) + p[:, 1] * np.cos(az)
    z1 = p[:, 2]
    y2 = y1 * np.cos(i) - z1 * np.sin(i)
    z2 = y1 * np.sin(i) + z1 * np.cos(i)
    return x1, y2, z2


def render(pos, outdir, name, view, seconds=40.0, max_frames=1000, incl=42.0):
    """pos: [T, N, 3]. Inclined camera with a slow azimuth sweep; depth -> brightness."""
    os.makedirs(outdir, exist_ok=True)
    T = pos.shape[0]
    stride = max(1, -(-T // max_frames))
    idx = list(range(0, T, stride))
    fps = max(1, round(len(idx) / seconds))

    def draw(ax, xyz, az):
        ax.clear(); ax.set_facecolor("black")
        X, Y, Z = project(xyz, incl, az)
        order = np.argsort(Z)                                    # far -> near (near drawn last)
        b = (Z - Z.min()) / (np.ptp(Z) + 1e-9)                  # depth brightness 0..1
        ax.scatter(X[order], Y[order], s=1.3, c=b[order], cmap="bone", vmin=-0.15, vmax=1.15,
                   linewidths=0)
        ax.set_xlim(-view, view); ax.set_ylim(-view, view)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1)
    draw(ax, pos[-1], 35.0)
    fig.savefig(os.path.join(outdir, "fig_final.png"), dpi=120, facecolor="black")
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=100):
        for j, t in enumerate(idx):
            az = 20.0 + 80.0 * j / max(len(idx) - 1, 1)          # slow 20->100 deg camera sweep
            draw(ax, pos[t], az); w.grab_frame()
    plt.close(fig)


def diagnostics(pos_last):
    d = pos_last - WORLD / 2.0
    r = np.sqrt((d ** 2).sum(-1))
    view = 3.6
    retain = float((r < view).mean())
    z_rms = float(np.sqrt((d[r < view, 2] ** 2).mean())) if retain > 0 else float("nan")
    r_half = float(np.median(r[r < view])) if retain > 0 else float("nan")
    # in-plane m=2 (spiral) on a mid annulus
    ann = (r > 0.2 * view) & (r < 0.7 * view)
    if ann.sum() > 10:
        th = np.arctan2(d[ann, 1], d[ann, 0]); A2 = float(np.abs(np.exp(2j * th).mean()))
    else:
        A2 = 0.0
    return dict(retain=round(retain, 3), A2=round(A2, 3), r_half=round(r_half, 3),
                z_rms=round(z_rms, 3))


def run_share(rank, nproc, device):
    cfgs = configs()[rank::nproc]
    print(f"[rank {rank}] {len(cfgs)} configs on {device}", flush=True)
    for i, (name, p) in enumerate(cfgs):
        odir = os.path.join(OUT, name)
        if os.path.exists(os.path.join(odir, "fig_final.png")):
            print(f"[rank {rank}] skip {name}", flush=True); continue
        try:
            print(f"[rank {rank}] ({i+1}/{len(cfgs)}) {name}", flush=True)
            sim, cfg = make_sim(p)
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, "spec.yaml"), "w") as sf:
                yaml.safe_dump(cfg, sf, sort_keys=False)
            _, out = engine_run(sim, device=device)
            pos = out["sets"]["star"]["pos"]                     # [T, N, 3]
            render(pos, odir, name, view=2.4 * p["disc_R"])
            diag = diagnostics(pos[-1]); diag.update(p)
            json.dump(diag, open(os.path.join(odir, "diag.json"), "w"), indent=1)
        except Exception:
            print(f"[rank {rank}] {name} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"[rank {rank}] done", flush=True)


def montage():
    from PIL import Image, ImageDraw
    figs = sorted(glob.glob(os.path.join(OUT, "*", "fig_final.png")))
    rows = []
    for f in figs:
        d = os.path.dirname(f); name = os.path.basename(d)
        try:
            diag = json.load(open(os.path.join(d, "diag.json")))
        except Exception:
            diag = {}
        rows.append((name, diag, f))
    rows.sort(key=lambda r: -(r[1].get("A2", 0) * min(r[1].get("retain", 0) / 0.8, 1.0)))
    thumbs = []
    for name, diag, f in rows:
        im = Image.open(f).convert("RGB").resize((240, 240)); dd = ImageDraw.Draw(im)
        dd.text((4, 4), name, fill=(255, 255, 255))
        dd.text((4, 224), f"A2={diag.get('A2','?')} z={diag.get('z_rms','?')}", fill=(150, 200, 255))
        thumbs.append(im)
    if thumbs:
        cols = 4; R = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 240, R * 240), (0, 0, 0))
        for k, t in enumerate(thumbs):
            sheet.paste(t, ((k % cols) * 240, (k // cols) * 240))
        sheet.save(os.path.join(OUT, "_montage.png"))
    with open(os.path.join(OUT, "_summary.md"), "w") as fh:
        fh.write("# galaxy 3D — ranked by spiral score (A2 · retain)\n\n")
        fh.write("| rank | config | A2 | retain | r_half | z_rms (disk thickness) |\n|--|--|--|--|--|--|\n")
        for i, (name, diag, _) in enumerate(rows):
            fh.write(f"| {i+1} | {name} | {diag.get('A2','?')} | {diag.get('retain','?')} "
                     f"| {diag.get('r_half','?')} | {diag.get('z_rms','?')} |\n")
    print(f"montage: {len(thumbs)} -> {OUT}/_montage.png + _summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
