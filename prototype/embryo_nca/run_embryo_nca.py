#!/usr/bin/env python
"""run_embryo2 -- grow / regenerate the Growing-NCA organism as a strict-Plexus field sim.

Builds a set of Plexus specs (a 16-channel `grid` field + the `growing_nca` field operator
loading the paper's pretrained weights), runs each through the Plexus engine, and renders
the RGBA organism (black background) to an mp4 + a growth-stage strip in archive/<name>/,
with a montage + a short "understanding" table. Splits across GPUs like galaxy_sweep:

    python run_embryo2.py --rank 0 --nproc 2 --device cuda:0 &
    python run_embryo2.py --rank 1 --nproc 2 --device cuda:1 &
    wait; python run_embryo2.py --montage
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

import plexus.operators           # noqa: F401  stock library
import embryo_nca_ops                 # noqa: F401  growing_nca + nca_seed + nca_damage
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
RES = 72                                           # 72x72 grid == the paper's canvas


def variants():
    """Each variant isolates one lever of the morphogenetic rule:
    growth/persistence (frames), async-update rate (fire_rate), and regeneration
    (a wound at a given frame -> the local rule must re-grow the missing tissue)."""
    V = []
    def add(name, frames, fire=0.5, dmg=None):
        V.append((name, dict(frames=frames, fire=fire, dmg=dmg)))
    # growth + persistence
    add("grow", 260)
    add("grow_long", 700)                          # does the organism hold its shape? (stability)
    # async-update sensitivity (stochastic firing = asynchronous cells)
    add("fire25", 320, fire=0.25)
    add("fire75", 260, fire=0.75)
    # regeneration: disc wounds of increasing size, cut after the body has formed
    add("regen_small", 460, dmg=[dict(frame=230, radius=0.12)])
    add("regen_disc",  460, dmg=[dict(frame=230, radius=0.25)])
    add("regen_big",   520, dmg=[dict(frame=230, radius=0.38)])
    # half-body amputation (left / right)
    add("regen_half_L", 520, dmg=[dict(frame=250, side="left")])
    add("regen_half_R", 520, dmg=[dict(frame=250, side="right")])
    # repeated injury: heal, wound again
    add("regen_twice", 640, dmg=[dict(frame=220, radius=0.25),
                                 dict(frame=430, side="left")])
    return V


def make_sim(p):
    ops = [
        {"op": "nca_seed", "at": "nca", "before_frame": 1},
        {"op": "growing_nca", "at": "nca", "fire_rate": p["fire"]},
    ]
    sched = ["nca_seed", "growing_nca"]
    for i, d in enumerate(p.get("dmg") or []):
        line = {"op": "nca_damage", "at": "nca", "frame": d["frame"]}
        if "side" in d:
            line["side"] = d["side"]
        else:
            line["radius"] = d.get("radius", 0.25)
        ops.append(line)
        if "nca_damage" not in sched:
            sched.append("nca_damage")
    cfg = {
        "general": {"name": "embryo2", "seed": 0, "n_frames": p["frames"], "dt": 1.0,
                    "boundary": "free", "world": [1.0, 1.0]},
        "sets": {"seed_cell": {"n": 1, "types": {"a": {"fraction": 1.0}}}},  # dummy set (engine needs >=1)
        "fields": {"nca": {"frame": "grid", "res": RES, "components": 16}},
        "operators": ops,
        "schedule": sched,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f); path = f.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def to_rgb_black(grid_frame):
    """[16, nx, ny] -> [nx, ny, 3] RGBA premultiplied over a BLACK background = rgb (clamped)."""
    rgb = np.clip(np.transpose(grid_frame[:3], (1, 2, 0)), 0.0, 1.0)
    return rgb


def render(grids, outdir, name, seconds=18.0, max_frames=520):
    """grids: [T, 16, nx, ny]. Black-background RGBA movie (~`seconds`) + a growth strip."""
    os.makedirs(outdir, exist_ok=True)
    T = grids.shape[0]
    stride = max(1, -(-T // max_frames))
    idx = list(range(0, T, stride))
    fps = max(1, round(len(idx) / seconds))
    # growth strip (5 snapshots)
    picks = [int(round(f * (T - 1))) for f in (0.03, 0.15, 0.4, 0.7, 1.0)]
    fig, ax = plt.subplots(1, len(picks), figsize=(len(picks) * 2.1, 2.3))
    fig.patch.set_facecolor("black")
    for a, t in zip(ax, picks):
        a.imshow(to_rgb_black(grids[t])); a.set_facecolor("black")
        a.set_title(f"{int(100*t/(T-1))}%", color="white", fontsize=9); a.axis("off")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.05)
    fig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black")
    plt.close(fig)
    # movie
    fig, ax = plt.subplots(figsize=(4, 4)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1); ax.axis("off")
    im = ax.imshow(to_rgb_black(grids[0]))
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=100):
        for t in idx:
            im.set_data(to_rgb_black(grids[t])); w.grab_frame()
    plt.close(fig)


def diagnostics(grids):
    """Morphogenesis observables from the alpha channel (living tissue):
    final size, growth curve, and -- for regen runs -- the recovery after the wound."""
    alpha = grids[:, 3]                                            # [T, nx, ny]
    live = (alpha > 0.1).reshape(alpha.shape[0], -1).sum(1)        # living-cell count per frame
    T = len(live)
    return dict(final_live=int(live[-1]), max_live=int(live.max()),
                final_frac_of_max=round(float(live[-1] / max(live.max(), 1)), 3),
                live_10=int(live[T // 10]), live_50=int(live[T // 2]))


def run_share(rank, nproc, device):
    V = variants()[rank::nproc]
    print(f"[rank {rank}] {len(V)} variants on {device}", flush=True)
    for i, (name, p) in enumerate(V):
        odir = os.path.join(OUT, name)
        if os.path.exists(os.path.join(odir, "movie.mp4")):
            print(f"[rank {rank}] skip {name}", flush=True); continue
        try:
            print(f"[rank {rank}] ({i+1}/{len(V)}) {name}", flush=True)
            sim, cfg = make_sim(p)
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, "spec.yaml"), "w") as sf:
                yaml.safe_dump(cfg, sf, sort_keys=False)
            _, out = engine_run(sim, device=device)
            grids = out["fields"]["nca"]["grid"]                  # [T, 16, nx, ny]
            render(grids, odir, name)
            diag = diagnostics(grids); diag.update({k: p[k] for k in ("frames", "fire")})
            diag["dmg"] = p["dmg"]
            json.dump(diag, open(os.path.join(odir, "diag.json"), "w"), indent=1)
        except Exception:
            print(f"[rank {rank}] {name} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"[rank {rank}] done", flush=True)


def montage():
    from PIL import Image
    strips = sorted(glob.glob(os.path.join(OUT, "*", "strip.png")))
    ims = []
    for f in strips:
        name = os.path.basename(os.path.dirname(f))
        im = Image.open(f).convert("RGB")
        im = im.resize((520, int(520 * im.height / im.width)))
        ims.append((name, im))
    if ims:
        w = max(i.width for _, i in ims); h = sum(i.height for _, i in ims)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, i in ims:
            sheet.paste(i, (0, y)); y += i.height
        sheet.save(os.path.join(OUT, "_montage.png"))
    with open(os.path.join(OUT, "_summary.md"), "w") as fh:
        fh.write("# embryo2 (Growing-NCA) — variants\n\n")
        fh.write("| variant | frames | fire | final_live | %of_max | note |\n|--|--|--|--|--|--|\n")
        for f in sorted(glob.glob(os.path.join(OUT, "*", "diag.json"))):
            d = json.load(open(f)); name = os.path.basename(os.path.dirname(f))
            note = "regen" if d.get("dmg") else "growth"
            fh.write(f"| {name} | {d.get('frames','?')} | {d.get('fire','?')} | "
                     f"{d.get('final_live','?')} | {d.get('final_frac_of_max','?')} | {note} |\n")
    print(f"montage: {len(ims)} variants -> {OUT}/_montage.png + _summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
