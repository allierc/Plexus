#!/usr/bin/env python
"""run_embryo_french_flag -- positional information (French Flag) as a strict-Plexus sim.

A morphogen `grid` field is driven from a boundary source and spread by diffuse+decay into a
standing gradient; cells read the concentration at their position and pick a fate (blue/white/red)
by thresholds -> three domains (the French flag). Renders cells coloured by fate over time (black
background). Sweeps the gradient steepness + threshold placement; includes a no-source control.

    python run_embryo_french_flag.py --device cuda:1
    python run_embryo_french_flag.py --montage
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

import plexus.operators           # noqa: F401  diffuse + decay
import embryo_french_flag_ops      # noqa: F401  morphogen_source + french_flag
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
# fate colours: 0 = far/low (red), 1 = mid (white), 2 = near-source/high (blue) -> the flag
FATE_COLORS = np.array([[0.85, 0.12, 0.15], [0.92, 0.92, 0.92], [0.12, 0.22, 0.80]])


def presets():
    """Levers: decay rate (gradient length λ=√(D/decay): steep -> stripes bunch near the source;
    shallow -> stripes spread), threshold placement (moves the domain boundaries), and a
    no-source control (no gradient -> a single uniform fate, no flag)."""
    P = [
        ("standard", dict(decay=0.0005, t1=0.12, t2=0.44, source=1.0)),  # even thirds
        ("steep",    dict(decay=0.0012, t1=0.12, t2=0.44, source=1.0)),  # short gradient -> big red
        ("shallow",  dict(decay=0.0003, t1=0.12, t2=0.44, source=1.0)),  # long gradient -> big blue
        ("shifted",  dict(decay=0.0005, t1=0.25, t2=0.62, source=1.0)),  # thresholds moved
        ("control",  dict(decay=0.0005, t1=0.12, t2=0.44, source=0.0)),  # no source -> no flag
    ]
    return [(n, dict(N=3000, frames=1000, res=32, diffuse=0.8, **d)) for n, d in P]


def make_sim(p):
    cfg = {
        "general": {"name": "ff", "seed": 1, "n_frames": p["frames"], "dt": 1.0,
                    "boundary": "free", "world": [1.0, 1.0]},
        "sets": {"cell": {"n": p["N"], "spawn": "random",
                          "types": {"red": {"fraction": 0.34}, "white": {"fraction": 0.33},
                                    "blue": {"fraction": 0.33}}}},
        "fields": {"morphogen": {"frame": "grid", "res": p["res"], "components": 1}},
        "operators": [
            {"op": "morphogen_source", "at": "morphogen", "width": 0.04, "value": p["source"]},
            {"op": "diffuse", "at": "morphogen", "rate": p["diffuse"]},
            {"op": "decay", "at": "morphogen", "rate": p["decay"]},
            {"op": "french_flag", "at": "cell", "from": "morphogen", "t1": p["t1"], "t2": p["t2"]},
        ],
        "schedule": ["morphogen_source", "diffuse", "decay", "french_flag"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _fate_at(nt, t):
    nt = np.asarray(nt)
    return nt[t] if nt.ndim == 2 else nt


def render(pos, node_type, outdir, name, seconds=12.0, max_frames=300):
    os.makedirs(outdir, exist_ok=True)
    xy = pos[0] if pos.ndim == 3 else pos
    x, y = xy[:, 0], xy[:, 1]
    nt = np.asarray(node_type)
    T = nt.shape[0] if nt.ndim == 2 else 1
    stride = max(1, -(-T // max_frames)); idx = list(range(0, T, stride))
    fps = max(1, round(len(idx) / seconds))

    def col(t):
        return FATE_COLORS[_fate_at(node_type, t)]

    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.1, 0.3, 0.6, 1.0)]
    fig, ax = plt.subplots(1, len(picks), figsize=(len(picks) * 2.3, 2.4)); fig.patch.set_facecolor("black")
    for a, t in zip(ax, picks):
        a.scatter(x, y, s=5, c=col(t), linewidths=0); a.set_facecolor("black")
        a.set_title(f"{int(100*t/max(T-1,1))}%", color="white", fontsize=9)
        a.set_aspect("equal"); a.axis("off")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.05)
    fig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4.2)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1); ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
    sc = ax.scatter(x, y, s=7, c=col(0), linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=100):
        for t in idx:
            sc.set_color(col(t)); w.grab_frame()
    plt.close(fig)


def diagnostics(pos, node_type):
    xy = pos[0] if pos.ndim == 3 else pos
    fate = _fate_at(node_type, -1 if np.asarray(node_type).ndim == 2 else 0)
    fr = {f"frac_{k}": round(float((fate == k).mean()), 3) for k in (0, 1, 2)}
    # ordered flag: high-morphogen fate (2) at low x, low fate (0) at high x
    mx = {k: (float(xy[fate == k, 0].mean()) if (fate == k).any() else float("nan")) for k in (0, 1, 2)}
    ordered = bool(mx[2] < mx[1] < mx[0]) if not any(np.isnan(v) for v in mx.values()) else False
    fr.update(dict(x_blue=round(mx[2], 3), x_red=round(mx[0], 3), ordered_flag=ordered))
    return fr


def run_share(rank, nproc, device):
    P = presets()[rank::nproc]
    print(f"[rank {rank}] {len(P)} presets on {device}", flush=True)
    for i, (name, p) in enumerate(P):
        odir = os.path.join(OUT, name)
        if os.path.exists(os.path.join(odir, "movie.mp4")):
            print(f"[rank {rank}] skip {name}", flush=True); continue
        try:
            print(f"[rank {rank}] ({i+1}/{len(P)}) {name}", flush=True)
            sim, cfg = make_sim(p)
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, "spec.yaml"), "w") as sf:
                yaml.safe_dump(cfg, sf, sort_keys=False)
            fate_hist = []
            def cap(H, tick, _h=fate_hist):
                lvl = H.level("cell")
                if hasattr(lvl, "node_type"):
                    _h.append(lvl.node_type.cpu().numpy().copy())
            _, out = engine_run(sim, device=device, on_frame=cap)
            pos = out["sets"]["cell"]["pos"]
            nt = np.stack(fate_hist) if fate_hist else out["sets"]["cell"].get("node_type")
            render(pos, nt, odir, name)
            diag = diagnostics(pos, nt); diag.update({k: p[k] for k in ("decay", "t1", "t2", "source")})
            json.dump(diag, open(os.path.join(odir, "diag.json"), "w"), indent=1)
        except Exception:
            print(f"[rank {rank}] {name} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"[rank {rank}] done", flush=True)


def montage():
    from PIL import Image
    strips = sorted(glob.glob(os.path.join(OUT, "*", "strip.png")))
    ims = []
    for fpath in strips:
        name = os.path.basename(os.path.dirname(fpath))
        im = Image.open(fpath).convert("RGB"); im = im.resize((580, int(580 * im.height / im.width)))
        ims.append((name, im))
    if ims:
        w = max(i.width for _, i in ims); h = sum(i.height for _, i in ims)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, i in ims:
            sheet.paste(i, (0, y)); y += i.height
        sheet.save(os.path.join(OUT, "_montage.png"))
    with open(os.path.join(OUT, "_summary.md"), "w") as fh:
        fh.write("# embryo_french_flag — positional information (Wolpert)\n\n")
        fh.write("| preset | decay | t1 | t2 | frac(red/white/blue) | ordered flag |\n|--|--|--|--|--|--|\n")
        for fpath in sorted(glob.glob(os.path.join(OUT, "*", "diag.json"))):
            d = json.load(open(fpath)); name = os.path.basename(os.path.dirname(fpath))
            fr = f"{d.get('frac_0','?')}/{d.get('frac_1','?')}/{d.get('frac_2','?')}"
            fh.write(f"| {name} | {d.get('decay','?')} | {d.get('t1','?')} | {d.get('t2','?')} "
                     f"| {fr} | {d.get('ordered_flag','?')} |\n")
    print(f"montage: {len(ims)} presets -> {OUT}/_montage.png + _summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
