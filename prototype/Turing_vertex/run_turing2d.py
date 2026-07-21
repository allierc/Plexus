#!/usr/bin/env python
"""run_turing2d -- Stage 1 of the Turing x vertex prototype: the SIGNALLING half on a
STATIC 2D cell disc. Reaction--diffusion on the cell--cell adjacency graph (plexus2
operators in turing_ops.py), cells coloured by activator concentration (paper's red).

Each preset -- good OR bad -- is archived to archive/<name>/ with its spec.yaml, a
strip.png (snapshots), a movie.mp4, and diag.json. Presets sweep the Gray-Scott
feed/kill regimes (spots / coral / labyrinth / holes), the spatial-scale knob chi,
the Gierer-Meinhardt implementation, and an over-diffused negative control.

    python run_turing2d.py            # run every preset, archive each
    python run_turing2d.py --montage  # stitch the strips + a summary table
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

import plexus.operators   # noqa: F401
import turing_ops         # noqa: F401  aggregate_seed + graph_diffuse + react
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
N, RADIUS, K = 4000, 19.0, 6
FRAMES, DT = 6000, 1.0


def presets():
    G = dict(d_a=0.08, d_h=0.16, chi=0.4)                       # Gray-Scott diffusion (substrate faster)
    return [
        # name          reaction params                                    note
        ("spots",     {**G, "impl": "gray_scott", "F": 0.037, "kk": 0.060}),   # classic Turing spots  (GOOD)
        ("coral",     {**G, "impl": "gray_scott", "F": 0.055, "kk": 0.062}),   # coral / branching worms
        ("labyrinth", {**G, "impl": "gray_scott", "F": 0.030, "kk": 0.057}),   # stripes / labyrinth
        ("holes",     {**G, "impl": "gray_scott", "F": 0.039, "kk": 0.058}),   # inverse spots (holes)
        ("chi_small", {**G, "chi": 0.2, "impl": "gray_scott", "F": 0.037, "kk": 0.060}),  # smaller scale -> more spots
        ("gierer",    {"d_a": 0.02, "d_h": 0.4, "chi": 0.6, "impl": "gierer_meinhardt", "rho": 0.06}),  # activator-inhibitor alt
        ("washed_bad", {"d_a": 0.08, "d_h": 0.16, "chi": 1.6, "impl": "gray_scott", "F": 0.037, "kk": 0.060}),  # over-diffused (BAD control)
    ]


def make_spec(name, p, frames=FRAMES):
    W = 2.6 * RADIUS
    react = {"op": "react", "at": "cell", "implementation": p["impl"]}
    react.update({k: p[k] for k in ("F", "kk", "rho", "kappa", "rho0") if k in p})
    cfg = {
        "general": {"name": f"turing2d_{name}", "seed": 0, "n_frames": frames, "dt": DT,
                    "boundary": "free", "dim": 2, "world": [W, W]},
        "sets": {"cell": {"n": N, "state": {
            "chem": {"width": 2, "integration": "first_order", "boundary": "free"},
            "xyz":  {"width": 2, "integration": "none", "boundary": "free"}}}},
        "fields": {},
        "operators": [
            {"op": "aggregate_seed", "at": "cell", "mode": "disc", "radius": RADIUS, "k": K,
             "seed_frac": 0.12, "before_frame": 1},
            {"op": "graph_diffuse", "at": "cell", "d_a": p["d_a"], "d_h": p["d_h"], "chi": p["chi"]},
            react,
        ],
        "schedule": ["aggregate_seed", "graph_diffuse", "react"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def render(xyz, act, outdir, name, diag, seconds=12.0, max_frames=240):
    """xyz [N,2] static cell centres; act [T,N] activator. Scatter, coloured Reds on black."""
    os.makedirs(outdir, exist_ok=True)
    T = act.shape[0]
    vmax = max(0.05, float(np.percentile(act, 99.5)))
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    fps = max(1, round(len(idx) / seconds))
    x, y = xyz[:, 0], xyz[:, 1]
    box = max(float(x.max() - x.min()), float(y.max() - y.min())) + 1.0
    cx, cy = float((x.max() + x.min()) / 2), float((y.max() + y.min()) / 2)
    lim = (cx - box / 2, cx + box / 2, cy - box / 2, cy + box / 2)
    spacing = box / (len(x) ** 0.5)              # ~ nearest-neighbour cell spacing
    # size markers to ~the cell spacing (just touching, tissue-like) PER figure width, so
    # the strip panels (2.6in) and the movie (4.6in) render at the SAME dot density.
    DOT_SCALE = 0.9                              # 1.0 = touching; <1 leaves a hairline gap
    dot = lambda fig_w: (fig_w * 72.0 / box * spacing * DOT_SCALE) ** 2
    s = dot(2.6); s_movie = dot(4.6)

    def draw(ax, t, ss=s):
        ax.clear(); ax.set_facecolor("black")
        ax.scatter(x, y, c=act[t], cmap="Reds", vmin=0, vmax=vmax, s=ss, edgecolors="none")
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal"); ax.axis("off")
        pct = int(100 * t / max(T - 1, 1))
        ax.text(0.02, 0.98, f"{name}\nt={t} ({pct}%)\nactivator (red)\nhi cells={diag['hi_cells']}",
                transform=ax.transAxes, color="white", fontsize=7, va="top", family="monospace")

    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.25, 0.5, 1.0)]
    sfig, sax = plt.subplots(1, len(picks), figsize=(len(picks) * 2.6, 2.7)); sfig.patch.set_facecolor("black")
    for a, t in zip(sax, picks):
        draw(a, t)                                   # no panel titles (black-bg, top-left labels only)
    sfig.subplots_adjust(0.01, 0.01, 0.99, 0.99, wspace=0.04)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)

    fig, ax = plt.subplots(figsize=(4.6, 4.6)); fig.patch.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=120):
        for t in idx:
            draw(ax, t, ss=s_movie); w.grab_frame()      # large dots in the mp4
    plt.close(fig)


def diagnostics(act):
    vT = act[-1]
    return dict(v_max=round(float(vT.max()), 3), v_std=round(float(vT.std()), 3),
                hi_cells=int((vT > 0.2).sum()),
                patterned=bool(vT.std() > 0.04 and vT.max() > 0.2))


def run_all():
    for name, p in presets():
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        print(f"[turing2d] {name}: {p}", flush=True)
        rec = {"preset": p}
        try:
            sim, cfg = make_spec(name, p)
            yaml.safe_dump(cfg, open(os.path.join(odir, "spec.yaml"), "w"), sort_keys=False)
            _, out = engine_run(sim, device="cpu")
            act = out["sets"]["cell"]["state"]["chem"][..., 0]
            xyz = out["sets"]["cell"]["state"]["xyz"][0]
            diag = diagnostics(act); rec.update(diag)
            render(xyz, act, odir, name, diag)
            print(f"           -> patterned={diag['patterned']} v_std={diag['v_std']} hi={diag['hi_cells']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def montage():
    from PIL import Image
    strips = [(os.path.basename(os.path.dirname(f)), Image.open(f).convert("RGB"))
              for f in sorted(glob.glob(os.path.join(OUT, "*", "strip.png")))]
    if strips:
        w = max(i.width for _, i in strips); h = sum(i.height for _, i in strips)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, im in strips:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(os.path.join(OUT, "_montage_turing2d.png"))
    with open(os.path.join(OUT, "_summary_turing2d.md"), "w") as fh:
        fh.write("# Stage 1 -- Turing RD on a static 2D cell disc\n\n")
        fh.write("| preset | patterned | v_std | hi cells | note |\n|--|--|--|--|--|\n")
        for f in sorted(glob.glob(os.path.join(OUT, "*", "diag.json"))):
            d = json.load(open(f)); name = os.path.basename(os.path.dirname(f))
            fh.write(f"| {name} | {d.get('patterned', d.get('error','?'))} | {d.get('v_std','?')} "
                     f"| {d.get('hi_cells','?')} | {d.get('preset',{}).get('impl','?')} |\n")
    print(f"[turing2d] montage + summary -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else (run_all(), montage())


if __name__ == "__main__":
    main()
