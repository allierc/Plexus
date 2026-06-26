"""Reproduce input_fig.mp4 with a Plexus spec (inverse problem).

Framework-respecting: the SIMULATION is a plain Plexus spec run through the stock
engine (sets / types / operators / schedule, MLS-MPM via the decomposed operators).
Only the RENDER + LOSS live here in the prototype layer. We colour particles by their
MATERIAL mask (is_liquid -> blue, is_snow -> white, else elastic -> red), matching the
target's 3-material scheme, then score against the GT frames.

  python reproduce.py --iter 1 --g 9 --frames 500 --rad 0.065 --ppc 900 \
        --you_jelly 90 --you_snow 120 --st 8
"""
import os, sys, math, argparse, json
sys.path.insert(0, "/groups/saalfeld/home/allierc/Graph/Plexus/src")
import numpy as np, yaml, torch
import plexus.operators  # noqa
from plexus.schema import load as load_spec
import plexus.engine as E
import imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = "/groups/saalfeld/home/allierc/Graph/Plexus/prototype/Unknown sequence"
GT = os.path.join(ROOT, "input_fig.mp4")
BLUE = (0.16, 0.45, 0.90); RED = (0.85, 0.12, 0.12); WHITE = (0.88, 0.88, 0.90)

# target 3x3 grid (from analysis); 3 materials x 3 cells each (engine assigns by fraction)
GRID = [(x, y) for y in (0.87, 0.62, 0.38) for x in (0.21, 0.45, 0.69)]


def build_spec(a):
    return {
        "general": {"name": f"unknown_iter{a.iter:02d}", "seed": a.seed, "n_frames": a.frames,
                    "dt": 1.0, "boundary": "wall"},
        "sets": {
            "cell": {"n": 9, "start": [list(p) for p in GRID], "types": {
                "water": {"fraction": 0.3334, "youngs": 200, "layers": [{"frac": 1.0, "youngs": 200, "material": "liquid"}]},
                "jelly": {"fraction": 0.3333, "youngs": a.you_jelly, "layers": [{"frac": 1.0, "youngs": a.you_jelly, "material": "elastic"}]},
                "snow":  {"fraction": 0.3333, "youngs": a.you_snow, "layers": [{"frac": 1.0, "youngs": a.you_snow, "material": "snow"}]}}},
            "mpm_particle": {"parent": "cell", "per_parent": a.ppc, "radius": a.rad, "density": 1.0}},
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 128}},
        "operators": [
            {"op": "aggregate", "at": "cell"},
            {"op": "gravity", "at": "cell", "g": a.g},
            {"op": "mpm_strain", "at": "mpm_particle", "dt_sub": 2.0e-4},
            {"op": "p2g", "at": "mpm_particle", "to": "mpm_grid", "dt_sub": 2.0e-4, "drag": a.drag, "a_max": 200},
            {"op": "mpm_grid_update", "at": "mpm_grid", "dt_sub": 2.0e-4, "surface_tension": a.st,
             "wall_damp": 0.5, "wall_contact": 0.06},
            {"op": "g2p", "at": "mpm_particle", "from": "mpm_grid", "dt_sub": 2.0e-4,
             "wall_damp": 0.5, "wall_contact": 0.06, "vmax": 1.0e9}],
        "schedule": ["aggregate", "gravity",
                     {"substep": 18, "dt": 2.0e-4, "steps": ["mpm_strain", "p2g", "mpm_grid_update", "g2p"]}],
        "plotting": {"background": "black"},
    }


def render(out, H, path, res=480):
    p = out["sets"]["mpm_particle"]["pos"]; occ = out["sets"]["mpm_particle"]["occ"]
    liq = getattr(H.level("mpm_particle"), "is_liquid", None)
    snow = getattr(H.level("mpm_particle"), "is_snow", None)
    liq = liq.cpu().numpy().astype(bool) if liq is not None else np.zeros(p.shape[1], bool)
    snow = snow.cpu().numpy().astype(bool) if snow is not None else np.zeros(p.shape[1], bool)
    elas = ~liq & ~snow
    col = np.zeros((p.shape[1], 3), np.float32)
    col[liq] = BLUE; col[snow] = WHITE; col[elas] = RED
    T = p.shape[0]
    fig = plt.figure(figsize=(5, 5), dpi=res / 5); ax = fig.add_axes([0, 0, 1, 1]); canvas = FigureCanvasAgg(fig)
    outframes = []
    for i in range(T):                                   # frames only (no gen.mp4 -- compare is enough)
        ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_facecolor("black"); ax.axis("off")
        m = occ[i]
        ax.scatter(p[i][m][:, 0], p[i][m][:, 1], s=7, c=col[m], linewidths=0)
        canvas.draw(); outframes.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    return np.array(outframes)


def color_masks(f):
    f = f.astype(float); r, g, b = f[..., 0], f[..., 1], f[..., 2]
    red = (r > 110) & (g < 90) & (b < 90)
    blue = (b > 100) & (r < 100)
    white = (r > 120) & (g > 120) & (b > 120) & ~red & ~blue
    return red, blue, white


def loss_vs_gt(gen, n_gt_sample=12, res=240):
    gt_all = [f for f in imageio.get_reader(GT)]
    gi = np.linspace(0, len(gt_all) - 1, n_gt_sample).astype(int)
    fi = np.linspace(0, len(gen) - 1, n_gt_sample).astype(int)
    import cv2
    tot = 0.0
    for a_, b_ in zip(gi, fi):
        ga = cv2.resize(gt_all[a_], (res, res)); gb = cv2.resize(gen[b_], (res, res))
        for mg, mb in zip(color_masks(ga), color_masks(gb)):
            inter = (mg & mb).sum(); union = (mg | mb).sum() + 1e-6
            tot += 1 - inter / union           # 1 - IoU per colour
    return tot / (n_gt_sample * 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--g", type=float, default=9); ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--rad", type=float, default=0.065); ap.add_argument("--ppc", type=int, default=900)
    ap.add_argument("--you_jelly", type=float, default=90); ap.add_argument("--you_snow", type=float, default=120)
    ap.add_argument("--st", type=float, default=8); ap.add_argument("--drag", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--note", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    spec_d = build_spec(a)
    outdir = os.path.join(ROOT, "archive", f"iter_{a.iter:02d}"); os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, "spec.yaml"), "w").write(f"# iter {a.iter}: {a.note}\n" + yaml.safe_dump(spec_d, sort_keys=False, width=120))
    tmp = os.path.join(outdir, "_run.yaml"); open(tmp, "w").write(yaml.safe_dump(spec_d, sort_keys=False))
    sim = load_spec(tmp)
    H, out = E.run(sim, device=a.device)
    gen = render(out, H, None)
    L = loss_vs_gt(gen)
    # side-by-side compare (gt | gen)
    gt_all = [f for f in imageio.get_reader(GT)]
    import cv2
    with imageio.get_writer(os.path.join(outdir, "compare.mp4"), fps=30, codec="libx264", quality=8, macro_block_size=None) as wr:
        N = min(len(gen), 150); gi = np.linspace(0, len(gt_all) - 1, N).astype(int); fi = np.linspace(0, len(gen) - 1, N).astype(int)
        for a_, b_ in zip(gi, fi):
            g = cv2.resize(gt_all[a_], (480, 480)); b = cv2.resize(gen[b_], (480, 480))
            wr.append_data(np.concatenate([g, b], axis=1))
    open(os.path.join(outdir, "loss.txt"), "w").write(f"loss={L:.4f}\n{json.dumps(vars(a))}\n")
    os.remove(tmp)
    print(f"ITER {a.iter}  loss={L:.4f}  -> {outdir}")


if __name__ == "__main__":
    main()
