"""Reproduce input_circle.mp4 (10 balls bouncing chaotically in a circular arena).

Framework-respecting: a plain Plexus MPM spec -- 10 small ELASTIC balls (one cell
each, coloured per cell) under gravity, confined by a CIRCULAR CONTAINER built as a
ring of disc-obstacles (general.obstacles), so balls bounce off the inner circle.
Render (with motion trails) + loss live in this prototype layer.

  python reproduce2.py --iter 1 --device cuda:1 --g 7 --frames 500 --youngs 400 \
        --ballr 0.028 --Rc 0.44 --ndisc 90 --rd 0.05
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
GT = os.path.join(ROOT, "input_circle.mp4")
CX, CY = 0.5, 0.5


def ring_obstacles(Rc, ndisc, rd):
    """A circular wall as a ring of overlapping disc-obstacles (balls bounce inside)."""
    return [[CX + Rc * math.cos(2 * math.pi * k / ndisc),
             CY + Rc * math.sin(2 * math.pi * k / ndisc), rd] for k in range(ndisc)]


def build_spec(a):
    rng = np.random.default_rng(a.seed)
    # 10 ball centres: random inside the upper part of the arena
    pts = []
    while len(pts) < 10:
        x = CX + (rng.random() * 2 - 1) * (a.Rc - a.ballr - 0.02)
        y = CY + (rng.random() * 2 - 1) * (a.Rc - a.ballr - 0.02)
        if (x - CX) ** 2 + (y - CY) ** 2 < (a.Rc - a.ballr - 0.03) ** 2 and y > CY - 0.1:
            if all((x - px) ** 2 + (y - py) ** 2 > (2.2 * a.ballr) ** 2 for px, py in pts):
                pts.append((x, y))
    return {
        "general": {"name": f"circle_iter{a.iter:02d}", "seed": a.seed, "n_frames": a.frames,
                    "dt": 1.0, "boundary": "wall", "obstacles": ring_obstacles(a.Rc, a.ndisc, a.rd)},
        "sets": {
            "cell": {"n": 10, "start": [list(p) for p in pts],
                     "types": {"ball": {"fraction": 1.0, "youngs": a.youngs,
                                        "layers": [{"frac": 1.0, "youngs": a.youngs, "material": "elastic"}]}}},
            "mpm_particle": {"parent": "cell", "per_parent": a.ppc, "radius": a.ballr, "density": 1.0}},
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 160}},
        "operators": [
            {"op": "aggregate", "at": "cell"},
            {"op": "gravity", "at": "cell", "g": a.g},
            {"op": "mpm_strain", "at": "mpm_particle", "dt_sub": 1.5e-4},
            {"op": "p2g", "at": "mpm_particle", "to": "mpm_grid", "dt_sub": 1.5e-4, "drag": a.drag, "a_max": 300},
            {"op": "mpm_grid_update", "at": "mpm_grid", "dt_sub": 1.5e-4, "surface_tension": 0.0,
             "wall_damp": 0.95, "wall_contact": 0.02},
            {"op": "g2p", "at": "mpm_particle", "from": "mpm_grid", "dt_sub": 1.5e-4,
             "wall_damp": 0.95, "wall_contact": 0.02, "vmax": 1.0e9}],
        "schedule": ["aggregate", "gravity",
                     {"substep": 20, "dt": 1.5e-4, "steps": ["mpm_strain", "p2g", "mpm_grid_update", "g2p"]}],
        "plotting": {"background": "black"},
    }


def render(out, Rc, res=480, trail=0.82):
    p = out["sets"]["mpm_particle"]["pos"]; occ = out["sets"]["mpm_particle"]["occ"]
    par = out["sets"]["mpm_particle"]["parent"]
    cmap = plt.get_cmap("tab10")
    col = np.array([cmap(int(c) % 10)[:3] for c in par], np.float32)
    T = p.shape[0]
    fig = plt.figure(figsize=(5, 5), dpi=res / 5); ax = fig.add_axes([0, 0, 1, 1]); canvas = FigureCanvasAgg(fig)
    acc = None; frames = []
    for i in range(T):                                   # frames only (compare.mp4 is enough)
        ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_facecolor("black"); ax.axis("off")
        th = np.linspace(0, 2 * math.pi, 200); ax.plot(CX + Rc * np.cos(th), CY + Rc * np.sin(th), color="0.4", lw=1)
        m = occ[i]
        ax.scatter(p[i][m][:, 0], p[i][m][:, 1], s=5, c=col[m], linewidths=0)
        canvas.draw(); fr = np.asarray(canvas.buffer_rgba())[..., :3].astype(np.float32)
        acc = fr if acc is None else np.maximum(fr, acc * trail)        # motion trail
        frames.append(acc.astype(np.uint8))
    return np.array(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--g", type=float, default=7); ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--youngs", type=float, default=400); ap.add_argument("--ballr", type=float, default=0.028)
    ap.add_argument("--Rc", type=float, default=0.44); ap.add_argument("--ndisc", type=int, default=90)
    ap.add_argument("--rd", type=float, default=0.05); ap.add_argument("--drag", type=float, default=0.02)
    ap.add_argument("--ppc", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    spec_d = build_spec(a)
    outdir = os.path.join(ROOT, "archive_circle", f"iter_{a.iter:02d}"); os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, "spec.yaml"), "w").write(f"# iter {a.iter}: {a.note}\n" + yaml.safe_dump(spec_d, sort_keys=False, width=120))
    tmp = os.path.join(outdir, "_run.yaml"); open(tmp, "w").write(yaml.safe_dump(spec_d, sort_keys=False))
    sim = load_spec(tmp)
    H, out = E.run(sim, device=a.device)
    gen = render(out, a.Rc)
    # compare.mp4: cropped GT arena (left of frame) | gen
    import cv2
    gt_all = [f for f in imageio.get_reader(GT)]; gh, gw, _ = gt_all[0].shape
    x0, x1, y0, y1 = int(0.045 * gw), int(0.52 * gw), int(0.10 * gh), int(0.96 * gh)   # crop the arena
    with imageio.get_writer(os.path.join(outdir, "compare.mp4"), fps=30, codec="libx264", quality=8, macro_block_size=None) as wr:
        N = min(len(gen), 150); gi = np.linspace(0, len(gt_all) - 1, N).astype(int); fi = np.linspace(0, len(gen) - 1, N).astype(int)
        for ga, gb in zip(gi, fi):
            g = cv2.resize(gt_all[ga][y0:y1, x0:x1], (480, 480)); b = cv2.resize(gen[gb], (480, 480))
            wr.append_data(np.concatenate([g, b], axis=1))
    os.remove(tmp)
    print(f"ITER {a.iter} (circle) -> {outdir}  ({len(gen)} frames)")


if __name__ == "__main__":
    main()
