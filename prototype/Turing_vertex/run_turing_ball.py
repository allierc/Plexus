#!/usr/bin/env python
"""run_turing_ball -- 3D Turing tests on cell aggregates (Okuda et al. 2018 Fig. 3).

Discrete reaction--diffusion on the cell--cell adjacency graph of a 3D aggregate
(plexus2 operators in turing_ops.py), cells coloured by activator (red), STATIC
geometry (no deformation). Two reaction kinetics, selected per preset by
`implementation:`:
  * brusselator  -> round activator spots (paper Fig. 3), seeded from noise;
  * gray_scott   -> coral / holes / labyrinth regimes, seeded from scattered nuclei.
Two aggregate modes: `ball` (solid) and `shell` (monolayer).

Spot size scales as sqrt(chi/gamma) (Brusselator); bigger spots => raise chi and lower
gamma at a smaller dt (stability: d_h*chi*dt <= 1 on the normalized Laplacian).

A solid ball hides its interior, so each ball test renders an external view (rotating
mp4) AND an internal cut (a hemisphere through the centre). Each test is archived to
archive/<name>/ : spec.yaml, strip.png, movie.mp4, diag.json.

    python run_turing_ball.py            # run every preset
    python run_turing_ball.py --montage
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
import turing_ops         # noqa: F401
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
PANEL = 4.4


def BR(gamma, A=1.0, B=3.0):
    return {"model": "brusselator", "gamma": gamma, "A": A, "B": B}


def GS(F, kk):
    return {"model": "gray_scott", "F": F, "kk": kk}


def presets():
    # --- bigger round spots: Brusselator, chi/gamma up ~4x -> ~2x wavelength ---
    ps = [
        dict(name="ball_big", mode="ball", N=4000, R=11.0, k=12, chi=6.4, dt=0.2, frames=20000,
             norm=True, seed="noise", d_a=0.05, d_h=0.5, react=BR(0.02)),
        dict(name="shell_big_spots", mode="shell", N=4000, R=11.0, k=6, chi=6.4, dt=0.2, frames=20000,
             norm=True, seed="noise", d_a=0.05, d_h=0.5, react=BR(0.02)),
    ]
    # --- Gray-Scott regimes (coral / holes / labyrinth), on BOTH the ball and the shell ---
    # dt small (0.35) so the reaction is SLOW relative to the recorded frames -> the pattern
    # forms gradually across the movie instead of in the first few frames.
    regimes = [("coral", 0.058, 0.063), ("holes", 0.039, 0.058), ("labyrinth", 0.029, 0.054)]
    for mode, N in (("ball", 2400), ("shell", 1600)):
        for nm, F, kk in regimes:
            ps.append(dict(name=f"{mode}_{nm}", mode=mode, N=N, R=11.0, k=6, chi=0.28, dt=0.35,
                           frames=10000, norm=False, seed="scatter", d_a=0.08, d_h=0.16, react=GS(F, kk)))
    return ps


def make_spec(p):
    W = 2.6 * p["R"]
    seed = {"op": "seed_aggregate", "at": "cell", "mode": p["mode"], "seed_mode": p["seed"],
            "radius": p["R"], "k": p["k"], "before_frame": 1}
    if p["seed"] == "noise":
        A, B = p["react"]["A"], p["react"]["B"]
        seed.update(a0=A, h0=B / A, noise=0.03)
    else:
        seed.update(seed_frac=0.04, noise=0.02)
    cfg = {
        "general": {"name": f"turing_{p['name']}", "seed": 0, "n_frames": p["frames"], "dt": p["dt"],
                    "boundary": "free", "dim": 3, "world": [W, W, W]},
        "sets": {"cell": {"n": p["N"], "state": {
            "chem": {"width": 2, "integration": "first_order", "boundary": "free"},
            "xyz":  {"width": 3, "integration": "none", "boundary": "free"}}}},
        "fields": {},
        "operators": [
            seed,
            {"op": "graph_diffuse", "at": "cell", "d_a": p["d_a"], "d_h": p["d_h"],
             "chi": p["chi"], "norm": p["norm"]},
            {"op": "react", "at": "cell", **p["react"]},
        ],
        "schedule": ["seed_aggregate", "graph_diffuse", "react"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _scatter(ax, x, y, z, c, vmin, vmax, s, box):
    ax.clear(); ax.set_facecolor("black")
    ax.scatter(x, y, z, c=c, cmap="Reds", vmin=vmin, vmax=vmax, s=s, edgecolors="none", depthshade=True)
    mx, my, mz = float(np.mean(x)), float(np.mean(y)), float(np.mean(z))
    ax.set_xlim(mx - box / 2, mx + box / 2)
    ax.set_ylim(my - box / 2, my + box / 2)
    ax.set_zlim(mz - box / 2, mz + box / 2)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


def render(xyz, act, outdir, name, mode, seconds=12.0, max_frames=170):
    os.makedirs(outdir, exist_ok=True)
    T = act.shape[0]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cx = float(x.mean())
    vmin, vmax = float(np.percentile(act, 1)), float(np.percentile(act, 99))
    box = float(np.ptp(np.stack([x, y, z]), axis=1).max()) + 1.0
    spacing = box / (len(x) ** 0.5)
    s = (PANEL * 72.0 / box * spacing * 0.85) ** 2
    cut = x <= cx
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    fps = max(1, round(len(idx) / seconds))

    def label(ax, txt):
        ax.text2D(0.02, 0.98, txt, transform=ax.transAxes, color="white", fontsize=7, va="top", family="monospace")

    # strip: external 3D @ 0/50/100%, then a 4th external back view (shell kept closed).
    sfig = plt.figure(figsize=(4 * PANEL, PANEL)); sfig.patch.set_facecolor("black")
    for i, fr in enumerate((0.0, 0.5, 1.0)):
        t = int(fr * (T - 1))
        ax = sfig.add_subplot(1, 4, i + 1, projection="3d"); ax.set_facecolor("black")
        _scatter(ax, x, y, z, act[t], vmin, vmax, s, box); ax.view_init(18, 35)
        label(ax, f"{name}\nexternal {int(100*fr)}%\nactivator (red)")
    # 4th panel: an external back view (no interior cutaway -- keep the shell closed).
    ax = sfig.add_subplot(1, 4, 4, projection="3d"); ax.set_facecolor("black")
    _scatter(ax, x, y, z, act[-1], vmin, vmax, s, box); ax.view_init(18, 215)
    label(ax, f"{name}\nexternal back\nactivator (red)")
    sfig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.02)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)

    # movie: rotating external
    fig = plt.figure(figsize=(PANEL, PANEL)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(projection="3d"); ax.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=120):
        for k, t in enumerate(idx):
            _scatter(ax, x, y, z, act[t], vmin, vmax, s, box)
            ax.view_init(18, 35 + 360.0 * k / len(idx))
            label(ax, f"{name}\nt={t} ({int(100*t/(T-1))}%)\nactivator (red)")
            w.grab_frame()
    plt.close(fig)


def diagnostics(act):
    vT = act[-1]
    return dict(v_std=round(float(vT.std()), 3), v_max=round(float(vT.max()), 2),
                hi_frac=round(float((vT > vT.mean()).mean()), 3), nan=bool(np.isnan(vT).any()),
                patterned=bool(vT.std() > 0.05 and not np.isnan(vT).any()))


def run_all():
    for p in presets():
        odir = os.path.join(OUT, p["name"]); os.makedirs(odir, exist_ok=True)
        print(f"[3d] {p['name']}: {p['mode']} {p['react']['implementation']} chi={p['chi']} dt={p['dt']}", flush=True)
        rec = {k: p[k] for k in ("mode", "N", "chi", "dt", "frames", "seed")}; rec["react"] = p["react"]
        try:
            sim, cfg = make_spec(p)
            yaml.safe_dump(cfg, open(os.path.join(odir, "spec.yaml"), "w"), sort_keys=False)
            _, out = engine_run(sim, device="cpu")
            act = out["sets"]["cell"]["state"]["chem"][..., 0]
            xyz = out["sets"]["cell"]["state"]["xyz"][0]
            diag = diagnostics(act); rec.update(diag)
            # cache a decimated trajectory so render tweaks re-render WITHOUT re-simulating
            T = act.shape[0]; keep = np.linspace(0, T - 1, min(T, 200)).astype(int)
            np.savez_compressed(os.path.join(odir, "traj.npz"), xyz=xyz.astype("float32"),
                                act=act[keep].astype("float32"), mode=p["mode"], name=p["name"])
            render(xyz, act[keep], odir, p["name"], p["mode"])
            print(f"     -> patterned={diag['patterned']} v_std={diag['v_std']} hi_frac={diag['hi_frac']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def rerender(only=None):
    """Re-render from cached traj.npz -- no re-simulation. `only` = list of preset names."""
    for p in presets():
        if only and p["name"] not in only:
            continue
        odir = os.path.join(OUT, p["name"]); tf = os.path.join(odir, "traj.npz")
        if not os.path.exists(tf):
            print(f"[rerender] {p['name']}: no traj.npz (run first)"); continue
        d = np.load(tf)
        render(d["xyz"], d["act"], odir, str(d["name"]), str(d["mode"]))
        print(f"[rerender] {p['name']} -> strip.png + movie.mp4", flush=True)


def montage():
    from PIL import Image
    names = [p["name"] for p in presets()]
    strips = [(n, Image.open(os.path.join(OUT, n, "strip.png")).convert("RGB"))
              for n in names if os.path.exists(os.path.join(OUT, n, "strip.png"))]
    if strips:
        w = max(i.width for _, i in strips); h = sum(i.height for _, i in strips)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, im in strips:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(os.path.join(OUT, "_montage_ball.png"))
    print(f"[3d] montage -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--rerender", action="store_true")   # re-render from cached traj.npz (no re-sim)
    ap.add_argument("--only", nargs="*", default=None)    # subset of preset names
    a = ap.parse_args()
    if a.montage:
        montage()
    elif a.rerender:
        rerender(a.only); montage()
    else:
        run_all(); montage()


if __name__ == "__main__":
    main()
