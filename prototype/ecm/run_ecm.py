#!/usr/bin/env python
"""run_ecm -- run one ECM experiment and leave the movie, the strip and the numbers behind.

    python run_ecm.py 01_first_contact --frames 320 --device cuda:0

Everything lands in `log/okuda_ECM/<name>/`: `movie.mp4`, `strip.png`, `spec_run.yaml` and
`metrics.json`. The spec is written beside the result because a movie without the spec that made
it is an anecdote -- and the sweep varies stiffness, cavity shape and growth rate, so "which run
was this" is a question that gets asked of every frame.

WHAT THE NUMBERS ARE FOR. The movie shows the stress front; `metrics.json` says whether it was
real. Three things decide that, and all three are cheap:

  contact_frame   the first frame any matrix particle is inside the ball. Before it, nothing this
                  experiment is about has happened; a run whose contact frame is 0 was seeded
                  wrong, and a run that never reaches contact is a null however good it looks.
  strained_frac   the fraction of matrix carrying |J-1| above the colour floor. This is the stress
                  FRONT as a number: it should be ~0 before contact and grow after it. If it is
                  large at frame 0 the material is exploding, not responding.
  max_disp        the furthest any particle has moved from where it was seeded. Distinguishes a
                  matrix being pushed from a matrix falling apart.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

LOG = os.path.join(ROOT, "log", "okuda_ECM")


def measure(out, spec, seeded=None):
    """The three numbers, from the recorded trajectory."""
    s = out["sets"]["mpm_particle"]
    pos = np.asarray(s["pos"])                      # [T, N, 3]
    T = pos.shape[0]
    op = next(o for o in spec["operators"] if o["op"] == "cell_to_ecm")
    c = np.asarray(op["centre"], float)
    r0, growth, r_max = op["r0"], op["growth"], op["r_max"]

    d = np.linalg.norm(pos - c, axis=2)             # [T, N]
    radii = np.minimum(r_max, r0 + growth * np.arange(T))
    inside = d < radii[:, None]
    hits = np.where(inside.any(axis=1))[0]
    contact = int(hits[0]) if hits.size else None

    # `node_type` IS RECORDED ONCE, NOT PER FRAME -- it is a buffer, and the recorder saves the
    # final state of it. So the band array is [N], the stress AT THE END, and `strained_frac` is
    # one number rather than a series. That is a real limit of this diagnostic and it is stated
    # rather than papered over: the MOVIE carries the propagation over time, the metric carries
    # only where the front had reached when the run stopped.
    band = s.get("node_type")
    strained = None
    if band is not None:
        b = np.asarray(band)
        strained = float((b > 0).mean())

    start = pos[0] if seeded is None else seeded
    disp = np.linalg.norm(pos - start[None], axis=2)
    return {"frames": int(T), "n_particles": int(pos.shape[1]),
            "contact_frame": contact,
            "ball_r_final": float(radii[-1]),
            "strained_frac_end": strained,
            "max_disp": float(disp.max()), "med_disp_final": float(np.median(disp[-1])),
            "exploded": bool(np.isnan(pos).any() or float(np.abs(pos).max()) > 5.0)}


def render(name, out, spec, out_dir, n_strip=6, max_frames=150):
    """Two views per frame: the matrix in 3D, and a SLICE through the plane of the cavity.

    THE SLICE IS THE POINT. A 3D cloud of 48,000 dots hides its own interior -- the bright
    particles nearest the camera occlude the front travelling behind them, and a stress wave
    moving outward reads as "the middle got brighter". A thin slab through the cavity's mid-plane
    has no interior to hide: the ball is a disc in the centre, the matrix is the region around it,
    and the front is a ring you can watch move.

    Both views are coloured by the SAME per-frame stress band, so the 3D panel shows the geometry
    of the deformation and the slice shows its timing, and neither is asked to do both.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import ecm_ops
    import ecm_spec as ES

    pos = np.asarray(out["sets"]["mpm_particle"]["pos"])            # [T, N, 3]
    hist = ecm_ops.STRESS_HISTORY
    radii = ecm_ops.BALL_RADIUS
    T = min(pos.shape[0], len(hist)) if hist else pos.shape[0]
    if not hist:
        print(f"[{name}] no stress history -- rendering positions only", flush=True)
    cmap = ListedColormap(ES.STRESS_COLORS)
    ax_i = int(next(o for o in spec["operators"] if o["op"] == "ecm_seed")["axis"])
    plane = [i for i in range(3) if i != ax_i]                       # the two free axes
    keep = np.arange(T)
    if T > max_frames:
        keep = np.unique(np.linspace(0, T - 1, max_frames).astype(int))

    frames_dir = os.path.join(out_dir, "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    written = []
    for j, t in enumerate(keep):
        band = hist[t] if hist else np.zeros(pos.shape[1], np.uint8)
        p3 = pos[t]
        fig = plt.figure(figsize=(11, 5.4), facecolor="black")
        # --- 3D ---
        a1 = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black")
        srt = np.argsort(band)                       # stressed particles drawn last, on top
        a1.scatter(p3[srt, 0], p3[srt, 2], p3[srt, 1], c=band[srt], cmap=cmap, vmin=0, vmax=7,
                   s=1.1, marker=".", linewidths=0, alpha=0.9)
        a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.set_zlim(0, 1)
        a1.set_axis_off(); a1.view_init(elev=18, azim=-60)
        # --- slice through the cavity mid-plane ---
        a2 = fig.add_subplot(1, 2, 2, facecolor="black")
        # A THICKER SLAB, because a thin one is not a picture of a material. At 0.035 the slice
        # held 1,128 of 20,000 particles -- sparse enough that the matrix read as scattered dust
        # and a front moving through it had nothing to move through.
        sl = np.abs(p3[:, ax_i] - 0.5) < 0.06
        a2.scatter(p3[sl][:, plane[0]], p3[sl][:, plane[1]], c=band[sl], cmap=cmap,
                   vmin=0, vmax=7, s=4.5, marker=".", linewidths=0)
        r = radii[t] if t < len(radii) else 0.0
        a2.add_patch(plt.Circle((0.5, 0.5), r, fill=False, ec="#39d0ff", lw=1.1, alpha=0.9))
        a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.set_aspect("equal"); a2.set_axis_off()
        fig.text(0.02, 0.95, f"{name}   frame {t}   ball r={r:.3f}", color="white", fontsize=9)
        fig.text(0.52, 0.95, "slice through the cavity plane", color="#888", fontsize=8)
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        f = os.path.join(frames_dir, f"f{j:05d}.png")
        fig.savefig(f, dpi=110, facecolor="black"); plt.close(fig)
        written.append(f)

    # STRIP: a few frames side by side, so the propagation is one still image.
    idx = np.unique(np.linspace(0, len(written) - 1, n_strip).astype(int))
    import PIL.Image as I
    ims = [I.open(written[i]) for i in idx]
    w, h = ims[0].size
    strip = I.new("RGB", (w * len(ims), h), "black")
    for i, im in enumerate(ims):
        strip.paste(im, (i * w, 0))
    strip.save(os.path.join(out_dir, "strip.png"))

    mp4 = os.path.join(out_dir, "movie.mp4")
    fps = int(spec.get("plotting", {}).get("fps", 30))
    # NEXT TO THE INTERPRETER, NOT ON PATH. The conda env ships ffmpeg but the shell does not see
    # it, so a bare `ffmpeg` returns 32512 (command not found) -- and os.system reports that in a
    # return code nobody reads, leaving a run that looks complete with no movie in it. This is
    # plexus.plot._ffmpeg's own rule, reused rather than re-derived.
    exe = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    if not os.path.exists(exe):
        import shutil as _sh
        exe = _sh.which("ffmpeg") or exe
    rc = os.system(f"{exe} -y -loglevel error -framerate {fps} -i "
                   f"{frames_dir}/f%05d.png -c:v libx264 -pix_fmt yuv420p -crf 20 {mp4}")
    if rc != 0 or not os.path.exists(mp4):
        raise RuntimeError(f"ffmpeg failed (rc={rc}) using {exe}")
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"[{name}] wrote movie.mp4 ({len(written)} frames) + strip.png"
          + ("" if rc == 0 else "  [ffmpeg rc=%d]" % rc), flush=True)


def run(name, spec, device="cuda:0", movie=True):
    import plexus.operators                                    # noqa: F401  register the stock ops
    import ecm_ops
    # THE HISTORY IS PER RUN. A module-level list survives between runs in a sweep, so a second
    # run would render the first run's stress on its own particles -- silently, and looking
    # entirely plausible.
    ecm_ops.STRESS_HISTORY.clear()
    ecm_ops.BALL_RADIUS.clear()
    import plexus.schema as S
    from plexus.engine import run as engine_run

    out_dir = os.path.join(LOG, name)
    os.makedirs(out_dir, exist_ok=True)
    spec_path = os.path.join(out_dir, "spec_run.yaml")
    with open(spec_path, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)

    t0 = time.time()
    sim = S.load(spec_path)
    H, out = engine_run(sim, device=device)
    wall = time.time() - t0

    m = measure(out, spec)
    m["wall_s"] = round(wall, 1)
    m["name"] = name
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    print(f"[{name}] {wall:.0f}s  contact_frame={m['contact_frame']}  "
          f"strained_frac_end={m['strained_frac_end']}  "
          f"max_disp={m['max_disp']:.3f}  exploded={m['exploded']}", flush=True)

    if movie:
        try:
            render(name, out, spec, out_dir)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{name}] render FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
    return m


def main():
    import ecm_spec as ES
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--youngs", type=float, default=40.0)
    ap.add_argument("--substep", type=float, default=2.0e-4)
    ap.add_argument("--cavity-r", type=float, default=0.22)
    ap.add_argument("--cavity-h", type=float, default=0.07)
    ap.add_argument("--align", type=float, default=0.0)
    ap.add_argument("--growth", type=float, default=0.0009)
    ap.add_argument("--k", type=float, default=900.0)
    ap.add_argument("--particles", type=int, default=48000)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()
    spec = ES.build_spec(a.name, n_frames=a.frames, substep_dt=a.substep, youngs=a.youngs,
                         cavity_r=a.cavity_r, cavity_h=a.cavity_h, align=a.align,
                         growth=a.growth, k_contact=a.k, n_particles=a.particles, n_grid=a.grid)
    run(a.name, spec, device=a.device, movie=not a.no_movie)


if __name__ == "__main__":
    main()
