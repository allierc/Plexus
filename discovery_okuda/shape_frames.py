#!/usr/bin/env python
"""The tissue's shape through time, as N square PNGs an image encoder can eat unmodified.

CEDRIC, 13 AUGUST, looking at a strip: *"one issue as above is the strip.png itself, we should just
use first row"*, then *"a strip with the right size of panels as used in transformers"*, then
*"render the best strip version for the transformer and fill up all 382 folders."*

WHAT IS WRONG WITH FEEDING `strip.png` TO AN ENCODER. Three of its four rows are not the shape:
row 2 is the same body from a second camera, row 3 is a PER-FRAME CONTRAST STRETCH OF CELL RADIUS
(`crew/strip.md` -- it paints large coloured domains on a body with 0.5% radial variation), and row 4
is a cross-section whose hollow centre is hardcoded at 0.82 of the outline. Measured on the strips on
disk, only 4.3-4.5% of row 1's pixels are even lit on a star. So three quarters of every image was
the same body again or a derived label, and an encoder asked how similar two runs are was being
handed mostly redundancy -- which is the direction that pushes similarity UP. CLIP already puts all
350 runs at cosine 0.9466 +/- 0.0297.

FOUR DECISIONS, each of which changes what the encoder sees:

  224 SQUARE, NOT 128 AND NOT A STRIP. Both cached encoders are 224 native -- CLIP ViT-B/32 (patch
  32, a 7x7 token grid) and SigLIP base (patch 16, 14x14). At 224 there is NO RESAMPLING anywhere in
  the path. And the frames are written as SEPARATE FILES rather than tiled: eight timepoints tiled
  into one 224 image would be 56x56 each, under two CLIP patches per timepoint. Separate files give
  every timepoint the whole token grid. A strip is a container the pipeline would only have to undo.

  SUPERSAMPLED 2x. Rendered at 448 and Lanczos-reduced to 224. VTK's MSAA x8 antialiases polygon
  EDGES; it does not help when a 20,000-cell mesh puts several cells inside one pixel, and the thin
  arms this campaign is looking for are exactly what aliasing eats.

  NO TEXT, and this one would have quietly poisoned the whole experiment. `vtk_render.evolve` stamps
  `"<run> <style> frame t/T <n> cells"` into the corner of every frame. Rendering the RUN'S OWN NAME
  into the image means an encoder can read the filename off the picture, and a spec-to-latent
  regression would then be predicting a caption it was never supposed to see. This module draws the
  mesh and nothing else.

  THE RUN'S OWN FIXED CAMERA BOX, as every other picture of it uses. Held constant across the N
  frames, so GROWTH WITHIN A RUN IS VISIBLE -- a per-frame refit is what "rescaling hid growth"
  means. It also means size is NOT comparable BETWEEN runs, which was already true of every artefact
  here and is why `camera_lbox` is recorded.

WHAT IT WRITES, per run:

    shape_strip.png                   the N frames in one row, losslessly. THE ARTEFACT THAT STAYS.
    shape/000.png .. shape/00N.png    224x224 RGB, one per timepoint -- what an image encoder would
                                      read. DELETED once the strip is written, unless --keep-frames:
                                      the encoder route was set aside and the strip holds the same
                                      pixels.
    shape.json                        n_frames, size, supersample, camera box, frame indices, the
                                      cell count at each -- so a later reader never has to guess
                                      what it is looking at or re-derive the mapping to traj.npz.

The strip is free: the frames are already in memory. Two products, one render pass.

    python shape_frames.py b_star                      one run
    python shape_frames.py --all --root _archive_...   every run under a root
    python shape_frames.py --submit 32                 fan the whole corpus over 32 cluster jobs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "log", "okuda")
for _p in (HERE, os.path.join(ROOT, "discovery_okuda", "ops"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# EGL BEFORE VTK, for the reason vtk_render.py gives: without it VTK finds the stub display and
# silently falls back to a software rasteriser -- same picture, no GPU, and nothing says so.
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

N_FRAMES = 8               # timepoints per run; the existing strip's column count
OUT_SIZE = 224             # both cached encoders' native input
SUPERSAMPLE = 2            # render at OUT_SIZE * this, then Lanczos down
STYLE = "nomesh"           # cell outlines smear to a dark field at 20k cells


def _pick(n, k):
    """k frame indices spanning the run, first and last always included."""
    if n <= k:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def render_one(run, n_frames=N_FRAMES, size=OUT_SIZE, ss=SUPERSAMPLE, force=False, quiet=False,
               keep_frames=False):
    """-> dict written to shape.json, or None if the run cannot be rendered."""
    import vtk_render as V
    from PIL import Image

    d = os.path.join(LOG, run)
    out_dir = os.path.join(d, "shape")
    meta_p = os.path.join(d, "shape.json")
    if os.path.exists(meta_p) and not force:
        try:
            m = json.load(open(meta_p))
            if m.get("n_frames") == n_frames and m.get("size") == size:
                return m                       # already done to this spec
        except Exception:
            pass
    if not os.path.exists(os.path.join(d, "traj.npz")):
        return None

    fr = V.frames_of(run)
    if not fr:
        return None
    L = V.box_of(run, fr)
    idx = _pick(len(fr), n_frames)

    # ONE ACTIVATOR RANGE ACROSS THE WHOLE RUN, exactly as `evolve` does it. Per-frame normalisation
    # would make a strengthening pattern look constant -- and would make the colour of a frame depend
    # on which frames happened to be sampled, so two renders of one run could disagree.
    vals = [np.asarray(a, float) for _p, _m, a in fr if a is not None]
    lo = float(min(np.nanmin(v) for v in vals)) if vals else 0.0
    hi = float(max(np.nanmax(v) for v in vals)) if vals else 1.0

    V.SIZE = size * ss
    p = V._plotter()
    os.makedirs(out_dir, exist_ok=True)
    tiles, cells, actor = [], [], None
    for j, t in enumerate(idx):
        pos, mt, act = fr[t]
        m = V.mesh_of(pos, mt, act, lo, hi, show_div=(STYLE == "mesh"))
        if m is None:
            continue
        if actor is not None:
            p.remove_actor(actor)              # NOT p.clear(): that removes the lights too
        actor = V.add(p, m, STYLE)
        V.aim(p, L)
        img = Image.fromarray(np.asarray(p.screenshot(return_img=True))[:, :, :3])
        if ss != 1:
            img = img.resize((size, size), Image.LANCZOS)
        img.save(os.path.join(out_dir, f"{j:03d}.png"))
        tiles.append(img)
        cells.append(int(mt["nF"]))
    p.close()
    if not tiles:
        return None

    # THE HUMAN STRIP, FREE. The frames are already decoded in memory, so this costs a paste and a
    # save -- and without it nobody can look at what the encoder was given, which is how the last
    # artefact went four rows deep with nobody knowing what three of them were.
    sheet = Image.new("RGB", (size * len(tiles), size), (0, 0, 0))
    for j, im in enumerate(tiles):
        sheet.paste(im, (j * size, 0))
    sheet.save(os.path.join(d, "shape_strip.png"))

    # THE PER-FRAME PNGs GO ONCE THE STRIP EXISTS. Cedric, 14 August: "folder shape in rxxx_xxx
    # should be deleted once used." They were written as an encoder's input -- 224 px native, one
    # file per timepoint so every frame gets the whole token grid -- and the encoder route was set
    # aside: a randomly-initialised ViT matched pretrained CLIP on this data, and the surrogate that
    # survived reads METRICS, not pictures. `shape_strip.png` holds the same eight frames losslessly
    # side by side, so nothing visual is lost and `--keep-frames` puts them back the moment an
    # encoder needs them. 18 MB across 107 runs at the time this was added; the point is the clutter
    # in every run directory rather than the bytes.
    if not keep_frames:
        for q in glob.glob(os.path.join(out_dir, "*.png")):
            os.remove(q)
        try:
            os.rmdir(out_dir)
        except OSError:
            pass

    meta = {"run": run, "n_frames": len(tiles), "size": size, "supersample": ss, "style": STYLE,
            "camera_lbox": float(L), "frame_index": idx[:len(tiles)], "n_cells": cells,
            "n_recorded": len(fr), "act_lo": lo, "act_hi": hi,
            "note": "shape/NNN.png are 224x224 RGB, no text, no scale bar, fixed camera. "
                    "shape_strip.png is the same frames in one row, for humans."}
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=1)
    if not quiet:
        print(f"  {run:52s} {len(tiles)} frames  {cells[0]:>6d}->{cells[-1]:>6d} cells")
    return meta


def corpus():
    """Every run with a traj.npz, current and archived, as paths relative to LOG."""
    import glob
    out = []
    # `_gates/*` WAS MISSED BY THE FIRST GLOB and it is not an empty corner: 17 runs, every one with
    # a traj.npz, and they are the instrument-gate specimens -- the runs the campaign uses to decide
    # whether a metric may be trusted. A pattern list that names two directories and silently omits a
    # third is the same shape as an allowlist, which is why the corpus is now three patterns and the
    # count is printed.
    for pat in ("*/traj.npz", "_archive*/*/traj.npz", "_gates/*/traj.npz"):
        for p in glob.glob(os.path.join(LOG, pat)):
            out.append(os.path.relpath(os.path.dirname(p), LOG))
    return sorted(set(out))


def submit(n_jobs, force=False, queue=None):
    """Fan the corpus over `n_jobs` cluster jobs.

    BATCHED, NOT ONE JOB PER RUN. Process start, imports and EGL context creation cost about 2 s and
    a render is about 4 -- so 382 separate bsubs would spend a third of the bill on starting up, and
    would put 382 entries in a queue this campaign shares with other people. Each job walks its own
    slice in one process and pays the startup once.
    """
    import cluster as C
    runs = corpus()
    if not runs:
        print("no runs with traj.npz")
        return 1
    n_jobs = max(1, min(n_jobs, len(runs)))
    lots = [runs[i::n_jobs] for i in range(n_jobs)]
    logdir = os.path.join(LOG, "_cluster")
    os.makedirs(logdir, exist_ok=True)
    q = queue or C.QUEUE
    cmds = []
    for i, lot in enumerate(lots):
        if not lot:
            continue
        sh = os.path.join(logdir, f"shape_{i:02d}.sh")
        with open(sh, "w") as f:
            f.write("\n".join([
                "#!/bin/bash -l",
                f"cd {C.cpath(HERE)}",
                f"export PYTHONPATH={C.cpath(os.path.join(ROOT, 'src'))}:"
                f"{C.cpath(os.path.join(HERE, 'ops'))}:{C.cpath(HERE)}",
                "export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow PYVISTA_OFF_SCREEN=true",
                "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4",
                f"conda run -n {C.ENV} python shape_frames.py {'--force ' if force else ''}"
                + " ".join(lot),
            ]) + "\n")
        os.chmod(sh, 0o755)
        out = C.cpath(os.path.join(logdir, f"shape_{i:02d}.out"))
        excl = "".join(f'-R "hname!={h}" ' for h in C.EXCLUDE_HOSTS if h)
        cmds.append(f"cd {C.cpath(HERE)} && bsub -n 4 -gpu num=1 {excl}-q {q} -W 60 "
                    f"-J {C.PREFIX}shape_{i:02d} -o {out} -e {out[:-4]}.err bash -l {C.cpath(sh)}")
    print(f"{len(runs)} runs over {len(cmds)} job(s) on {q}, "
          f"{min(len(l) for l in lots if l)}-{max(len(l) for l in lots)} runs each")
    r = C._ssh("\n".join(cmds))
    print((r.stdout or "")[-1500:] or (r.stderr or "")[-800:])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every run with a traj.npz")
    ap.add_argument("--submit", type=int, metavar="N", help="fan the corpus over N cluster jobs")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--size", type=int, default=OUT_SIZE)
    ap.add_argument("--supersample", type=int, default=SUPERSAMPLE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-frames", action="store_true", dest="keep_frames",
                    help="keep shape/NNN.png -- what an image encoder would read")
    a = ap.parse_args()

    if a.submit:
        return submit(a.submit, force=a.force)

    runs = corpus() if a.all else a.runs
    if not runs:
        ap.error("name runs, or --all, or --submit N")
    t0, ok, skip = time.perf_counter(), 0, []
    for r in runs:
        try:
            m = render_one(r, a.frames, a.size, a.supersample, force=a.force,
                           keep_frames=a.keep_frames)
        except Exception as e:
            print(f"  {r:52s} FAILED: {type(e).__name__}: {str(e)[:90]}")
            skip.append(r)
            continue
        ok += m is not None
        if m is None:
            skip.append(r)
    dt = time.perf_counter() - t0
    print(f"\n{ok}/{len(runs)} rendered in {dt / 60:.1f} min ({dt / max(ok, 1):.1f} s/run)")
    # SAID, NOT SWALLOWED. A run silently absent from the corpus reads downstream as a run the
    # encoder chose not to embed rather than one that was never offered.
    if skip:
        print(f"NOT rendered ({len(skip)}): {', '.join(skip[:8])}"
              + (f" ... and {len(skip) - 8} more" if len(skip) > 8 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
