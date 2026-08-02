#!/usr/bin/env python
"""reel -- assemble the accumulating video battery.

Every run already produces a movie. Nothing assembled them, so a round's result could only be
inspected one file at a time and there was no way to SEE the campaign progress -- which is the
one thing a multi-week mechanism search most needs, because the numbers are exactly what the
Watcher exists to distrust.

Two products:

  round montage   the slots of one round side by side, each labelled with its edit, its score and
                  its verdict, so eye/number divergence is visible at a glance. A vetoed slot is
                  labelled VETOED rather than dropped -- the whole point is to see what the
                  metric liked and the eye did not.

  progress reel   one tile per round (its best surviving run), concatenated oldest to newest.
                  This is the artefact that answers "is the campaign getting anywhere".

Labels are burned in so a montage is self-describing when it turns up in a talk folder six weeks
later with no context.

⚠ TILES ARE NOT ON A COMMON SPATIAL SCALE, and the label says so on every tile.
Each per-run movie was rendered with its own camera box (`Lbox = 1.06 * Rmax` for that run), so a
run with a long thin spike gets a wide box and its body is drawn SMALL. Placing such tiles side by
side invites exactly the comparison the rendering cannot support: in the round-2 montage the
control looks like a tiny body with a stub next to two large spheres, and the size difference is
the camera, not the biology. A genuinely comparable montage needs the runs re-rendered against a
shared `Lbox`; until then the montage is for reading SHAPE and PHENOTYPE, never relative size.
This is recorded rather than fixed silently because an unlabelled montage is itself an instrument
that lies, which is the failure mode this project keeps paying for.

    python reel.py --round 2
    python reel.py --study            # the Turing x vertex waves
    python reel.py --progress
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")
CAMP = os.path.join(HERE, "campaign")
REEL = os.path.join(HERE, "_reel")

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"


def _label_png(text, width, path, height=30):
    """Render a label strip as an image.

    The bundled imageio-ffmpeg binary is built WITHOUT libfreetype, so it has no `drawtext`
    filter and there is no system ffmpeg here. Burning labels in is not optional -- an unlabelled
    montage is the thing that turns up in a talk folder six weeks later and cannot be read -- so
    the text is rendered with matplotlib and composited with `overlay`, which the binary does have.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("black")
    fig.text(0.012, 0.5, str(text), color="white", fontsize=8.5, va="center", ha="left",
             family="DejaVu Sans")
    fig.savefig(path, facecolor="black", dpi=100)
    plt.close(fig)
    return path


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # Never swallow: a montage that silently did not build is worse than a crash, because the
        # absence of a file reads as "there was nothing to show".
        print(p.stderr[-1500:], file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed ({p.returncode})")
    return p


def montage(tiles, out, cols=3, tile_w=420, fps=12, seconds=None):
    """tiles: [(mp4_path, label)] -> a grid movie with burned-in labels."""
    tiles = [(p, l) for p, l in tiles if p and os.path.exists(p)]
    if not tiles:
        print("[reel] no input movies -- nothing to assemble")
        return None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = len(tiles)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    tile_h = int(tile_w * 1.04)

    lab_dir = os.path.join(REEL, "_labels")
    os.makedirs(lab_dir, exist_ok=True)
    cmd = [FFMPEG, "-y"]
    for p, _ in tiles:
        if seconds:
            cmd += ["-t", str(seconds)]
        cmd += ["-i", p]
    for i, (_, label) in enumerate(tiles):
        lp = _label_png(label, tile_w, os.path.join(lab_dir, f"l{i}.png"))
        cmd += ["-i", lp]

    fc = []
    for i, (_, label) in enumerate(tiles):
        fc.append(
            f"[{i}:v]scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
            f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black[t{i}]")
        # eof_action=repeat holds the single-frame label for the whole clip
        fc.append(f"[t{i}][{n + i}:v]overlay=0:0:eof_action=repeat:shortest=0[v{i}]")
    # pad the grid so xstack always gets a full rectangle
    blanks = rows * cols - n
    for b in range(blanks):
        fc.append(f"color=c=black:s={tile_w}x{tile_h}:d=1[v{n + b}]")
    total = n + blanks
    layout = "|".join(f"{(k % cols) * tile_w}_{(k // cols) * tile_h}" for k in range(total))
    ins = "".join(f"[v{k}]" for k in range(total))
    fc.append(f"{ins}xstack=inputs={total}:layout={layout}:fill=black[out]")

    cmd += ["-filter_complex", ";".join(fc), "-map", "[out]",
            "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out]
    _run(cmd)
    print(f"[reel] {out}  ({n} tiles, {rows}x{cols})")
    return out


# --------------------------------------------------------------------------- sources
def round_tiles(rid, mode="c"):
    """One tile per slot of a round, labelled with edit / score / verdict."""
    hyp = {}
    hp = os.path.join(CAMP, "hypotheses.jsonl")
    if os.path.exists(hp):
        for line in open(hp):
            if line.strip():
                d = json.loads(line)
                if d.get("round_id") == rid:
                    hyp[d["hid"]] = d
    tiles = []
    for d in sorted(glob.glob(os.path.join(LOG, f"r{rid:03d}{mode}_*"))):
        nm = os.path.basename(d)
        mv = os.path.join(d, "movie.mp4")
        summ = {}
        dj = os.path.join(d, "diag.json")
        if os.path.exists(dj):
            summ = json.load(open(dj)).get("summary", {})
        edit = next((h.get("edit", "") for h in hyp.values()
                     if nm.endswith(h["comp_hash"][1:7])), nm)
        pk = summ.get("protr_peak")
        w = summ.get("watcher_verdict", "")
        tag = "VETOED" if summ.get("watcher_blocks") else (w or "")
        lab = f"{edit[:30]}  pk {pk:.2f}" if isinstance(pk, (int, float)) else edit[:30]
        if tag:
            lab += f"  [{tag}]"
        lab += "   (own scale)"      # tiles are NOT spatially comparable -- see module docstring
        tiles.append((mv, lab))
    return tiles


def study_tiles(tag=None):
    """The Turing x vertex waves, labelled with the knob and the two numbers that matter."""
    tiles = []
    for d in sorted(glob.glob(os.path.join(HERE, "_turing_vertex", "wave*_*"))):
        if not os.path.isdir(d) or (tag and not os.path.basename(d).startswith(tag)):
            continue
        dj = os.path.join(d, "diag.json")
        mv = os.path.join(d, "movie.mp4")
        if not os.path.exists(dj):
            continue
        r = json.load(open(dj))
        if not r.get("ok"):
            continue
        k = r["knobs"]
        c = r.get("corr_act_rad")
        nm = r["name"]
        knob = ("ca=%g" % k["conserve_amount"] if nm.startswith("waveC") else
                "a_sw=%g" % k["a_sw"] if nm.startswith(("waveA", "waveD")) else
                "rho=%g" % k["rho"] if nm.startswith(("waveB", "waveE")) else
                "chi=%g" % k["chi"] if nm.startswith("waveF") else "rate=%g" % k["rate"])
        integ = ("DESTROYED" if r["hollow_frac"] > 0.5
                 else "straining" if r["hollow_frac"] > 0.05 else "intact")
        lab = (f"{knob}  protr {r['protr']:.2f}  "
               f"corr {'n/a' if c is None else f'{c:+.2f}'}  [{integ}]   (own scale)")
        tiles.append((mv, lab))
    return tiles


def progress_tiles():
    """One tile per round: the best SURVIVING run (watcher-vetoed runs are not progress)."""
    lm = os.path.join(CAMP, "lever_map.jsonl")
    best = {}
    if os.path.exists(lm):
        for line in open(lm):
            if not line.strip():
                continue
            o = json.loads(line)
            rid = str(o.get("run_id") or "")
            if not rid:
                continue
            key = rid.split("_")[0]
            if o.get("score", -1e9) > best.get(key, (None, -1e9))[1]:
                best[key] = (rid, o.get("score", 0.0))
    tiles = []
    for key in sorted(best):
        rid, sc = best[key]
        tiles.append((os.path.join(LOG, rid, "movie.mp4"), f"{key}  best score {sc:.2f}"))
    return tiles


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--mode", default="c")
    ap.add_argument("--study", action="store_true")
    ap.add_argument("--wave", default=None)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(REEL, exist_ok=True)
    made = []
    if a.round is not None:
        made.append(montage(round_tiles(a.round, a.mode),
                            os.path.join(REEL, f"round{a.round:03d}{a.mode}_montage.mp4"),
                            cols=a.cols, seconds=a.seconds))
    if a.study:
        made.append(montage(study_tiles(a.wave),
                            os.path.join(REEL, f"study_{a.wave or 'all'}.mp4"),
                            cols=a.cols, seconds=a.seconds))
    if a.progress:
        made.append(montage(progress_tiles(), os.path.join(REEL, "progress_reel.mp4"),
                            cols=a.cols, seconds=a.seconds))
    if not any(m for m in made):
        print("[reel] nothing produced")
        sys.exit(1)
