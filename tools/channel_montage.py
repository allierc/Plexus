"""One contact sheet of a batch: the frame at `t` seconds of every run's movie, labelled, in one PNG -- what the
batch's single judge looks at (experiments/exp04_membrane_channels/AGENT_channel_judge.md), instead of opening twenty movies.

    PYTHONPATH=src python tools/channel_montage.py 10 [--t 7.5] [--cols 5]
        -> experiments/exp04_membrane_channels/figs/batch10_t7.5.png

The labels are the run's version and its row's channel and state (channel_spec.VERSIONS' `why`, cut short).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
FFMPEG = "/workspace/.conda_envs/neural-graph-linux/bin/ffmpeg"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch", type=int)
    ap.add_argument("--t", type=float, default=7.5)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--width", type=int, default=640, help="pixels per tile")
    a = ap.parse_args()
    import channel_spec as S
    from PIL import Image, ImageDraw, ImageFont
    from plexus.paths import graphs_data_path
    labels = sorted([k for k in S.VERSIONS if k.startswith(str(a.batch)) and k[len(str(a.batch)):].isalpha()
                     and len(k) == len(str(a.batch)) + 1])
    tiles = []
    with tempfile.TemporaryDirectory() as td:
        for lab in labels:
            mp4 = os.path.join(graphs_data_path(), "channel", f"exp04_v{lab}", "movie.mp4")
            png = os.path.join(td, f"{lab}.png")
            ok = os.path.isfile(mp4) and subprocess.run(
                [FFMPEG, "-loglevel", "error", "-y", "-ss", str(a.t), "-i", mp4, "-frames:v", "1", png]).returncode == 0
            im = Image.open(png).convert("RGB") if ok and os.path.isfile(png) else Image.new("RGB", (1808, 1280), (40, 40, 40))
            h = int(im.height * a.width / im.width)
            im = im.resize((a.width, h))
            d = ImageDraw.Draw(im)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
            except OSError:
                font = ImageFont.load_default()
            txt = f"{lab}  {S.VERSIONS[lab]['why'][:60]}"
            d.rectangle([0, 0, a.width, 26], fill=(0, 0, 0))
            d.text((6, 3), txt, fill=(255, 255, 255), font=font)
            tiles.append(im)
    if not tiles:
        raise SystemExit(f"no versions of batch {a.batch}")
    w, h = tiles[0].width, max(t.height for t in tiles)
    rows = (len(tiles) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (w * a.cols, h * rows), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % a.cols) * w, (i // a.cols) * h))
    out = os.path.join(ROOT, "experiments", "exp04_membrane_channels", "figs", f"batch{a.batch}_t{a.t}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print(f"[montage] {len(tiles)} runs -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
