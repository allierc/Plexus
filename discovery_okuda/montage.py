#!/usr/bin/env python
"""Every generated shape on one sheet: all the `3d.png` end-frames, reduced and tiled.

Cedric, 11 August: *"make a montage with all 3d.png so that I can see all generated shapes in one
big png, of course reduce each 3d.png first."*

WHY REDUCE FIRST AND NOT AFTER. There are 246 end-frames in `log/okuda` alone at 614x614 RGBA --
about 370 MB decoded if they are all held at full size, and Pillow will do exactly that if the
resize happens after the paste. Each is opened, thumbnailed with `draco`-quality resampling and
closed before the next, so the peak cost is one full image plus the finished sheet.

THE TILES ARE CROPPED TO THEIR CONTENT. A `3d.png` is a matplotlib figure: the shape occupies the
middle and the rest is black margin, so a naive grid spends most of its pixels on nothing. The
bounding box of the non-black pixels is taken first, which roughly doubles the shape's size in the
same sheet area.

ORDER IS THE ARGUMENT. Sorting by name puts r001_00 beside r001_01, which compares two runs that
differ by one edit -- useful. Sorting by a metric puts the best specimen first and the failures
last, which compares the campaign against itself. `--by` chooses; the label under each tile always
carries the run name and the sorted quantity, so a tile can be traced back.

    python montage.py                          the current campaign, by name
    python montage.py --by protr_final         best protrusion first
    python montage.py --glob 'sc_*' --by n_apop     one series, most deaths first
"""
import argparse
import json
import os
import sys
from fnmatch import fnmatch

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(HERE), "log", "okuda")

from PIL import Image, ImageDraw, ImageFont          # noqa: E402

TILE = 190          # px per tile before the label strip
LABEL = 16          # px of caption under each tile
PAD = 3
BG = (0, 0, 0)
FG = (235, 235, 235)


def _font(sz=11):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def _crop_to_content(im, thresh=12):
    """The bounding box of everything that is not background, so the tile is mostly shape."""
    g = im.convert("L")
    bb = g.point(lambda v: 255 if v > thresh else 0).getbbox()
    if not bb:
        return im
    # keep it square so the aspect is not distorted, and give it a small margin
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.06
    W, H = im.size
    return im.crop((int(max(0, cx - half)), int(max(0, cy - half)),
                    int(min(W, cx + half)), int(min(H, cy + half))))


def _metric(name, key):
    p = os.path.join(LOG, name, "diag.json")
    if not os.path.exists(p):
        return None
    try:
        s = json.load(open(p)).get("summary") or {}
    except Exception:
        return None
    v = s.get(key)
    return v if isinstance(v, (int, float)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*", help="which runs, e.g. 'r01*' or 'sc_*'")
    ap.add_argument("--by", default=None, help="a summary metric to sort by, descending")
    ap.add_argument("--cols", type=int, default=0, help="0 = square-ish")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = sorted(d for d in os.listdir(LOG)
                  if not d.startswith("_") and fnmatch(d, a.glob)
                  and os.path.exists(os.path.join(LOG, d, "3d.png")))
    if not runs:
        print(f"no run matching {a.glob!r} has a 3d.png"); return 1

    if a.by:
        vals = {r: _metric(r, a.by) for r in runs}
        # A RUN WITH NO VALUE IS NOT A RUN WITH ZERO -- it goes last and says so, rather than
        # sorting among the worst as though it had been measured and found wanting.
        runs = sorted(runs, key=lambda r: (vals[r] is None, -(vals[r] or 0)))
        cap = {r: (f"{vals[r]:.3f}" if vals[r] is not None else "--") for r in runs}
    else:
        cap = {r: "" for r in runs}

    n = len(runs)
    cols = a.cols or max(1, int(n ** 0.5 * 1.35))
    rows = (n + cols - 1) // cols
    cw, ch = TILE + 2 * PAD, TILE + LABEL + 2 * PAD
    sheet = Image.new("RGB", (cols * cw, rows * ch), BG)
    dr = ImageDraw.Draw(sheet)
    f = _font(11)

    for i, r in enumerate(runs):
        try:
            with Image.open(os.path.join(LOG, r, "3d.png")) as im:
                im = im.convert("RGB")
                im = _crop_to_content(im)
                im.thumbnail((TILE, TILE), Image.LANCZOS)
                x = (i % cols) * cw + PAD + (TILE - im.size[0]) // 2
                y = (i // cols) * ch + PAD
                sheet.paste(im, (x, y))
        except Exception as e:
            print(f"  skipped {r}: {type(e).__name__}")
            continue
        txt = f"{r} {cap[r]}".strip()
        dr.text(((i % cols) * cw + PAD, (i // cols) * ch + PAD + TILE + 2), txt[:34],
                fill=FG, font=f)

    out = a.out or os.path.join(LOG, "_gates",
                                f"montage_{a.glob.strip('*') or 'all'}"
                                f"{'_by_' + a.by if a.by else ''}.png".replace("*", ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out, optimize=True)
    print(f"{n} shapes, {cols}x{rows}, {sheet.size[0]}x{sheet.size[1]} px"
          f"{' sorted by ' + a.by if a.by else ''}")
    print(f"  -> {os.path.relpath(out, os.path.dirname(HERE))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
