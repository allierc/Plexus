"""am2_montage.py -- tile a batch's slot panels next to the paper reference figure.

The agentic loops call `batch_montage(...)` after each batch so the next iteration's
agent can Read ONE image that puts our reproduced panels beside the paper's figure --
the visual "adequation" check that drives the loop (understanding, not a scalar loss).
"""
from __future__ import annotations

import os
from PIL import Image, ImageDraw, ImageFont


def _load(path, h):
    im = Image.open(path).convert("RGB")
    w = int(im.width * h / im.height)
    return im.resize((w, h))


def _label(im, text):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    d.text((6, 3), text, fill=(255, 255, 255), font=f)
    return im


def batch_montage(jobs, paper_ref, out, title="", panel_h=420):
    """Tile each slot's panel.png (labelled by slot name + progress) + the paper figure."""
    tiles = []
    for j in jobs:
        p = os.path.join(j["dir"], "panel.png")
        if not os.path.exists(p):
            continue
        prog = ""
        pp = os.path.join(j["dir"], "progress.txt")
        if os.path.exists(pp):
            prog = " " + open(pp).read().strip().split("\n")[0]
        tiles.append(_label(_load(p, panel_h), f"s{j['slot']} {j['name']}{prog}"[:60]))
    if os.path.exists(paper_ref):
        ref = _load(paper_ref, panel_h)
        tiles.append(_label(ref, "PAPER REFERENCE"))
    if not tiles:
        print("[montage] no panels to tile"); return None
    gap, pad = 8, 34
    W = sum(t.width for t in tiles) + gap * (len(tiles) - 1) + 2 * pad
    H = panel_h + pad + 12
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    x = pad
    for t in tiles:
        canvas.paste(t, (x, pad)); x += t.width + gap
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    d.text((pad, 6), title, fill=(255, 255, 255), font=f)
    canvas.save(out)
    print(f"[montage] -> {out}")
    return out
