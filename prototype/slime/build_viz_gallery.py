"""Tile the multi-viz exploration into one sheet: rows = morphologies, columns =
viz modes (trail / flow / stream / orient / graph), each cell the final frame.

    python build_viz_gallery.py [spec1 spec2 ...]
"""

import os
import sys

from PIL import Image

MODES = ["trail", "flow", "stream", "orient", "graph"]
DEFAULT_SPECS = ["slime_default", "slime_fine", "slime_coarse", "slime_filaments",
                 "slime_curly", "slime_dense", "slime_ring_in", "slime_point",
                 "slime_random_full", "slime_two_repel", "slime_three", "slime_four"]


def last_panel(path):
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB"); w, h = im.size
    return im.crop((w - w // 6, 0, w, h))                 # montages are 6 panels wide


def main(specs):
    cw = ch = 240
    rows = []
    for s in specs:
        cells = []
        for m in MODES:
            p = "specs/%s_%s_montage.png" % (s, m)
            im = last_panel(p)
            cells.append(im.resize((cw, ch)) if im else Image.new("RGB", (cw, ch), "black"))
        rows.append(cells)
    gal = Image.new("RGB", (cw * len(MODES), ch * len(rows)), "black")
    for r, cells in enumerate(rows):
        for c, im in enumerate(cells):
            gal.paste(im, (c * cw, r * ch))
    gal.save("viz_gallery.png")
    print("viz_gallery.png: %d morphologies x %d modes" % (len(rows), len(MODES)))


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_SPECS)
