#!/usr/bin/env python
"""montage -- tile the archived tests into one overview PNG (the "exploration process" view,
like active_matter2's fig*_b*_montage.png). Each row = one test's blob evolution strip, labelled
with its name + key metrics (escape / disc_growth / aniso / polar). Successes AND failures.

    python prototype/embryogenesis/montage.py                # all archived tests
    python prototype/embryogenesis/montage.py water          # substring filter
    python prototype/embryogenesis/montage.py --out name.png water elastic
"""
import os, sys, glob, json
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "archive")
MONT = os.path.join(HERE, "montages")


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    argv = sys.argv[1:]
    out = "montage.png"
    if "--out" in argv:
        i = argv.index("--out"); out = argv[i + 1]; del argv[i:i + 2]
    filters = argv
    dirs = sorted(d for d in glob.glob(os.path.join(ARCHIVE, "*")) if os.path.isdir(d))
    if filters:
        dirs = [d for d in dirs if any(f in os.path.basename(d) for f in filters)]
    rows = []
    for d in dirs:
        strip = os.path.join(d, "blob_evolution.png")
        if not os.path.isfile(strip):
            strip = os.path.join(d, "fig_mpm_particle_evolution.png")
        if not os.path.isfile(strip):
            continue
        m = {}
        mj = os.path.join(d, "metrics.json")
        if os.path.isfile(mj):
            m = json.load(open(mj))
        label = (f"{os.path.basename(d)}   n={m.get('n_cells','?')} collapsed={m.get('collapsed','?')} "
                 f"nn_min={m.get('nn_min','?')} deform={m.get('deform','?')} flow={m.get('flow','?')} "
                 f"migr={m.get('migration','?')} seg={m.get('segregation','?')} accel={m.get('accel','?')} "
                 f"({m.get('seconds','?')}s)")
        rows.append((strip, label))
    if not rows:
        print("no archived tests matched", filters); return

    W = 1600; lab_h = 30; pad = 6
    imgs = []
    for strip, label in rows:
        im = Image.open(strip).convert("RGB")
        scale = W / im.width
        im = im.resize((W, int(im.height * scale)))
        canvas = Image.new("RGB", (W, im.height + lab_h), "black")
        canvas.paste(im, (0, lab_h))
        dr = ImageDraw.Draw(canvas)
        dr.text((8, 6), label, fill=(230, 230, 230), font=_font(16))
        imgs.append(canvas)
    total_h = sum(i.height + pad for i in imgs)
    board = Image.new("RGB", (W, total_h), "black")
    y = 0
    for im in imgs:
        board.paste(im, (0, y)); y += im.height + pad
    os.makedirs(MONT, exist_ok=True)
    path = os.path.join(MONT, out)
    board.save(path)
    print(f"[montage] {len(imgs)} tests -> {path}")


if __name__ == "__main__":
    main()
