#!/usr/bin/env python
"""A run and everything bred from it, drawn as a family tree with the edits on the branches.

Cedric, 12 August: *"make a genealogy montage starting top from r015_06 with vertical/horizontal
white lines."*

WHY A TREE AND NOT A ROW. `montage.py --lineage` walks ANCESTRY, which is a chain -- one `parent`
per record, so a run has exactly one line back to the basis. Descent is the other direction and it
branches: r015_06 has eight children and twenty-two grandchildren, and the interesting fact about
that family is not any one specimen, it is that r016_01 and r016_03 came off the SAME parent by one
edit each and went opposite ways -- one to broad petals, one to a bare white cross. A row cannot
show a fork. This is the picture that can.

THE EDIT IS ON THE BRANCH, not in a caption. Each tile carries the single edit that separates it
from the tile above it, so the tree reads as "this shape, plus this one change, gives that shape".
That is the only claim a one-edit-per-slot campaign can actually support, and writing it on the
connector is what makes the sheet an argument rather than a contact sheet.

LAYOUT. Leaves are laid out left to right in birth order and every parent is centred over the span
of its own children, which is the standard tidy-tree rule and the one that keeps a branch visually
contiguous. Connectors are orthogonal -- down from the parent, across the sibling span, down into
each child -- because a diagonal between two tiles at these densities reads as part of the picture.

    python genealogy.py r015_06
    python genealogy.py r015_06 --image 3d_c3.png --tile 240
"""
import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "log", "okuda")
RECORDS = os.path.join(HERE, "campaign", "records.jsonl")

from PIL import Image, ImageDraw, ImageFont          # noqa: E402

BG = (0, 0, 0)
FG = (235, 235, 235)
LINE = (255, 255, 255)
DIM = (150, 150, 150)


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def _records():
    by = {}
    if os.path.exists(RECORDS):
        with open(RECORDS) as f:
            for line in f:
                try:
                    r = json.loads(line); by[r["name"]] = r
                except Exception:
                    pass
    return by


def edit_label(rec):
    """The one edit that separates a run from its parent, short enough to sit on a branch."""
    e = (rec or {}).get("edit")
    if not e:
        return ""
    if isinstance(e[0], list):                 # several edits: name them all, compactly
        return " + ".join(edit_label({"edit": x}) for x in e)
    kind = e[0]
    if kind == "set_param":
        p = str(e[1]).split(".")[-1]
        v = e[2]
        return f"{p} {v:g}" if isinstance(v, (int, float)) else f"{p} {v}"
    if kind == "add_op":
        return f"+{e[1]}"
    if kind == "remove_op":
        return f"-{str(e[1]).rstrip('0123456789')}"
    if kind == "set_impl":
        return f"={e[1]}:{e[2]}"
    if kind == "connect":
        return f"~{e[1]}->{e[2]}"
    return kind


def _crop(im, thresh=12):
    g = im.convert("L")
    bb = g.point(lambda v: 255 if v > thresh else 0).getbbox()
    if not bb:
        return im
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.06
    W, H = im.size
    return im.crop((int(max(0, cx - half)), int(max(0, cy - half)),
                    int(min(W, cx + half)), int(min(H, cy + half))))


def _claim_tree(a):
    """The knowledge tree: claims, their parents, their status and their weights."""
    import claims as K
    spec = K.load_spec()
    cur, _ = K.load()
    if not cur:
        print("the claim ledger is empty"); return 1
    kids = defaultdict(list)
    for cid, c in cur.items():
        for p in (c.get("parents") or []) or [None]:
            kids[p].append(cid)
    for k in kids:
        kids[k].sort()
    roots = sorted(kids.get(None, []))

    xs, order, ctr = {}, [], [0]

    def place(n, d):
        order.append((n, d))
        ch = kids.get(n, [])
        if not ch:
            xs[n] = ctr[0]; ctr[0] += 1
        else:
            for c in ch:
                place(c, d + 1)
            xs[n] = (xs[ch[0]] + xs[ch[-1]]) / 2.0

    for r in roots:
        place(r, 0)
    depth = max(d for _, d in order)

    TILE, k = a.tile, a.tile / 200.0
    cw, rh = int(TILE * 1.35), int(TILE * 0.72)
    W, H = int(ctr[0] * cw) + 20, int((depth + 1) * rh) + 40
    sheet = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(sheet)
    f_id, f_txt = _font(int(15 * k)), _font(int(11 * k))
    COL = {"contested": (250, 190, 60), "supported": (120, 220, 120),
           "refuted": (230, 90, 90), "proposed": (200, 200, 200),
           "stale": (130, 130, 130), "superseded": (110, 90, 140)}

    def cx(n): return int(xs[n] * cw + cw / 2) + 10
    def ty(d): return int(d * rh) + 30

    for n, d in order:
        for c in kids.get(n, []):
            ym = (ty(d) + int(TILE * 0.44) + ty(d + 1)) // 2
            dr.line([(cx(n), ty(d) + int(TILE * 0.44)), (cx(n), ym)], fill=LINE, width=max(2, int(2 * k)))
            dr.line([(cx(n), ym), (cx(c), ym)], fill=LINE, width=max(2, int(2 * k)))
            dr.line([(cx(c), ym), (cx(c), ty(d + 1))], fill=LINE, width=max(2, int(2 * k)))
    for n, d in order:
        c = cur[n]
        fo, ag = K.weigh(c, spec)
        x, y = cx(n) - int(cw * 0.46), ty(d)
        col = COL.get(c.get("status"), FG)
        dr.rectangle([x, y, x + int(cw * 0.92), y + int(TILE * 0.44)], outline=col,
                     width=max(2, int(2 * k)))
        dr.text((x + 8, y + 5), f"{n}  {c.get('status')}", fill=col, font=f_id)
        words, line, lines = c["statement"].split(), "", []
        while words:
            w = words.pop(0)
            if len(line) + len(w) > 34:
                lines.append(line); line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        for i, ln in enumerate(lines[:3]):
            dr.text((x + 8, y + int(24 * k) + i * int(13 * k)), ln, fill=FG, font=f_txt)
        # PLACED BELOW THE LAST LINE OF THE STATEMENT, not a fixed offset from the bottom: a
        # three-line statement overlapped it and the two texts drew on top of each other.
        dr.text((x + 8, y + int(24 * k) + len(lines[:3]) * int(13 * k) + int(4 * k)),
                f"{c['kind']}   for {fo:.1f} / against {ag:.1f}", fill=DIM, font=f_txt)

    out = a.out or os.path.join(LOG, "_gates", "genealogy_claims.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    lv = depth + 1
    print(f"claim tree: {len(order)} claims over {lv} level(s), {len(roots)} root(s), "
          f"{W}x{H} px")
    print(f"  -> {os.path.relpath(out, ROOT)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--image", default="3d_c3.png")
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--out", default=None)
    # PHASE 4 -- THE SECOND GENEALOGY. The composition tree answers "what was bred from what"; this
    # answers "what was learned from what". Same renderer, same tidy layout, same generated-not-
    # curated rule -- a claim's `parents` field is written by the Analyst when a claim derives
    # another, exactly as a run's `parent` is written when a slot breeds from one.
    #
    # Tiles are TEXT, not pictures: a claim has no 3d.png. The panel carries the id, the status,
    # the weights for and against, and the act that derived it.
    ap.add_argument("--claims", action="store_true",
                    help="draw the CLAIM tree from campaign/claims.jsonl instead of the run tree")
    a = ap.parse_args()

    if a.claims:
        return _claim_tree(a)

    by = _records()
    kids = defaultdict(list)
    for n, r in by.items():
        kids[r.get("parent")].append(n)
    for k in kids:
        kids[k].sort()

    def has(n):
        return os.path.exists(os.path.join(LOG, n, a.image))

    # ONLY RUNS WITH A PICTURE take a slot, and a childless one without a picture is dropped
    # entirely rather than left as a black square -- but one WITH children is kept, because
    # deleting it would silently reparent its subtree and the tree would assert a descent that
    # never happened.
    def prune(n):
        ch = [c for c in kids.get(n, []) if prune(c)]
        kids[n] = ch
        return has(n) or bool(ch)

    prune(a.root)

    # tidy layout: leaves left to right, parents centred over their children
    xs, order = {}, []
    counter = [0]

    def place(n, depth):
        order.append((n, depth))
        ch = kids.get(n, [])
        if not ch:
            xs[n] = counter[0]; counter[0] += 1
        else:
            for c in ch:
                place(c, depth + 1)
            xs[n] = (xs[ch[0]] + xs[ch[-1]]) / 2.0

    place(a.root, 0)
    if not order:
        print(f"{a.root}: nothing to draw"); return 1
    depth = max(d for _, d in order)
    ncol = counter[0]

    # EVERYTHING SCALES WITH THE TILE. A 27-leaf tree is 6.7 times wider than it is tall, so it is
    # always read zoomed in; a label pinned at 13 px is then unreadable beside a 300 px specimen,
    # and connectors 2 px wide vanish. Proportions hold at any --tile.
    TILE = a.tile
    k = TILE / 200.0
    LAB = int(34 * k)            # two text lines under each tile
    GAP = int(54 * k)            # where the connectors live
    PAD = int(8 * k)
    LW = max(2, int(2 * k))
    cw = TILE + 2 * PAD
    rh = TILE + LAB + GAP
    # A HEADER, BECAUSE THE ROWS DO NOT MEAN WHAT THEY LOOK LIKE. Cedric, reading the first
    # version: *"r017 r018 r019 r020 cannot be on the same row??"* They can, and they are: a row is
    # EDIT DISTANCE from the root, not time. r016_03 was still a parent in r017, r018, r019 and
    # r020, so its children carry four different round numbers and are all one edit away. A picture
    # whose axis has to be explained in conversation is a picture that should say it itself.
    HEAD = int(46 * k)
    W, H = int(ncol * cw), int((depth + 1) * rh + PAD + HEAD)
    sheet = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(sheet)
    f_name, f_edit = _font(int(13 * k)), _font(int(12 * k))

    def cx_of(n):
        return int(xs[n] * cw + cw / 2)

    def top_of(d):
        return int(d * rh + PAD + HEAD)

    # CONNECTORS FIRST, so the tiles are drawn over them and no line crosses a specimen.
    for n, d in order:
        ch = kids.get(n, [])
        if not ch:
            continue
        y0 = top_of(d) + TILE + LAB           # just under the parent's label
        y1 = top_of(d + 1) - int(10 * k)      # just above the children
        ym = (y0 + y1) // 2
        dr.line([(cx_of(n), y0), (cx_of(n), ym)], fill=LINE, width=LW)
        if len(ch) > 1:
            dr.line([(cx_of(ch[0]), ym), (cx_of(ch[-1]), ym)], fill=LINE, width=LW)
        for c in ch:
            dr.line([(cx_of(c), ym), (cx_of(c), y1)], fill=LINE, width=LW)

    f_head = _font(int(17 * k))
    spread = {}
    for n, _d in order:
        for c in kids.get(n, []):
            spread.setdefault(n, set()).add(c.split("_")[0])
    worst = max(spread.items(), key=lambda kv: len(kv[1])) if spread else (None, set())
    note = (f"rows are GENERATIONS -- edits from {a.root} -- NOT rounds.")
    if worst[0] and len(worst[1]) > 1:
        note += (f"   {worst[0]} was a parent in {', '.join(sorted(worst[1]))}, "
                 f"so its {len(kids[worst[0]])} children share a row across "
                 f"{len(worst[1])} rounds.")
    dr.text((int(10 * k), int(12 * k)), note, fill=FG, font=f_head)

    drawn = 0
    for n, d in order:
        x, y = cx_of(n) - TILE // 2, top_of(d)
        p = os.path.join(LOG, n, a.image)
        if os.path.exists(p):
            try:
                with Image.open(p) as im:
                    im = _crop(im.convert("RGB"))
                    im.thumbnail((TILE, TILE), Image.LANCZOS)
                    sheet.paste(im, (x + (TILE - im.size[0]) // 2, y))
                    drawn += 1
            except Exception as e:
                print(f"  {n}: {type(e).__name__}")
        dr.text((x, y + TILE + int(2 * k)), n, fill=FG, font=f_name)
        lab = edit_label(by.get(n))
        if lab:
            dr.text((x, y + TILE + int(18 * k)), lab[:26], fill=DIM, font=f_edit)

    out = a.out or os.path.join(LOG, "_gates", f"genealogy_{a.root}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out, optimize=True)
    print(f"{a.root}: {len(order)} runs over {depth + 1} generations, {drawn} with a picture "
          f"({ncol} leaves, {W}x{H} px)")
    print(f"  -> {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
