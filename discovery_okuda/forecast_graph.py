#!/usr/bin/env python
"""The exploration graph, with the Forecaster's accuracy painted on every node.

CEDRIC, 14 AUGUST: *"if you pinpoint the forecaster precision on the exploration graph what do we
see... make a 3x3 pixel red (bad) green (good) results per node, put 3d.png small icon."*

WHAT ONE NODE IS. A run. Its picture is the tissue it produced; the little grid beside it is how
well the Forecaster called that run, ONE CELL PER SLOT, red = missed, green = called it. The edges
run parent -> child, so a green patch spreading down a branch is knowledge that transferred and a
red one is a branch the campaign is walking blind.

SIX CELLS, NOT NINE, and the difference is worth stating because the request assumed ten. The
Forecaster does not predict the ten admitted METRICS -- it fills `crew/description.md`'s form, which
is seven slots of which six are scored (`free` is deliberately unscored: it is where "like a flower"
goes, and scoring it would make it a slot the writer games). So the grid is 3x2. `CELLS` below is
`foresight.SCORED`; the layout is computed from its length, so adding a slot to the schema changes
this picture without touching this file.

WHY THE LAYOUT IS ROUND-BY-ROUND AND NOT A TREE. `genealogy.py` already draws the ancestry tree, and
a tree answers "what came from what". This answers "when did the campaign learn", which is a
question about TIME: a column per round, so a slot's accuracy can be read down the page as the
campaign accumulates knowledge, and the parent edges show which of those columns actually inherited.

    python forecast_graph.py                      the whole campaign
    python forecast_graph.py --rounds 5 12        a window
    python forecast_graph.py --slot count         one slot only, full-size cells
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.environ.get("OKUDA_LOG", os.path.join(ROOT, "log", "okuda"))
for _p in (HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CAMPAIGN = os.path.join(HERE, "campaign")
ICON = 62               # the run's own 3d.png
CELL = 13               # one slot's verdict
GAP, PADX, PADY = 6, 54, 30      # PADX is the gutter the edges are routed through
BG, FG, DIM = (0, 0, 0), (235, 235, 235), (110, 110, 110)


def _colour(v):
    """red -> amber -> green over [0, 1]. Grey when the slot was not scored at all.

    NOT A DIVERGING MAP AND NOT A RAINBOW. The question a cell answers is "did the Forecaster get
    this right", which is one-directional, so the ramp is one-directional. Grey is reserved for
    ABSENT -- a slot one side did not fill is not a slot the Forecaster got wrong, and `foresight.py`
    already refuses to average it in.
    """
    if v is None:
        return (52, 52, 58)
    v = max(0.0, min(1.0, float(v)))
    if v < 0.5:
        t = v / 0.5
        return (int(206 + 32 * t), int(38 + 130 * t), int(38 + 20 * t))     # red -> amber
    t = (v - 0.5) / 0.5
    return (int(238 - 150 * t), int(168 + 22 * t), int(58 + 26 * t))        # amber -> green


def load(rounds=None):
    """-> {run: {slot: score}}, {run: parent}, [round ids]. From the ledger the loop already writes."""
    import foresight as F
    fp = os.path.join(CAMPAIGN, "foresight.jsonl")
    per, rids = {}, []
    if os.path.exists(fp):
        for line in open(fp):
            try:
                d = json.loads(line)
            except Exception:
                continue
            rid = str(d.get("round") or "")
            n = int(rid[1:4]) if rid[1:4].isdigit() else None
            if rounds and n is not None and not (rounds[0] <= n <= rounds[1]):
                continue
            rids.append(rid)
            for run, r in (d.get("runs") or {}).items():
                per[run] = dict(r.get("per_slot") or {})
                per[run]["_mean"] = r.get("foresight")
    parent = {}
    rp = os.path.join(CAMPAIGN, "records.jsonl")
    if os.path.exists(rp):
        for line in open(rp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("name"):
                parent[r["name"]] = r.get("parent")
    return per, parent, sorted(set(rids)), F.SCORED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs=2, type=int, metavar=("FROM", "TO"))
    ap.add_argument("--slot", default=None, help="draw ONE slot per node, at full size")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont
    per, parent, rids, SCORED = load(a.rounds)
    if not per:
        print("no campaign/foresight.jsonl -- the Forecaster has not been scored yet")
        return 1
    cells = [a.slot] if a.slot else list(SCORED)
    if a.slot and a.slot not in SCORED:
        print(f"{a.slot!r} is not a scored slot. One of: {', '.join(SCORED)}")
        return 1

    # COLUMN PER ROUND. A run's name carries its round, so nothing has to be joined to get this.
    cols = {}
    for run in per:
        cols.setdefault(run.split("_")[0], []).append(run)
    order = sorted(cols)
    for k in cols:
        cols[k].sort()

    # BARYCENTRIC ORDERING, and without it the picture is unreadable. Sorted by NAME, a run sits
    # wherever the Proposer happened to number it, so a child of the top-left parent lands at the
    # bottom of the next column and its edge crosses every other edge in the gap. The first version
    # drew 183 such lines and Cedric could not see them at all.
    #
    # The fix is the standard layered-graph one: place each node at the mean row of its parents,
    # sweep left to right, repeat. Three sweeps is enough here -- the graph is shallow (13 layers)
    # and every node has at most one parent, so there is little for further passes to gain.
    for _ in range(3):
        for ci in range(1, len(order)):
            prev = {r: i for i, r in enumerate(cols[order[ci - 1]])}
            here = cols[order[ci]]
            # a run whose parent is not in the previous column keeps its place rather than being
            # swept to row 0 -- its position is unknown, not zero
            cols[order[ci]] = sorted(
                here, key=lambda r: (prev.get(parent.get(r), here.index(r) + 0.5)))

    gw = 1 if a.slot else 3
    gh = 1 if a.slot else (len(cells) + gw - 1) // gw
    cw = ICON + GAP + gw * CELL + PADX
    ch = max(ICON, gh * CELL) + PADY
    W, H = len(order) * cw + 40, max(len(v) for v in cols.values()) * ch + 92
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except Exception:
        f = fs = ImageFont.load_default()

    at = {}
    for ci, rid in enumerate(order):
        x = 20 + ci * cw
        dr.text((x, 8), rid, fill=FG, font=f)
        for ri, run in enumerate(cols[rid]):
            y = 44 + ri * ch
            at[run] = (x, y)

    # EDGES FIRST so a node's picture is never cut by a line. A parent outside the window is drawn
    # as a stub rather than dropped -- "this came from somewhere you are not looking at" is
    # information, and silently rootless nodes would read as a campaign with no lineage at all.
    # AND THE EDGE SAYS SOMETHING. A grey line only tells you a child had a parent, which the layout
    # already shows. Coloured by whether the child's foresight BEAT ITS PARENT'S, it answers the
    # question the graph is for: does breeding from a run make the campaign better at predicting the
    # next one? Green up, red down, grey when either end was not scored.
    for run, (x, y) in at.items():
        p = parent.get(run)
        if not p:
            continue
        y0, y1 = y + ICON // 2, y + ICON // 2
        if p in at:
            px, py = at[p]
            a_, b_ = per.get(p, {}).get("_mean"), per.get(run, {}).get("_mean")
            if a_ is None or b_ is None:
                col, w = (70, 70, 78), 2
            else:
                d = b_ - a_
                col = (90, 200, 110) if d > 0.05 else (206, 60, 55) if d < -0.05 else (120, 120, 130)
                w = 2 if abs(d) < 0.2 else 3
            x0, y0 = px + ICON, py + ICON // 2
            # ORTHOGONAL, NOT DIAGONAL. A straight line between two columns crosses every node in
            # between; a line that leaves the parent horizontally, steps across the gap and drops
            # into the child stays inside the gutter where nothing else is drawn.
            mx = x0 + (x - 4 - x0) // 2
            dr.line([(x0, y0), (mx, y0)], fill=col, width=w)
            dr.line([(mx, y0), (mx, y1)], fill=col, width=w)
            dr.line([(mx, y1), (x - 4, y1)], fill=col, width=w)
        else:
            dr.line([(x - 14, y1), (x - 4, y1)], fill=(80, 80, 88), width=2)

    for run, (x, y) in at.items():
        p3 = os.path.join(LOG, run, "3d.png")
        if os.path.exists(p3):
            try:
                with Image.open(p3) as t:
                    # CROPPED TO THE BODY FIRST. `3d.png` is a fixed camera box with the tissue
                    # somewhere inside it, so a raw thumbnail at 62 px spends most of its pixels on
                    # black and the spheroid comes out a smudge. `montage._crop_to_content` is the
                    # same trim the montage sheets use -- one definition of "where the tissue is".
                    from montage import _crop_to_content
                    t = _crop_to_content(t.convert("RGB"))
                    t.thumbnail((ICON, ICON), Image.LANCZOS)
                    im.paste(t, (x + (ICON - t.size[0]) // 2, y + (ICON - t.size[1]) // 2))
            except Exception:
                pass
        gx = x + ICON + GAP
        for k, s in enumerate(cells):
            v = per[run].get(s)
            cx, cy = gx + (k % gw) * CELL, y + (k // gw) * CELL
            dr.rectangle([cx, cy, cx + CELL - 2, cy + CELL - 2], fill=_colour(v))
        m = per[run].get("_mean")
        dr.text((x, y + max(ICON, gh * CELL) + 2),
                f"{run.split('_', 1)[1]} {'' if m is None else f'{m:.2f}'}", fill=DIM, font=fs)

    # THE LEGEND NAMES THE CELLS IN ORDER, or the grid is nine coloured squares meaning nothing.
    ly = H - 34
    dr.text((20, ly - 14), "each node: its 3d.png, then one cell per forecast slot "
                           f"(reading left to right, top to bottom): {', '.join(cells)}",
            fill=DIM, font=fs)
    for i, (lab, v) in enumerate([("missed", 0.0), ("", 0.35), ("", 0.65), ("called it", 1.0),
                                  ("not scored", None)]):
        bx = 20 + i * 92
        dr.rectangle([bx, ly, bx + CELL - 2, ly + CELL - 2], fill=_colour(v))
        if lab:
            dr.text((bx + CELL + 4, ly), lab, fill=DIM, font=fs)
    for i, (lab, col) in enumerate([("child predicted BETTER than its parent", (90, 200, 110)),
                                    ("worse", (206, 60, 55)), ("same", (120, 120, 130))]):
        bx = 500 + i * 230
        dr.line([(bx, ly + 5), (bx + 26, ly + 5)], fill=col, width=3)
        dr.text((bx + 32, ly), lab, fill=DIM, font=fs)

    out = a.out or os.path.join(LOG, "analysis",
                                f"forecast_graph{'_' + a.slot if a.slot else ''}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im.save(out, optimize=True)
    scored = [v["_mean"] for v in per.values() if v.get("_mean") is not None]
    print(f"{len(per)} runs over {len(order)} round(s), {len(cells)} cell(s) each; "
          f"mean foresight {sum(scored) / max(len(scored), 1):.3f}")
    print(f"  -> {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
