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


def _ancestry(run):
    """RUN and every ancestor, oldest first. One parent per record, so this is a chain."""
    rec = os.path.join(HERE, "campaign", "records.jsonl")
    by = {}
    if os.path.exists(rec):
        with open(rec) as f:
            for line in f:
                try:
                    r = json.loads(line); by[r["name"]] = r
                except Exception:
                    pass
    chain, n, seen = [], run, set()
    while n and n not in seen:
        seen.add(n); chain.append(n)
        n = (by.get(n) or {}).get("parent")
    if n:                      # a cycle cannot happen with one parent per record, but say so
        print(f"  lineage stopped: {n} repeats")
    return list(reversed(chain))


def _identical(runs):
    """-> ([cluster, ...], {run: label}) for runs whose traj.npz is byte-for-byte the same.

    WHY THE TRAJECTORY AND NOT THE METRICS. Two runs can agree on every admitted metric and be
    different experiments; only an identical trajectory proves the substrate did the same thing
    twice. It is also the cheapest possible check -- one md5 per run, no parsing.

    AND WHY THE LABEL NAMES THE DIFFERENCE. Three collisions in this campaign, three distinct
    causes: `r003_01`/`r004_01` differ in NOTHING (a re-proposal a round apart that the dedupe
    missed); `r002_01`/`r004_02` differ only in `comp_hash` and `src_op`, which are provenance and
    not physics; `r004_12`/`r004_13` differ in `vth_frac`, which turns out to be INERT on that
    composition. "Duplicate" would have flattened three different bugs into one word.
    """
    import hashlib
    import yaml
    h = {}
    for r in runs:
        p = os.path.join(LOG, r, "traj.npz")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            h.setdefault(hashlib.md5(fh.read()).hexdigest(), []).append(r)
    clusters = [v for v in h.values() if len(v) > 1]

    def flat(d, pre=""):
        """Flatten a spec, keying an operator by its NAME and not its position in the list.

        THE POSITIONAL VERSION LIED, and it lied confidently. `operators[2]` is a different operator
        in two specs whose schedules differ, so comparing `operators[2].K_A` across them reported
        the entire mechanics block as differing when both specs carried IDENTICAL mechanics -- and
        the label read "INERT: Gamma,K_A,K_P,K_R,K_V,K_bend,K_lumen,K_purse,Lambda" on a pair whose
        only real difference was `cone_deg`. A diff keyed on position measures the ordering, not the
        parameters.
        """
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if k != "name":
                    out.update(flat(v, pre + "." + str(k)))
        elif isinstance(d, list):
            for i, v in enumerate(d):
                # an operator entry is keyed by op name (+ an index for a repeated operator)
                tag = f"[{v['op']}]" if isinstance(v, dict) and "op" in v else f"[{i}]"
                out.update(flat(v, pre + tag))
        else:
            out[pre] = d
        return out

    why = {}
    for c in clusters:
        fs = []
        for r in c:
            try:
                fs.append(flat(yaml.safe_load(open(os.path.join(LOG, r, "spec_run.yaml")))))
            except Exception:
                fs.append({})
        allk = set().union(*[set(f) for f in fs]) if fs else set()
        diff = sorted({k.split(".")[-1] for k in allk if len({f.get(k) for f in fs}) > 1})
        # PROVENANCE IS NOT PHYSICS. A pair differing only in these ran the same experiment and the
        # dedupe key was reading a serial number.
        prov = {"comp_hash", "src_op", "run_key", "parent"}
        if not diff:
            tag = "SAME SPEC"
        elif set(diff) <= prov:
            tag = "PROVENANCE ONLY: " + ",".join(diff)
        else:
            tag = "INERT: " + ",".join(d for d in diff if d not in prov)
        for r in c:
            why[r] = tag
    return clusters, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*", help="which runs, e.g. 'r01*' or 'sc_*'")
    ap.add_argument("--by", default=None, help="a summary metric to sort by, descending")
    ap.add_argument("--cols", type=int, default=0, help="0 = square-ish")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="3d.png",
                    help="which end-frame file to tile, e.g. 3d_c3.png from restyle3d.py")
    # LINEAGE. `parent` is ONE name on every record, so a run's ancestry is a CHAIN and not a DAG:
    # the campaign as a whole is a tree, but tracing any single run back gives a line. Ordered
    # oldest first, so the sheet reads left to right as the edits that built the specimen, and
    # `--by` is ignored -- the order IS the information.
    ap.add_argument("--lineage", default=None, metavar="RUN",
                    help="tile RUN and its ancestors, oldest first")
    # THE SHEET MUST BE ABLE TO ACCUSE THE RUN. Cedric spotted `r004_12` and `r004_13` as an
    # identical pair by eye, in a 53-tile montage, and they were: same `traj.npz` to the byte from
    # two specs differing in `vth_frac`. A human found in one glance what no check was looking for,
    # and the campaign had already spent 6 of 50 runs on three such clusters. So the sheet computes
    # it now -- runs whose trajectory is bit-identical are grouped adjacent and labelled with the
    # reason they collided, which is not the same reason each time.
    ap.add_argument("--identical", action="store_true",
                    help="group runs with a bit-identical traj.npz and label why they collided")
    a = ap.parse_args()

    if a.lineage:
        runs = _ancestry(a.lineage)
        missing = [r for r in runs if not os.path.exists(os.path.join(LOG, r, a.image))]
        if missing:
            print(f"  no {a.image} for: {', '.join(missing)}")
        runs = [r for r in runs if r not in missing]
    else:
        runs = sorted(d for d in os.listdir(LOG)
                      if not d.startswith("_") and fnmatch(d, a.glob)
                      and os.path.exists(os.path.join(LOG, d, a.image)))
    if not runs:
        print(f"no run matching {a.glob!r} has a {a.image}"); return 1

    if a.identical and not a.lineage:
        clusters, why = _identical(runs)
        # duplicates first and adjacent, then everything else -- the sheet is FOR the collisions
        order = [r for c in clusters for r in c] + [r for r in runs if not any(r in c for c in clusters)]
        runs = order
        cap = {r: why.get(r, "") for r in runs}
        n_dup = sum(len(c) for c in clusters)
        print(f"{n_dup} of {len(runs)} runs share a trajectory with another, in {len(clusters)} "
              f"cluster(s)" if clusters else "no two runs share a trajectory")
    elif a.by and not a.lineage:
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
            with Image.open(os.path.join(LOG, r, a.image)) as im:
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
