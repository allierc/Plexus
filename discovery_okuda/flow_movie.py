#!/usr/bin/env python
"""Where information is made in this loop, where it flows, and where it dies. One mp4 per round.

CEDRIC, 14 AUGUST: *"make a small mp4 video to show where information is generated and where it
flows (left to right) to see where are the blockers... a small/big green ball from an agent that
goes to a file or another agent, information sums up on nodes and should move left to right on
edges, the edges becomes green."*

WHY A PICTURE AND NOT A CHECK. `round.load_flow` already refuses a graph where something is emitted
and nothing names it -- and that check has passed on every round of this campaign while the loop
lost its entire inductive output. The Analyst wrote seven claims into `analysis.md` for thirteen
rounds; the `analyst -> claims_update` edge was well-formed and EMPTY. A topology check cannot see
that by construction: it proves the pipe is connected, never that water went through it.

So this measures the water. A ball's size is how much a node emitted; an edge lights when something
crossed it and stays dark when nothing did. A dark edge in a well-formed graph is the defect class
this project has now hit six times.

TWO SOURCES, AND THE PICTURE SAYS WHICH IT USED.

    MEASURED      `campaign/flow_trace.jsonl`, written by `round._trace` since 14 August: the exact
                  size of what every node emitted, every round.
    RECONSTRUCTED  for rounds that ran before the trace existed, rebuilt from the artefacts those
                  rounds left -- records.jsonl, foresight.jsonl, claims.jsonl, analysis.md. It is
                  an approximation and every frame it produces is labelled RECONSTRUCTED, because a
                  reconstruction that looks like a measurement is worse than no picture at all.

    python flow_movie.py                    every round, plus the concatenation
    python flow_movie.py --round r012       one round
    python flow_movie.py --fps 12 --secs 3  slower
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.environ.get("OKUDA_LOG", os.path.join(ROOT, "log", "okuda"))
CAMPAIGN = os.path.join(HERE, "campaign")
OUT = os.path.join(LOG, "_gates", "flow")
for _p in (HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

W, H = 1600, 896      # both divisible by 8: ffmpeg silently resizes otherwise, and a
                      # resized frame is a picture nobody chose
NODE_R, BALL_MAX = 26, 15
BG, FG, DIM = (8, 8, 12), (238, 238, 242), (120, 120, 132)
LIVE, DEAD, PEND = (70, 210, 120), (150, 46, 46), (52, 52, 62)
UNK = (96, 96, 112)     # measurable only from a trace: not empty, not proven to carry


def graph():
    """(nodes in topological order, edges) from crew/flow.yaml -- the design, not a copy of it."""
    import round as R
    order = R.load_flow(R.FLOW)
    emits = {n.get("out", n["id"]): n["id"] for n in order}
    edges = []
    for n in order:
        for d in (n.get("in") or []):
            if d in emits:
                edges.append((emits[d], n["id"], d))
        if n.get("each") and n["each"] in emits:
            edges.append((emits[n["each"]], n["id"], n["each"]))
    return order, edges


def layout(order, edges):
    """x = topological depth, y = spread within the depth. Left to right, as asked."""
    dep = {n["id"]: 0 for n in order}
    for _ in range(len(order)):
        for a, b, _k in edges:
            dep[b] = max(dep[b], dep[a] + 1)
    cols = {}
    for n in order:
        cols.setdefault(dep[n["id"]], []).append(n["id"])
    pos, nx = {}, max(cols) + 1
    for d, ids in cols.items():
        for i, nid in enumerate(ids):
            x = 90 + d * (W - 200) / max(nx - 1, 1)
            y = 90 + (i + 0.5) * (H - 240) / max(len(ids), 1)
            pos[nid] = (x, y)
    return pos, dep


def measured(rid):
    """{node: (chars, empty)} from the trace, or None if that round predates it."""
    p = os.path.join(CAMPAIGN, "flow_trace.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("round") == rid:
            out[d["node"]] = (int(d.get("chars", 0)), bool(d.get("empty")))
    return out or None


def reconstruct(rid):
    """{node: (chars, empty)} rebuilt from what the round left on disk. AN APPROXIMATION.

    Each entry says what it is standing in for, because a reader who cannot tell a measured size
    from a plausible one will trust both equally.
    """
    v = {}

    def put(node, chars):
        v[node] = (int(chars), not chars)

    runs = [json.loads(l) for l in open(os.path.join(CAMPAIGN, "records.jsonl"))] \
        if os.path.exists(os.path.join(CAMPAIGN, "records.jsonl")) else []
    mine = [r for r in runs if str(r.get("round")) == rid]
    put("record", sum(len(json.dumps(r, default=str)) for r in mine))
    put("build", sum(len(json.dumps(r.get("edit"), default=str)) for r in mine))
    put("launch", 8 * len(mine))
    put("measure", sum(len(json.dumps(r.get("metrics") or {}, default=str)) for r in mine))
    put("score", sum(len(json.dumps(r.get("scored") or {}, default=str)) for r in mine))
    put("proposer", sum(len(json.dumps({k: r.get(k) for k in ("edit", "why", "act", "on")},
                                       default=str)) for r in mine))

    fs = os.path.join(CAMPAIGN, "foresight.jsonl")
    if os.path.exists(fs):
        for line in open(fs):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("round") != rid:
                continue
            put("eye", sum(len(json.dumps(r.get("observed") or {})) for r in d["runs"].values()))
            put("forecaster", sum(len(json.dumps(r.get("forecast") or {}))
                                  for r in d["runs"].values()))
            put("foresight", len(json.dumps(d, default=str)))

    cl = os.path.join(CAMPAIGN, "claims.jsonl")
    if os.path.exists(cl):
        n = sum(1 for l in cl and open(cl) if f'"round": "{rid}"' in l)
        put("claims_update", 400 * n)

    am = os.path.join(CAMPAIGN, "analysis.md")
    if os.path.exists(am):
        t = open(am, errors="ignore").read()
        m = re.search(rf"#+[^\n]*{rid}(.*?)(?=\n#+ |\Z)", t, re.S)
        put("analyst", len(m.group(1)) if m else 0)

    # UNMEASURABLE IS NOT EMPTY, and defaulting these to 0 made the movie lie in exactly the way
    # this file's own docstring warns about. These nodes leave no artefact a later reader can size --
    # `menu`, `coverage`, `history` and the rest are computed, handed to a role, and discarded -- so
    # a reconstruction CANNOT know what they emitted. Marked None, drawn grey, and counted as
    # UNMEASURED rather than as a dead edge. The first version reported "27 of 56 edges carried
    # nothing" identically on all thirteen rounds, which is the signature of an artefact, not a
    # finding: a real blockage would vary with what the round did.
    for n in ("parents", "menu", "coverage", "diagnosis", "history", "claim_ledger",
              "metric_floors", "metric_bank", "grounding", "user_input", "refusals",
              "route_a", "control", "observations", "morphology", "route_a_results",
              "grounder", "planned"):
        v.setdefault(n, None)
    return v


def rounds():
    ids = set()
    for p, key in ((os.path.join(CAMPAIGN, "flow_trace.jsonl"), "round"),
                   (os.path.join(CAMPAIGN, "foresight.jsonl"), "round")):
        if os.path.exists(p):
            for line in open(p):
                try:
                    ids.add(json.loads(line)[key])
                except Exception:
                    pass
    if os.path.exists(os.path.join(CAMPAIGN, "records.jsonl")):
        for line in open(os.path.join(CAMPAIGN, "records.jsonl")):
            try:
                ids.add(str(json.loads(line).get("round")))
            except Exception:
                pass
    return sorted(i for i in ids if re.fullmatch(r"r\d{3}", str(i or "")))


def draw(rid, vol, src, order, edges, pos, dep, fps, secs):
    """-> [PIL frames]. Depth by depth, so the eye reads it left to right."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
        fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        f = fb = fsm = ImageFont.load_default()

    top = max(1, max((c for c, _e in (v for v in vol.values() if v)), default=1))
    def rad(c):
        return 3 + BALL_MAX * (c / top) ** 0.45

    maxd = max(dep.values())
    per = max(2, int(fps * secs / max(maxd, 1)))
    frames, acc, lit = [], {}, {}
    for d in range(maxd + 1):
        step = [(a, b, k) for a, b, k in edges if dep[a] == d]
        for t in range(per):
            u = (t + 1) / per
            im = Image.new("RGB", (W, H), BG)
            dr = ImageDraw.Draw(im)
            dr.text((28, 22), f"{rid}", fill=FG, font=fb)
            dr.text((28, 50), f"{'MEASURED from flow_trace.jsonl' if src == 'measured' else 'RECONSTRUCTED from artefacts -- an approximation'}",
                    fill=(90, 200, 120) if src == "measured" else (215, 150, 60), font=fsm)
            for a, b, k in edges:
                (x0, y0), (x1, y1) = pos[a], pos[b]
                on = lit.get((a, b))
                col = PEND if on is None else (LIVE if on is True else
                                              UNK if on == "?" else DEAD)
                dr.line([(x0 + NODE_R, y0), (x1 - NODE_R, y1)], fill=col,
                        width=3 if on is True else 1)
            for nid, (x, y) in pos.items():
                vv = vol.get(nid)
                c, empty = (0, True) if vv is None else vv
                got = acc.get(nid, 0)
                node = next(n for n in order if n["id"] == nid)
                ring = (95, 95, 110) if node.get("agent") is None else (150, 130, 220)
                dr.ellipse([x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R], outline=ring, width=2)
                if got:
                    r = rad(got)
                    dr.ellipse([x - r, y - r, x + r, y + r], fill=LIVE if not empty else DEAD)
                lab = nid if len(nid) < 15 else nid[:14]
                dr.text((x - NODE_R, y + NODE_R + 4), lab, fill=FG if got else DIM, font=fsm)
                if vv is None:
                    dr.text((x - NODE_R, y + NODE_R + 16), "unmeasured", fill=UNK, font=fsm)
                elif c:
                    dr.text((x - NODE_R, y + NODE_R + 16),
                            f"{c/1000:.1f}k" if c >= 1000 else str(c), fill=DIM, font=fsm)
            for a, b, k in step:
                vv = vol.get(a)
                if vv is None:
                    continue
                c, empty = vv
                (x0, y0), (x1, y1) = pos[a], pos[b]
                bx, by = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
                r = rad(c) if c else 3
                dr.ellipse([bx - r, by - r, bx + r, by + r], fill=LIVE if c else DEAD)
            if t == per - 1:
                for a, b, k in step:
                    vv = vol.get(a)
                    c = 0 if vv is None else vv[0]
                    lit[(a, b)] = "?" if vv is None else bool(c)
                    if c:
                        acc[b] = acc.get(b, 0) + c
                    acc.setdefault(a, (vol.get(a) or (0, 0))[0])
            dead = [f"{a}->{b}" for (a, b), on in lit.items() if on is False]
            unk = sum(1 for on in lit.values() if on == "?")
            dr.text((28, H - 40), f"edges carrying nothing: {len(dead)}"
                    + (f"   unmeasured: {unk}" if unk else ""), fill=DEAD, font=f)
            if dead:
                dr.text((28, H - 22), ", ".join(dead[:6])[:150], fill=DIM, font=fsm)
            frames.append(im)
    frames += [frames[-1]] * int(fps * 0.8)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--secs", type=float, default=4.0)
    a = ap.parse_args()

    import imageio.v2 as imageio
    import numpy as np
    order, edges = graph()
    pos, dep = layout(order, edges)
    rids = [a.round] if a.round else rounds()
    if not rids:
        print("no rounds on disk")
        return 1
    os.makedirs(OUT, exist_ok=True)
    made, allf = [], []
    for rid in rids:
        vol = measured(rid)
        src = "measured" if vol else "reconstructed"
        if not vol:
            vol = reconstruct(rid)
        fr = draw(rid, vol, src, order, edges, pos, dep, a.fps, a.secs)
        p = os.path.join(OUT, f"flow_{rid}.mp4")
        imageio.mimsave(p, [np.asarray(x) for x in fr], fps=a.fps, quality=8,
                        macro_block_size=8)
        made.append(p)
        allf += fr
        dead = sum(1 for (x, y, k) in edges if vol.get(x) is not None and not vol[x][0])
        unk = sum(1 for (x, y, k) in edges if vol.get(x) is None)
        print(f"  {rid}  {src:14s} {len(fr):4d} frames   "
              f"{dead} edge(s) carried nothing, {unk} unmeasured, {len(edges)-dead-unk} alive")
    if len(made) > 1:
        p = os.path.join(OUT, "flow_all.mp4")
        imageio.mimsave(p, [np.asarray(x) for x in allf], fps=a.fps, quality=8,
                        macro_block_size=8)
        made.append(p)
    print(f"\n{len(made)} file(s) -> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
