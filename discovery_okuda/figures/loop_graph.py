#!/usr/bin/env python
"""loop_graph -- what ROLES.md PROMISES against what round.py / campaign_loop.py ACTUALLY WIRE.

Drawn by an independent audit on 2026-08-03. Every edge below was traced to a call site or to a
file read; nothing here is taken from a docstring. The classification is:

    (a) declared in ROLES.md AND wired in code       teal,  solid
    (b) declared in ROLES.md and NOT wired           red,   dashed + x   -- a promise nobody keeps
    (c) wired in code and NOT declared in ROLES.md   amber, dotted       -- undocumented flow

Artefacts drawn in red are WRITTEN by the round and READ BY NOTHING.
The bundle along the bottom and left margin is the return path into the next round's Proposer;
drawing it as one channel is the point -- it is the only route by which anything a round learns
reaches the round after it.

    python loop_graph.py        # rewrites loop_graph.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "loop_graph.png")

BG, FG = "#080808", "#f2f2f2"
AGENT, CHECK, ART, DEAD = "#7aa2f7", "#9ece6a", "#8f8f8f", "#ff5f56"
CA, CB, CC = "#2fd6ab", "#ff5f56", "#ffb31f"

# ------------------------------------------------------------- nodes: name -> (x, y, kind)
N = {
    # ------------------------------- ACT 1
    "Grounder":                   (2.2, 11.05, "agent"),
    "Proposer":                   (6.1, 11.05, "agent"),
    "Peer-review":                (10.0, 11.05, "agent"),
    "Critic":                     (13.9, 11.05, "check"),
    "Biologist\nstatic + probe":  (18.2, 11.05, "check"),
    "frontier.json":              (2.2, 9.75, "art"),
    "proposal.json":              (6.1, 9.75, "art"),
    "peer_review.jsonl":         (10.0, 9.75, "art"),
    "batch_refusals\n.jsonl":    (13.9, 9.75, "art"),
    # ------------------------------- ACT 2
    "cluster + run_one\n(the engine)": (2.2, 7.85, "engine"),
    "Biologist\npassive":         (5.9, 7.85, "check"),
    "Reader":                     (9.6, 7.85, "agent"),
    "Eye-check":                 (13.0, 7.85, "agent"),
    "Metrologist":               (16.3, 7.85, "check"),
    "Diagnostician":             (20.0, 7.85, "agent"),
    "spec_run.yaml\ncomposition.json": (2.2, 6.55, "art"),
    "diag.json\nmetrics.json":    (5.9, 6.55, "art"),
    "description.txt\n(VLM caption)": (11.3, 6.55, "art"),
    "diagnoses.jsonl":           (20.0, 6.55, "deadart"),
    "Collector":                 (11.3, 5.05, "check"),
    # ------------------------------- ACT 3 + artefacts of the record
    "round_records\n.jsonl":      (5.6, 3.55, "art"),
    "analysis.md":                (8.9, 3.55, "art"),
    "hypotheses.jsonl":          (12.2, 3.55, "art"),
    "logic_report.jsonl":        (15.4, 3.55, "deadart"),
    "holes.jsonl":               (18.3, 3.55, "deadart"),
    "operator_requests\n.jsonl": (21.4, 3.55, "deadart"),
    "Interpreter":                (4.4, 1.95, "agent"),
    "Meta-review":                (8.5, 1.95, "agent"),
    "Supervisor":                (12.6, 1.95, "check"),
    "Archivist":                 (16.7, 1.95, "agent"),
    "campaign_loop\n(driver)":   (20.6, 1.95, "check"),
    "causal_descriptions\n.md":   (3.5, 0.70, "deadart"),
    "memory.md":                  (6.3, 0.70, "art"),
    "instruction.md":             (8.9, 0.70, "art"),
    "state.json":                (12.0, 0.70, "art"),
    "archivist.jsonl":           (15.6, 0.70, "deadart"),
    "every OTHER agent's\nprompt": (19.4, 0.70, "art"),
}

W = {"agent": 1.75, "check": 1.75, "engine": 1.85, "art": 1.72, "deadart": 1.78}
H = {"agent": 0.52, "check": 0.52, "engine": 0.52, "art": 0.46, "deadart": 0.46}

# ------------------------------ edges: (src, dst, class, label, rad, tpos)
# `rad` None  => routed through the return channel (bottom lane + left margin) into Act 1.
E = [
    # ================= (a) DECLARED IN ROLES.md AND WIRED =================
    ("Grounder", "Proposer", "a", "Okuda setup:\nn_cells + quote", 0.0, 0.52),
    ("Proposer", "proposal.json", "a", "", 0.0, 0.5),
    ("proposal.json", "Critic", "a", "graphs to admit", -0.16, 0.55),
    ("Critic", "batch_refusals\n.jsonl", "a", "", 0.0, 0.5),
    ("batch_refusals\n.jsonl", "Proposer", "a", "refusal reasons\n(next round)", 0.20, 0.5),
    ("Proposer", "Peer-review", "a", "slots (edit stripped)", 0.0, 0.5),
    ("Peer-review", "peer_review.jsonl", "a", "", 0.0, 0.5),
    ("peer_review.jsonl", "Proposer", "a", "last batch's issues", 0.16, 0.5),
    ("Proposer", "cluster + run_one\n(the engine)", "a", "the batch", -0.18, 0.5),
    ("cluster + run_one\n(the engine)", "Biologist\npassive", "a", "series", 0.0, 0.5),
    ("Biologist\npassive", "diag.json\nmetrics.json", "a", "premises\npremises_broken", 0.0, 0.5),
    ("diag.json\nmetrics.json", "Reader", "a", "premise brief\n+ endpoints", -0.14, 0.55),
    ("diag.json\nmetrics.json", "Collector", "a", "specimen verdict", -0.10, 0.5),
    ("Reader", "Collector", "a", "phenotype", -0.18, 0.62),
    ("Eye-check", "Collector", "a", "watcher verdict", 0.20, 0.60),
    ("Collector", "round_records\n.jsonl", "a", "", 0.0, 0.5),
    ("Collector", "analysis.md", "a", "", 0.0, 0.5),
    ("analysis.md", "Meta-review", "a", "STALE: written\nAFTER Meta-review ran", 0.26, 0.62),
    ("round_records\n.jsonl", "Archivist", "a", "the history — missing the round being decided", -0.30, 0.50),
    ("Supervisor", "state.json", "a", "", 0.0, 0.5),
    ("Meta-review", "memory.md", "a", "", 0.0, 0.5),
    ("Meta-review", "instruction.md", "a", "LEARNED PATTERNS", 0.0, 0.5),
    # ---- the return channel into the next round's Proposer
    ("state.json", "Proposer", "a", "the Supervisor's steer: mix_why", None, 9.2),
    ("memory.md", "Proposer", "a", "memory.md — a path in the prompt, never parsed", None, 3.6),
    ("instruction.md", "Proposer", "a", "instruction.md — the prompt write-back", None, 13.4),
    ("Archivist", "Proposer", "a", "the branch table: evidence / sound per branch", None, 17.6),

    # ================= (b) DECLARED IN ROLES.md, NO CODE PATH =================
    ("Peer-review", "Critic", "b", "the Critic ALREADY RAN\n(admit @ round.py:798,\nreflect @ round.py:837)", 0.30, 0.5),
    ("Critic", "Biologist\nstatic + probe", "b", "no Act-1 Biologist call site", 0.0, 0.5),
    ("Biologist\nstatic + probe", "Reader", "b", "static runs on the CLUSTER (run_one:209);\nthe probe never runs at all", 0.26, 0.5),
    ("Biologist\nstatic + probe", "Collector", "b", "", 0.36, 0.5),
    ("Metrologist", "Reader", "b", "the admissible list is a\nliteral in predict.py", 0.42, 0.20),
    ("Metrologist", "Collector", "b", "no metrologist field\nin the record", 0.24, 0.66),
    ("Diagnostician", "Critic", "b", "guard_to_add never\nreaches critic.py", -0.24, 0.5),
    ("Diagnostician", "Supervisor", "b", "", 0.30, 0.5),
    ("Diagnostician", "Proposer", "b", "", 0.42, 0.72),
    ("Collector", "Interpreter", "b", "the record is built,\nnot passed", 0.18, 0.90),
    ("Collector", "Supervisor", "b", "observe() ran FIRST,\non rows not the record", -0.44, 0.84),
    ("Interpreter", "Meta-review", "b", "causal_descriptions.md is\nnot in the prompt", 0.0, 0.5),
    ("Interpreter", "Supervisor", "b", "", -0.30, 0.5),
    ("Meta-review", "Supervisor", "b", "", 0.24, 0.5),
    ("Meta-review", "every OTHER agent's\nprompt", "b", "only the Proposer is told\ninstruction.md exists", -0.14, 0.5),
    ("Archivist", "Supervisor", "b", "continue / roll_back / stop\nis read by NOTHING", 0.0, 0.5),

    # ================= (c) WIRED IN CODE, NOT DECLARED =================
    ("cluster + run_one\n(the engine)", "spec_run.yaml\ncomposition.json", "c", "", 0.0, 0.5),
    ("spec_run.yaml\ncomposition.json", "Archivist", "c", "cold start: what to breed from", -0.22, 0.14),
    ("Archivist", "frontier.json", "c", "cold_start() picks the frontier", None, 20.4),
    ("frontier.json", "Proposer", "c", "parents", 0.0, 0.5),
    ("description.txt\n(VLM caption)", "Reader", "c", "", -0.22, 0.5),
    ("description.txt\n(VLM caption)", "Eye-check", "c", "the movie, in words", 0.22, 0.28),
    ("cluster + run_one\n(the engine)", "Critic", "c", "post-hoc: inert ops /\nsaturation / divergence", -0.34, 0.24),
    ("Critic", "Collector", "c", "post-hoc refusals\n+ reasons", 0.34, 0.88),
    ("Reader", "Interpreter", "c", "analyst_consensus\n+ agreement", 0.26, 0.86),
    ("Biologist\npassive", "Meta-review", "c", "round_tally", 0.24, 0.58),
    ("Supervisor", "Collector", "c", "steer = mix_why", 0.28, 0.28),
    ("Archivist", "Collector", "c", "decision, recorded unread", 0.30, 0.30),
    ("Diagnostician", "Collector", "c", "diagnosis, STOP path ONLY", 0.26, 0.36),
    ("Diagnostician", "diagnoses.jsonl", "c", "", 0.0, 0.5),
    ("Collector", "hypotheses.jsonl", "c", "outcome / surprise", 0.0, 0.60),
    ("Collector", "logic_report.jsonl", "c", "logic.check_file", -0.10, 0.58),
    ("Collector", "holes.jsonl", "c", "note_hole", -0.16, 0.62),
    ("Collector", "operator_requests\n.jsonl", "c", "unmeasured property\n→ instrument request", -0.22, 0.40),
    ("Collector", "Meta-review", "c", "", -0.20, 0.5),
    ("Interpreter", "causal_descriptions\n.md", "c", "", 0.0, 0.5),
    ("Archivist", "archivist.jsonl", "c", "", 0.0, 0.5),
    ("hypotheses.jsonl", "Proposer", "c", "the refused-run summary", None, 6.2),
    ("round_records\n.jsonl", "campaign_loop\n(driver)", "c", "_aborts_in_a_row", -0.24, 0.5),
]

LANE_Y = -0.32           # the return bundle's horizontal lane, below everything
LANE_X = -0.92           # ... and its vertical leg, left of the act bands


def _box(ax, name):
    x, y, k = N[name]
    w, h = W[k], H[k]
    ec = {"agent": AGENT, "check": CHECK, "engine": CHECK, "art": ART, "deadart": DEAD}[k]
    fc = {"agent": "#101728", "check": "#111a0f", "engine": "#111a0f",
          "art": "#151515", "deadart": "#26100f"}[k]
    style = "round,pad=0.06" if k == "agent" else "square,pad=0.05"
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                linewidth=2.1 if k == "deadart" else 1.4,
                                edgecolor=ec, facecolor=fc, zorder=20))
    role = k in ("agent", "check", "engine")
    ax.text(x, y, name, ha="center", va="center", color=FG, fontsize=8.0 if role else 6.9,
            zorder=21, fontweight="bold" if role else "normal", linespacing=1.12)
    if k == "deadart":
        ax.text(x, y - h / 2 - 0.16, "written · no reader", ha="center", va="center",
                color=DEAD, fontsize=6.0, zorder=21, style="italic")


def _edge_pts(a, b):
    ax_, ay, ak = N[a]
    bx, by, bk = N[b]
    dx, dy = bx - ax_, by - ay

    def hit(cx, cy, w, h, dx, dy):
        if dx == 0 and dy == 0:
            return cx, cy
        tx = (w / 2 + 0.06) / abs(dx) if dx else 1e9
        ty = (h / 2 + 0.06) / abs(dy) if dy else 1e9
        t = min(tx, ty)
        return cx + dx * t, cy + dy * t
    return hit(ax_, ay, W[ak], H[ak], dx, dy), hit(bx, by, W[bk], H[bk], -dx, -dy)


def _arrow_head(ax, p, q, col, zorder=15):
    """A small filled head at q, pointing along p->q."""
    import numpy as np
    d = np.array(q) - np.array(p)
    n = (d[0] ** 2 + d[1] ** 2) ** 0.5 or 1.0
    d = d / n
    perp = (-d[1], d[0])
    L, Wd = 0.20, 0.085
    tip = q
    b1 = (q[0] - d[0] * L + perp[0] * Wd, q[1] - d[1] * L + perp[1] * Wd)
    b2 = (q[0] - d[0] * L - perp[0] * Wd, q[1] - d[1] * L - perp[1] * Wd)
    ax.add_patch(plt.Polygon([tip, b1, b2], closed=True, color=col, zorder=zorder, lw=0))


def draw():
    fig, ax = plt.subplots(figsize=(25.5, 15.2), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-2.15, 23.2)
    ax.set_ylim(-1.55, 13.55)
    ax.axis("off")

    for y0, y1, lab, col in ((9.20, 11.85, "ACT 1  ·  PROPOSE   — before a second of compute is spent", "#161f31"),
                             (5.90, 8.55, "ACT 2  ·  MEASURE   — the batch runs", "#0f1d16"),
                             (1.35, 4.35, "ACT 3  ·  DECIDE  +  cross-run", "#1e1729")):
        ax.add_patch(Rectangle((-0.70, y0), 23.85, y1 - y0, facecolor=col, edgecolor="none",
                               zorder=0))
        ax.text(-0.52, y1 - 0.16, lab, ha="left", va="top", color="#8d99b8", fontsize=11.0,
                fontweight="bold", zorder=1)

    counts = {"a": 0, "b": 0, "c": 0}
    lane_i = [0]
    for a, b, cl, lab, rad, tpos in E:
        counts[cl] += 1
        col = {"a": CA, "b": CB, "c": CC}[cl]
        ls = {"a": "-", "b": (0, (5, 3)), "c": (0, (1.1, 2.0))}[cl]
        lw = {"a": 1.6, "b": 1.5, "c": 1.4}[cl]

        if rad is None:                       # ---- routed through the return channel
            i = lane_i[0]
            lane_i[0] += 1
            ly = LANE_Y - i * 0.175
            lx = LANE_X - i * 0.175
            sx, sy, sk = N[a]
            tx, ty, tk = N[b]
            p0 = (sx, sy - H[sk] / 2 - 0.06)
            # come up the left margin, then run in UNDER the target and stub up into its base
            ybot = ty - H[tk] / 2 - 0.05
            yin = ybot - 0.30 - i * 0.105
            xin = tx - W[tk] / 2 + 0.28 + i * 0.24
            verts = [p0, (sx, ly), (lx, ly), (lx, yin), (xin, yin), (xin, ybot)]
            ax.add_patch(PathPatch(Path(verts, [Path.MOVETO] + [Path.LINETO] * 5),
                                   fill=False, edgecolor=col, lw=lw, linestyle=ls, zorder=6))
            _arrow_head(ax, (xin, yin), (xin, ybot), col, zorder=7)
            if lab:
                ax.text(tpos, ly, lab, ha="center", va="center", color=col, fontsize=6.6,
                        zorder=9, bbox=dict(boxstyle="round,pad=0.16", facecolor=BG,
                                            edgecolor="none", alpha=0.94))
            continue

        p0, p1 = _edge_pts(a, b)
        # quadratic bezier with the same geometry matplotlib's arc3 uses
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        ctrl = (mx - dy * rad, my + dx * rad)
        ax.add_patch(PathPatch(Path([p0, ctrl, p1], [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                               fill=False, edgecolor=col, lw=lw, linestyle=ls, zorder=6,
                               alpha=0.95))
        _arrow_head(ax, ctrl, p1, col, zorder=7)
        # bezier point at t=tpos, and the x marker for a broken promise
        def bz(t):
            u = 1 - t
            return (u * u * p0[0] + 2 * u * t * ctrl[0] + t * t * p1[0],
                    u * u * p0[1] + 2 * u * t * ctrl[1] + t * t * p1[1])
        if cl == "b":
            hx, hy = bz(0.5)
            ax.plot([hx - 0.10, hx + 0.10], [hy - 0.10, hy + 0.10], color=CB, lw=2.0, zorder=10)
            ax.plot([hx - 0.10, hx + 0.10], [hy + 0.10, hy - 0.10], color=CB, lw=2.0, zorder=10)
        if lab:
            lxp, lyp = bz(tpos)
            ax.text(lxp, lyp, lab, ha="center", va="center", color=col, fontsize=6.2,
                    zorder=11, linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor=BG, edgecolor="none",
                              alpha=0.88))

    for n in N:
        _box(ax, n)

    # ---------------------------------------------------------------- header (its own strip)
    ax.text(-2.10, 13.45, "the Okuda discovery loop  ·  what ROLES.md promises, against what "
                          "round.py and campaign_loop.py actually wire",
            ha="left", va="top", color=FG, fontsize=14.5, fontweight="bold")
    leg = [(CA, "-", f"(a)  declared in ROLES.md  AND  wired in code", counts["a"]),
           (CB, (0, (5, 3)), f"(b)  declared in ROLES.md, NO code path  —  a promise nobody keeps", counts["b"]),
           (CC, (0, (1.1, 2.0)), f"(c)  wired in code, NOT declared  —  undocumented flow", counts["c"])]
    for i, (c, ls, t, n) in enumerate(leg):
        yy = 12.80 - i * 0.34
        ax.plot([-2.02, -0.78], [yy, yy], color=c, linestyle=ls, linewidth=2.0)
        ax.text(-0.55, yy, t, ha="left", va="center", color=c, fontsize=9.4)
        ax.text(8.15, yy, f"{n} edges", ha="right", va="center", color=c, fontsize=9.4,
                fontweight="bold")
    for i, (c, t) in enumerate([(AGENT, "LLM role"), (CHECK, "deterministic check / code"),
                                (ART, "artefact on disk"),
                                (DEAD, "artefact WRITTEN by the round and READ BY NOTHING")]):
        yy = 12.80 - i * 0.34 + (0.34 if i == 3 else 0)
        xx = 9.6
        yy = 12.80 - i * 0.30
        ax.add_patch(FancyBboxPatch((xx, yy - 0.075), 0.42, 0.15,
                                    boxstyle="square,pad=0.02", edgecolor=c,
                                    facecolor="#151515", linewidth=1.6))
        ax.text(xx + 0.62, yy, t, ha="left", va="center", color=c, fontsize=9.4)
    ax.text(23.15, 12.80, "the return channel (bottom + left margin) is the ONLY route by which\n"
                          "anything a round learns reaches the round after it",
            ha="right", va="top", color="#8d99b8", fontsize=9.0, linespacing=1.3, style="italic")

    n_role = sum(1 for v in N.values() if v[2] in ("agent", "check", "engine"))
    n_art = sum(1 for v in N.values() if v[2] == "art")
    n_dead = sum(1 for v in N.values() if v[2] == "deadart")
    ax.text(23.15, 11.95, f"{n_role} roles + checks   ·   {n_art} live artefacts   ·   "
                          f"{n_dead} write-only artefacts   ·   {sum(counts.values())} edges",
            ha="right", va="top", color="#8d99b8", fontsize=9.4)

    fig.savefig(OUT, facecolor=BG, bbox_inches="tight", pad_inches=0.22)
    print(f"wrote {OUT}")
    for k, name in (("a", "declared + wired      "), ("b", "declared, NOT wired   "),
                    ("c", "wired, NOT declared   ")):
        print(f"  ({k}) {name} {counts[k]:>3} edges")
    print(f"  nodes: {n_role} roles/checks, {n_art} artefacts, {n_dead} write-only artefacts, "
          f"{len(N)} total")


if __name__ == "__main__":
    draw()
