#!/usr/bin/env python
"""07c, part one -- OWNERSHIP: the plaque belongs to a cell, and divides with it.

    python plaque_lineage.py            # the self-test, against the replay's own lineage

WHAT CHANGES AND WHAT DOES NOT. The plaque keeps binding barycentrically to a triangle of TISSUE
VERTICES -- that law is certified (G9--G13), it returns its reaction to both endpoints in one call, and
vertex ids are stable because the cache only ever appends them. What is added is a label: every plaque
carries the CELL it belongs to. The mechanics does not read that label; the bookkeeping does.

WHY A LABEL IS ENOUGH TO FIX WHAT 07a MEASURED. `plaque_seed` made one edge per live SHEET NODE, so
the set followed the membrane's mesh: 2,562 -> 40,962 in two jumps of up to +300%, while the tissue
divided smoothly to 4,069 cells, and plaques per cell fell 12.81 -> 6.42. With the cell as the owner
the count is a per-cell constant by construction, and it grows one division at a time.

THE DIVISION RULE, and each clause is a gate:

    mother m divides, daughter d appears     (from the lineage: m is the face whose `ndiv` rose,
                                              d is the appended face -- exact on 3,869 of 3,869)
    each of m's plaques goes to whichever     a plaque is a patch of membrane; it ends up on the
    of m, d its attachment point is nearer    daughter that inherited that piece of the mother
    its bond number N_b travels WITH IT       G75: division must SPLIT the pool, never duplicate it.
                                              A plaque that moves takes its own bonds; nothing is
                                              created, so the total is unchanged by the division
    each of m, d is topped up to N0           G70: plaques per cell holds. A plaque added here is
                                              NEW -- it starts at the current geometry with no
                                              bonds, and the receptor supply fills it over tau_i
                                              (G73: born at rest, not born stretched)

The last two are deliberately different: moving a plaque conserves, adding one does not, and the
distinction is what lets G74 and G75 disagree if the implementation is wrong.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM"))


class PlaqueOwner:
    """Which cell owns each plaque, and what a division does to that.

    Holds only the bookkeeping: `cell[p]` for every plaque, and the per-cell target `N0`. Positions,
    weights and bonds live where they already do; this decides which of them move.
    """

    def __init__(self, cell_of_plaque, n_cells, N0):
        self.cell = np.asarray(cell_of_plaque, np.int64).copy()
        self.n_cells = int(n_cells)
        self.N0 = float(N0)

    # -- the event ---------------------------------------------------------------------------
    def divide(self, mothers, daughters, attach, centroid):
        """Re-label the mothers' plaques between mother and daughter, by which centroid is nearer.

        `mothers` and `daughters` are the pairing the lineage gate certified: the face whose `ndiv`
        rose, and the face appended in the same step. Returns the plaques that MOVED, so the caller
        can assert that nothing was created -- a division re-labels, it does not seed.
        """
        moved = []
        for m, d in zip(np.asarray(mothers), np.asarray(daughters)):
            p = np.flatnonzero(self.cell == m)
            if not p.size:
                continue
            dm = np.linalg.norm(attach[p] - centroid[m], axis=1)
            dd = np.linalg.norm(attach[p] - centroid[d], axis=1)
            take = p[dd < dm]
            self.cell[take] = d
            moved.append(take)
        self.n_cells = int(max(self.n_cells, int(np.max(daughters)) + 1)) if len(daughters) else \
            self.n_cells
        return np.concatenate(moved) if moved else np.zeros(0, np.int64)

    def deficit(self):
        """How many plaques each cell is short of `N0`. This is what a top-up seeds, and it is the
        only place the count is allowed to GROW."""
        have = np.bincount(self.cell, minlength=self.n_cells)
        return np.maximum(np.round(self.N0).astype(int) - have, 0)

    def per_cell(self):
        have = np.bincount(self.cell, minlength=self.n_cells)
        return have


# =============================================================================================
def selftest():
    """Replay the cache's own divisions through the rule and check the three invariants.

    No rig, no GPU: the lineage is in the cache and the bookkeeping is arithmetic, so the rule can be
    wrong here and found out here, before it is wired into anything that takes seventeen minutes.
    """
    import glob
    import json
    z = np.load(sorted(glob.glob(os.path.join(LOG, "_tissue",
                                              "cellfix_B_new_f401_x4_*.npz")))[0], mmap_mode="r")
    frames = np.asarray(z["mesh_frames"])
    n = len(frames)

    def cents(j):
        pos = np.asarray(z[f"m{j}_pos"], float)
        ef, es = np.asarray(z[f"m{j}_E_face"]), np.asarray(z[f"m{j}_E_srce"])
        nF = int(z[f"m{j}_nF"])
        live = ef < nF
        C = np.zeros((nF, 3)); c = np.zeros(nF)
        np.add.at(C, ef[live], pos[es[live]])
        np.add.at(c, ef[live], 1.0)
        return C / np.maximum(c, 1.0)[:, None], nF

    N0 = 12.0
    C0, nF0 = cents(0)
    own = PlaqueOwner(np.repeat(np.arange(nF0), int(N0)), nF0, N0)
    # every plaque starts at its cell's centroid, jittered inside the cell so the two daughters can
    # be told apart; the jitter is deterministic
    rng = np.random.default_rng(0)
    r = float(np.median(np.linalg.norm(C0 - C0.mean(0), axis=1))) * 0.15
    att = C0[own.cell] + rng.normal(0.0, 0.35 * r, size=(own.cell.size, 3))
    nb = np.full(own.cell.size, 1.0)                     # one unit of integrin per plaque
    tot0, n0 = float(nb.sum()), own.cell.size
    S = {k: [] for k in ("t", "cells", "plaques", "ppc", "moved", "seeded", "receptor_total")}
    nd_prev = np.asarray(z["m0_ndiv"], float)
    for j in range(1, n):
        C1, nF1 = cents(j)
        nd = np.asarray(z[f"m{j}_ndiv"], float)
        kk = min(len(nd_prev), len(nd))
        moth = np.flatnonzero(nd[:kk] - nd_prev[:kk] > 0)
        daug = np.arange(nF0, nF1)
        if len(moth) != len(daug):                       # the lineage gate says this cannot happen
            raise SystemExit(f"[07c] step {j}: {len(moth)} mothers but {len(daug)} daughters")
        own.n_cells = nF1
        mv = own.divide(moth, daug, att, C1) if len(moth) else np.zeros(0, np.int64)
        # top up: a NEW plaque, at the current geometry, with no bonds
        d = own.deficit()
        need = np.repeat(np.arange(nF1), d)
        if need.size:
            add = C1[need] + rng.normal(0.0, 0.35 * r, size=(need.size, 3))
            own.cell = np.concatenate([own.cell, need])
            att = np.concatenate([att, add])
            nb = np.concatenate([nb, np.zeros(need.size)])
        S["t"].append(int(frames[j])); S["cells"].append(int(nF1))
        S["plaques"].append(int(own.cell.size))
        S["ppc"].append(float(own.cell.size / nF1))
        S["moved"].append(int(mv.size)); S["seeded"].append(int(need.size))
        S["receptor_total"].append(float(nb.sum()))
        C0, nF0, nd_prev = C1, nF1, nd

    ppc = np.asarray(S["ppc"])
    step = np.abs(np.diff(np.asarray(S["plaques"], float))) / np.maximum(
        np.asarray(S["plaques"], float)[:-1], 1.0)
    res = {
        "N0": N0, "cells": [int(nF0), int(S["cells"][-1])],
        "plaques": [n0, int(S["plaques"][-1])],
        "G70 plaques per cell holds": {
            "seeded": float(ppc[0]), "final": float(ppc[-1]),
            "worst_deviation": float(np.max(np.abs(ppc / ppc[0] - 1.0))),
            "threshold": 0.10, "pass": bool(np.max(np.abs(ppc / ppc[0] - 1.0)) <= 0.10)},
        "G71 the count grows smoothly": {
            "max_step": float(step.max()), "threshold": 0.05,
            "steps_that_move": int((step > 1e-12).sum()), "of": int(step.size),
            "pass": bool(step.max() <= 0.05)},
        "G75 division splits, never creates": {
            "receptor total": [tot0, float(S['receptor_total'][-1])],
            "moved by division": int(np.sum(S["moved"])),
            "seeded empty": int(np.sum(S["seeded"])),
            "note": "the total may only rise through a SEEDED plaque, and a seeded plaque starts "
                    "at zero bonds -- so with no supply term the total must not move at all",
            "pass": bool(abs(float(S["receptor_total"][-1]) - tot0) < 1e-9)},
        "series": S,
    }
    out = os.path.join(LOG, "07c_cell_plaque")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "ownership_selftest.json"), "w"), indent=1)
    g = res["G70 plaques per cell holds"]; h = res["G71 the count grows smoothly"]
    k = res["G75 division splits, never creates"]
    print(f"[07c] cells {res['cells'][0]} -> {res['cells'][1]}, plaques {res['plaques'][0]} -> "
          f"{res['plaques'][1]}", flush=True)
    print(f"[07c] G70 {'PASS' if g['pass'] else 'FAIL'} (ppc {g['seeded']:.2f} -> {g['final']:.2f}, "
          f"worst {100*g['worst_deviation']:.2f}%)  "
          f"G71 {'PASS' if h['pass'] else 'FAIL'} (max step {100*h['max_step']:.2f}%, moves on "
          f"{h['steps_that_move']} of {h['of']})  "
          f"G75 {'PASS' if k['pass'] else 'FAIL'} (total {k['receptor total'][0]:.1f} -> "
          f"{k['receptor total'][1]:.1f}, {k['moved by division']} moved, {k['seeded empty']} "
          f"seeded) -> {out}", flush=True)


if __name__ == "__main__":
    selftest()
