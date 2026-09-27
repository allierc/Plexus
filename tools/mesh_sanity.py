#!/usr/bin/env python
"""Geometry sanity of a mesh run, row by row: is every cell still a cell of the surface?

    PYTHONPATH=src:tools python tools/mesh_sanity.py tissue/exp14_neutral_r2_s1 [--every 5] [--all]

WHY IT EXISTS (exp 14, 2026-09-26). exp 3's growth auditor reads whole-tissue quantities -- the shell's
asphericity, the inverted fraction, the largest vertex jump -- and a single exploding cell moves none
of them until the whole shell has gone. On `exp14_neutral_s1` single cells reached 4,740 x the median
area and vertices flew to radius 116 against a shell of radius 6 at row 110, while the tissue-wide
numbers crossed their lines only later; the movie, whose camera frames the run's whole extent, showed
a dot. This checks every CELL and every VERTEX the cells use, on every sampled row.

THE LINES, each a multiple of a scale the same row supplies, so they hold on any tissue size:

    radial   the largest |r - median r| of a used vertex from the shell's centre, in median edge
             lengths: a vertex off the surface. Closed shells only (a sheet has no radius).
    edge     the longest edge over the median edge
    area     the largest cell over the median cell
    euler    V - E + F of the used vertices (2 on a closed shell, 1 on a disc)
    finite   no non-finite position
    orphans  vertices inside the live prefix that no cell uses (reported, not a wreck: bookkeeping)

CALIBRATED on `exp14_neutral_s3`, the intact seed of exp 14's first batch, 437 rows: worst radial 2.17,
edge 4.62, area 4.36, all during the opening burst of divisions. The lines are about twice those.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RADIAL = 4.0        # median edge lengths off the median radius
EDGE = 8.0          # x the median edge
AREA = 10.0         # x the median cell area


def row(T, t, closed=None):
    """The checks of one row: a dict of numbers and `bad`, the list of broken lines (empty = sane)."""
    es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
    P = np.asarray(T.pos(t), float)
    nF = T.nF(t)
    used = np.unique(np.concatenate([es, et])) if len(es) else np.zeros(0, np.int64)
    out = {"row": t, "cells": nF, "vertices": int(len(used)), "orphans": int(len(P) - len(used))}
    bad = []
    if not len(used):
        return {**out, "bad": ["no mesh"]}
    Pu = P[used]
    if not np.isfinite(Pu).all():
        return {**out, "bad": ["non-finite positions"]}
    chi = len(used) - len(es) // 2 + nF
    closed = (chi == 2) if closed is None else closed
    out["euler"] = int(chi)
    if chi != (2 if closed else 1):
        bad.append(f"Euler characteristic {chi}")
    L = np.linalg.norm(P[es] - P[et], axis=1)
    Lm = float(np.median(L)) or 1e-12
    out["edge"] = float(L.max() / Lm)
    fc = np.zeros((nF, 3))
    np.add.at(fc, ef, P[es])
    fc /= np.maximum(np.bincount(ef, minlength=nF), 1)[:, None]
    A = np.zeros(nF)
    np.add.at(A, ef, 0.5 * np.linalg.norm(np.cross(P[es] - fc[ef], P[et] - fc[ef]), axis=1))
    Am = float(np.median(A)) or 1e-12
    out["area"] = float(A.max() / Am)
    out["worst_cell"] = int(np.argmax(A))
    if closed:
        c = np.median(Pu, 0)
        r = np.linalg.norm(Pu - c, axis=1)
        out["radial"] = float(np.abs(r - np.median(r)).max() / Lm)
        if out["radial"] > RADIAL:
            bad.append(f"a vertex {out['radial']:.1f} edge lengths off the shell")
    if out["edge"] > EDGE:
        bad.append(f"an edge {out['edge']:.1f} x the median")
    if out["area"] > AREA:
        bad.append(f"a cell {out['area']:.1f} x the median area")
    return {**out, "bad": bad}


def scan(T, every=1):
    """Every sampled row's checks, and the first row that breaks a line (None if none does)."""
    rows = [row(T, t) for t in range(0, T.n_rows(), max(1, int(every)))]
    first = next((r for r in rows if r["bad"]), None)
    return rows, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="group/name under graphs_data, or a run directory")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="print every sampled row, not only the broken ones")
    a = ap.parse_args()
    from exp_measures.common import open_run
    T = open_run(a.run)
    rows, first = scan(T, a.every)
    for r in rows:
        if a.all or r["bad"]:
            print(f"row {r['row']:5d}  cells {r['cells']:5d}  orphans {r['orphans']:5d}  "
                  f"radial {r.get('radial', float('nan')):7.2f}  edge {r.get('edge', float('nan')):7.2f}  "
                  f"area {r.get('area', float('nan')):9.2f}  {'; '.join(r['bad']) or 'ok'}")
    worst = {k: max((r.get(k, 0) or 0) for r in rows) for k in ("radial", "edge", "area", "orphans")}
    print(f"{a.run}: {len(rows)} rows checked, {sum(bool(r['bad']) for r in rows)} broken; "
          f"first broken row {first['row'] if first else None}; worst radial {worst['radial']:.2f} "
          f"(line {RADIAL}), edge {worst['edge']:.2f} (line {EDGE}), area {worst['area']:.2f} (line {AREA}), "
          f"orphans {worst['orphans']}")


if __name__ == "__main__":
    main()
