#!/usr/bin/env python
"""metrics -- one name, one quantity. The registry, not a new set of measurements.

PHASE 2. WHY THIS FILE EXISTS, AND WHY IT INVENTS NOTHING
================================================================================================
The measurements this campaign needs already exist. They were written over three weeks in
`prototype/cardio_mpm`, audited on 4 July, and the audit's own remedy survived. The failure was
never that the quantities were missing -- it is that **the same quantity acquired four names, in
four places, with four supports, and nothing said which was which.**

    loopscore_residual   size | orientation | openness/aspect | chirality | shape-detail(k>=2)
    enclosure_row        energy peak | area loopiness | chir_match | minor
    morphology_row       size | open | chir+                  (SIM-ONLY -- withdrawn, still printed)
    descriptors.py       magnitude_peak | opening_area | direction_chirality | orientation_rad ...

The fourth is mine, added in Phase 1, and it is the same defect one turn later: a second
vocabulary for quantities that already had names. This file is the correction. It does not define
new measurements; it **names the ones that exist, points each at the code that computes it, and
records what may be believed about it.**

THE READING SURFACE
------------------------------------------------------------------------------------------------
The campaign reads loops on a **10x10 grid of the tissue** -- `select_grid_nodes(10, 10, side=137,
margin=10)` -- and compares each node's trajectory with the recording's. That is the picture on
every dashboard and the thing a human actually judges.

It also carries a known defect, found by the 4 July audit and never fixed: **a margin of 10 nodes
is INSIDE the pinned band.** The band is 0.06 of the domain, which is 11.7 nodes of the 137 grid,
so the outer ring of the 10x10 selection sits on particles pinned to the answer -- 36 of the 100.
Numbers read there are the anchor, not the model. The grid is kept, because it is what the eye
reads; the margin is corrected, and both versions are reported so the change is visible.

WHAT A TIER MEANS
------------------------------------------------------------------------------------------------
  certified    it has a measured null, a measured noise floor, and it moved when it should and
               held still when it should not. Admissible evidence.
  provisional  it computes, and one of those is missing. Usable, never citable.
  withdrawn    a defect was demonstrated. Kept with its cause of death, never quoted, and
               **no definition here may name one** -- the check below refuses that.

Three checks, taken from the Okuda folder's Phase 10, where the same class of fault cost weeks:
every registered metric must have something that computes it; every one must have a written
definition; and no definition may name a withdrawn metric.

    python metrics.py --check      # the three checks
    python metrics.py --list       # the registry
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CERTIFIED, PROVISIONAL, WITHDRAWN = "certified", "provisional", "withdrawn"

# The reading surface, reconstructed from the prototype. `margin` is the number of lattice nodes
# skipped at each edge; the band occupies bwidth * 137 = 8.2 nodes at bwidth 0.06 measured from the
# sheet edge, and the recording's lattice and the sheet are the same square, so a margin below that
# lands inside the anchor.
GRID_SIDE = 137
GRID_N = 10
MARGIN_INHERITED = 10
MARGIN_SAFE = 20                       # outside the band with room to spare; see grid_report()


def select_grid_nodes(nx=GRID_N, ny=GRID_N, side=GRID_SIDE, margin=MARGIN_INHERITED):
    """The canonical 10x10 dashboard selection, exactly as the prototype defines it."""
    rows = np.linspace(margin, side - 1 - margin, ny).round().astype(int)
    cols = np.linspace(margin, side - 1 - margin, nx).round().astype(int)
    return (rows[:, None] * side + cols[None, :]).ravel()


def grid_report(bwidth=0.06):
    """How much of each grid selection sits inside the pinned band. The defect, as a number."""
    out = {}
    for name, margin in (("inherited", MARGIN_INHERITED), ("corrected", MARGIN_SAFE)):
        idx = select_grid_nodes(margin=margin)
        r, c = idx // GRID_SIDE, idx % GRID_SIDE
        u = np.stack([c, r], 1) / (GRID_SIDE - 1)                     # node -> unit square
        # ...and the unit square is mapped onto the SHEET, which spans 0.70 of the world, not 1.0.
        # The first version of this check omitted that factor and reported 0 nodes in the band,
        # against the 4 July audit's 36. The audit was right: the outer ring of a 10x10 grid is
        # 4*10-4 = 36 nodes, and at margin 10 it sits 0.052 from the edge against a band of 0.06.
        edge = 0.70 * np.minimum(u, 1 - u).min(1)                     # distance to the sheet edge
        inside = edge < bwidth
        out[name] = {"margin_nodes": margin, "n": int(idx.size),
                     "in_band": int(inside.sum()),
                     "min_edge_distance": float(edge.min()),
                     "band_width": bwidth}
    return out


# ---------------------------------------------------------------------------------------------
# THE REGISTRY. Each entry: what it is, in one sentence a person can check, and what computes it.
# `compute` takes (sim, real, mask) and returns a float, or None where the quantity is a whole
# decomposition rather than a scalar.
# ---------------------------------------------------------------------------------------------
def _harm():
    import harmonic_inherited as H
    return H


def _t(a, mask=None):
    import torch
    x = a[:, mask] if mask is not None else a
    return torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)


REGISTRY = {
    # ---- the objective ----------------------------------------------------------------------
    "loopscore": {
        "definition": "per node, the beat is a closed path; compare its low-order Fourier "
                      "description with the recording's, normalised by that node's own energy, "
                      "and average over nodes. 1 = the loops match.",
        "tier": PROVISIONAL,
        "source": "cardio_harmonic.harmonic_score (26 June, the objective shift)",
        "compute": lambda s, r, m: float(_harm().harmonic_score(_t(s, m), _t(r, m), None)),
        "null_measured": 0.0700,
        "domain": "one closed beat, >= 8 frames, on moving interior nodes",
        "known_defects": ["blind to coordination: randomise every node's timing and it returns "
                          "exactly 1.0000 (measured on all 10 corpus runs)",
                          "its zero is +0.070, not 0, and its own docstring says otherwise",
                          "the area term is weighted x3, hard-coded in a signature"],
    },
    "loopscore_sd": {
        "definition": "the spread of the per-node loopscore. Uniformly mediocre tissue and a few "
                      "excellent nodes among wrong ones score the same mean; this separates them.",
        "tier": PROVISIONAL,
        "source": "cardio_harmonic.harmonic_stats",
        "compute": lambda s, r, m: float(_harm().harmonic_stats(_t(s, m), _t(r, m), None)[1]),
        "domain": "as loopscore",
    },
    "interior_r2": {
        "definition": "frame-locked goodness of fit, pooled over nodes, with the temporal mean "
                      "removed from the recording only.",
        "tier": PROVISIONAL,
        "source": "cardio_harmonic.interior_r2 (23 June, the original objective)",
        "compute": lambda s, r, m: float(_harm().interior_r2(_t(s, m), _t(r, m), None)),
        "null_measured": -0.875,
        "domain": "any window; blind to whether the path is a loop at all",
        "known_defects": ["every one of the 324 archived fits scores below its own null"],
    },

    # ---- the five residual dimensions. THE campaign vocabulary; do not rename them. ----------
    "residual/size": {
        "definition": "how much loopscore is recovered by scaling the simulated loops to the "
                      "recording's size, leaving everything else alone.",
        "tier": PROVISIONAL, "source": "cardio_harmonic.loopscore_residual",
        "compute": None, "domain": "as loopscore"},
    "residual/orientation": {
        "definition": "how much is recovered by rotating the simulated loops onto the recording's "
                      "axis.",
        "tier": PROVISIONAL, "source": "cardio_harmonic.loopscore_residual",
        "compute": None, "domain": "as loopscore"},
    "residual/openness_aspect": {
        "definition": "how much is recovered by matching the aspect ratio -- how fat the loop is "
                      "against how long -- while keeping its size.",
        "tier": PROVISIONAL, "source": "cardio_harmonic.loopscore_residual",
        "compute": None, "domain": "as loopscore"},
    "residual/chirality": {
        "definition": "how much is recovered by making each loop circulate the same way round as "
                      "the recording's.",
        "tier": PROVISIONAL, "source": "cardio_harmonic.loopscore_residual",
        "compute": None, "domain": "as loopscore"},
    "residual/shape_detail": {
        "definition": "how much is recovered by taking the higher harmonics from the recording -- "
                      "everything the ellipse does not describe.",
        "tier": PROVISIONAL, "source": "cardio_harmonic.loopscore_residual",
        "compute": None, "domain": "as loopscore"},

    # ---- per-loop quantities, real-referenced (the 4 July audit's remedy) --------------------
    "openness": {
        "definition": "enclosed area divided by the area of the box the loop fits in. 0 is a "
                      "straight line; a circle is about 0.79.",
        "tier": PROVISIONAL,
        "source": "cardio_mpm_train._openness (audit remedy, 4-5 July)",
        "compute": lambda s, r, m: float(np.median(_openness_pernode(s, m))),
        "domain": "a closed path; undefined for a path of zero extent"},
    "path_length": {
        "definition": "how far a node travels over the beat, summed frame to frame. A loop and a "
                      "line of the same width differ here; the enclosed area alone does not say so.",
        "responds_to": ["size", "openness"],
        "note": "NOT an independent axis, and the battery is what showed it: flattening an ellipse "
                "onto its long axis changes the distance travelled (the perimeter goes from about "
                "pi*(a+b) to 4a), so path length mixes how big the loop is with how open it is. "
                "Useful, but it may not be quoted as a size measure on its own.",
        "tier": PROVISIONAL,
        "source": "this file -- the one quantity the prototype named and never computed",
        "compute": lambda s, r, m: float(np.median(_path_length(s, m))),
        "domain": "any window"},
    "peak_excursion": {
        "definition": "how far a node reaches from the CENTRE of its own path, at its furthest. "
                      "Centred, so sliding the loop somewhere else does not change it.",
        "tier": PROVISIONAL,
        "source": "cardio_mpm_train.enclosure_row (peak), centred here",
        "compute": lambda s, r, m: float(np.median(_peak_excursion(s, m))),
        "domain": "any window",
        "known_defects": ["the first version measured distance from the ORIGIN, so merely "
                          "translating a loop changed it -- the same uncentred defect the July "
                          "audit found in morphology_row/size. Caught by the certification "
                          "battery, which is what it is for."]},
    "chirality_match": {
        "definition": "the fraction of nodes circulating the same way round as the recording's.",
        "tier": PROVISIONAL,
        "source": "cardio_mpm_train.enclosure_row (chir_match)",
        "compute": lambda s, r, m: float((np.sign(_shoelace(s, m)) == np.sign(_shoelace(r, m))).mean()),
        "domain": "a closed path with non-zero area"},

    # ---- withdrawn. Kept with the cause of death; may never be quoted. -----------------------
    "morphology_row/size": {
        "definition": "the mean over the 10x10 grid of each node's largest displacement, "
                      "computed on the SIMULATION ALONE and uncentred.",
        "tier": WITHDRAWN,
        "source": "cardio_mpm_train.morphology_row (~24 June)",
        "compute": None, "domain": "none",
        "cause_of_death": "sim-only, so it cannot see the residual at all; and 36 of its 100 "
                          "nodes are pinned to the recording, so it largely reported the anchor. "
                          "The campaign read it as 'size is flat against every lever, therefore a "
                          "structural limit' and chased that for four rounds. (Audit, 4 July.)"},
    "morphology_row/open": {
        "definition": "openness over the 10x10 grid, on the simulation alone.",
        "tier": WITHDRAWN, "source": "cardio_mpm_train.morphology_row", "compute": None,
        "domain": "none", "cause_of_death": "same defect as morphology_row/size"},
    "morphology_row/chir": {
        "definition": "the fraction of grid nodes turning anticlockwise, on the simulation alone.",
        "tier": WITHDRAWN, "source": "cardio_mpm_train.morphology_row", "compute": None,
        "domain": "none", "cause_of_death": "same defect as morphology_row/size"},
    "ampL": {
        "definition": "the ratio of total motion energy, simulated against recorded, over the "
                      "whole field.",
        "tier": WITHDRAWN,
        "source": "cardio_mpm_train (23 June, Batch 1)",
        "compute": None, "domain": "none",
        "cause_of_death": "a GLOBAL ratio dominated by a few large-motion nodes. The best run on "
                          "record read 0.002 -- 'the cleanest ever' -- while the median node was "
                          "at 0.57 of the recorded amplitude. It was read as a shape score and is "
                          "not one. (Audit, 4 July.)"},
}


# ---------------------------------------------------------------------------------------------
def _sel(a, m):
    return a[:, m] if m is not None else a


def _shoelace(a, m=None):
    p = _sel(np.asarray(a, float), m)
    x, y = p[..., 0], p[..., 1]
    return 0.5 * (x * np.roll(y, -1, 0) - np.roll(x, -1, 0) * y).sum(0)


def _openness_pernode(a, m=None):
    p = _sel(np.asarray(a, float), m)
    box = np.ptp(p[..., 0], axis=0) * np.ptp(p[..., 1], axis=0)
    return np.abs(_shoelace(p)) / (box + 1e-12)


def _peak_excursion(a, m=None):
    """Furthest reach from the path's own centre. Translation-invariant by construction."""
    p = _sel(np.asarray(a, float), m)
    return np.linalg.norm(p - p.mean(axis=0, keepdims=True), axis=-1).max(0)


def _path_length(a, m=None):
    p = _sel(np.asarray(a, float), m)
    return np.linalg.norm(np.diff(p, axis=0), axis=-1).sum(0)


# ---------------------------------------------------------------------------------------------
# RECONCILIATION. `descriptors.py` (Phase 1) measures several of these under different names. The
# names here are canonical -- they are the campaign's, and they came first. The alias table exists
# so the two cannot silently diverge, and `check()` proves they agree NUMERICALLY rather than
# assuming it. Renaming without checking is how four vocabularies became four measurements.
# ---------------------------------------------------------------------------------------------
ALIASES = {                                   # descriptors.py name -> canonical registry name
    "opening_loopiness": "openness",
    "magnitude_peak": "peak_excursion",
    "direction_chirality": "chirality_match",
}


def check_aliases():
    """Do the two implementations of each shared quantity return the same number?"""
    import descriptors as DS
    rng = np.random.default_rng(3)
    G, M = 24, 60
    t = np.linspace(0, 2 * np.pi, G, endpoint=False)
    a, b = rng.uniform(0.5, 1.5, M), rng.uniform(0.2, 0.9, M)
    th = rng.uniform(0, np.pi, M)
    u, v = np.cos(t)[:, None] * a, np.sin(t)[:, None] * a * b
    real = np.stack([u * np.cos(th) - v * np.sin(th), u * np.sin(th) + v * np.cos(th)], -1)
    sim = real * 1.4
    res = DS.loop_residual(sim, real)
    rows = []
    for dname, cname in ALIASES.items():
        mine = res[dname]["sim"] if dname in res else None
        theirs = REGISTRY[cname]["compute"](sim, real, None)
        if dname == "direction_chirality":
            mine = res[dname]["sim"]
        ok = mine is not None and abs(float(mine) - float(theirs)) < 1e-6 * max(1.0, abs(theirs))
        rows.append({"descriptors": dname, "canonical": cname, "descriptors_value": float(mine),
                     "registry_value": float(theirs), "agree": bool(ok)})
    return rows


# ---------------------------------------------------------------------------------------------
# THE CERTIFICATION BATTERY. A metric is admitted when it MOVES on the axis it claims to measure
# and HOLDS STILL on the axes it claims to ignore. Both halves are required: a metric that always
# moves is not measuring anything in particular, and one that never moves is not measuring at all.
#
# The distortions are applied to the RECORDING, so the answer is known by construction. This is
# the third of the three things a tier needs; the null came from floors.py and the noise floor is
# still outstanding, which is why nothing here can reach `certified` yet.
# ---------------------------------------------------------------------------------------------
def _population(G=24, M=200, seed=0):
    """A population of ellipses standing in for the recording: random size, aspect, angle, phase."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, G, endpoint=False)
    a = rng.uniform(0.5, 1.5, M)
    b = a * rng.uniform(0.25, 0.9, M)
    th = rng.uniform(0, np.pi, M)
    ph = rng.integers(0, G, M)
    u, v = np.cos(t)[:, None] * a, np.sin(t)[:, None] * b
    p = np.stack([u * np.cos(th) - v * np.sin(th), u * np.sin(th) + v * np.cos(th)], -1)
    return np.stack([np.roll(p[:, j], int(ph[j]), 0) for j in range(M)], 1), th


def distortions(real, th):
    """(name, distorted, {axis: should_move}) -- what each distortion is supposed to touch."""
    G, M = real.shape[0], real.shape[1]
    rng = np.random.default_rng(1)
    out = []
    out.append(("identity", real.copy(), set()))
    out.append(("scale x2", real * 2.0, {"size"}))
    rot = np.pi / 5
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    out.append(("rotate by pi/5", real @ R.T, {"orientation"}))
    mir = real.copy(); mir[..., 1] *= -1
    out.append(("mirror", mir, {"chirality", "orientation"}))
    ca, sa = np.cos(th), np.sin(th)
    proj = real[..., 0] * ca[None] + real[..., 1] * sa[None]
    # Collapsing onto the major axis destroys the enclosed area and, with it, any sense of
    # rotation -- a straight line has no handedness. It does NOT shorten the reach along the axis
    # that survives, so the size axis is expected to HOLD. My first expectation had both of those
    # backwards, and the battery reported two disagreements that were mine, not the metrics'.
    out.append(("collapse to a line", np.stack([proj * ca[None], proj * sa[None]], -1),
                {"openness", "chirality"}))
    out.append(("shift the whole beat", np.stack([np.roll(real[:, j], 7, 0) for j in range(M)], 1),
                set()))
    out.append(("translate", real + np.array([3.0, -2.0]), set()))
    sc = np.stack([np.roll(real[:, j], int(rng.integers(0, G)), 0) for j in range(M)], 1)
    out.append(("scramble each node's timing", sc, {"coordination"}))
    return out


# which registry entry speaks for which axis
# Which axes each metric legitimately responds to. Declared per METRIC rather than per axis,
# because at least one of them is a composite and pretending otherwise is how a ruler acquires a
# reputation for lying. `responds_to` in the registry is the source of truth; this mirrors it.
METRIC_AXES = {
    "peak_excursion": {"size"},
    "path_length": {"size", "openness"},          # a composite -- see its registry note
    "openness": {"openness"},
    "chirality_match": {"chirality"},
    "orientation_error": {"orientation"},         # not implemented yet
    "coordination": {"coordination"},              # not implemented yet -- the known hole
}


def certify(verbose=True, tol=0.02):
    real, th = _population()
    rows = []
    for name, sim, should in distortions(real, th):
        rec = {"distortion": name, "should_move": sorted(should), "axes": {}}
        for mname, axes in METRIC_AXES.items():
            if mname not in REGISTRY or REGISTRY[mname].get("compute") is None:
                rec["axes"][mname] = None
                continue
            base = REGISTRY[mname]["compute"](real, real, None)
            got = REGISTRY[mname]["compute"](sim, real, None)
            rel = abs(got - base) / max(abs(base), 1e-12)
            moved = rel > tol
            expect = bool(axes & should)
            rec["axes"][mname] = {"base": base, "value": got, "rel_change": rel,
                                  "moved": bool(moved), "responds_to": sorted(axes),
                                  "expected_to_move": expect, "ok": bool(moved) == expect}
        rows.append(rec)
    if verbose:
        print(f"\n{'=' * 108}\n  CERTIFICATION -- does each metric move on its own axis and hold "
              f"still on the others?\n{'=' * 108}")
        names = list(METRIC_AXES)
        print(f"  {'distortion':<28s} " + " ".join(f"{m[:13]:>14s}" for m in names))
        for r in rows:
            cells = []
            for m in names:
                a = r["axes"].get(m)
                if a is None:
                    cells.append(f"{'--':>14s}")
                else:
                    cells.append(f"{('MOVE' if a['moved'] else 'hold') + ('' if a['ok'] else ' X!'):>14s}")
            print(f"  {r['distortion']:<28s} " + " ".join(cells))
        bad = [(r["distortion"], m) for r in rows for m, a in r["axes"].items()
               if a is not None and not a["ok"]]
        missing = sorted({m for r in rows for m, a in r["axes"].items() if a is None})
        print(f"\n  X! marks a metric that moved when it should not, or held when it should have.")
        if missing:
            print(f"  NOT IMPLEMENTED, so untestable: {missing}")
        print(f"  {len(bad)} disagreements" + (f": {bad[:6]}" if bad else ""))
        print("=" * 108)
    return rows


def admitted():
    """The only names a claim may cite. Empty until Phase 2 certifies something -- deliberately."""
    return {k: v for k, v in REGISTRY.items() if v["tier"] == CERTIFIED}


def withdrawn():
    return {k: v for k, v in REGISTRY.items() if v["tier"] == WITHDRAWN}


# ---------------------------------------------------------------------------------------------
# THE THREE CHECKS. Taken from the Okuda folder's Phase 10, where this class of fault cost weeks.
# ---------------------------------------------------------------------------------------------
def check(verbose=True):
    rows = []

    def add(name, ok, detail=""):
        rows.append({"check": name, "pass": bool(ok), "detail": detail})
        return ok

    # 1. every metric that is not withdrawn must have something that computes it
    missing = [k for k, v in REGISTRY.items()
               if v["tier"] != WITHDRAWN and v["compute"] is None
               and not k.startswith("residual/")]
    add("every live metric has code that computes it", not missing,
        f"{len(REGISTRY)} registered" + (f"; NO COMPUTE: {missing}" if missing else ""))

    # the residual/* family is one decomposition, so it is checked as one
    try:
        import torch
        import harmonic_inherited as H
        G, M = 20, 40
        rng = np.random.default_rng(0)
        t = np.linspace(0, 2 * np.pi, G, endpoint=False)
        a = rng.uniform(0.5, 1.5, M)
        p = np.stack([np.cos(t)[:, None] * a, np.sin(t)[:, None] * a * 0.6], -1)
        q = p * 1.3
        base, d = H.loopscore_residual(_t(q), _t(p), None)
        want = {"size", "orientation", "openness/aspect", "chirality", "shape-detail(k>=2)"}
        add("the residual decomposition returns its five named dimensions",
            set(d) == want, ", ".join(sorted(d)))
    except Exception as e:
        add("the residual decomposition returns its five named dimensions", False,
            f"{type(e).__name__}: {e}")

    # 2. every metric must have a written definition
    nodef = [k for k, v in REGISTRY.items() if not v.get("definition", "").strip()]
    add("every metric has a written definition", not nodef,
        "all present" if not nodef else str(nodef))

    # 3. no definition may name a withdrawn metric
    wd = set(withdrawn())
    bad = []
    for k, v in REGISTRY.items():
        if v["tier"] == WITHDRAWN:
            continue
        for w in wd:
            leaf = w.split("/")[-1]
            if leaf in v["definition"].split():
                bad.append(f"{k} names {w}")
    add("no live definition names a withdrawn metric", not bad, "clean" if not bad else str(bad))

    # 4. a withdrawn metric must carry its cause of death
    nocause = [k for k, v in withdrawn().items() if not v.get("cause_of_death")]
    add("every withdrawn metric records why", not nocause,
        f"{len(wd)} withdrawn, all with a cause" if not nocause else str(nocause))

    # 5. nothing is admitted yet, and that must be true rather than assumed
    add("nothing is cited as certified before Phase 2 certifies it", not admitted(),
        f"{len(admitted())} certified, {sum(1 for v in REGISTRY.values() if v['tier'] == PROVISIONAL)} "
        f"provisional, {len(wd)} withdrawn")

    # 6. the two implementations of each shared quantity must agree numerically
    try:
        al = check_aliases()
        bad = [r for r in al if not r["agree"]]
        add("descriptors.py and the registry agree on the shared quantities", not bad,
            "; ".join(f"{r['descriptors']}={r['descriptors_value']:.4g} vs "
                      f"{r['canonical']}={r['registry_value']:.4g}" for r in al))
    except Exception as e:
        add("descriptors.py and the registry agree on the shared quantities", False,
            f"{type(e).__name__}: {e}")

    # 7. the reading surface: the defect, MEASURED on a real model rather than by arithmetic
    meas = os.path.join(HERE, "_metrology", "archive_metric_test.json")
    if os.path.exists(meas):
        m = json.load(open(meas))["dashboard_grid"]
        add("the grid defect is measured on a real fit, not just counted",
            m["overlap"] == m["pinned_to_the_recording"] == m["scoring_1.000"],
            f"all {m['overlap']} pinned panels score 1.000; the picture reads "
            f"{m['mean_all_panels']:+.3f} against {m['mean_unpinned']:+.3f} on the unpinned ones "
            f"-- inflation {m['inflation']:+.3f}")

    # 8. the reading surface, and the defect in it
    g = grid_report()
    add("the 10x10 reading surface is outside the pinned band",
        g["corrected"]["in_band"] == 0,
        f"inherited margin {g['inherited']['margin_nodes']}: {g['inherited']['in_band']}/100 nodes "
        f"INSIDE the band; corrected margin {g['corrected']['margin_nodes']}: "
        f"{g['corrected']['in_band']}/100")

    if verbose:
        print(f"\n{'=' * 100}\n  METRICS -- one name, one quantity\n{'=' * 100}")
        for r in rows:
            print(f"  [{'  ok  ' if r['pass'] else ' FAIL '}] {r['check']:<52s} {r['detail']}")
        ok = all(r["pass"] for r in rows)
        print(f"\n  REGISTRY: {'PASS' if ok else 'FAIL'}  ({sum(r['pass'] for r in rows)}/{len(rows)})")
        print("=" * 100)
    return all(r["pass"] for r in rows), rows


def show():
    print(f"\n{'=' * 108}\n  THE REGISTRY\n{'=' * 108}")
    for tier in (CERTIFIED, PROVISIONAL, WITHDRAWN):
        ks = [k for k, v in REGISTRY.items() if v["tier"] == tier]
        if not ks:
            print(f"\n  {tier.upper()}: none")
            continue
        print(f"\n  {tier.upper()}")
        for k in ks:
            v = REGISTRY[k]
            print(f"    {k:<26s} {v['definition'][:74]}")
            if v.get("known_defects"):
                for d in v["known_defects"]:
                    print(f"      {'defect:':<8s} {d[:86]}")
            if v.get("cause_of_death"):
                print(f"      {'why:':<8s} {v['cause_of_death'][:86]}")
    g = grid_report()
    print(f"\n  READING SURFACE -- the 10x10 grid the campaign actually looks at")
    for k, v in g.items():
        print(f"    {k:<12s} margin {v['margin_nodes']:>3d} nodes -> {v['in_band']:>3d}/100 inside "
              f"the pinned band (nearest node {v['min_edge_distance']:.3f} from the edge, "
              f"band {v['band_width']})")
    print("=" * 108)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)
    if a.list:
        show()
        return 0
    ok, rows = check()
    json.dump({"pass": ok, "checks": rows,
               "registry": {k: {kk: vv for kk, vv in v.items() if kk != "compute"}
                            for k, v in REGISTRY.items()},
               "grid": grid_report()},
              open(os.path.join(HERE, "_metrology", "metrics_registry.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
