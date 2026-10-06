"""exp20: THE SUM-UP FROM BATCH 6 ON, one change at a time (Cedric, 2026-10-06: "a sum-up table starting at batch 6,
with one-to-one comparisons that allow conclusions; what jobs would complete the comparison").

    PYTHONPATH=src:tools python tools/exp20_summary.py

Every run on glucose fish 1 from batch 6 on (and batch 4's fish-1 law, 4.1, their common reference), scored the same way
on its own free rollout of the whole session:

  brain-mean r     Pearson r of the learned brain-mean dF/F against the recorded (exp17_slides.bm_metrics)
  per-neuron r     per cell, learned against recorded after each is regressed on its own brain mean; the mean over the
                   cells (exp17_slides.local_r) -- the deck's results print
  gut kept         the gut-responsive cells' evoked change after every gut pulse (0-20 s minus the 10 s before), the
                   law's over the recorded (tools/exp20_batch6.py's measure, from <run>_freetrial.json)

A comparison is two runs that differ by ONE thing. Its verdict per metric: "=" when the difference is within twice the
seed-to-seed difference (4.1 against 5.5, the same spec trained with seed 1: the one seed pair on fish 1), else up or
down. The two batch-7 graphs at their end (49,000 updates, landed 2026-10-06).
Writes data/summary_b6.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
DATA = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data")
RUNS = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "log", "training", "gutbrain")
B6 = "gb_b6_glucose_f1_"
LAWS = {"4.1": ("gb_sx_f1_mask_siren", ""), "5.5": ("gb_b5_glucose_f1_s1", ""),
        **{a: (B6 + s, "") for a, s in (("6.1", "mesh3"), ("6.2", "mesh4"), ("6.3", "mesh5"), ("6.4", "bio"),
                                       ("6.5", "mesh4_bio"), ("6.6", "paper2"), ("6.7", "paper3"), ("6.8", "mesh4_paper2"),
                                       ("6.9", "mesh4_paper3"), ("6.10", "anat_apvg"), ("6.11", "mesh4_anat_apvg"),
                                       ("6.12", "anat_dvc"), ("6.13", "mesh4_anat_dvc"))},
        "7.1": ("gb_b7_glucose_f1_mesh3_anat_apvg", ""), "7.2": ("gb_b7_glucose_f1_mf_anat_apvg", ""),
        "7.3": ("gb_b7_glucose_f1_now_anat_apvg", ""), "7.4": ("gb_b7_glucose_f1_mesh3_anat_dvc", ""),
        "7.5": ("gb_b7_glucose_f1_mf_anat_dvc", ""), "7.6": ("gb_b7_glucose_f1_now_anat_dvc", "")}
# (group, A, B, what changes from A to B)
COMPARE = [
    ("the graph", "4.1", "6.1", "neuron graph -> 3-level mesh"),
    ("the graph", "4.1", "6.2", "neuron graph -> 4-level mesh"),
    ("the graph", "4.1", "6.3", "neuron graph -> 5-level mesh"),
    ("the graph", "6.4", "6.5", "neuron graph -> 4-level mesh (10 % rule)"),
    ("the graph", "6.6", "6.8", "neuron graph -> 4-level mesh (paper 2 SD)"),
    ("the graph", "6.7", "6.9", "neuron graph -> 4-level mesh (paper 3 SD)"),
    ("the graph", "6.10", "6.11", "neuron graph -> 4-level mesh (atlas AP + VG)"),
    ("the graph", "6.12", "6.13", "neuron graph -> 4-level mesh (atlas + DVC)"),
    ("the graph", "7.1", "6.11", "3-level -> 4-level mesh (atlas AP + VG)"),
    ("the graph", "7.4", "6.13", "3-level -> 4-level mesh (atlas + DVC)"),
    ("the gut-input cells", "4.1", "6.4", "batches 1-5 -> 10 % gut-not-control"),
    ("the gut-input cells", "4.1", "6.6", "batches 1-5 -> paper 2 SD"),
    ("the gut-input cells", "4.1", "6.7", "batches 1-5 -> paper 3 SD"),
    ("the gut-input cells", "4.1", "6.10", "batches 1-5 -> atlas AP + VG"),
    ("the gut-input cells", "4.1", "6.12", "batches 1-5 -> atlas + DVC"),
    ("the gut-input cells", "6.2", "6.9", "batches 1-5 -> paper 3 SD (4-level mesh)"),
    ("the gut-input cells", "6.2", "6.11", "batches 1-5 -> atlas AP + VG (4-level mesh)"),
    ("the gut-input cells", "6.11", "6.13", "atlas AP + VG -> + DVC (4-level mesh)"),
    ("the gut-input cells", "7.1", "7.4", "atlas AP + VG -> + DVC (3-level mesh)"),
    ("the network", "7.2", "7.1", "mean field -> graph (AP + VG)"),
    ("the network", "7.3", "7.2", "no W -> mean field (AP + VG)"),
    ("the network", "7.5", "7.4", "mean field -> graph (+ DVC)"),
    ("the network", "7.6", "7.5", "no W -> mean field (+ DVC)"),
    ("the noise", "4.1", "5.5", "seed 0 -> seed 1 (the reference)"),
]
EV_S = 20.0


def gut_kept(run, stem, arm="full"):
    """The gut response kept by `arm` ("full", or "W0": the same law with W = 0) over the RECORDED response."""
    p = os.path.join(RUNS, run, "results", f"{run}{stem}_freetrial.json")
    if not os.path.exists(p):
        return None
    ft = json.load(open(p))
    pre, ev = ft["window"][0], int(round(EV_S / 1.117))
    P = [q for q in ft["arms"].get(arm, {}).get("pulses", []) if q["site"] != 1]
    R = [q for q in ft["arms"]["full"].get("pulses", []) if q["site"] != 1]
    if not P or not R:
        return None
    e = lambda Q, key: float(np.mean([np.mean(q[key][pre:pre + ev]) - np.mean(q[key][:pre]) for q in Q]))   # noqa: E731
    return e(P, "trace_free") / e(R, "trace_rec")


def main():
    import exp17_slides as E17
    M = {}
    for a, (run, stem) in LAWS.items():
        res = os.path.join(RUNS, run, "results")
        npz = os.path.join(res, f"{run}{stem}_movie.npz")
        tj = os.path.join(res, f"{run}{stem}_test.json")
        if not (os.path.exists(npz) and os.path.exists(tj)):
            print(f"[summary] {a} {run}{stem}: not scored yet")
            continue
        rec = json.load(open(tj)).get("trace_recording")
        if rec not in E17._REC:
            E17._REC.clear()
        bm, lr = E17.bm_metrics(npz), E17.local_r(npz, rec)
        M[a] = {"run": run + stem, "brain_r": bm["r"] if bm else None, "local_r": lr["mean"] if lr else None,
                "gut_kept": gut_kept(run, stem)}
        print(f"[summary] {a:5s} {run}{stem}: brain r {M[a]['brain_r']:+.3f}, per-neuron r {M[a]['local_r']:+.3f}, "
              f"gut kept {M[a]['gut_kept'] if M[a]['gut_kept'] is not None else float('nan'):.2f}")
    keys = ("brain_r", "local_r", "gut_kept")
    seed = {k: abs(M["5.5"][k] - M["4.1"][k]) for k in keys}
    rows = []
    for grp, a, b, what in COMPARE:
        if a not in M or b not in M:
            continue
        d = {k: (M[b][k] - M[a][k]) if M[a][k] is not None and M[b][k] is not None else None for k in keys}
        tol = {k: max(2 * seed[k], 0.01) for k in keys}
        v = {k: ("" if d[k] is None else "=" if abs(d[k]) <= tol[k] else ("up" if d[k] > 0 else "down")) for k in keys}
        rows.append({"group": grp, "a": a, "b": b, "what": what, "delta": d, "verdict": v,
                     "a_val": {k: M[a][k] for k in keys}, "b_val": {k: M[b][k] for k in keys}})
    out = {"laws": M, "seed_pair": seed, "tolerance": {k: max(2 * seed[k], 0.01) for k in keys}, "rows": rows}
    json.dump(out, open(os.path.join(DATA, "summary_b6.json"), "w"), indent=1)
    print("[summary] seed-to-seed |difference| (4.1 vs 5.5):", {k: round(v, 3) for k, v in seed.items()})
    for r in rows:
        print(f"  {r['a']:>5s} -> {r['b']:5s} {r['what']:48s} " + "  ".join(
            f"{k} {r['delta'][k]:+.3f} {r['verdict'][k]:4s}" if r['delta'][k] is not None else f"{k} --" for k in keys))


if __name__ == "__main__":
    main()
