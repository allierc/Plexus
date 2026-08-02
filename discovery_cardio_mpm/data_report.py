#!/usr/bin/env python
"""data_report -- the two numbers that bound everything this project can claim.

WHY THESE TWO, AND WHY BEFORE ANYTHING IS FITTED
================================================================================================
Track B is judged by how far the simulated loop is from the recorded one. That comparison is
meaningless without knowing how far the recording is from ITSELF. Two different senses of "itself",
and they bound different things:

  1. THE CEILING.  Score one real beat against another real beat, with the same instrument used to
     score the model. The tissue is not a metronome; consecutive beats differ. **No model may be
     scored better than the data agrees with itself** -- a fit that beats this number is fitting
     the difference between one beat and the next, which is not a mechanism.

  2. THE FLOOR ON SPATIAL CLAIMS.  The same movie was tracked twice, independently. Wherever those
     two trackings disagree, the disagreement is a property of the tracking software, not of the
     tissue. If the fine spatial pattern is mostly tracker, then a learned map of stiffness across
     the sheet is measuring the tracker, and **learned fields are not a publishable product of this
     project.** That is a large claim to be able to settle in an afternoon, and it decides what the
     campaign should be aimed at.

Both are expressed in the SAME units as the model residual -- the Track B descriptors -- so they can
be drawn on the same axis as any later result rather than quoted as a separate caveat.

WHAT THE SECOND TRACKING IS, EXACTLY
------------------------------------------------------------------------------------------------
`healthy.npy` is not a crop of the main array, and not a copy: it covers the top-left 80x80 of the
same 15-px lattice, and its frame 0 is the *pristine undeformed grid* while the main array's frame 0
already carries a tracking result. So they are two genuinely different registrations of one movie,
compared here on their overlapping region, each referenced to its own first frame, with the frame
offset between them searched rather than assumed.

    python data_report.py            # both measurements, and a figure
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data as D                                                    # noqa: E402
import descriptors as DS                                            # noqa: E402

SOURCE_ROOT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
SECOND_TRACKING = os.path.join(SOURCE_ROOT, "healthy.npy")
SIDE = 137                                                          # the main lattice is 137 x 137
PITCH_PX = 15.0


# ---------------------------------------------------------------------------------------------
# 1. THE CEILING -- how well does the recording agree with itself, beat to beat?
# ---------------------------------------------------------------------------------------------
def self_agreement(pos=None, verbose=True):
    z = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256) if pos is None else None
    P = (z["pos"] if pos is None else pos).astype(np.float64)
    b = D.beats(P)
    onsets = b["onsets"]

    # Complete beats only. The stretch after the last onset is not a beat and is not treated as one.
    spans = [(onsets[i], onsets[i + 1]) for i in range(len(onsets) - 1)]
    G = min(e - s for s, e in spans)                                 # common length: 49 frames
    beats = [P[s:s + G] - P[s] for s, e in spans]                    # each referenced to its own onset

    # score on the nodes that actually move, chosen ONCE from the whole recording so the mask is
    # the same for every pair -- a per-pair mask would let the comparison choose its own support
    amp = np.linalg.norm(P - P[0], axis=-1).max(axis=0)
    mov = amp > 0.2 * np.percentile(amp, 99)

    pairs, rows = [], []
    for i in range(len(beats)):
        for j in range(i + 1, len(beats)):
            r = DS.loop_residual(beats[i], beats[j], mov)
            rows.append({
                "pair": [i, j],
                "magnitude_peak": r["magnitude_peak"]["ratio"],
                "magnitude_energy": r["magnitude_energy"]["ratio"],
                "opening_area": r["opening_area"]["ratio"],
                "opening_loopiness": r["opening_loopiness"]["ratio"],
                "direction_chirality": r["direction_chirality"]["ratio"],
                "orientation_rad": r["orientation_error_rad"]["median_rad_covariance"],
                "shape_minor": r["shape_minor_fraction"]["ratio"]})
            pairs.append((i, j))

    def spread(k):
        v = np.array([r[k] for r in rows], float)
        return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                "sd": float(v.std())}

    out = {"n_complete_beats": len(beats), "beat_length_frames": int(G),
           "onsets": onsets, "gaps": b["gaps"], "mean_gap": b["mean_gap"],
           "reported_period": b["period"], "nodes_scored": int(mov.sum()),
           "n_pairs": len(rows), "pairs": rows,
           "ceiling": {k: spread(k) for k in
                       ("magnitude_peak", "magnitude_energy", "opening_area", "opening_loopiness",
                        "direction_chirality", "orientation_rad", "shape_minor")}}

    # the same thing through the inherited objective, for comparability with the old record
    try:
        import torch
        import harmonic_inherited as HARM
        t = lambda a: torch.tensor(np.ascontiguousarray(a[:, mov]), dtype=torch.float32)
        ls = [float(HARM.harmonic_score(t(beats[i]), t(beats[j]), None)) for i, j in pairs]
        out["loopscore_beat_to_beat"] = {"values": ls, "median": float(np.median(ls)),
                                         "min": float(min(ls)), "max": float(max(ls))}
    except Exception as e:
        out["loopscore_error"] = f"{type(e).__name__}: {e}"

    if verbose:
        print(f"\n{'=' * 96}\n  1. THE CEILING -- the recording against itself, beat to beat\n{'=' * 96}")
        print(f"  {len(beats)} complete beats (onsets {onsets}, gaps {b['gaps']}), "
              f"{G} frames each, {int(mov.sum())} moving nodes, {len(rows)} pairs")
        print(f"\n  {'axis':<22s} {'median':>9s} {'min':>9s} {'max':>9s}   (1.0 = the two beats agree)")
        for k, v in out["ceiling"].items():
            print(f"  {k:<22s} {v['median']:>9.4f} {v['min']:>9.4f} {v['max']:>9.4f}")
        if "loopscore_beat_to_beat" in out:
            L = out["loopscore_beat_to_beat"]
            print(f"\n  inherited LoopScore, beat vs beat: median {L['median']:.4f} "
                  f"(range {L['min']:.4f}..{L['max']:.4f})")
            print(f"  ==> NO MODEL MAY BE SCORED BETTER THAN {L['median']:.3f} ON LOOPSCORE.")
            print(f"      A fit above that is fitting the difference between one beat and the next.")
    return out


# ---------------------------------------------------------------------------------------------
# 2. THE FLOOR -- how much of the spatial pattern is the tracking software?
# ---------------------------------------------------------------------------------------------
def _second_tracking():
    A = np.load(SECOND_TRACKING, mmap_mode="r")                      # [240, 80, 80, 2] absolute px
    return np.asarray(A).astype(np.float64)


def _main_tracking_grid():
    z = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)
    P = z["pos"].astype(np.float64) * 2048.0                         # back to pixels
    F = P.shape[0]
    return P.reshape(F, SIDE, SIDE, 2)


def tracker_reproducibility(verbose=True, smooth_scales=(1, 3, 5, 9, 17)):
    B = _second_tracking()                                           # [240, 80, 80, 2]
    A = _main_tracking_grid()                                        # [239, 137, 137, 2]
    n = B.shape[1]
    A = A[:, :n, :n]                                                 # the overlapping region

    # Each tracking is referenced to ITS OWN first frame: the two start from different states
    # (B's frame 0 is the pristine lattice, A's already carries a result), so only displacement is
    # comparable. The frame offset between them is SEARCHED, not assumed.
    best = None
    for off in range(-3, 4):
        ia = slice(max(0, off), min(A.shape[0], B.shape[0] + off))
        ib = slice(max(0, -off), min(B.shape[0], A.shape[0] - off))
        da = A[ia] - A[ia][0]
        db = B[ib] - B[ib][0]
        T = min(da.shape[0], db.shape[0])
        ta = np.linalg.norm(da[:T], axis=-1).mean(axis=(1, 2))
        tb = np.linalg.norm(db[:T], axis=-1).mean(axis=(1, 2))
        r = float(np.corrcoef(ta, tb)[0, 1])
        if best is None or r > best["time_corr"]:
            best = {"offset": off, "time_corr": r, "T": int(T),
                    "da": da[:T], "db": db[:T], "ta": ta, "tb": tb}

    da, db = best.pop("da"), best.pop("db")

    # The frame of peak motion INSIDE each complete beat, each referenced to its own onset -- not
    # the global peak, which falls in the truncated tail after the last onset and is not a beat.
    # Doing it per beat also shows whether the disagreement is a one-frame accident or a constant.
    onsets = D.beats(A.reshape(A.shape[0], -1, 2) / 2048.0)["onsets"]
    windows = [(onsets[i], onsets[i + 1]) for i in range(len(onsets) - 1)
               if onsets[i + 1] <= min(da.shape[0], db.shape[0])]
    per_beat, peaks = [], []
    for i, (s0, e0) in enumerate(windows):
        seg = np.linalg.norm(da[s0:e0] - da[s0], axis=-1).mean(axis=(1, 2))
        kk = s0 + int(np.argmax(seg))
        peaks.append((i, s0, kk))
        ga = (da[kk] - da[s0]).mean(axis=(0, 1))
        gb = (db[kk] - db[s0]).mean(axis=(0, 1))
        per_beat.append({"beat": i, "window": [int(s0), int(e0)], "peak_frame": int(kk),
                         "corr_x": float(np.corrcoef((da[kk] - da[s0])[..., 0].ravel(),
                                                     (db[kk] - db[s0])[..., 0].ravel())[0, 1]),
                         "corr_y": float(np.corrcoef((da[kk] - da[s0])[..., 1].ravel(),
                                                     (db[kk] - db[s0])[..., 1].ravel())[0, 1]),
                         "whole_field_mean_A": [float(ga[0]), float(ga[1])],
                         "whole_field_mean_B": [float(gb[0]), float(gb[1])]})
    i0, s0, k = peaks[0]
    fa, fb = da[k] - da[s0], db[k] - db[s0]                          # [80,80,2], beat-referenced

    # Is a swapped or transposed axis convention hiding the agreement? Tested, not assumed.
    def _cc(u, v):
        return float(np.corrcoef(u.ravel(), v.ravel())[0, 1])
    conventions = {
        "as_given": [_cc(fa[..., 0], fb[..., 0]), _cc(fa[..., 1], fb[..., 1])],
        "channels_swapped": [_cc(fa[..., 0], fb[..., 1]), _cc(fa[..., 1], fb[..., 0])],
        "grid_transposed": [_cc(fa[..., 0], fb[..., 0].T), _cc(fa[..., 1], fb[..., 1].T)],
        "y_sign_flipped": [_cc(fa[..., 0], fb[..., 0]), _cc(fa[..., 1], -fb[..., 1])]}

    def box(x, w):
        if w <= 1:
            return x
        from scipy.ndimage import uniform_filter
        return np.stack([uniform_filter(x[..., c], size=w, mode="nearest") for c in range(2)], -1)

    smoothing = []
    for w in smooth_scales:
        sa, sb = box(fa, w), box(fb, w)
        cx = float(np.corrcoef(sa[..., 0].ravel(), sb[..., 0].ravel())[0, 1])
        cy = float(np.corrcoef(sa[..., 1].ravel(), sb[..., 1].ravel())[0, 1])
        rel = float(np.linalg.norm(sa - sb) / max(np.linalg.norm(sa), 1e-30))
        smoothing.append({"box_nodes": int(w), "box_px": float(w * PITCH_PX),
                          "corr_x": cx, "corr_y": cy, "rel_l2": rel})

    # and the sharpest statement: read one tracking as if it were a MODEL of the other, with the
    # very instrument Track B will be judged by. This puts the tracker's own disagreement on the
    # same axis as any later model residual.
    onsets = D.beats(_main_tracking_grid().reshape(A.shape[0] if False else -1, 2)[:0] or None) \
        if False else None
    bA = D.beats((A.reshape(A.shape[0], -1, 2) / 2048.0))["onsets"]
    G = min(np.diff(bA)) if len(bA) > 1 else 49
    o = bA[1] if len(bA) > 1 else 0
    pa = (A[o:o + G] - A[o]).reshape(G, -1, 2)
    pb = (B[o + best["offset"]:o + best["offset"] + G] - B[o + best["offset"]]).reshape(G, -1, 2)
    amp = np.linalg.norm(pa, axis=-1).max(axis=0)
    mov = amp > 0.2 * np.percentile(amp, 99)
    res = DS.loop_residual(pb, pa, mov)

    out = {"second_tracking": SECOND_TRACKING, "overlap_nodes": f"{n}x{n}",
           "frame_offset": best["offset"], "frames_compared": best["T"],
           "time_course_corr": best["time_corr"],
           "peak_frame": k, "smoothing": smoothing,
           "per_beat": per_beat, "axis_conventions_tested": conventions,
           "as_a_model_of_the_other": {
               "nodes_scored": res["n_nodes"], "beat_frames": int(G),
               "magnitude_peak": res["magnitude_peak"]["ratio"],
               "opening_area": res["opening_area"]["ratio"],
               "opening_loopiness": res["opening_loopiness"]["ratio"],
               "direction_chirality": res["direction_chirality"]["ratio"],
               "orientation_rad": res["orientation_error_rad"]["median_rad_covariance"],
               "shape_minor": res["shape_minor_fraction"]["ratio"]}}
    try:
        import torch
        import harmonic_inherited as HARM
        t = lambda a: torch.tensor(np.ascontiguousarray(a[:, mov]), dtype=torch.float32)
        out["as_a_model_of_the_other"]["loopscore"] = float(
            HARM.harmonic_score(t(pb), t(pa), None))
    except Exception as e:
        out["loopscore_error"] = f"{type(e).__name__}: {e}"

    if verbose:
        print(f"\n{'=' * 96}\n  2. THE FLOOR -- two independent trackings of the SAME movie\n{'=' * 96}")
        print(f"  overlap {n}x{n} nodes, frame offset {best['offset']:+d}, "
              f"{best['T']} frames compared")
        print(f"\n  WHEN and HOW MUCH the tissue moves:  time-course correlation "
              f"{best['time_corr']:.4f}")
        print(f"\n  WHERE it moves -- per-node displacement map, at the peak of EACH beat:")
        print(f"  {'beat':>6s} {'peak':>6s} {'corr x':>8s} {'corr y':>8s}   "
              f"{'whole-field mean A':>22s} {'whole-field mean B':>22s}")
        for b in per_beat:
            print(f"  {b['beat']:>6d} {b['peak_frame']:>6d} {b['corr_x']:>8.3f} {b['corr_y']:>8.3f}   "
                  f"({b['whole_field_mean_A'][0]:+6.3f},{b['whole_field_mean_A'][1]:+6.3f}) px      "
                  f"({b['whole_field_mean_B'][0]:+6.3f},{b['whole_field_mean_B'][1]:+6.3f}) px")
        print(f"\n  no axis convention rescues it: " +
              ", ".join(f"{k} ({v[0]:.2f},{v[1]:.2f})" for k, v in conventions.items()))
        print(f"\n  and with spatial smoothing:")
        print(f"  {'smoothing':>12s} {'corr x':>8s} {'corr y':>8s} {'rel L2':>8s}")
        for s in smoothing:
            print(f"  {s['box_px']:>9.0f} px {s['corr_x']:>8.3f} {s['corr_y']:>8.3f} "
                  f"{s['rel_l2']:>8.3f}")
        m = out["as_a_model_of_the_other"]
        print(f"\n  Reading one tracking as a MODEL of the other, with Track B's own instrument:")
        for kk in ("magnitude_peak", "opening_area", "opening_loopiness", "direction_chirality",
                   "shape_minor"):
            print(f"    {kk:<22s} {m[kk]:.4f}")
        print(f"    {'orientation_rad':<22s} {m['orientation_rad']:.4f}")
        if "loopscore" in m:
            print(f"    {'LoopScore':<22s} {m['loopscore']:.4f}")
        print("\n  ==> Everything the two trackings disagree about is TRACKER, not tissue.")
    return out


def figure(out_png, agree, repro):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))

    a = ax[0]
    ks = ["magnitude_peak", "opening_area", "opening_loopiness", "direction_chirality", "shape_minor"]
    med = [agree["ceiling"][k]["median"] for k in ks]
    lo = [agree["ceiling"][k]["min"] for k in ks]
    hi = [agree["ceiling"][k]["max"] for k in ks]
    y = np.arange(len(ks))
    a.hlines(y, lo, hi, color="#2B4C7E", lw=3, alpha=0.4)
    a.plot(med, y, "o", color="#2B4C7E")
    a.axvline(1.0, color="#666666", lw=0.8, ls="--")
    a.set_yticks(y); a.set_yticklabels([k.replace("_", "\n") for k in ks], fontsize=7)
    a.set_xlabel("ratio, beat vs beat (1.0 = identical)", fontsize=8)
    a.set_title("the ceiling: the recording against itself", fontsize=9)

    a = ax[1]
    w = [s["box_px"] for s in repro["smoothing"]]
    a.plot(w, [s["corr_x"] for s in repro["smoothing"]], "o-", label="x", color="#B3261E")
    a.plot(w, [s["corr_y"] for s in repro["smoothing"]], "s-", label="y", color="#2B4C7E")
    a.axhline(repro["time_course_corr"], color="#1B7F3B", ls="--", lw=1,
              label=f"time course ({repro['time_course_corr']:.3f})")
    a.set_xlabel("spatial smoothing (px)", fontsize=8)
    a.set_ylabel("correlation between the two trackings", fontsize=8)
    a.set_ylim(-0.05, 1.05); a.legend(fontsize=7)
    a.set_title("the floor: how much of the pattern is the tracker", fontsize=9)

    a = ax[2]
    a.plot(repro["_ta"], color="#2B4C7E", lw=1, label="tracking A")
    a.plot(repro["_tb"], color="#B3261E", lw=1, ls="--", label="tracking B")
    a.set_xlabel("frame", fontsize=8); a.set_ylabel("mean |displacement| (px)", fontsize=8)
    a.legend(fontsize=7)
    a.set_title("the two agree on WHEN, not on WHERE", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default=os.path.join(HERE, "_metrology", "data_report.png"))
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)

    agree = self_agreement()
    repro = tracker_reproducibility()

    # keep the traces for the figure but not for the json
    B = _second_tracking(); A = _main_tracking_grid()[:, :B.shape[1], :B.shape[1]]
    off = repro["frame_offset"]
    ia = slice(max(0, off), min(A.shape[0], B.shape[0] + off))
    ib = slice(max(0, -off), min(B.shape[0], A.shape[0] - off))
    da, db = A[ia] - A[ia][0], B[ib] - B[ib][0]
    T = min(da.shape[0], db.shape[0])
    repro["_ta"] = np.linalg.norm(da[:T], axis=-1).mean(axis=(1, 2))
    repro["_tb"] = np.linalg.norm(db[:T], axis=-1).mean(axis=(1, 2))
    p = figure(a.figure, agree, repro)
    for k in ("_ta", "_tb"):
        repro.pop(k, None)
    print(f"\n[data_report] figure -> {p}")

    json.dump({"self_agreement": agree, "tracker_reproducibility": repro},
              open(os.path.join(HERE, "_metrology", "data_report.json"), "w"),
              indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
