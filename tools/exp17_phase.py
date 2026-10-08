"""exp17: WHICH NEURONS OSCILLATE IN ANTIPHASE (Cedric, 2026-10-07: "look at all pairs of cells that oscillate with pi
dephasing [in the rotation block], then plot a red / blue map on the fish"). Local, the destriped recording in the
Z-Brain atlas (tools/exp17_atlas.py), no model.

The rotation block's stimulus is a +-1 square wave (stimulus column 19: grating rotating one way, then the other)
of period P_S = 60 s. Every neuron's dF/F over the block (its first SKIP_S s dropped: the onset response) is fitted by
    x(t) = b0 + b1 t + a cos(2 pi t / P_S) + c sin(2 pi t / P_S),
the stimulus by the same; R2 = the share of the detrended variance the sinusoid explains, and dphi = the neuron's
phase minus the stimulus's, in (-pi, pi]. Two neurons oscillate in antiphase when their dphi differ by pi, so all
such pairs are: one neuron with cos(dphi) < 0 (RED, half a cycle off the stimulus) and one with cos(dphi) > 0
(BLUE, in phase). The colours chosen so red is mostly the left hemisphere and blue the right, as on the ARTR slide
(red = ARTR left there; all 100 left ARTR cells picked on rotation fall in this red group, the 100 right in blue).
A neuron counts when its R2 beats the NULL: the same fit at off periods (OFF_S, no stimulus power there), the 99.9th
percentile over all neurons and off periods.

    PYTHONPATH=src:tools python tools/exp17_phase.py [--movie]
-> presentation/figs/phase_rotation.png [, presentation/Movies/phase_rotation.mp4], data/phase_rotation.json, data/phase_rotation.npz
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
DT, P_S, SKIP_S, COL = 0.914, 60.0, 60.0, 19
OFF_S = (41.0, 47.0, 77.0, 89.0)
MID, W_UM = (621 - 1) * 0.798 / 2, (621 - 1) * 0.798


def fit(X, period):
    """Per column of X [T, n]: R2 of the sinusoid over a linear trend, and its phase (cos coefficient a, sin c)."""
    t = np.arange(len(X)) * DT
    D0 = np.column_stack([np.ones_like(t), t - t.mean()])
    w = 2 * np.pi * t / period
    D = np.column_stack([D0, np.cos(w), np.sin(w)])
    B = np.linalg.lstsq(D, X, rcond=None)[0]
    R0 = X - D0 @ np.linalg.lstsq(D0, X, rcond=None)[0]
    R = X - D @ B
    r2 = 1.0 - (R * R).sum(0) / np.maximum((R0 * R0).sum(0), 1e-12)
    return r2, np.arctan2(B[3], B[2])                  # x ~ cos(w - phi), phi = atan2(c, a)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    from exp17_atlas import REGIONS
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, reg, names, ins = za["atlas_um"].astype(np.float64), za["regions"], [str(x) for x in za["names"]], za["inside"]
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    off, bn = z["offsets"], [str(x) for x in z["names"]]
    k = bn.index("rotation")
    f0, f1 = int(off[k]) + int(round(SKIP_S / DT)), int(off[k + 1])
    X = np.asarray(z["dff"][f0:f1], np.float64)
    ok = X.std(0) > 1e-6
    r2s, phs = fit(z["stimulus"][f0:f1, COL:COL + 1].astype(np.float64), P_S)
    r2, ph = fit(X, P_S)
    null = np.concatenate([fit(X[:, ok], p)[0] for p in OFF_S])
    thr = float(np.percentile(null, 99.9))
    dphi = np.angle(np.exp(1j * (ph - phs[0])))
    sig = ok & (r2 > thr)
    red, blue = sig & (np.cos(dphi) < 0), sig & (np.cos(dphi) > 0)
    left = A[:, 0] < MID
    # per region: red / blue on each side (a chess board = red on one side, blue on the other)
    rlist = [(f, s) for f, s in REGIONS if f in names]
    per_region = {}
    for full, short in rlist:
        m = reg[:, names.index(full)] & ins
        per_region[short] = {"neurons": int(m.sum()),
                             "left_red_blue": [int((m & left & red).sum()), int((m & left & blue).sum())],
                             "right_red_blue": [int((m & ~left & red).sum()), int((m & ~left & blue).sum())]}
    doc = {"period_s": P_S, "frames": f1 - f0, "stimulus_r2": float(r2s[0]), "null_r2_p999": thr,
           "significant": int(sig.sum()), "red": int(red.sum()), "blue": int(blue.sum()),
           "red_left_right": [int((red & left).sum()), int((red & ~left).sum())],
           "blue_left_right": [int((blue & left).sum()), int((blue & ~left).sum())],
           "antiphase_pairs": int(red.sum()) * int(blue.sum()),
           # each group's circular-mean phase behind the stimulus, in s of the 60-s period
           "red_lag_s": float(np.angle(np.exp(1j * dphi[red]).mean()) / (2 * np.pi) * P_S),
           "blue_lag_s": float(np.angle(np.exp(1j * dphi[blue]).mean()) / (2 * np.pi) * P_S),
           "red_blue_phase_gap_deg": float(np.degrees(np.abs(np.angle(np.exp(1j * dphi[red]).mean()
                                                                      / np.exp(1j * dphi[blue]).mean())))),
           "per_region": per_region}
    json.dump(doc, open(os.path.join(EXP, "data", "phase_rotation.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(EXP, "data", "phase_rotation.npz"), r2=r2.astype(np.float32),
                        dphi=dphi.astype(np.float32), sig=sig, thr=thr)
    print(json.dumps({k_: v for k_, v in doc.items() if k_ != "per_region"}, indent=1))

    # the figure, and as a MOVIE (Cedric, 2026-10-07, "make a movie in slide 14"): every red / blue neuron lit by its own
    # dF/F at the frame (z over the window, a 3-frame mean), a white bar sweeping the traces; open loop -> rotation -> dark,
    # one movie frame per recorded frame; the layout twin of the ARTR slide's (exp17_artr.twin_figure).
    xd, yd = A[:, 1], W_UM - A[:, 0]
    o = np.argsort(r2[sig])                              # the strongest drawn last
    ii = np.flatnonzero(sig)[o]
    base = np.where(np.cos(dphi[ii])[:, None] < 0, [[1.0, 0.19, 0.19]], [[0.23, 0.42, 1.0]])
    g0, g1 = int(off[bn.index("open loop")]), int(off[bn.index("dark") + 1])
    t = (np.arange(g0, g1) - g0) * DT / 60
    Xg = np.asarray(z["dff"][g0:g1][:, ii], np.float32)

    from exp17_artr import twin_figure, smooth3, zs, region_rows
    Zm = smooth3(zs(Xg))                                 # each neuron's dF/F over the window, z, the 3-frame mean
    is_red = np.cos(dphi[ii]) < 0
    stim = np.asarray(z["stimulus"][g0:g1, COL], np.float32)
    tr = [(Zm[:, is_red].mean(1), "#ff3030", f"red, mostly left ({int(is_red.sum()):,})"),
          (Zm[:, ~is_red].mean(1), "#3a6bff", f"blue, mostly right ({int((~is_red).sum()):,})")]
    marks = [(0.0, "open loop")] + [((int(off[bn.index(b)]) - g0) * DT / 60, b) for b in ("rotation", "dark")]

    rows_ = region_rows(reg, names, ii, is_red)

    def build():
        return twin_figure(A, ins, ii, base, t, tr, stim, marks, Z=Zm, rows=rows_,
                           caption="")
    fig, _, _ = build()
    fig.savefig(os.path.join(EXP, "presentation", "figs", "phase_rotation.png"), dpi=130, facecolor="black")
    plt.close(fig)
    if "--movie" in sys.argv:
        from exp17_artr import movie
        movie(build, scs_rgb=base, Zc=Zm, n=g1 - g0, name="phase_rotation")

if __name__ == "__main__":
    main()
