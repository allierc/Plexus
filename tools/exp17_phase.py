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

    PYTHONPATH=src:tools python tools/exp17_phase.py [--movie] [--model <run>] [--select antiphase|quiet|lagged]
-> presentation/figs/<name>.png [, presentation/Movies/<name>.mp4], data/<name>.json (, .npz)
   <name> = phase_rotation[_quiet|_lagged][_model | _model_<run>] (_model alone: 22.3, zap_n22_markall)

THE TWO OTHER SELECTIONS (Cedric, 2026-10-10: "a twin of 29 for neurons that are not correlated to the stimuli and
not silent, sorted by non-correlation; a twin for neurons that are correlated but have a large lag, sorted by lag"),
both made on the RECORDING, so the model's panel shows the same neurons in the same rows:
  quiet    not correlated: R2 below the null's median (as little 60-s power as a typical neuron at an off period);
           not silent: the dF/F SD over the block at least the median SD of the significant (antiphase-map) neurons;
           rows by R2, the least correlated first; one colour (orange)
  lagged   significant (R2 above the null, as the antiphase map) and in QUADRATURE: |sin dphi| > |cos dphi|, the phase
           lag (dphi mod 2 pi) / 2 pi x P_S between 7.5 and 22.5 s or 37.5 and 52.5 s behind the stimulus -- neither
           in phase (the blue group) nor half a cycle off (the red); rows by lag; colour by lag (a cyclic map)
With --model the antiphase map is the model's own selection (as before); quiet and lagged keep the recording's.
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


MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
SEL = sys.argv[sys.argv.index("--select") + 1] if "--select" in sys.argv else "antiphase"
if SEL not in ("antiphase", "quiet", "lagged"):
    raise SystemExit("--select antiphase | quiet | lagged")
SUF = ("" if SEL == "antiphase" else f"_{SEL}") + ("" if not MODEL else "_model" if MODEL == "zap_n22_markall"
                                                   else f"_model_{MODEL}")


def traces(z):
    """[T, N] dF/F: the recording, or with --model the learned law's free rollout (tools/exp17_model_traces.py)."""
    if MODEL:
        from exp17_model_traces import path as mpath
        return np.load(mpath(MODEL), mmap_mode="r")
    return z["dff"]


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
    X = np.asarray(traces(z)[f0:f1], np.float64)
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
    if SEL != "antiphase":
        return other(z, A, ins, reg, names, off, bn, f0, f1, ok, r2, dphi, sig, thr, null)
    if MODEL:
        # THE TWIN'S AGREEMENT with the recording's map: of the neurons oscillating in both, the share in the same group
        rp = np.load(os.path.join(EXP, "data", "phase_rotation.npz"))
        both = rp["sig"] & sig
        same = (np.cos(rp["dphi"]) < 0) == (np.cos(dphi) < 0)
        doc["model"] = MODEL
        doc["agreement"] = {"oscillating_in_both": int(both.sum()), "recording_only": int((rp["sig"] & ~sig).sum()),
                            "model_only": int((~rp["sig"] & sig).sum()),
                            "same_group_share": float(same[both].mean()) if both.any() else None}
        print("[agree]", doc["agreement"])
    json.dump(doc, open(os.path.join(EXP, "data", f"phase_rotation{SUF}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(EXP, "data", f"phase_rotation{SUF}.npz"), r2=r2.astype(np.float32),
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
    Xg = np.asarray(traces(z)[g0:g1][:, ii], np.float32)

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
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"phase_rotation{SUF}.png"), dpi=130, facecolor="black")
    plt.close(fig)
    if "--movie" in sys.argv:
        from exp17_artr import movie
        movie(build, scs_rgb=base, Zc=Zm, n=g1 - g0, name=f"phase_rotation{SUF}")

def other(z, A, ins, reg, names, off, bn, f0, f1, ok, r2, dphi, sig, thr, null):
    """The quiet and lagged twins (see the docstring): the selection on the recording, the traces of MODEL or of the
    recording, the layout of the antiphase map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp17_artr import twin_figure, smooth3, zs
    if MODEL:                                             # the selection is the recording's: its fit again
        Xr = np.asarray(z["dff"][f0:f1], np.float64)
        ok = Xr.std(0) > 1e-6
        r2s, phs = fit(z["stimulus"][f0:f1, COL:COL + 1].astype(np.float64), P_S)
        r2, ph = fit(Xr, P_S)
        null = np.concatenate([fit(Xr[:, ok], p)[0] for p in OFF_S])
        thr = float(np.percentile(null, 99.9))
        dphi = np.angle(np.exp(1j * (ph - phs[0])))
        sig = ok & (r2 > thr)
        sd = Xr.std(0)
    else:
        sd = np.asarray(z["dff"][f0:f1], np.float64).std(0)
    lag = np.mod(dphi, 2 * np.pi) / (2 * np.pi) * P_S                 # s behind the stimulus, [0, P_S)
    if SEL == "quiet":
        sd_min, r2_max = float(np.median(sd[sig])), float(np.median(null))
        m = ok & (sd >= sd_min) & (r2 < r2_max)
        ii = np.flatnonzero(m)[np.argsort(r2[m])]                     # the least correlated first
        rgb = np.tile([[1.0, 0.62, 0.10]], (len(ii), 1))
        key, rule = r2[ii], {"sd_min": sd_min, "r2_max_null_median": r2_max}
        seg = [(f"R2 {r2[ii].min():.3f} .. {r2[ii].max():.3f}", 0, len(ii))]
    else:
        m = sig & (np.abs(np.sin(dphi)) > np.abs(np.cos(dphi)))
        ii = np.flatnonzero(m)[np.argsort(lag[m])]
        rgb = plt.get_cmap("hsv")(lag[ii] / P_S)[:, :3]
        key, rule = lag[ii], {"r2_min": thr, "quadrature": "|sin dphi| > |cos dphi|"}
        seg = [(f"lag {lag[ii].min():.0f} .. {lag[ii].max():.0f} s", 0, len(ii))]
    g0, g1 = int(off[bn.index("open loop")]), int(off[bn.index("dark") + 1])
    t = (np.arange(g0, g1) - g0) * DT / 60
    Xg = np.asarray(traces(z)[g0:g1][:, ii], np.float32)
    Zm = smooth3(zs(Xg))
    stim = np.asarray(z["stimulus"][g0:g1, COL], np.float32)
    if SEL == "quiet":
        tr = [(Zm.mean(1), "white", f"their mean ({len(ii):,})")]          # white: the stimulus line is orange
    else:
        q = (lag[ii] < P_S / 2)
        tr = [(Zm[:, q].mean(1), "#ffd23a", f"a quarter cycle behind, {int(q.sum()):,}"),
              (Zm[:, ~q].mean(1), "#3ad0ff", f"a quarter cycle ahead, {int((~q).sum()):,}")]
    marks = [(0.0, "open loop")] + [((int(off[bn.index(b)]) - g0) * DT / 60, b) for b in ("rotation", "dark")]
    rows_ = (np.arange(len(ii)), seg, max(1, int(np.ceil(len(ii) / 600))))
    doc = {"select": SEL, "rule": rule, "period_s": P_S, "neurons": int(len(ii)),
           "key_quantiles_10_50_90": [float(np.quantile(key, q_)) for q_ in (0.1, 0.5, 0.9)],
           "left_right": [int((A[ii, 0] < MID).sum()), int((A[ii, 0] >= MID).sum())]}
    if MODEL:
        R = np.asarray(z["dff"][g0:g1][:, ii], np.float32)
        Rc, Mc = R - R.mean(0), Xg - Xg.mean(0)
        rr = (Rc * Mc).sum(0) / np.maximum(np.sqrt((Rc ** 2).sum(0) * (Mc ** 2).sum(0)), 1e-12)
        doc["model"] = MODEL
        doc["model_vs_recording_r_10_50_90"] = [float(np.quantile(rr, q_)) for q_ in (0.1, 0.5, 0.9)]
    json.dump(doc, open(os.path.join(EXP, "data", f"phase_rotation{SUF}.json"), "w"), indent=1)
    print(json.dumps(doc, indent=1))

    def build():
        return twin_figure(A, ins, ii, rgb, t, tr, stim, marks, Z=Zm, rows=rows_, caption="")
    fig, _, _ = build()
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"phase_rotation{SUF}.png"), dpi=130, facecolor="black")
    plt.close(fig)
    if "--movie" in sys.argv:
        from exp17_artr import movie
        movie(build, scs_rgb=rgb, Zc=Zm, n=g1 - g0, name=f"phase_rotation{SUF}")


if __name__ == "__main__":
    main()
