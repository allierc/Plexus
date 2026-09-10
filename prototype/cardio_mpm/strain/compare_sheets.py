"""compare_sheets -- healthy against HCM, the same model fitted the same way to each.

Inputs: two fits (E fixed, band cells masked, per-cell g, phi, delta + one clock). Outputs a table
and one figure of distributions over interior cells: fitted shortening at full activation g, clock
delay delta (seconds), the recording's own peak shortening per cell (so the model's number sits
beside the measurement it explains), the fibre-axis order (how aligned the cells' axes are, 0 =
random, 1 = all parallel), and the two clocks in seconds. n = 1 specimen per condition: these are
descriptions of two sheets, not a statistic, and the text says so.

    python compare_sheets.py --healthy out/fits/s4_live_r5_Efixed_mask_delay --hcm out/fits/hcm_r1_Efixed_mask_delay
"""
import argparse, json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import recording as R
RED, BLUE, WHITE, GREY = "#e0604a", "#4a90e0", "#f2f2f2", "#8a8a8a"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "axes.edgecolor": GREY, "axes.labelcolor": WHITE, "xtick.color": WHITE, "ytick.color": WHITE,
                     "text.color": WHITE, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def sheet(fit_dir, specimen, beat):
    z = np.load(os.path.join(fit_dir, "params.npz"))
    cfg = json.load(open(os.path.join(fit_dir, "fit.json")))["config"]
    inter = z["interior"].astype(bool) if "interior" in z.files else np.ones(z["g"].shape[0], bool)
    rec = R.load(specimen=specimen)
    win = R.beat_window(rec, beat); A, u = R.window_affine(rec, win)
    sh = R.shortening(A).numpy()                     # [T,C]
    peak_cell = sh.max(0)
    t0, tr, du, td = z["clock"][0], np.exp(z["clock"][1]), np.exp(z["clock"][2]), np.exp(z["clock"][3])
    T = len(win["frames"]); t = np.arange(T)
    s = 1 / (1 + np.exp(-(t - t0) / tr)) / (1 + np.exp(-(t0 + du - t) / td)); gam = (s - s[0]) / (1 - s[0])
    phi = z["phi"][inter]
    order = float(np.abs(np.mean(np.exp(2j * phi))))
    d = dict(specimen=specimen, n_cells=int(inter.sum()), n_all=int(z["g"].shape[0]),
             g=z["g"][inter], delay_s=z["delay"][inter] * R.DT_S, phi=phi, peak_cell=peak_cell[inter],
             logE=z["logE"][inter], E_free=bool(np.std(z["logE"]) > 1e-6),
             E_shrink=cfg.get("E_shrink", 0.0),
             gamma=gam, t_s=t * R.DT_S, axis_order=order,
             clock=dict(t0_s=float(t0 * R.DT_S), rise_s=float(tr * R.DT_S), dur_s=float(du * R.DT_S), decay_s=float(td * R.DT_S)),
             mean_curve=sh.mean(1), fit_dir=fit_dir)
    return d


def q(v):
    return dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                sd=float(np.std(v)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--healthy", required=True); ap.add_argument("--hcm", required=True)
    ap.add_argument("--beat-healthy", type=int, default=3); ap.add_argument("--beat-hcm", type=int, default=3)
    args = ap.parse_args()
    H = sheet(args.healthy, "healthy", args.beat_healthy); D = sheet(args.hcm, "hcm", args.beat_hcm)
    table = {}
    for name, S in (("healthy", H), ("hcm", D)):
        table[name] = dict(n_cells_interior=S["n_cells"], n_cells=S["n_all"], g=q(S["g"]), delay_s=q(S["delay_s"]),
                           logE=q(S["logE"]), E_shrink=S["E_shrink"],
                           recorded_peak_shortening=q(S["peak_cell"]), axis_order=S["axis_order"], clock=S["clock"],
                           mean_peak_shortening=float(S["mean_curve"].max()),
                           fraction_cells_g_below_0_01=float((S["g"] < 0.01).mean()))
    print(f"{'':<34s}{'healthy':>12s}{'HCM':>12s}")
    rows = [("interior cells (of all)", f"{H['n_cells']}/{H['n_all']}", f"{D['n_cells']}/{D['n_all']}"),
            ("recorded peak shortening, mean", f"{H['mean_curve'].max():.4f}", f"{D['mean_curve'].max():.4f}"),
            ("recorded peak shortening/cell, median", f"{np.median(H['peak_cell']):.4f}", f"{np.median(D['peak_cell']):.4f}"),
            ("fitted g, median", f"{np.median(H['g']):.4f}", f"{np.median(D['g']):.4f}"),
            ("fitted g, p10 / p90", f"{np.percentile(H['g'],10):.3f} / {np.percentile(H['g'],90):.3f}", f"{np.percentile(D['g'],10):.3f} / {np.percentile(D['g'],90):.3f}"),
            ("cells with g < 0.01 (silent)", f"{(H['g']<0.01).mean():.2f}", f"{(D['g']<0.01).mean():.2f}"),
            ("clock delay sd (s)", f"{H['delay_s'].std():.3f}", f"{D['delay_s'].std():.3f}"),
            ("axis order (0 random, 1 parallel)", f"{H['axis_order']:.2f}", f"{D['axis_order']:.2f}"),
            ("fitted E, median (spec units; 80 = init)", f"{np.exp(np.median(H['logE'])):.0f}", f"{np.exp(np.median(D['logE'])):.0f}"),
            ("fitted E, p10 / p90", f"{np.exp(np.percentile(H['logE'],10)):.0f} / {np.exp(np.percentile(H['logE'],90)):.0f}",
             f"{np.exp(np.percentile(D['logE'],10)):.0f} / {np.exp(np.percentile(D['logE'],90)):.0f}"),
            ("log E sd (shrink lambda)", f"{H['logE'].std():.2f} ({H['E_shrink']:g})", f"{D['logE'].std():.2f} ({D['E_shrink']:g})"),
            ("clock rise / duration / decay (s)", " / ".join(f"{H['clock'][k]:.2f}" for k in ("rise_s", "dur_s", "decay_s")),
             " / ".join(f"{D['clock'][k]:.2f}" for k in ("rise_s", "dur_s", "decay_s")))]
    for a, b, c in rows:
        print(f"{a:<34s}{b:>12s}{c:>12s}")

    withE = H["E_free"] and D["E_free"]
    fig, ax = plt.subplots(1, 5 if withE else 4, figsize=(22 if withE else 18, 4.4), gridspec_kw=dict(wspace=0.32))
    keys = [("peak_cell", "recorded peak shortening per cell (strain)"), ("g", "fitted g per cell (strain at full activation)"),
            ("delay_s", "fitted clock delay per cell (s)")]
    if withE:
        keys.append(("logE", f"fitted log E per cell (80 = init; shrink lambda {H['E_shrink']:g})"))
    for a, (key, xl) in zip(ax[:len(keys)], keys):
        lo = min(H[key].min(), D[key].min()); hi = max(np.percentile(H[key], 99), np.percentile(D[key], 99))
        bins = np.linspace(lo, hi, 36)
        a.hist(H[key], bins, histtype="step", lw=2, color=BLUE, density=True, label=f"healthy ({H['n_cells']} cells)")
        a.hist(D[key], bins, histtype="step", lw=2, color=RED, density=True, label=f"HCM ({D['n_cells']} cells)")
        a.set_xlabel(xl); a.set_ylabel("density over interior cells")
    ax[0].legend(loc="upper right", fontsize=9)
    a = ax[-1]
    a.plot(H["t_s"], H["gamma"], color=BLUE, lw=2, label="healthy clock")
    a.plot(D["t_s"], D["gamma"], color=RED, lw=2, label="HCM clock")
    a.plot(H["t_s"], H["mean_curve"] / H["mean_curve"].max(), color=BLUE, lw=1, ls="--", label="healthy recorded (normalised)")
    a.plot(D["t_s"], D["mean_curve"] / D["mean_curve"].max(), color=RED, lw=1, ls="--", label="HCM recorded (normalised)")
    a.set_xlabel("time in the beat window (s)"); a.set_ylabel("fitted clock gamma(t), 0-1\n(dashed: recorded mean shortening, normalised)")
    a.legend(loc="upper right", fontsize=8)
    for a, s in zip(ax, "ABCDE"):
        a.text(-0.02, 1.01, s, transform=a.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")
    os.makedirs(os.path.join(HERE, "out", "figures"), exist_ok=True)
    name = "fig5_healthy_vs_hcm" + ("_withE" if withE else "")
    fig.savefig(os.path.join(HERE, "out", "figures", f"{name}.png"), dpi=130, bbox_inches="tight")
    json.dump(table, open(os.path.join(HERE, "out", f"compare_sheets{'_withE' if withE else ''}.json"), "w"), indent=1)
    print(f"  -> out/figures/{name}.png")


if __name__ == "__main__":
    main()
