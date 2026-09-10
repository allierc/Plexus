"""rounds_table -- every live round side by side: held-out R^2 (beats 1 and 2, the clean ones;
beat 0 starts mid-contraction at frame 0 and violates the rest-start premise), in-sample R^2,
per-cell agreement at the peak, and what the parameters did (spread of log E, median g)."""
import glob, json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "out", "fits", "s4_*", "s4_score.json"))):
    d = json.load(open(f)); r = d["results"]; tag = os.path.basename(os.path.dirname(f))
    ho = [r[k]["model"] for k in ("1", "2") if k in r]; fit = r.get("3", {}).get("model", {})
    rep = [r[k]["replay"] for k in ("1", "2") if k in r]
    z = np.load(os.path.join(os.path.dirname(f), "params.npz"))
    cfg = json.load(open(os.path.join(os.path.dirname(f), "fit.json")))["config"]
    rows.append(dict(tag=tag, free=cfg["free"], clock=cfg.get("clock_mode", "sigmoid"), E_shrink=cfg.get("E_shrink", 0),
                     ho_r2A=np.mean([m["r2_A"] for m in ho]), ho_r2u=np.mean([m["r2_u"] for m in ho]),
                     fit_r2A=fit.get("r2_A", np.nan), fit_r2u=fit.get("r2_u", np.nan),
                     ho_short=np.mean([m["shortening_corr"] for m in ho]), ho_axis=np.mean([m["axis_agreement"] for m in ho]),
                     rep_r2A=np.mean([m["r2_A"] for m in rep]), logE_sd=float(np.std(z["logE"])), g_med=float(np.median(z["g"]))))
print(f"{'round':<36s}{'free':<18s}{'clock':<8s}{'Eshr':>5s}{'held-out R2A':>13s}{'R2u':>6s}{'fit R2A':>8s}{'R2u':>6s}{'short r':>8s}{'axis':>6s}{'replay':>7s}{'logE sd':>8s}{'g med':>7s}")
for r in rows:
    print(f"{r['tag']:<36s}{r['free']:<18s}{r['clock']:<8s}{r['E_shrink']:>5.1f}{r['ho_r2A']:>13.3f}{r['ho_r2u']:>6.3f}{r['fit_r2A']:>8.3f}{r['fit_r2u']:>6.3f}{r['ho_short']:>8.3f}{r['ho_axis']:>6.3f}{r['rep_r2A']:>7.3f}{r['logE_sd']:>8.2f}{r['g_med']:>7.4f}")
json.dump(rows, open(os.path.join(HERE, "out", "rounds_table.json"), "w"), indent=1, default=float)
