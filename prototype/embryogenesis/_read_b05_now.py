import json, glob, os

def series(sc, fam, key):
    try:
        v = sc[fam][key]
        if isinstance(v, list):
            return [round(x, 5) if isinstance(x, (int, float)) else x for x in v]
        return v
    except Exception:
        return None

for d in sorted(glob.glob("archive/embryo_1A_b05_s*")):
    print("=" * 5, os.path.basename(d), "=" * 5)
    try:
        m = json.load(open(d + "/metrics.json"))
    except Exception as e:
        print("  metrics err", e); m = {}
    for k in ["collapsed", "escape", "agent_escaped", "nn_min", "r0", "accel",
              "vmax", "n_cells", "n_div_events", "seconds", "r_cell_max"]:
        if k in m:
            print(f"  {k}={m[k]}")
    try:
        sc = json.load(open(d + "/scorecard.json"))
    except Exception as e:
        print("  scorecard err", e); sc = None
    if sc:
        for fam, key in [("shape", "circularity"), ("shape", "deform_rms"),
                         ("shape", "fourier_m2"), ("shape", "fourier_m3"),
                         ("organization", "nn_min"), ("organization", "nn_mean"),
                         ("organization", "gr_peak_r"), ("organization", "gr_peak"),
                         ("organization", "nn_cv"),
                         ("flow", "speed"), ("flow", "msd"),
                         ("flow", "polar_order"), ("flow", "net_circulation")]:
            s = series(sc, fam, key)
            if s is not None:
                print(f"  {fam}.{key}={s}")
