import json, glob, os

base = "/workspace/Plexus/prototype/embryogenesis/archive"
dirs = sorted(glob.glob(os.path.join(base, "embryo_1B_b12_s*")))

def g(d, path):
    cur = d
    for p in path.split('.'):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

for dd in dirs:
    name = os.path.basename(dd)
    print("="*70)
    print(name)
    mp = os.path.join(dd, "metrics.json")
    sp = os.path.join(dd, "scorecard.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        gate = {k: m.get(k) for k in ['collapsed','escape','nn_min','accel','speed','vmax','n_cells'] if k in m}
        print("  GATE:", gate)
        # sometimes nested
        if not gate:
            print("  metrics keys:", list(m.keys())[:20])
    if os.path.exists(sp):
        s = json.load(open(sp))
        # scorecard structure: family -> metric -> list of 5 values
        def series(fam, met):
            v = g(s, fam+'.'+met)
            return v
        for fam, mets in [('shape',['deform_rms','fourier_m1','fourier_m2','fourier_m3','circularity','shape_index']),
                          ('flow',['speed','polar_order','net_circulation','enstrophy','msd']),
                          ('coupling',['deform_cell_corr'])]:
            for met in mets:
                v = series(fam, met)
                if v is not None:
                    if isinstance(v, list):
                        print(f"  {fam}.{met}: "+", ".join(f"{x:.5g}" if isinstance(x,(int,float)) else str(x) for x in v))
                    else:
                        print(f"  {fam}.{met}: {v}")
