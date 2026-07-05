import json, glob, os

dirs = sorted(glob.glob('archive/embryo_1B_b07_s*'))
for d in dirs:
    name = os.path.basename(d)
    mp = os.path.join(d, 'metrics.json')
    sp = os.path.join(d, 'scorecard.json')
    print('='*70)
    print(name)
    if os.path.exists(mp):
        m = json.load(open(mp))
        gate = {k: m.get(k) for k in ['collapsed','escape','nn_min','accel','vmax','n_cells','n_div_events'] if k in m}
        print('  GATE:', gate)
    else:
        print('  NO metrics.json')
    if os.path.exists(sp):
        sc = json.load(open(sp))
        # scorecard structure: family -> metric -> list of 5 values? print shape family
        def get(path):
            cur = sc
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
            return cur
        # try to find shape metrics
        shape = sc.get('shape', {})
        flow = sc.get('flow', {})
        coup = sc.get('coupling', {})
        for fam, keys in [('shape',['deform_rms','fourier_m1','fourier_m2','fourier_m3','circularity','shape_index']),
                          ('flow',['speed','polar_order','net_circulation','enstrophy','msd']),
                          ('coupling',['deform_cell_corr','flow_deform_lag','stress_cell_corr'])]:
            fd = sc.get(fam, {})
            for k in keys:
                v = fd.get(k)
                if v is not None:
                    if isinstance(v, list):
                        print(f'    {fam}.{k}: ' + ' '.join(f'{x:.5g}' if isinstance(x,(int,float)) else str(x) for x in v))
                    else:
                        print(f'    {fam}.{k}: {v}')
    else:
        print('  NO scorecard.json')
