import json, glob, os

# 1) spot-check b07 scorecard/metrics vs the analysis table
slots = ['s3_fast_mass4x','s7_quiescent_ctrl','s1_mass10x','s5_combo']
for s in slots:
    ds = glob.glob(f'archive/embryo_1B_b07_*{s}')
    if not ds:
        print(s, 'NO ARCHIVE'); continue
    d = ds[0]
    m = json.load(open(os.path.join(d,'metrics.json')))
    def g(*ks):
        for k in ks:
            if k in m: return m[k]
        return None
    print(f'--- {s} ---')
    for k in ['collapsed','escape','nn_min','accel','deform_rms','fourier_m2','fourier_m3','speed','msd','polar_order','deform_cell_corr','net_circulation']:
        if k in m: print(f'  {k:18s} {m[k]}')
    print('  keys:', sorted(m.keys())[:25])
