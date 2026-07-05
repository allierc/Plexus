import json, glob, os
rows=[]
for d in sorted(glob.glob('archive/embryo_1B_b08_s*')):
    m=json.load(open(os.path.join(d,'metrics.json')))
    sc=json.load(open(os.path.join(d,'scorecard.json')))
    ev=sc['evolution']
    name=d.split('_',3)[-1]
    rows.append((name, m))
    print(f"=== {os.path.basename(d)} ===")
    print(f"  collapsed={m['collapsed']} escape={m['escape']} nn_min={m['nn_min']} accel={m['accel']:.5f} speed={m['speed']:.5f}")
    print(f"  deform_rms={m['deform_rms']} circ={m['circularity']} shape_idx={m['shape_index']}")
    print(f"  fourier m1={m['fourier_m1']} m2={m['fourier_m2']} m3={m['fourier_m3']} m4={m['fourier_m4']}")
    print(f"  deform_cell_corr={m['deform_cell_corr']} stress_cell_corr={m['stress_cell_corr']} flow_deform_lag={m['flow_deform_lag']}")
    print(f"  msd={m['msd']} polar={m['polar_order']} net_circ={m['net_circulation']} enstrophy={m['enstrophy']} t1={m['t1_rate']}")
    print(f"  traj deform_rms: {ev['deform_rms']}")
    print(f"  traj fourier_m2: {ev['fourier_m2']}")
    print(f"  traj circularity: {ev['circularity']}")
    print(f"  traj deform_cell_corr: {ev['deform_cell_corr']}")
