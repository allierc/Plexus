import json, os
base='/workspace/Plexus/prototype/embryogenesis/archive'
slots=['s0_base_n44_ref','s1_confine1p0','s2_confine0p5','s1_confine0','s3_quiet_substrate',
       's5_drag0','s2_confine0_k0','s7_confine0_dense','s4_confined_dense','s0_baseline',
       's1_no_div','s6_spin0','s7_field_mass','s0_no_align']
keys=['n_cells','collapsed','nn_min','nn_mean','deform','flow','migration','segregation','accel','escape','r_cell_max']
for s in slots:
    d=os.path.join(base,'embryo_base_eb_b01_'+s,'metrics.json')
    if os.path.exists(d):
        m=json.load(open(d))
        row=' '.join(f"{k}={round(m[k],4) if isinstance(m.get(k),float) else m.get(k)}" for k in keys if k in m)
        print(f"{s:22s} {row}")
    else:
        print(f"{s:22s} MISSING")
