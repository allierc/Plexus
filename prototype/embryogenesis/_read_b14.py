import json, glob, os
for d in sorted(glob.glob('archive/embryo_1E_b14_s*')):
    p = os.path.join(d, 'metrics.json')
    if not os.path.exists(p):
        print(d, 'NO METRICS'); continue
    m = json.load(open(p))
    keys = ['n_cells','collapsed','nn_min','nn_mean','escape','r_cell_max','deform','flow','migration','segregation','accel']
    print(d.replace('archive/embryo_1E_b14_',''))
    print('  ' + '  '.join(f'{k}={m.get(k)}' for k in keys if k in m))
