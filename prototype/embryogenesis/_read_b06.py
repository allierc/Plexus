import json, glob, os
keys=['n_cells','collapsed','nn_min','nn_mean','deform','flow','migration','segregation','escape','r_cell_max','accel']
for d in sorted(glob.glob('archive/embryo_1A_eb_b06_s*')):
    p=os.path.join(d,'metrics.json')
    if not os.path.exists(p):
        print(os.path.basename(d),'NO metrics.json'); continue
    m=json.load(open(p))
    print(os.path.basename(d))
    print('  '+' '.join(f'{k}={m.get(k)}' for k in keys))
