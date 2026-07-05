import json, glob, os
keys=['n_cells','collapsed','nn_min','deform','escape','r_cell_max','flow','migration','segregation','accel']
for d in sorted(glob.glob('archive/embryo_1C_b09_s*')):
    m=json.load(open(os.path.join(d,'metrics.json')))
    name=d.split('/')[-1]
    print(name)
    print('  '+' '.join(f"{k}={m.get(k)}" for k in keys))
