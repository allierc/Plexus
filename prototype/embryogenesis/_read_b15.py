import json, glob, os
for d in sorted(glob.glob('archive/embryo_1C_b15_s*')):
    name = os.path.basename(d)
    try:
        m = json.load(open(os.path.join(d, 'metrics.json')))
        pick = {k: m.get(k) for k in ('collapsed', 'escape', 'nn_min', 'accel', 'n_cells')}
    except Exception as e:
        pick = {'ERR': str(e)}
    print(name, pick)
