import json, glob, os
for d in sorted(glob.glob('archive/embryo_INT_b24_s*')):
    m = json.load(open(os.path.join(d, 'metrics.json')))
    print('%-32s escape=%.4f r_max=%.3f seg=%.4f deform=%.4f n=%s migr=%.3f coll=%.4f nnmin=%.4f' % (
        os.path.basename(d), m.get('escape', -1), m.get('r_cell_max', -1), m.get('segregation', -1),
        m.get('deform', -1), m.get('n_cells', -1), m.get('migration', -1), m.get('collapsed', -1), m.get('nn_min', -1)))
