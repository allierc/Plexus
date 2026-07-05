import json, glob, os
for d in sorted(glob.glob('archive/embryo_INT_b25_*/')):
    m=json.load(open(os.path.join(d,'metrics.json')))
    name=os.path.basename(d.rstrip('/'))
    print(f"{name}: esc={m.get('escape',-1):.4f} r_max={m.get('r_cell_max',-1):.3f} "
          f"deform={m.get('deform',-1):.4f} migr={m.get('migration',-1):.3f} "
          f"seg={m.get('segregation',-1):.3f} flow={m.get('flow',-1):.5f} "
          f"coll={m.get('collapsed',-1):.4f} nn={m.get('nn_min',-1):.4f} n={m.get('n_cells',-1)}")
