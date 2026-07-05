import json, glob, os
keys = ["n_cells","collapsed","nn_min","nn_mean","deform","flow","migration","segregation","escape","r_cell_max","accel"]
for d in sorted(glob.glob("/workspace/Plexus/prototype/embryogenesis/archive/*eb_b05*")):
    p = os.path.join(d, "metrics.json")
    if os.path.exists(p):
        m = json.load(open(p))
        row = ' '.join(f"{k}={round(m[k],4) if isinstance(m.get(k),float) else m.get(k)}" for k in keys if k in m)
        print(f"{os.path.basename(d)[-24:]:26s} {row}")
