import json, glob, os
os.chdir(os.path.join(os.path.dirname(__file__), "archive"))
for d in sorted(glob.glob("embryo_INT_b28_s*")):
    p = os.path.join(d, "metrics.json")
    try:
        m = json.load(open(p))
    except Exception as e:
        print(d, "ERR", e); continue
    keys = ["escape","r_cell_max","collapsed","migration","segregation","deform","flow","n_cells"]
    print(d.replace("embryo_INT_b28_",""),
          {k: (round(m[k],4) if isinstance(m.get(k),float) else m.get(k)) for k in keys})
