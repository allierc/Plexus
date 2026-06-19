import json
d = json.load(open("sweep_results.json"))
print("Total sweeps:", len(d["sweeps"]))
all_nan_count = 0
for i, s in enumerate(d["sweeps"]):
    losses = s["loss"]
    n_nan = sum(1 for x in losses if x is None or (isinstance(x,float) and x!=x))
    if n_nan == 16: all_nan_count += 1
print("All-NaN sweeps:", all_nan_count, "/", len(d["sweeps"]))
for i,s in enumerate(d["sweeps"]):
    losses = s["loss"]
    valid = [(s["values"][k], x, s["inner_mass"][k], s["n_mounds"][k], s.get("morph_score", [None]*16)[k]) for k,x in enumerate(losses) if x is not None and not (isinstance(x,float) and x!=x)]
    if valid:
        print(f"\nSweep {i} {s['param']}: {len(valid)} valid")
        for v,l,im,nm,ms in valid:
            try:
                print(f"  v={v} loss={l:.4f} inner={im:.4f} nm={nm} morph={ms}")
            except:
                print(f"  v={v} loss={l} inner={im} nm={nm} morph={ms}")
