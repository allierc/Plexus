import json, glob, os
base="/workspace/Plexus/prototype/embryogenesis/archive"
dirs=sorted(glob.glob(base+"/embryo_base_eb_b01_*"))
keys=["n_cells","collapsed","nn_min","nn_mean","escape","accel","n_div_events",
      "circularity","deform_rms","fourier_m2","fourier_m3","shape_index",
      "gr_peak","nn_cv","density_cv","polar_order","msd","speed",
      "net_circulation","segregation_index","t1_rate"]
rows=[]
for d in dirs:
    name=os.path.basename(d).replace("embryo_base_eb_b01_","")
    m=json.load(open(d+"/metrics.json"))
    rows.append((name,m))
# header
hdr="slot".ljust(16)+ "".join(k[:9].rjust(11) for k in keys)
print(hdr)
for name,m in rows:
    line=name.ljust(16)+"".join((f"{m.get(k,'-')}").rjust(11) for k in keys)
    print(line)

print("\n=== evolution: n_cells, collapsed-proxy(nn_cv), deform_rms across 5/25/50/75/100 ===")
for d in dirs:
    name=os.path.basename(d).replace("embryo_base_eb_b01_","")
    sc=json.load(open(d+"/scorecard.json"))["evolution"]
    print(name)
    for k in ["n_cells","nn_cv","deform_rms","fourier_m2","gr_peak","msd"]:
        print("   ",k.ljust(12),sc.get(k))
