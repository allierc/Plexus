import json, glob
def last(d, k):
    v = d.get(k)
    return v[-1] if isinstance(v, list) else v
for name in ["s5_selfattr","s0_xr_g20_s1","s1_xr_g20_s2","s2_xr_g10_s1","s3_xr_g10_s2","s6_sharp_g10","s7_ctrl","s4_asym"]:
    p = glob.glob("archive/embryo_1E_b29_%s/scorecard.json" % name)
    if not p:
        print(name, "NO scorecard"); continue
    d = json.load(open(p[0]))
    part = d.get('partition', {}); org = d.get('organization', {})
    m = glob.glob("archive/embryo_1E_b29_%s/metrics.json" % name)
    md = json.load(open(m[0])) if m else {}
    print("%-16s seg=%.3f mi_x=%.3f contact=%.3f | collapsed=%s nn_min=%.4f escape=%.4f" % (
        name, last(part,'segregation_index'), last(part,'mi_type_x'),
        last(org,'contact_same'), md.get('collapsed'), md.get('nn_min',-1), md.get('escape',-1)))
