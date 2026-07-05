import json, os
dirs = [
    "embryo_1B_b07_s0_ctrl_cf0_m5e4",
    "embryo_1B_b07_s1_cf0p03_m5e4",
    "embryo_1B_b07_s2_cf0p07_m5e4",
    "embryo_1B_b07_s3_cf0p05_m3e4",
    "embryo_1B_b07_s4_cf0p04_m4e4",
    "embryo_1B_b07_s5_cf0p05_fieldmass",
    "embryo_1B_b07_s6_cf0p05_div",
    "embryo_1B_b07_s7_cf0p10_dense",
]
base = "archive"
for d in dirs:
    p = os.path.join(base, d, "metrics.json")
    if not os.path.exists(p):
        print(d, "NO METRICS")
        continue
    m = json.load(open(p))
    print("%-38s esc=%.4f rmax=%.3f deform=%.4f coll=%.4f nn=%.4f n=%s migr=%.3f flow=%.5f" % (
        d, m.get("escape", -1), m.get("r_cell_max", -1), m.get("deform", -1),
        m.get("collapsed", -1), m.get("nn_min", -1), m.get("n_cells", "?"),
        m.get("migration", -1), m.get("flow", -1)))
