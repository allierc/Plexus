import json, glob, os
slots = ["s0_g10_2x","s1_g10_3x","s2_g10_4x","s3_g20_2x","s4_g20_3x","s5_g20_4x","s6_slowfill_g10_4x","s7_ctrl_g10_nodiv"]
base = "/workspace/Plexus/prototype/embryogenesis/archive/embryo_INT_b32_"
for s in slots:
    d = base + s
    sc = json.load(open(d+"/scorecard.json"))
    m = json.load(open(d+"/metrics.json"))
    ev = sc["evolution"]; f = sc["final"]
    def tr(k): return "->".join(f"{x:.3f}" if isinstance(x,(int,float)) else str(x) for x in ev[k])
    print(f"\n==== {s} ====")
    print("  metrics.json gate:", json.dumps(m))
    print("  n_cells   :", tr("n_cells"))
    print("  seg_index :", tr("segregation_index"), " final", f["segregation_index"])
    print("  contact_sm:", tr("contact_same"))
    print("  interface :", tr("interface_frac"))
    print("  mix_entr  :", tr("mixing_entropy"))
    print("  mi_type_x :", tr("mi_type_x"))
    print("  nn_min    :", tr("nn_min"), " final", f["nn_min"])
    print("  deform_rms:", tr("deform_rms"))
    print("  circular  :", tr("circularity"))
    print("  speed     :", tr("speed"))
    print("  msd       :", tr("msd"))
    print("  n_div     :", f.get("n_div_events"), " div_angle", f.get("div_stress_angle"))
