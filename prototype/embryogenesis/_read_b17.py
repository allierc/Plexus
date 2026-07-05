import json, glob, os
base = '/workspace/Plexus/prototype/embryogenesis/archive'
def g(sc, fam, key):
    # scorecard.json: fam -> key -> list of 5 (5/25/50/75/100)
    try:
        return sc[fam][key]
    except Exception:
        return None
for d in sorted(glob.glob(os.path.join(base, 'embryo_1C_b17_s*'))):
    name = d.split('_', 3)[3]
    m = json.load(open(os.path.join(d, 'metrics.json')))
    sc = json.load(open(os.path.join(d, 'scorecard.json')))
    esc = m.get('escape'); rcm = m.get('r_cell_max'); nc = m.get('n_cells')
    coll = m.get('collapsed'); nnm = m.get('nn_min')
    # final-frame fourier + deform + circ from scorecard shape family
    def last(fam, key):
        v = g(sc, fam, key)
        if isinstance(v, list): return round(v[-1], 5)
        return v
    print(f"== {name}")
    print(f"   escape={esc} collapsed={coll} nn_min={nnm} r_cell_max={round(rcm,4) if rcm else rcm} n={nc}")
    print(f"   deform_rms={last('shape','deform_rms')} circ={last('shape','circularity')} area={last('shape','area')}")
    print(f"   f_m1={last('shape','fourier_m1')} f_m2={last('shape','fourier_m2')} f_m3={last('shape','fourier_m3')}")
    # trajectory of deform and f_m2
    print(f"   deform_traj={g(sc,'shape','deform_rms')}")
    print(f"   f_m2_traj={g(sc,'shape','fourier_m2')}")
