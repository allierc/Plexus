import json, glob, os

dirs = sorted(glob.glob('/workspace/Plexus/prototype/embryogenesis/archive/embryo_1D_b22_s*'))
for d in dirs:
    print('=====', os.path.basename(d), '=====')
    try:
        m = json.load(open(d+'/metrics.json'))
        print('GATE:', {k: m.get(k) for k in ['collapsed','escape','nn_min','accel','n_cells']})
    except Exception as e:
        print('metrics err', e)
    try:
        s = json.load(open(d+'/scorecard.json'))
        def tr(fam, key):
            try:
                v = s[fam][key]
                if isinstance(v, list):
                    return [round(x,4) if isinstance(x,(int,float)) else x for x in v]
                return v
            except:
                return None
        for fam,key in [('flow','polar_order'),('flow','net_circulation'),('flow','msd'),
                        ('flow','speed'),('shape','deform_rms'),('shape','fourier_m2'),
                        ('shape','fourier_m3'),('shape','circularity')]:
            print(f'  {fam}.{key:16s}', tr(fam,key))
    except Exception as e:
        print('sc err', e)
