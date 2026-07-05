import json, glob, os

dirs = sorted(glob.glob('archive/embryo_1E_b31_s*/'))
keys = ['segregation_index','mi_type_x','contact_same','interface_frac','mixing_entropy','nn_min']

def find_series(d, k):
    if k in d:
        return d[k]
    for fam, v in d.items():
        if isinstance(v, dict) and k in v:
            return v[k]
    return None

def fmt(s):
    if s is None: return 'MISSING'
    if isinstance(s, list):
        return '[' + ', '.join(f'{x:.4f}' if isinstance(x,(int,float)) else str(x) for x in s) + ']'
    return f'{s:.4f}' if isinstance(s,(int,float)) else str(s)

for d in dirs:
    name = os.path.basename(d.rstrip('/'))
    print('===', name, '===')
    scp = os.path.join(d, 'scorecard.json')
    mp = os.path.join(d, 'metrics.json')
    if os.path.exists(scp):
        sc = json.load(open(scp))
        for k in keys:
            print(f'  {k:18s}', fmt(find_series(sc, k)))
    else:
        print('  NO scorecard.json')
    if os.path.exists(mp):
        m = json.load(open(mp))
        print('  metrics:', {k: m[k] for k in ('collapsed','escape','nn_min','agent_escaped') if k in m})
