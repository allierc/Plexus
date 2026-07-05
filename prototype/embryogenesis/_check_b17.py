import sys, yaml, torch
sys.path.insert(0, '/workspace/Plexus/src')
specs = ['embryo_1E_split_xrep','embryo_1E_split_selfagg','embryo_1E_split_combo',
         'embryo_1E_split_hin','embryo_1E_mix_xrep']
for f in specs:
    yaml.safe_load(open('specs/%s.yaml' % f))
    print('YAML OK', f)

import plexus.schema as S
from plexus.engine import build

for f, layout in [('embryo_1E_split_xrep','split'), ('embryo_1E_mix_xrep','mixed')]:
    sim = S.load('specs/%s.yaml' % f)
    H = build(sim, device='cpu')
    lvl = H.levels['agent'] if hasattr(H, 'levels') else None
    # find agent level
    ag = None
    for name in dir(H):
        pass
    ag = H['agent'] if hasattr(H, '__getitem__') else lvl
    x = ag.state[:ag.n, 0]
    nt = ag.node_type[:ag.n]
    xa = x[nt == 0]; xb = x[nt == 1]
    print(f'{f} [{layout}]: mean x_a={xa.mean():.4f} x_b={xb.mean():.4f}  '
          f'overlap-check a<b? {(xa.mean() < xb.mean()).item()}')
