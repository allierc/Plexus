"""bisect the shrink: strip the op list down and watch the radius."""
import sys, copy, numpy as np
sys.path.insert(0,'/workspace/Plexus/discovery_okuda'); sys.path.insert(0,'/workspace/Plexus/prototype/Tyssue')
sys.path.insert(0,'/workspace/Plexus/src')
import plexus.operators, tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, tyssue_monolayer, ckpt  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
import yaml

CFG='/workspace/Plexus/config/okuda/p1_control.yaml'
raw = yaml.safe_load(open(CFG))
ALL = [o['op'] for o in raw['operators']]
print('operators in p1_control:', ALL)

def radii(keep, nf=20):
    d = copy.deepcopy(raw)
    d['general']['n_frames'] = nf
    d['operators'] = [o for o in d['operators'] if o['op'] in keep]
    if 'schedule' in d.get('general', {}):
        d['general']['schedule'] = [s for s in d['general']['schedule'] if s in keep or s in ('integrate','render','advect')]
    for k in ('schedule',):
        if k in d: d[k] = [s for s in d[k] if s in keep or s in ('integrate','render','advect')]
    import tempfile, os
    fn = tempfile.mktemp(suffix='.yaml'); yaml.safe_dump(d, open(fn,'w'))
    sim = S.load(fn); os.unlink(fn)
    Hf, out = engine_run(sim, device='cpu')
    pos = out['sets']['vertex']['pos']; Nv = Hf.level('vertex')._mesh['Nv']
    return [float(np.linalg.norm(pos[t][:Nv],axis=1).mean()) for t in range(nf+1)]

if __name__ == '__main__':
    which = sys.argv[1:] if len(sys.argv)>1 else None
    keep = ALL if which is None else ['seed_mesh_3d'] + which
    r = radii(keep)
    print(f'\nkeep = {keep}')
    print('  radius: ' + '  '.join(f'{x:.3f}' for x in r[:12]))
    print(f'  frame0 {r[0]:.4f} -> frame{len(r)-1} {r[-1]:.4f}   ratio {r[-1]/max(r[0],1e-9):.4f}')
