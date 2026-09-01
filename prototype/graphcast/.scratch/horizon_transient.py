"""Longer horizon ON THE TRANSIENT: does the loss MINIMUM in K move to the true 0.90?"""
import sys; sys.path[:0]=['.','../../src']
import torch, numpy as np, zarr, yaml, plexus.operators, ops_toy, ops_known_ode
from plexus.models.registry import get_operator
dev='cuda'; d='log/toy2d_fine_noed_simple_p1_none'
kf = next(o for o in yaml.safe_load(open('config/toy2d_fine.yaml'))['operators'] if o['op']=='kuramoto_field')
gen = get_operator('kuramoto_field', variant='phase')({**kf,'_at':'v'}, device=dev)
_, om_true, mask = gen._build(torch.zeros(1024,1024, device=dev)); mask=(mask>0).float()
v = torch.as_tensor(np.asarray(zarr.open(d+'/field.zarr','r')['v']['grid']), device=dev).float()
stride = round(1200/(v.shape[0]-1))
cls = get_operator('kuramoto_known_ode', variant='phase_fit')
class F: pass
class Hh: pass
KS = (0.1, 0.3, 0.5, 0.7, 0.9, 1.1)

def build(K):
    op = cls({'_at':'v','dt':kf['dt'],'substeps':kf['substeps']}, device=dev)
    op.bind((1024,1024), mask); op.omega.data.copy_(om_true); op.omega.requires_grad_(False)
    op.K.data.fill_(K); return op

def loss_at(K, starts, horizon):
    op = build(K); H=Hh(); H.fields={'v':F()}; tot = 0.0
    with torch.no_grad():
        for j in starts:
            s = v[j].clone()
            for k in range(horizon):
                for _ in range(stride):
                    H.fields['v'].grid = s; op.forward(H); s = H.fields['v'].grid
                tot += float(((s - v[j+k+1])**2).mean())
    return tot/(len(starts)*horizon)

print("TRANSIENT (frames 0..20), omega frozen at truth. WHERE is the loss in K minimised?")
print("  horizon  n_starts   " + "  ".join("K=%.1f" % k for k in KS) + "     min")
for H_ in (1, 2, 4, 8, 16):
    st = list(range(0, max(1, 20 - H_)))
    Ls = [loss_at(K, st, H_) for K in KS]
    print("  %5d   %6d    %s   -> K=%.1f"
          % (H_, len(st), "  ".join("%5.2f" % (x*1e3) for x in Ls), KS[int(np.argmin(Ls))]),
          flush=True)
print("  (loss x 1e3; true K = 0.90)")
