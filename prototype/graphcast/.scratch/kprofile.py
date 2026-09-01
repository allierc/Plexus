"""The loss AND its gradient in K, at the true omega, averaged over many starts."""
import sys, os; sys.path[:0]=['.','../../src']
import torch, numpy as np, zarr, yaml, plexus.operators, ops_toy, ops_known_ode
from plexus.models.registry import get_operator
dev='cuda'; d='log/toy2d_fine_noed_simple_p1_none'
kf = next(o for o in yaml.safe_load(open('config/toy2d_fine.yaml'))['operators'] if o['op']=='kuramoto_field')
gen = get_operator('kuramoto_field', variant='phase')({**kf,'_at':'v'}, device=dev)
_, om_true, mask = gen._build(torch.zeros(1024,1024, device=dev)); mask=(mask>0).float()
v = torch.as_tensor(np.asarray(zarr.open(d+'/field.zarr','r')['v']['grid']), device=dev).float()
n=v.shape[0]; stride=round(1200/(n-1)); train=v[:int(870*n/1200)]
cls = get_operator('kuramoto_known_ode', variant='phase_fit')
class F: pass
class Hh: pass
import sys as _s
SETS = {"early 0..9 (consecutive)": list(range(10)),
        "spread 0..216 (step 24)":  list(range(0, 240, 24)),
        "late 130..238 (step 12)":  list(range(130, 240, 12))}
STARTS = []
def loss_and_grad(K, horizon=2):
    op = cls({'_at':'v','dt':kf['dt'],'substeps':kf['substeps']}, device=dev)
    op.bind((1024,1024), mask); op.omega.data.copy_(om_true); op.omega.requires_grad_(False)
    op.K.data.fill_(K)
    # ONE START AT A TIME, backward immediately. The gradient of a MEAN is the mean of the
    # gradients, so accumulating per start is exact -- and holding all 10 rollouts on the tape at
    # once is 720 stored 2x1024^2 states, which is what OOMed.
    H=Hh(); H.fields={'v':F()}; tot=0.0
    for j in STARTS:
        s = train[j].clone(); Lj = 0.0
        for k in range(horizon):
            for _ in range(stride):
                H.fields['v'].grid = s; op.forward(H); s = H.fields['v'].grid
            Lj = Lj + ((s - train[j+k+1])**2).mean()
        (Lj/(len(STARTS)*horizon)).backward()
        tot += float(Lj)
    return tot/(len(STARTS)*horizon), float(op.K.grad)
print("true omega, horizon 2 -- WHERE the loss in K is minimised, by which starts are used")
print("  %-26s %s" % ("start set", "  ".join("K=%.1f" % k for k in (0.1,0.3,0.5,0.7,0.9,1.1))))
for name, st in SETS.items():
    STARTS[:] = st
    Ls = [loss_and_grad(K)[0] for K in (0.1,0.3,0.5,0.7,0.9,1.1)]
    best = (0.1,0.3,0.5,0.7,0.9,1.1)[int(np.argmin(Ls))]
    print("  %-26s %s   -> min at K=%.1f" % (name, "  ".join("%5.2f" % (x*1e3) for x in Ls), best))
print()
print("  (loss x 1e3; true K = 0.90)")
