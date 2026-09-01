"""LR and batch size, as asked -- and the fix the profile implies: fit the TRANSIENT."""
import sys; sys.path[:0]=['.','../../src']
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

def fit_K(lr, batch, n_iter, hi, horizon=2, seed=0):
    """omega frozen at the TRUTH; fit K alone. `hi` caps which start frames are sampled."""
    torch.manual_seed(seed)
    op = cls({'_at':'v','dt':kf['dt'],'substeps':kf['substeps']}, device=dev)
    op.bind((1024,1024), mask); op.omega.data.copy_(om_true); op.omega.requires_grad_(False)
    op.K.data.fill_(0.0)
    opt = torch.optim.AdamW([op.K], lr=lr, betas=(0.9,0.95))
    H=Hh(); H.fields={'v':F()}
    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        for _ in range(batch):                       # accumulate: the gradient of a mean
            j = int(torch.randint(0, hi, (1,)))
            s = train[j].clone(); L = 0.0
            for k in range(horizon):
                for _ in range(stride):
                    H.fields['v'].grid = s; op.forward(H); s = H.fields['v'].grid
                L = L + ((s - train[j+k+1])**2).mean()
            (L/(batch*horizon)).backward()
        torch.nn.utils.clip_grad_norm_([op.K], 32.0); opt.step()
    return float(op.K.detach())

print("ALL FRAMES (starts 0..%d) -- lr x batch, omega frozen at the truth. true K = 0.90" % (train.shape[0]-3))
print("   lr\\batch      1        4        8")
for lr in (0.005, 0.02, 0.05):
    row = [fit_K(lr, b, 200, train.shape[0]-3) for b in (1, 4, 8)]
    print("   %.3f     %s" % (lr, "  ".join("%7.4f" % x for x in row)), flush=True)
print()
print("TRANSIENT ONLY (starts 0..19) -- the same sweep, the only change is WHICH frames")
print("   lr\\batch      1        4        8")
for lr in (0.005, 0.02, 0.05):
    row = [fit_K(lr, b, 200, 20) for b in (1, 4, 8)]
    print("   %.3f     %s" % (lr, "  ".join("%7.4f" % x for x in row)), flush=True)
