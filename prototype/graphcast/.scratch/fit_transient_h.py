"""Fit K on the transient at long horizon -- the combination the profile says should work."""
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
def fit(horizon, n_iter=150, lr=0.02, batch=2, seed=0):
    torch.manual_seed(seed)
    op = cls({'_at':'v','dt':kf['dt'],'substeps':kf['substeps']}, device=dev)
    op.bind((1024,1024), mask); op.omega.data.copy_(om_true); op.omega.requires_grad_(False)
    op.K.data.fill_(0.0)
    opt = torch.optim.AdamW([op.K], lr=lr, betas=(0.9,0.95)); H=Hh(); H.fields={'v':F()}
    hi = max(1, 20 - horizon)
    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        for _ in range(batch):
            j = int(torch.randint(0, hi, (1,)))
            s = v[j].clone(); L = 0.0
            for k in range(horizon):
                for _ in range(stride):
                    H.fields['v'].grid = s; op.forward(H); s = H.fields['v'].grid
                L = L + ((s - v[j+k+1])**2).mean()
            (L/(batch*horizon)).backward()
        torch.nn.utils.clip_grad_norm_([op.K], 32.0); opt.step()
    return float(op.K.detach())
print("FIT K on the transient, omega frozen at truth, lr 0.02 batch 2. true K = 0.90")
print("  horizon    K learned    %% of true")
for h in (2, 4, 8, 16):
    try:
        K = fit(h); print("  %5d      %8.4f     %5.1f%%" % (h, K, 100*K/0.9), flush=True)
    except torch.OutOfMemoryError:
        print("  %5d      OOM (%d states on the tape)" % (h, h*stride*12), flush=True)
        torch.cuda.empty_cache()
