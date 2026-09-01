"""Is K's collapse OPTIMISATION or IDENTIFIABILITY? Freeze one, fit the other."""
import sys, os; sys.path[:0]=['.','../../src']
import torch, numpy as np, zarr, yaml, metrics, plexus.operators, ops_toy, ops_known_ode
from plexus.models.registry import get_operator
dev='cuda'; d='log/toy2d_fine_noed_simple_p1_none'
kf = next(o for o in yaml.safe_load(open('config/toy2d_fine.yaml'))['operators'] if o['op']=='kuramoto_field')
gen = get_operator('kuramoto_field', variant='phase')({**kf,'_at':'v'}, device=dev)
_, om_true, mask = gen._build(torch.zeros(1024,1024, device=dev))
mask = (mask > 0).float(); K_true = float(kf['K'])
v = torch.as_tensor(np.asarray(zarr.open(d+'/field.zarr','r')['v']['grid']), device=dev).float()
n = v.shape[0]; stride = round(1200/(n-1)); train = v[:int(870*n/1200)]
cls = get_operator('kuramoto_known_ode', variant='phase_fit')

class F: pass
class Hh: pass
def make(omega, K):
    op = cls({'_at':'v','dt':kf['dt'],'substeps':kf['substeps']}, device=dev)
    op.bind((1024,1024), mask); op.omega.data.copy_(omega); op.K.data.fill_(K)
    return op

def rollout_loss(op, H, j, horizon=2):
    s = train[j].clone(); tot = 0.0
    for k in range(horizon):
        for _ in range(stride):
            H.fields['v'].grid = s; op.forward(H); s = H.fields['v'].grid
        tot = tot + ((s - train[j+k+1])**2).mean()
    return tot / horizon

def fit(op, params, n_iter, lr, tag):
    H = Hh(); H.fields = {'v': F()}
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9,0.95))
    for it in range(n_iter):
        j = int(torch.randint(0, train.shape[0]-2, (1,)))
        loss = rollout_loss(op, H, j)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 32.0); opt.step()
        if it % 100 == 0:
            print("   %s it %4d  loss %.4e  K %.4f" % (tag, it, float(loss), float(op.K)), flush=True)
    return op

print("ABLATION 1 -- omega FROZEN AT THE TRUTH, fit K alone (1 parameter).")
op = make(om_true, 0.0)
op.omega.requires_grad_(False)
fit(op, [op.K], 600, 0.03, "K-only")
print("   RESULT: K = %.4f   (true %.2f)" % (float(op.K), K_true))
print()
print("ABLATION 2 -- K FROZEN AT THE TRUTH, fit omega alone (free per-pixel).")
op2 = make(torch.zeros_like(om_true), K_true)
op2.K.requires_grad_(False)
fit(op2, [op2.omega], 600, 0.01, "om-only")
r = metrics.recovery(om_true[mask>0].cpu().numpy(), op2.omega.detach()[mask>0].cpu().numpy())
print("   RESULT: omega R^2 %.4f  pearson %.4f  mean %.5f (true %.5f)"
      % (r["r2"], r["pearson"], r["learned_mean"], r["gt_mean"]))
