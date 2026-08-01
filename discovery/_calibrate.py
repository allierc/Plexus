"""Calibrate the pattern to Okuda: ~5 spots on a 2000-cell ball, then freeze.

The knob is NOT chi. Wavelength goes as sqrt(D / reaction rate), so a coarser pattern can be had
either by raising diffusion or by SLOWING THE REACTION -- and raising diffusion is what destabilises
the explicit step (chi 13 saturated the integrator in the earlier sweep). Slowing the reaction moves
the same ratio in the same direction and makes the step MORE stable, not less. So sweep rd_rate.
"""
import sys, itertools, json, numpy as np
sys.path.insert(0,'/workspace/Plexus/prototype/Tyssue'); sys.path.insert(0,'/workspace/Plexus/src')
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
from pattern_scale import cell_graph, pattern_metrics
import torch

v,es,et,ef,nF = build_sphere_mesh(2000,5.0,0.0,0)
_,_,cen,_ = face_geometry_3d(torch.as_tensor(v),torch.as_tensor(es),torch.as_tensor(et),torch.as_tensor(ef),nF)
cen=cen.numpy(); src,dst=cell_graph(es,et,ef,nF)
deg=np.bincount(src,minlength=nF).astype(float); deg[deg==0]=1

def run(chi, rate, steps, F=0.055, kk=0.062, d_a=0.08, d_h=0.16, seed_frac=0.06, seed=0):
    rng=np.random.default_rng(seed); a=0.04*rng.random(nF); u=np.ones(nF)
    n=rng.random(nF)<seed_frac; a[n]=0.5; u[n]=0.25
    Da,Du=d_a*chi,d_h*chi
    for s in range(steps):
        la=(np.bincount(src,weights=a[dst],minlength=nF)/deg)-a
        lu=(np.bincount(src,weights=u[dst],minlength=nF)/deg)-u
        uaa=u*a*a
        a=a+Da*la+rate*(uaa-(F+kk)*a); u=u+Du*lu+rate*(-uaa+F*(1.0-u))
        a=np.clip(a,0,10); u=np.clip(u,0,10)
    return a

TARGET_SPOTS = 5
print(f"CALIBRATING to Okuda's ~{TARGET_SPOTS} spots on a {nF}-cell ball")
print(f"target spacing for {TARGET_SPOTS} spots = R*sqrt(4pi/k)/L")
L=float(np.mean(np.linalg.norm(cen[dst]-cen[src],axis=1)))
print(f"  mesh spacing {L:.4f}, target spot spacing {5.0*np.sqrt(4*np.pi/TARGET_SPOTS)/L:.1f} cells\n")
print(f"{'chi':>6}{'rd_rate':>9}{'steps':>7} | {'a_max':>7}{'n_spots':>9}{'spacing':>9}  verdict")
rows=[]
for chi, rate in itertools.product((1.3, 2.6, 5.2), (1.0, 0.3, 0.1, 0.03)):
    steps=int(3000/max(rate,0.03)) if rate<1 else 3000
    steps=min(steps, 30000)
    a=run(chi,rate,steps)
    if a.max()<0.2:
        print(f"{chi:6.1f}{rate:9.2f}{steps:7d} | {a.max():7.3f}{'-':>9}{'-':>9}  DEAD"); continue
    m=pattern_metrics(a,es,et,ef,nF,cen=cen)
    ns, sp = m['n_spots'], m['spot_spacing_cells']
    v_ = "<-- TARGET" if TARGET_SPOTS-2 <= ns <= TARGET_SPOTS+3 else ""
    print(f"{chi:6.1f}{rate:9.2f}{steps:7d} | {a.max():7.3f}{ns:9d}{(sp if sp else 0):9.2f}  {v_}")
    rows.append(dict(chi=chi, rd_rate=rate, steps=steps, n_spots=ns, spacing=sp, a_max=float(a.max())))
json.dump(rows, open("_calibrate.json","w"), indent=1)
