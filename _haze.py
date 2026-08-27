import sys, yaml, tempfile, os
sys.path.insert(0,'src')
import numpy as np, torch
torch.cuda.set_device(1)
import plexus.operators, plexus.operators.mpm_warp
from plexus.schema import load
from plexus import engine as E
NG=192
def go(mf, cf, frames=200):
    s=yaml.safe_load(open('config/material/material_3d_water_dam_20m.yaml'))
    for o in s['operators']:
        if o['op']=='mpm_grid_update':
            o['mass_floor']=float(mf); o['csf_mass_floor']=float(cf)
    s['general']['n_frames']=frames; s['general']['record_cap']=2
    f=tempfile.NamedTemporaryFile('w',suffix='.yaml',delete=False); yaml.safe_dump(s,f); f.close()
    H,_=E.run(load(f.name), out_path=None, device='cuda:1', progress=False); os.unlink(f.name)
    lvl=H.level('mpm_particle')
    q=lvl.get('pos').detach().cpu().numpy(); v=lvl.get('vel').detach().cpu().numpy()
    idx=np.clip((q*NG).astype(np.int32),0,NG-1)
    flat=(idx[:,0].astype(np.int64)*NG+idx[:,1])*NG+idx[:,2]
    cnt=np.bincount(flat,minlength=NG**3); per=cnt[flat]
    bulk=per>=8
    yt=np.percentile(q[bulk,1],99.9)
    haze=(q[:,1] > yt+0.05) & (~bulk)
    return dict(yt=yt, n=int(haze.sum()), pct=haze.mean()*100,
                maxy=float(q[:,1].max()),
                vy=float(v[haze,1].mean()) if haze.any() else 0.0,
                lone=float((per<2).mean()*100))
print("RESULT  floors (mass / csf)        bulk top   haze n     haze %   max y   mean vy of haze  lone-cell %", flush=True)
for mf, cf, lbl in ((1.7e-14, 1.7e-14, "SCALED  1.7e-14 / 1.7e-14"),
                    (1e-10,   1e-8,    "DEFAULT 1e-10   / 1e-8  ")):
    r=go(mf,cf)
    print(f"RESULT  {lbl}   {r['yt']:8.3f}{r['n']:>9,}{r['pct']:>10.3f}%{r['maxy']:>8.3f}"
          f"{r['vy']:>17.3f}{r['lone']:>12.3f}%", flush=True)
