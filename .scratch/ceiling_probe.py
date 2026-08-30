import sys, os, tempfile, yaml, numpy as np
sys.path.insert(0,'src')
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
s=yaml.safe_load(open("config/material/si_water_splash.yaml"))
s["general"]["n_frames"]=600; s["general"]["record_cap"]=10000
s["sets"]["mpm_particle"]["per_parent"]=300000
f=tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False); yaml.safe_dump(s,f); f.close()
CFL(f.name); sim=load(f.name); os.unlink(f.name)
W=0.30; snaps={}
def cb(H,t):
    if t in (350,400,450,500,550,600):
        L=H.level("mpm_particle")
        snaps[t]=(L.get("pos").detach().cpu().numpy().copy(),
                  L.get("vel").detach().cpu().numpy().copy())
E.run(sim,out_path=None,device=sys.argv[1],progress=False,on_frame=cb)
print(f"\n  PARTICLES IN THE TOP 2 cm OF A 0.3 m BOX -- are they the same ones?\n")
print(f"  {'frame':>7}{'n above 0.28':>14}{'median vy':>12}{'free-fall dv/50fr':>19}{'shared with f350':>18}")
base=None
for t in sorted(snaps):
    X,V=snaps[t]; m=X[:,1]>0.28; idx=set(np.where(m)[0].tolist())
    if base is None: base=idx
    sh=f"{100*len(idx&base)/max(len(base),1):.0f}%"
    print(f"  {t:>7}{int(m.sum()):>14}{np.median(V[m,1]) if m.any() else float('nan'):>12.4f}"
          f"{-9.81*50*8.3333e-4:>19.3f}{sh:>18}")
print()
