import sys, os, tempfile, yaml, numpy as np, torch as T
sys.path.insert(0,'src')
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
n=sys.argv[1]
s=yaml.safe_load(open(f"config/material/{n}.yaml")); s["general"]["n_frames"]=5
s["general"]["record_cap"]=10
f=tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False); yaml.safe_dump(s,f); f.close()
CFL(f.name); sim=load(f.name); os.unlink(f.name)
H,_=E.run(sim,out_path=None,device=sys.argv[2],progress=False)
g=H.field("mpm_grid"); p=H.level("mpm_particle"); X=p.get("pos").detach().cpu().numpy()
V=float(p.p_vol[0])*p.n
print(f"\n  {n}: dx {g.dx*1000:.3f} mm  shape {g.shape}  ppc {g.dx**3/float(p.p_vol[0]):.2f}")
print(f"    m_p {float(p.mass[0]):.4e} kg   p_vol {float(p.p_vol[0]):.4e} m3 = {float(p.p_vol[0])*1e9:.3f} mm3")
print(f"    N {p.n:,}  ->  V = N*p_vol = {V*1e3:.4f} L   mass {V*1000:.4f} kg")
print(f"    cube side derived {V**(1/3)*1000:.2f} mm   measured span "
      f"{(X[:,0].max()-X[:,0].min())*1000:.2f} x {(X[:,1].max()-X[:,1].min())*1000:.2f} x "
      f"{(X[:,2].max()-X[:,2].min())*1000:.2f} mm")
print(f"    K {float(p.la[0]):.3e} Pa  mu {float(p.mu[0]):.1e}  OK\n")
