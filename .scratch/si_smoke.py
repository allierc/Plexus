import sys, os, tempfile, yaml, numpy as np, torch as T
sys.path.insert(0,'src')
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
s=yaml.safe_load(open("config/material/si_water_column.yaml"))
s["general"]["n_frames"]=150; s["sets"]["mpm_particle"]["per_parent"]=300000
f=tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False); yaml.safe_dump(s,f); f.close()
CFL(f.name); sim=load(f.name); os.unlink(f.name)
H,_=E.run(sim,out_path=None,device=sys.argv[1],progress=False)
g=H.field("mpm_grid"); p=H.level("mpm_particle"); X=p.get("pos").detach().cpu().numpy()
print(f"\n  grid   dx {g.dx*1000:.3f} mm   inv_dx {g.inv_dx:.1f}   shape {g.shape}   cells {g.n_cells:,}")
print(f"  mass   {float(p.mass[0]):.4e} kg   p_vol {float(p.p_vol[0]):.4e} m3   -> rho {float(p.mass[0]/p.p_vol[0]):.1f} kg/m3")
print(f"  moduli la(=K) {float(p.la[0]):.4e} Pa   mu {float(p.mu[0]):.4e} Pa")
print(f"  mean(J) {float(T.linalg.det(p.F).mean()):.6f}")
print(f"  y {X[:,1].min():.4f}..{X[:,1].max():.4f} m   x {X[:,0].min():.4f}..{X[:,0].max():.4f} m   (box 0.1)")
print(f"  SMOKE TEST OK\n")
