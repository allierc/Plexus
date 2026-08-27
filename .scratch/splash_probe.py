import sys, os, tempfile, yaml, numpy as np, math
sys.path.insert(0,'src')
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
spec, dev, frames, npar = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
wd = float(sys.argv[5]) if len(sys.argv) > 5 else None
s=yaml.safe_load(open(f"config/material/{spec}.yaml"))
s["general"]["n_frames"]=frames; s["general"]["record_cap"]=10000
s["sets"]["mpm_particle"]["per_parent"]=npar
if wd is not None:
    for o in s["operators"]:
        if "wall_damp" in o: o["wall_damp"]=wd
f=tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False); yaml.safe_dump(s,f); f.close()
CFL(f.name); sim=load(f.name); os.unlink(f.name)
W=float(s["general"]["world"][1]); g=next(float(o["g"]) for o in s["operators"] if o["op"]=="gravity")
rho=float(s["sets"]["mpm_particle"]["density"])
prev={}
rows=[]
def cb(H,t):
    L=H.level("mpm_particle"); X=L.get("pos"); V=L.get("vel")
    sp=V.norm(dim=1)
    m=float(L.mass[0])
    ke=float(0.5*m*(sp**2).sum()); pe=float(m*g*X[:,1].sum())
    rows.append((t,float(sp.max()),float(sp.mean()),float(X[:,1].max()),ke,pe))
E.run(sim,out_path=None,device=dev,progress=False,on_frame=cb)
print(f"\n  {spec}  {npar:,} particles, wall_damp {'as written' if wd is None else wd}")
print(f"  free-fall impact speed would be {math.sqrt(2*g*0.226):.2f} m/s; "
      f"escape-to-ceiling needs {math.sqrt(2*g*W):.2f} m/s\n")
print(f"  {'frame':>7}{'v max':>9}{'v mean':>9}{'y max':>9}{'KE (J)':>11}{'PE (J)':>11}{'KE+PE':>11}")
for r in rows[::max(1,len(rows)//12)]:
    print(f"  {r[0]:>7}{r[1]:>9.3f}{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>11.5f}{r[5]:>11.5f}{r[4]+r[5]:>11.5f}")
print()
