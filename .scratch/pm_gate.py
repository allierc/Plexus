"""GATE for particle_mass: N + m_p declared, VOLUME derived, and a conflict is loud."""
import sys, os, tempfile, yaml, warnings, numpy as np
sys.path.insert(0,'src')
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
def run(patch, dev, frames=3):
    s=yaml.safe_load(open("config/material/si_water_quarter.yaml"))
    s["general"]["n_frames"]=frames; s["general"]["record_cap"]=10
    s["sets"]["mpm_particle"]["per_parent"]=200000
    patch(s)
    f=tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False); yaml.safe_dump(s,f); f.close()
    sim=load(f.name); os.unlink(f.name)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        H,_=E.run(sim,out_path=None,device=dev,progress=False)
        msgs=[str(x.message) for x in w if "VOLUME" in str(x.message)]
    p=H.level("mpm_particle"); X=p.get("pos").detach().cpu().numpy()
    return dict(pv=float(p.p_vol[0]), mp=float(p.mass[0]), N=p.n,
                V=float(p.p_vol[0])*p.n, span=[float(X[:,k].max()-X[:,k].min()) for k in range(3)],
                warn=msgs)
dev=sys.argv[1]; rho=1000.0
print(f"\n  {'case':<34}{'p_vol':>12}{'m_p (kg)':>12}{'V = N*p_vol':>14}{'x span':>10}  warn")
print("  "+"-"*94)
# 1. baseline: block declared, no particle_mass  -> p_vol = block/N (today's behaviour)
a=run(lambda s: None, dev)
print(f"  {'block only (unchanged)':<34}{a['pv']:>12.4e}{a['mp']:>12.4e}{a['V']:>14.6g}"
      f"{a['span'][0]:>10.4f}  {len(a['warn'])}")
# 2. particle_mass matching the block exactly -> no warning, same p_vol
mp_match = a['pv']*rho
def p2(s):
    s["sets"]["mpm_particle"]["particle_mass"]=mp_match
b=run(p2, dev)
print(f"  {'block + matching particle_mass':<34}{b['pv']:>12.4e}{b['mp']:>12.4e}{b['V']:>14.6g}"
      f"{b['span'][0]:>10.4f}  {len(b['warn'])}")
# 3. particle_mass 4x too big, block kept -> MUST warn
def p3(s):
    s["sets"]["mpm_particle"]["particle_mass"]=mp_match*4
c=run(p3, dev)
print(f"  {'block + 4x particle_mass':<34}{c['pv']:>12.4e}{c['mp']:>12.4e}{c['V']:>14.6g}"
      f"{c['span'][0]:>10.4f}  {len(c['warn'])}")
# 4. NO block, particle_mass only -> volume and box DERIVED
def p4(s):
    s["sets"]["mpm_particle"]["particle_mass"]=mp_match
    list(s["sets"]["cell"]["types"].values())[0].pop("block")
d=run(p4, dev)
print(f"  {'particle_mass only, block DERIVED':<34}{d['pv']:>12.4e}{d['mp']:>12.4e}{d['V']:>14.6g}"
      f"{d['span'][0]:>10.4f}  {len(d['warn'])}")
print(f"\n  1 vs 2: p_vol identical? {abs(b['pv']/a['pv']-1)<1e-6}   "
      f"3 warns? {len(c['warn'])==1}   3 p_vol is 4x? {abs(c['pv']/a['pv']-4)<1e-6}")
print(f"  4: derived cube side {d['V']**(1/3):.6g} m, measured span {d['span'][0]:.6g} m "
      f"-> {'MATCHES' if abs(d['span'][0]/d['V']**(1/3)-1)<0.02 else 'MISMATCH'}")
if c['warn']: print(f"\n  the warning:\n    {c['warn'][0][:300]}")
print()
