import sys, os, tempfile, yaml, torch
sys.path.insert(0,'src')
import plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
N=int(sys.argv[1]); dev=sys.argv[2]
s=yaml.safe_load(open("config/si_material/si_bench_100m.yaml"))
s["general"]["n_frames"]=3
s["sets"]["mpm_particle"]["per_parent"]=N//27
f=tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False); yaml.safe_dump(s,f); f.close()
CFL(f.name); sim=load(f.name); os.unlink(f.name)
torch.cuda.set_device(int(dev.split(":")[-1]))
torch.cuda.init()
torch.cuda.reset_peak_memory_stats()
E.run(sim,out_path=None,device=dev,progress=False)
pk=torch.cuda.max_memory_allocated()/2**30
n=27*(N//27)
print(f"  {n/1e6:>6.1f}M particles -> peak {pk:.2f} GiB  ({pk/(n/1e6)*1000:.1f} GiB per 1000M)")
