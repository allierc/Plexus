"""Peak memory and ms/frame for a spec, over a few iterations. No output written."""
import sys, os, time, tempfile, yaml, torch
ROOT=sys.argv[1]; SPEC=sys.argv[2]; DEV=sys.argv[3]; NF=int(sys.argv[4])
sys.path.insert(0, os.path.join(ROOT,"src"))
import plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
s=yaml.safe_load(open(os.path.join(ROOT,"config",*SPEC.split("/"))+".yaml"))
s["general"]["n_frames"]=NF
if "record_cap" in s["general"]: s["general"]["record_cap"]=2
s["general"]["save_data"]=False
f=tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False); yaml.safe_dump(s,f); f.close()
CFL(f.name); sim=load(f.name); os.unlink(f.name)
torch.cuda.set_device(int(DEV.split(":")[-1])); torch.cuda.init()
torch.cuda.reset_peak_memory_stats()
ts=[]
def cb(H,t): ts.append(time.perf_counter())
E.run(sim,out_path=None,device=DEV,progress=False,on_frame=cb)
pk=torch.cuda.max_memory_allocated()/2**30
d=[(ts[i+1]-ts[i])*1000 for i in range(len(ts)-1)]
warm=d[len(d)//2:] or d
ng=int(list(s["fields"].values())[0]["n_grid"])
N=s["sets"]["cell"]["n"]*s["sets"]["mpm_particle"]["per_parent"]
print(f"  {SPEC.split('/')[-1]:<30}{N/1e6:>6.0f}M  n_grid {ng:>4} ({ng**3/1e6:>5.1f}M cells)  "
      f"peak {pk:>6.2f} GiB   {sum(warm)/len(warm):>8.0f} ms/frame")
