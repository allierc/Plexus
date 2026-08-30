import sys, os, tempfile, yaml, torch, time
sys.path.insert(0,'src')
import plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
N=int(sys.argv[1]); dev=sys.argv[2]; cap=(sys.argv[3]=="on")
s=yaml.safe_load(open("config/si_material/si_bench_100m.yaml"))
s["general"]["n_frames"]=6; s["sets"]["mpm_particle"]["per_parent"]=N//27
for blk in s["schedule"]:
    if isinstance(blk,dict) and "substep_dt" in blk: blk["capture"]=cap
f=tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False); yaml.safe_dump(s,f); f.close()
CFL(f.name); sim=load(f.name); os.unlink(f.name)
torch.cuda.set_device(int(dev.split(":")[-1])); torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
ts=[]
E.run(sim,out_path=None,device=dev,progress=False,on_frame=lambda H,t: ts.append(time.perf_counter()))
d=[(ts[i+1]-ts[i])*1000 for i in range(len(ts)-1)]; w=d[len(d)//2:] or d
print(f"  {27*(N//27)/1e6:>5.0f}M  capture {'ON ' if cap else 'OFF'} -> peak "
      f"{torch.cuda.max_memory_allocated()/2**30:>6.2f} GiB   {sum(w)/len(w):>7.0f} ms/frame")
