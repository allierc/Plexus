import sys, os, tempfile, yaml, numpy as np
sys.path.insert(0,'src')
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
from plexus.generators.graph_data_generator import *  # noqa
s=yaml.safe_load(open("config/si_material/si_gate.yaml"))
s["general"]["n_frames"]=int(sys.argv[2]); s["sets"]["mpm_particle"]["per_parent"]=60000
f=tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False); yaml.safe_dump(s,f); f.close()
CFL(f.name); sim=load(f.name); os.unlink(f.name)
print(f"  sim.save_data = {sim.save_data!r}")
from plexus.live_movie import LiveMovie
lm=LiveMovie(out="/tmp/fps_test.mp4", world=list(sim.world_size), n_frames=sim.n_frames,
             up=1, name=sim.name, sim=sim, style=(sim.plotting or {}), render_n=60000,
             max_frames=300, stills=0, dt=sim.dt,
             time_s=(sim.units.time_s if sim.units.declared else None))
print(f"  stride {lm.stride}  fps {getattr(lm,'fps',None)}  speed {lm.speed}  "
      f"duration {getattr(lm,'duration_s',None)} s")
lm.close()
