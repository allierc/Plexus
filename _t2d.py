import sys, yaml, tempfile, os, time
sys.path.insert(0,'src')
import torch; torch.cuda.set_device(0)
import plexus.operators
from plexus.schema import load
from plexus import engine as E
def go(ng, per, gu):
    s=yaml.safe_load(open('config/material/material_dam_break.yaml'))
    s['fields']['mpm_grid']['n_grid']=ng
    s['sets']['mpm_particle']['per_parent']=per
    for o in s['operators']:
        if o['op']=='mpm_grid_update':
            o.pop('implementation',None)
            if gu!='default': o['implementation']=gu
    s['general']['n_frames']=24; s['general']['record_cap']=2
    f=tempfile.NamedTemporaryFile('w',suffix='.yaml',delete=False); yaml.safe_dump(s,f); f.close()
    sim=load(f.name); os.unlink(f.name)
    blk=next(x for x in yaml.safe_load(open('config/material/material_dam_break.yaml'))['schedule'] if isinstance(x,dict))
    m={}
    def on_frame(H,t):
        if t==6: torch.cuda.synchronize(0); m['a']=time.perf_counter()
        elif t==24: torch.cuda.synchronize(0); m['b']=time.perf_counter()
    E.run(sim,out_path=None,device='cuda:0',on_frame=on_frame,progress=False)
    return (m['b']-m['a'])/18*1000
print("RESULT  n_grid  per_parent  p/cell   grid_update   ms/frame", flush=True)
import math
V=None
s0=yaml.safe_load(open('config/material/material_dam_break.yaml'))
pt=s0['sets']['mpm_particle']; par=s0['sets'].get(pt.get('parent'),{})
b=list((par.get('types') or {}).values())[0].get('block')
V=abs(b[2]-b[0])*abs(b[3]-b[1]) if b else 0.05
for ng, per in ((96,9000),(96,36000),(192,36000),(192,144000),(288,324000)):
    for gu in ('default','nosync'):
        try:
            t=go(ng,per,gu)
            print(f"RESULT  {ng:>6}  {per:>10,}  {per/(V*ng**2):6.1f}   {gu:<12}{t:8.1f}", flush=True)
        except Exception as e:
            print(f"RESULT  {ng:>6}  {per:>10,}  {gu:<12} ERROR {type(e).__name__}", flush=True)
