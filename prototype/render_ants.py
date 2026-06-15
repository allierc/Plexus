"""Render the 10 ant-variant gifs (periodic), resumable via markers.
    python render_ants.py   # re-run until 'ALL DONE'
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scenario_schema import load
import engine2
V = {1:('baseline',{},{}), 2:('rich_network',{'motility':{'rot':0.20},'trail':{'turn':0.3}},{'decay':0.10}),
     3:('loose_wander',{'trail':{'turn':0.15}},{}), 4:('tight_trails',{'motility':{'rot':0.03},'trail':{'turn':0.7}},{}),
     5:('blurry',{},{'diffusion':0.10}), 6:('sharp_permanent',{},{'diffusion':0.01,'decay':0.015}),
     7:('far_sensor',{'trail':{'sensor_dist':0.09}},{}), 8:('wiggly',{'trail':{'sensor_dist':0.02,'sensor_angle':0.9}},{}),
     9:('strong_pheromone',{'secrete':{'rate':4.0}},{}), 10:('ephemeral',{'motility':{'rot':0.20},'secrete':{'rate':0.8}},{'decay':0.12})}
for i,(lab,opov,fov) in V.items():
    mark="/tmp/pant_%d.done"%i
    if os.path.exists(mark): print("skip ant_%d (done)"%i,flush=True); continue
    sc=load(os.path.join(HERE,"scenarios","ant.yaml")); sc.n_frames=900; sc.record_every=1
    for o in sc.operators:
        if o.op in opov: o.params.update(opov[o.op])
    sc.fields["pheromone"].update(fov)
    _,a=engine2.run(sc,None,device="cuda")
    pp,par=a["particle_pos"],a["parent"]; pt=a["cell_type"][par]; f=a["field"]; T=f.shape[0]; soft,stiff=pt==0,pt==1
    fig,ax=plt.subplots(figsize=(5,5)); fig.patch.set_facecolor("black")
    im=ax.imshow(f[0].T,origin="lower",extent=[0,1,0,1],cmap="inferno",vmin=0,vmax=max(f.max()*0.35,1e-6))
    sb=ax.scatter(pp[0][stiff,0],pp[0][stiff,1],s=0.4,c="#3a6ea5",alpha=0.5)
    sr=ax.scatter(pp[0][soft,0],pp[0][soft,1],s=0.5,c="cyan")
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xticks([]);ax.set_yticks([]); tt=ax.set_title("",color="white",fontsize=7)
    def upd(fr,im=im,sr=sr,sb=sb,tt=tt,pp=pp,f=f,soft=soft,stiff=stiff,lab=lab,T=T,i=i):
        im.set_data(f[fr].T); sr.set_offsets(pp[fr][soft]); sb.set_offsets(pp[fr][stiff])
        tt.set_text("ant_%d %s (periodic)  %d/%d"%(i,lab,fr,T-1)); return [im,sr,sb,tt]
    FuncAnimation(fig,upd,frames=T,blit=False).save(os.path.join(HERE,"ant_%d.gif"%i),writer=PillowWriter(fps=20)); plt.close(fig)
    open(mark,"w").close(); print("ant_%d.gif = %s"%(i,lab),flush=True)
print("ALL DONE",flush=True)
