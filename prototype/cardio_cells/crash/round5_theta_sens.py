"""Does a 1.8e-15 relative change in theta move the crash test?  (round 5 vs round 4's naive)"""
import sys, os, json
from types import SimpleNamespace
import numpy as np, torch
for p in ("/workspace/Plexus/src","/workspace/Plexus/prototype/cardio_cells/algebraic",
          "/workspace/Plexus/discovery_cardio_mpm","/workspace/Plexus/prototype/cardio_cells/crash"):
    sys.path.insert(0,p)
from assemble import SUBSTEP_TOKENS
from recover import install_E
import metrics as MET, crash_test as CT
args=SimpleNamespace(device="cuda:1",cells=100,per_parent=100,n_grid=128,warmup=165,window=150,
                     dtype="float64",mode="full",e_lo=40.,e_hi=220.,g_lo=.5,g_hi=1.5)
torch.manual_seed(0)
with torch.no_grad():
    sy,_=CT.plant_and_warm(args,lambda s:None)
    C,W,G,n=sy.C,165,150,sy.n_sub_per_frame
    x0=sy.x0.clone()
    tracers={MET.MARGIN_SAFE:CT.tracer_indices(x0,CT.probe_points(MET.MARGIN_SAFE))}
    band=0.06/MET.SHEET_SPAN
    anchor=((x0[:,0]<band)|(x0[:,0]>1-band)|(x0[:,1]<band)|(x0[:,1]>1-band)); interior=~anchor
    ref=torch.zeros(G,sy.Np,2,device=sy.device,dtype=sy.dtype)
    sy.restore(); install_E(sy,sy.E_true)
    for k in range(G):
        sy._outer(W+k,gain_cell=sy.gain_true); sy.H.sub_dt=sy.dt_sub
        for _ in range(n):
            for tok in SUBSTEP_TOKENS: sy._tok(tok)
        sy.H.sub_dt=None; ref[k]=sy.p.get("pos")
    d=ref-x0[None]; dm=d[:,interior].mean(0,keepdim=True); ss=(d[:,interior]-dm).pow(2).sum()
    real=ref[:,tracers[MET.MARGIN_SAFE]].cpu().numpy()
    t5=torch.as_tensor(np.load("theta_round5.npz")["round5_norm_s90210_sF0.0039|T8|naive"],device=sy.device,dtype=torch.float64)
    t4=torch.as_tensor(np.load("theta_round4_stack_T8.npz")["naive"],device=sy.device,dtype=torch.float64)
    out={}
    for nm,t in (("round5",t5),("round4",t4),("round5_again",t5)):
        tr,_,co=CT.rollout(sy,t,W,G,tracers,ref_full=ref,anchor=None,interior=interior,ss_tot=ss)
        m=CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(),real)
        out[nm]={"loop":m["loopscore"],"R2":co["R2_displacement_interior"],
                 "rms":co["rms_pos_err_dx_mean"],"t1":co["motion_energy_ratio_interior"]}
        print(nm,out[nm],flush=True)
    out["theta_rel_diff"]=float((t5-t4).norm()/t5.norm())
    out["theta_max_abs_diff"]=float((t5-t4).abs().max())
    json.dump(out,open("/workspace/Plexus/prototype/cardio_cells/crash/round5_theta_sens.json","w"),indent=1)
    print("rel diff",out["theta_rel_diff"])
