import os, sys, json, numpy as np
# ROOT-RELATIVE PATHS, because this file runs on the cluster where the repo is mounted at
# /groups/saalfeld/home/allierc/Graph/Plexus, not /workspace. Same NFS export, different mount point.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "discovery_okuda")):
    sys.path.insert(0, p)
import combine as C, run_ecm as R, tissue as TIS, membrane_ops, aniso
rate, dev = float(sys.argv[1]), sys.argv[2]
# TEST 1: THE SECRETION / GROWTH RACE. Area demand grows as R^2 while deposition is a fixed FRACTION of
# existing material per frame, so there is a threshold rate below which the sheet cannot keep up. It is
# predictable in advance: 3,333 -> 42,575 particles over ~400 frames needs ln(12.77)/400 = 0.0064 per
# frame. The sweep brackets that, so the prediction is falsifiable rather than fitted afterwards.
gate = os.path.join(_ROOT, "log", "okuda_ECM", "49_aniso_i0_fibres", "load.npz")
npz = TIS.load_or_build(frames=401, device=dev, buffer_x=4, myosin=1.0, gate_npz=gate,
                        gate_p_half="auto", gate_hill=6.0, gate_floor=0.08, gate_smooth_frames=25,
                        gate_smooth_phi=360.0, tag_extra="_gated_myo")
name = f"_race_r{rate:g}"
for t in (membrane_ops.BOND_TRACE, membrane_ops.MEMBRANE_STRAIN, membrane_ops.SECRETE_TRACE): t.clear()
cfg = dict(aniso.BASE)
cfg.update(membrane=npz, membrane_particles=45000, membrane_cutoff=0.008, membrane_break=0.35,
           membrane_bond_k=2.0e5, membrane_adhesion=4.0e4, membrane_tau=60.0, membrane_jitter=0.35,
           membrane_reserve=12.5, membrane_secrete_rate=rate)
spec, info = C.build(name, npz, **cfg)
R.run(name, spec, device=dev, movie=False, render_kw={"strip_only": True})

bt = np.asarray(membrane_ops.BOND_TRACE, float)
z = np.load(os.path.join(R.LOG, name, "traj.npz"))
mp = np.asarray(z["mpos"]); al = np.asarray(z["malive"]) if "malive" in z.files else np.ones(mp.shape[1], bool)
P = mp[-1][al]; u = P - P.mean(0); u /= np.linalg.norm(u, axis=1)[:, None]
# COVERAGE: fraction of angular bins that hold any membrane at all. A sheet can keep every bond it has
# and still leave half the surface bare, which is the failure the polar-cap bug made visible.
th = np.arccos(np.clip(u[:,2],-1,1)); ph = np.arctan2(u[:,1],u[:,0])
bi = (np.clip((th/np.pi*16).astype(int),0,15)*32 + np.clip(((ph+np.pi)/(2*np.pi)*32).astype(int),0,31))
cov = len(np.unique(bi)) / (16*32)
out = dict(rate=rate, n_alive_end=int(al.sum()),
           bonds_start=int(bt[0,0]), bonds_end=int(bt[-1,0]),
           lcc_end=float(bt[-1,3]) if bt.shape[1] > 3 else None,
           strain_end=float(np.mean(np.asarray(membrane_ops.MEMBRANE_STRAIN[-1])[al])),
           coverage=cov, secretion_events=len(membrane_ops.SECRETE_TRACE))
print("RACE " + json.dumps(out), flush=True)
