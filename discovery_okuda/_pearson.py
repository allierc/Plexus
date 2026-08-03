"""Where in Pearson's (F, k) diagram do we get Okuda's ~5 STABLE spots on a 2000-cell ball?

The minisite values F=0.055 k=0.062 do not give spots at all: they coarsen monotonically to ONE
bicontinuous domain at 53% coverage and sit there. That is Gray-Scott's labyrinth regime, and it is
what the numbers say we have been running all along. Okuda's figure shows persistent discrete
spots, so the parameters must be somewhere else -- and where is a measurement, not a guess.

STABILITY is the point. n_spots is read at two well-separated times; a pattern that is still
coarsening is not a spot pattern, whatever it looks like at the moment you happen to stop.
"""
import os
import sys, itertools, json, numpy as np, torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,'/workspace/Plexus/prototype/Tyssue'); sys.path.insert(0,'/workspace/Plexus/src')
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
from pattern_scale import cell_graph, pattern_metrics

v,es,et,ef,nF = build_sphere_mesh(2000,5.0,0.0,0)
_,_,cen,_ = face_geometry_3d(torch.as_tensor(v),torch.as_tensor(es),torch.as_tensor(et),torch.as_tensor(ef),nF)
cen=cen.numpy(); src,dst=cell_graph(es,et,ef,nF)
deg=np.bincount(src,minlength=nF).astype(float); deg[deg==0]=1

def run(F, kk, chi=1.3, rate=1.0, d_a=0.08, d_h=0.16, seed_frac=0.06, seed=0, checks=(1500,3000)):
    rng=np.random.default_rng(seed); a=0.04*rng.random(nF); u=np.ones(nF)
    n=rng.random(nF)<seed_frac; a[n]=0.5; u[n]=0.25
    Da,Du=d_a*chi,d_h*chi; out=[]
    for s in range(max(checks)+1):
        if s in checks:
            out.append(pattern_metrics(a,es,et,ef,nF,cen=cen) if a.max()>0.15 else None)
        la=(np.bincount(src,weights=a[dst],minlength=nF)/deg)-a
        lu=(np.bincount(src,weights=u[dst],minlength=nF)/deg)-u
        uaa=u*a*a
        a=a+Da*la+rate*(uaa-(F+kk)*a); u=u+Du*lu+rate*(-uaa+F*(1.0-u))
        a=np.clip(a,0,10); u=np.clip(u,0,10)
    return out, float(a.max())



# A GUARD, BECAUSE THIS FILE DOES WORK. Everything below ran AT IMPORT: a package scan,
# an editor indexing the tree, or `import _pearson` from a sibling would silently start a
# GPU sweep. Found during the pre-flight for the next campaign, when importing every module
# in the package printed two full calibration tables. A script and a module are different
# things and the difference is this line.
if __name__ == "__main__":
    print("PEARSON SWEEP -- looking for STABLE spots, not a moment during coarsening")
    print(f"target: ~5 spots on {nF} cells, unchanged between step 1500 and 3000\n")
    print(f"{'F':>7}{'k':>7} | {'spots@1500':>11}{'spots@3000':>11}{'cover':>8}{'spacing':>9}  verdict")
    rows=[]
    for F, kk in itertools.product((0.020,0.030,0.037,0.046,0.055), (0.055,0.060,0.062,0.065)):
        (m1,m2), amax = run(F,kk)
        if m1 is None or m2 is None:
            print(f"{F:7.3f}{kk:7.3f} | {'-':>11}{'-':>11}{'-':>8}{'-':>9}  DEAD"); continue
        n1,n2 = m1['n_spots'], m2['n_spots']
        stable = (n2>0) and abs(n1-n2) <= max(1, 0.2*n1)
        v = ("STABLE" if stable else "coarsening")
        if stable and 3 <= n2 <= 9: v = "STABLE  <-- OKUDA RANGE"
        print(f"{F:7.3f}{kk:7.3f} | {n1:11d}{n2:11d}{m2['spot_frac']:8.3f}{(m2['spot_spacing_cells'] or 0):9.2f}  {v}")
        rows.append(dict(F=F,kk=kk,n1500=n1,n3000=n2,cover=m2['spot_frac'],
                         spacing=m2['spot_spacing_cells'],stable=bool(stable)))
    json.dump(rows, open("_pearson.json","w"), indent=1)
    print("\n-> _pearson.json")
