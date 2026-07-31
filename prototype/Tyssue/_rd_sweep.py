"""Pure Gray-Scott on a fixed sphere: what makes the pattern SURVIVE and form big domains?

No mechanics, no growth, no division -- only cell_adjacency + cell_diffuse + cell_react on a rigid
mesh. That isolates the chemistry completely, which nothing in this campaign has ever done, and it
is fast enough to sweep properly.
"""
import sys, itertools, json, numpy as np, torch
sys.path.insert(0,'/workspace/Plexus/prototype/Tyssue'); sys.path.insert(0,'/workspace/Plexus/src')
from tyssue_ops3d import build_sphere_mesh
from tyssue_topology_ops3d import rings_from_flat_3d

def adjacency(es, et, ef, nF):
    """cell-cell pairs: two faces sharing a mesh edge"""
    key = np.minimum(es,et).astype(np.int64)*(max(es.max(),et.max())+1)+np.maximum(es,et)
    o=np.argsort(key,kind='stable'); k,f=key[o],ef[o]; pairs=[]
    i=0
    while i < len(k):
        j=i
        while j+1 < len(k) and k[j+1]==k[i]: j+=1
        if j>i:
            for x in range(i,j+1):
                for y in range(x+1,j+1):
                    if f[x]!=f[y]: pairs.append((f[x],f[y])); pairs.append((f[y],f[x]))
        i=j+1
    p=np.array(pairs,dtype=np.int64)
    return p[:,0], p[:,1]

def run(nF_target=2000, seed_frac=0.06, chi=1.3, d_a=0.08, d_h=0.16, F=0.055, kk=0.062,
        rate=1.0, steps=4000, seed=0, patch=0):
    v,es,et,ef,nF = build_sphere_mesh(nF_target, 5.0, 0.0, 0)
    src,dst = adjacency(es,et,ef,nF)
    deg = np.bincount(src, minlength=nF).astype(np.float64); deg[deg==0]=1
    rng = np.random.default_rng(seed)
    a = 0.04*rng.random(nF); u = np.ones(nF)
    if patch > 0:
        # contiguous patches instead of scattered single cells
        rings = rings_from_flat_3d(es,et,ef,nF)
        cen = np.array([v[r].mean(0) if r is not None and len(r) else [9e9]*3 for r in rings])
        n_patch = max(1, int(seed_frac*nF/patch))
        centres = rng.choice(nF, n_patch, replace=False)
        for c in centres:
            d = np.linalg.norm(cen-cen[c],axis=1)
            for i in np.argsort(d)[:patch]: a[i]=0.5; u[i]=0.25
    else:
        nucl = rng.random(nF) < seed_frac
        a[nucl]=0.5; u[nucl]=0.25
    Da, Du = d_a*chi, d_h*chi
    hist=[]
    for s in range(steps):
        la = (np.bincount(src, weights=a[dst], minlength=nF)/deg) - a
        lu = (np.bincount(src, weights=u[dst], minlength=nF)/deg) - u
        uaa = u*a*a
        a = a + Da*la + rate*(uaa - (F+kk)*a)
        u = u + Du*lu + rate*(-uaa + F*(1.0-u))
        a = np.clip(a,0,10); u = np.clip(u,0,10)
        if s % (steps//8) == 0 or s == steps-1:
            hist.append((s, float(a.max()), float((a>0.2).mean())))
    # how many CONTIGUOUS domains, and how big
    hi = a > 0.2
    ndom, sizes = 0, []
    seen = np.zeros(nF,bool)
    nbr = [[] for _ in range(nF)]
    for x,y in zip(src,dst): nbr[x].append(y)
    for i in range(nF):
        if hi[i] and not seen[i]:
            ndom+=1; stack=[i]; seen[i]=True; n=0
            while stack:
                c=stack.pop(); n+=1
                for j in nbr[c]:
                    if hi[j] and not seen[j]: seen[j]=True; stack.append(j)
            sizes.append(n)
    return dict(a_max=float(a.max()), frac_hi=float(hi.mean()), n_domains=ndom,
                domain_med=int(np.median(sizes)) if sizes else 0,
                domain_max=int(max(sizes)) if sizes else 0, hist=hist)

if __name__ == "__main__":
    print(f"{'chi':>6}{'seedfrac':>9}{'patch':>6}{'steps':>7} | {'a_max':>7}{'frac_hi':>8}{'ndom':>6}{'med':>5}{'max':>6}  verdict")
    rows=[]
    for chi, sf, patch in itertools.product((1.3, 4.0, 13.0, 40.0), (0.06,), (0, 7)):
        r = run(chi=chi, seed_frac=sf, patch=patch, steps=4000)
        alive = r["a_max"] > 0.2
        v = ("DEAD -- activator extinguished" if not alive else
             f"alive: {r['n_domains']} domains, median {r['domain_med']} cells")
        print(f"{chi:6.1f}{sf:9.2f}{patch:6d}{4000:7d} | {r['a_max']:7.3f}{r['frac_hi']:8.3f}"
              f"{r['n_domains']:6d}{r['domain_med']:5d}{r['domain_max']:6d}  {v}")
        rows.append(dict(chi=chi, seed_frac=sf, patch=patch, **{k:v2 for k,v2 in r.items() if k!='hist'}))
    json.dump(rows, open("_rd_sweep.json","w"), indent=1)
