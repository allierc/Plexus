import json, os, sys, numpy as np
sys.path.insert(0, "/workspace/Plexus/discovery_okuda/ops")
LOG = "/workspace/Plexus/log/okuda_ECM"

def loops(F):
    """Boundary edges (in exactly one live face), grouped into loops -> how many holes."""
    e = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            e[(min(u, v), max(u, v))] = e.get((min(u, v), max(u, v)), 0) + 1
    bnd = [k for k, n in e.items() if n == 1]
    if not bnd:
        return 0, 0
    adj = {}
    for u, v in bnd:
        adj.setdefault(u, []).append(v); adj.setdefault(v, []).append(u)
    seen, nl = set(), 0
    for s in adj:
        if s in seen: continue
        nl += 1; st = [s]
        while st:
            x = st.pop()
            if x in seen: continue
            seen.add(x); st.extend(adj[x])
    return nl, len(bnd)

def go(run):
    z = np.load(os.path.join(LOG, run, "bm_frames.npz"))
    n = int(z["n_kept"]); c = np.asarray(z["centre"], float); sc = float(z["scale"])
    S = {k: [] for k in ("t", "torn_frac", "n_loops", "n_bnd", "area", "R")}
    for i in range(n):
        x = (np.asarray(z[f"x{i}"], float) - c) / sc
        F = np.asarray(z[f"f{i}"], np.int64)
        v = x[F]
        A = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum()
        used = np.unique(F)
        R = np.linalg.norm(x[used], axis=1).mean()
        nl, nb = loops(F)
        S["t"].append(int(z[f"t{i}"])); S["area"].append(float(A)); S["R"].append(float(R))
        S["torn_frac"].append(float(max(0.0, 1.0 - A / (4 * np.pi * R * R))))
        S["n_loops"].append(nl); S["n_bnd"].append(nb)
    a = {k: np.asarray(v, float) for k, v in S.items()}
    last = -1
    third = len(a["t"]) // 3
    creep = float(a["torn_frac"][-1] - a["torn_frac"][-third])
    out = dict(run=run, frames=n,
               torn_frac_final=float(a["torn_frac"][last]),
               torn_frac_max=float(a["torn_frac"].max()),
               rim_loops_final=int(a["n_loops"][last]),
               boundary_edges_final=int(a["n_bnd"][last]),
               first_torn_frame=int(a["t"][np.argmax(a["n_loops"] > 0)]) if (a["n_loops"] > 0).any() else None,
               creep_over_last_third=creep,
               arrested=bool(abs(creep) < 0.002),
               series={k: v.tolist() for k, v in a.items()})
    json.dump(out, open(os.path.join(LOG, run, "hole_metrics.json"), "w"), indent=1)
    print(f"[hole] {run}: torn {100*out['torn_frac_final']:.2f}% of the sphere it would be, "
          f"{out['rim_loops_final']} rim loop(s), {out['boundary_edges_final']} boundary edges, "
          f"first tear at frame {out['first_torn_frame']}, creep over the last third "
          f"{100*creep:+.3f}% -> {'ARRESTED' if out['arrested'] else 'STILL GROWING'}", flush=True)

for r in sys.argv[1:]:
    go(r)
