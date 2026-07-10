"""pf_bootstrap -- run the phase-field forward model over the mechanism tree, score every spec with the
TIGHTENED readout + its DISTANCE to the real SMG morphology, and write a dataset for the surrogate/UCB.

Replaces the old sparse-agent bootstrap: the substrate is now pf_sim (dense connected tissue that clefts),
so specs land on real-like lobular morphology instead of fragments. Each row = (branch, params) ->
morphology VECTOR + failure class + target_distance to real. Determinism: fixed seed per spec.

  python pf/pf_bootstrap.py --n 48 [--stride 130 --nrec 6 --out pf/_boot]
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "search"))
import numpy as np
import pf_sim
import pf_tree
import smg_reward as R

# morphology dims compared to real (target_distance); generations/branch normalized to the real scale
_TD_DIMS = dict(duct_score=1.0, bud_score=1.0, cluster_score=1.0, elongation=0.5, connectedness=0.5)


def phi_to_points(phi, thr=0.5, max_pts=9000, rng=None):
    ys, xs = np.nonzero(phi > thr)
    if len(xs) < 20:
        return None
    P = np.c_[xs, ys].astype(float)
    if len(P) > max_pts:
        rng = rng or np.random.default_rng(0)
        P = P[rng.choice(len(P), max_pts, replace=False)]
    return (P - P.min(0)) / (np.ptp(P, 0) + 1e-9)


def score(phi):
    Pn = phi_to_points(phi)
    if Pn is None:
        return None
    o = R.obs_2d(Pn, W=1.0)
    return o, R.value_vector(o), R.classify(o)


def target_distance(v, real_v):
    """Weighted L1 between a spec's morphology vector and the real gland's -> 0 = matches real."""
    num = sum(w * abs(v.get(k, 0) - real_v.get(k, 0)) for k, w in _TD_DIMS.items())
    gen_d = abs(v.get("generations", 0) - real_v.get("generations", 0)) / max(real_v.get("generations", 1), 1)
    br_d = abs(v.get("branch_count", 0) - real_v.get("branch_count", 0)) / max(real_v.get("branch_count", 1), 1)
    return round(float((num + 0.5 * gen_d + 0.5 * br_d) / (sum(_TD_DIMS.values()) + 1.0)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--stride", type=int, default=130)
    ap.add_argument("--nrec", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "_boot"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0"

    phi0 = np.load(os.path.join(HERE, "_real", "phi0.npy"))
    real_o = R._real_obs(552); real_v = R.value_vector(real_o)
    print(f"real target vector: duct={real_v['duct_score']} bud={real_v['bud_score']} "
          f"gen={real_v['generations']} branch={real_v['branch_count']} class={R.classify(real_o)}\n", flush=True)

    branches = list(pf_tree.BRANCHES)
    rng = np.random.default_rng(args.seed)
    ds_path = os.path.join(args.out, "dataset.jsonl")
    rows, encs = [], []
    t0 = time.time()
    with open(ds_path, "w") as fh:
        for i in range(args.n):
            branch = branches[i % len(branches)]                 # stratified across hypotheses
            params = pf_tree.sample_params(branch, rng)
            seed = int(rng.integers(0, 100000))
            snaps = pf_sim.simulate(phi0, pf_tree.build_params(branch, params), n_record=args.nrec,
                                    stride=args.stride, device=dev, seed=seed)
            sc = score(snaps[-1])
            enc, names = pf_tree.encode(branch, params); encs.append(enc)
            if sc is None:
                row = dict(i=i, branch=branch, cleft_mode=pf_tree.BRANCHES[branch]["cleft_mode"],
                           params=params, failure="collapsed", value={}, target_distance=1.0, area=0.0)
            else:
                o, v, cls = sc
                td = target_distance(v, real_v)
                area = float((snaps[-1] > 0.5).mean())
                row = dict(i=i, branch=branch, cleft_mode=pf_tree.BRANCHES[branch]["cleft_mode"],
                           params=params, failure=cls, value=v, target_distance=td, area=round(area, 3))
            rows.append(row); fh.write(json.dumps(row) + "\n"); fh.flush()
            if (i + 1) % 4 == 0 or i == args.n - 1:
                v = row.get("value", {})
                print(f"[{i+1}/{args.n}] {branch:22} {row['failure']:11} duct={v.get('duct_score','-')} "
                      f"clust={v.get('cluster_score','-')} td={row['target_distance']} "
                      f"[{(time.time()-t0)/(i+1):.1f}s/spec]", flush=True)

    np.save(os.path.join(args.out, "encodings.npy"), np.array(encs, np.float32))
    json.dump({"feature_names": names, "n": args.n}, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    from collections import Counter
    hist = Counter(r["failure"] for r in rows)
    on = sum(1 for r in rows if r["failure"] == "branch-like")
    best = min((r for r in rows if r.get("value")), key=lambda r: r["target_distance"], default=None)
    print(f"\n=== pf bootstrap done: {args.n} specs in {(time.time()-t0)/60:.1f} min ===")
    print("failure manifold:", dict(hist))
    print(f"branch-like: {on}/{args.n}")
    if best:
        print(f"closest-to-real: {best['branch']} td={best['target_distance']} "
              f"duct={best['value']['duct_score']} gen={best['value']['generations']}")
    print("dataset:", ds_path)


if __name__ == "__main__":
    main()
