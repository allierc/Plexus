"""pf_ucb -- the search controller for the phase-field SMG loop: UCB over the mechanism-tree HYPOTHESIS
BRANCHES + a learned surrogate that predicts the MORPHOLOGY VECTOR, driving the forward model toward the
real gland (minimize target_distance). This is the AlphaZero-lite + Bayesian-optimization design, now on a
substrate that actually makes dense connected clefting glands.

Loop (all local; pf_sim ~0.8 s/spec):
  1. SEED: run `seed_n` random specs stratified across branches -> archive + train surrogate.
  2. UCB round: pick the branch maximizing  (best -td in branch) + c*sqrt(ln N / n_branch);
     propose `cands` param sets in that branch, SURROGATE pre-screens them (predict vector -> td),
     RUN the best-predicted one for real, score, update surrogate + branch stats + archive.
  3. Repeat for `rounds`; track the global best (lowest target_distance to real).

  python pf/pf_ucb.py --seed_n 40 --rounds 300 --cands 24 [--out pf/_ucb]
"""
import os, sys, json, time, math, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "search"))
import numpy as np
import pf_sim, pf_tree
import smg_reward as R
from pf_bootstrap import score, target_distance
from sklearn.ensemble import RandomForestRegressor

VEC_DIMS = ["duct_score", "bud_score", "cluster_score", "elongation", "connectedness",
            "generations", "branch_count"]
BAD_TD = 1.0                                    # collapsed / no-growth specs -> worst distance


def vec_of(v):
    return [float(v.get(k, 0)) for k in VEC_DIMS]


class Surrogate:
    """RandomForest predicting the morphology VECTOR from a spec encoding (reviewer's ask: predict
    morphology, not a scalar). td is computed from the predicted vector -> cheap pre-screen."""
    def __init__(self):
        self.rf = None

    def fit(self, X, Y):
        if len(X) >= 12:
            self.rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1,
                                            random_state=0).fit(np.array(X), np.array(Y))

    def pred_td(self, enc, real_v):
        if self.rf is None:
            return None
        vec = self.rf.predict(enc[None])[0]
        return target_distance(dict(zip(VEC_DIMS, vec)), real_v)


def run_spec(phi0, branch, params, stride, nrec, seed, real_v):
    snaps = pf_sim.simulate(phi0, pf_tree.build_params(branch, params), n_record=nrec,
                            stride=stride, device="cuda:0", seed=seed)
    sc = score(snaps[-1])
    enc, _ = pf_tree.encode(branch, params)
    if sc is None:
        return dict(branch=branch, cleft_mode=pf_tree.BRANCHES[branch]["cleft_mode"], params=params,
                    failure="collapsed", value={}, target_distance=BAD_TD), enc, [0.0] * len(VEC_DIMS)
    o, v, cls = sc
    td = target_distance(v, real_v) if cls != "no-growth" else min(1.0, target_distance(v, real_v) + 0.15)
    return dict(branch=branch, cleft_mode=pf_tree.BRANCHES[branch]["cleft_mode"], params=params,
                failure=cls, value=v, target_distance=td,
                area=round(float((snaps[-1] > 0.5).mean()), 3)), enc, vec_of(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_n", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--cands", type=int, default=24)
    ap.add_argument("--c", type=float, default=0.12)      # UCB exploration weight (td scale ~0..0.5)
    ap.add_argument("--stride", type=int, default=130)
    ap.add_argument("--nrec", type=int, default=6)
    ap.add_argument("--retrain", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "_ucb"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    phi0 = np.load(os.path.join(HERE, "_real", "phi0.npy"))
    real_v = R.value_vector(R._real_obs(552))
    print(f"real target: duct={real_v['duct_score']} bud={real_v['bud_score']} gen={real_v['generations']} "
          f"branch={real_v['branch_count']}\n", flush=True)

    branches = list(pf_tree.BRANCHES)
    rng = np.random.default_rng(0)
    surr = Surrogate()
    X, Y, archive = [], [], []
    stats = {b: dict(n=0, best=BAD_TD) for b in branches}
    ds = open(os.path.join(args.out, "dataset.jsonl"), "w")
    t0 = time.time()

    def record(row, enc, y):
        X.append(enc); Y.append(y); archive.append(row); ds.write(json.dumps(row) + "\n"); ds.flush()
        b = row["branch"]; stats[b]["n"] += 1; stats[b]["best"] = min(stats[b]["best"], row["target_distance"])

    # 1. SEED
    print("=== seeding ===", flush=True)
    for i in range(args.seed_n):
        b = branches[i % len(branches)]
        row, enc, y = run_spec(phi0, b, pf_tree.sample_params(b, rng), args.stride, args.nrec,
                               int(rng.integers(1e5)), real_v)
        record(row, enc, y)
    surr.fit(X, Y)
    best = min(archive, key=lambda r: r["target_distance"])
    print(f"seed done: {args.seed_n} specs, best td={best['target_distance']} ({best['branch']})\n"
          f"=== UCB ===", flush=True)

    # 2. UCB rounds
    for t in range(args.rounds):
        N = sum(s["n"] for s in stats.values())
        ucb = {b: (1.0 - stats[b]["best"]) + args.c * math.sqrt(math.log(N + 1) / (stats[b]["n"] + 1))
               for b in branches}
        branch = max(ucb, key=ucb.get)
        cands = [pf_tree.sample_params(branch, rng) for _ in range(args.cands)]
        if surr.rf is not None:                          # surrogate pre-screen: pick best predicted
            preds = [surr.pred_td(pf_tree.encode(branch, c)[0], real_v) for c in cands]
            params = cands[int(np.argmin(preds))]
        else:
            params = cands[0]
        row, enc, y = run_spec(phi0, branch, params, args.stride, args.nrec, int(rng.integers(1e5)), real_v)
        record(row, enc, y)
        if (t + 1) % args.retrain == 0:
            surr.fit(X, Y)
        if row["target_distance"] < best["target_distance"]:
            best = row
        if (t + 1) % 20 == 0 or t == args.rounds - 1:
            picks = " ".join(f"{b[:5]}:{stats[b]['n']}({stats[b]['best']:.3f})" for b in branches)
            print(f"[{t+1}/{args.rounds}] best td={best['target_distance']:.3f} {best['branch']} | {picks} "
                  f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    ds.close()

    # 3. results
    top = sorted(archive, key=lambda r: r["target_distance"])[:10]
    json.dump({"best": best, "top10": top, "branch_stats": stats,
               "n_specs": len(archive)}, open(os.path.join(args.out, "best.json"), "w"), indent=2)
    np.save(os.path.join(args.out, "encodings.npy"), np.array(X, np.float32))
    print(f"\n=== UCB done: {len(archive)} specs in {(time.time()-t0)/60:.1f} min ===")
    print(f"BEST: {best['branch']} td={best['target_distance']} duct={best['value'].get('duct_score')} "
          f"gen={best['value'].get('generations')} bud={best['value'].get('bud_score')}")
    print("per-branch best td:", {b: round(stats[b]["best"], 3) for b in branches})
    print("per-branch visits:", {b: stats[b]["n"] for b in branches})
    print("archive:", os.path.join(args.out, "dataset.jsonl"))


if __name__ == "__main__":
    main()
