"""surrogate -- the value/policy net: predict the MORPHOLOGY VECTOR from a spec encoding.

Per the reviews: the surrogate predicts the full morphology vector
  {duct, cluster, bud, branch_count, branch_length_ratio, generations, migration, growth}
NOT a scalar reward (richer supervision, smoother landscape, interpretable errors, reusable). The
reward is computed AFTERWARDS from the predicted vector. Ensemble = a RandomForest whose per-tree
spread gives the epistemic UNCERTAINTY that UCB uses to explore. A separate RF classifier predicts
the FAILURE CLASS (the surrogate learns what KIND of failure a spec produces).

Trains on the bootstrap dataset (search/_bootstrap/{dataset.jsonl, encodings.npy}). A quality GATE
(k-fold R2 per morphology dim + class-separation accuracy) MUST pass before UCB trusts it.

  python search/surrogate.py            # train on _bootstrap, print the gate; self-test if no data yet
"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
import smg_reward as R

TARGETS = ["duct_score", "cluster_score", "bud_score", "branch_count", "branch_length_ratio",
           "generations", "migration_coherence", "growth"]
BOOT = os.path.join(HERE, "_bootstrap")


# ------------------------------------------------------------------ data
def load_dataset(boot=BOOT):
    """Reconstruct the feature matrix from each row's (branch, params) -- robust to wall-killed shards
    that never wrote encodings.npy (dataset.jsonl is the single source of truth)."""
    import mechanism_tree as mt
    rows = [json.loads(l) for l in open(os.path.join(boot, "dataset.jsonl")) if l.strip()]
    X, Y, valid, cls, names = [], [], [], [], None
    for r in rows:
        enc, names = mt.encode(r["branch"], r["params"])
        X.append(enc)
        v = r.get("value") or {}
        cls.append(r.get("failure", "unstable"))
        if v:
            Y.append([float(v.get("duct_score", 0)), float(v.get("cluster_score", 0)),
                      float(v.get("bud_score", 0)), float(v.get("branch_count", 0)),
                      float(v.get("branch_length_ratio", 0)), float(v.get("generations", 0)),
                      float(v.get("migration_coherence", 0)), float(r.get("growth", 1.0))])
            valid.append(True)
        else:
            Y.append([0, 1, 0, 0, 0, 0, 0, 1]); valid.append(False)   # placeholder (excluded from regression)
    return (np.array(X, np.float32), np.array(Y, np.float32), np.array(valid),
            np.array(cls), names)


# ------------------------------------------------------------------ model
class Surrogate:
    def __init__(self, n_trees=300, seed=0):
        self.reg = RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1,
                                         min_samples_leaf=2)
        self.clf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1,
                                          min_samples_leaf=2)
        self.trained = False

    def fit(self, X, Y, valid, cls):
        self.reg.fit(X[valid], Y[valid])
        self.clf.fit(X, cls)
        self.trained = True
        return self

    def predict(self, X):
        """morphology vector (mean), per-tree UNCERTAINTY (std), failure class."""
        X = np.atleast_2d(X).astype(np.float32)
        per_tree = np.stack([t.predict(X) for t in self.reg.estimators_])   # [T, n, d]
        mean, std = per_tree.mean(0), per_tree.std(0)
        return mean, std, self.clf.predict(X)

    def value_dict(self, x):
        m, s, c = self.predict(x)
        m, s = m[0], s[0]
        return ({t: float(m[i]) for i, t in enumerate(TARGETS)},
                {t: float(s[i]) for i, t in enumerate(TARGETS)}, c[0])


# ------------------------------------------------------------------ reward from the predicted vector
def reward_from_vector(v, stage="connect"):
    w = R.STAGE_WEIGHTS[stage]
    bc = min(max(v.get("branch_count", 0) / 4.0, 0), 1)
    return round(float(w["duct"] * v["duct_score"] + w["bud"] * v["bud_score"] + w["branch"] * bc
                       + w["cluster"] * v["cluster_score"] + w["migr"] * v["migration_coherence"]), 3)


# ------------------------------------------------------------------ quality gate (before UCB)
def quality_gate(X, Y, valid, cls, k=5):
    print(f"dataset: {len(X)} specs ({valid.sum()} with morphology, {len(X)-valid.sum()} unstable)")
    ok = True
    if valid.sum() >= 2 * k:
        Xv, Yv = X[valid], Y[valid]
        reg = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1, min_samples_leaf=2)
        pred = cross_val_predict(reg, Xv, Yv, cv=KFold(k, shuffle=True, random_state=0))
        print(f"\n{'morphology dim':22} {'CV R^2':>7}")
        r2s = []
        for i, t in enumerate(TARGETS):
            r2 = r2_score(Yv[:, i], pred[:, i]); r2s.append(r2)
            print(f"  {t:20} {r2:7.3f}")
        avg = float(np.mean(r2s))
        key = float(np.mean([r2s[TARGETS.index(t)] for t in ("duct_score", "cluster_score")]))
        print(f"  {'mean':20} {avg:7.3f}   (duct+cluster mean {key:.3f})")
        ok = key > 0.3
    else:
        print("  (too few valid specs for CV -- bootstrap more)"); ok = False
    if len(np.unique(cls)) >= 2 and len(X) >= 2 * k:
        cpred = cross_val_predict(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1,
                                  min_samples_leaf=2), X, cls, cv=KFold(k, shuffle=True, random_state=0))
        acc = accuracy_score(cls, cpred)
        base = max(np.bincount([list(np.unique(cls)).index(c) for c in cls])) / len(cls)
        print(f"\nfailure-class CV accuracy {acc:.3f}  (majority baseline {base:.3f})")
        ok = ok and acc > base + 0.1
    print(f"\n=== SURROGATE GATE: {'PASS -- ready for UCB' if ok else 'FAIL -- bootstrap more / weak features'} ===")
    return ok


# ------------------------------------------------------------------ self-test / run
def _self_test():
    print("[self-test] no bootstrap dataset yet -> synthetic sanity check")
    import mechanism_tree as mt
    rng = np.random.default_rng(0)
    Xs, Ys, cs = [], [], []
    for _ in range(120):
        b = rng.choice(list(mt.BRANCHES)); p = mt.sample_params(b, rng)
        enc, _ = mt.encode(b, p)
        duct = 0.6 * enc[list(mt.OPERATOR_VOCAB).index("ecm_boundary") if "ecm_boundary" in mt.OPERATOR_VOCAB else 0] + rng.normal(0, 0.1)
        Xs.append(enc); Ys.append([duct, 1 - duct, 0.1, 2, 1.0, 3, 0.3, 1.2]); cs.append("branch-like" if duct > 0.4 else "cluster")
    X, Y, cls = np.array(Xs, np.float32), np.array(Ys, np.float32), np.array(cs)
    s = Surrogate(n_trees=100).fit(X, Y, np.ones(len(X), bool), cls)
    m, sd, c = s.value_dict(X[0])
    print(f"[self-test] predict OK: duct={m['duct_score']:.2f} +/- {sd['duct_score']:.2f}, class={c}, "
          f"reward={reward_from_vector(m)}")


def main():
    ds = os.path.join(BOOT, "dataset.jsonl")
    n = sum(1 for _ in open(ds)) if os.path.isfile(ds) else 0
    if n < 20:
        print(f"[surrogate] only {n} specs in {ds} -- running self-test instead.")
        _self_test(); return
    X, Y, valid, cls, names = load_dataset()
    quality_gate(X, Y, valid, cls)
    s = Surrogate().fit(X, Y, valid, cls)
    imp = s.reg.feature_importances_
    top = np.argsort(imp)[::-1][:8]
    print("\ntop morphology-predictive features:")
    for i in top:
        print(f"  {names[i]:24} {imp[i]:.3f}")
    from collections import Counter
    print("\nfailure manifold:", dict(Counter(cls)))


if __name__ == "__main__":
    main()
