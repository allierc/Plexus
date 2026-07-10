"""rescore_sample -- re-run a stratified sample of bootstrap specs and score them with the
TIGHTENED readout (sigma 2.5 + skeleton tissue-support), so we can see how many of the old
'branch-like duct=1.0' winners were readout artifacts (skeleton bridging fragment gaps).

The bootstrap saved morphology records but NOT raw point clouds, so we must re-simulate. We re-run
each spec once (fixed seed) with real-init and apply the NEW readout to the final cloud. We report,
per spec, the OLD recorded duct/class vs the NEW duct/support/class + a 'survives' flag.

  python search/rescore_sample.py [--n_win 28 --n_frag 8 --frames 250]
"""
import os, sys, json, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import bootstrap as B
import smg_reward as R


def _new_score(caps):
    aX = np.array(caps["aX"]); occ = np.array(caps["occ"]) > 0
    P = aX[-1][occ[-1]]
    if len(P) < 20:
        return dict(duct_score=0.0, cluster_score=0.0, skel_support=0.0), "unstable", 0.0
    Pn = (P - P.min(0)) / (np.ptp(P, axis=0) + 1e-9)
    o = R.obs_2d(Pn, W=1.0)
    growth = float(occ[-1].sum() / max(occ[0].sum(), 1))
    v = R.value_vector(o); cls = R.classify(o, growth_ratio=growth)
    return v, cls, float(o.get("skel_support", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_win", type=int, default=28)
    ap.add_argument("--n_frag", type=int, default=8)
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for f in glob.glob(os.path.join(HERE, "_bootstrap", "shard_*", "dataset.jsonl"))
            for l in open(f) if l.strip()]
    rows = [r for r in rows if r.get("value")]
    winners = [r for r in rows if r["failure"] == "branch-like" or r["value"].get("duct_score", 0) > 0.4]
    frags = [r for r in rows if r["failure"] == "fragment"]

    # stratify winners across branches; take highest-old-duct per branch first, then fill
    by_branch = {}
    for r in sorted(winners, key=lambda r: -r["value"]["duct_score"]):
        by_branch.setdefault(r["branch"], []).append(r)
    sel_win, k = [], 0
    while len(sel_win) < min(args.n_win, len(winners)):
        added = False
        for b in by_branch:
            if k < len(by_branch[b]):
                sel_win.append(by_branch[b][k]); added = True
                if len(sel_win) >= args.n_win:
                    break
        k += 1
        if not added:
            break
    rng = np.random.default_rng(args.seed)
    sel_frag = [frags[i] for i in rng.choice(len(frags), min(args.n_frag, len(frags)), replace=False)] if frags else []
    sel = [("winner", r) for r in sel_win] + [("fragment", r) for r in sel_frag]
    print(f"re-scoring {len(sel)} specs ({len(sel_win)} old-winners + {len(sel_frag)} old-fragments) "
          f"@ frames={args.frames} with TIGHTENED readout\n", flush=True)
    print(f"{'kind':8} {'branch':20} {'duct_old':>8} {'duct_new':>8} {'suppt':>6} {'class_new':11} survives")

    survive = frag_flip = 0
    for kind, r in sel:
        caps, err = B.run_spec(B.mt.build_spec(r["branch"], r["params"], seed=args.seed, frames=args.frames),
                               args.frames, max(1, args.frames // 3), args.seed, "cuda:0")
        if caps is None:
            print(f"{kind:8} {r['branch']:20} {'RERUN ERR':>8} {err}"); continue
        vnew, cls, supp = _new_score(caps)
        surv = (cls == "branch-like" and vnew["duct_score"] > 0.4)
        if kind == "winner":
            survive += int(surv); frag_flip += int(cls == "fragment")
        print(f"{kind:8} {r['branch']:20} {r['value']['duct_score']:>8} {vnew['duct_score']:>8} "
              f"{supp:>6.2f} {cls:11} {'YES' if surv else 'no'}", flush=True)

    nwin = len(sel_win)
    print(f"\n=== of {nwin} old-'winners' re-scored: {survive} survive as connected ducts "
          f"({100*survive/max(nwin,1):.0f}%), {frag_flip} flip to FRAGMENT ({100*frag_flip/max(nwin,1):.0f}%) ===")


if __name__ == "__main__":
    main()
