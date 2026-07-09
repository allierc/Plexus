"""bootstrap -- learn the FAILURE MANIFOLD before UCB.

Run N stratified-random specs over the mechanism branches; label each with the value vector +
failure class + the spec encoding -> a dataset the surrogate (next) trains on. Without this prior
the tree has no prior and rediscovers clusters. Pre-flight: the reward `calibration_gate()` MUST pass.

NOTE (v1): runs on the DISC substrate (real-init is TODO in the search worker); gray_scott Turing is
not yet wired, so `signaling_like_field` uses growth_gate's prescribed-field fallback.

  python search/bootstrap.py --n 320 [--frames 700 --stride 20 --out search/_bootstrap]
"""
import os, sys, json, time, argparse, copy, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, os.path.join(HERE, "..", "..", "active_matter2"))
sys.path.insert(0, HERE)
import numpy as np
import torch
import yaml
import plexus.operators   # noqa
import am2_ops            # noqa
import operators_smg      # noqa  registers ecm_boundary / growth_field / slow_field / growth_gate / stiffness_field
import plexus.schema as S
from plexus.engine import run
import mechanism_tree as mt
import smg_reward as R


def run_spec(spec, frames, stride, seed, dev):
    spec = copy.deepcopy(spec)
    spec["general"].pop("init", None)
    spec["sets"]["agent"]["spawn"] = "disc"                 # real-init TODO (search worker)
    spec["general"]["seed"] = int(seed); spec["general"]["n_frames"] = int(frames)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(spec, f); f.close()
    caps = {"aX": [], "occ": []}

    def hook(H, frame):
        if frame % stride:
            return
        a = H.level("agent")
        caps["aX"].append(a.get("pos").detach().cpu().numpy().copy())
        caps["occ"].append(a.occ.detach().cpu().numpy().copy())

    try:
        sim = S.load(f.name)
        run(sim, out_path=None, device=dev, on_frame=hook)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:90]}"
    return caps, None


def observe(caps):
    aX = np.array(caps["aX"]); occ = np.array(caps["occ"]) > 0
    T = len(aX)
    if T < 2:
        return None
    n0, nT = int(occ[0].sum()), int(occ[-1].sum())
    P = aX[-1][occ[-1]]
    P0 = aX[0][occ[0]]
    if len(P) < 20:
        return None                                          # collapsed -> unstable
    Pn = (P - P.min(0)) / (np.ptp(P, axis=0) + 1e-9)         # normalize like the real projection
    o = R.obs_2d(Pn, W=1.0)
    tail = max(2, T // 5); live = occ[-1]
    v = np.diff(aX[-tail:], axis=0)[:, live]
    sp = np.linalg.norm(v, axis=-1)
    polar = float(np.linalg.norm((v / np.clip(sp[..., None], 1e-9, None)).mean((0, 1))))
    growth = float(nT / max(n0, 1))
    area_ratio = float((np.ptp(P, axis=0).prod() + 1e-12) / (np.ptp(P0, axis=0).prod() + 1e-9))
    return o, polar, growth, area_ratio


def _plausible(branch, rng):
    """A hand-plausible spec: mid-range numerics (less blow-up than extremes) + sensible choices."""
    p = {}
    for k, r in mt.BRANCHES[branch]["params"].items():
        if isinstance(r, list):
            p[k] = str(r[0])
        else:
            lo, hi = r
            p[k] = float(np.clip((lo + hi) / 2 + rng.normal(0, 0.15 * (hi - lo)), lo, hi))
    return p


def _perturb(branch, params, rng):
    """Small gaussian perturbation around a known-good spec (exploit the best morphology so far)."""
    p = dict(params)
    for k, r in mt.BRANCHES[branch]["params"].items():
        if k in p and not isinstance(r, list):
            lo, hi = r
            p[k] = float(np.clip(p[k] + rng.normal(0, 0.12 * (hi - lo)), lo, hi))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--frames", type=int, default=700)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "_bootstrap"))
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    print("=== pre-flight: reward calibration gate ===", flush=True)
    if not args.skip_gate and not R.calibration_gate():
        print("CALIBRATION GATE FAILED -> aborting bootstrap"); return
    print()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    branches = list(mt.BRANCHES)
    rng = np.random.default_rng(args.seed)
    ds_path = os.path.join(args.out, "dataset.jsonl")
    enc_list, rows = [], []
    best = None
    t0 = time.time()
    with open(ds_path, "w") as fh:
        for i in range(args.n):
            # biased sampling: 50% random, 25% hand-plausible, 25% perturb-best (avoids a
            # 95%-fragment dataset -> the surrogate sees failures AND 'almost-working' tissue)
            roll = rng.random()
            if roll >= 0.75 and best is not None:
                branch = best["branch"]; params = _perturb(branch, best["params"], rng); mode = "perturb"
            elif 0.5 <= roll < 0.75:
                branch = branches[i % len(branches)]; params = _plausible(branch, rng); mode = "plausible"
            else:
                branch = branches[i % len(branches)]; params = mt.sample_params(branch, rng); mode = "random"
            seed = int(rng.integers(0, 100000))
            spec = mt.build_spec(branch, params, seed=seed, frames=args.frames)
            caps, err = run_spec(spec, args.frames, args.stride, seed, dev)
            enc, names = mt.encode(branch, params); enc_list.append(enc)
            if caps is None:
                row = dict(i=i, branch=branch, params=params, failure="unstable", err=err, value={})
            else:
                out = observe(caps)
                if out is None:
                    row = dict(i=i, branch=branch, params=params, failure="unstable", value={})
                else:
                    o, polar, growth, area_ratio = out
                    v = R.value_vector(o, migration_coherence=polar, growth_ratio=growth)
                    cls = R.classify(o, growth_ratio=growth, area_ratio=area_ratio)
                    row = dict(i=i, branch=branch, params=params, failure=cls, value=v,
                               polar=round(polar, 3), growth=round(growth, 3))
            rows.append(row); fh.write(json.dumps(row) + "\n"); fh.flush()
            if (i + 1) % 8 == 0 or i == args.n - 1:
                v = row.get("value", {})
                print(f"[{i+1}/{args.n}] {branch:22} -> {row['failure']:11} "
                      f"duct={v.get('duct_score','-')} clust={v.get('cluster_score','-')} "
                      f"[{(time.time()-t0)/(i+1):.1f}s/spec]", flush=True)

    np.save(os.path.join(args.out, "encodings.npy"), np.array(enc_list, np.float32))
    json.dump({"feature_names": names, "n": args.n}, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    from collections import Counter
    hist = Counter(r["failure"] for r in rows)
    print(f"\n=== bootstrap done: {args.n} specs in {(time.time()-t0)/60:.1f} min ===")
    print("failure manifold:", dict(hist))
    on_path = sum(1 for r in rows if r["failure"] == "branch-like"
                  or r.get("value", {}).get("duct_score", 0) > 0.4)
    print(f"on-path (branch-like or duct>0.4): {on_path}/{args.n}")
    print(f"dataset: {ds_path}  +  encodings.npy  +  meta.json")


if __name__ == "__main__":
    main()
