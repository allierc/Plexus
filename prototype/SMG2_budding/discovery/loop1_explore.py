"""loop1_explore -- Mechanism-space exploration (Loop I).

Explore CompositionSpace by ONE stage-gated legal edit at a time from a seed; run each composition over
a seed BASIN on a substrate backend; store every run as an immutable RunRecord in the archive; and
judge each mechanism claim by NECESSITY (ablate an operator → does emergence collapse?), SUFFICIENCY
(does the composition emerge at all?) and ROBUSTNESS (emergence consistent across seeds). Knowledge is
distilled from the RunRecords, never written directly. The forward MODEL is the *result* of this
search, not its target.

  simulation → RunRecord → Knowledge      (never simulation → Knowledge)

  python discovery/loop1_explore.py [--basin 2 --node_cap 12 --max_stage 2]
"""
from __future__ import annotations
import os, sys, argparse
from collections import deque
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "pf"))
import numpy as np
from composition_space import CompositionGraph, seed, OPERATORS
from run_record import RunRecord, RunArchive, comp_hash
import metrics, knowledge
import substrate


def _param_basin(g, rng, n):
    """Default params + (n-1) small perturbations within each operator's range -> emergence is tested
    across a PARAMETER BASIN (not one point), per the robustness rule (avoids single-point artifacts)."""
    base = g.default_params()
    out = [base]
    for _ in range(max(0, n - 1)):
        p = dict(base)
        for node in g.ops:
            for pn, (lo, hi, _d) in OPERATORS[node["op"]]["params"].items():
                k = f"{node['id']}.{pn}"
                p[k] = float(np.clip(base[k] + rng.normal(0, 0.12 * (hi - lo)), lo, hi))
        out.append(p)
    return out


def evaluate(g, phi0, archive, seeds, rng, param_basin=2, metric="metric_v0", parent_id=None,
             edit=None, n_record=6, stride=150):
    """Run g across a PARAMETER BASIN x SEED basin -> store RunRecords; return (emergence_rate,
    representative_obs, run_id). emergence_rate = fraction of (theta, seed) samples in the real regime."""
    n_emerge, total, obs_rep, rid = 0, 0, None, None
    for params in _param_basin(g, rng, param_basin):
        pg = g.with_params(params)
        for s in seeds:
            traj = substrate.run(pg, phi0, seed_=s, n_record=n_record, stride=stride)
            rec = RunRecord(pg, params=params, seed=s, backend=substrate.BACKEND,
                            parent_id=parent_id, edit=edit)
            ref = archive.save_trajectory(rec.run_id, traj[-1])
            object.__setattr__(rec, "trajectory_ref", ref)
            obs = metrics.METRICS[metric](traj)
            rec.add_analysis(metric, {**obs, "verdict": "Established" if obs["emergent"] else "Open"})
            archive.add(rec)
            n_emerge += int(obs["emergent"]); total += 1
            obs_rep = obs_rep or obs; rid = rid or rec.run_id
    return n_emerge / total, obs_rep, rid


def explore(basin=2, node_cap=12, max_stage=2, param_basin=2, out=None):
    out = out or os.path.join(HERE, "_archive")
    phi0 = np.load(os.path.join(ROOT, "pf", "_real", "phi0.npy"))
    archive = RunArchive(out)
    seeds = [1, 2, 3][:basin]
    rng = np.random.default_rng(0)
    ADD = {"interface_relax", "tissue_grow", "cleft_induce", "confine"}   # supported single-cleft subtree

    # stage-gated BFS from EMPTY: visits growth-only / cleft-only / tension-only (impossibility cases) too
    start = seed("empty")
    frontier = deque([(start, None, None)]); seen = {comp_hash(start)}
    evals = {}
    print(f"exploring (basin={basin}, node_cap={node_cap}, max_stage={max_stage})\n"
          f"{'composition':40} {'region':30} {'rate':>4} class", flush=True)
    while frontier and len(evals) < node_cap:
        g, parent_id, edit = frontier.popleft()
        if g.ops and not substrate.translate(g)[2]:                        # supported + non-empty
            rate, obs, rid = evaluate(g, phi0, archive, seeds, rng, param_basin=param_basin, parent_id=parent_id, edit=edit)
            evals[comp_hash(g)] = dict(g=g, rate=rate, region=g.name_region(), obs=obs, run_id=rid)
            print(f"{'+'.join(sorted(set(g.op_names()))):40} {g.name_region():30} "
                  f"{rate:>4.2f} {obs['cls']}", flush=True)
        else:
            rid = None
        for e, lbl in g.legal_edits(max_stage):
            if e[0] == "add_op" and e[1] in ADD:
                g2, _ = g.apply(e)
                h = comp_hash(g2)
                if h not in seen:
                    seen.add(h); frontier.append((g2, rid, lbl))

    # NECESSITY: ablate each non-substrate-unique operator of every emergent composition
    print("\nnecessity (ablation) tests:", flush=True)
    necessity = {}
    for h, rec in list(evals.items()):
        if rec["rate"] < 0.5:
            continue
        g = rec["g"]
        for node in g.ops:
            g_ab, _ = g.apply(("remove_op", node["id"]))
            if not g_ab.ops or substrate.translate(g_ab)[2]:
                continue
            ab_rate, ab_obs, _ = evaluate(g_ab, phi0, archive, seeds, rng, param_basin=param_basin,
                                          parent_id=rec["run_id"], edit=f"-{node['op']}")
            evals.setdefault(comp_hash(g_ab), dict(g=g_ab, rate=ab_rate, region=g_ab.name_region(),
                                                   obs=ab_obs, run_id=None))
            nec = (ab_rate < 0.5)
            necessity[(h, node["op"])] = nec
            print(f"  {'+'.join(sorted(set(g.op_names()))):32} ablate {node['op']:16} "
                  f"→ rate {ab_rate:.2f} {'NECESSARY' if nec else 'not necessary'}", flush=True)

    kout = os.path.join(HERE, "knowledge.md")
    stats = knowledge.distill(evals, necessity, kout)
    n_runs = len(archive.all())
    print(f"\n=== Loop I done: {len(evals)} compositions, {n_runs} RunRecords archived ===")
    print(f"ledger: Established {stats['established']} · Structural-limitation {stats['structural']} · "
          f"Open {stats['open']}  →  {kout}")
    print(f"archive (source of truth): {out}/records.jsonl + analyses.jsonl + trajectories/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", type=int, default=2)
    ap.add_argument("--node_cap", type=int, default=12)
    ap.add_argument("--max_stage", type=int, default=2)
    ap.add_argument("--param_basin", type=int, default=2)
    a = ap.parse_args()
    explore(basin=a.basin, node_cap=a.node_cap, max_stage=a.max_stage, param_basin=a.param_basin)
