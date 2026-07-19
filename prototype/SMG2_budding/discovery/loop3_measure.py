"""loop3_measure -- Measurement discovery (Loop III).

metric_v0 reached a NAMED FAILURE: from the real (already-branch-like) t=0 IC it cannot measure
developmental subdivision, so it cannot separate the in-regime compositions (focal-ECM vs Turing) nor
prove any operator necessary. Loop III searches a BANK of candidate OBSERVABLES for one that resolves
that failure under the TRIPLE CRITERION (a promotion gate):
  1. GT-agreement   -- orders the real gland t=0 < t=552 in subdivision (the trusted anchor),
  2. separation     -- distinguishes two Loop-I-distinct compositions (focal-ECM vs Turing),
  3. nuisance-invariance -- low coefficient of variation across seeds.
The winner is promoted to metric_v1 and the ARCHIVED RunRecords are RE-SCORED by APPENDING a versioned
analysis -- never re-simulated (immutability guardrail). This is measurement science: the observable is
a new biological measurement (lobule count / subdivision), not a better loss function.

  python discovery/loop3_measure.py
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "pf"))
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from run_record import RunArchive
import substrate


# ------------------------------------------------------------------ candidate observable bank
def _body(phi):
    b = np.asarray(phi) > 0.5
    lbl, nc = ndi.label(b)
    if nc == 0:
        return b
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    return lbl == int(sizes.argmax())


def obs_lobules(phi):
    """Number of distinct lobules = EDT peaks of the connected body (a DEVELOPMENTAL count of buds)."""
    body = _body(phi)
    if body.sum() < 20:
        return 0.0
    edt = ndi.distance_transform_edt(body)
    return float(len(peak_local_max(edt, min_distance=6, threshold_abs=3.0)))


def obs_solidity(phi):
    """1 - area/filled-bbox-ish: a subdivided (clefted) mass has lower solidity than a smooth blob."""
    body = _body(phi)
    a = body.sum()
    if a < 20:
        return 0.0
    filled = ndi.binary_fill_holes(ndi.binary_closing(body, iterations=6))
    return float(1.0 - a / max(filled.sum(), 1))


def obs_perimeter_ratio(phi):
    """Boundary length / sqrt(area): clefting increases perimeter for fixed area (a shape-complexity obs)."""
    body = _body(phi)
    a = body.sum()
    if a < 20:
        return 0.0
    per = np.logical_xor(body, ndi.binary_erosion(body)).sum()
    return float(per / np.sqrt(a))


BANK = {"lobule_count": obs_lobules, "cleft_solidity": obs_solidity, "perimeter_ratio": obs_perimeter_ratio}


# ------------------------------------------------------------------ triple criterion
def _stats(vals):
    v = np.asarray(vals, float)
    return v.mean(), v.std()


def evaluate_observable(fn, gt_frames, focal_finals, turing_finals):
    e0 = fn(gt_frames[0]); eT = fn(gt_frames[-1])
    gt_agree = eT > e0 + 1e-6                                  # real gland subdivides more by t=552
    mf, sf = _stats([fn(p) for p in focal_finals])
    mt, st = _stats([fn(p) for p in turing_finals])
    sep = abs(mf - mt) / (0.5 * (sf + st) + 1e-6)             # effect size (separation)
    cv = 0.5 * (sf / (abs(mf) + 1e-6) + st / (abs(mt) + 1e-6))  # nuisance CV across seeds
    invariant = cv < 0.35
    return dict(gt_t0=round(e0, 2), gt_tT=round(eT, 2), gt_agree=bool(gt_agree),
                focal=round(mf, 2), turing=round(mt, 2), separation=round(sep, 2),
                cv=round(cv, 2), invariant=bool(invariant),
                admissible=bool(gt_agree and invariant and sep > 1.0))


def main():
    phi0 = np.load(os.path.join(ROOT, "pf", "_real", "phi0.npy"))
    tgt = np.load(os.path.join(ROOT, "pf", "_real", "targets.npz"))
    gt_frames = [tgt["phis"][0], tgt["phis"][-1]]             # real t=0 and t=552 (the GT anchor)

    bm = substrate.benchmarks()
    seeds = [1, 2, 3]
    focal = [substrate.run(bm["focal_ecm"], phi0, seed_=s, n_record=6, stride=150)[-1] for s in seeds]
    turing = [substrate.run(bm["turing"], phi0, seed_=s, n_record=6, stride=150)[-1] for s in seeds]

    print("Loop III — searching the observable bank for a metric_v1 that resolves the named failure\n")
    print(f"{'observable':16} {'GT t0→tT':12} {'agree':6} {'focal':6} {'turing':6} {'sep':>5} "
          f"{'cv':>5} {'invar':6} admissible")
    results = {}
    for name, fn in BANK.items():
        r = evaluate_observable(fn, gt_frames, focal, turing)
        results[name] = r
        print(f"{name:16} {str(r['gt_t0'])+'→'+str(r['gt_tT']):12} {str(r['gt_agree']):6} "
              f"{r['focal']:>6} {r['turing']:>6} {r['separation']:>5} {r['cv']:>5} "
              f"{str(r['invariant']):6} {r['admissible']}")

    admissible = {k: v for k, v in results.items() if v["admissible"]}
    if not admissible:
        print("\nno observable passed the triple criterion — bank must be extended.")
        return
    winner = max(admissible, key=lambda k: admissible[k]["separation"])
    w = results[winner]
    print(f"\n=== PROMOTE metric_v1: observable '{winner}' ===")
    print(f"  GT-agreement {w['gt_t0']}→{w['gt_tT']} · separation {w['separation']} "
          f"(focal {w['focal']} vs turing {w['turing']}) · CV {w['cv']}")

    # re-score the archived Loop-I RunRecords: APPEND a metric_v1 analysis; NEVER re-simulate
    arch = RunArchive(os.path.join(HERE, "_archive"))
    fn = BANK[winner]; n = 0
    for rec in arch.all():
        phi = arch.load_trajectory(rec)
        if phi is None or "metric_v1" in rec.analyses:
            continue
        val = fn(phi)
        arch.add_analysis(rec.run_id, "metric_v1", {"metric": "metric_v1", "observable": winner,
                                                    winner: round(float(val), 3)})
        n += 1
    print(f"  re-scored {n} archived RunRecords with metric_v1 (versioned analysis appended, no re-sim)")
    print(f"\nNamed failure resolved: metric_v0 could not separate focal-ECM from Turing (both in-regime); "
          f"metric_v1 '{winner}' separates them by effect size {w['separation']} while agreeing with GT "
          f"and staying seed-invariant. → reopen Loop I necessity tests under metric_v1.")


if __name__ == "__main__":
    main()
