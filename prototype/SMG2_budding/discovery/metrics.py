"""metrics -- VERSIONED measurement adapters (this is Loop III's object). A metric maps a trajectory
(list of phi fields) -> an observables dict + an `emergent` flag (is the phenotype in the real regime?).

metric_v1 is the current smg_reward topology readout. Loop III (measurement discovery) will register
metric_v2 with a new observable that separates compositions metric_v1 cannot. RunRecords store the
result of EACH metric_version as an immutable, appended analysis -- raw runs are never re-simulated
when the metric changes.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "search"))
import numpy as np
import smg_reward as R

_REAL = None


def real_vector():
    global _REAL
    if _REAL is None:
        _REAL = R.value_vector(R._real_obs(552))
    return _REAL


def phi_to_points(phi, thr=0.5, max_pts=9000, rng=None):
    ys, xs = np.nonzero(np.asarray(phi) > thr)
    if len(xs) < 20:
        return None
    P = np.c_[xs, ys].astype(float)
    if len(P) > max_pts:
        rng = rng or np.random.default_rng(0)
        P = P[rng.choice(len(P), max_pts, replace=False)]
    return (P - P.min(0)) / (np.ptp(P, 0) + 1e-9)


def _readout(phi):
    Pn = phi_to_points(phi)
    if Pn is None:
        return None, None
    o = R.obs_2d(Pn, W=1.0)
    return o, R.value_vector(o)


def measure_v0(traj):
    """metric_v0 -- the FROZEN, trusted, coarse anchor (bootstrap ladder rung 0). Emergence = the final
    shape STAYS IN THE REAL REGIME: a connected, branch-like gland (not fragment/blob/cluster/collapse).
    It reliably detects gross STRUCTURAL failures (fragmentation, collapse) but by design CANNOT measure
    developmental subdivision -- the IC is the real t=0 gland, already branch-like (bud 0.47/gen 7), and
    the readout saturates. That inadequacy is the NAMED FAILURE handed to Loop III (measurement
    discovery), which will add a subdivision observable as metric_v1. We report d_bud/d_gen for that
    later use, but they are NOT in the v0 emergence test."""
    phiT, phi0 = np.asarray(traj[-1]), np.asarray(traj[0])
    area = float((phiT > 0.5).mean())
    oT, vT = _readout(phiT)
    if vT is None:
        return dict(metric="metric_v0", cls="collapsed", duct=0.0, cluster=0.0, bud=0.0,
                    generations=0, d_bud=0.0, d_gen=0, area=area, emergent=False)
    _o0, v0 = _readout(phi0)
    growth = area / max(float((phi0 > 0.5).mean()), 1e-6)
    cls = R.classify(oT, growth_ratio=growth)
    emergent = bool(cls == "branch-like" and vT["duct_score"] > 0.55 and vT["cluster_score"] < 0.25)
    return dict(metric="metric_v0", cls=cls, duct=round(vT["duct_score"], 3),
                cluster=round(vT["cluster_score"], 3), bud=round(vT["bud_score"], 3),
                generations=int(vT["generations"]),
                d_bud=round(float(vT["bud_score"] - (v0["bud_score"] if v0 else 0)), 3),
                d_gen=int(vT["generations"] - (v0["generations"] if v0 else 0)),
                area=round(area, 3), emergent=emergent)


def _lobules(phi):
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    b = np.asarray(phi) > 0.5
    lbl, nc = ndi.label(b)
    if nc == 0 or b.sum() < 20:
        return 0.0
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    body = lbl == int(sizes.argmax())
    edt = ndi.distance_transform_edt(body)
    return float(len(peak_local_max(edt, min_distance=6, threshold_abs=3.0)))


def measure_v1(traj):
    """metric_v1 -- promoted by Loop III (measurement discovery). It EXTENDS metric_v0's regime check
    with the subdivision observable that resolved the named failure: the composition must not only stay
    branch-like/connected but SUBDIVIDE the already-branched initial condition (more lobules than t=0).
    This is the observable metric_v0 could not measure; under it, cleft/growth operators become
    necessary (the reopened Loop-I conclusion)."""
    v0 = measure_v0(traj)
    lob_T = _lobules(traj[-1]); lob0 = _lobules(traj[0])
    subdivided = lob_T > 1.3 * max(lob0, 1.0)
    v0.update(metric="metric_v1", lobules=int(lob_T), lobules0=int(lob0),
              emergent=bool(v0["emergent"] and subdivided))
    return v0


METRICS = {"metric_v0": measure_v0, "metric_v1": measure_v1}   # v0 frozen anchor; v1 promoted by Loop III


if __name__ == "__main__":
    rv = real_vector()
    print("real regime target: duct", rv["duct_score"], "cluster", rv["cluster_score"],
          "gen", rv["generations"])
