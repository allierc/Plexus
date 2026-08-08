"""Relaxation alone, on a frozen sheet. No growth, no secretion, no moving surface.

WHY THIS EXISTS. Four mechanisms were tested against the holes inside full 402-frame runs -- anchor
turnover, ongoing crosslinking, the deposition rule, and a common rest length -- and only the last moved
the packing at all (d/hex 0.472 -> 0.522, against 0.885 for the seeded sheet). Inside a growing run every
one of those is confounded: the tissue is stretching, material is being deposited, and the network is
relaxing, all at once, so a null could mean the mechanism does nothing OR that growth re-injects disorder
faster than it can be removed.

Here the sheet is frozen at a mid-run state and ONLY the network acts: spring forces, rest-length
remodelling, and crosslink formation. If a configuration does not equilibrate to a uniform sheet with
everything else switched off, no amount of tuning inside a growing run will make it.

The sheet is held on its sphere -- nodes may move tangentially but not radially -- which stands in for
the tether without imposing a fixed angular position, since 81 showed the anchors do not constrain
packing anyway.
"""
import math
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def load(run, frac=0.5):
    z = np.load(f"/workspace/Plexus/log/okuda_ECM/{run}/traj.npz")
    mp = np.asarray(z["mpos"])
    f = int(frac * (mp.shape[0] - 1))
    P = mp[f]
    c = np.array([.5, .5, .5])
    r = np.linalg.norm(P - c, axis=1)
    keep = r > 0.35 * np.median(r[r > 1e-9])
    idx = np.where(keep)[0]
    remap = -np.ones(P.shape[0], np.int64)
    remap[idx] = np.arange(len(idx))
    # the bond snapshot nearest this frame
    bf = np.asarray(z["bond_frames"])
    k = int(np.argmin(np.abs(bf - f * (bf.max() / max(mp.shape[0] - 1, 1)))))
    a, b = z["bond_off"][k], z["bond_off"][k + 1]
    bi, bj = np.asarray(z["bond_i"][a:b]), np.asarray(z["bond_j"][a:b])
    st = np.asarray(z["bond_s"][a:b], float)
    ok = (remap[bi] >= 0) & (remap[bj] >= 0)
    bi, bj, st = remap[bi[ok]], remap[bj[ok]], st[ok]
    return P[keep] - c, bi, bj, st


def metrics(X, q):
    R = float(np.median(np.linalg.norm(X, axis=1)))
    lam = len(X) / (4 * math.pi * R ** 2)
    hexd = math.sqrt(2.0 / (math.sqrt(3) * lam))
    d = cKDTree(X).query(X, k=2)[0][:, 1]
    u = X / np.linalg.norm(X, axis=1)[:, None]
    gap = np.percentile(cKDTree(u).query(q, k=1)[0] * R, 99.9) / hexd
    return d.mean() / hexd, d.std() / d.mean(), gap


def run(X0, bi, bj, st, iters=4000, dt=4e-3, gamma=2e3, k_b=5e3,
        tau_r=0.0, mesh_w=1.0, rebond_every=0, cutoff=0.008, report=None):
    X = torch.tensor(X0, dtype=torch.float32, device=DEV)
    I = torch.tensor(bi, dtype=torch.long, device=DEV)
    J = torch.tensor(bj, dtype=torch.long, device=DEV)
    L0 = (X[J] - X[I]).norm(dim=1) / (1.0 + torch.tensor(st, dtype=torch.float32, device=DEV))
    rad = X.norm(dim=1, keepdim=True).clone()
    out = []
    for t in range(iters):
        dvec = X[J] - X[I]
        L = dvec.norm(dim=1).clamp_min(1e-12)
        f = (k_b * (L - L0) / gamma)[:, None] * (dvec / L[:, None])
        acc = torch.zeros_like(X).index_add_(0, I, f).index_add_(0, J, -f)
        X = X + acc * dt
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12) * rad     # stay on the sphere
        if tau_r > 0:
            tgt = (1.0 - mesh_w) * L + mesh_w * L.mean()
            L0 = L0 + (tgt - L0) / tau_r
        if rebond_every and (t + 1) % rebond_every == 0:
            # REWIRE: rebuild the connectivity from proximity. Holding the topology fixed is why the
            # network stalls at d/hex 0.64 -- an irregular connectivity has an irregular equilibrium, and
            # no amount of relaxation escapes it, because the springs are pulling toward whatever
            # arrangement the existing edges encode. Bonds must be able to LEAVE as well as arrive.
            Xn = X.detach().cpu().numpy()
            tr = cKDTree(Xn)
            _, nb = tr.query(Xn, k=7)
            a_ = np.repeat(np.arange(len(Xn)), 6)
            b_ = nb[:, 1:].ravel()
            keyp = np.unique(np.minimum(a_, b_) * (len(Xn) + 1) + np.maximum(a_, b_))
            ai, aj = keyp // (len(Xn) + 1), keyp % (len(Xn) + 1)
            I = torch.tensor(ai, dtype=torch.long, device=DEV)
            J = torch.tensor(aj, dtype=torch.long, device=DEV)
            dv = (X[J] - X[I]).norm(dim=1)
            L0 = torch.full_like(dv, float(dv.mean()))      # a new edge takes the common mesh length
        if report and (t + 1) % report == 0:
            # CV OF THE BOND LENGTHS ABOUT THEIR COMMON TARGET, as a time series. The endpoint alone
            # cannot tell a slow exponential from a plateau; if the network really is relaxing toward one
            # mesh size this decays as exp(-t/T) toward zero, and the fitted T is the relaxation time.
            # If it flattens above zero, the network has an equilibrium that is not uniform and no
            # further relaxation time will help.
            Ln = (X[J] - X[I]).norm(dim=1)
            out.append((t + 1, float(Ln.std() / Ln.mean())) + metrics(X.cpu().numpy(), run.q))
    return X.cpu().numpy(), out


run.q = None

if __name__ == "__main__":
    g = np.random.default_rng(0)
    q = g.normal(size=(40000, 3)); q /= np.linalg.norm(q, axis=1)[:, None]
    run.q = q
    X0, bi, bj, st = load("82_mesh_restlength", 0.5)
    print(f"frozen sheet from 82 at mid-run: {len(X0):,} nodes, {len(bi):,} crosslinks")
    print(f"  start                          d/hex {metrics(X0,q)[0]:.3f}  cv {metrics(X0,q)[1]:.3f}  gap {metrics(X0,q)[2]:.2f}")
    print()
    import numpy as _np
    print(f"  {'iters':>7} {'cv(L)':>8} {'d/hex':>8}   fit cv = A exp(-t/T) + C")
    for lbl, kw in (("topology FIXED", dict(tau_r=60.0, mesh_w=1.0)),
                    ("REWIRE every 20", dict(tau_r=60.0, mesh_w=1.0, rebond_every=20))):
        _, series = run(X0, bi, bj, st, iters=6000, report=200, **kw)
        t = _np.array([r[0] for r in series], float)
        cv = _np.array([r[1] for r in series], float)
        # fit an exponential with an offset: the offset IS the question
        from scipy.optimize import curve_fit
        try:
            popt, _ = curve_fit(lambda x, A, T, C: A * _np.exp(-x / T) + C, t, cv,
                                p0=[cv[0] - cv[-1], 1000.0, cv[-1]], maxfev=20000)
            A, T, C = popt
            fit = f"A {A:.3f}  T {T:>7.0f} iters  C {C:.3f}"
        except Exception as e:
            fit = f"fit failed: {e}"
        print(f"  {lbl:<16} cv {cv[0]:.3f} -> {cv[-1]:.3f}   {fit}")
    raise SystemExit
    for lbl, kw in (("springs only, topology FIXED",       dict(tau_r=0.0)),
                    ("+ common mesh rest length",         dict(tau_r=60.0, mesh_w=1.0)),
                    ("+ REWIRE every 500",                dict(tau_r=60.0, mesh_w=1.0, rebond_every=500)),
                    ("+ REWIRE every 100",                dict(tau_r=60.0, mesh_w=1.0, rebond_every=100)),
                    ("+ REWIRE every 20",                 dict(tau_r=60.0, mesh_w=1.0, rebond_every=20))):
        X, _ = run(X0, bi, bj, st, iters=4000, **kw)
        dh, cv, gp = metrics(X, q)
        print(f"  {lbl:<34} d/hex {dh:.3f}  cv {cv:.3f}  gap {gp:.2f}")
