"""exp18's rulers: one broadcast angle selects among laws on the zebrafish oculomotor connectome.

    run_measure("exp18.fit", "training/neural/exp18_m3_alpha_zf285")

A training run has no trajectory. `fit` reads the trainer's own test (`results/<name>_test.json`,
`results/report.json`) and recomputes nothing it scored. `spectrum` and `swap` restore the kept
checkpoint (`models/best.pt`) and roll the trained circuit out again, in the devcontainer -- they are
the analysis GNN_Transformer.tex Part IV asks for, which the trainer's generic `analyse` cannot do: its
pole panel reads the RAW connectome W and ignores the rotation by alpha.

    fit       nmse_<law>            held-out MSE of that law, as a fraction of its own target variance
              worst_cell_nmse       the largest of them -- one circuit holds every law only if this is small
              diverged_frac         diverged steps (restored and halved) / all optimiser steps
    spectrum  (phase_rotated runs only; at each law's own operating point, the first test trial of that law)
              J(alpha) = diag(g) [W o cos(Phi - alpha)] diag(rho'),  W the Dale-signed recurrent weights,
              Phi_ij = varphi[type(j), type(i)], rho' = 1 - tanh(v)^2 time-averaged, g the coupling gain
              gap_<law>, gap_min    (Re l1 - Re l2) / (Re l2 - Re lN): the leading eigenvalue's distance to
                                    the next, as a fraction of the rest of the spectrum's width
              mode_switch           1 - |<u1(alpha_a), u1(alpha_b)>|, leading eigenvectors of two laws
              halfturn_err          max_k |l_k(J(alpha+pi)) + l_k'(J(alpha))| / max |l| -- an IDENTITY (0)
              alpha_<law>, alpha_sep    the learned angles (rad) and their smallest circular distance
    swap      swap_ratio_<law>, swap_ratio_min   that law's nmse with the angles shifted by one law (two laws: exchanged) / its own
"""
from __future__ import annotations

import os

import numpy as np

from .common import finite, register_run


# ----------------------------------------------------------------------------- pure functions (tested)
def coupling(W_post_pre, phi_pair, t_post, t_pre, alpha, gain, slope, nonneg=False):
    """J(alpha) = diag(gain) [W o F] diag(slope), F_ij = mean_l cos(Phi^(l)_ij - alpha_l), Phi^(l)_ij =
    phi_pair[l, t_pre[j], t_post[i]]. A circle: phi_pair [T, T], alpha a scalar; a torus: [L, T, T], [L]."""
    P = np.asarray(phi_pair, np.float64)
    P = P[None] if P.ndim == 2 else P
    al = np.atleast_1d(np.asarray(alpha, np.float64))
    F = np.mean([np.cos(P[l][np.asarray(t_pre)[None, :], np.asarray(t_post)[:, None]] - al[l])
                 for l in range(P.shape[0])], axis=0)
    F = 0.5 * (1.0 + F) if nonneg else F                                     # the gain-only control
    M = np.asarray(W_post_pre, np.float64) * F
    return (np.asarray(gain, float)[:, None] * M) * np.asarray(slope, float)[None, :]


def gap(eigs):
    """(Re l1 - Re l2) / (Re l2 - Re lN), MODES ordered by real part, largest first.

    A real non-symmetric J has complex-conjugate pairs: the two members are ONE oscillatory mode with
    one real part, so each pair is counted once (Im >= 0). Without this the gap of a leading oscillatory
    mode reads exactly 0 against its own conjugate (found on the exp18 smoke, 2026-10-02)."""
    lam = np.asarray(eigs, complex)
    r = np.sort(np.real(lam[lam.imag >= -1e-12 * max(1.0, float(np.abs(lam).max()))]))[::-1]
    if r.size < 3 or r[1] - r[-1] <= 0:
        return None
    return float((r[0] - r[1]) / (r[1] - r[-1]))


def leading(J):
    """The eigenvector of the eigenvalue with the largest real part, unit norm."""
    lam, V = np.linalg.eig(J)
    u = V[:, int(np.argmax(lam.real))]
    return u / np.linalg.norm(u)


def overlap(u, v):
    return float(abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)))


def halfturn_err(J_at, J_at_plus_pi):
    """The spectrum of J(alpha + pi) must be minus that of J(alpha): matched by sorting both."""
    a = np.sort_complex(np.linalg.eigvals(J_at))
    b = np.sort_complex(-np.linalg.eigvals(J_at_plus_pi))
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30))


def circ_dist(a, b):
    d = (a - b) % (2 * np.pi)
    return float(min(d, 2 * np.pi - d))


# ----------------------------------------------------------------------------- fit
def fit(T):
    res = T.results.get(f"{T.name}_test")
    rep = T.results.get("report")
    if res is None or rep is None:
        raise ValueError(f"{T.name}: no results/{T.name}_test.json or report.json")
    names = res.get("cell_names") or []
    per = res["normalised_per_cell"]
    out = {}
    for c, v in per.items():
        nm = names[int(c)] if int(c) < len(names) else c
        out[f"nmse_{nm}"] = finite(v)
    vals = [v for v in per.values() if v is not None]
    out["worst_cell_nmse"] = finite(max(vals)) if vals else None
    out["nmse"] = finite(res.get("normalised_mse"))
    steps = max(1, int(rep["n_trials"]) // max(1, int(rep["batch"]))) * int(rep["epochs"])
    out["diverged_frac"] = finite(sum(int(h.get("diverged_steps", 0)) for h in rep["history"]) / steps)
    return out


# ----------------------------------------------------------------------------- the trained circuit, again
def _restore(T, device):
    import plexus.operators  # noqa: F401
    from plexus import engine
    from plexus import trainer as TR
    spec = TR.load(os.path.join(T.dir, "config.yaml"))
    spec["_model_dir"] = os.path.basename(os.path.dirname(T.dir.rstrip("/")))
    root = os.path.dirname(os.path.dirname(os.path.dirname(T.dir.rstrip("/"))))
    engine.quiet(True)
    sim, learn, ck, _ = TR._restore(spec, device, root)
    U, Y, cond = TR._data(spec, "test", int(ck.get("n_cond", 1)), device)
    cond = np.asarray(cond.cpu() if hasattr(cond, "cpu") else cond).astype(int).reshape(-1)
    return TR, spec, sim, learn, ck, U, Y, cond


def _phase_op(H):
    op = dict(zip(H.operator_names, H.operators)).get("neuron_signal")
    return op if op is not None and hasattr(op, "phi") else None


def _device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def spectrum(T):
    import torch
    dev = _device()
    TR, spec, sim, learn, ck, U, Y, cond = _restore(T, dev)
    names = (T.results.get(f"{T.name}_test") or {}).get("cell_names") or []
    cells = sorted(set(cond.tolist()))
    out, U1 = {}, {}
    for c in cells:
        i = int(np.flatnonzero(cond == c)[0])
        vs = []
        with torch.no_grad():
            H, _ = TR.rollout(sim, learn, U[i], spec["task"], dev, grad=False,
                              watch=lambda H: vs.append(H.level("neuron").get("voltage").squeeze(-1).clone()))
        op = _phase_op(H)
        if op is None:
            return {"phase_model": 0.0}
        lvl, es = H.level("neuron"), H.level(op.edge_set)
        n = lvl.n
        w = es.get("w").detach().reshape(-1)
        if op.dale:
            w = w.abs() * op._dale_sign(lvl, es).reshape(-1).to(w.dtype)
        Wm = np.zeros((n, n))
        Wm[es.post.cpu().numpy(), es.pre.cpu().numpy()] = w.cpu().numpy()
        nt = lvl.node_type.cpu().numpy()
        g = lvl.type_params[lvl.node_type][:, 2].detach().cpu().numpy()
        slope = (1.0 - torch.tanh(torch.stack(vs)) ** 2).mean(0).cpu().numpy()
        alpha = op.alpha.detach().cpu().numpy()
        phi = op.phi.detach().cpu().numpy()
        a_c = alpha[c] if len(alpha) > c else alpha[0]                          # scalar, or [L] on a torus
        nm = names[int(c)] if int(c) < len(names) else str(c)
        nn = bool(getattr(op, "nonneg", False))
        J = coupling(Wm, phi, nt, nt, a_c, g, slope, nn)
        lam = np.linalg.eigvals(J)
        out[f"gap_{nm}"] = gap(lam)
        out[f"alpha_{nm}"] = float(np.atleast_1d(a_c)[0])
        out[f"lead_re_{nm}"] = float(np.max(lam.real))
        U1[nm] = leading(J)
        if nt.max() == 1:
            # A QL BIT (two cell types, D4): is the leading mode Scholes' in-phase (a1 + a2) or anti-phase
            # (a1 - a2) emergent state? `ql_inphase` / `ql_antiphase` = |<u1, (1_a1 +/- 1_a2)/sqrt(n)>|, the
            # share of the leading eigenvector on each; together near 1 when u1 is the uniform pair.
            e1, e2 = (nt == 0).astype(float), (nt == 1).astype(float)
            out[f"ql_inphase_{nm}"] = overlap(U1[nm], e1 + e2)
            out[f"ql_antiphase_{nm}"] = overlap(U1[nm], e1 - e2)
        if c == cells[0] and not nn:                                       # a gain has no half-turn identity
            # THE IDENTITY, in float64 (exp18 Decisions 2026-10-02): (1) per edge, the OPERATOR's own factor
            # Re(e^{i varphi} conj z) at this law's context against the ruler's cos(Phi - alpha_c); (2) the matrix
            # identity J(alpha + pi) = -J(alpha). The eigenvalue comparison stays as `halfturn_eig_err`, for
            # information only: on a trained 285 x 285 non-normal J the eigensolver alone moved it to 6e-7.
            # the operator's formula on its own context read and parameters, in float64 (its float32 arithmetic
            # differs by rounding only, kept as `edge_factor_err_f32`)
            pre, post = es.pre.cpu().numpy(), es.post.cpu().numpy()
            a64 = alpha.astype(np.float64)
            a64 = a64[:, None] if a64.ndim == 1 else a64                           # [K, L]
            P64 = phi.astype(np.float64)
            P64 = P64[None] if P64.ndim == 2 else P64                              # [L, T, T]
            if op.ctx_set is None:
                zr, zi = np.cos(a64[0]), np.sin(a64[0])
            else:
                lo, hi = op.ctx_lines
                cc = H.level(op.ctx_set).get(op.ctx_block)[..., lo:hi, 0].detach().cpu().double().numpy().reshape(-1)
                zr, zi = (cc[:, None] * np.cos(a64)).sum(0), (cc[:, None] * np.sin(a64)).sum(0)   # [L]
            ph = P64[:, nt[pre], nt[post]].T                                       # [E, L]
            rot_op = (np.cos(ph) * zr + np.sin(ph) * zi).mean(-1)
            rot_ruler = np.mean(np.cos(ph - np.atleast_1d(a_c)), -1)
            zr32, zi32 = (x.reshape(-1).detach().cpu().double().numpy() for x in op.broadcast(H))
            out["edge_factor_err_f32"] = float(np.max(np.abs((np.cos(ph) * zr32 + np.sin(ph) * zi32).mean(-1) - rot_ruler)))
            Jpi = coupling(Wm, phi, nt, nt, a_c + np.pi, g, slope)
            out["edge_factor_err"] = float(np.max(np.abs(rot_op - rot_ruler)))
            out["matrix_halfturn_err"] = float(np.max(np.abs(Jpi + J)) / max(np.max(np.abs(J)), 1e-30))
            out["halfturn_err"] = max(out["edge_factor_err"], out["matrix_halfturn_err"])
            out["halfturn_eig_err"] = halfturn_err(J, Jpi)
    gaps = [v for k, v in out.items() if k.startswith("gap_") and v is not None]
    out["gap_min"] = min(gaps) if gaps else None
    ks = list(U1)
    if len(ks) >= 2:
        out["mode_switch"] = 1.0 - min(overlap(U1[a], U1[b]) for a in ks for b in ks if a < b)
        al = [out[f"alpha_{k}"] for k in ks]
        out["alpha_sep"] = min(circ_dist(x, y) for i, x in enumerate(al) for y in al[i + 1:])
    out["phase_model"] = 1.0
    return {k: finite(v) for k, v in out.items()}


def swap(T):
    import torch
    dev = _device()
    TR, spec, sim, learn, ck, U, Y, cond = _restore(T, dev)
    key = "neuron_signal.alpha"
    if key not in learn.p:
        return {"phase_model": 0.0}
    res = T.results.get(f"{T.name}_test") or {}
    names, own = res.get("cell_names") or [], res.get("normalised_per_cell") or {}
    n = min(int(spec["task"]["reference"].get("n_test", 24)), U.shape[0])
    ch = int(spec["task"]["observe"].get("channel", 0))
    a = learn.p[key].detach().clone()
    if a.numel() < 2:
        return {"phase_model": 1.0, "n_angles": float(a.numel())}
    # each law gets the NEXT law's angle (a cyclic shift); for two laws this is the exchange of batch 1
    learn.restore({key: a.roll(1, 0)})
    with torch.no_grad():
        _, Yp = TR.rollout(sim, learn, U[:n], spec["task"], dev, grad=False)
    learn.restore({key: a})
    nn_ = min(Yp.shape[-2], Y.shape[-2])
    per = ((Yp[:n, :nn_, ch] - Y[:n, :nn_, 0]) ** 2).mean(-1).cpu().numpy()
    c_np = cond[:n]
    out, ratios = {"phase_model": 1.0}, []
    for c in sorted(set(c_np.tolist())):
        var = float((Y[:n][torch.as_tensor(c_np == c, device=Y.device), :, 0] ** 2).mean())
        sw = float(per[c_np == c].mean()) / max(var, 1e-30)
        nm = names[int(c)] if int(c) < len(names) else str(c)
        out[f"nmse_swapped_{nm}"] = finite(sw)
        o = own.get(str(c))
        if o:
            out[f"swap_ratio_{nm}"] = finite(sw / o)
            ratios.append(sw / o)
    out["swap_ratio_min"] = finite(min(ratios)) if ratios else None
    return out


def jitter(T, sigmas=(0.0, 0.05, 0.1, 0.2), spread=0.3):
    """Robustness of a TRAINED circuit to an imprecise broadcast: the test set re-run with every angle offset by a
    per-trial Gaussian of standard deviation sigma (rad; `alpha_jitter`), and with a per-receiver offset of
    standard deviation `spread` (rad; `alpha_spread`, coherence R = exp(-spread^2/2)). Keys: worst_jit<100 sigma>,
    worst_spread<100 spread> -- the worst law's nmse; and the same over the run's own worst law (`_ratio`)."""
    import torch
    dev = _device()
    TR, spec, sim, learn, ck, U, Y, cond = _restore(T, dev)
    if "neuron_signal.alpha" not in learn.p:
        return {"phase_model": 0.0}
    res = T.results.get(f"{T.name}_test") or {}
    own = max((res.get("normalised_per_cell") or {"0": float("nan")}).values())
    n = min(int(spec["task"]["reference"].get("n_test", 24)), U.shape[0])
    ch = int(spec["task"]["observe"].get("channel", 0))
    op = next(o for o in sim.operators if o.op == "neuron_signal")
    base = dict(op.params)

    def worst(**kw):
        op.params = {**base, **kw}
        torch.manual_seed(0)
        with torch.no_grad():
            _, Yp = TR.rollout(sim, learn, U[:n], spec["task"], dev, grad=False)
        nn_ = min(Yp.shape[-2], Y.shape[-2])
        per = ((Yp[:n, :nn_, ch] - Y[:n, :nn_, 0]) ** 2).mean(-1).cpu().numpy()
        c = cond[:n]
        return max(float(per[c == k].mean()) / max(float((Y[:n][torch.as_tensor(c == k, device=Y.device), :, 0] ** 2).mean()), 1e-30)
                   for k in sorted(set(c.tolist())))
    out = {"phase_model": 1.0, "worst_own": finite(own)}
    for sg in sigmas:
        w = worst(alpha_jitter=float(sg), alpha_spread=0.0)
        out[f"worst_jit{int(round(100 * sg)):02d}"] = finite(w)
        out[f"worst_jit{int(round(100 * sg)):02d}_ratio"] = finite(w / own)
    w = worst(alpha_jitter=0.0, alpha_spread=float(spread))
    out[f"worst_spread{int(round(100 * spread)):02d}"] = finite(w)
    out[f"worst_spread{int(round(100 * spread)):02d}_ratio"] = finite(w / own)
    op.params = base
    return out


register_run("exp18.fit", fit, doc="held-out error per law, of its own variance; worst law; diverged steps")
register_run("exp18.spectrum", spectrum,
             doc="J(alpha_k) at each law's operating point: gap, leading-mode switch, half-turn identity")
register_run("exp18.swap", swap, doc="each law's error with the two broadcast angles exchanged, over its own")
register_run("exp18.jitter", jitter, doc="worst law under per-trial angle noise and per-receiver angle spread")
