"""exp17: THE LINEARISED DYNAMICS OF A TRAINED LAW (Cedric, 2026-10-10: "a Jacobian or impulse-response test ... launch
them on 22 to 26"). Local, a landed run's models/best.pt, no training.

THE LAW (cell_ops StateDiffuseNeuronGraph, ...Phase, ...Grid and their _video twins), in its forward's normalised units
z = (x - mu) / sd (mu, sd the recording's dF/F mean and SD over its frames, as trainer._trace_setup hands them to the
law), t in seconds (one frame = the recording's clock, 0.914 s):
    tau_i dz_i/dt = -z_i + V_i [+ dV_ik] + Omega_i(t) m_i(z) + B_i . (M_i (.) u(t))      1/tau_i = rate_i / 0.914 s
    neuron_graph        m_i = sum_s sum_{j -> i} W^s_ij tanh z_j                          (short / mid / long edges)
    neuron_graph_phase  m_i = sum_s sum_{j -> i} W^s_ij cos(phi_ij - alpha(t)) tanh z_j   (Omega = 1 unless set)
    neuron_grid         m_i = g_i (1/8) sum_{l in cube(i)} sum_{k -> l} w_kl c_k,  c_k = (1/n_k) sum_{j -> k} a_j tanh z_j
                        (a_j = A_send, w_kl = W_grid^2, g_i = G_recv^2: sign: neuron)
LINEARISED at a reference state z* (the input and the rest are constants: they drop out):
    J = diag(1/tau_i) (-I + diag(Omega_i) W_eff diag(tanh'(z*_j))),      tanh' = 1 - tanh^2,   [1/s]
    W_eff   neuron_graph: the W^s summed over the sets (a sparse N x N matrix, row the receiver)
            neuron_graph_phase: W^s_ij cos(phi_ij - alpha_k) at block k's circular-mean angle alpha_k (exp17_angle's):
                                one J per stimulus block
            neuron_grid: diag(g) (1/8) Dec Hop diag(1/n) Enc diag(a), kept as its four sparse factors (never densified)
    Omega_i the time mean of Omega_i(t) over every frame of the recording (the law's own SIREN, evaluated per frame);
            per block, over the block's frames, for the per-block J; 1 for a law without modulation
    z*      (a) each neuron's mean over the whole recording -- the laws without a block term in the coupling (22.3, 23.3);
            (b) each neuron's mean over block k's frames -- the per-block J of the laws whose coupling changes with the
            block (24.10: alpha_k); (a) is also computed there with each alpha_k, its abscissa in the json
THE SPECTRUM: the K = 50 eigenvalues with the largest real part (ARPACK, scipy.sparse.linalg.eigs, which='LR') and the
20 with the largest imaginary part (which='LI'), against the leak-only spectrum -1/tau_i (W = 0). Reported: the spectral
abscissa (the largest Re lambda) against the leak-only one (-1/tau_max, the slowest leak); how many of the K have a time
constant -1/Re lambda longer than the slowest leak; any Re lambda > 0; the largest |Im lambda|; per mode the share of
|v_i|^2 in each Z-Brain region (data/atlas_destripe.npz, the table regions of exp17_atlas.REGIONS, masks overlapping)
and its participation ratio 1 / sum p_i^2 (p_i = |v_i|^2 / |v|^2: the number of neurons it spans).
THE IMPULSE RESPONSE: z(0) = 1 on one region's neurons, 0 elsewhere, propagated by dz/dt = J z for 60 s (RK4 on the GPU,
the step 1 / the Gershgorin bound of J, at most 0.05 s), |z_i(t)| integrated over the 60 s and summed per region; six
regions (PULSE, made disjoint: a neuron in two of them belongs to neither) and the rest of the brain as a seventh row.
Against W = 0, where z_i(t) = exp(-t / tau_i) on the pulsed region only (the diagonal, exactly).

THE NULL: batch 17's random graph 17.7 (zap_g17_random: the same degrees, the senders drawn uniformly) is the plain
neuron_graph law with Omega = 1 and no rate bounds, on the ephys recording: comparable to its spatial twin 17.9
(zap_g17_mesh3, the same law on the multi-level mesh), not to 22.3-24.10. Run both: `tools/exp17_jacobian.py
zap_g17_random zap_g17_mesh3`.

BATCHES 25 AND 26 are run with this tool when they land (results/<run>_test.json present): `--landed` lists them.

    PYTHONPATH=src:tools python tools/exp17_jacobian.py zap_n22_markall zap_n23_markall zap_n24_ph_edge_blk
    PYTHONPATH=src:tools python tools/exp17_jacobian.py --landed
-> data/jacobian_<run>.json, presentation/figs/jacobian_<run>.png
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
K_LR = 50                     # eigenvalues with the largest real part
K_LI = 20                     # eigenvalues with the largest imaginary part
T_IR = 60.0                   # the impulse response's duration, s
H_MAX = 0.05                  # its largest RK4 step, s
# the pulsed regions: (label, the exp17_atlas.REGIONS full names whose union it is)
PULSE = [("pretectum", ["Diencephalon - Pretectum"]),
         ("tectum (periventricular)", ["Mesencephalon - Tectum Stratum Periventriculare"]),
         ("dorsal thalamus", ["Diencephalon - Dorsal Thalamus"]),
         ("rhombomere 7", ["Rhombencephalon - Rhombomere 7"]),
         ("cerebellum", ["Rhombencephalon - Cerebellum"]),
         ("telencephalon", ["Telencephalon - Olfactory Bulb", "Telencephalon - Pallium", "Telencephalon - Subpallium"])]
BATCHES_LATE = ("zap_n25_", "zap_n26_")


def landed(prefixes=BATCHES_LATE):
    """{run: True / False} for every training spec of batches 25 and 26: True once results/<run>_test.json exists."""
    from plexus import trainer as T
    d = os.path.join(ROOT, "config", "training", "zapbench")
    out = {}
    for f in sorted(os.listdir(d)):
        r = f[:-5]
        if f.endswith(".yaml") and r.startswith(prefixes):
            out[r] = os.path.exists(os.path.join(T.out_dir(T.load(r), None), "results", f"{r}_test.json"))
    return out


def _norm(spec, X):
    """(mu, sd) as trainer._trace_setup computes them: over every frame, or the training frames of a held-out split."""
    from plexus.tasks import trace_recording as TR
    split = spec["task"]["reference"].get("split", "all")
    if split in ("zapbench", "recording"):
        rec = TR.load(spec["task"]["reference"]["trace_recording"])
        lab = (TR.zapbench_split(rec["offsets"], int(X.shape[0])) if split == "zapbench"
               else np.asarray(rec["split"]).astype(np.int64))
        Xt = X[lab == 0]
        return float(Xt.mean(dtype=np.float64)), float(Xt.std(dtype=np.float64))
    return float(X.mean(dtype=np.float64)), float(X.std(dtype=np.float64))


def load_law(run, device="cuda:0"):
    """The run's operator with its learned values, the recording's per-neuron and per-block means, the rates, Omega's
    time means and (phase laws) the block angles. -> dict."""
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from plexus.operators.cell_ops import StateDiffuseNeuronGraph, StateDiffuseNeuronGraphPhase, StateDiffuseNeuronGrid
    from exp17_ablation import neuron_graph_op
    from exp17_angle import circ
    spec = T.load(run)
    out = T.out_dir(spec, None)
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    op = neuron_graph_op(spec, device)
    if op is None or not isinstance(op, StateDiffuseNeuronGraph):
        raise SystemExit(f"{run}: no neuron-graph law")
    kind = ("grid" if isinstance(op, StateDiffuseNeuronGrid) else
            "phase" if isinstance(op, StateDiffuseNeuronGraphPhase) else "graph")
    if getattr(op, "synapse", "current") != "current" or getattr(op, "adapt", None) or op.activation != "tanh":
        raise SystemExit(f"{run}: J is written for current synapses, tanh, no adaptation")
    for e in spec["learnable"]:                                   # every operator parameter at its learned value
        if "param" in e:
            setattr(op, e["param"], fit[T.Learnables.key(e)].to(device).float())
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    X = np.asarray(rec["dff"], np.float32)
    F, N = X.shape
    off, names = np.asarray(rec["offsets"], np.int64), [str(n) for n in rec["names"]]
    fs = T._frame_s(rec)
    mu, sd = _norm(spec, X)
    zbar = (X.mean(0, dtype=np.float64) - mu) / sd
    zblk = np.stack([(X[off[k]:off[k + 1]].mean(0, dtype=np.float64) - mu) / sd for k in range(len(names))])
    del X
    with torch.no_grad():
        rate = op._rate(fit["neuron.tau"].to(device).float()).double().cpu().numpy().reshape(-1)
    lam = rate / fs                                               # 1/tau_i, 1/s
    om_bar, om_blk = np.ones(N), np.ones((len(names), N))
    if op.modulation != "none":                                   # the law's own Omega_i(t) at every frame
        op.n_frames_ref = F
        acc = torch.zeros(N, dtype=torch.float64, device=device)
        accb = torch.zeros(len(names), N, dtype=torch.float64, device=device)
        blk = np.searchsorted(off, np.arange(F), side="right") - 1
        with torch.no_grad():
            fs_ = op._space_features() if op.modulation == "hash" else None
            for f in range(F):
                op.frame = f
                o = op._omega(fs_).reshape(-1).double()
                acc += o
                accb[min(blk[f], len(names) - 1)] += o
        om_bar = (acc / F).cpu().numpy()
        om_blk = (accb / torch.as_tensor(np.diff(off), device=device, dtype=torch.float64)[:, None]).cpu().numpy()
    alpha_k = alpha_sd = None
    if kind == "phase":                                           # alpha(t) per frame, no training jitter
        op._train = False
        op.n_frames_ref = F
        a_ = np.empty(F)
        with torch.no_grad():
            for f in range(F):
                op.frame = f
                a_[f] = float(op.angle())
        alpha_k = np.array([circ(a_[off[k]:off[k + 1]])[0] for k in range(len(names))])
        alpha_sd = np.array([circ(a_[off[k]:off[k + 1]])[1] for k in range(len(names))])
    pos = np.load(op._pos_file[0])[op._pos_file[1]].astype(np.float64)
    usp = float(np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean()))      # a mode spread evenly over every neuron
    return dict(pos=pos, uniform_spread_um=usp, run=run, spec=spec, op=op, fit=fit, kind=kind, N=N, F=F, off=off, names=names, frame_s=fs, mu=mu, sd=sd,
                zbar=zbar, zblk=zblk, lam=lam, rate=rate, om_bar=om_bar, om_blk=om_blk, alpha_k=alpha_k, alpha_sd=alpha_sd)


def w_factors(L, alpha=None):
    """W_eff as a list of scipy sparse factors, their product the N x N matrix (row the receiver)."""
    import scipy.sparse as sp
    op, N = L["op"], L["N"]
    if L["kind"] == "grid":
        nm = op._grid["n_mesh"]
        gs, gr = (t.cpu().numpy() for t in op._grid["g2m"])
        ms, mr = (t.cpu().numpy() for t in op._grid["mm"])
        cs, cr = (t.cpu().numpy() for t in op._grid["m2g"])
        a = op.A_send.double().cpu().numpy() if op.sign == "neuron" else np.ones(N)
        w = op.W_grid.double().cpu().numpy()
        w = w ** 2 if op.sign == "neuron" else w
        g = op.G_recv.double().cpu().numpy() ** 2
        cnt = np.maximum(np.bincount(gr, minlength=nm), 1).astype(np.float64)
        Enc = sp.csr_matrix((a[gs] / cnt[gr], (gr, gs)), shape=(nm, N))          # c = Enc tanh z
        Hop = sp.csr_matrix((w, (mr, ms)), shape=(nm, nm))                        # h = Hop c
        Dec = sp.csr_matrix((g[cr] / 8.0, (cr, cs)), shape=(N, nm))               # m = Dec h
        return [Dec, Hop, Enc]
    rows, cols, vals = [], [], []
    for s in op.EDGE_SETS:
        snd, rcv = (t.cpu().numpy() for t in op._E[s])
        if not len(snd):
            continue
        w = getattr(op, f"W_{s}").double().cpu().numpy().reshape(-1)
        if L["kind"] == "phase":
            ph = (getattr(op, f"phi_{s}").double().cpu().numpy().reshape(-1) if op.phi_per == "edge"
                  else op.phi.double().cpu().numpy().reshape(-1)[op._pair[s].cpu().numpy()])
            c = np.cos(ph - alpha)
            w = w * (0.5 * (1.0 + c) if op.nonneg else c)
        rows.append(rcv), cols.append(snd), vals.append(w)
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    W.sum_duplicates()
    return [W]


class Jac:
    """J = diag(lam) (-I + diag(om) F_1 ... F_n diag(s)), as a scipy LinearOperator (CPU, float64) and a torch matvec."""

    def __init__(self, lam, om, factors, s, device, leak=1.0):
        self.lam, self.om, self.F, self.s, self.leak = lam, om, factors, s, leak    # leak 0: J = diag(lam om) W diag(s)
        self.N = len(lam)
        self.dev = device
        self.t = [self._torch(f) for f in factors]
        self.lam_t = torch.as_tensor(lam, device=device, dtype=torch.float32)[:, None]
        self.om_t = torch.as_tensor(om, device=device, dtype=torch.float32)[:, None]
        self.s_t = torch.as_tensor(s, device=device, dtype=torch.float32)[:, None]

    def _torch(self, M, dtype=torch.float32):
        import warnings
        M = M.tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                       # "sparse CSR tensor support is in beta"
            return torch.sparse_csr_tensor(torch.as_tensor(M.indptr, dtype=torch.int64),
                                           torch.as_tensor(M.indices, dtype=torch.int64),
                                           torch.as_tensor(M.data, dtype=dtype), size=M.shape).to(self.dev)

    def mv(self, v):
        y = self.s * v
        for f in reversed(self.F):
            y = f @ y
        return self.lam * (-self.leak * v + self.om * y)

    def rmv(self, v):                                             # J^T v
        y = self.om * (self.lam * v)
        for f in self.F:
            y = f.T @ y
        return -self.leak * self.lam * v + self.s * y

    def linop(self):
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator((self.N, self.N), matvec=self.mv, rmatvec=self.rmv, dtype=np.float64)

    def mv_c(self, x):                                            # complex [N] on the GPU, float64
        if not hasattr(self, "t64"):
            self.t64 = [self._torch(f, torch.float64) for f in self.F]
            self.v64 = [torch.as_tensor(a, device=self.dev, dtype=torch.float64)[:, None] for a in (self.lam, self.om, self.s)]
        lam, om, s = self.v64
        X = torch.stack([x.real, x.imag], 1)
        y = s * X
        for f in reversed(self.t64):
            y = f @ y
        y = lam * (-self.leak * X + om * y)
        return torch.complex(y[:, 0], y[:, 1])

    def mv_t(self, V):                                            # [N, B] on the GPU
        y = self.s_t * V
        for f in reversed(self.t):
            y = f @ y
        return self.lam_t * (-self.leak * V + self.om_t * y)

    def gershgorin(self):
        """max_i lam_i (1 + |om_i| sum_j |W_eff,ij| s_j): a bound on |lambda|, for the RK4 step."""
        y = self.s.copy()
        for f in reversed(self.F):
            y = abs(f) @ y
        return float((self.lam * (1.0 + np.abs(self.om) * y)).max())


SOLVER = {"name": "krylov_schur"}     # or arpack (scipy.sparse.linalg.eigs, CPU)
VERBOSE = False


def krylov_schur(J, k, which, m=None, tol=1e-5, maxit=4000, seed=0):
    """THE k EIGENPAIRS OF J WITH THE LARGEST Re (LR) OR |Im| (LI), Krylov-Schur on the GPU (Stewart 2002, SIAM J.
    Matrix Anal. Appl. 23:601): an m-step Arnoldi factorisation A V_m = V_{m+1} H (complex, classical Gram-Schmidt twice),
    the complex Schur form of H_m reordered so the wanted Ritz values lead, truncated to them, extended again. ARPACK
    (scipy's eigs) gives the same pairs; its dense steps on the CPU took ~8 min per call on this machine. Converged when
    every wanted Ritz pair's residual |J x - lambda x| / |x| < tol max(|lambda|, 0.01): 1e-7 /s at the slowest leak's
    0.01 /s. Each restart keeps the k wanted and a buffer of k / 2 more (the k-th sits in a dense cluster of slow modes,
    and without the buffer it converges in thousands of restarts). -> (eigenvalues, vectors, info)."""
    import scipy.linalg as sl
    dev = J.dev
    m = m or max(4 * k, k + 80)
    N = J.N
    g = np.random.default_rng(seed).standard_normal(N)
    V = torch.zeros(N, m + 1, dtype=torch.complex128, device=dev)
    H = np.zeros((m + 1, m), np.complex128)
    V[:, 0] = torch.as_tensor(g / np.linalg.norm(g), device=dev)
    key = {"LR": lambda x: x.real, "LI": lambda x: np.abs(x.imag), "LM": lambda x: np.abs(x)}[which]
    p, nmv = 0, 0
    for it in range(maxit):
        for j in range(p, m):
            w = J.mv_c(V[:, j])
            nmv += 1
            Vj = V[:, :j + 1]
            h = Vj.conj().T @ w
            w = w - Vj @ h
            h2 = Vj.conj().T @ w                                   # the second pass (DGKS)
            w = w - Vj @ h2
            h = (h + h2).cpu().numpy()
            b = float(torch.linalg.vector_norm(w))
            H[:j + 1, j] = h
            H[j + 1, j] = b
            V[:, j + 1] = w / b
        T, Q = sl.schur(H[:m, :m], output="complex")
        kb = min(k + max(10, k // 2), m - 20)                      # kept: the k wanted and a buffer (ARPACK's idea)
        th_ = np.sort(key(np.diag(T)))[::-1][kb - 1]
        T, Q, sdim = sl.schur(H[:m, :m], output="complex", sort=lambda x: key(np.array([x]))[0] >= th_ - 1e-14 * max(1, abs(th_)))
        p = min(max(sdim, kb), m - 10)
        ev, Y = np.linalg.eig(T[:p, :p])
        bq = H[m, m - 1] * Q[m - 1, :p]                            # the residual row after the truncation
        res = np.abs(bq @ Y) / np.linalg.norm(Y, axis=0)
        o = np.argsort(-key(ev))[:k]
        ok = res[o] < tol * np.maximum(np.abs(ev[o]), 1e-2)
        if VERBOSE and it % 25 == 0:
            print(f"    [ks {which}] restart {it}: {int(ok.sum())}/{k} converged, residual max {res[o].max():.2e}, "
                  f"wanted {key(ev[o]).min():+.5f} .. {key(ev[o]).max():+.5f}", flush=True)
        Qt = torch.as_tensor(Q[:, :p], device=dev)
        Vn = V[:, :m] @ Qt
        if ok.all() or it == maxit - 1:
            X = (Vn @ torch.as_tensor(Y[:, o], device=dev)).cpu().numpy()
            return ev[o], X / np.linalg.norm(X, axis=0), {"restarts": it + 1, "matvecs": nmv, "max_residual": float(res[o].max()),
                                                           "converged": bool(ok.all()), "m": m}
        V[:, :p] = Vn
        V[:, p] = V[:, m]
        H[:] = 0
        H[:p, :p] = T[:p, :p]
        H[p, :p] = bq


def spectrum(J, k, which, seed=0):
    """(eigenvalues, eigenvectors, info), sorted by real part (LR) or |imaginary part| (LI), largest first."""
    if SOLVER["name"] == "arpack":
        from scipy.sparse.linalg import eigs
        v0 = np.random.default_rng(seed).standard_normal(J.N)
        ev, V = eigs(J.linop(), k=k, which=which, ncv=max(4 * k, 120), tol=1e-9, maxiter=20000, v0=v0)
        info = {"solver": "arpack"}
    else:
        ev, V, info = krylov_schur(J, k, which, seed=seed)
        info["solver"] = "krylov_schur"
    o = np.argsort(-{"LR": ev.real, "LI": np.abs(ev.imag), "LM": np.abs(ev)}[which])
    return ev[o], V[:, o], info


def regions(N):
    """The table regions (exp17_atlas.REGIONS, >= 50 neurons; masks may overlap) and the disjoint pulse sets."""
    from exp17_atlas import REGIONS
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    nm, reg = [str(n) for n in za["names"]], za["regions"]
    assert reg.shape[0] == N, f"atlas_destripe.npz has {reg.shape[0]} neurons, the law {N}"
    tab = [(short, reg[:, nm.index(full)]) for full, short in REGIONS if full in nm and reg[:, nm.index(full)].sum() >= 50]
    P = np.stack([np.any([reg[:, nm.index(f)] for f in fulls], 0) for _, fulls in PULSE], 1)   # [N, 6]
    P &= (P.sum(1) == 1)[:, None]                                 # disjoint: a neuron in two pulse regions is in neither
    return tab, P


def participation(V, tab, pos):
    """Per mode: the share of p_i = |v_i|^2 / |v|^2 per region [modes, R], outside every table region, its participation
    ratio 1 / sum p_i^2 (neurons), its spatial spread sqrt(sum p_i |x_i - xbar|^2) (um, the anatomy frame) and the mass
    on its 20 largest p_i."""
    p = np.abs(V) ** 2
    p /= p.sum(0, keepdims=True)
    M = np.stack([m for _, m in tab], 1).astype(np.float64)       # [N, R]
    share = p.T @ M                                               # [modes, R]
    none_ = p[~M.any(1)].sum(0)
    xb = p.T @ pos                                                # [modes, 3]
    spread = np.sqrt(np.maximum((p * ((pos ** 2).sum(1)[:, None])).sum(0) - (xb ** 2).sum(1), 0.0))
    top20 = -np.sort(-p, axis=0)[:20].sum(0)
    return share, none_, 1.0 / (p ** 2).sum(0), spread, top20


def impulse(J, P, lam, device):
    """sum over each row region of int_0^T |z_i| dt (and of z_i), for z(0) = the pulse region's indicator; the W = 0
    twin in closed form. -> (R_abs [7, 6], R_signed [7, 6], R_w0 [7, 6], h, steps)."""
    rho = J.gershgorin()
    h = min(H_MAX, 1.0 / rho)
    n = int(np.ceil(T_IR / h))
    h = T_IR / n
    Z = torch.as_tensor(P.astype(np.float32), device=device)
    A = torch.zeros_like(Z, dtype=torch.float64)
    S_ = torch.zeros_like(Z, dtype=torch.float64)
    a0 = Z.abs().double()
    s0 = Z.double()
    with torch.no_grad():
        for _ in range(n):
            k1 = J.mv_t(Z)
            k2 = J.mv_t(Z + 0.5 * h * k1)
            k3 = J.mv_t(Z + 0.5 * h * k2)
            k4 = J.mv_t(Z + h * k3)
            Z = Z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            a1, s1 = Z.abs().double(), Z.double()
            A += 0.5 * h * (a0 + a1)
            S_ += 0.5 * h * (s0 + s1)
            a0, s0 = a1, s1
    A, S_ = A.cpu().numpy(), S_.cpu().numpy()
    rows = np.concatenate([P, ~P.any(1, keepdims=True)], 1).astype(np.float64)   # [N, 7]: the six, then the rest
    w0 = (1.0 - np.exp(-lam * T_IR)) / lam
    R_w0 = rows.T @ (P * w0[:, None])
    return rows.T @ A, rows.T @ S_, R_w0, h, n


def offdiag(R):
    """(per source: the share of its integrated |response| outside the pulsed region, all neurons; the same over the 6 x
    6 block only; overall, both)."""
    tot = R.sum(0)
    out_ = 1.0 - np.diag(R[:6]) / tot
    B = R[:6]
    blk = 1.0 - np.diag(B) / B.sum(0)
    return out_, blk, float(1.0 - np.trace(R[:6]) / R.sum()), float(1.0 - np.trace(B) / B.sum())


def analyse_one(L, s, om, alpha, tab, P, device, impulse_on=True, li=True):
    """One J: its spectrum, participation, impulse response. -> dict (json-ready) and the arrays for the figure."""
    t0 = time.time()
    J = Jac(L["lam"], om, w_factors(L, alpha), s, device)
    ev, V, inf_lr = spectrum(J, K_LR, "LR")
    t_lr = time.time() - t0
    lam = L["lam"]
    slow_leak = float(lam.min())                                  # the slowest leak's rate, 1/s
    share, none_, pr, spread, top20 = participation(V, tab, L["pos"])
    d = {"abscissa": float(ev.real.max()), "eig_re": ev.real.tolist(), "eig_im": ev.imag.tolist(),
         "n_slower_than_slowest_leak": int((ev.real > -slow_leak).sum()), "n_unstable": int((ev.real > 0).sum()),
         "max_abs_im_LR": float(np.abs(ev.imag).max()),
         "mode_participation_ratio": pr.tolist(), "mode_spread_um": spread.tolist(), "mode_mass_top20": top20.tolist(),
         "uniform_spread_um": L["uniform_spread_um"],
         "mode_region_share": share.tolist(), "mode_share_outside_table": none_.tolist(),
         "mode_top_regions": [[((tab + [("outside the table regions", None)])[j][0], float(sh_[j]))
                               for j in np.argsort(-sh_)[:3]] for sh_ in np.concatenate([share, none_[:, None]], 1)],
         "seconds_LR": t_lr, "solver_LR": inf_lr}
    if li:
        t1 = time.time()
        ei, _, inf_li = spectrum(J, K_LI, "LI")
        d.update({"eig_LI_re": ei.real.tolist(), "eig_LI_im": ei.imag.tolist(), "max_abs_im": float(np.abs(ei.imag).max()),
                  "max_abs_im_its_re": float(ei.real[np.argmax(np.abs(ei.imag))]), "seconds_LI": time.time() - t1,
                  "solver_LI": inf_li})
    if impulse_on:
        R, Rs, R0, h, n = impulse(J, P, lam, device)
        o_, b_, oa_, ob_ = offdiag(R)
        o0_, b0_, oa0_, ob0_ = offdiag(R0)
        d["impulse"] = {"rows": [p for p, _ in PULSE] + ["rest of the brain"], "cols": [p for p, _ in PULSE],
                        "abs": R.tolist(), "signed": Rs.tolist(), "w0": R0.tolist(), "rk4_step_s": h, "rk4_steps": n,
                        "outside_share": o_.tolist(), "block_offdiag_share": b_.tolist(),
                        "outside_share_all": oa_, "block_offdiag_share_all": ob_,
                        "w0_outside_share_all": oa0_, "w0_block_offdiag_share_all": ob0_,
                        "gain_vs_w0": (R.sum(0) / R0.sum(0)).tolist(),
                        "pulse_neurons": P.sum(0).astype(int).tolist()}
    return d, ev, share


def crosscheck_rest(L, device):
    """THE AUDIT'S CROSS-CHECK (an audit session, 2026-10-10: M_ij = W_ij sech^2(V_j) over the three edge sets, z* = V the
    rest; A = diag(rate) (c M - I), rate per frame, c = 1 or the mean Omega): the spectral radius of M, the abscissa of A
    at c = 1 and at c = the mean over neurons of Omega_i's time mean, and this tool's J at z* = V with Omega_i per neuron;
    then this tool's J at z* = the recording mean with Omega = 1. Each with the localisation of its leading mode."""
    V = L["fit"]["neuron.rest"].double().numpy().reshape(-1)
    th = lambda z: 1.0 - np.tanh(z) ** 2                          # noqa: E731
    Fs, fs, lam = w_factors(L), L["frame_s"], L["lam"]
    one = np.ones(L["N"])
    c = float(L["om_bar"].mean())
    out = {"omega_mean_over_neurons": c, "rest_quantiles_10_50_90": [float(np.quantile(V, q)) for q in (0.1, 0.5, 0.9)]}
    ev, X, _ = spectrum(Jac(one, one, Fs, th(V), device, leak=0.0), 6, "LM")
    out["rho_M_rest"] = float(np.abs(ev[0]))
    out["rho_M_rest_eig"] = [float(ev[0].real), float(ev[0].imag)]
    for name, om, s_ in (("rest_c1", one, th(V)), ("rest_cmean", c * one, th(V)), ("rest_omega_i", L["om_bar"], th(V)),
                         ("zbar_c1", one, th(L["zbar"]))):
        ev, X, _ = spectrum(Jac(lam, om, Fs, s_, device), 10, "LR")
        _, _, pr, sp, t20 = participation(X, [("all", np.ones(L["N"], bool))], L["pos"])
        out[name] = {"abscissa_per_s": float(ev.real[0]), "abscissa_per_frame": float(ev.real[0] * fs),
                     "top10_re_per_s": ev.real.tolist(), "n_unstable_of_10": int((ev.real > 0).sum()),
                     "lead_mode_participation_ratio": float(pr[0]), "lead_mode_spread_um": float(sp[0]),
                     "lead_mode_mass_top20": float(t20[0])}
        print(f"[jac] crosscheck {name}: abscissa {ev.real[0]:+.5f} /s ({ev.real[0] * fs:+.5f} per frame), "
              f"{int((ev.real > 0).sum())}/10 unstable, lead mode PR {pr[0]:.1f}, spread {sp[0]:.1f} um", flush=True)
    return out


def run_one(run, device="cuda:0", cross=False):
    t0 = time.time()
    L = load_law(run, device)
    print(f"[jac] {run}: {L['kind']}, loaded in {time.time() - t0:.0f} s", flush=True)
    tab, P = regions(L["N"])
    lam = L["lam"]
    th = lambda z: 1.0 - np.tanh(z) ** 2                          # noqa: E731
    leak_sorted = np.sort(-lam)[::-1]
    doc = {"run": run, "kind": L["kind"], "N": L["N"], "frames": L["F"], "frame_s": L["frame_s"], "mu": L["mu"],
           "sd": L["sd"], "blocks": L["names"], "modulation": L["op"].modulation,
           "leak": {"slowest_rate_per_s": float(lam.min()), "slowest_tau_s": float(1.0 / lam.min()),
                    "fastest_tau_s": float(1.0 / lam.max()), "abscissa": float(-lam.min()),
                    "slowest_K": leak_sorted[:K_LR].tolist(),
                    "all_neg_rates_hist": {"edges": np.linspace(-lam.max(), -lam.min(), 41).tolist(),
                                           "counts": np.histogram(-lam, np.linspace(-lam.max(), -lam.min(), 41))[0].tolist()},
                    "tau_quantiles_s_10_50_90": [float(np.quantile(1.0 / lam, q)) for q in (0.1, 0.5, 0.9)]},
           "regions": [r for r, _ in tab], "region_neurons": [int(m.sum()) for _, m in tab],
           "pulse": {p: f for p, f in PULSE}, "K_LR": K_LR, "K_LI": K_LI, "T_impulse_s": T_IR}
    if L["kind"] == "phase":
        doc["reference"] = "per block k: z*_i the mean of z_i over block k's frames; alpha_k the circular mean of alpha(t) over them"
        doc["omega"] = ("Omega_i the mean over block k's frames" if L["op"].modulation != "none" else "no Omega (1)")
        doc["alpha_block_mean_deg"] = np.degrees(L["alpha_k"]).tolist()
        doc["alpha_block_spread_deg"] = np.degrees(L["alpha_sd"]).tolist()
        doc["per_block"] = []
        for k, nm in enumerate(L["names"]):
            s_ = th(L["zblk"][k])
            d, ev, share = analyse_one(L, s_, L["om_blk"][k], L["alpha_k"][k], tab, P, device, li=True)
            # (a) as a check: the recording mean with the same alpha_k, the abscissa only (the 10 with the largest Re)
            Ja = Jac(lam, L["om_blk"][k], w_factors(L, L["alpha_k"][k]), th(L["zbar"]), device)
            ea, _, _ = spectrum(Ja, 10, "LR")
            d.update({"block": nm, "mean_tanh_prime": float(s_.mean()), "abscissa_zbar": float(ea.real.max()),
                      "n_unstable_of_10_zbar": int((ea.real > 0).sum())})
            doc["per_block"].append(d)
            print(f"[jac] {run} block {nm}: abscissa {d['abscissa']:+.5f} /s (z* recording mean: {d['abscissa_zbar']:+.5f}), "
                  f"slow {d['n_slower_than_slowest_leak']}/{K_LR}, unstable {d['n_unstable']}, max|Im| {d['max_abs_im']:.4f}, "
                  f"outside share {d['impulse']['outside_share_all']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        pb = doc["per_block"]
        doc["summary"] = {"abscissa_range": [min(d["abscissa"] for d in pb), max(d["abscissa"] for d in pb)],
                          "n_slow_range": [min(d["n_slower_than_slowest_leak"] for d in pb),
                                           max(d["n_slower_than_slowest_leak"] for d in pb)],
                          "n_unstable_total": sum(d["n_unstable"] for d in pb),
                          "max_abs_im": max(d["max_abs_im"] for d in pb),
                          "outside_share_range": [min(d["impulse"]["outside_share_all"] for d in pb),
                                                  max(d["impulse"]["outside_share_all"] for d in pb)],
                          "block_offdiag_share_range": [min(d["impulse"]["block_offdiag_share_all"] for d in pb),
                                                        max(d["impulse"]["block_offdiag_share_all"] for d in pb)]}
    else:
        doc["reference"] = "z*_i the mean of z_i over the whole recording"
        doc["omega"] = ("Omega_i the mean of Omega_i(t) over every frame" if L["op"].modulation != "none" else "no Omega (1)")
        s_ = th(L["zbar"])
        d, ev, share = analyse_one(L, s_, L["om_bar"], None, tab, P, device)
        d.update({"mean_tanh_prime": float(s_.mean()), "omega_mean": float(L["om_bar"].mean()),
                  "omega_quantiles_10_50_90": [float(np.quantile(L["om_bar"], q)) for q in (0.1, 0.5, 0.9)]})
        doc["J"] = d
        print(f"[jac] {run}: abscissa {d['abscissa']:+.5f} /s against the leak's {-lam.min():+.5f}; slow "
              f"{d['n_slower_than_slowest_leak']}/{K_LR}, unstable {d['n_unstable']}, max|Im| (LI) {d['max_abs_im']:.4f}; "
              f"outside share {d['impulse']['outside_share_all']:.3f}, 6x6 off-diagonal {d['impulse']['block_offdiag_share_all']:.3f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
    if cross:
        doc["crosscheck_rest"] = crosscheck_rest(L, device)
    doc["seconds"] = time.time() - t0
    json.dump(doc, open(os.path.join(EXP, "data", f"jacobian_{run}.json"), "w"), indent=1)
    figure(doc, os.path.join(EXP, "presentation", "figs", f"jacobian_{run}.png"))
    return doc


def figure(doc, path):
    """From the json alone. a: the K eigenvalues of J with the largest Re lambda against the K slowest leaks (W = 0), and
    inset the whole range: every leak -1/tau_i, the K, and the 20 with the largest |Im lambda|; b: the region
    impulse-response matrix, each pulse's (column's) integrated |response| split by row region (24.10: the mean of the
    blocks' shares); c: each mode's Z-Brain participation (24.10: per block, the mean over its K modes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    TS, LS, KS = 16, 12, 11
    per = doc.get("per_block")
    Js = per if per else [doc["J"]]
    labs = [d["block"] for d in per] if per else ["J"]
    fig = plt.figure(figsize=(12, 10), facecolor="black")

    def style(ax, ks=KS):
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=ks)
        for s_ in ax.spines.values():
            s_.set_color("0.5")
    cm = plt.get_cmap("tab10")
    col = (lambda i: "#e6a03c") if not per else (lambda i: cm(i % 10))   # noqa: E731
    ax = fig.add_axes([0.085, 0.53, 0.50, 0.40])
    style(ax)
    lk = np.array(doc["leak"]["slowest_K"])
    ax.plot(lk, np.zeros_like(lk), "|", ms=18, mew=1.6, color="0.8",
            label=(f"leak only, the {len(lk)} slowest" if per else f"leak only ($W = 0$): the {len(lk)} slowest $-1/\\tau_i$"))
    allre, allim = [lk], [np.zeros(1)]
    for i, d in enumerate(Js):
        re_, im_ = np.array(d["eig_re"]), np.array(d["eig_im"])
        allre.append(re_)
        allim.append(im_)
        ax.plot(re_, im_, "o", ms=5, mfc=col(i) if not per else "none", mec=col(i), mew=1.2,
                label=(f"$J$: the {len(re_)} with the largest Re $\\lambda$" if not per else labs[i]))
    sl_ = -doc["leak"]["slowest_rate_per_s"]
    ax.axvline(sl_, color="0.6", lw=0.8, ls="--")
    lo_, hi_ = min(a.min() for a in allre), max(max(a.max() for a in allre), sl_)
    pad = 0.06 * (hi_ - lo_ + 1e-6)
    ax.set_xlim(lo_ - pad, hi_ + 4 * pad)
    if hi_ > 0:
        ax.axvline(0, color="#ff5050", lw=0.8)
    ym = max(float(np.abs(np.concatenate(allim)).max()), 1e-4)
    ax.set_ylim(-(2.4 if per else 1.25) * ym, 2.6 * ym)        # room for the legend (above) and the inset
    ax.set_xlabel("Re $\\lambda$ (1/s)", color="white", fontsize=LS)
    ax.set_ylabel("Im $\\lambda$ (1/s)", color="white", fontsize=LS)
    ax.set_title("a  eigenvalues of $J$ and of the leak alone", color="white", fontsize=TS, loc="left")
    ax.legend(loc="upper left", fontsize=9, facecolor="black", edgecolor="0.4", labelcolor="white",
              ncol=4 if per else 1, markerscale=1.0, handletextpad=0.3, columnspacing=0.7)
    ax.text(sl_, 0.02, " slowest leak", transform=ax.get_xaxis_transform(), color="0.7", fontsize=KS, ha="left", va="bottom")
    # inset: the whole range
    ai = fig.add_axes([0.40, 0.55, 0.175, 0.10] if per else [0.40, 0.70, 0.175, 0.15])   # per block: lower right
    style(ai, 8)
    lam_all = np.array(doc["leak"]["all_neg_rates_hist"]["edges"])
    h_ = np.array(doc["leak"]["all_neg_rates_hist"]["counts"], float)
    ai.bar(0.5 * (lam_all[1:] + lam_all[:-1]), 0.25 * h_ / h_.max(), width=np.diff(lam_all), color="0.6", bottom=0.0)
    for i, d in enumerate(Js):
        ai.plot(d["eig_re"], d["eig_im"], ".", ms=3, color=col(i))
        ai.plot(d["eig_LI_re"], d["eig_LI_im"], "x", ms=4, mew=0.9, color="#4fc3f7")
    ai.axvline(0, color="#ff5050", lw=0.6)
    ai.set_title("the whole range\nx: the 20 with the largest |Im $\\lambda$|\ngrey: every $-1/\\tau_i$", color="white",
                 fontsize=8, loc="left")
    # b: the impulse-response matrix
    Sh = np.mean([np.array(d["impulse"]["abs"]) / np.array(d["impulse"]["abs"]).sum(0, keepdims=True) for d in Js], 0)
    ax2 = fig.add_axes([0.725, 0.53, 0.205, 0.40])
    style(ax2)
    im = ax2.imshow(np.maximum(Sh, 1e-4), cmap="magma", norm=LogNorm(1e-4, 1), aspect="auto")
    rows = ["pretectum", "tectum (PVL)", "dorsal thalamus", "rhombomere 7", "cerebellum", "telencephalon", "rest of the brain"]
    ax2.set_yticks(range(7))
    ax2.set_yticklabels(rows, fontsize=KS)
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(rows[:6], rotation=40, ha="right", fontsize=KS)
    for i in range(7):
        for j in range(6):
            v = 100 * Sh[i, j]
            ax2.text(j, i, f"{v:.0f}" if v >= 9.5 else (f"{v:.1f}" if v >= 0.05 else ""), ha="center", va="center",
                     fontsize=9, color="black" if Sh[i, j] > 0.1 else "white")
    ax2.set_title("b  a pulse's response, %", color="white", fontsize=TS, loc="left")
    cax = fig.add_axes([0.94, 0.53, 0.012, 0.40])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(colors="white", labelsize=9)
    # c: participation
    reg = doc["regions"] + ["outside the table regions"]
    if not per:
        d = Js[0]
        Pm = np.concatenate([np.array(d["mode_region_share"]), np.array(d["mode_share_outside_table"])[:, None]], 1).T
        xl, xt = "mode, by Re $\\lambda$ (1 = the largest)", np.arange(0, Pm.shape[1], 5)
        xtl = [str(x + 1) for x in xt]
    else:
        Pm = np.stack([np.concatenate([np.array(d["mode_region_share"]).mean(0), [np.mean(d["mode_share_outside_table"])]])
                       for d in Js], 1)
        xl, xt, xtl = "stimulus block (the mean over its modes)", np.arange(len(Js)), labs
    ax3 = fig.add_axes([0.20, 0.06, 0.73, 0.31])
    style(ax3)
    im3 = ax3.imshow(Pm, cmap="viridis", aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    ax3.set_yticks(range(len(reg)))
    ax3.set_yticklabels(reg, fontsize=7.5)
    ax3.set_xticks(xt)
    ax3.set_xticklabels(xtl, fontsize=KS if not per else 10)
    ax3.set_xlabel(xl, color="white", fontsize=LS)
    ax3.set_title("c  each mode's share of $|v_i|^2$ by Z-Brain region", color="white", fontsize=TS, loc="left")
    cax3 = fig.add_axes([0.94, 0.06, 0.012, 0.31])
    cb3 = fig.colorbar(im3, cax=cax3)
    cb3.ax.tick_params(colors="white", labelsize=9)
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--landed", action="store_true", help="list batch 25 / 26 runs and whether each has landed")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--crosscheck", action="store_true", help="also the audit's rest-state spectra (crosscheck_rest)")
    ap.add_argument("--arpack", action="store_true", help="scipy's eigs (CPU) in place of the GPU Krylov-Schur")
    ap.add_argument("--figure-only", action="store_true", help="redraw the figures from data/jacobian_<run>.json")
    a_ = ap.parse_args()
    if a_.arpack:
        SOLVER["name"] = "arpack"
    if a_.landed:
        for r_, ok_ in landed().items():
            print(f"{r_:32s} {'landed' if ok_ else 'not landed'}")
    for r_ in a_.runs:
        if a_.figure_only:
            figure(json.load(open(os.path.join(EXP, "data", f"jacobian_{r_}.json"))),
                   os.path.join(EXP, "presentation", "figs", f"jacobian_{r_}.png"))
        else:
            run_one(r_, a_.device, cross=a_.crosscheck)
