"""exp17: how much of a trained neuron-graph law is STIMULUS and how much is NETWORK (Cedric, 2026-10-02: "can we do a
mutual information analysis to figure out the percent of stimulus vs network in the model?").

Local analysis (no cluster). The law (state_diffuse[neuron_graph], cell_ops.StateDiffuseNeuronGraph.step), in the
normalised units z = (x - mu) / sd of the recording (mu, sd: box["norm"]), takes each frame M substeps of

    z_i <- z_i + f_i (-z_i + V_i + n_i + s_i [- g_i a_i])

    s_i(t) = mask_i B_i . u(t)                                 the STIMULUS drive: neuron i's 22 learned input weights
                                                               times the 22 stimulus features of frame t (0 for a
                                                               neuron the `input_mask` drops)
    n_i(t) = Omega_i(t) sum_sets sum_{j->i} W_e phi(z_j)       the NETWORK drive (current synapse; phi = tanh, relu or
             Omega_i(t) sum W_e^2 relu(z_j) (E_j - z_i)          linear); conductance synapse; Omega = 1 without the
                                                               modulation
    -z_i + V_i [- g_i a_i]                                     the neuron's OWN term: its leak toward rest (and its
                                                               adaptation state a_i, when the run has one)

TEACHER-FORCED: every term is evaluated on the RECORDED z(t) of every frame, never on a rollout, so no drift enters.
The drive terms are those of the frame's start (the first substep); the law's full one-frame change p_i(t) (all M
substeps, from the recorded z(t)) is computed too, and checked against the operator's own `step` (--validate).

Per neuron, over the frames of a group (all frames; each of the 9 stimulus conditions; ZAPBench's test frames, label
2, when the run used that split), with t and t+1 in the same condition:

  VARIANCE SHARE OF THE DRIVE   var(s), var(n), 2 cov(s, n) over var(s + n): which term moves the law's drive.
                                Brain-wide = the sums over neurons of each part over the sum of var(s + n) (one scale
                                for every neuron: z is normalised by ONE brain-wide mean and sd).
  INFORMATION ABOUT THE CHANGE  the target y = the RECORDED one-frame change z(t+1) - z(t) (and, separately, the law's
                                own predicted change p), CONDITIONED ON THE NEURON'S OWN STATE (z(t), and a(t) when
                                present): the leak alone predicts much of the change (mean reversion) and is neither
                                stimulus nor network, so it is partialled out of y, s and n before any MI is taken.
                                Gaussian MI in bits, from the partial correlations rho:
                                    I(y; s) = -1/2 log2(1 - rho_ys^2)       I(y; n) likewise
                                    I(y; s, n) = -1/2 log2(1 - R^2)         R^2 the multiple correlation of y on (s, n)
                                Split by the MINIMUM-MUTUAL-INFORMATION partial information decomposition (Barrett
                                2015, the Gaussian MMI PID): redundancy = min(I(y;s), I(y;n)),
                                unique_s = I(y;s) - red, unique_n = I(y;n) - red, synergy = I(y;s,n) - max(I(y;s),
                                I(y;n)); the four sum to I(y; s, n). Per neuron one of the two uniques is 0 by
                                construction (MMI's known property); brain-wide sums are not.
                                PERCENT STIMULUS = sum_i unique_s / sum_i I(y; s, n), the same for network, redundant,
                                synergy. The per-neuron STIMULUS SHARE (map, histogram) = I(y;s) / (I(y;s) + I(y;n)).
  CHANCE LEVEL (NULL)           a finite recording gives every Gaussian MI a positive bias (~ predictors / (2 n_eff
                                ln 2) bits, n_eff the frames' effective count, small for a 120-frame test condition).
                                The NULL drives s0, n0 are the same terms taken at a frame SHIFTED within the same
                                condition by a lag chosen in [L/4, 3L/4] of the condition's length L to minimise the
                                stimulus features' circular autocorrelation (the stimuli repeat: a half-length shift
                                lands on the same trial phase, r 0.98 for flash), so they keep each term's own time
                                structure and lose its alignment with the change. DEBIASED per neuron:
                                I_s' = max(I_s - I(y; s0), 0), I_n' likewise, I_sn' = max(I_sn - I(y; s0) - I(y; n0),
                                max(I_s', I_n')); the PID split above is then taken of the debiased values. The
                                headline percents are the debiased ones; the raw ones are kept beside them.
  NON-GAUSSIAN CHECK            on 2,000 random neurons, all frames: I(y; s), I(y; n), I(y; s, n) and the two nulls by
                                the Kraskov-Stoegbauer-Grassberger k-NN estimator (algorithm 1, k = 4, Chebyshev), on
                                the same own-state-partialled series, against the Gaussian values of the same neurons.

Writes experiments/exp17_zapbench_graphcast/data/stim_vs_network_<run>.json (+ _per_neuron.npz) and
data/figs/stim_vs_network_<run>.png.

    python tools/exp17_stim_vs_network.py zap_zs_ng_base [--validate] [--device cuda:0] [--no-ksg]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
LN2 = math.log(2.0)


# ------------------------------------------------------------------------------------------------ the trained law
def setup(name, device):
    """The run's operator instance with its learned values, the per-neuron blocks it reads, and the recording box."""
    import plexus.trainer as TRN
    from plexus import engine
    spec = TRN.load(name)
    engine.quiet(True)
    out = TRN.out_dir(spec, None)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = TRN.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = TRN._trace_setup(spec, device)
    sim = TRN._model(spec, train=False, n_frames=1)
    cap = {}

    def seeded(H):
        learn.inject(H)

    def ready(H):
        learn.ready(H)
        for op in H.operators:
            if getattr(op, "NORM_FROM_REFERENCE", False):
                op.norm = box["norm"]
        cap["H"] = H

    with torch.no_grad():
        engine.run(sim, device=device, progress=False, on_seeded=seeded, on_ready=ready)
    H = cap["H"]
    ops = [o for o in H.operators if type(o).__name__ == "StateDiffuseNeuronGraph"]
    if len(ops) != 1:
        raise ValueError(f"{name}: {len(ops)} neuron-graph operators (this tool reads exactly one)")
    op = ops[0]
    lvl = H.level(op.at)
    nb = {}
    for k, blk in op.blocks.items():
        if k == "input" and not op.forcing_dim:
            continue
        c0, c1 = lvl.state_schema[blk]
        nb[k] = lvl.state[:, c0:c1].detach().clone()
    if op.adapt:
        for k, blk in op.adapt_blocks.items():
            c0, c1 = lvl.state_schema[blk]
            nb[k] = lvl.state[:, c0:c1].detach().clone()
    if op.modulation == "hash":
        nb["_omega_fs"] = op._space_features().detach()
    return spec, op, nb, box, ck.get("it"), learn


class Law:
    """The law's terms on a chunk of C recorded frames at once ([C, N] tensors, z units), reproducing `step`."""

    def __init__(self, op, nb, S, T):
        import torch.nn.functional as Fnn
        self.op, self.nb, self.S, self.T = op, nb, S, T
        self.V = nb["rest"][:, 0][None]
        self.fz = op._frac(op._rate(nb["tau"]))[:, 0][None]                     # fraction of the way per substep
        self.B = nb["input"] if op.forcing_dim else None                         # [N, 22]
        self.mask = op.input_mask[:, 0][None].to(self.V.device) if (op.input_mask is not None and op.forcing_dim) else None
        self.E = nb["reversal"][:, 0] if op.synapse == "conductance" else None
        if op.adapt:
            self.ga = Fnn.softplus(nb["adapt_gain"])[:, 0][None]
            self.fa = op._frac(op._rate(nb["adapt_rate"]))[:, 0][None]

    def msg(self, z):
        """[C, N] the summed message into each neuron (before Omega)."""
        op = self.op
        out = torch.zeros_like(z)
        if op.synapse == "conductance":
            r = torch.relu(z)
            q1, q2 = r * self.E[None], r
            g = torch.zeros_like(z)
            for s in op.EDGE_SETS:
                snd, rcv = op._E[s]
                if snd.numel():
                    w2 = getattr(op, f"W_{s}") ** 2
                    out.index_add_(1, rcv, w2[None] * q1[:, snd])
                    g.index_add_(1, rcv, w2[None] * q2[:, snd])
            return out - z * g
        act = op._act(z)
        for s in op.EDGE_SETS:
            snd, rcv = op._E[s]
            if snd.numel():
                out.index_add_(1, rcv, getattr(op, f"W_{s}")[None] * act[:, snd])
        return out

    def omega(self, frames):
        op = self.op
        if op.modulation == "none":
            return None
        oms = []
        for t in frames:
            op.frame, op.n_frames_ref = int(t), int(self.T)
            oms.append(op._omega(self.nb.get("_omega_fs"))[:, 0])
        return torch.stack(oms, 0)

    def drive(self, frames):
        if self.B is None:
            return torch.zeros(len(frames), self.V.shape[1], device=self.V.device)
        d = self.S[frames] @ self.B.T                                             # [C, N]
        return d * self.mask if self.mask is not None else d

    def drives(self, Z, frames):
        """s, n at the frame's start only (no substeps): the null terms at shifted frames."""
        om = self.omega(frames)
        n = self.msg(Z)
        return self.drive(frames), (om * n if om is not None else n)

    def terms(self, Z, frames, A=None):
        """s, n at the frame's start; p the full one-frame change (M substeps); da the adaptation state's change."""
        s = self.drive(frames)
        om = self.omega(frames)
        z, ad = Z, A
        n0 = None
        for k in range(self.op.substeps):
            agg = self.msg(z)
            if om is not None:
                agg = om * agg
            if k == 0:
                n0 = agg
            if self.op.adapt:
                z, ad = (z + self.fz * (-z + self.V + agg + s - self.ga * ad), ad + self.fa * (z - ad))
            else:
                z = z + self.fz * (-z + self.V + agg + s)
        return s, n0, z - Z, (ad - A if self.op.adapt else None)


def validate(op, nb, law, X, S, norm, frames, A=None):
    """max |law's own step - this reproduction| over a few frames, dF/F units, and the step's typical size."""
    mu, sd, _ = norm
    worst, size = 0.0, 0.0
    for t in frames:
        nbt = dict(nb)
        if op.adapt:
            nbt["adapt"] = A[t][:, None]
        if op.modulation != "none":
            op.frame, op.n_frames_ref = int(t), int(X.shape[0])
        u = S[t].reshape(-1, 1) if op.forcing_dim else None
        ref = op.step(X[t][:, None], None, None, u, nb=nbt)
        ref = ref[0] if isinstance(ref, tuple) else ref
        Z = ((X[t] - mu) / sd)[None]
        _, _, p, _ = law.terms(Z, torch.as_tensor([t], device=X.device), None if A is None else A[t][None])
        worst = max(worst, float((ref[:, 0] - sd * p[0]).abs().max()))
        size = max(size, float(ref.abs().max()))
    return {"frames": [int(t) for t in frames], "max_abs_diff_dff": worst, "max_abs_step_dff": size}


def validate_pipeline(spec, learn, box, law, frames, device):
    """The trainer's own one-frame forecast (`_trace_rollout`, horizon 1, a fresh model) against X(t) + sd p(t) of
    this reproduction: checks the learned blocks, the normalisation and the stimulus frame end to end."""
    import plexus.trainer as TRN
    if TRN._warmup(spec) or law.op.adapt:
        return {"pipeline_check": "skipped (warm-up or adaptation: the rollout's hidden state differs)"}
    X, (mu, sd, _) = box["X"], box["norm"]
    sim = TRN._trace_sim(spec, False, 1)
    worst = 0.0
    for t in frames:
        with torch.no_grad():
            f = TRN._trace_rollout(sim, learn, spec, box, int(t), 1, device, False)[0]
            _, _, p, _ = law.terms(((X[t] - mu) / sd)[None], torch.as_tensor([t], device=X.device))
        worst = max(worst, float((f - (X[t] + sd * p[0])).abs().max()))
    return {"pipeline_max_abs_diff_dff": worst}



# ------------------------------------------------------------------------------------------------ the statistics
VARS = ("y_rec", "y_law", "s", "n", "s0", "n0")      # after the own-state variables: z (and a)


def groups_of(rec, lab, T):
    """Frame t -> the groups it belongs to (t and t+1 in the same condition): 'all', each condition, and the test
    frames (label 2 at t and t+1) when the run used ZAPBench's split."""
    cond = np.asarray(rec["condition"])
    names = list(rec["names"])
    gnames = ["all"] + names
    if lab is not None:
        gnames += ["test"] + [f"test {n}" for n in names]
    G = np.zeros((len(gnames), T), bool)
    ok = np.zeros(T, bool)
    ok[:-1] = cond[:-1] == cond[1:]
    G[0] = ok
    for c in range(len(names)):
        G[1 + c] = ok & (cond == c)
    if lab is not None:
        te = np.zeros(T, bool)
        te[:-1] = (lab[:-1] == 2) & (lab[1:] == 2)
        G[1 + len(names)] = ok & te
        for c in range(len(names)):
            G[2 + len(names) + c] = ok & te & (cond == c)
    return gnames, G


def null_map(rec, T):
    """Frame t -> the frame its NULL terms are taken at: shifted circularly within t's condition by the lag in
    [L/4, 3L/4] that minimises the mean |circular autocorrelation| of the condition's varying stimulus features."""
    S = np.asarray(rec["stimulus"], np.float64)
    off = np.asarray(rec["offsets"])
    nf = np.arange(T)
    lags = {}
    for c, nm in enumerate(rec["names"]):
        a, b = int(off[c]), int(off[c + 1])
        L = b - a
        seg = S[a:b]
        v = seg[:, seg.std(0) > 1e-6]
        cand = np.arange(L // 4, 3 * L // 4 + 1)
        if v.shape[1]:
            v = (v - v.mean(0)) / v.std(0)
            F = np.fft.rfft(v, axis=0)
            ac = np.fft.irfft(F * np.conj(F), n=L, axis=0) / L                    # circular autocorrelation [L, f]
            score = np.abs(ac[cand]).mean(1)
            lag = int(cand[np.argmin(score)])
            lags[nm] = {"lag_frames": lag, "length_frames": L, "mean_abs_autocorr_at_lag": float(score.min())}
        else:
            lag = L // 2
            lags[nm] = {"lag_frames": lag, "length_frames": L, "mean_abs_autocorr_at_lag": None}
        nf[a:b] = a + (np.arange(L) + lag) % L
    return nf, lags


def gauss_mi(C, n_own):
    """Per neuron, from the covariance C [N, k, k] of (own state..., VARS): the drive's variance parts and, for each
    target (recorded change, law's change), the Gaussian MIs (bits) conditioned on the own state, their nulls, and
    the MMI split raw and debiased."""
    o = n_own
    ix = {v: o + i for i, v in enumerate(VARS)}
    out = {}
    vs, vn, cv = C[:, ix["s"], ix["s"]], C[:, ix["n"], ix["n"]], C[:, ix["s"], ix["n"]]
    out["var_s"], out["var_n"], out["cov2"] = vs, vn, 2 * cv
    # partial covariance of VARS given the own state (Schur complement)
    Coo = C[:, :o, :o] + 1e-12 * torch.eye(o, device=C.device, dtype=C.dtype)[None]
    Cor = C[:, :o, o:]
    P = C[:, o:, o:] - Cor.transpose(1, 2) @ torch.linalg.solve(Coo, Cor)
    p = {v: i for i, v in enumerate(VARS)}
    eps = 1e-14
    cap = 1 - 1e-12
    I = lambda r2: -0.5 * torch.log1p(-r2.clamp(0, cap)) / LN2                  # noqa: E731

    def r2(j, q):
        yy, qq, yq = P[:, j, j], P[:, q, q], P[:, j, q]
        return torch.where(qq > eps, yq ** 2 / (yy * qq).clamp(min=eps), torch.zeros_like(yy))

    ss, nn, sn = P[:, p["s"], p["s"]], P[:, p["n"], p["n"]], P[:, p["s"], p["n"]]
    for key in ("rec", "law"):
        j = p[f"y_{key}"]
        yy, ys, yn = P[:, j, j], P[:, j, p["s"]], P[:, j, p["n"]]
        r2s, r2n = r2(j, p["s"]), r2(j, p["n"])
        det = ss * nn - sn ** 2
        full = (ss > eps) & (nn > eps) & (det > 1e-9 * ss * nn)
        r2j = (nn * ys ** 2 - 2 * sn * ys * yn + ss * yn ** 2) / (yy * det).clamp(min=eps)
        r2j = torch.where(full, r2j, torch.maximum(r2s, r2n))
        Is, In = I(r2s), I(r2n)
        Ij = torch.maximum(I(r2j), torch.maximum(Is, In))
        Is0, In0 = I(r2(j, p["s0"])), I(r2(j, p["n0"]))
        m = {"I_s": Is, "I_n": In, "I_sn": Ij, "I_s_null": Is0, "I_n_null": In0}
        for tag, (a_, b_, c_) in (("raw", (Is, In, Ij)),
                                  ("debiased", ((Is - Is0).clamp(min=0), (In - In0).clamp(min=0), None))):
            if c_ is None:
                c_ = torch.maximum(Ij - Is0 - In0, torch.maximum(a_, b_))
            red = torch.minimum(a_, b_)
            m[tag] = {"I_s": a_, "I_n": b_, "I_sn": c_, "unique_s": a_ - red, "unique_n": b_ - red,
                      "redundant": red, "synergy": c_ - torch.maximum(a_, b_)}
        # the own state's information about the change (unconditioned): how much the leak alone predicts
        Coy = C[:, :o, o + j:o + j + 1]
        r2o = (Coy.transpose(1, 2) @ torch.linalg.solve(Coo, Coy))[:, 0, 0] / C[:, o + j, o + j].clamp(min=eps)
        m["I_own"] = I(r2o)
        out[key] = m
    return out


def summarise(C, mean, n_own, n_frames):
    g = gauss_mi(C, n_own)
    tot = lambda x: float(x.sum())                                                # noqa: E731
    med = lambda x: float(x.median()) if x.numel() else float("nan")              # noqa: E731
    V = g["var_s"] + g["var_n"] + g["cov2"]
    vok = V > 1e-14
    res = {"n_frames": int(n_frames),
           "drive_variance": {
               "brainwide_pct_stimulus": 100 * tot(g["var_s"]) / max(tot(V), 1e-30),
               "brainwide_pct_network": 100 * tot(g["var_n"]) / max(tot(V), 1e-30),
               "brainwide_pct_2cov": 100 * tot(g["cov2"]) / max(tot(V), 1e-30),
               "median_pct_stimulus": 100 * med((g["var_s"] / V.clamp(min=1e-30))[vok]),
               "median_pct_network": 100 * med((g["var_n"] / V.clamp(min=1e-30))[vok]),
               "median_pct_2cov": 100 * med((g["cov2"] / V.clamp(min=1e-30))[vok]),
               "median_var_s_z2": med(g["var_s"]), "median_var_n_z2": med(g["var_n"])}}
    o = n_own
    for key in ("rec", "law"):
        m = g[key]
        r = {}
        for tag in ("debiased", "raw"):
            d = m[tag]
            J = tot(d["I_sn"])
            r[tag] = {f"brainwide_pct_{q}": 100 * tot(d[q]) / max(J, 1e-30)
                      for q in ("unique_s", "unique_n", "redundant", "synergy")}
            r[tag]["brainwide_pct_stimulus_of_single_MIs"] = 100 * tot(d["I_s"]) / max(tot(d["I_s"]) + tot(d["I_n"]), 1e-30)
            r[tag]["mean_bits_per_neuron"] = {q: float(d[q].mean()) for q in ("I_s", "I_n", "I_sn")}
            sh = d["I_s"] / (d["I_s"] + d["I_n"])
            r[tag]["median_neuron_stimulus_share"] = med(sh[torch.isfinite(sh)])
        r["mean_bits_per_neuron"] = {q: float(m[q].mean()) for q in ("I_s", "I_n", "I_sn", "I_s_null", "I_n_null", "I_own")}
        r["null_over_measured"] = {"stimulus": tot(m["I_s_null"]) / max(tot(m["I_s"]), 1e-30),
                                   "network": tot(m["I_n_null"]) / max(tot(m["I_n"]), 1e-30)}
        res[f"info_{key}"] = r
    # the law's one-frame forecast against the recorded change: 1 - E[(dz - p)^2] / var(dz), per neuron
    j, l = o, o + 1
    mse = C[:, j, j] + C[:, l, l] - 2 * C[:, j, l] + (mean[:, j] - mean[:, l]) ** 2
    r2 = 1 - mse / C[:, j, j].clamp(min=1e-30)
    res["law_one_step"] = {"median_r2_of_recorded_change": med(r2),
                           "brainwide_r2_of_recorded_change": 1 - tot(mse) / tot(C[:, j, j])}
    return res, g


# ------------------------------------------------------------------------------------------------ KSG check
def ksg_np(x, y, k=4):
    """Kraskov-Stoegbauer-Grassberger estimator 1 (Kraskov et al. 2004, Phys Rev E 69:066138), bits, Chebyshev."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    n = len(x)
    j = np.hstack([x, y])
    d, _ = cKDTree(j).query(j, k=k + 1, p=np.inf)
    eps = np.nextafter(d[:, -1], 0)                                               # strictly inside the k-th distance
    nx = cKDTree(x).query_ball_point(x, eps, p=np.inf, return_length=True) - 1
    ny = cKDTree(y).query_ball_point(y, eps, p=np.inf, return_length=True) - 1
    return max(float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))), 0.0) / LN2


def _ksg_one(args):
    """One neuron: [n, 6] own-state-partialled (y_rec, y_law, s, n, s0, n0) -> Gaussian and KSG MIs, bits."""
    R, seed = args
    rng = np.random.default_rng(seed)
    y, s, n, s0, n0 = R[:, 0], R[:, 2], R[:, 3], R[:, 4], R[:, 5]
    sd = [v.std() for v in (y, s, n, s0, n0)]
    if sd[0] < 1e-12 or sd[2] < 1e-12:
        return None
    has_s = sd[1] > 1e-12 and sd[3] > 1e-12
    col = lambda v, q: (v / q + 1e-6 * rng.standard_normal(len(v)))[:, None]     # noqa: E731
    Y, Nn, N0 = col(y, sd[0]), col(n, sd[2]), col(n0, sd[4])
    gI = lambda r2: -0.5 * math.log(max(1 - r2, 1e-12)) / LN2                    # noqa: E731
    c2 = lambda a, b: float(np.corrcoef(a, b)[0, 1] ** 2)                         # noqa: E731
    out = {"g_n": gI(c2(y, n)), "g_n0": gI(c2(y, n0)), "k_n": ksg_np(Y, Nn), "k_n0": ksg_np(Y, N0)}
    if has_s:
        Ss, S0 = col(s, sd[1]), col(s0, sd[3])
        Cm = np.cov(np.stack([y, s, n]))
        sxy = Cm[0, 1:]
        r2j = float(sxy @ np.linalg.pinv(Cm[1:, 1:]) @ sxy) / Cm[0, 0]
        out.update({"g_s": gI(c2(y, s)), "g_s0": gI(c2(y, s0)), "g_sn": gI(r2j),
                    "k_s": ksg_np(Y, Ss), "k_s0": ksg_np(Y, S0), "k_sn": ksg_np(Y, np.hstack([Ss, Nn]))})
    else:
        out.update({"g_s": 0.0, "g_s0": 0.0, "g_sn": out["g_n"], "k_s": 0.0, "k_s0": 0.0, "k_sn": out["k_n"]})
    return out


def ksg_check(series, mask, n_own, workers=32):
    """series [T, k, M] of the subsample (float32): the group's frames, own state partialled out of every other
    variable (least squares per neuron), then Gaussian and KSG MIs per neuron in a process pool."""
    from multiprocessing import get_context
    V = series[mask].astype(np.float64)                                           # [n, k, M]
    V = V - V.mean(0)
    O, R = V[:, :n_own], V[:, n_own:]
    jobs = []
    for i in range(V.shape[2]):
        beta, *_ = np.linalg.lstsq(O[:, :, i], R[:, :, i], rcond=None)
        jobs.append((R[:, :, i] - O[:, :, i] @ beta, i))
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        res = pool.map(_ksg_one, jobs, chunksize=8)
    print(f"[stim_vs_network] KSG on {len(jobs)} neurons: {time.time() - t0:.0f} s", flush=True)
    keys = ("g_s", "g_n", "g_sn", "g_s0", "g_n0", "k_s", "k_n", "k_sn", "k_s0", "k_n0")
    return {k: np.asarray([r[k] if r else np.nan for r in res], float) for k in keys}


def ksg_summary(kc):
    from scipy.stats import spearmanr
    ok = np.isfinite(kc["g_n"])
    res = {"n_neurons": int(ok.sum()),
           "estimator": "KSG algorithm 1, k = 4, Chebyshev, 1e-6 sd jitter; same own-state-partialled series as Gaussian"}
    for q in ("s", "n", "sn", "s0", "n0"):
        g, k = kc[f"g_{q}"][ok], kc[f"k_{q}"][ok]
        res[f"I_{q}"] = {"gauss_mean_bits": float(g.mean()), "ksg_mean_bits": float(k.mean()),
                         "spearman_gauss_vs_ksg_across_neurons": float(spearmanr(g, k).correlation) if g.std() > 0 else None}
    for est, p in (("gauss", "g"), ("ksg", "k")):
        for tag in ("raw", "debiased"):
            Is, In = kc[f"{p}_s"][ok], kc[f"{p}_n"][ok]
            Ij = np.maximum(kc[f"{p}_sn"][ok], np.maximum(Is, In))
            if tag == "debiased":
                s0, n0 = kc[f"{p}_s0"][ok], kc[f"{p}_n0"][ok]
                Is, In = np.maximum(Is - s0, 0), np.maximum(In - n0, 0)
                Ij = np.maximum(Ij - s0 - n0, np.maximum(Is, In))
            red = np.minimum(Is, In)
            J = Ij.sum()
            res[f"{est}_{tag}_pct"] = {"unique_s": 100 * (Is - red).sum() / J, "unique_n": 100 * (In - red).sum() / J,
                                       "redundant": 100 * red.sum() / J,
                                       "synergy": 100 * (Ij - np.maximum(Is, In)).sum() / J,
                                       "stimulus_of_single_MIs": 100 * Is.sum() / max(Is.sum() + In.sum(), 1e-30)}
    return res


# ------------------------------------------------------------------------------------------------ main
def brain_view(pos):
    """Positions (um) as the run movies draw them: horizontal, head left (the convention of tools/exp17_slides.py)."""
    pos = np.asarray(pos, dtype=np.float64)
    if np.ptp(pos[:, 0]) > np.ptp(pos[:, 1]):
        pos = np.stack([pos[:, 1], -pos[:, 0], pos[:, 2]], 1)
    return np.stack([-pos[:, 1], pos[:, 0], pos[:, 2]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="zap_zs_ng_base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--validate", action="store_true", help="only check the reproduction against the operator's step")
    ap.add_argument("--no-ksg", action="store_true")
    ap.add_argument("--n-ksg", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    spec, op, nb, box, it, learn = setup(a.run, dev)
    X, S, rec, lab, norm = box["X"], box["S"], box["rec"], box["lab"], box["norm"]
    T, N = X.shape
    mu, sd, _ = norm
    if S is None:
        S = torch.zeros(T, max(op.forcing_dim, 1), device=dev)
    law = Law(op, nb, S, T)
    print(f"[stim_vs_network] {a.run}: {N} neurons x {T} frames; synapse {op.synapse}, activation {op.activation}, "
          f"modulation {op.modulation}, integrator {op.integrator}, adaptation {bool(op.adapt)}, "
          f"input mask {op.input_mask is not None}; set-up {time.time() - t0:.0f} s", flush=True)

    A = None
    with torch.no_grad():
        if op.adapt:                     # the adaptation state, teacher-forced: a(0) = z(0), then the law's update
            A = torch.empty(T, N, device=dev)
            A[0] = (X[0] - mu) / sd
            for t in range(T - 1):
                Z = ((X[t] - mu) / sd)[None]
                _, _, _, da = law.terms(Z, torch.as_tensor([t], device=dev), A[t][None])
                A[t + 1] = A[t] + da[0]
        vframes = [int(T * q) for q in (0.1, 0.35, 0.6, 0.85)]
        val = validate(op, nb, law, X, S, norm, vframes, A)
        val.update(validate_pipeline(spec, learn, box, law, vframes, dev))
    print(f"[stim_vs_network] validation (operator step vs this reproduction, dF/F): {val}", flush=True)
    if a.validate:
        return

    gnames, G = groups_of(rec, lab, T)
    nullf, lags = null_map(rec, T)
    nullf_t = torch.as_tensor(nullf, device=dev)
    n_own = 2 if op.adapt else 1
    k = n_own + len(VARS)
    nG = len(gnames)
    S1 = torch.zeros(nG, k, N, device=dev, dtype=torch.float64)
    S2 = torch.zeros(nG, k, k, N, device=dev, dtype=torch.float64)
    cnt = G.sum(1)
    sub = np.sort(rng.choice(N, min(a.n_ksg, N), replace=False))
    sub_t = torch.as_tensor(sub, device=dev)
    series = np.zeros((T, k, len(sub)), np.float32)
    Gt = torch.as_tensor(G, device=dev)
    with torch.no_grad():
        for c0 in range(0, T - 1, a.chunk):
            fr = torch.arange(c0, min(c0 + a.chunk, T - 1), device=dev)
            Z = (X[fr] - mu) / sd
            Z1 = (X[fr + 1] - mu) / sd
            s, n, p, _ = law.terms(Z, fr, None if A is None else A[fr])
            frn = nullf_t[fr]
            s0, n0 = law.drives((X[frn] - mu) / sd, frn)
            own = [Z] + ([A[fr]] if op.adapt else [])
            Vv = torch.stack(own + [Z1 - Z, p, s, n, s0, n0], 1)                 # [C, k, N]
            series[fr.cpu().numpy()] = Vv[:, :, sub_t].cpu().numpy()
            Vd = Vv.double()
            for g in range(nG):
                m = Gt[g, fr]
                if not bool(m.any()):
                    continue
                W = Vd[m]
                S1[g] += W.sum(0)
                S2[g] += torch.einsum("tan,tbn->abn", W, W)
            if (c0 // a.chunk) % 50 == 0:
                print(f"[stim_vs_network] frame {c0}/{T}  {time.time() - t0:.0f} s", flush=True)

    drv = law.drive(torch.arange(T, device=dev))
    res = {"run": a.run, "checkpoint_iteration": it, "n_neurons": int(N), "n_frames": int(T),
           "recording": spec["task"]["reference"]["trace_recording"], "split": box["split"],
           "law": {"synapse": op.synapse, "activation": op.activation, "modulation": op.modulation,
                   "integrator": op.integrator, "substeps": op.substeps, "adaptation": bool(op.adapt),
                   "input_mask": op.input_mask is not None,
                   "neurons_with_time_varying_stimulus_drive": int((drv.std(0) > 1e-9).sum())},
           "norm_mu_sd_of_dff": [mu, sd],
           "validation": val,
           "null_lags": lags,
           "method": {
               "units": "z = (dF/F - mu) / sd, mu and sd ONE brain-wide pair (the run's normalisation)",
               "s": "stimulus drive s_i(t) = mask_i * B_i . u(t), 22 learned weights times the 22 stimulus features",
               "n": "network drive n_i(t) = Omega_i(t) * sum over edges j->i of W phi(z_j) (current) or W^2 relu(z_j)"
                    "(E_j - z_i) (conductance), at the frame's start (first substep), on the RECORDED z(t)",
               "target_rec": "the recorded one-frame change z(t+1) - z(t)",
               "target_law": "the law's own one-frame change from the recorded z(t), all substeps (teacher-forced)",
               "conditioning": "target, s and n are partialled on the neuron's own state (z(t), and a(t) if adapting): "
                               "the leak -z + V is neither stimulus nor network",
               "MI": "Gaussian, bits: -1/2 log2(1 - rho^2) of the partial correlation; joint from the multiple R^2",
               "PID": "minimum mutual information (Barrett 2015): redundant = min(I_s, I_n); unique_s = I_s - red; "
                      "unique_n = I_n - red; synergy = I_sn - max(I_s, I_n)",
               "null": "s0, n0 = the same terms at a frame shifted within the condition (null_lags); debiased "
                       "I_s' = max(I_s - I_s0, 0), I_n' likewise, I_sn' = max(I_sn - I_s0 - I_n0, max(I_s', I_n'))",
               "percent": "brain-wide = sum over neurons of each part / sum over neurons of I(target; s, n)",
               "stimulus_share_per_neuron": "I(target; s) / (I(target; s) + I(target; n)), raw (all frames)",
               "frames": "t and t+1 in the same condition; 'test' = ZAPBench label 2 at both"},
           "groups": {}}
    per_neuron = {}
    for g, nm in enumerate(gnames):
        if cnt[g] < 10:
            continue
        mean = (S1[g] / cnt[g]).T                                                 # [N, k]
        C = (S2[g] / cnt[g]).permute(2, 0, 1) - mean[:, :, None] * mean[:, None, :]
        r, gm = summarise(C, mean, n_own, cnt[g])
        res["groups"][nm] = r
        if nm in ("all", "test"):
            d = gm["rec"]["raw"]
            V = gm["var_s"] + gm["var_n"] + gm["cov2"]
            per_neuron[nm] = {"info_share_stimulus": (d["I_s"] / (d["I_s"] + d["I_n"])).float().cpu().numpy(),
                              "var_share_stimulus": (gm["var_s"] / V.clamp(min=1e-30)).float().cpu().numpy(),
                              "I_s_bits": d["I_s"].float().cpu().numpy(), "I_n_bits": d["I_n"].float().cpu().numpy(),
                              "I_sn_bits": d["I_sn"].float().cpu().numpy(),
                              "I_s_null_bits": gm["rec"]["I_s_null"].float().cpu().numpy(),
                              "I_n_null_bits": gm["rec"]["I_n_null"].float().cpu().numpy()}
        ir = r["info_rec"]["debiased"]
        print(f"[stim_vs_network] {nm:>18s} ({cnt[g]} frames): debiased info about the recorded change: unique "
              f"stimulus {ir['brainwide_pct_unique_s']:.1f} %, unique network {ir['brainwide_pct_unique_n']:.1f} %, "
              f"redundant {ir['brainwide_pct_redundant']:.1f} %, synergy {ir['brainwide_pct_synergy']:.1f} %; "
              f"null/measured stimulus {r['info_rec']['null_over_measured']['stimulus']:.2f}, network "
              f"{r['info_rec']['null_over_measured']['network']:.2f}; drive variance from the stimulus "
              f"{r['drive_variance']['brainwide_pct_stimulus']:.1f} %", flush=True)
    del S2

    if not a.no_ksg:
        kc = ksg_check(series, G[0], n_own, a.workers)
        res["ksg_check_all_frames"] = ksg_summary(kc)
        print(f"[stim_vs_network] KSG check: {json.dumps(res['ksg_check_all_frames'])}", flush=True)

    os.makedirs(os.path.join(EXP, "data", "figs"), exist_ok=True)
    np.savez_compressed(os.path.join(EXP, "data", f"stim_vs_network_{a.run}_per_neuron.npz"),
                        **{f"{g}_{k}": v for g, d in per_neuron.items() for k, v in d.items()})
    figure(res, per_neuron, rec, os.path.join(EXP, "data", "figs", f"stim_vs_network_{a.run}.png"))
    json.dump(res, open(os.path.join(EXP, "data", f"stim_vs_network_{a.run}.json"), "w"), indent=1)
    print(f"[stim_vs_network] done in {time.time() - t0:.0f} s", flush=True)


def figure(res, per_neuron, rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(17, 8.6), facecolor="black")
    RED, BLUE, GREY, LGREY = "#e8483b", "#4d8fe8", "#8c8c8c", "#d0d0d0"

    def tidy(ax):
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

    # a: brain-wide information split per group (debiased)
    a = fig.add_axes([0.09, 0.17, 0.29, 0.70])
    rows = [g for g in ["all"] + list(rec["names"]) + (["test"] if "test" in res["groups"] else []) if g in res["groups"]]
    parts = [("brainwide_pct_unique_s", "unique stimulus", RED), ("brainwide_pct_unique_n", "unique network", BLUE),
             ("brainwide_pct_redundant", "redundant (either drive)", GREY),
             ("brainwide_pct_synergy", "synergy (only both together)", LGREY)]
    for r_i, g in enumerate(rows):
        left = 0.0
        d = res["groups"][g]["info_rec"]["debiased"]
        for key, lab, col in parts:
            v = d[key]
            a.barh(r_i, v, left=left, color=col, height=0.72, edgecolor="black", linewidth=1.5,
                   label=lab if r_i == 0 else None)
            left += v
        vs = res["groups"][g]["drive_variance"]["brainwide_pct_stimulus"]
        a.plot([vs], [r_i], marker="D", color="white", ms=6, mec="black", ls="none",
               label="stimulus % of the drive's variance" if r_i == 0 else None)
        a.text(102, r_i, f"{d['brainwide_pct_unique_s']:.0f} / {d['brainwide_pct_unique_n']:.0f}", va="center",
               fontsize=8, color="0.8")
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([f"{g} ({res['groups'][g]['n_frames']} frames)" for g in rows], fontsize=8)
    a.invert_yaxis()
    a.set_xlim(0, 115)
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_xlabel("% of the information the two drives carry about each neuron's recorded one-frame change,\n"
                 "given its own present value, summed over neurons (chance level subtracted)", fontsize=8)
    a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3)
    tidy(a)
    fig.text(0.01, 0.95, "a   stimulus vs network, brain-wide, per condition (Gaussian MI, minimum-MI split)\n"
                         "     numbers right: unique stimulus % / unique network %", fontsize=10)

    # b: per-neuron stimulus share
    b = fig.add_axes([0.47, 0.20, 0.18, 0.67])
    pn = per_neuron["all"]
    bins = np.linspace(0, 1, 41)
    sh = pn["info_share_stimulus"]
    b.hist(sh[np.isfinite(sh)], bins=bins, color=RED, alpha=0.85, label="share of the information")
    b.hist(np.clip(pn["var_share_stimulus"], 0, 1), bins=bins, histtype="step", color="white", lw=1.4,
           label="share of the drive's variance")
    b.set_yscale("log")
    b.set_xlabel("stimulus share per neuron, all frames\n0 = all network, 1 = all stimulus", fontsize=8)
    b.set_ylabel("neurons", fontsize=8)
    b.legend(frameon=False, fontsize=7, loc="upper center")
    tidy(b)
    fig.text(0.44, 0.95, "b   per-neuron stimulus share", fontsize=10)

    # c: map
    P = brain_view(rec["pos_um"])
    order = np.argsort(P[:, 2])
    cm = LinearSegmentedColormap.from_list("netstim", [BLUE, "#6a6a6a", RED])
    c = fig.add_axes([0.70, 0.16, 0.28, 0.71])
    c.set_facecolor("black")
    c.axis("off")
    v = np.nan_to_num(sh, nan=0.0)[order]
    sc = c.scatter(P[order, 0], P[order, 1], c=v, s=0.2, cmap=cm, vmin=0, vmax=1, linewidths=0)
    c.set_aspect("equal")
    cb = fig.colorbar(sc, ax=c, orientation="horizontal", fraction=0.04, pad=0.02)
    cb.set_label("I(change; s) / (I(change; s) + I(change; n)), all frames:\n0 = all network, 1 = all stimulus",
                 fontsize=8)
    fig.text(0.70, 0.95, "c   stimulus share on the brain (horizontal, head left)", fontsize=10)
    fig.text(0.01, 0.01, f"{res['run']}: {res['n_neurons']:,} neurons; law teacher-forced on the recording "
                         f"(synapse {res['law']['synapse']}, modulation {res['law']['modulation']}, input mask "
                         f"{res['law']['input_mask']}); {res['law']['neurons_with_time_varying_stimulus_drive']:,} "
                         f"neurons have a time-varying stimulus drive", fontsize=8, color="0.75")
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    plt.style.use("default")


if __name__ == "__main__":
    main()
