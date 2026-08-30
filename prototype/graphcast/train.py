"""Training, testing, and the closed-form gate measurements.

Defaults follow GraphCast where they are load-bearing and measured (supplement secs. 4.2, 4.4, 4.5):
AdamW with beta2 = 0.95, a global gradient-norm clip, a linear warmup then cosine decay TO ZERO,
and the target normalised by the inverse variance of the INCREMENT rather than of the state. The
last of those is `s_j` in Eq 19 and it is what makes one loss weight transfer across neurons of
very different activity; the cosine-to-zero is the fix for the one-checkpoint collapses of 0.2-0.5
in R^2_W that the connectome-gnn weekend benchmark had to work around with trailing medians.

Rollout is a TAIL, not the objective. GraphCast spends 96% of its updates at one step and then
11,000 updates at K rising 2->12 at a learning rate 3,300x below peak; the weekend grid found
rollout-as-objective loses on every arm that cleared its resolution floor. `training.rollout.tail`
is that tail and it defaults to zero length.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import model as model_mod
import viz

_CGNN = "/workspace/connectome-gnn/src"


def _connectome_gnn():
    """connectome-gnn's metrics, imported not reimplemented. Returns None if unavailable."""
    if _CGNN not in sys.path:
        sys.path.insert(0, _CGNN)
    try:
        from connectome_gnn.metrics import compute_r_squared_lin_fit
        from connectome_gnn.sparsify import clustering_evaluation
        return compute_r_squared_lin_fit, clustering_evaluation
    except Exception:                                    # noqa: BLE001
        return None


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    """R^2 of `pred` against `true` after a linear fit, the convention connectome-gnn scores in."""
    got = _connectome_gnn()
    if got is not None:
        try:
            return float(got[0](np.asarray(true).ravel(), np.asarray(pred).ravel())[0])
        except Exception:                                # noqa: BLE001
            pass
    t, p = np.asarray(true).ravel(), np.asarray(pred).ravel()
    A = np.stack([p, np.ones_like(p)]).T
    c = np.linalg.lstsq(A, t, rcond=None)[0]
    return float(1 - ((t - A @ c) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-30))


def adjusted_rand(a: np.ndarray, labels: np.ndarray) -> float:
    got = _connectome_gnn()
    if got is not None:
        try:
            out = got[1](np.asarray(a), list(np.asarray(labels)))
            return float(out[0] if isinstance(out, (tuple, list)) else out)
        except Exception:                                # noqa: BLE001
            pass
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    k = int(labels.max()) + 1
    pred = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(np.asarray(a))
    return float(adjusted_rand_score(labels, pred))


# --------------------------------------------------------------------------------------- #

def load_toy(run_dir: str, device: str, fit_stimulus_is_none: bool = False):
    gt = np.load(os.path.join(run_dir, "ground_truth.npz"))
    v = torch.tensor(gt["voltage"], dtype=torch.float32, device=device)[:, :, None]
    ei = torch.tensor(gt["edge_index"], dtype=torch.long, device=device)
    # WITHHOLDING THE DRIVE IS THE POINT ON THE WAVE TOY, not an omission. For a travelling wave
    # u = f(x - ct) the gradient is du/dx = -(1/c) du/dt, so a model handed u_i can recover the
    # gradient from that node's own history and the graph is never necessary -- measured: loss
    # 0.005 with r2_grad 0.000. With the drive withheld the phase can only come from the SPATIAL
    # pattern of the neighbours, and since each neighbour's state is its own signed gain times
    # that gradient, the model has to learn the gains to read them. That is what makes the
    # heterogeneity load-bearing rather than incidental.
    if fit_stimulus_is_none:
        return gt, v, ei, torch.zeros(v.shape[0], v.shape[1], device=device)
    key = "stim" if "stim" in gt.files else "stim_t"
    st = torch.tensor(gt[key], dtype=torch.float32, device=device)
    if st.ndim == 1:                       # a global clock, broadcast to every node
        st = st[:, None].expand(-1, v.shape[1])
    return gt, v, ei, st.contiguous()


def build_schedule(tc):
    def lr_at(step):
        if tc.lr_scheduler == "none":
            return 1.0
        if step < tc.warmup_iters:
            return step / max(tc.warmup_iters, 1)
        p = (step - tc.warmup_iters) / max(tc.n_iter - tc.warmup_iters, 1)
        return 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))
    return lr_at


def train(fit, run_dir: str, device: str = "cuda") -> dict:
    device = device if torch.cuda.is_available() else "cpu"
    tc, ms = fit.training, fit.model
    torch.manual_seed(tc.seed)
    gt, v, ei, stim_t = load_toy(run_dir, device, str(fit.data.stimulus).lower() in ("none", "null", ""))
    T, N, _ = v.shape
    E = ei.shape[1]

    net = model_mod.GraphCastModel(ms, N, E, device=device, seed=tc.seed)
    dv = (v[1:] - v[:-1])                                     # the increment, GraphCast's target
    # s_j: per-neuron inverse standard deviation of the increment (supplement sec. 4.2)
    if tc.target_norm == "inverse_increment_variance":
        s = 1.0 / dv.std(dim=0, keepdim=True).clamp(min=1e-6)
    else:
        s = torch.ones(1, N, 1, device=device)

    tr0, tr1 = fit.data.split.get("train", [0, T - 1])
    # THREE PARAMETER GROUPS, and the split matters. GraphCast applies weight decay "on the weight
    # matrices" (supplement sec. 4.4) -- not on everything. Decaying W, a and b is not a
    # regularisation choice, it is a bias toward zero on the three parameters that ARE the
    # scientific output: the interaction weights, the embedding and the stimulus gain. The first
    # version of this loop decayed all three at 0.1 and W came out at a correlation of 0.049 with
    # the truth while the loss fell by four orders of magnitude.
    structural = {"W", "a", "b"}
    mats, others, struct = [], [], []
    for n, prm in net.named_parameters():
        root = n.split(".")[0]
        if root in structural:
            struct.append(prm)
        elif prm.ndim >= 2:
            mats.append(prm)
        else:
            others.append(prm)
    groups = [{"params": mats, "lr": tc.lr, "weight_decay": tc.weight_decay},
              {"params": others, "lr": tc.lr, "weight_decay": 0.0}]
    if net.a is not None:
        groups.append({"params": [net.a], "lr": tc.lr_embedding or tc.lr, "weight_decay": 0.0})
        struct = [q for q in struct if q is not net.a]
    groups.append({"params": struct, "lr": tc.lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=tc.betas)
    lr_at = build_schedule(tc)
    base = [g["lr"] for g in opt.param_groups]

    snap_dir = os.path.join(run_dir, "tmp_training")
    os.makedirs(snap_dir, exist_ok=True)
    hist = []
    for step in range(tc.n_iter):
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * lr_at(step)
        idx = torch.randint(tr0, min(tr1, T - 1), (tc.batch_frames,), device=device)
        pred = torch.stack([net(v[t], ei, stim_t[t][:, None]) for t in idx])
        loss = (((pred - dv[idx]) * s) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tc.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), tc.grad_clip)
        opt.step()
        if step % max(tc.checkpoint_every, 1) == 0 or step == tc.n_iter - 1:
            m = measure(net, gt, v, ei, stim_t, device)
            m.update(step=step, loss=float(loss))
            hist.append(m)
            print(f"  step {step:6d}  loss {float(loss):.5f}  "
                  + "  ".join(f"{k} {m[k]:.3f}" for k in ("r2_grad", "r2_gain", "r2_tau", "ari_a")))
    torch.save(net.state_dict(), os.path.join(run_dir, "model.pt"))
    np.savez(os.path.join(run_dir, "history.npz"),
             **{k: np.array([h[k] for h in hist]) for k in hist[0]})
    return {"history": hist, "net": net, "gt": gt}


@torch.no_grad()
def _learned_arrays(net, device):
    a = net.embedding()
    return (net.W.detach().cpu().numpy(),
            None if a is None else a.detach().cpu().numpy(),
            net.b.detach().cpu().numpy())


def effective_tau(net, v, ei, stim, device, n_frames: int = 8) -> np.ndarray:
    """Per-neuron tau from the model's own Jacobian: d(dv_i)/dv_i is -1/tau_i for a leaky unit.

    Read off the OPERATOR rather than from a fitted parameter, so it works for both message forms
    and does not assume the model wrote tau down anywhere.
    """
    slopes = []
    N = v.shape[1]
    for t in np.linspace(0, v.shape[0] - 2, n_frames).astype(int):
        x = v[t].clone().requires_grad_(True)
        out = net(x, ei, stim[t][:, None])
        g = torch.autograd.grad(out.sum(), x, retain_graph=False)[0]
        slopes.append(g.detach().squeeze(-1).cpu().numpy())
    leak = -np.mean(slopes, axis=0)
    return 1.0 / np.clip(leak, 1e-6, None)


def gauge_normalise(W: np.ndarray, sender: np.ndarray) -> np.ndarray:
    """Remove the per-sender scale, which the data does not determine.

    The message is W_e * g_phi(v_j, a_j) and g_phi depends on the SENDER only, so for any per-
    sender constant c_j the pair (W_e / c_j, c_j * g_phi(., a_j)) produces identical messages. W is
    therefore identifiable only up to a per-sender scaling, and scoring the raw W scores a quantity
    the observations do not fix -- the same shape of degeneracy as connectome-gnn's
    docs/degeneracy_identifiability.tex. Normalising each sender's outgoing weights to unit norm
    quotients the gauge out, so what is left is what the data can actually determine: the RELATIVE
    weights among one neuron's outgoing edges, which is the distance kernel.
    """
    out = np.zeros_like(W, dtype=np.float64)
    for j in np.unique(sender):
        m = sender == j
        n = np.linalg.norm(W[m])
        out[m] = W[m] / n if n > 1e-12 else 0.0
    return out


def measure(net, gt, v, ei, stim, device, n_frames: int = 12) -> dict:
    """The closed-form gates, restated for the two-scale wave toy.

    The fine rule is dv_i = -v_i/tau_i + g_i du/dx(r_i), so the two things to recover are a
    SPATIAL DERIVATIVE (does the message become a gradient operator?) and a SIGNED PER-NODE GAIN
    (the heterogeneity). Both are read off the trained operator rather than off a named parameter,
    so the same measurement works for either message form.
    """
    W, a, b = _learned_arrays(net, device)
    N = v.shape[1]
    ts = np.linspace(0, v.shape[0] - 2, n_frames).astype(int)

    msgs, dvs, slopes = [], [], []
    for t in ts:
        x = v[t].clone().requires_grad_(True)
        out, m = net(x, ei, stim[t][:, None], return_msg=True)
        g = torch.autograd.grad(out.sum(), x, retain_graph=False)[0]
        slopes.append(g.detach().squeeze(-1).cpu().numpy())
        msgs.append(m.detach().squeeze(-1).cpu().numpy())
        dvs.append(out.detach().squeeze(-1).cpu().numpy())
    leak = -np.mean(slopes, axis=0)
    msg = np.stack(msgs)                                     # [n_frames, N]

    # G9: the message against the true field gradient, per node, pooled
    grad = gt["grad"][ts]
    out = {"r2_grad": r2(grad.reshape(-1), msg.reshape(-1))}
    # per-node gain: d(dv)/d(msg) via the ratio of the model's own emitted dv to its message,
    # fitted per node across frames -- the slope IS the gain up to the message's own scale.
    gains = []
    dv_hat = np.stack(dvs)
    for i in range(N):
        A = np.stack([msg[:, i], np.ones(len(ts))]).T
        gains.append(np.linalg.lstsq(A, dv_hat[:, i], rcond=None)[0][0])
    out["r2_gain"] = r2(gt["gain"], np.array(gains))
    out["r2_tau"] = r2(gt["tau"], 1.0 / np.clip(leak, 1e-6, None))
    out["ari_a"] = adjusted_rand(a, gt["node_type"]) if a is not None else float("nan")
    return out
