"""exp_profile.py -- identifiability of sensor_angle via its LIKELIHOOD PROFILE.

A point estimate can't tell "unidentifiable" from "the optimiser missed". So scan
sensor_angle across its whole range (turn_speed + sensor_dist fixed at TRUTH) and
evaluate the curriculum per-frame (cos,sin) loss at each value -- the profile of the
(neg-log) distribution of sensor_angle given the data. Shape tells the story:
  flat            -> genuinely unidentifiable
  dip at truth    -> identifiable (the fit just failed)
  dip elsewhere   -> biased objective / model misspec
Saves a plot per target to archive/profile_sensor_angle.png and logs the summary.
"""
import sys, os, math, datetime
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autorun as A
from sense_trainer import _curr_batch, _wrap, DEFAULT
from operators import LearnableSense

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "logbook2.md")


def logw(s):
    open(LOG, "a").write(s + "\n"); print(s, flush=True)


def curriculum_loss(s, bat, K):
    """Free-run per-frame (cos,sin) loss for a frozen LearnableSense over one batch."""
    wid, init, fld_k, pos_k, tgt_k, bnc_k = bat
    cx, sx = torch.cos(init), torch.sin(init)
    loss, wsum = 0.0, 0.0
    for k in range(K):
        mu = s.mu_vec(fld_k[k], wid, pos_k[k], cx, sx, R=R_GLOBAL)
        cm, sm = torch.cos(mu), torch.sin(mu)
        cx, sx = cx * cm - sx * sm, cx * sm + sx * cm
        tc, ts_ = torch.cos(tgt_k[k]), torch.sin(tgt_k[k])
        valid = (~bnc_k[k]).float()
        loss = loss + (((cx - tc) ** 2 + (sx - ts_) ** 2) * valid).sum(); wsum = wsum + valid.sum()
        rs = bnc_k[k].float(); cx = cx * (1 - rs) + tc * rs; sx = sx * (1 - rs) + ts_ * rs
    return float(loss / (wsum + 1e-6))


def profile_target(T, cfg=None, n=25, K=30):
    """Return (angs, losses, true_ang_u, argmin_u, well_depth) for one target -- the
    sensor_angle likelihood profile (turn,dist at truth). Reusable by the PDF builder."""
    global R_GLOBAL
    cfg = cfg or {**DEFAULT}
    lo, hi = A.SENSE_LO, A.SENSE_HI
    angs = np.linspace(0.02, 0.98, n)
    R_GLOBAL = T["R"]
    d = T["pos"][1:] - T["pos"][:-1]; theta = torch.atan2(d[..., 1], d[..., 0])
    dth = _wrap(theta[1:] - theta[:-1])
    move, sdist = float(T["p"][4]), float(T["p"][7]); margin = move + sdist + 0.02
    g = np.random.default_rng(0)
    starts = g.integers(1, max(2, min(theta.shape[0] - 1, 140) - K), size=24)
    bat = _curr_batch(T["grid"], T["pos"], theta, dth, starts, K, int(cfg["field_offset"]),
                      margin, 256, g, T["pos"].device, cfg["max_turn"])
    u_ts, u_sd, u_true = float(T["u_true"][5]), float(T["u_true"][7]), float(T["u_true"][6])
    losses = []
    for ua in angs:
        s = LearnableSense(u_ts, ua, u_sd, lo, hi, beta=cfg["beta0"], temp=cfg["temp0"]).to(T["pos"].device)
        for p in s.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            losses.append(curriculum_loss(s, bat, K))
    losses = np.array(losses)
    amin = float(angs[int(losses.argmin())])
    depth = float((losses.max() - losses.min()) / (losses.min() + 1e-9))
    return angs, losses, u_true, amin, depth


def main():
    global R_GLOBAL
    targets = A.load_targets()
    cfg = {**DEFAULT}
    n = 25; K = 30
    lo, hi = A.SENSE_LO, A.SENSE_HI
    angs = np.linspace(0.02, 0.98, n)
    logw(f"\n## PROFILE [{datetime.datetime.now().strftime('%H:%M:%S')}]: likelihood profile of "
         f"sensor_angle (turn,dist fixed at truth), curriculum loss, K={K}.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 3.2))
    except Exception:
        plt = None
    for ti, T in enumerate(targets):
        R_GLOBAL = T["R"]
        d = T["pos"][1:] - T["pos"][:-1]; theta = torch.atan2(d[..., 1], d[..., 0])
        dth = _wrap(theta[1:] - theta[:-1])
        move, sdist = float(T["p"][4]), float(T["p"][7]); margin = move + sdist + 0.02
        g = np.random.default_rng(0)
        starts = g.integers(1, max(2, min(theta.shape[0] - 1, 140) - K), size=24)
        bat = _curr_batch(T["grid"], T["pos"], theta, dth, starts, K, int(cfg["field_offset"]),
                          margin, 256, g, T["pos"].device, cfg["max_turn"])
        u_ts = float((T["u_true"][5])); u_sd = float(T["u_true"][7]); u_true_ang = float(T["u_true"][6])
        losses = []
        for ua in angs:
            s = LearnableSense(u_ts, ua, u_sd, lo, hi, beta=cfg["beta0"], temp=cfg["temp0"]).to(T["pos"].device)
            for p in s.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                losses.append(curriculum_loss(s, bat, K))
        losses = np.array(losses)
        amin = angs[int(losses.argmin())]
        flat = float((losses.max() - losses.min()) / (losses.min() + 1e-9))   # relative depth of the well
        l_true = float(np.interp(u_true_ang, angs, losses))
        verdict = ("UNIDENTIFIABLE (flat profile)" if flat < 0.15
                   else "IDENTIFIABLE, fit missed (dip at truth)" if abs(amin - u_true_ang) < 0.12
                   else "BIASED (dip away from truth)")
        logw(f"- target{ti}: true_ang_u={u_true_ang:.2f}, argmin={amin:.2f}, "
             f"relative well-depth={flat:.2f}, loss(true)={l_true:.4f}, loss(min)={losses.min():.4f}  -> {verdict}")
        if plt is not None:
            ax = axes[ti] if len(targets) > 1 else axes
            ax.plot(angs, losses, "-o", ms=3); ax.axvline(u_true_ang, color="g", ls="--", label="true")
            ax.axvline(amin, color="r", ls=":", label="argmin"); ax.set_title(f"target{ti}")
            ax.set_xlabel("sensor_angle (u)"); ax.set_ylabel("curriculum loss"); ax.legend(fontsize=7)
    if plt is not None:
        os.makedirs(os.path.join(os.path.dirname(__file__), "archive"), exist_ok=True)
        out = os.path.join(os.path.dirname(__file__), "archive", "profile_sensor_angle.png")
        fig.tight_layout(); fig.savefig(out, dpi=110); logw(f"- saved plot -> archive/profile_sensor_angle.png")


if __name__ == "__main__":
    main()
