"""opt_dicty.py -- UCB optimization of the dicty model against the real movie.

Same kNN-surrogate UCB search as opt_ant.py / race_opt.py, retargeted at matching
the simulation's cell-DENSITY evolution to the real Dictyostelium movie. The loss is
the mean squared difference between the simulation's coarse density map and the movie's
(target_density.npz from make_target.py) over K matched timepoints, each normalized
COM-centred + RMS-scaled so the comparison is invariant to the movie's rotating box /
field-of-view drift. The objective the UCB maximizes is -loss.

Levers (11) -- exactly the knobs the user asked to sweep (N cells, division rate,
chemotaxis rules) plus the spring/field params that shape the aggregate:

    fleet     : cell.n
    influx    : inflow.rate
    chemotaxis: sense.gain, secrete.rate, field.diffusion, field.decay
    mechanics : spring.k_rep, spring.r0, spring.mu_f
    motility  : random_walk.strength, dt

Whenever the best loss improves by >= IMPROVE it writes a winner spec
(specs/dicty_opt_winner_<k>.yaml), renders its gif, and rewrites WINNERS_dicty.md.

    PYTHONPATH=../../src python opt_dicty.py            # run until killed
    PYTHONPATH=../../src python opt_dicty.py 3600       # stop after N seconds
    PYTHONPATH=../../src python opt_dicty.py --test     # 3-eval smoke test
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _PROTO)
sys.path.insert(0, os.path.join(os.path.dirname(_PROTO), "src"))
os.chdir(_HERE)

import torch
from scenario_schema import load
import dicty_engine

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EVAL_FRAMES = 400                 # coalescence into few mounds needs time
IMPROVE = 0.004                   # write a winner when loss drops by at least this
GRID = 48

#         fleet  division  ------------- chemotaxis -------------  ---------- mechanics ----------  motility  dt
PARAMS = ["cell.n", "inflow.rate", "sense.gain", "secrete.rate", "field.diffusion",
          "field.decay", "spring.k_rep", "spring.r0", "spring.kadh", "spring.r_on", "spring.mu_f",
          "random_walk.strength", "dt"]
BOUNDS = np.array([
    [300.,  900.],    # cell.n  (seeded from real frame-0 positions regardless of count)
    [0.0,   4.0],     # inflow.rate            (cells ENTERING the volume per frame)
    [2.0,   40.0],    # sense.gain             (chemotactic sensitivity)
    [1.0,   10.0],    # secrete.rate           (cAMP deposit)
    [0.001, 0.060],   # field.diffusion        (longer range -> fewer/larger mounds)
    [0.03,  0.30],    # field.decay
    [10.,   80.],     # spring.k_rep           (repulsion stiffness)
    [0.010, 0.040],   # spring.r0              (cell radius / rest length)
    [0.0,   200.],    # spring.kadh            (ADHESION: merges mounds -> coalescence)
    [0.02,  0.25],    # spring.r_on            (adhesion range: long -> mounds merge)
    [0.02,  0.10],    # spring.mu_f            (force scale)
    [0.0,   0.03],    # random_walk.strength
    [0.2,   0.8],     # dt
])
D = len(PARAMS)


from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

W_VEL = 0.50                      # weight of the velocity-distribution loss vs the density loss
SMOOTH = 1.5                      # gaussian smoothing (bins) applied to density maps before SSIM


def _target():
    z = np.load(os.path.join(_HERE, "target_density.npz"))
    v = np.load(os.path.join(_HERE, "velocity_target.npz"))
    return z["t_frac"], z["dens"].astype(np.float32), v["vfeat"].astype(np.float32)


NR = 12                           # radial-profile bins


def _smooth(d):
    return gaussian_filter(d.astype(np.float64), SMOOTH)


def density_ssim(sim_d, tgt_d):
    """Structural similarity between two (smoothed) density maps, in [0,1] (1=identical).
    NOTE: kept for the compare_*.py displays; the LOSS uses radial_profile instead because
    SSIM demands the mounds sit at the same positions (they don't -- mound locations are
    stochastic + the movie is a rotating 3-D projection), so SSIM saturates and can't rank."""
    a, b = _smooth(sim_d), _smooth(tgt_d)
    return float(ssim(a, b, data_range=max(a.max(), b.max(), 1e-9)))


_RGRID = None
def radial_profile(dmap, nr=NR):
    """Angularly-averaged mass vs radius from the (COM-centred) map centre -- a position- and
    rotation-INVARIANT measure of how CONCENTRATED the cells are. Concentrated -> mass at small
    radius; diffuse -> flat. Normalized to sum 1. Mound *positions* don't matter, only the degree
    of aggregation, which is exactly what's reproducible across sim and the rotating real movie."""
    global _RGRID
    g = dmap.shape[0]
    if _RGRID is None or _RGRID[0] != g:
        yy, xx = np.mgrid[0:g, 0:g]
        r = np.sqrt((xx - (g - 1) / 2) ** 2 + (yy - (g - 1) / 2) ** 2)
        rb = np.clip((r / (g / 2) * nr).astype(int), 0, nr - 1)
        _RGRID = (g, rb)
    rb = _RGRID[1]
    prof = np.zeros(nr)
    for b in range(nr):
        prof[b] = dmap[rb == b].sum()
    s = prof.sum()
    return (prof / s) if s > 0 else np.full(nr, 1.0 / nr)


IMG_G = 96                        # full-FOV image resolution for SSIM


def fov_image(xy, g=IMG_G, smooth=1.5):
    """Full-FOV cell-occupancy image (NO COM-recentering) from points in [0,1]^2, smoothed and
    peak-normalized. Position-sensitive on purpose: the sim is seeded from the real frame-0 cell
    positions, so a correct model grows mounds WHERE the data does -> absolute layout is meaningful."""
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=g, range=[[0, 1], [0, 1]])
    h = gaussian_filter(h.astype(np.float64), smooth)
    m = h.max()
    return h / m if m > 0 else h


def image_ssim(a, b):
    """SSIM between two full-FOV cell images (1 = identical). Sound here because the IC is shared."""
    return float(ssim(a, b, data_range=1.0))


def radial_autocorr(img, nbin=NR):
    """Angularly-averaged AUTOCORRELATION g(r) of a (FOV) image -- translation+rotation invariant
    AND structure-preserving: a multi-mound pattern shows a correlation peak at the inter-mound
    spacing, which the COM radial profile destroys. Normalized by the zero-lag value (scale-free)."""
    d = img - img.mean()
    F = np.fft.rfft2(d)
    ac = np.fft.fftshift(np.fft.irfft2(F * np.conj(F), s=d.shape))
    g = img.shape[0]
    yy, xx = np.mgrid[0:g, 0:g]
    r = np.sqrt((xx - g // 2) ** 2 + (yy - g // 2) ** 2)
    rb = np.clip((r / (g / 2) * nbin).astype(int), 0, nbin - 1)
    prof = np.array([ac[rb == b].mean() for b in range(nbin)])
    return prof / (abs(prof[0]) + 1e-9)


def n_mounds(img, smooth=2.0, rel=0.30):
    """Count distinct density peaks (mounds) = local maxima above rel*max -- the multi-mound scalar."""
    from scipy.ndimage import maximum_filter
    d = gaussian_filter(img.astype(np.float64), smooth)
    mx = maximum_filter(d, size=7)
    return int(((d == mx) & (d > rel * d.max())).sum())


def sim_speed_feats(hist, rec_idx, periodic=True):
    """Per-frame [mean, std] of the cell SPEED distribution, normalized by the sequence-mean
    speed (scale-free) -- the sim counterpart of make_velocity_target's vfeat. Speed = magnitude
    of the slot-consistent per-cell displacement into each recorded frame."""
    speeds = []
    for ri in rec_idx:
        rj = max(ri - 1, 0)
        both = hist[rj]["active"] & hist[ri]["active"]
        if both.sum() < 20:
            speeds.append(np.zeros(1, np.float32)); continue
        disp = (hist[ri]["pos_full"][both] - hist[rj]["pos_full"][both]).astype(np.float32)
        if periodic:
            disp = (disp + 0.5) % 1.0 - 0.5
        speeds.append(np.linalg.norm(disp, axis=1))
    seqmean = np.mean(np.concatenate(speeds)) + 1e-9
    return np.array([[s.mean() / seqmean, s.std() / seqmean] for s in speeds], np.float32)


def _circ_com(p):
    ang = p * 2 * np.pi
    return np.angle(np.exp(1j * ang).mean(0)) / (2 * np.pi) % 1.0


def sim_density_seq(pos_list, periodic=True, grid=GRID):
    """K sim density maps with PER-FRAME (circular) COM but a SHARED frame-0 scale -- the exact
    counterpart of make_target.seq_densities, so concentration over time is preserved and
    comparable. This is what the loss uses (not the per-frame sim_density)."""
    def center(pos):
        pos = pos.astype(np.float32)
        if periodic:
            c = _circ_com(pos); return (pos - c + 0.5) % 1.0 - 0.5
        return pos - pos.mean(0)
    u0 = center(pos_list[0])
    scale0 = np.sqrt((u0 ** 2).sum(1).mean()) + 1e-6
    out = []
    for pos in pos_list:
        u = center(pos)
        h, _, _ = np.histogram2d(*( (u / (2.5 * scale0)).T ), bins=grid, range=[[-1, 1], [-1, 1]])
        s = h.sum()
        out.append((h / s).astype(np.float32) if s > 0 else np.full((grid, grid), 1.0 / (grid * grid), np.float32))
    return out


def sim_density(pos, grid=GRID, periodic=True):
    """COM-centre (circular mean on the torus) + RMS-scale + grid histogram, sum 1 --
    the SAME normalization as make_target.normalized_density, so sim and movie maps are
    directly comparable."""
    if len(pos) < 10:
        return np.full((grid, grid), 1.0 / (grid * grid), np.float32)
    p = pos.astype(np.float32).copy()
    if periodic:                                          # unwrap around the circular mean
        ang = p * 2 * np.pi
        com = np.angle(np.exp(1j * ang).mean(0)) / (2 * np.pi) % 1.0
        p = (p - com + 0.5) % 1.0 - 0.5                   # min-image offset from COM, in [-0.5,0.5]
    else:
        p = p - p.mean(0)
    rms = np.sqrt((p ** 2).sum(1).mean()) + 1e-6
    p = p / (2.5 * rms)
    h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=grid, range=[[-1, 1], [-1, 1]])
    s = h.sum()
    return (h / s).astype(np.float32) if s > 0 else np.full((grid, grid), 1.0 / (grid * grid), np.float32)


def _apply(sc, p):
    sc.sets["cell"]["n"] = int(round(p[0]))
    sc.dt = float(p[12])
    f = sc.fields["camp"]; f["diffusion"] = float(p[4]); f["decay"] = float(p[5])
    for o in sc.operators:
        if o.op == "inflow": o.params["rate"] = float(p[1])
        elif o.op == "sense": o.params["gain"] = float(p[2])
        elif o.op == "secrete": o.params["rate"] = float(p[3])
        elif o.op == "spring":
            o.params["k_rep"] = float(p[6]); o.params["r0"] = float(p[7])
            o.params["kadh"] = float(p[8]); o.params["r_on"] = float(p[9]); o.params["mu_f"] = float(p[10])
        elif o.op == "random_walk": o.params["strength"] = float(p[11])
    return sc


def evaluate(u, t_frac, target, vtarget, frames=EVAL_FRAMES):
    """loss = mean over timepoints of (1 - SSIM(density)) + W_VEL * speed-distribution MSE.
    SSIM matches the spatial STRUCTURE of aggregation; the velocity term matches the mean+std
    of the (scale-free) speed distribution evolving over the same frames."""
    p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
    sc = _apply(load("specs/dicty_inflow.yaml"), p)
    sc.n_frames = frames; sc.record_every = max(1, frames // 60)
    try:
        _, hist = dicty_engine.run(sc, device=DEVICE)
    except Exception as e:
        print("  [run failed]", e, flush=True)
        return 10.0
    T = len(hist)
    rec_idx = (t_frac * (T - 1)).astype(int)
    K = len(rec_idx)
    fw = np.linspace(0.5, 1.5, K)                          # weight late (aggregated) frames more
    fw = fw / fw.sum()
    simd = sim_density_seq([hist[ri]["pos"] for ri in rec_idx])
    # position/rotation-invariant radial concentration profile, weighted to late frames
    dloss = float(sum(fw[k] * ((radial_profile(simd[k]) - radial_profile(target[k])) ** 2).sum()
                      for k in range(K))) * 30.0          # scale ~comparable to the velocity term
    vfeat = sim_speed_feats(hist, rec_idx)
    vloss = float((fw[:, None] * (vfeat - vtarget) ** 2).sum())
    return dloss + W_VEL * vloss


def render_winner(u, k):
    p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
    import yaml
    raw = yaml.safe_load(open("specs/dicty_inflow.yaml"))
    raw["sets"]["cell"]["n"] = int(round(p[0])); raw["dt"] = float(p[12])
    raw["fields"]["camp"]["diffusion"] = float(p[4]); raw["fields"]["camp"]["decay"] = float(p[5])
    for o in raw["operators"]:
        if o["op"] == "inflow": o["rate"] = float(p[1])
        elif o["op"] == "sense": o["gain"] = float(p[2])
        elif o["op"] == "secrete": o["rate"] = float(p[3])
        elif o["op"] == "spring":
            o["k_rep"] = float(p[6]); o["r0"] = float(p[7]); o["kadh"] = float(p[8]); o["r_on"] = float(p[9]); o["mu_f"] = float(p[10])
        elif o["op"] == "random_walk": o["strength"] = float(p[11])
    name = f"dicty_opt_winner_{k}"; raw["name"] = name
    open(f"specs/{name}.yaml", "w").write(yaml.safe_dump(raw, sort_keys=False))
    return name


def render_winner_gif(u, k, loss):
    """Write the winner spec + an INDEXED sim-vs-real comparison gif (watch the optimization)."""
    name = render_winner(u, k)
    try:
        import winner_gif
        out = winner_gif.save(name, k, loss, device=DEVICE)
        print(f"  [gif] {os.path.basename(out)}", flush=True)
    except Exception as e:
        print("  [winner gif failed]", e, flush=True)


def write_index(winners):
    head = "| # | loss | " + " | ".join(PARAMS) + " | gif |"
    sep = "|" + "---|" * (len(PARAMS) + 3)
    lines = ["# dicty optimization winners (auto-indexed by opt_dicty.py)\n",
             "Objective = mean density-map MSE vs the real movie; each row beat the previous "
             f"best by >= {IMPROVE}.\n", head, sep]
    for w in winners:
        vals = " | ".join(f"{v:.4f}" for v in w["params"])
        lines.append(f"| {w['k']} | {w['loss']:.5f} | {vals} | [{w['gif']}]({w['gif']}) |")
    open("WINNERS_dicty.md", "w").write("\n".join(lines) + "\n")


def main(budget=float("inf")):
    rng = np.random.default_rng(0)
    t_frac, target, vtarget = _target()
    t0 = time.time()
    print(f"[start] dicty opt: {D} levers, density-map loss over {len(t_frac)} timepoints, "
          f"{EVAL_FRAMES} frames/eval on {DEVICE}", flush=True)
    X, Y, winners_list, winners = [], [], [], 0
    best = 1e9

    def maybe_winner(u, loss):
        nonlocal winners, best
        if loss > best - IMPROVE:
            return
        winners += 1; gif = f"dicty_opt_winner_{winners}.gif"
        render_winner_gif(u, winners, loss)
        p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
        winners_list.append({"k": winners, "loss": loss, "params": p, "gif": gif})
        write_index(winners_list); best = loss
        print(f"[WINNER {winners}] loss={loss:.5f} -> {gif}  "
              + ", ".join(f"{k}={v:.3f}" for k, v in zip(PARAMS, p)), flush=True)

    for _ in range(8):                                          # random exploration seed
        u = rng.random(D); loss = evaluate(u, t_frac, target, vtarget)
        X.append(u); Y.append(-loss)
        print(f"  eval {len(Y):3d} (seed): loss={loss:.5f}  best={-max(Y):.5f}  ({time.time()-t0:.0f}s)", flush=True)
        maybe_winner(u, loss)

    while time.time() - t0 < budget:
        Xa, Ya = np.array(X), np.array(Y)
        best_u = Xa[Ya.argmax()]
        glob = rng.random((2500, D))
        loc = np.clip(best_u + rng.normal(0, 0.07, (2500, D)), 0, 1)
        cand = np.vstack([glob, loc])
        d = np.linalg.norm(cand[:, None, :] - Xa[None, :, :], axis=2)
        w = np.exp(-(d / 0.22) ** 2) + 1e-9
        mean = (w * Ya[None, :]).sum(1) / w.sum(1)
        unc = d.min(1)
        scale = (Ya.max() - Ya.min()) or 1.0
        u = cand[(mean + 0.6 * scale * unc).argmax()]
        loss = evaluate(u, t_frac, target, vtarget); X.append(u); Y.append(-loss)
        print(f"  eval {len(Y):3d}: loss={loss:.5f}  best={-max(Y):.5f}  ({time.time()-t0:.0f}s)", flush=True)
        maybe_winner(u, loss)

    print(f"[DONE] {len(Y)} evals, {winners} winners, best loss={-max(Y):.5f}", flush=True)


if __name__ == "__main__":
    if "--test" in sys.argv:
        tf, tg, tv = _target()
        print("[TEST] 3 evaluations (short) ...", flush=True)
        rng = np.random.default_rng(1); t0 = time.time()
        for i in range(3):
            u = rng.random(D); loss = evaluate(u, tf, tg, tv, frames=120)
            p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
            print(f"  test {i+1}: loss={loss:.5f}  n={int(p[0])} rate={p[1]:.2f} gain={p[2]:.1f}  ({time.time()-t0:.0f}s)", flush=True)
        print("[TEST OK]", flush=True)
    else:
        budget = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace('.', '').isdigit() else float("inf")
        main(budget)
