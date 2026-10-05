"""exp17: THE INFORMATION FLUX AS A WEATHER MAP (Cedric, 2026-10-03: "not great, mostly flickering dots; take the
example of weather broadcasts, wind flows, pressure"). The traffic movie of tools/exp17_flux.py spawned a particle per
unit of flux on each road and let it die at the road's end: dots appear and vanish. A wind map does the reverse: a
continuous VECTOR FIELD, and long-lived particles that drift through it leaving fading trails, so the eye reads
streamlines (earth.nullschool.net, windy.com).

Local analysis.
  WIND   every edge's message m_ji(t) = W_ji phi(z_j(t)) Omega_i(t) (the law's own term, as exp17_flux.py) moves
         information from sender j to receiver i: it deposits the vector |m_ji| (p_i - p_j) / |p_i - p_j| at three
         points along the edge (1/4, 1/2, 3/4), in the top view (horizontal, head left), on a grid of `--cells` cells
         across the brain's length, smoothed (Gaussian, SIGMA_UM). Its excitatory and inhibitory parts are gridded
         apart: a particle is red where the local flux is mostly excitatory, blue where mostly inhibitory, brighter
         where it is stronger. One field per movie frame (800 over the 2 h), linearly interpolated between frames.
  PARTICLES  `--particles` of them, advected (RK2, 3 substeps per frame) at a speed proportional to the local wind
         (the 99th percentile of |wind| over the movie moves 1.5 cells per substep), each living 40-120 frames and
         reborn at a random place in the brain; their trails fade by `FADE` per frame.
  PRESSURE   under them, the LEARNED dF/F of the free rollout, gridded and dimmed: where the brain is active.
Below: the recorded and learned brain-mean dF/F with the stimulus conditions and a cursor.

    PYTHONPATH=src:tools python tools/exp17_wind.py zap_e15_cur_siren
Writes <run>/results/movie_wind.mp4 (+ a copy and a still in experiments/exp17_zapbench_graphcast/png/).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FADE = 0.82                    # trail decay per frame (Cedric, 2026-10-03: too many filaments -> shorter trails)
SIGMA_UM = 25.0                # the field's smoothing, um: the large-scale flow, as a weather map, not every edge
SUB = 3


def fields(name, cells=220, device="cuda:0", sigma=SIGMA_UM, view="top"):
    """(wind [F, 2, ny, nx], excitatory share [F, ny, nx], speed [F, ny, nx], activity [F, ny, nx], inside [ny, nx],
    grid (x0, y0, h), the run's pieces)."""
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import neuron_graph_op, _brain_view
    spec = T.load(name)
    out = T.out_dir(spec, None)
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    op = neuron_graph_op(spec, "cpu")
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    mu, sd = float(rec["dff"].mean()), float(rec["dff"].std())
    mv = np.load(os.path.join(out, "results", f"{name}_movie.npz"))
    fr, pred = mv["frames"], np.nan_to_num(mv["pred"].astype(np.float32))
    bv = _brain_view(np.asarray(rec["pos_um"], np.float64))
    # `view`: top (head left, from above) or side (head left, the planes stacked: the messages projected on the sagittal plane;
    # Cedric, 2026-10-04: "side flows view")
    if view == "top":
        P = bv[:, :2]
    elif view == "side":
        P = np.stack([bv[:, 0], bv[:, 2]], 1)                                  # side: flipped (Cedric, 2026-10-04)
    else:                                   # oblique: head left, seen from 45 deg above (Cedric, 2026-10-04: "3D oblique")
        P = np.stack([bv[:, 0], np.cos(np.pi / 4) * bv[:, 1] + np.sin(np.pi / 4) * bv[:, 2]], 1)
    pad = 0.03 * np.ptp(P[:, 0])
    x0, y0 = P[:, 0].min() - pad, P[:, 1].min() - pad
    h = (np.ptp(P[:, 0]) + 2 * pad) / cells
    nx, ny = cells, int(np.ceil((np.ptp(P[:, 1]) + 2 * pad) / h))
    Pt = torch.as_tensor(P, device=device, dtype=torch.float32)
    cell_of = lambda xy: (((xy[:, 1] - y0) / h).long().clamp(0, ny - 1) * nx + ((xy[:, 0] - x0) / h).long().clamp(0, nx - 1))
    inside = torch.zeros(ny * nx, device=device).index_add_(0, cell_of(Pt), torch.ones(len(P), device=device))
    z = torch.as_tensor((pred - mu) / sd, device=device)
    phi = {"tanh": torch.tanh, "relu": torch.relu, "linear": lambda v: v}[op.activation](z)
    omp = os.path.join(out, "results", f"{name}_omega.npz")
    om = (torch.as_tensor(np.load(omp)["omega"].astype(np.float32), device=device) if op.modulation != "none"
          else torch.ones_like(z))
    E = []
    for s in ("short", "mid", "long"):
        e = next((l for l in spec["learnable"] if l.get("param") == f"W_{s}"), None)
        if e is None:                                       # an edge set this graph does not have (batch 17)
            continue
        w = fit[T.Learnables.key(e)].float().to(device).reshape(-1)
        snd, rcv = (t.to(device) for t in op._E[s])
        d = Pt[rcv] - Pt[snd]
        u = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)
        cells_ = torch.stack([cell_of(Pt[snd] + f_ * d) for f_ in (0.25, 0.5, 0.75)], 0)       # [3, E]
        E.append((w, snd, rcv, u, cells_))
    sg = SIGMA_UM / h                                                  # in cells
    R = int(np.ceil(3 * sg))
    g = torch.exp(-0.5 * (torch.arange(-R, R + 1, device=device, dtype=torch.float32) / sg) ** 2)
    g = (g / g.sum())
    K = (g[:, None] * g[None, :])[None, None]

    def smooth(a):                                                       # [C, ny, nx]
        return torch.nn.functional.conv2d(a[:, None], K, padding=R)[:, 0]

    def smooth_w(a):                     # the WIND's smoothing: `sigma` um (Cedric, 2026-10-04: a twin with none, sigma 0)
        if sigma <= 0:
            return a
        sw = sigma / h
        Rw = int(np.ceil(3 * sw))
        gw = torch.exp(-0.5 * (torch.arange(-Rw, Rw + 1, device=device, dtype=torch.float32) / sw) ** 2)
        gw = gw / gw.sum()
        return torch.nn.functional.conv2d(a[:, None], (gw[:, None] * gw[None, :])[None, None], padding=Rw)[:, 0]
    suf = ("" if sigma == SIGMA_UM else f"_sigma{sigma:g}") + ("" if view == "top" else f"_{view}")
    # Cedric, 2026-10-03: "two maps, one excitatory and one inhibitory": two wind fields, built from the messages'
    # excitatory part (m > 0) and inhibitory part (m < 0) apart, so neither cancels the other
    wind = {k: np.zeros((len(fr), 2, ny, nx), np.float32) for k in ("ex", "in")}
    act = np.zeros((len(fr), ny, nx), np.float32)
    cnt = inside.clamp(min=1)
    for f in range(len(fr)):
        V = torch.zeros(4, ny * nx, device=device)
        for w, snd, rcv, u, cells_ in E:
            m = w * phi[f, snd] * om[f, rcv]
            ae, ai = m.clamp(min=0), (-m).clamp(min=0)
            for c in cells_:
                V[0].index_add_(0, c, ae * u[:, 0])
                V[1].index_add_(0, c, ae * u[:, 1])
                V[2].index_add_(0, c, ai * u[:, 0])
                V[3].index_add_(0, c, ai * u[:, 1])
        S = smooth_w(V.reshape(4, ny, nx)).cpu().numpy()
        wind["ex"][f], wind["in"][f] = S[:2], S[2:]
        A_ = torch.zeros(ny * nx, device=device).index_add_(0, cell_of(Pt), torch.as_tensor(pred[f], device=device)) / cnt
        act[f] = smooth(A_.reshape(1, ny, nx))[0].cpu().numpy()
    ins = (smooth((inside > 0).float().reshape(1, ny, nx))[0] > 0.2).cpu().numpy()
    sp = {k: np.linalg.norm(v, axis=1) for k, v in wind.items()}
    print(f"[wind] {name}: {len(fr)} frames, grid {nx} x {ny} ({h:.1f} um cells); 99th percentile of the wind "
          f"speed: excitatory {np.percentile(sp['ex'][:, ins], 99):.3g}, inhibitory {np.percentile(sp['in'][:, ins], 99):.3g}")
    # the time-averaged fields, kept for comparing runs (exp17 batch 17: same graph or not, same flows?)
    np.savez_compressed(os.path.join(out, "results", f"{name}_wind_fields{suf}.npz"), ex=wind["ex"].mean(0),
                        inh=wind["in"].mean(0), inside=ins, grid=np.array([x0, y0, h]))
    return dict(wind=wind, speed=sp, act=act, inside=ins, grid=(x0, y0, h), fr=fr, rec=rec, out=out, pred=pred, P=P,
                sigma=sigma, suffix=suf)


class _Map:
    """One wind map: its particles, their ages and its trail canvas."""

    def __init__(self, wind, ins, vref, colour, n, rng, S=3):
        self.w, self.ins, self.vref, self.col, self.rng, self.S = wind, ins, vref, np.asarray(colour), rng, S
        F, _, self.ny, self.nx = wind.shape
        self.iy, self.ix = np.nonzero(ins)
        self.xy, self.life = self.spawn(n)
        self.age = rng.integers(0, 15, n)
        self.canvas = np.zeros((self.ny * S, self.nx * S, 3), np.float32)

    def spawn(self, n):
        k = self.rng.integers(0, len(self.ix), n)
        return np.stack([self.ix[k] + self.rng.random(n), self.iy[k] + self.rng.random(n)], 1), self.rng.integers(15, 45, n)

    def sample(self, arr, xy):                                          # bilinear, arr [..., ny, nx]
        x = np.clip(xy[:, 0] - 0.5, 0, self.nx - 1.001)
        y = np.clip(xy[:, 1] - 0.5, 0, self.ny - 1.001)
        i, j = y.astype(int), x.astype(int)
        fy, fx = y - i, x - j
        return (arr[..., i, j] * (1 - fy) * (1 - fx) + arr[..., i, j + 1] * (1 - fy) * fx
                + arr[..., i + 1, j] * fy * (1 - fx) + arr[..., i + 1, j + 1] * fy * fx)

    def step(self, f):
        F = self.w.shape[0]
        f1 = min(f + 1, F - 1)
        S = self.S
        for sub in range(SUB):
            a_ = sub / SUB
            W = (1 - a_) * self.w[f] + a_ * self.w[f1]
            v1 = self.sample(W, self.xy).T / self.vref * 1.5
            v2 = self.sample(W, self.xy + 0.5 * v1).T / self.vref * 1.5
            new = self.xy + v2
            c = self.col[None] * np.clip(0.2 + np.linalg.norm(v2, axis=1) / 1.5, 0, 1.4)[:, None] * 0.30
            for q in (0.25, 0.5, 0.75, 1.0):
                p = self.xy + q * (new - self.xy)
                px = np.clip((p[:, 0] * S).astype(int), 0, self.nx * S - 1)
                py = np.clip((p[:, 1] * S).astype(int), 0, self.ny * S - 1)
                np.add.at(self.canvas, (py, px), c)
            self.xy = new
            self.canvas *= FADE ** (1 / SUB)
        self.age += 1
        xy = self.xy
        out_ = (self.age > self.life) | (xy[:, 0] < 0) | (xy[:, 0] >= self.nx) | (xy[:, 1] < 0) | (xy[:, 1] >= self.ny)
        out_ |= ~self.ins[np.clip(xy[:, 1].astype(int), 0, self.ny - 1), np.clip(xy[:, 0].astype(int), 0, self.nx - 1)]
        if out_.any():
            self.xy[out_], self.life[out_] = self.spawn(int(out_.sum()))
            self.age[out_] = 0


def render(name, D, n_part=6000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks import trace_recording as TR
    rng = np.random.default_rng(seed)
    act, ins = D["act"], D["inside"]
    vref = max(np.percentile(D["speed"]["ex"][:, ins], 99), np.percentile(D["speed"]["in"][:, ins], 99))   # ONE scale
    maps = {"ex": _Map(D["wind"]["ex"], ins, vref, (1.0, 0.35, 0.22), n_part, rng),
            "in": _Map(D["wind"]["in"], ins, vref, (0.35, 0.62, 1.0), n_part, rng)}
    F, _, ny, nx = D["wind"]["ex"].shape
    S = maps["ex"].S
    rec, fr = D["rec"], D["fr"]
    cond = rec["condition"][fr]
    tm = fr * 0.914 / 60
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    names = [str(x) for x in rec["names"]]
    mo, mp = rec["dff"][fr].mean(1), D["pred"].mean(1)
    amax = np.percentile(act[:, ins], 99)
    tmp = tempfile.mkdtemp(prefix="wind_")
    fig = plt.figure(figsize=(16, 6.4), facecolor="black")
    ims = {}
    for j_, (k, lab) in enumerate((("ex", "EXCITATORY flux (W phi(z) > 0)"), ("in", "INHIBITORY flux (W phi(z) < 0)"))):
        ax = fig.add_axes([0.005 + 0.5 * j_, 0.30, 0.49, 0.60])
        ax.axis("off")
        ims[k] = ax.imshow(np.zeros((ny * S, nx * S, 3)), origin="lower", interpolation="bilinear")
        fig.text(0.01 + 0.5 * j_, 0.965, lab, color="white", fontsize=12, va="top")
    sg_ = D.get("sigma", SIGMA_UM)
    fig.text(0.20, 0.925, ("wind = the messages, sender to receiver, " + (f"smoothed over {sg_:g} um" if sg_ > 0 else
             "NOT smoothed") + "; brighter = stronger, one scale for both maps; under "
             "them the learned dF/F"), color="0.8", fontsize=9, va="top")
    t_txt = fig.text(0.01, 0.925, "", color="0.75", fontsize=9, va="top")
    m_ = fig.add_axes([0.05, 0.08, 0.90, 0.15])
    m_.set_facecolor("black")
    for k_, sp in m_.spines.items():
        sp.set_visible(k_ in ("left", "bottom"))
        sp.set_color("0.5")
    for i_, (a_, b_) in enumerate(zip(st, en)):
        m_.axvspan(tm[a_], tm[b_], color=("0.30" if i_ % 2 else "0.18"), alpha=0.6, lw=0)
        m_.text((tm[a_] + tm[b_]) / 2, 1.02, names[int(cond[a_])], color="0.75", fontsize=7, ha="center", va="bottom",
                transform=m_.get_xaxis_transform())
    m_.plot(tm, mo, color="#2ca02c", lw=0.7)
    m_.plot(tm, mp, color="white", lw=0.7)
    m_.set_xlim(tm[0], tm[-1])
    m_.set_yticks([])
    m_.tick_params(colors="0.6", labelsize=7)
    fig.text(0.05, 0.255, "brain-mean dF/F: recorded (green), learned (white); time, min", color="0.7", fontsize=8)
    cur = m_.axvline(tm[0], color="#ff7f0e", lw=0.9)
    for f in range(F):
        bg = np.clip(act[f] / amax, 0, 1)
        if D.get("bg_style") == "activity":             # the activity itself, as the run movies draw it (inferno)
            bgc = (matplotlib.colormaps["inferno"](bg)[..., :3] * ins[..., None]).repeat(S, 0).repeat(S, 1) * 0.75
        else:
            bgc = (np.stack([bg * 0.45, bg * 0.40, bg * 0.45], -1) * ins[..., None]).repeat(S, 0).repeat(S, 1) * 0.45
        for k, mp_ in maps.items():
            mp_.step(f)
            ims[k].set_data(np.clip(bgc + mp_.canvas, 0, 1))
        t_txt.set_text(f"{names[int(cond[f])]}   t = {tm[f]:5.1f} min")
        cur.set_xdata([tm[f]] * 2)
        fig.savefig(os.path.join(tmp, f"{f:05d}.png"), dpi=100, facecolor="black")
    plt.close(fig)
    path = os.path.join(D["out"], "results", f"movie_wind{D.get('suffix', '')}.mp4")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-framerate", "25", "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", path], check=True)
    shutil.copy(os.path.join(tmp, f"{F // 2:05d}.png"), os.path.join(EXP, "png", f"wind_frame_{name}{D.get('suffix', '')}.png"))
    shutil.copy(path, os.path.join(EXP, "png", f"wind_movie_{name}{D.get('suffix', '')}.mp4"))          # Cedric follows the work in png/
    shutil.rmtree(tmp)
    print(f"[wind] {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cells", type=int, default=160)
    ap.add_argument("--particles", type=int, default=6000, help="per map")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sigma", type=float, default=SIGMA_UM, help="the wind's smoothing, um (0: none)")
    a = ap.parse_args()
    render(a.run, fields(a.run, a.cells, a.device, a.sigma), a.particles)
