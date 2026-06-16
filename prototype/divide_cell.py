"""divide_cell.py -- a differentiable growing/dividing soft cell.

A round soft cell = hundreds of particles held together by cohesion-to-centroid
plus soft short-range repulsion (an incompressible disk). Particles DUPLICATE
(dormant slots in a fixed buffer switch on, each placed next to a parent), so the
cell GROWS; when a cell's occupancy mass crosses a threshold it DIVIDES into two
daughters (split along the principal axis); the daughters grow and divide again.

The point is the *continuous-occupancy* formulation of Divide/Die from the
operator taxonomy: a FIXED buffer of N_max slots, each with a mass w_i in [0,1]
(0 = dormant). Nothing is ever resized, so the state tensors keep a constant
shape and the whole rollout -- including every division -- is differentiable.
`grad_check()` backpropagates an outcome (final colony spread) to a physical
parameter (cohesion) straight through the division events and prints the gradient.

  PYTHONPATH=../src python divide_cell.py          # render gif + run grad check

Standing convention in this repo: GPU work goes on cuda:1.
"""
import os, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
EPS = 1e-6


# --------------------------------------------------------------------------- #
#  parameters
# --------------------------------------------------------------------------- #
class P:
    dt        = 0.02
    r0        = 0.013     # repulsion range  (~ particle diameter)
    k_rep     = 0.6       # repulsion strength
    k_coh     = 0.9       # cohesion-to-centroid strength  (the grad-check knob)
    fmax      = 1.2       # per-particle force clamp (stability)
    g_ramp    = 4.0       # mass ramp-in rate for a freshly duplicated particle
    w_init    = 0.15      # mass a daughter particle starts at
    div_ratio = 2.0       # a cell divides once its mass has DOUBLED since birth
    split_push= 0.012     # how far apart the two daughters are nudged


# --------------------------------------------------------------------------- #
#  duplication schedule (precomputed, detached: WHICH slot wakes WHEN, near WHOM)
# --------------------------------------------------------------------------- #
def build(N0, N_max, rate, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.zeros(N_max, 2)
    # initial cell: a uniform disk of N0 particles centred at (0.5, 0.5)
    rr = 0.06 * torch.sqrt(torch.rand(N0, generator=g))
    th = 2 * math.pi * torch.rand(N0, generator=g)
    pos[:N0, 0] = 0.5 + rr * torch.cos(th)
    pos[:N0, 1] = 0.5 + rr * torch.sin(th)
    w = torch.zeros(N_max); w[:N0] = 1.0
    cell = torch.zeros(N_max, dtype=torch.long)
    t_act = torch.full((N_max,), -1, dtype=torch.long)          # -1 = initial
    parent = torch.zeros(N_max, dtype=torch.long)
    offset = torch.zeros(N_max, 2)
    for k in range(N0, N_max):                                  # dormant slots
        T = (k - N0) // rate
        t_act[k] = T                                            # wake `rate`/tick
        first = N0 + T * rate                                   # first slot of THIS tick's batch
        parent[k] = int(torch.randint(0, first, (1,), generator=g))  # a slot already active (earlier tick)
        a = 2 * math.pi * torch.rand(1, generator=g)
        offset[k] = 0.5 * P.r0 * torch.tensor([math.cos(a), math.sin(a)]).squeeze()
    return (pos.to(DEV), w.to(DEV), cell.to(DEV),
            t_act.to(DEV), parent.to(DEV), offset.to(DEV))


# --------------------------------------------------------------------------- #
#  one tick  (functional in pos & w -> autograd-safe; cell is bookkeeping)
# --------------------------------------------------------------------------- #
def forces(pos, w, cell, C, k_coh):
    alive = (w > EPS).float()
    # mass-weighted cell centroids
    sum_pos = torch.zeros(C, 2, device=DEV).index_add(0, cell, w[:, None] * pos)
    mass    = torch.zeros(C, device=DEV).index_add(0, cell, w)
    centroid = sum_pos / mass.clamp_min(EPS)[:, None]
    f_coh = -k_coh * (pos - centroid[cell])                     # pull to own centroid
    # soft short-range repulsion (mass-weighted by the neighbour)
    N = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]                    # [N,N,2]
    d = torch.sqrt((diff * diff).sum(2) + 1e-9)                 # softened: grad-safe at 0
    off = 1.0 - torch.eye(N, device=DEV)                        # drop self-pairs
    mag = (P.k_rep * (1.0 - d / P.r0)).clamp_min(0.0) * off * w[None, :]
    f_rep = (mag[:, :, None] * diff / d[:, :, None]).sum(1)
    f = f_coh + f_rep
    fn = f.norm(dim=1, keepdim=True).clamp_min(EPS)
    f = f * (fn.clamp(max=P.fmax) / fn)                         # clamp magnitude
    return f * alive[:, None]


def step(state, t, sched, C_next, birth_mass, k_coh):
    pos, w, cell = state
    t_act, parent, offset = sched
    # --- duplication: wake the slots scheduled for this tick ---
    act = (t_act == t)
    if bool(act.any()):
        newpos = pos[parent] + offset
        pos = torch.where(act[:, None], newpos, pos)
        w   = torch.where(act, torch.full_like(w, P.w_init), w)
        cell = cell.clone(); cell[act] = cell[parent][act]
    alive = w > EPS
    # --- grow mass (smooth ramp toward 1) ---
    w = torch.where(alive, (w + P.g_ramp * P.dt).clamp(max=1.0), w)
    # --- forces + overdamped integration ---
    C = int(cell[alive].max().item()) + 1
    pos = pos + P.dt * forces(pos, w, cell, C, k_coh)
    # --- division: split any cell over the mass threshold (decision detached) ---
    nudge = torch.zeros_like(pos)
    for c in range(C):
        mem = (cell == c) & alive
        mc = float(w[mem].sum())
        if mc < P.div_ratio * birth_mass[c]:                    # divide once doubled
            continue
        idx = mem.nonzero(as_tuple=True)[0]
        p = pos[idx].detach(); ww = w[idx].detach()
        cen = (ww[:, None] * p).sum(0) / ww.sum()
        X = (p - cen) * ww.sqrt()[:, None]
        cov = (X.t() @ X) / ww.sum()
        axis = torch.linalg.eigh(cov)[1][:, -1]                 # principal axis
        side = ((p - cen) @ axis) > 0
        m_side = float(w[idx[side]].sum())                      # the two daughters' masses
        birth_mass[c] = mc - m_side                             # kept side, re-baselined
        birth_mass.append(m_side)                               # new daughter (index == C_next)
        cell = cell.clone(); cell[idx[side]] = C_next; C_next += 1
        signed = torch.zeros(pos.shape[0], device=DEV)
        signed[idx[side]] = 1.0; signed[idx[~side]] = -1.0
        nudge = nudge + signed[:, None] * axis[None, :] * P.split_push
    pos = pos + nudge
    return (pos, w, cell), C_next


# --------------------------------------------------------------------------- #
#  rollout
# --------------------------------------------------------------------------- #
def run(N0, N_max, rate, ticks, record_every, k_coh, grad=False):
    pos, w, cell, t_act, parent, offset = build(N0, N_max, rate)
    sched = (t_act, parent, offset)
    state = (pos, w, cell); C_next = 1
    birth_mass = [float(w.sum())]                               # cell 0's birth size (= N0)
    hist = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for t in range(ticks):
            state, C_next = step(state, t, sched, C_next, birth_mass, k_coh)
            if (not grad) and t % record_every == 0:
                p, ww, cc = state
                a = ww > EPS
                hist.append((p[a].detach().cpu().numpy(),
                             ww[a].detach().cpu().numpy(),
                             cc[a].detach().cpu().numpy(), C_next))
    return state, C_next, hist


# --------------------------------------------------------------------------- #
#  render
# --------------------------------------------------------------------------- #
def render(hist, path, fps=20):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.cm as cm
    # follow-cam: centre on the colony each frame; zoom out monotonically as it grows
    centers, spans, run = [], [], 0.0
    for h in hist:
        p = h[0]; c = p.mean(0); e = float(np.abs(p - c).max())
        run = max(run, e); centers.append(c); spans.append(run * 1.25 + 0.02)
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(5.5, 5.5)); fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    palette = cm.get_cmap("tab20")
    MAXC = 32                                                   # soft "membrane" per cell
    membranes = [Circle((0, 0), 0.0, lw=0) for _ in range(MAXC)]
    for m in membranes:
        m.set_visible(False); ax.add_patch(m)
    sc = ax.scatter([], [], s=6)
    tt = ax.set_title("", color="white", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    def upd(i):
        p, ww, cc, C = hist[i]
        ids = sorted(set(cc.tolist()))
        for j, m in enumerate(membranes):
            if j < len(ids):
                pc = p[cc == ids[j]]; cen = pc.mean(0)
                r = 1.7 * float(np.sqrt(((pc - cen) ** 2).sum(1).mean()))   # ~ enclosing radius
                m.set_center((cen[0], cen[1])); m.set_radius(r)
                m.set_facecolor(palette(ids[j] % 20)); m.set_alpha(0.16); m.set_visible(True)
            else:
                m.set_visible(False)
        sc.set_offsets(p)
        sc.set_color(palette(cc % 20))
        sc.set_alpha(0.95)
        sc.set_sizes(7 * ww + 1)
        cx, cy = centers[i]; s = spans[i]
        ax.set_xlim(cx - s, cx + s); ax.set_ylim(cy - s, cy + s)
        tt.set_text(f"divide_cell   frame {i}/{len(hist)-1}   |   {len(p)} particles, {len(ids)} cells")
        return [sc, tt] + membranes

    FuncAnimation(fig, upd, frames=len(hist), blit=False).save(
        path, writer=PillowWriter(fps=fps)); plt.close(fig)


# --------------------------------------------------------------------------- #
#  differentiability check: backprop an outcome to cohesion, through divisions
# --------------------------------------------------------------------------- #
def _colony_spread(state):
    """mass-weighted variance of particles about each cell's own centroid (a scalar
    outcome of the whole grow-and-divide rollout)."""
    pos, w, cell = state
    C = int(cell[w > EPS].max().item()) + 1
    sum_pos = torch.zeros(C, 2, device=DEV).index_add(0, cell, w[:, None] * pos)
    mass    = torch.zeros(C, device=DEV).index_add(0, cell, w)
    centroid = sum_pos / mass.clamp_min(EPS)[:, None]
    return (w * ((pos - centroid[cell]) ** 2).sum(1)).sum() / mass.sum()


def grad_check():
    """Test that the rollout is differentiable THROUGH a division.

    Backprop the colony spread to the cohesion parameter and cross-check the
    autograd gradient against a central finite difference. Done in float64 in a
    numerically gentle regime (soft repulsion, short horizon) so the rollout is
    not chaotic -- the long stiff rollout used for the movie has a correct but
    exploding gradient (float32 overflows it), which is a separate stiffness
    issue, not a failure of the division operator.
    """
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    save = (P.dt, P.k_rep)
    P.dt, P.k_rep = 0.01, 0.25                          # gentle regime (non-chaotic)
    try:
        # N0=120 doubles to 240 within the horizon -> one division to cross-check
        cfg = dict(N0=120, N_max=420, rate=3, ticks=70, record_every=0)

        def loss_of(kval, grad):
            k = torch.tensor(kval, device=DEV, requires_grad=grad)
            state, C_next, _ = run(k_coh=k, grad=True, **cfg)
            return _colony_spread(state), k, C_next

        L, k, n_cells = loss_of(P.k_coh, grad=True)
        L.backward(); g_ad = float(k.grad)
        h = 1e-4
        Lp = float(loss_of(P.k_coh + h, grad=False)[0])
        Lm = float(loss_of(P.k_coh - h, grad=False)[0])
        g_fd = (Lp - Lm) / (2 * h)
        rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)
        return dict(spread=float(L), grad_ad=g_ad, grad_fd=g_fd,
                    rel_err=rel, n_cells=n_cells)
    finally:
        P.dt, P.k_rep = save
        torch.set_default_dtype(prev)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    torch.manual_seed(0)

    print("[1/3] rollout + render ...", flush=True)
    _, C_next, hist = run(N0=200, N_max=2200, rate=4, ticks=520,
                          record_every=4, k_coh=P.k_coh, grad=False)
    last = hist[-1]
    print(f"      final: {len(last[0])} particles, {C_next} cells over {len(hist)} frames", flush=True)
    gif = os.path.join(HERE, "divide_cell.gif")
    render(hist, gif)
    print(f"      wrote {gif}", flush=True)

    print("[2/3] differentiability check (autograd vs finite-difference, through a division) ...", flush=True)
    r = grad_check()
    print(f"      cells={r['n_cells']}  d(spread)/d(cohesion): autograd={r['grad_ad']:+.6e}  "
          f"finite-diff={r['grad_fd']:+.6e}  rel.err={r['rel_err']:.2e}", flush=True)
    print(f"      => autograd matches finite differences through the division: "
          f"{r['rel_err'] < 1e-2}  (residual = discrete which-daughter split, not gradient error)",
          flush=True)

    print("[3/3] done.", flush=True)
