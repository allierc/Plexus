"""A gallery of flow visualizations for the squirmer microswimmer.

Pure-flow viz only needs the analytical Stokes field (squirmer.flow_on_grid) plus
numpy tracer integration -- no chemical solver -- so this renders many styles fast:

  piv         PIV-style tracer streaks (short fading trails in the flow)
  pathlines   long integrated pathlines coloured by speed, drawn progressively
  streak      streaklines from an upstream rake in an ambient current (flow past a cell)
  quiver      animated arrow field coloured by |u| (motile cell swimming)
  vorticity   curl of the flow (the puller/pusher counter-rotating lobes)
  lic         animated line-integral-convolution "smoke" texture along streamlines
  slipwave    the metachronal "wavy surface velocity" travelling around the sphere
  dye         a passive dye blob advected + stretched by the feeding current

    PYTHONPATH=../../src python viz_swim.py            # render the whole gallery
    PYTHONPATH=../../src python viz_swim.py piv lic    # just some styles

Reference: Liu, Costello & Kanso, Nat Commun 16, 4154 (2025),
doi:10.1038/s41467-025-59413-x.
"""
from __future__ import annotations

import os, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import squirmer

BLUE = (0.0, 0.447, 0.741)
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
#  flow + sampling helpers (all numpy after one analytic eval)
# --------------------------------------------------------------------------- #
def get_flow(life="sessile", feeding=0.3, frame="body", center=(0.5, 0.5),
             axis=0.0, R=0.14, res=180, world=1.0, nmode=40):
    ny = res; nx = int(round(world * ny)); dx = 1.0 / ny
    xs = (torch.arange(nx) + 0.5) * dx
    ys = (torch.arange(ny) + 0.5) * dx
    X = xs[:, None].expand(nx, ny); Y = ys[None, :].expand(nx, ny)
    B = torch.tensor(squirmer.design_modes(feeding, life, nmode), dtype=torch.float32)
    u = squirmer.flow_on_grid(X, Y, center, axis, R, B, life, frame=frame)
    return dict(u=u.numpy(), nx=nx, ny=ny, dx=dx, W=world, R=R, center=np.array(center),
                xs=xs.numpy(), ys=ys.numpy(), life=life, feeding=feeding, frame=frame, axis=axis)


def samp(u, pos, W, dx, nx, ny):
    gx = np.clip(pos[:, 0], 0, W - 1e-6) / dx; gy = np.clip(pos[:, 1], 0, 1 - 1e-6) / dx
    i0 = np.floor(gx).astype(int); fx = (gx - i0)[:, None]; i1 = np.minimum(i0 + 1, nx - 1)
    j0 = np.floor(gy).astype(int); fy = (gy - j0)[:, None]; j1 = np.minimum(j0 + 1, ny - 1)
    a = u[i0, j0]; b = u[i1, j0]; c = u[i0, j1]; d = u[i1, j1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def advect(u, pos, dt, F, drift, W, dx, nx, ny):
    v = F * samp(u, pos, W, dx, nx, ny) + drift                 # RK2
    mid = pos + 0.5 * dt * v
    v2 = F * samp(u, mid, W, dx, nx, ny) + drift
    return pos + dt * v2


def _circle(ax, fl, color="0.85"):
    ax.add_patch(Circle(tuple(fl["center"]), fl["R"], fc=color, ec="k", lw=1.3, zorder=6))


def _frame_ax(fl, figh=5.2, bg="white"):
    fig, ax = plt.subplots(figsize=(figh * max(fl["W"], 1), figh)); fig.patch.set_facecolor(bg)
    ax.set_xlim(0, fl["W"]); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


def _save(anim, name, fps=22):
    anim.save(f"{name}.gif", writer=PillowWriter(fps=fps)); plt.close("all")
    print(f"[viz] {name}.gif", flush=True)


# --------------------------------------------------------------------------- #
#  styles
# --------------------------------------------------------------------------- #
def piv(name, fl, n=2400, dt=0.6, F=0.02, drift=(0, 0), trail=6, T=120):
    """PIV-style: many tracers advected by the flow, drawn as motion-blur streaks
    (the head plus a few fading past positions)."""
    drift = np.array(drift, float)
    p = RNG.uniform([0, 0], [fl["W"], 1], (n, 2))
    d0 = p - fl["center"]; inside = np.hypot(*d0.T) < fl["R"] * 1.15
    p[inside] = fl["center"] + fl["R"] * 1.3 * d0[inside] / np.hypot(*d0[inside].T)[:, None]
    hist = [p.copy()]
    fig, ax = _frame_ax(fl, bg="black")
    # one scatter layer per trail age (oldest faintest), drawn under the white head
    layers = [ax.scatter(p[:, 0], p[:, 1], s=1.4, c=[(0.35, 0.75, 1.0)],
                         alpha=0.55 * (1 - k / (trail + 1)), zorder=3) for k in range(trail)]
    head = ax.scatter(p[:, 0], p[:, 1], s=2.0, c="white", zorder=5)
    _circle(ax, fl, color="0.2")

    def upd(fr):
        nonlocal p
        p = advect(fl["u"], p, dt, F, drift, fl["W"], fl["dx"], fl["nx"], fl["ny"])
        out = (p[:, 0] < 0) | (p[:, 0] > fl["W"]) | (p[:, 1] < 0) | (p[:, 1] > 1)
        if out.any():
            lo, hi = ([0, 0], [fl["W"] * 0.04, 1]) if drift[0] > 0 else ([0, 0], [fl["W"], 1])
            p[out] = RNG.uniform(lo, hi, (int(out.sum()), 2))
        hist.append(p.copy())
        head.set_offsets(p)
        for k, lyr in enumerate(layers):
            if len(hist) > k + 2:
                lyr.set_offsets(hist[-(k + 2)])
        ax.set_title(f"{name}  PIV streaks", color="white", fontsize=10)
        return [head, *layers]
    _save(FuncAnimation(fig, upd, frames=T, blit=False), name)


def pathlines(name, fl, nseed=44, dt=0.5, F=0.025, steps=160, drift=(0, 0)):
    """Long pathlines from a seed grid, coloured by speed, revealed progressively."""
    drift = np.array(drift, float)
    sy = np.linspace(0.06, 0.94, nseed)
    seeds = np.stack([np.full(nseed, 0.04 * fl["W"] if drift[0] > 0 else fl["W"] * 0.5), sy], 1)
    if drift[0] == 0:
        ang = np.linspace(0, 2 * np.pi, nseed, endpoint=False)
        seeds = fl["center"] + fl["R"] * 1.6 * np.stack([np.cos(ang), np.sin(ang)], 1)
    paths = [seeds.copy()]
    for _ in range(steps):
        paths.append(advect(fl["u"], paths[-1], dt, F, drift, fl["W"], fl["dx"], fl["nx"], fl["ny"]))
    P = np.array(paths)                                          # [steps,nseed,2]
    spd = np.hypot(*samp(fl["u"], seeds, fl["W"], fl["dx"], fl["nx"], fl["ny"]).T)
    fig, ax = _frame_ax(fl)
    lines = [ax.plot([], [], "-", lw=1.3, color=plt.cm.viridis(min(s / (spd.max() + 1e-9), 1)))[0]
             for s in spd]
    _circle(ax, fl)

    def upd(fr):
        k = int((fr + 1) / 90 * steps)
        for i, ln in enumerate(lines):
            ln.set_data(P[:k, i, 0], P[:k, i, 1])
        ax.set_title(f"{name}  pathlines", fontsize=10)
        return lines
    _save(FuncAnimation(fig, upd, frames=90, blit=False), name)


def streak(name, fl, dt=0.5, F=0.03, drift=(0.01, 0), T=150, emit=26):
    """Streaklines: dye continuously released from an upstream rake, in a current."""
    drift = np.array(drift, float)
    ys = np.linspace(0.1, 0.9, emit)
    parts = np.empty((0, 2)); age = np.empty(0)
    fig, ax = _frame_ax(fl, bg="black")
    sc = ax.scatter([], [], s=3, c=[], cmap="cool", vmin=0, vmax=T)
    _circle(ax, fl, color="0.2")

    def upd(fr):
        nonlocal parts, age
        newp = np.stack([np.full(emit, 0.02 * fl["W"]), ys], 1)
        parts = np.vstack([parts, newp]); age = np.concatenate([age, np.full(emit, fr)])
        parts = advect(fl["u"], parts, dt, F, drift, fl["W"], fl["dx"], fl["nx"], fl["ny"])
        keep = (parts[:, 0] < fl["W"]) & (parts[:, 0] > 0) & (parts[:, 1] > 0) & (parts[:, 1] < 1)
        parts, age = parts[keep], age[keep]
        sc.set_offsets(parts); sc.set_array(fr - age)
        ax.set_title(f"{name}  streaklines (flow past a cell)", color="white", fontsize=10)
        return [sc]
    _save(FuncAnimation(fig, upd, frames=T, blit=False), name)


def quiver(name, fl_fn, T=120, step=7):
    """Animated arrow field coloured by |u| (recomputes flow if fl_fn moves the cell)."""
    fl0 = fl_fn(0)
    Xc, Yc = np.meshgrid(fl0["xs"][::step], fl0["ys"][::step], indexing="ij")
    fig, ax = _frame_ax(fl0)
    u0 = fl0["u"][::step, ::step]
    Q = ax.quiver(Xc, Yc, u0[..., 0], u0[..., 1], np.hypot(u0[..., 0], u0[..., 1]),
                  cmap="turbo", scale=18, width=0.0025)
    body = Circle(tuple(fl0["center"]), fl0["R"], fc="0.85", ec="k", lw=1.3, zorder=6); ax.add_patch(body)

    def upd(fr):
        fl = fl_fn(fr); u = fl["u"][::step, ::step]
        Q.set_UVC(u[..., 0], u[..., 1], np.hypot(u[..., 0], u[..., 1]))
        body.center = tuple(fl["center"])
        ax.set_title(f"{name}  velocity field", fontsize=10)
        return [Q, body]
    _save(FuncAnimation(fig, upd, frames=T, blit=False), name)


def vorticity(name, fl_fn, T=120):
    """Curl of the flow -> the squirmer's counter-rotating lobes (puller/pusher)."""
    fl0 = fl_fn(0)
    fig, ax = _frame_ax(fl0)

    def curl(fl):
        u = fl["u"]; dux = np.gradient(u[..., 0], fl["dx"], axis=1)
        duy = np.gradient(u[..., 1], fl["dx"], axis=0)
        return duy - dux
    w0 = curl(fl0); lim = np.percentile(np.abs(w0), 99) + 1e-9
    im = ax.imshow(w0.T, origin="lower", extent=[0, fl0["W"], 0, 1], cmap="RdBu_r",
                   vmin=-lim, vmax=lim)
    body = Circle(tuple(fl0["center"]), fl0["R"], fc="0.9", ec="k", lw=1.3, zorder=6); ax.add_patch(body)
    fig.colorbar(im, ax=ax, fraction=0.046, label="vorticity")

    def upd(fr):
        fl = fl_fn(fr); im.set_data(curl(fl).T); body.center = tuple(fl["center"])
        ax.set_title(f"{name}  vorticity", fontsize=10)
        return [im, body]
    _save(FuncAnimation(fig, upd, frames=T, blit=False), name)


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def vortzoom(name, life="sessile", feeding=0.25, R=0.16, res=520, T=90,
             m=8, omega=2.0, amp=0.9, delta=0.18, half_end=0.105, target=None):
    """Vorticity with a travelling METACHRONAL surface wave + a Ken Burns zoom onto
    the surface. The base squirmer flow is steady, so we add a tangential ripple
    A*exp(-(r-1)/delta)*sin(m*theta - omega*t) that decays away from the surface --
    a real travelling surface wave whose curl visibly marches around the cell. The
    camera eases from the full frame down to a tight crop on the surface."""
    center = np.array([0.5, 0.5])
    fl = get_flow(life, feeding, "body", tuple(center), 0.0, R, res=res, world=1.0)
    u0 = fl["u"]; xs, ys = fl["xs"], fl["ys"]
    X = xs[:, None]; Y = ys[None, :]
    dxc = X - center[0]; dyc = Y - center[1]
    rr = np.hypot(dxc, dyc) + 1e-9
    r = rr / R
    th = np.arctan2(dyc, dxc)                                  # angle around the cell
    outside = (r >= 1.0).astype(float)
    decay = np.exp(-(np.clip(r - 1.0, 0, None)) / delta) * outside
    etx, ety = -dyc / rr, dxc / rr                            # unit tangential (swirl) direction
    dx = fl["dx"]

    def curl(u):
        return np.gradient(u[..., 1], dx, axis=0) - np.gradient(u[..., 0], dx, axis=1)

    fig, ax = _frame_ax(fl)
    lim = np.percentile(np.abs(curl(u0)), 99.5) + 1e-9
    im = ax.imshow(curl(u0).T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
                   vmin=-1.6 * lim, vmax=1.6 * lim, interpolation="bilinear")
    body = Circle(tuple(center), R, fc="0.9", ec="k", lw=1.4, zorder=6); ax.add_patch(body)

    def upd(fr):
        t = 2 * np.pi * fr / T
        ripple = amp * decay * np.sin(m * th - omega * t)
        u = u0.copy()
        u[..., 0] += ripple * etx; u[..., 1] += ripple * ety
        im.set_data(curl(u).T)
        # Ken Burns: ease from the full frame all the way down onto the surface --
        # the camera both zooms (half -> half_end) and pans from the cell centre to
        # a point ON the surface, so a surface arc with the travelling wave fills it.
        tgt = target if target is not None else (center[0] + R, center[1])
        z = _smoothstep(min(fr / (T * 0.55), 1.0))
        half = 0.5 * (1 - z) + half_end * z
        panx = 0.5 * (1 - z) + tgt[0] * z + 0.012 * z * np.cos(0.8 * t)
        pany = 0.5 * (1 - z) + tgt[1] * z + 0.012 * z * np.sin(0.8 * t)
        ax.set_xlim(panx - half, panx + half); ax.set_ylim(pany - half, pany + half)
        ax.set_title(f"{name}  metachronal surface vorticity (zoom)", fontsize=10)
        return [im, body]
    _save(FuncAnimation(fig, upd, frames=T, blit=False), name, fps=20)


def lic(name, fl, T=40, L=18):
    """Animated line-integral-convolution: a smoke texture flowing along streamlines."""
    u = fl["u"]; nx, ny = fl["nx"], fl["ny"]
    spd = np.hypot(u[..., 0], u[..., 1]) + 1e-9
    vx = u[..., 0] / spd; vy = u[..., 1] / spd
    noise = RNG.random((nx, ny))
    Ix = np.arange(nx)[:, None].repeat(ny, 1).astype(float)
    Iy = np.arange(ny)[None, :].repeat(nx, 0).astype(float)

    def lic_frame(phase):
        out = np.zeros((nx, ny)); wsum = np.zeros((nx, ny))
        for direction in (1, -1):
            px, py = Ix.copy(), Iy.copy()
            for k in range(L):
                ix = np.clip(px.astype(int), 0, nx - 1); iy = np.clip(py.astype(int), 0, ny - 1)
                w = 0.5 + 0.5 * np.cos(2 * np.pi * 3 * k / L - direction * phase)
                out += w * noise[ix, iy]; wsum += w
                px += direction * vx[ix, iy]; py += direction * vy[ix, iy]
        return out / (wsum + 1e-9)

    base = lic_frame(0.0)
    shade = (spd / spd.max())
    fig, ax = _frame_ax(fl, bg="black")
    im = ax.imshow((base * (0.3 + 0.7 * shade)).T, origin="lower", extent=[0, fl["W"], 0, 1],
                   cmap="bone", vmin=0, vmax=1)
    _circle(ax, fl, color="0.15")

    def upd(fr):
        phase = 2 * np.pi * fr / T
        im.set_data((lic_frame(phase) * (0.3 + 0.7 * shade)).T)
        ax.set_title(f"{name}  LIC flow texture", color="white", fontsize=10)
        return [im]
    _save(FuncAnimation(fig, upd, frames=T, blit=False), name, fps=14)


def slipwave(name, life="sessile", feeding=0.3, R=0.18, T=60, nmode=40, nnode=160):
    """The 'wavy surface velocity': slip vectors around the sphere with a travelling
    metachronal wave (cilia beat coordination), beside the flow it drives."""
    center = np.array([0.5, 0.5]); axis = 0.0
    alpha = np.linspace(0, 2 * np.pi, nnode, endpoint=False)
    B = squirmer.design_modes(feeding, life, nmode)
    slipmag = squirmer.slip_profile(B, life, np.arccos(np.clip(np.cos(alpha), -1, 1)))
    mu1 = 1 - 2 * feeding; mouth = np.cos(alpha) > mu1
    ez = np.array([1.0, 0.0]); eperp = np.array([0.0, 1.0])
    pos = center + R * (np.cos(alpha)[:, None] * ez + np.sin(alpha)[:, None] * eperp)
    tang = (-np.sin(alpha)[:, None] * ez + np.cos(alpha)[:, None] * eperp)
    fl = get_flow(life, feeding, "body", tuple(center), axis, R, res=200, world=1.0, nmode=nmode)
    sp = np.hypot(*fl["u"].transpose(2, 0, 1))
    fig, ax = _frame_ax(fl)
    ax.imshow(sp.T, origin="lower", extent=[0, 1, 0, 1], cmap="Blues", vmin=0, vmax=sp.max(), alpha=0.7)
    _circle(ax, fl, color="0.92")
    ax.scatter(pos[mouth, 0], pos[mouth, 1], s=22, c="#d62728", zorder=9)
    amp0 = np.abs(slipmag) + 1e-6
    Q = ax.quiver(pos[:, 0], pos[:, 1], tang[:, 0], tang[:, 1], amp0, cmap="turbo",
                  scale=6.0, width=0.006, zorder=8, pivot="tail",
                  scale_units="width", clim=(0, 1.2))

    def upd(fr):
        wave = 0.5 + 0.5 * np.cos(alpha * 6 - 2 * np.pi * fr / T)     # metachronal travelling wave
        amp = slipmag * (0.35 + 0.65 * wave)
        Q.set_UVC(amp * tang[:, 0], amp * tang[:, 1], np.abs(amp))
        ax.set_title(f"{name}  wavy surface velocity (metachronal wave)", fontsize=10)
        return [Q]
    _save(FuncAnimation(fig, upd, frames=T, blit=False), name, fps=20)


def dye(name, fl, dt=0.45, F=0.03, drift=(0.012, 0), T=150, res_dye=None):
    """Advect a passive dye blob through the feeding current (semi-Lagrangian)."""
    drift = np.array(drift, float)
    nd = res_dye or fl["ny"]; dx = 1.0 / nd; nxg = int(round(fl["W"] * nd))
    xs = (np.arange(nxg) + 0.5) * dx; ys = (np.arange(nd) + 0.5) * dx
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    dye0 = np.exp(-(((Xg - 0.26 * fl["W"]) ** 2 + (Yg - 0.5) ** 2) / (2 * 0.045 ** 2)))
    grid = dye0.copy()
    flat = np.stack([Xg.ravel(), Yg.ravel()], 1)
    fig, ax = _frame_ax(fl, bg="black")
    im = ax.imshow(grid.T, origin="lower", extent=[0, fl["W"], 0, 1], cmap="magma", vmin=0, vmax=1)
    _circle(ax, fl, color="0.2")

    def sample_scalar(g, pts):
        gx = np.clip(pts[:, 0], 0, fl["W"] - 1e-6) / dx; gy = np.clip(pts[:, 1], 0, 1 - 1e-6) / dx
        i0 = np.floor(gx).astype(int); fx = gx - i0; i1 = np.minimum(i0 + 1, nxg - 1)
        j0 = np.floor(gy).astype(int); fy = gy - j0; j1 = np.minimum(j0 + 1, nd - 1)
        a = g[i0, j0]; b = g[i1, j0]; c = g[i0, j1]; d = g[i1, j1]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

    def upd(fr):
        nonlocal grid
        back = flat - dt * (F * samp(fl["u"], flat, fl["W"], fl["dx"], fl["nx"], fl["ny"]) + drift)
        grid = sample_scalar(grid, back).reshape(nxg, nd)
        grid[0, :] = 0
        im.set_data(grid.T)
        ax.set_title(f"{name}  dye advection", color="white", fontsize=10)
        return [im]
    _save(FuncAnimation(fig, upd, frames=T, blit=False), name)


# --------------------------------------------------------------------------- #
#  moving-cell flow factory (for motile quiver/vorticity)
# --------------------------------------------------------------------------- #
def swimmer_factory(feeding=0.3, frame="lab", world=1.8, R=0.12, res=170, speed=0.0055):
    B1 = squirmer.design_modes(feeding, "motile", 40)[0]
    U = (2.0 / 3.0) * B1 * speed

    def fl_fn(fr):
        cx = 0.22 + U * fr
        return get_flow("motile", feeding, frame, (cx, 0.5), 0.0, R, res=res, world=world)
    return fl_fn


GALLERY = {
    "piv":       lambda: [piv("piv_sessile", get_flow("sessile", 0.25, "body", res=190), F=0.03),
                          piv("piv_motile_body", get_flow("motile", 0.3, "body", res=190), F=0.03),
                          piv("piv_motile_lab", get_flow("motile", 0.3, "lab", res=190), F=0.03)],
    "pathlines": lambda: [pathlines("pathlines_sessile", get_flow("sessile", 0.25, "body", res=190)),
                          pathlines("pathlines_motile", get_flow("motile", 0.3, "lab", res=190)),
                          pathlines("pathlines_streamflow", get_flow("sessile", 0.25, "body", res=190),
                                    drift=(0.012, 0))],
    "streak":    lambda: [streak("streak_sessile", get_flow("sessile", 0.25, "body", res=190)),
                          streak("streak_motile", get_flow("motile", 0.3, "body", res=190))],
    "quiver":    lambda: [quiver("quiver_motile", swimmer_factory(0.3)),
                          quiver("quiver_sessile", lambda fr: get_flow("sessile", 0.25, "body", res=170))],
    "vorticity": lambda: [vorticity("vorticity_motile", swimmer_factory(0.3)),
                          vorticity("vorticity_puller", swimmer_factory(0.08)),
                          vorticity("vorticity_pusher", swimmer_factory(0.6))],
    "vortzoom":  lambda: [vortzoom("vortzoom_sessile", "sessile", 0.25),
                          vortzoom("vortzoom_motile", "motile", 0.3),
                          vortzoom("vortzoom_fast", "sessile", 0.2, m=12, omega=3.0)],
    "lic":       lambda: [lic("lic_sessile", get_flow("sessile", 0.25, "body", res=150)),
                          lic("lic_motile_lab", get_flow("motile", 0.3, "lab", res=150)),
                          lic("lic_motile_body", get_flow("motile", 0.3, "body", res=150))],
    "slipwave":  lambda: [slipwave("slipwave_sessile", "sessile", 0.3),
                          slipwave("slipwave_narrow", "sessile", 0.12),
                          slipwave("slipwave_motile", "motile", 0.3)],
    "dye":       lambda: [dye("dye_sessile", get_flow("sessile", 0.25, "body", res=170)),
                          dye("dye_motile", get_flow("motile", 0.3, "body", res=170))],
}

if __name__ == "__main__":
    want = sys.argv[1:] or list(GALLERY)
    for fam in want:
        print(f"=== {fam} ===", flush=True)
        GALLERY[fam]()
    print("[done] gallery", flush=True)
