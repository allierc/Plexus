#!/usr/bin/env python
"""cardio_mpm_forward.py -- FORWARD model on the NEW Plexus MLS-MPM operators.

Replaces the old spring stand-in (cardio_train08_09.forward_beat_loop) with the real
decomposed MLS-MPM material model now in the codebase:

  aggregate -> apply_material_map(stiffness->youngs) -> pacemaker -> pulse_stimulus
            -> pulse_to_contraction(directional) -> mpm_drag
            -> [mpm_strain -> p2g -> mpm_grid_update -> g2p] x substeps

A 128^2 elastic sheet filling [0.15,0.85]^2 is contracted by a periodically pulsed active
force whose ORIENTATION is the learnable `direction` (vector_grid) field and whose local
stiffness is the learnable `stiffness_map` (image) field -- the two maps a UNet will output
in the inverse problem. The OUTER BAND of the sheet is Dirichlet-anchored to the real
cardiomyocyte data each frame (`--anchor`); the interior is the model's prediction, and the
real trajectories are overlaid in green -- exactly as the old cardio fit/render did.

Run:
  PYTHONPATH=../../src python cardio_mpm_forward.py material/material_directional_cardio --anchor
  PYTHONPATH=../../src python cardio_mpm_forward.py material/material_map_circle --device cuda:0
Outputs -> archive/<spec name>/
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from plexus.schema import load
from plexus.paths import resolve_config
import plexus.engine as E
import cardio_mpm_data as D

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
#  per-frame rollout hook: (1) anchor the outer band to the real data (Dirichlet),
#  (2) capture post-anchor pos + the per-particle MPM continuum (F/C/vel + activation).
# --------------------------------------------------------------------------- #
class Roll:
    def __init__(self, n_frames, anchor=True, bwidth=0.06, set_name="mpm_particle", act_field="activation"):
        self.n_frames, self.anchor, self.bwidth = n_frames, anchor, bwidth
        self.set_name, self.act_field = set_name, act_field
        self.rest = self.real_disp = self.bnd = None
        self.pos, self.Fn, self.Jp, self.Cn, self.spd, self.act = [], [], [], [], [], []

    def __call__(self, H, frame):
        lvl = H.level(self.set_name); dev = lvl.state.device
        pa, pb = lvl.state_schema["pos"]; va, vb = lvl.state_schema["vel"]
        if self.rest is None:                                          # frame 0: fix rest, map real data
            self.rest = lvl.state[:, pa:pb].clone()
            rd, bnd = D.load_real_for(self.rest.cpu().numpy(), self.n_frames, self.bwidth)
            self.real_disp = torch.tensor(rd, device=dev)              # [T, N, 2]
            self.bnd = torch.tensor(bnd, device=dev)
        if self.anchor and frame < self.real_disp.shape[0]:           # pin the band to real (interior is free)
            tgt = self.rest + self.real_disp[frame]
            st = lvl.state.clone()
            st[self.bnd, pa:pb] = tgt[self.bnd]
            st[self.bnd, va:vb] = 0.0
            lvl.state = st
        # capture (post-anchor)
        F = lvl.F; J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        C = lvl.C
        pos = lvl.state[:, pa:pb]
        af = H.fields[self.act_field] if self.act_field in H.fields else None
        if af is not None:
            gx, gy = af.pix(pos[:, 0], pos[:, 1]); a = af.grid[0][gx, gy]
        else:
            a = torch.zeros(pos.shape[0], device=dev)
        for buf, val in ((self.pos, pos), (self.Fn, F.reshape(F.shape[0], -1).norm(dim=1)),
                         (self.Jp, J), (self.Cn, C.reshape(C.shape[0], -1).norm(dim=1)),
                         (self.spd, lvl.get("vel").norm(dim=1)), (self.act, a)):
            buf.append(val.detach().cpu().numpy().copy())

    def fields(self):
        return {k: np.stack(getattr(self, k)) for k in ("Fn", "Jp", "Cn", "spd", "act")}


def grid_sample_idx(rest, n=12, lo=D.DOM_LO, hi=D.DOM_HI):
    targets = np.stack(np.meshgrid(np.linspace(lo, hi, n), np.linspace(lo, hi, n), indexing="ij"), -1).reshape(-1, 2)
    d = ((rest[None] - targets[:, None]) ** 2).sum(-1)
    return d.argmin(1)


def interior_r2(sim_pos, real_disp, bnd):
    """Honest fit metric: motion-normalised variance-explained over INTERIOR MOVING nodes.
    sim_pos [T,N,2]; real_disp [T,N,2]; bnd [N] bool. R2<0 = worse than predicting no motion."""
    sim_d = sim_pos - sim_pos[0]
    mov = (np.linalg.norm(real_disp, axis=2).max(0) > 0.1 * np.linalg.norm(real_disp, axis=2).max())
    mk = (~bnd) & mov
    if mk.sum() == 0:
        return float("nan")
    res = ((sim_d[:, mk] - real_disp[:, mk]) ** 2).sum()
    tot = ((real_disp[:, mk] - real_disp[:, mk].mean(0, keepdims=True)) ** 2).sum() + 1e-12
    return float(1 - res / tot)


# --------------------------------------------------------------------------- #
#  renders
# --------------------------------------------------------------------------- #
def render_traj(sim_pos, idx, out, amp=None, real_disp=None, bnd=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    P = sim_pos[:, idx]; rest = P[0]
    if amp is None:
        amp = 0.12 / max(1e-9, float(np.abs(P - rest[None]).max()))
    A = rest[None] + amp * (P - rest[None])
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black"); ax.set_facecolor("black"); ax.axis("off")
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    # anchored band: the ring between the domain border and the inner border
    ax.add_patch(Rectangle((D.DOM_LO, D.DOM_LO), D.DOM, D.DOM, fill=False, ec="#3a6", lw=0.8, ls="--"))
    if bnd is not None:
        bw = 0.06
        ax.add_patch(Rectangle((D.DOM_LO + bw, D.DOM_LO + bw), D.DOM - 2 * bw, D.DOM - 2 * bw,
                               fill=False, ec="#3a6", lw=0.6, ls=":"))
    if real_disp is not None:
        Rr = rest[None] + amp * real_disp[:, idx]
        rsegs = np.stack([Rr[:-1], Rr[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(list(rsegs), colors=(0.2, 1.0, 0.2, 0.7), linewidths=1.0))
    segs = np.stack([A[:-1], A[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    age = np.tile(np.linspace(0, 1, A.shape[0] - 1), A.shape[1])
    col = np.zeros((age.size, 4), np.float32); col[:, 0] = 1.0; col[:, 3] = 0.15 + 0.85 * age
    ax.add_collection(LineCollection(list(segs), colors=col, linewidths=1.0))
    ax.scatter(A[0, :, 0], A[0, :, 1], s=9, c="red", edgecolors="black", linewidths=0.3)
    ax.set_title(f"trajectories (amp x{amp:.0f}, sim red / real green; band dashed)", color="#ccc")
    fig.savefig(out, dpi=130, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def render_params(H, out, scalars=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sm = H.fields["stiffness_map"].grid[0].cpu().numpy()
    dr = H.fields["direction"].grid.cpu().numpy()
    ang = (np.arctan2(dr[1], dr[0]) % np.pi)
    fig, axs = plt.subplots(1, 3, figsize=(17, 6), facecolor="black")

    def panel(ax, m, cmap, title, **kw):
        ax.set_facecolor("black"); im = ax.imshow(m.T, origin="lower", cmap=cmap, **kw)
        ax.set_title(title, color="#ccc"); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=7); plt.setp(cb.ax.get_yticklabels(), color="white")

    panel(axs[0], sm, "viridis", "stiffness map (-> youngs)")
    panel(axs[1], ang, "twilight", "direction angle (mod pi)")
    ax = axs[2]; ax.set_facecolor("black"); nx = dr.shape[1]; s = max(1, nx // 26)
    yy, xx = np.meshgrid(np.arange(0, nx, s), np.arange(0, nx, s), indexing="ij")
    ax.imshow(sm.T, origin="lower", cmap="gray", alpha=0.4)
    ax.quiver(xx, yy, dr[0, ::s, ::s].T, dr[1, ::s, ::s].T, color="red", pivot="mid", scale=26)
    ax.set_title("direction field over stiffness", color="#ccc"); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    if scalars:
        fig.text(0.5, 0.02, "  ".join(f"{k}={v}" for k, v in scalars.items()), color="#9f9",
                 fontsize=9, family="monospace", ha="center")
    fig.savefig(out, dpi=120, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def render_fields(pos, roll, mu, out, fps=24, stride=2):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    plt.rcParams["savefig.facecolor"] = "black"
    T, N = pos.shape[0], pos.shape[1]; S = roll.fields()
    strain = np.abs(S["Jp"] - 1.0); mu_b = np.broadcast_to(mu[None], (T, N))
    pc = lambda a, q: max(float(np.nanpercentile(a, q)), 1e-9)
    panels = [("activation", S["act"], "hot", 0.0, pc(S["act"], 99)),
              ("stiffness mu", mu_b, "viridis", float(mu.min()), float(mu.max())),
              ("|v| velocity", S["spd"], "viridis", 0.0, pc(S["spd"], 99)),
              ("C (vel jacobian)", S["Cn"], "viridis", 0.0, pc(S["Cn"], 99)),
              ("F (deformation)", S["Fn"], "coolwarm", 1.34, 1.54),
              ("Jp (det F)", S["Jp"], "viridis", 0.85, 1.15),
              ("strain |J-1|", strain, "hot", 0.0, pc(strain, 99)),
              ("momentum |v|", S["spd"], "viridis", 0.0, pc(S["spd"], 99))]
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), facecolor="black"); axs = axs.ravel(); arts = []
    for k, (title, arr, cmap, vmin, vmax) in enumerate(panels):
        ax = axs[k]; ax.set_facecolor("black"); ax.set_title(title, color="#ccc")
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        sc = ax.scatter(pos[0][:, 0], pos[0][:, 1], c=arr[0], s=2, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=6); plt.setp(cb.ax.get_yticklabels(), color="white")
        arts.append((sc, arr))
    plt.tight_layout()
    w = FFMpegWriter(fps=fps, bitrate=5000)
    with w.saving(fig, out, dpi=100):
        for t in range(0, T, stride):
            for sc, arr in arts:
                sc.set_offsets(pos[t]); sc.set_array(arr[t])
            w.grab_frame()
    plt.close(fig)
    print(f"saved {out}  ({os.path.getsize(out)/1e6:.1f} MB, MPM-style fields)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_frames", type=int, default=0)
    ap.add_argument("--anchor", action="store_true", help="Dirichlet-anchor the outer band to the real data")
    ap.add_argument("--bwidth", type=float, default=0.06)
    args = ap.parse_args()

    spec = load(resolve_config(args.spec)[0])
    if args.n_frames:
        spec.n_frames = args.n_frames
    print(f"=== cardio_mpm forward: {spec.name} (dev={args.device}, n_frames={spec.n_frames}, anchor={args.anchor}) ===")
    roll = Roll(spec.n_frames, anchor=args.anchor, bwidth=args.bwidth)
    H, out = E.run(spec, device=args.device, on_frame=roll)
    pos = np.stack(roll.pos)                                           # [T, N, 2] post-anchor
    mu = H.level("mpm_particle").mu.detach().cpu().numpy()
    real_disp = roll.real_disp.cpu().numpy(); bnd = roll.bnd.cpu().numpy()
    r2 = interior_r2(pos, real_disp, bnd)
    print(f"  particles={pos.shape[1]} frames={pos.shape[0]} disp_max={np.abs(pos - pos[0]).max():.5f} "
          f"band_nodes={int(bnd.sum())} interior_R2(vs real)={r2:+.3f}")

    d = os.path.join(HERE, "archive", spec.name); os.makedirs(d, exist_ok=True)
    idx = grid_sample_idx(pos[0], n=12)
    scal = {k: v for o in spec.operators if o.op in ("pulse_to_contraction", "pacemaker", "mpm_drag", "apply_material_map")
            for k, v in o.params.items() if k in ("amplitude", "period", "duration", "k", "min", "max")}
    scal["R2"] = f"{r2:+.3f}"
    render_traj(pos, idx, os.path.join(d, f"{spec.name}_trajectories.png"),
                real_disp=real_disp, bnd=bnd)
    render_params(H, os.path.join(d, f"{spec.name}_params.png"), scalars=scal)
    render_fields(pos, roll, mu, os.path.join(d, f"{spec.name}_fields.mp4"))
    print(f"  archived -> {d}")


if __name__ == "__main__":
    main()
