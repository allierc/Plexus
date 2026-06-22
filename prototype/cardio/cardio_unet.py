#!/usr/bin/env python
"""Training process, Experiment 3: a small UNet that maps the microscope image to a
MECHANICAL TENSOR (per-node stiffness, gain, fibre), instead of free-fitting 3xN
per-node values from scratch.

    microscope frame  --UNet-->  (stiffness, gain, fibre)  --forward mechanics-->
    predicted trajectory  --MSE vs target-->  backprop into the UNet.

The UNet is an image-conditioned, amortized prior: mechanical properties are physically
tied to visible tissue texture (density/alignment), so a conv net can regress them, and
the whole chain stays differentiable. `pulse_field` is GIVEN; mechanics scalars
(beta,k_anchor,gamma,aniso) are GIVEN; only the property MAPS are produced by the UNet.

Sanity setup: derive ground-truth maps FROM the image (stiffness~texture contrast,
fibre~local gradient orientation, gain~intensity), roll them forward to a GT trajectory,
then train UNet(image)->maps to reproduce that trajectory. Recovering image-derived maps
end-to-end validates the pipeline before training against the real trajectory.

Archives to archive/train_03_unet_mechanical/ (comparison PNG + run.log).

Run:
    cd /workspace/Plexus
    PYTHONPATH=src .../python prototype/cardio/cardio_unet.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import cardio_stage2 as C            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TIF = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif"
GRID = (40, 40)
T = 160
DT = 0.1
MECH = {"beta": 0.4, "k_anchor": 0.1, "gamma": 0.3, "aniso": 0.85}   # GIVEN scalars
K_LO, K_HI, G_LO, G_HI = 0.6, 1.8, 0.4, 1.6


# --------------------------------------------------------------------------- #
#  Small UNet : 1 image channel -> 4 maps (stiffness, gain, fcos, fsin)
# --------------------------------------------------------------------------- #
class UNet(nn.Module):
    def __init__(self, ch=16, out=4):
        super().__init__()
        def cbr(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU())
        self.e1 = cbr(1, ch); self.e2 = cbr(ch, ch * 2)
        self.b = cbr(ch * 2, ch * 2)
        self.d2 = cbr(ch * 4, ch); self.d1 = cbr(ch * 2, ch)
        self.head = nn.Conv2d(ch, out, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        b = self.b(F.max_pool2d(e2, 2))
        # upsample to the skip tensor's exact size (handles odd/non-power-of-2 inputs e.g. 137)
        d2 = self.d2(torch.cat([F.interpolate(b, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        return self.head(d1)


def maps_from_unet(out):
    """[1,4,H,W] -> per-node stiffness, gain, fibre (flattened)."""
    o = out[0]
    stiff = K_LO + (K_HI - K_LO) * torch.sigmoid(o[0])
    gain = G_LO + (G_HI - G_LO) * torch.sigmoid(o[1])
    fiber = torch.atan2(o[3], o[2])
    return stiff.reshape(-1), gain.reshape(-1), fiber.reshape(-1)


# --------------------------------------------------------------------------- #
#  Image + ground-truth maps derived from it
# --------------------------------------------------------------------------- #
def load_image(shape):
    import tifffile
    img = np.asarray(tifffile.TiffFile(TIF).pages[0].asarray()).astype(np.float32)
    t = torch.tensor(img)[None, None]
    t = F.interpolate(t, size=shape, mode="area")[0, 0]
    t = (t - t.mean()) / (t.std() + 1e-6)
    return t                                              # [H,W], standardized


def gt_maps_from_image(img):
    """A plausible causal link image->mechanics (for the sanity target)."""
    sm = img[None, None]
    k = torch.ones(1, 1, 5, 5) / 25
    blur = F.conv2d(sm, k, padding=2)[0, 0]
    stiff = K_LO + (K_HI - K_LO) * torch.sigmoid(blur)
    gain = G_LO + (G_HI - G_LO) * torch.sigmoid(-blur)
    gy, gx = torch.gradient(blur)
    fiber = torch.atan2(gy, gx)
    return stiff.reshape(-1), gain.reshape(-1), fiber.reshape(-1)


# --------------------------------------------------------------------------- #
#  Differentiable forward: maps -> trajectory (mechanics scalars + pulse GIVEN)
# --------------------------------------------------------------------------- #
def setup(shape):
    pat = C._patterns({"pattern_wavelength": 6})
    H = C.build_tissue(shape, pat); C.GridGraph({"neighbours": 8}).forward(H)
    lvl = H.level("tissue_particle")
    i, j = lvl.edge_index
    d = lvl.X[j] - lvl.X[i]; L0 = d.norm(dim=1).clamp(min=1e-6)
    edir = d / L0[:, None]
    # base activation sequence from the GIVEN pulse + nagumo (gain=1), reused every step
    lvl.state[:, 4] = -1.1994; lvl.state[:, 5] = (-1.1994 + 0.7) / 0.8
    pulse = C.PulseField({"period": 180, "dur": 6, "amp": 2.0, "t_start": 1})
    nag = C.ExcitableNagumo({"D": 2.0, "a": 0.7, "b": 0.8, "eps": 0.3, "dt": DT})
    sig = C.SignalToMpmForce({"theta": 0.0, "eta": 0.3})
    base = []
    for t in range(T):
        H.tick = t; pulse.forward(H); nag.forward(H)
        u = lvl.get("u").squeeze(1)
        base.append(torch.sigmoid((u - 0.0) / 0.3).detach().clone())   # gain=1 activation
    return (lvl.edge_index, L0.detach(), edir.detach(), lvl.X.clone().detach()), torch.stack(base)


def forward_traj(stiff, gain, fiber, geo, base_act):
    edge_index, L0, edir, X = geo
    i, j = edge_index
    kedge = 0.5 * (stiff[i] + stiff[j])
    fvec = torch.stack([torch.cos(fiber[i]), torch.sin(fiber[i])], 1)
    align = (edir * fvec).sum(1) ** 2
    ani = (1 - MECH["aniso"]) + MECH["aniso"] * align
    dt = DT / 4
    pos = X.clone(); preds = []
    for t in range(base_act.shape[0]):
        act = gain * base_act[t]
        a_edge = 0.5 * (act[i] + act[j])
        Lt = L0 * (1.0 - MECH["beta"] * a_edge * ani)
        for _ in range(4):
            d = pos[j] - pos[i]; L = d.norm(dim=1).clamp(min=1e-6)
            fv = (kedge * (L - Lt) / L)[:, None] * d
            Fc = torch.zeros_like(pos).index_add(0, i, fv) + MECH["k_anchor"] * (X - pos)
            pos = pos + (dt / MECH["gamma"]) * Fc
        preds.append(pos)
    return torch.stack(preds)


# --------------------------------------------------------------------------- #
def main(n_iter=150, lr=3e-3):
    print("=== Experiment 3: UNet image -> mechanical tensor (end-to-end) ===")
    img = load_image(GRID)
    geo, base_act = setup(GRID)
    sk_gt, gn_gt, fb_gt = gt_maps_from_image(img)
    gt_traj = forward_traj(sk_gt, gn_gt, fb_gt, geo, base_act).detach()
    print(f"  image {tuple(img.shape)}  GT traj {tuple(gt_traj.shape)}  (maps derived from the microscope frame)")

    net = UNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    x = img[None, None]
    hist = []
    for it in range(n_iter):
        opt.zero_grad()
        sk, gn, fb = maps_from_unet(net(x))
        pred = forward_traj(sk, gn, fb, geo, base_act)
        loss = (pred - gt_traj).pow(2).mean()
        loss.backward(); opt.step()
        if it % 25 == 0 or it == n_iter - 1:
            hist.append((it, loss.item())); print(f"  it {it:4d}  traj MSE {loss.item():.3e}")

    with torch.no_grad():
        sk, gn, fb = maps_from_unet(net(x))
        pred = forward_traj(sk, gn, fb, geo, base_act)
        map_err = {"stiffness": (sk - sk_gt).abs().mean().item(),
                   "gain": (gn - gn_gt).abs().mean().item(),
                   "fibre_cos": (torch.cos(fb) - torch.cos(fb_gt)).abs().mean().item()}
    print(f"  final traj MSE {((pred-gt_traj)**2).mean():.3e}  map MAE {map_err}")

    d = os.path.join(HERE, "archive", "train_03_unet_mechanical"); os.makedirs(d, exist_ok=True)
    _figure(img, (sk_gt, gn_gt, fb_gt), (sk, gn, fb), gt_traj, pred, GRID,
            os.path.join(d, "unet_image_to_mechanics.png"))
    with open(os.path.join(d, "run.log"), "w") as f:
        f.write("experiment=UNet image->mechanical tensor (end-to-end through forward mechanics)\n")
        f.write(f"grid={GRID} T={T}\ngiven: pulse_field, mechanics scalars {MECH}\n")
        f.write(f"learned: UNet -> (stiffness,gain,fibre) maps\n")
        f.write(f"final_traj_mse={float(((pred-gt_traj)**2).mean()):.3e}\nmap_mae={map_err}\n")
        f.write(f"loss_history={hist}\n")
    print(f"  archived -> {d}")


def _figure(img, gt, pred, gt_traj, pred_traj, shape, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Hy, Wx = shape
    sk_g, gn_g, fb_g = (m.detach().cpu().numpy().reshape(Hy, Wx) for m in gt)
    sk_p, gn_p, fb_p = (m.detach().cpu().numpy().reshape(Hy, Wx) for m in pred)
    fig, axs = plt.subplots(3, 4, figsize=(15, 11))
    axs[0, 0].imshow(img.cpu(), cmap="gray"); axs[0, 0].set_title("microscope frame (UNet input)")
    for ax, m, t, cm in [(axs[0, 1], sk_g, "GT stiffness", "viridis"),
                         (axs[0, 2], gn_g, "GT gain", "magma"),
                         (axs[0, 3], fb_g % np.pi, "GT fibre", "twilight"),
                         (axs[1, 1], sk_p, "UNet stiffness", "viridis"),
                         (axs[1, 2], gn_p, "UNet gain", "magma"),
                         (axs[1, 3], fb_p % np.pi, "UNet fibre", "twilight")]:
        ax.imshow(m, cmap=cm); ax.set_title(t)
    axs[1, 0].axis("off")
    # trajectory comparison (10x10, amplified)
    ii = np.linspace(0, Hy - 1, 10).round().astype(int); jj = np.linspace(0, Wx - 1, 10).round().astype(int)
    sel = (ii[:, None] * Wx + jj[None, :]).ravel()
    for ax, traj, col, ttl in [(axs[2, 1], gt_traj, "deepskyblue", "GT trajectory"),
                               (axs[2, 2], pred_traj, "lime", "UNet trajectory"),
                               (axs[2, 3], None, None, "overlay")]:
        ax.set_facecolor("black"); ax.set_xlim(-.15, 1.15); ax.set_ylim(-.15, 1.15); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl)
        for tr, cc in ([(gt_traj, "deepskyblue"), (pred_traj, "lime")] if traj is None else [(traj, col)]):
            P = tr.detach().cpu().numpy()[:, sel]; rest = P[0]; P = rest[None] + 4 * (P - rest[None])
            for m in range(P.shape[1]):
                ax.plot(P[:, m, 0], P[:, m, 1], c=cc, lw=0.6)
    axs[2, 0].axis("off")
    for a in axs.flat:
        if a.images or a.lines:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Experiment 3: UNet maps the microscope image to a mechanical tensor", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
