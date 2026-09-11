"""model -- the cardiac sheet as ACTIVE STRAIN on the differentiable Plexus MPM engine.

Nothing in src/plexus is edited. The spec below is stock operators; the contraction is injected
the way `plexus.morph` injects its growth field -- through `engine.run(on_frame=...)`, functionally,
so the tape survives -- and the per-cell stiffness leaf is bound at tick 0 the same way.

THE MODEL. Every particle p belongs to one measured cell j = cid[p]. Once a frame,

    F_p  <-  ( I + c_j(t) f_j f_j^T ) F_p            f_j = (cos phi_j, sin phi_j), the fibre axis

with c_j(t) chosen so that the accumulated active part is EXACTLY  F_a,j(t) = I + gamma(t) g_j f_j f_j^T:

    c_j(t) = ( gamma(t) - gamma(t-1) ) g_j  /  ( 1 + gamma(t-1) g_j )

THE SIGN. The engine's F is the ELASTIC deformation gradient: stress is computed from it. Writing
F <- (I + c f f^T) F tells the constitutive law the material is stretched along f beyond its rest
length, and it pulls itself shorter along f to relax -- that is contraction. The other sign,
F <- (I - c f f^T) F, is morph's: it declares the material compressed, so it GROWS along f. Measured
with the wrong sign: per-cell axis agreement with the recording -0.85 (i.e. perpendicular),
expansion twice the shortening. One character.

gamma(t) in [0,1] is ONE clock for the whole sheet (the contraction is synchronous in the
recording: onset-vs-position r ~ 0), g_j the fibre shortening of cell j at full activation
(dimensionless strain), phi_j its axis, E_j its Young's modulus. Rank-1 updates along the same axis
commute, so when gamma returns to 0 the active part returns to I exactly -- the model rests where
the tissue rests, and no warm-up is needed (the recording starts every beat from rest).

Why F on the left, in the spatial frame: morph does the same, and at the strains here (a few per
cent) left and right differ at second order. Why a body-force model is NOT used: `active_force` is
force ~ grad(activation), so a UNIFORM activation gives zero force -- the old campaign could only
move a synchronous sheet by painting a Gaussian bump, and its per-cell maps then measured that bump
(STATUS s10.4). Active strain has no such term.

Parameters (all torch leaves):  g [C]   phi [C]   logE [C]   clock [4] = (t0, log tau_r, log dur, log tau_d)
    gamma(t) = (s(t) - s(0)) / (1 - s(0)),   s(t) = sigmoid((t - t0)/tau_r) * sigmoid((t0 + dur - t)/tau_d)
so gamma(0) = 0 exactly: the window starts at rest, and the sheet's declared rest (F = I) IS the
clock's zero. (Measured before this normalisation: a clock at 0.256 on frame 0 left the sheet
expanded by 0.256 g along every fibre once gamma returned to 0 -- 64% of the peak strain, and it
looked like a model that does not rest.)

The substrate: the sheet sits on a 15 kPa gel, attached, not floating. `mpm_anchor` with
`applies_to: substrate` is a spring a_p = kappa (x_rest - x_p) on every particle -- an elastic
foundation. It sets the length over which one cell's contraction is felt, sqrt(E / (rho kappa)),
and the time the sheet takes to settle, ~drag / kappa. Without it the sheet-wide mode needs ~90
frames to relax under drag alone (k L^2 rho / E at L = 0.7), longer than the beat.
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_TIF = os.path.join(HERE, "data", "cells_2560.tif")
DOM_LO, DOM_HI = 0.15, 0.85


def build_spec(n_grid=128, per_parent=100, n_frames=50, dt=0.002, sub=2e-4, n_cells=472,
               youngs=80.0, drag_k=30.0, anchor_k=0.0, differentiable=True, compile=True,
               wall_damp=0.5, wall_contact=0.06, a_max=200.0, label_tif=LABEL_TIF,
               name="cardio_strain", seed=0, density=1.0):
    """A 2D sheet of `n_cells` measured cells, `per_parent` material points each, no forces of
    its own: the contraction comes from `on_frame`. `implementation: differentiable` names the
    operator bodies that rebind instead of writing in place (what autograd needs)."""
    impl = dict(implementation="differentiable") if differentiable else {}
    anchor = ([dict(op="mpm_anchor", at="mpm_particle", k=float(anchor_k), applies_to="substrate")]
              if anchor_k > 0 else [])
    return dict(
        general=dict(name=name, seed=seed, n_frames=n_frames, dt=dt, record_cap=3,
                     boundary="wall", dim=2),
        sets=dict(
            tissue=dict(n=1, start=[[0.5, 0.5]],
                        types=dict(sheet=dict(block=[DOM_LO, DOM_LO, DOM_HI, DOM_HI],
                                              fraction=1.0, youngs=youngs))),
            cell=dict(parent="tissue", per_parent=n_cells, radius=0.02,
                      types=dict(myocyte=dict(fraction=1.0, youngs=youngs))),
            mpm_particle=dict(density=float(density), parent="cell", per_parent=per_parent)),
        fields=dict(mpm_grid=dict(frame="mpm_grid", n_grid=n_grid),
                    cells=dict(frame="label_image", source=label_tif)),
        operators=[
            dict(op="seed_from_segmentation", at="mpm_particle", **{"from": "cells"},
                 youngs_min=youngs, youngs_max=youngs),
            dict(op="drag", at="mpm_particle", k=drag_k, emit="mpm_acceleration"),
            *anchor,
            dict(op="mpm_strain", at="mpm_particle", **impl),
            dict(op="mpm_scatter", at="mpm_particle", to="mpm_grid", drag=0.0, a_max=a_max, **impl),
            dict(op="mpm_grid_update", at="mpm_grid", wall_contact=wall_contact,
                 wall_damp=wall_damp, **impl),
            dict(op="mpm_gather", at="mpm_particle", **{"from": "mpm_grid"},
                 wall_contact=wall_contact, wall_damp=wall_damp, **impl)],
        schedule=["seed_from_segmentation", "drag", *(["mpm_anchor"] if anchor else []),
                  dict(substep_dt=sub, compile=bool(compile),
                       steps=["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"])],
        plotting={})


def sim_label_tif(sim):
    """The label image a loaded spec reads (so a rollout can re-seed onto it)."""
    for f in getattr(sim, "fields", []) or []:
        src = getattr(f, "params", {}).get("source") if hasattr(f, "params") else None
        if src and str(src).endswith(".tif"):
            return src
    return LABEL_TIF


def load_sim(raw):
    from plexus import operators  # noqa: F401  -- importing registers every operator
    from plexus.schema import load
    f = os.path.join(tempfile.mkdtemp(prefix="cardio_strain_"), "spec.yaml")
    yaml.safe_dump(raw, open(f, "w"), sort_keys=False)
    return load(f)


class Params:
    """The learnables, as leaves. `E0` in the spec's stress units (the old recipes used 80)."""

    def __init__(self, n_cells, device, g0=0.03, phi0=None, E0=80.0,
                 t0=4.0, tau_r=1.5, dur=8.0, tau_d=3.0, nu=0.3, clock_mode="sigmoid", n_frames=0):
        C = n_cells
        self.nu = float(nu)                 # Poisson ratio of the sheet (plane strain), fixed
        # THE CLOCK'S FORM. `sigmoid` is the 4-number rise x fall; `free` is one number per frame
        # (gamma(0) pinned at 0), because the recording's relaxation is not a sigmoid: it drops
        # 0.015 -> 0.005 in four frames and then trails, and the first live fit stretched the
        # sigmoid's tail to compromise (tau_d 2.4 -> 4.8) and lost the relaxation phase entirely
        # (residual/signal > 1 after frame 24). A free clock is 57 numbers shared by 472 cells.
        self.clock_mode = clock_mode
        self.shift = 0.0                    # whole-frame re-alignment used only by the scorer
        # PER-CELL TIMING. The recording's beat is 88% one temporal mode and 6.7% a second whose
        # time course is early-negative / late-positive: cells lead or lag. Per-cell peak frames
        # spread p10 7 / p50 13 / p90 21 -- 14 frames (0.6 s) across the sheet, with no spatial
        # gradient (so not a wave). delay_j shifts cell j's clock: gamma_j(t) = gamma(t - delay_j).
        # Zero by default, which is the one-clock model exactly.
        self.delay = torch.zeros(C, device=device, requires_grad=True)
        # SECOND ACTIVE AXIS. g2_j is the active strain ACROSS the fibre (positive = thickening) at
        # full activation, so the active part is I + gamma (g f f^T + g2 f_perp f_perp^T). Zero = the
        # rank-1 model. The recording is area-preserving at the cell level (expansion/shortening
        # 1.11); an isotropic sheet with nu = 0.3 reaches 0.95 with rank-1 shortening alone.
        self.g2 = torch.zeros(C, device=device, requires_grad=True)
        # PER-CELL RELAXATION. The recording's second temporal mode (6.7% of variance) is early-negative /
        # late-positive: cells differ not only in WHEN they start (delay) but in how long they stay
        # contracted. logtau_j scales cell j's decay time: tau_d,j = tau_d * exp(logtau_j). Zero = shared.
        self.logtau = torch.zeros(C, device=device, requires_grad=True)
        # ... and its own rise time and plateau duration: with delay and logtau this gives every cell
        # its own 4-number excitation-contraction time course around the shared one (the recording's
        # temporal modes 2-3 are per-cell time-course differences; rank ceiling 0.94 / 0.97).
        self.logtr = torch.zeros(C, device=device, requires_grad=True)
        self.logdur = torch.zeros(C, device=device, requires_grad=True)
        self.g = torch.full((C,), float(g0), device=device, requires_grad=True)
        phi = (torch.zeros(C, device=device) if phi0 is None
               else torch.as_tensor(phi0, device=device, dtype=torch.float32).clone())
        self.phi = phi.requires_grad_(True)
        self.logE = torch.full((C,), math.log(E0), device=device, requires_grad=True)
        self.clock = torch.tensor([t0, math.log(tau_r), math.log(dur), math.log(tau_d)],
                                  device=device, requires_grad=True)
        if clock_mode == "free":
            with torch.no_grad():
                init = torch.stack([self._gamma_sigmoid(t) for t in range(max(n_frames, 2))])
            self.gfree = init.clone().requires_grad_(True)

    def leaves(self):
        clk = self.gfree if self.clock_mode == "free" else self.clock
        return dict(g=self.g, phi=self.phi, logE=self.logE, clock=clk, delay=self.delay, g2=self.g2,
                    logtau=self.logtau, logtr=self.logtr, logdur=self.logdur)

    def _s(self, t, logtau=None, percell=False):
        t0, tr, d, td = self.clock[0], self.clock[1].exp(), self.clock[2].exp(), self.clock[3].exp()
        if logtau is not None:
            td = td * logtau.exp()                     # per-cell decay time, [C]
        if percell:
            tr = tr * self.logtr.exp(); d = d * self.logdur.exp()     # per-cell rise and plateau
        t = torch.as_tensor(t, device=self.clock.device, dtype=self.clock.dtype)
        return torch.sigmoid((t - t0) / tr) * torch.sigmoid((t0 + d - t) / td)

    def _gamma_sigmoid(self, t):
        s0 = self._s(0)
        return (self._s(t) - s0) / (1.0 - s0)

    def gamma_cells(self, t):
        """[C] clock value of every cell at frame t, its own delay applied (sigmoid clock only;
        a free clock has no per-cell timing). gamma_j(0) = 0 is kept exactly by the same
        normalisation, so a delayed cell still starts at rest."""
        if self.clock_mode == "free":
            return self.gamma(t).expand(self.delay.shape[0])
        tt = float(t) - self.shift - self.delay
        s0 = self._s(0.0 - self.delay, self.logtau, percell=True)
        return ((self._s(tt, self.logtau, percell=True) - s0) / (1.0 - s0)).clamp(min=0.0)

    def cell_clock_penalty(self):
        """Mean squared log-scales of the per-cell time-course numbers (0 = the shared clock)."""
        return (self.logtau.pow(2) + self.logtr.pow(2) + self.logdur.pow(2)).mean()

    def gamma(self, t):
        if self.clock_mode == "free":
            i = int(round(t - self.shift))
            if i <= 0:
                return self.gfree[0] * 0.0                      # gamma(0) = 0, on the tape
            return self.gfree[min(i, self.gfree.shape[0] - 1)]
        return self._gamma_sigmoid(t - self.shift)

    def smoothness(self):
        """Sum of squared frame-to-frame steps of a free clock (0 for the sigmoid)."""
        if self.clock_mode != "free":
            return torch.zeros((), device=self.g.device)
        return (self.gfree[1:] - self.gfree[:-1]).pow(2).sum()

    def lame(self, E):
        nu = self.nu
        return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

    def state_dict(self):
        d = {k: v.detach().cpu().numpy() for k, v in self.leaves().items()}
        d["clock_mode"] = np.array(self.clock_mode)
        return d

    def load(self, d):
        with torch.no_grad():
            for k, v in d.items():
                if k == "clock_mode":
                    continue
                if k in ("delay", "g2", "logtau", "logtr", "logdur") and np.asarray(v).shape != tuple(self.delay.shape):
                    continue
                if k == "clock" and self.clock_mode == "free":
                    self.gfree = torch.as_tensor(np.asarray(v), device=self.g.device,
                                                 dtype=torch.float32).clone().requires_grad_(True)
                else:
                    getattr(self, k).copy_(torch.as_tensor(np.asarray(v)))


def lattice_positions(H, label_tif, per_parent, device):
    """Regular placement: each cell's `per_parent` particles on a square lattice inside its own mask,
    spacing sqrt(area / per_parent), instead of the seed's uniform-random pixels. Same cell ids,
    same counts; only WHERE the material points sit changes. Returns [N,2] world positions ordered
    like the particle set (cell j's particles are contiguous in parent order)."""
    import tifffile
    img = tifffile.imread(label_tif)[::-1, :]                    # rows = world y (LabelImageField's flip)
    n = img.shape[0]
    q = H.level("mpm_particle"); cid = q.cell_id.long().cpu().numpy(); N = cid.shape[0]
    pos = np.zeros((N, 2), np.float32)
    ys, xs = np.nonzero(img)
    labs = img[ys, xs]
    order = np.argsort(labs, kind="stable"); labs, ys, xs = labs[order], ys[order], xs[order]
    starts = np.searchsorted(labs, np.arange(1, labs.max() + 2))
    rng = np.random.default_rng(0)
    for j in range(1, labs.max() + 1):
        members = np.nonzero(cid == j)[0]
        if members.size == 0:
            continue
        py, px = ys[starts[j - 1]:starts[j]], xs[starts[j - 1]:starts[j]]
        if py.size == 0:
            continue
        area = py.size; s_px = np.sqrt(area / members.size)      # lattice spacing in canvas pixels
        # lattice anchored on the cell's bounding box, kept to pixels inside the mask
        gy = np.arange(py.min() + s_px / 2, py.max() + 1, s_px); gx = np.arange(px.min() + s_px / 2, px.max() + 1, s_px)
        GY, GX = np.meshgrid(gy, gx, indexing="ij"); GY, GX = GY.ravel(), GX.ravel()
        iy, ix = np.clip(GY.round().astype(int), 0, n - 1), np.clip(GX.round().astype(int), 0, n - 1)
        inside = img[iy, ix] == j
        cand = np.stack([GX[inside], GY[inside]], 1)
        if cand.shape[0] >= members.size:                      # thin evenly to the count
            sel = np.linspace(0, cand.shape[0] - 1, members.size).round().astype(int)
            pts = cand[sel]
        else:                                                  # top up with random mask pixels
            extra = rng.choice(py.size, members.size - cand.shape[0], replace=True)
            pts = np.concatenate([cand, np.stack([px[extra], py[extra]], 1).astype(float)])
        pos[members] = (pts + 0.5) / n                         # canvas pixel -> world [0,1)
    return torch.as_tensor(pos, device=device)


def rollout(sim, params, device, n_cells, grad=True, prescribe=None, keep_pos=False,
            progress=False, lattice=False, label_tif=None):
    """Run one window of `sim.n_frames` frames from rest. Returns per-frame per-cell affine maps.

    prescribe: optional (band [N] bool, u_band [T+1, N, 2]) -- particles in `band` are pinned
    each tick to rest + u_band[tick] (the recording's own motion, interpolated per cell), which is
    how the unobserved tissue beyond the crop enters. Velocities in the band are zeroed.
    Returns dict(A [T+1,C,2,2], u [T+1,C,2], X0 [N,2], cid [N], pos [T+1,N,2] if keep_pos, J).
    """
    from plexus import engine, operators  # noqa: F401
    from recording import cell_affine

    dev = torch.device(device)
    eye = torch.eye(2, device=dev)
    box = {}
    A_list, u_list, pos_list = [], [], []

    def cb(H, tick):
        q = H.level("mpm_particle")
        if tick == 0:
            cid = q.cell_id.long()
            box["cid"] = cid
            if lattice:
                # the seed's random pixels are replaced by a per-cell lattice BEFORE the rest state is
                # taken (a state write at tick 0, as morph does; the material is at rest either way)
                tif = label_tif or sim_label_tif(sim)
                newpos = lattice_positions(H, tif, None, q.state.device)
                p0, p1 = q.state_schema["pos"]
                S = q.state.clone(); S[:, p0:p1] = newpos; q.state = S
            box["X0"] = q.get("pos").detach().clone()
            f = torch.stack([torch.cos(params.phi), torch.sin(params.phi)], 1)   # [C,2]
            fp = torch.stack([-torch.sin(params.phi), torch.cos(params.phi)], 1)
            box["ff"] = (f[:, :, None] * f[:, None, :])[cid - 1]               # [N,2,2]
            box["pp"] = (fp[:, :, None] * fp[:, None, :])[cid - 1]
            box["g"] = params.g[cid - 1]
            box["g2"] = params.g2[cid - 1]
            box["gam_prev"] = params.gamma_cells(0)[cid - 1]
            # the stiffness LEAF, bound after the seed wrote its own per-cell value
            E_p = params.logE.exp()[cid - 1]
            q.mu, q.la = params.lame(E_p)
        else:
            cid = box["cid"]
            gam = params.gamma_cells(tick)[cid - 1]
            c = (gam - box["gam_prev"]) * box["g"] / (1.0 + box["gam_prev"] * box["g"])
            c2 = (gam - box["gam_prev"]) * box["g2"] / (1.0 + box["gam_prev"] * box["g2"])
            box["gam_prev"] = gam
            G = eye + c[:, None, None] * box["ff"] + c2[:, None, None] * box["pp"]
            q.F = torch.bmm(G, q.F)
        if prescribe is not None:
            band, u_band = prescribe
            p0, p1 = q.state_schema["pos"]
            v0, v1 = q.state_schema["vel"]
            S0 = q.state
            S = S0.clone()                          # a fresh tensor: in-place writes below are
            target = box["X0"] + u_band[min(tick, u_band.shape[0] - 1)]   # autograd-safe on it
            m = band[:, None]
            S[:, p0:p1] = torch.where(m, target, S0[:, p0:p1])
            S[:, v0:v1] = torch.where(m, torch.zeros_like(S0[:, v0:v1]), S0[:, v0:v1])
            q.state = S
        x = q.get("pos")
        A, u = cell_affine(x, box["X0"], box["cid"], n_cells)
        A_list.append(A); u_list.append(u)
        if keep_pos:
            pos_list.append(x.detach().cpu().numpy().copy())

    Hh, _ = engine.run(sim, device=device, on_frame=cb, progress=progress, grad=grad)
    q = Hh.level("mpm_particle")
    out = dict(A=torch.stack(A_list), u=torch.stack(u_list), X0=box["X0"], cid=box["cid"],
               J=torch.linalg.det(q.F).detach())
    if keep_pos:
        out["pos"] = np.stack(pos_list)
    return out


def affine_loss(A, u, A_ref, u_ref, w_u=None, w_A=None, frames=None, cells=None):
    """Mean squared mismatch of the per-cell affine maps, each channel scaled by the reference's
    own spread so a 2x2 strain (~1e-2) and a centroid displacement (~1e-3 world) weigh alike.
    `cells` [C] bool restricts everything to those cells: a cell with particles in the prescribed
    edge band has part of its motion dictated, not modelled, so its g is unconstrained by the loss
    (measured: g inflated along the band in round 4); such cells are left out of loss and claim."""
    eye = torch.eye(2, device=A.device)
    if frames is not None:
        A, u, A_ref, u_ref = A[frames], u[frames], A_ref[frames], u_ref[frames]
    if cells is not None:
        A, u, A_ref, u_ref = A[:, cells], u[:, cells], A_ref[:, cells], u_ref[:, cells]
    if w_A is None:
        w_A = 1.0 / ((A_ref - eye) ** 2).mean().clamp(min=1e-12)
    if w_u is None:
        w_u = 1.0 / (u_ref ** 2).mean().clamp(min=1e-12)
    return w_A * ((A - A_ref) ** 2).mean() + w_u * ((u - u_ref) ** 2).mean()


def interior_cells(cid, band, n_cells, max_frac=0.0):
    """[C] bool: cells with at most `max_frac` of their particles in the prescribed band."""
    idx = cid - 1
    ones = torch.ones(cid.shape[0], device=cid.device)
    cnt = torch.zeros(n_cells, device=cid.device).index_add(0, idx, ones).clamp(min=1)
    inb = torch.zeros(n_cells, device=cid.device).index_add(0, idx, band.float())
    return (inb / cnt) <= max_frac


def band_prescription(rec_A, rec_u, X0, cid, band):
    """The recording's per-cell affine motion, evaluated at the band particles' rest positions:
    u_p(t) = (A_j(t) - I)(X_p - Xbar_j) + u_j(t).  Returns [T, N, 2] (zeros off the band)."""
    C = rec_A.shape[1]
    eye = torch.eye(2, device=X0.device)
    idx = cid - 1
    ones = torch.ones(X0.shape[0], device=X0.device)
    cnt = torch.zeros(C, device=X0.device).index_add(0, idx, ones).clamp(min=1)
    Xbar = torch.zeros(C, 2, device=X0.device).index_add(0, idx, X0) / cnt[:, None]
    dX = X0 - Xbar[idx]                                              # [N,2]
    T = rec_A.shape[0]
    out = torch.zeros(T, X0.shape[0], 2, device=X0.device)
    for t in range(T):
        Ap = (rec_A[t] - eye)[idx]                                   # [N,2,2]
        up = torch.einsum("nij,nj->ni", Ap, dX) + rec_u[t][idx]
        out[t] = torch.where(band[:, None], up, torch.zeros_like(up))
    return out
