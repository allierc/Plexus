"""operators_smg -- the mechanism-tree operators the stock palette is missing.

Built so the random bootstrap can REACH every branch (else it only makes clusters). Import this
module BEFORE S.load so the operators register. Kept deliberately MODEST (extract the contract, do
not overfit the source frameworks); specialize later.

  ecm_boundary     boundary-guided confinement/adhesion. Contract: a signed-distance boundary phi(x)
                   (ellipse; circular when aspect=1) and a force  F = -k grad(phi) psi(phi)  that
                   pushes cells that leave Omega back in. A CONSTRAINT/GUIDE, not a growth source:
                   circular -> rounds, anisotropic -> guides elongation. strength=0 == byte-identical no-op.
  growth_field     a STATIC spatial program g(x) in [0,1] (patch/gradient/ring) that gates a local
                   soft-repulsion = LOCALIZED growth pressure (cells push apart where g is high).
  slow_field       growth_field modulated slowly in time (omega).
  growth_gate      growth_field gated by a morphogen field channel (Turing/gray_scott) if present.
  stiffness_field  a spatial Young's-modulus gradient on the MPM material (soft ducts / stiff stroma).
"""
from __future__ import annotations
import math
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator


# ------------------------------------------------------------------ geometry / field helpers
def _ellipse_phi_n(pos, c, a, b, theta):
    """Signed-distance-like phi = rho-1 (phi<0 inside, >0 outside) + outward unit normal grad(phi)."""
    ct, st = math.cos(theta), math.sin(theta)
    d = pos - c
    ux = d[:, 0] * ct + d[:, 1] * st
    uy = -d[:, 0] * st + d[:, 1] * ct
    rho = torch.sqrt((ux / a) ** 2 + (uy / b) ** 2).clamp(min=1e-6)
    phi = rho - 1.0
    gx, gy = (ux / a ** 2) / rho, (uy / b ** 2) / rho          # grad rho in boundary frame
    wx, wy = gx * ct - gy * st, gx * st + gy * ct              # rotate back to world
    n = torch.stack([wx, wy], -1)
    return phi, n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _spatial_gate(pos, mode, center, radius, axis):
    """A static program g(x) in [0,1]: patch (disc), gradient (along axis), ring (annulus)."""
    d = pos - center
    r = d.norm(dim=-1)
    if mode == "gradient":
        proj = d[:, 0] if axis == "x" else d[:, 1] if axis == "y" else r
        g = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    elif mode == "ring":
        g = torch.exp(-((r - radius) / (0.4 * radius + 1e-6)) ** 2)
    else:                                                       # patch
        g = torch.sigmoid((radius - r) / (0.25 * radius + 1e-6))
    return g.clamp(0, 1)


# ------------------------------------------------------------------ ecm_boundary (constraint/guide)
@register_operator("ecm_boundary", level="cell", kind="lateral")
class EcmBoundary(Lateral):
    """F = -strength * grad(phi) * relu(phi + gap):  push cells that leave the boundary back in.
    aspect!=1 (an ELLIPSE) guides elongation; a small `adhesion` band inside lets cells cling.
    strength=0 -> returns exactly zero (byte-identical no-op)."""
    PREDICTION = "velocity"
    SUPPORTED_DIMS = [2]
    PARAM_ROLES = {"strength": "boundary_stiffness", "aspect": "boundary_anisotropy",
                   "radius": "boundary_size", "gap": "band"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params.get("strength", 20.0))
        self.radius = float(params.get("radius", 0.30))
        self.aspect = float(params.get("aspect", 1.0))          # a/b ; 1 = circle
        self.theta = float(params.get("angle", 0.0))
        self.gap = float(params.get("gap", 0.0))                # containment margin (>0 pulls in earlier)
        self.adhesion = float(params.get("adhesion", 0.0))      # mild cell-matrix cling just inside
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        if self.k == 0.0:
            return {}                                            # byte-identical no-op
        lvl = H.level(self.at)
        pos, occ = lvl.get("pos"), lvl.occ
        if pos.shape[-1] != 2:
            return {}
        c = getattr(H, "world_center", None)
        c = pos.new_tensor([0.5, 0.5]) if c is None else pos.new_tensor(c)
        a = self.radius * math.sqrt(self.aspect)
        b = self.radius / math.sqrt(self.aspect)
        phi, n = _ellipse_phi_n(pos, c, a, b, self.theta)
        contain = torch.relu(phi + self.gap)[:, None]            # >0 outside (or within gap) -> push in
        vel = -self.k * n * contain
        if self.adhesion:                                        # band just inside -> mild outward cling
            band = ((phi < 0) & (phi > -0.05)).float()[:, None]
            vel = vel + self.adhesion * n * band
        return {self.at: vel * occ[:, None].float()}


# ------------------------------------------------------------------ gated localized growth
class _GatedGrowth(Lateral):
    """Localized growth = a soft repulsion gated by a program g(x) in [0,1]: cells push apart where g
    is high (local volume increase). Subclasses set how g is computed. Uses the radius_graph edges."""
    PREDICTION = "velocity"
    SUPPORTED_DIMS = [2, 3]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.gain = float(params.get("gain", params.get("rate", 0.5)))
        self.range = float(params.get("range", 0.035))
        self.mode = str(params.get("mode", "patch"))
        self.radius = float(params.get("gate_radius", 0.18))
        self.axis = str(params.get("axis", "x"))
        self.at = params.get("_at", "cell")

    def _gate(self, H, pos):
        c = pos.new_tensor([0.5, 0.5]) if pos.shape[-1] == 2 else pos.mean(0)
        return _spatial_gate(pos[:, :2], self.mode, c[:2], self.radius, self.axis)

    def forward(self, H, mask=None):
        if self.gain == 0.0:
            return {}
        lvl = H.level(self.at)
        pos, occ, ei = lvl.get("pos"), lvl.occ, lvl.edge_index
        g = self._gate(H, pos)
        vel = torch.zeros_like(pos)
        if ei.numel() > 0:
            i, j = ei[0], ei[1]
            d = pos[i] - pos[j]
            dist = d.norm(dim=-1)
            push = (self.range - dist).clamp(min=0.0) * occ[j]
            f = self.gain * (d / dist.clamp(min=1e-6)[:, None]) * push[:, None] * g[i][:, None]
            vel = vel.index_add_(0, i, f)
        return {self.at: vel * occ[:, None].float()}


@register_operator("growth_field", level="cell", kind="lateral")
class GrowthField(_GatedGrowth):
    """Static spatial growth program (patch/gradient/ring)."""


@register_operator("slow_field", level="cell", kind="lateral")
class SlowField(_GatedGrowth):
    """growth_field modulated slowly in time: g(x)*(0.5+0.5 sin(omega t))."""
    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.omega = float(params.get("omega", 1.0))

    def _gate(self, H, pos):
        g = super()._gate(H, pos)
        t = float(getattr(H, "t", getattr(H, "frame", 0)) or 0) * float(getattr(H, "dt", 0.002))
        return g * (0.5 + 0.5 * math.sin(self.omega * t))


@register_operator("growth_gate", level="cell", kind="lateral")
class GrowthGate(_GatedGrowth):
    """growth_field gated by a morphogen FIELD channel (Turing/gray_scott) if present, else prescribed."""
    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.channel = int(params.get("channel", 1))
        self.thresh = float(params.get("thresh", 0.2))

    def _gate(self, H, pos):
        fld = getattr(H, "fields", {})
        grid = fld.get("mpm_grid") if isinstance(fld, dict) else None
        val = getattr(grid, "value", None) if grid is not None else None
        if val is None or val.dim() < 2:                         # no field -> fall back to prescribed
            return super()._gate(H, pos)
        ch = val[..., min(self.channel, val.shape[-1] - 1)] if val.shape[-1] > 1 else val.squeeze(-1)
        n = ch.shape[0]
        ij = (pos[:, :2].clamp(0, 1) * (n - 1)).long()
        samp = ch[ij[:, 0], ij[:, 1]]
        return torch.sigmoid((samp - self.thresh) * 10.0).clamp(0, 1)


# ------------------------------------------------------------------ stiffness_field (MPM)
@register_operator("stiffness_field", level="mpm_particle", kind="lateral")
class StiffnessField(Lateral):
    """Spatial Young's-modulus gradient on the MPM material (soft ducts / stiff stroma). Sets the
    Lame parameters per particle ONCE from position; returns {} (mutates material, no motion delta)."""
    SUPPORTED_DIMS = [2, 3]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.lo = float(params.get("lo", 50.0))
        self.hi = float(params.get("hi", 200.0))
        self.axis = str(params.get("axis", "x"))
        self.at = params.get("_at", "mpm_particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if getattr(lvl, "_stiff_set", False) or not hasattr(lvl, "mu"):
            return {}
        pos = lvl.get("pos")
        if self.axis == "radial":
            proj = (pos[:, :2] - pos[:, :2].mean(0)).norm(dim=-1)
        else:
            proj = pos[:, 0] if self.axis == "x" else pos[:, 1]
        frac = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
        target = self.lo + (self.hi - self.lo) * frac
        base = float(lvl.mu.mean().clamp(min=1e-6))
        scale = (target / (base * 2 + 1e-6)).clamp(0.1, 10.0)    # relative rescale of stiffness
        lvl.mu = (lvl.mu * scale).detach()
        lvl.la = (lvl.la * scale).detach()
        lvl._stiff_set = True
        return {}


@register_operator("chemotax_field", level="cell", kind="lateral")
class ChemotaxField(Lateral):
    """Chemotaxis up a prescribed morphogen gradient: velocity = gain * normalize(grad g(x)).
    A migration bias (the chemotaxis hypothesis), distinct from growth. strength/gain=0 -> no-op."""
    PREDICTION = "velocity"
    SUPPORTED_DIMS = [2]
    PARAM_ROLES = {"gain": "chemotactic_sensitivity"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.gain = float(params.get("gain", 0.5))
        self.mode = str(params.get("mode", "gradient"))
        self.radius = float(params.get("gate_radius", 0.18))
        self.axis = str(params.get("axis", "x"))
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        if self.gain == 0.0:
            return {}
        lvl = H.level(self.at)
        pos, occ = lvl.get("pos"), lvl.occ
        if pos.shape[-1] != 2:
            return {}
        c = pos.new_tensor([0.5, 0.5])
        e = 0.01
        def g(p):
            return _spatial_gate(p, self.mode, c, self.radius, self.axis)
        gx = (g(pos + pos.new_tensor([e, 0])) - g(pos - pos.new_tensor([e, 0]))) / (2 * e)
        gy = (g(pos + pos.new_tensor([0, e])) - g(pos - pos.new_tensor([0, e]))) / (2 * e)
        grad = torch.stack([gx, gy], -1)
        n = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return {self.at: self.gain * n * occ[:, None].float()}


# ------------------------------------------------------------------ tests
def _run(spec, frames=60, seed=0, hook=None):
    import os, sys, copy, tempfile, yaml
    sys.path.insert(0, "/workspace/Plexus/src")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "active_matter2"))
    import plexus.operators, am2_ops, plexus.schema as S       # noqa
    from plexus.engine import run
    spec = copy.deepcopy(spec)
    spec["general"]["seed"] = seed
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(spec, f); f.close()
    sim = S.load(f.name); sim.n_frames = frames
    box = {"aX": []}
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _h(H, frame):
        if hook:
            hook(H, frame)
        if frame == frames:
            lvl = H.level("agent")
            box["aX"] = lvl.get("pos").detach().cpu().numpy()[lvl.occ.detach().cpu().numpy() > 0]
    run(sim, out_path=None, device=dev, on_frame=_h)
    return box["aX"]


def _tests():
    import os, sys, copy
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mechanism_tree as mt
    base = copy.deepcopy(mt.BASE)
    base["general"].pop("init", None); base["sets"]["agent"]["spawn"] = "disc"; base["general"]["n_frames"] = 60

    # TEST 1: strength=0 -> byte-identical no-op
    noop = copy.deepcopy(base)
    noop["operators"].append({"op": "ecm_boundary", "at": "agent", "strength": 0.0})
    noop["schedule"].insert(5, "ecm_boundary")
    a0 = _run(copy.deepcopy(base), 40, seed=1)
    a1 = _run(noop, 40, seed=1)
    ident = a0.shape == a1.shape and np.allclose(a0, a1, atol=0, rtol=0)
    print(f"[T1] ecm strength=0 byte-identical no-op: {'PASS' if ident else 'FAIL'} "
          f"(max|Δ|={0.0 if a0.shape != a1.shape else np.abs(a0-a1).max():.2e})")

    # TEST 2: circular boundary pushes outward cells inward (radius smaller than the disc)
    circ = copy.deepcopy(base)
    circ["operators"].append({"op": "ecm_boundary", "at": "agent", "strength": 40.0, "radius": 0.15, "aspect": 1.0})
    circ["schedule"].insert(5, "ecm_boundary")
    ac = _run(circ, 120, seed=1)
    rc = np.linalg.norm(ac - ac.mean(0), axis=1)
    r_free = np.linalg.norm(a0 - a0.mean(0), axis=1)
    print(f"[T2] circular boundary contains: {'PASS' if rc.max() < r_free.max() else 'FAIL'} "
          f"(max radius {rc.max():.3f} < free {r_free.max():.3f})")

    # TEST 3: an ANISOTROPIC boundary elongates the tissue MORE than a circular one (confine relaxed
    # so the boundary -- not the re-rounding MPM confine -- sets the shape).
    def aspect(P):
        ev = np.sort(np.linalg.eigvalsh(np.cov((P - P.mean(0)).T)))[::-1]
        return (ev[0] / max(ev[1], 1e-9)) ** 0.5

    def ecm_run(aspect_val):
        sp = copy.deepcopy(base)
        for o in sp["operators"]:
            if o["op"] == "mpm_to_agent":
                o["confine"] = 0.3                               # let the boundary set shape
        sp["operators"].append({"op": "ecm_boundary", "at": "agent", "strength": 70.0,
                                "radius": 0.20, "aspect": aspect_val})
        sp["schedule"].insert(5, "ecm_boundary")
        return aspect(_run(sp, 220, seed=1))
    asp_e, asp_c = ecm_run(4.0), ecm_run(1.0)
    print(f"[T3] anisotropic boundary elongates: {'PASS' if asp_e > asp_c + 0.3 else 'FAIL'} "
          f"(ellipse aspect {asp_e:.2f} > circular {asp_c:.2f})")

    # smoke-run the growth / stiffness operators (register + run without crashing)
    for op, extra in [("growth_field", {"gain": 0.5, "mode": "patch"}),
                      ("slow_field", {"gain": 0.5, "omega": 1.0}),
                      ("growth_gate", {"gain": 0.5}),
                      ("stiffness_field", {"lo": 50, "hi": 250})]:
        sp = copy.deepcopy(base)
        at = "mpm_particle" if op == "stiffness_field" else "agent"
        sp["operators"].append({"op": op, "at": at, **extra})
        sp["schedule"].insert(5, op)
        try:
            _run(sp, 40, seed=1)
            print(f"[smoke] {op}: PASS (registers + runs)")
        except Exception as e:
            print(f"[smoke] {op}: FAIL {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    _tests()
