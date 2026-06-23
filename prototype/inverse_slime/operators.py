"""operators.py -- DECOMPOSED learnable Plexus operators (one module per operator).

Instead of one monolithic "slime" model, each registered operator gets its own
differentiable module with its own learnable parameter, mirroring the engine
operator 1:1 -- the Plexus philosophy (operators are the primitives). The smooth
operators are validated EXACT against the engine (see validate_ops.py):

    LearnableDeposit (amount)      set->field   exact
    LearnableDiffuse (rate)        field->field exact
    LearnableDecay   (rate)        field->field exact
    LearnableAdvance (move_speed)  set          exact (||displacement|| = dt*move_speed)
    LearnableSense   (turn_speed, sensor_angle, sensor_dist)  field->set  SURROGATE

`sense` in the engine is non-differentiable (hard torch.where + a random turn), so
LearnableSense is a differentiable *expected-turn* surrogate: bilinear field reads at
the 3 sensors (so sensor geometry gets a gradient) + a smooth gate matching the
engine's "turn toward the stronger side only when the front sensor is the median".
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _u_param(u0):
    """A scalar parameter stored in logit space so sigmoid(theta) stays in [0,1]."""
    u0 = float(min(max(u0, 1e-3), 1 - 1e-3))
    return nn.Parameter(torch.tensor(math.log(u0 / (1 - u0)), dtype=torch.float32))


def _phys(theta, lo, hi):
    return lo + torch.sigmoid(theta) * (hi - lo)


# ---- field operators (EXACT vs engine) ---------------------------------------
class LearnableDeposit(nn.Module):
    def __init__(self, u0, lo, hi):
        super().__init__(); self.t = _u_param(u0); self.lo, self.hi = lo, hi

    def amount(self): return _phys(self.t, self.lo, self.hi)

    def forward(self, grid, pos, nt, R, width=1.0, dt=1.0):
        C, nx, ny = grid.shape
        gx = (pos[:, 0].clamp(0, width - 1e-6) * R).long().clamp(0, nx - 1)
        gy = (pos[:, 1].clamp(0, 1 - 1e-6) * R).long().clamp(0, ny - 1)
        flat = nt.long() * (nx * ny) + gx * ny + gy
        add = torch.zeros(C * nx * ny, device=grid.device, dtype=grid.dtype)
        add = add.index_add(0, flat, (self.amount().to(grid.dtype) * dt).expand(pos.shape[0]))
        return (grid + add.view(C, nx, ny)).clamp(max=1.0)


class LearnableDiffuse(nn.Module):
    def __init__(self, u0, lo, hi):
        super().__init__(); self.t = _u_param(u0); self.lo, self.hi = lo, hi

    def rate(self): return _phys(self.t, self.lo, self.hi)

    def forward(self, g, dt=1.0):
        gp = F.pad(g.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        blur = F.avg_pool2d(gp, 3, stride=1).squeeze(0)
        w = (self.rate() * dt).clamp(0.0, 1.0)
        return g * (1.0 - w) + blur * w


class LearnableDecay(nn.Module):
    def __init__(self, u0, lo, hi):
        super().__init__(); self.t = _u_param(u0); self.lo, self.hi = lo, hi

    def rate(self): return _phys(self.t, self.lo, self.hi)

    def forward(self, g, dt=1.0):
        return (g - self.rate() * dt).clamp(min=0.0)


class LearnableAdvance(nn.Module):
    """Self-propulsion: displacement = dt*move_speed*(cos h, sin h). Exact given h."""
    def __init__(self, u0, lo, hi):
        super().__init__(); self.t = _u_param(u0); self.lo, self.hi = lo, hi

    def move_speed(self): return _phys(self.t, self.lo, self.hi)

    def forward(self, heading, dt=1.0):
        return dt * self.move_speed() * torch.stack([torch.cos(heading), torch.sin(heading)], -1)


# ---- bilinear field read (differentiable in the sample position) -------------
def bilinear(grid_c, fx, fy, nx, ny):
    """Sample a single channel grid_c:[nx,ny] at continuous pixel coords (fx,fy)."""
    fx = fx.clamp(0, nx - 1.001); fy = fy.clamp(0, ny - 1.001)
    x0 = fx.long(); y0 = fy.long(); x1 = x0 + 1; y1 = y0 + 1
    wx = fx - x0.float(); wy = fy - y0.float()
    g00 = grid_c[x0, y0]; g10 = grid_c[x1, y0]; g01 = grid_c[x0, y1]; g11 = grid_c[x1, y1]
    return (g00 * (1 - wx) * (1 - wy) + g10 * wx * (1 - wy)
            + g01 * (1 - wx) * wy + g11 * wx * wy)


def std3(a, b, c):
    """Standardize the 3 sensor readings per cell (zero-mean, unit-spread) so the
    steering's DECISIVENESS comes from beta, not the reading-magnitude -- removes the
    bias that inflated sensor_angle to manufacture a large wL-wR gap."""
    W = torch.stack([a, b, c], -1)
    W = (W - W.mean(-1, keepdim=True)) / (W.std(-1, keepdim=True) + 1e-3)
    return W[..., 0], W[..., 1], W[..., 2]


def box3(g2d):
    """3x3 replicate-box blur of a single field [nx,ny] -- matches the engine sensor's
    (2*sensor_size+1)^2 window read (sensor_size=1). Sum vs mean is an overall scale
    absorbed by the learnable temp/beta, so we use the mean."""
    x = F.pad(g2d[None, None], (1, 1, 1, 1), mode="replicate")
    return F.avg_pool2d(x, 3, stride=1)[0, 0]


def box3b(fields):
    """3x3 replicate-box blur of a batch of fields [F,nx,ny]."""
    x = F.pad(fields.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    return F.avg_pool2d(x, 3, stride=1).squeeze(1)


def bilinear_batched(fields, fidx, fx, fy, nx, ny):
    """Per-sample bilinear read: fields:[F,nx,ny], fidx/fx/fy:[M] -> [M]. One gather
    per corner (no [M,nx,ny] intermediate) so all frames train in a single batch."""
    fx = fx.clamp(0, nx - 1.001); fy = fy.clamp(0, ny - 1.001)
    x0 = fx.long(); y0 = fy.long(); x1 = x0 + 1; y1 = y0 + 1
    wx = fx - x0.float(); wy = fy - y0.float()
    g00 = fields[fidx, x0, y0]; g10 = fields[fidx, x1, y0]
    g01 = fields[fidx, x0, y1]; g11 = fields[fidx, x1, y1]
    return (g00 * (1 - wx) * (1 - wy) + g10 * wx * (1 - wy)
            + g01 * (1 - wx) * wy + g11 * wx * wy)


class LearnableSense(nn.Module):
    """Differentiable mixture surrogate of the engine `sense` (single-channel).

    The engine's per-cell turn is a 4-component mixture decided by the ORDERING of
    the three sensor readings (wF/wL/wR), each scaled by turn_speed `ts`:
        front strongest (p_max) : turn = 0
        front weakest   (p_min) : turn ~ U(-ts, ts)     mean 0,     var ts^2/3
        front median, wL>wR     : turn ~ U(0,  ts)       mean +ts/2, var ts^2/12
        front median, wR>wL     : turn ~ U(-ts, 0)       mean -ts/2, var ts^2/12
    We model the branch probabilities with a softmax over the orderings (temperature
    `temp`) and the L/R split with `sigmoid(beta*(wL-wR))`, then return the mixture
    (mu, sigma). `ts` enters BOTH mu and sigma; sensor_angle/dist enter through the
    bilinear sample positions (hence the ordering probabilities) -> a real gradient.
    `beta` and `temp` are learnable so the surrogate self-calibrates to the field
    scale. Reparameterized sampling turn = mu + sigma*eps is differentiable."""
    def __init__(self, u_ts, u_ang, u_sd, lo, hi, beta=12.0, temp=0.05,
                 u_cross=0.5, cross_lo=-1.5, cross_hi=1.5):
        super().__init__()
        self.t_ts = _u_param(u_ts); self.t_ang = _u_param(u_ang); self.t_sd = _u_param(u_sd)
        self.lo, self.hi = lo, hi          # arrays [turn_speed, sensor_angle, sensor_dist]
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta), dtype=torch.float32))
        self.log_temp = nn.Parameter(torch.tensor(math.log(temp), dtype=torch.float32))
        self.t_cross = _u_param(u_cross); self.cross_lo, self.cross_hi = cross_lo, cross_hi

    def cross_val(self):                                # inter-type sensing weight (multi-type)
        return _phys(self.t_cross, self.cross_lo, self.cross_hi)

    def mu_vec_mc(self, fields_flat, total, fidx, tidx, pos, hcos, hsin, R, C):
        """Multi-channel expected turn (heading as cos/sin). Per cell the read is
        (1-cross)*own_type_channel + cross*total_over_channels, so `cross` is fitted.
        fields_flat:[F*C,nx,ny] (per-frame per-channel grids), total:[F,nx,ny]=sum_C."""
        ts, ang, sd = self.params(); cross = self.cross_val()
        ffl = box3b(fields_flat); tot = box3b(total)
        nx, ny = total.shape[1], total.shape[2]
        ca, sa = torch.cos(ang), torch.sin(ang)
        def read(rc, rs):
            dx = hcos * rc - hsin * rs; dy = hcos * rs + hsin * rc
            fx = (pos[:, 0] + dx * sd) * R; fy = (pos[:, 1] + dy * sd) * R
            own = bilinear_batched(ffl, fidx * C + tidx, fx, fy, nx, ny)
            allc = bilinear_batched(tot, fidx, fx, fy, nx, ny)
            return (1.0 - cross) * own + cross * allc
        wF, wL, wR = read(1.0, 0.0), read(ca, sa), read(ca, -sa)
        wF, wL, wR = std3(wF, wL, wR)
        temp = self.log_temp.exp().clamp(1e-3, 5.0); beta = self.log_beta.exp().clamp(1.0, 200.0)
        s = torch.stack([wF, wL, wR], -1) / temp
        p_max = torch.softmax(s, -1)[:, 0]; p_min = torch.softmax(-s, -1)[:, 0]
        p_mid = (1.0 - p_max - p_min).clamp(0.0, 1.0)
        p_left = torch.sigmoid(beta * (wL - wR))
        return p_mid * (ts / 2.0) * (2.0 * p_left - 1.0)

    def params(self):
        ts = _phys(self.t_ts, self.lo[0], self.hi[0])
        ang = _phys(self.t_ang, self.lo[1], self.hi[1]) * (math.pi / 180.0)
        sd = _phys(self.t_sd, self.lo[2], self.hi[2])
        return ts, ang, sd

    def _reads(self, grid, pos, heading, R):
        ts, ang, sd = self.params()
        g0 = box3(grid[0]); nx, ny = g0.shape          # 3x3-window read (matches engine)
        def read(aoff):
            a = heading + aoff
            return bilinear(g0, (pos[:, 0] + torch.cos(a) * sd) * R,
                            (pos[:, 1] + torch.sin(a) * sd) * R, nx, ny)
        return ts, read(0.0), read(ang), read(-ang)

    def dist(self, grid, pos, heading, R, beta_scale=1.0):
        ts, wF, wL, wR = self._reads(grid, pos, heading, R)
        wF, wL, wR = std3(wF, wL, wR)                   # scale-invariant steering (de-bias angle)
        temp = self.log_temp.exp().clamp(1e-3, 5.0)
        beta = self.log_beta.exp().clamp(1.0, 200.0) * beta_scale
        s = torch.stack([wF, wL, wR], -1) / temp                    # ordering softmax
        p_max = torch.softmax(s, -1)[:, 0]                          # P(front strongest)
        p_min = torch.softmax(-s, -1)[:, 0]                         # P(front weakest)
        p_mid = (1.0 - p_max - p_min).clamp(0.0, 1.0)
        p_left = torch.sigmoid(beta * (wL - wR))                    # within mid: toward L if wL>wR
        mu = p_mid * (ts / 2.0) * (2.0 * p_left - 1.0)
        ex2 = (ts * ts / 3.0) * (p_min + p_mid)                     # E[turn^2] (mixture)
        var = (ex2 - mu * mu).clamp(min=0.0) + 1e-6
        return mu, var.sqrt()

    def forward(self, grid, pos, heading, R):                       # E[turn]
        return self.dist(grid, pos, heading, R)[0]

    def nll(self, grid, pos, heading, turn_obs, R, beta_scale=1.0):
        mu, sig = self.dist(grid, pos, heading, R, beta_scale)
        return (0.5 * ((turn_obs - mu) / sig) ** 2 + torch.log(sig)).mean()

    # ---- batched path: all frames in one gather (the fast trainer uses this) ----
    def dist_b(self, fields, fidx, pos, heading, R, beta_scale=1.0):
        ts, ang, sd = self.params()
        fields = box3b(fields)                          # 3x3-window read (matches engine)
        nx, ny = fields.shape[1], fields.shape[2]
        def read(aoff):
            a = heading + aoff
            return bilinear_batched(fields, fidx, (pos[:, 0] + torch.cos(a) * sd) * R,
                                    (pos[:, 1] + torch.sin(a) * sd) * R, nx, ny)
        wF, wL, wR = read(0.0), read(ang), read(-ang)
        wF, wL, wR = std3(wF, wL, wR)                   # scale-invariant steering (de-bias angle)
        temp = self.log_temp.exp().clamp(1e-3, 5.0)
        beta = self.log_beta.exp().clamp(1.0, 200.0) * beta_scale
        s = torch.stack([wF, wL, wR], -1) / temp
        p_max = torch.softmax(s, -1)[:, 0]; p_min = torch.softmax(-s, -1)[:, 0]
        p_mid = (1.0 - p_max - p_min).clamp(0.0, 1.0)
        p_left = torch.sigmoid(beta * (wL - wR))
        mu = p_mid * (ts / 2.0) * (2.0 * p_left - 1.0)
        var = ((ts * ts / 3.0) * (p_min + p_mid) - mu * mu).clamp(min=0.0) + 1e-6
        return mu, var.sqrt()

    def nll_b(self, fields, fidx, pos, heading, turn_obs, R, beta_scale=1.0):
        mu, sig = self.dist_b(fields, fidx, pos, heading, R, beta_scale)
        return (0.5 * ((turn_obs - mu) / sig) ** 2 + torch.log(sig)).mean()

    def mu_vec(self, fields, fidx, pos, hcos, hsin, R):
        """Expected turn with heading given as a UNIT VECTOR (cos,sin) -- sensor
        directions are computed by ROTATION (no atan2/wrap), the circular-variable
        representation from the head-direction integrator work. Returns mu [M]."""
        ts, ang, sd = self.params()
        fields = box3b(fields)                          # 3x3-window read (matches engine)
        nx, ny = fields.shape[1], fields.shape[2]
        ca, sa = torch.cos(ang), torch.sin(ang)
        def read(rc, rs):                                   # rotate (hcos,hsin) by (rc,rs)
            dx = hcos * rc - hsin * rs; dy = hcos * rs + hsin * rc
            return bilinear_batched(fields, fidx, (pos[:, 0] + dx * sd) * R,
                                    (pos[:, 1] + dy * sd) * R, nx, ny)
        wF, wL, wR = read(1.0, 0.0), read(ca, sa), read(ca, -sa)
        wF, wL, wR = std3(wF, wL, wR)                   # scale-invariant steering (de-bias angle)
        temp = self.log_temp.exp().clamp(1e-3, 5.0); beta = self.log_beta.exp().clamp(1.0, 200.0)
        s = torch.stack([wF, wL, wR], -1) / temp
        p_max = torch.softmax(s, -1)[:, 0]; p_min = torch.softmax(-s, -1)[:, 0]
        p_mid = (1.0 - p_max - p_min).clamp(0.0, 1.0)
        p_left = torch.sigmoid(beta * (wL - wR))
        return p_mid * (ts / 2.0) * (2.0 * p_left - 1.0)


class LearnableSenseMargin(nn.Module):
    """No-Gumbel `sense` fit: the branch is DETERMINISTIC given the readings, and the
    observed turn reveals it, so geometry is fit by a structured MARGIN (hinge) loss
    that enforces the reading-ordering each branch implies -- no softmax/Gumbel/sampling.

    Engine orderings (single channel): straight -> wF>wL,wF>wR ; left(turn>0) -> wL>wF>wR ;
    right(turn<0) -> wR>wF>wL. turn_speed is estimated closed-form from |turn| (E|turn|=ts/2
    in a side branch), so only sensor_angle/dist are learned -- through the bilinear sample
    positions, giving sensor_angle a clean, relaxation-free gradient."""
    def __init__(self, u_ang, u_sd, lo, hi):
        super().__init__()
        self.t_ang = _u_param(u_ang); self.t_sd = _u_param(u_sd)
        self.lo, self.hi = lo, hi              # arrays [ts, sensor_angle, sensor_dist]

    def geom(self):
        ang = _phys(self.t_ang, self.lo[1], self.hi[1]) * (math.pi / 180.0)
        sd = _phys(self.t_sd, self.lo[2], self.hi[2])
        return ang, sd

    def _reads(self, fields, fidx, pos, heading, R):
        ang, sd = self.geom()
        nx, ny = fields.shape[1], fields.shape[2]
        def read(aoff):
            a = heading + aoff
            return bilinear_batched(fields, fidx, (pos[:, 0] + torch.cos(a) * sd) * R,
                                    (pos[:, 1] + torch.sin(a) * sd) * R, nx, ny)
        return read(0.0), read(ang), read(-ang)

    def margin_loss(self, fields, fidx, pos, heading, turn, R, eps=0.02, m=0.0):
        wF, wL, wR = self._reads(fields, fidx, pos, heading, R)
        relu = torch.nn.functional.relu
        left = (turn > eps).float(); right = (turn < -eps).float()
        straight = (turn.abs() <= eps).float()
        L = (left * (relu(m - (wL - wF)) + relu(m - (wF - wR)))            # wL>wF>wR
             + right * (relu(m - (wR - wF)) + relu(m - (wF - wL)))         # wR>wF>wL
             + straight * (relu(m - (wF - wL)) + relu(m - (wF - wR))))     # wF>wL,wR
        return L.sum() / (left.sum() + right.sum() + straight.sum() + 1e-6)


# =============================================================================
#  PER-OPERATOR TESTS vs the real engine. Each builds a tiny Hierarchy, runs the
#  registered engine operator one step, runs the learnable operator on the same
#  state, and reports the discrepancy. deposit/diffuse/decay/advance must match
#  EXACTLY (~1e-6); sense is a surrogate -> reported as correlation, not equality.
# =============================================================================
def _tiny_H(device="cpu", n=400, res=32, seed=1):
    import os, sys, numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RL"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    import plexus.operators  # noqa
    from plexus.engine import build
    from spec_space import SLIME
    u = np.random.default_rng(0).uniform(0.25, 0.75, SLIME.D)
    spec, p = SLIME.apply_u(u, seed=seed)
    spec.sets["cell"]["n"] = n
    spec.fields["chemical"]["res"] = res
    H = build(spec, device)
    fld = H.fields["chemical"]                                    # develop a non-trivial field
    g = torch.rand_like(fld.grid)
    for _ in range(6):
        g = F.avg_pool2d(F.pad(g.unsqueeze(0), (1, 1, 1, 1), mode="replicate"), 3, 1).squeeze(0)
    fld.grid = g / g.max()
    return H, p, SLIME


def _amax(a, b):
    return (a - b).abs().max().item()


def test_deposit(device="cpu"):
    from plexus.operators.deposit import Deposit
    H, p, S = _tiny_H(device); fld = H.fields["chemical"]; lvl = H.level("cell")
    g0, pos, nt = fld.grid.clone(), lvl.get("pos"), lvl.node_type
    Deposit({"to": "chemical", "amount": p[0], "_at": "cell"}, device).forward(H)   # engine, in place
    lo, hi = S.lo[0], S.hi[0]
    out = LearnableDeposit((p[0] - lo) / (hi - lo), lo, hi).forward(g0.clone(), pos, nt, fld.R)
    return _amax(out, fld.grid)


def test_diffuse(device="cpu"):
    from plexus.operators.diffuse import Diffuse
    H, p, S = _tiny_H(device); fld = H.fields["chemical"]
    g0 = fld.grid.clone()
    Diffuse({"_at": "chemical", "rate": p[1]}, device).forward(H)
    lo, hi = S.lo[1], S.hi[1]
    out = LearnableDiffuse((p[1] - lo) / (hi - lo), lo, hi).forward(g0.clone())
    return _amax(out, fld.grid)


def test_decay(device="cpu"):
    from plexus.operators.decay import Decay
    H, p, S = _tiny_H(device); fld = H.fields["chemical"]
    g0 = fld.grid.clone()
    Decay({"_at": "chemical", "rate": p[2]}, device).forward(H)
    lo, hi = S.lo[2], S.hi[2]
    out = LearnableDecay((p[2] - lo) / (hi - lo), lo, hi).forward(g0.clone())
    return _amax(out, fld.grid)


def test_advance(device="cpu"):
    from plexus.operators.advance import Advance
    H, p, S = _tiny_H(device); lvl = H.level("cell")
    h = lvl.heading.clone()
    vel = Advance({"_at": "cell"}, device).forward(H)["cell"]          # engine velocity delta
    lo, hi = S.lo[4], S.hi[4]
    out = LearnableAdvance((p[4] - lo) / (hi - lo), lo, hi).forward(h)  # displacement = dt*vel
    return _amax(out, vel)


def test_sense(device="cpu"):
    """Surrogate: report correlation between predicted E[turn] and the engine's
    (stochastic) actual turn, plus sign-agreement among cells that turned."""
    import math, numpy as np
    from plexus.operators.sense import Sense
    H, p, S = _tiny_H(device); fld = H.fields["chemical"]; lvl = H.level("cell")
    h0 = lvl.heading.clone()
    Sense({"from": "chemical", "cross": p[3], "_at": "cell"}, device).forward(H)
    turn = (lvl.heading - h0 + math.pi) % (2 * math.pi) - math.pi      # wrapped actual turn
    lo = np.array([S.lo[5], S.lo[6], S.lo[7]]); hi = np.array([S.hi[5], S.hi[6], S.hi[7]])
    u = (np.array([p[5], p[6], p[7]]) - lo) / (hi - lo)
    pred = LearnableSense(u[0], u[1], u[2], lo, hi).forward(fld.grid, lvl.get("pos"), h0, fld.R)
    t = turn.detach().numpy(); q = pred.detach().numpy()
    moved = np.abs(t) > 1e-6
    r = float(np.corrcoef(t[moved], q[moved])[0, 1]) if moved.sum() > 5 else float("nan")
    sign = float((np.sign(t[moved]) == np.sign(q[moved])).mean()) if moved.sum() > 5 else float("nan")
    return r, sign


def run_all_tests(device="cpu"):
    print("== operator self-tests vs engine ==")
    for name, fn in [("deposit", test_deposit), ("diffuse", test_diffuse),
                     ("decay", test_decay), ("advance", test_advance)]:
        e = fn(device)
        print(f"  {name:8s} max|err| = {e:.2e}   {'EXACT ✓' if e < 1e-4 else 'MISMATCH ✗'}")
    r, sign = test_sense(device)
    print(f"  {'sense':8s} SURROGATE: corr(E[turn], engine_turn) = {r:.3f}, "
          f"sign-agreement = {sign:.0%}")


if __name__ == "__main__":
    run_all_tests()
