"""cardio_harmonic.py -- a morphology-aligned LOSS for the cardio-MPM inverse fit.

WHY: the interior R² used so far is a frame-locked squared-displacement error normalised by
temporal variance. By Parseval it is (up to the DC term) the same as matching the raw trajectory
spectrum -- so it is dominated by the BULK radial breathing and is near-blind to the small
area-enclosing LOOP (chirality / openness / axis) that is the actual deliverable. It also charges
loop POSITION and TIMING as shape error.

THIS loss compares each node's beat as a CLOSED CURVE via its low-order complex Fourier
(elliptic-Fourier) harmonics, in a representation that is invariant to the two confounds R² conflates
with shape, and that explicitly surfaces the loop:

  z(t) = dx(t) + i dy(t) ;  c_{+k}, c_{-k} = the +k / -k Fourier coefficients (k>=1; DC dropped).

  * |c_{+k}|, |c_{-k}|              -> size / aspect of the k-th counter-rotating pair
  * signed area  S = sum_k (|c_{+k}|^2 - |c_{-k}|^2)
                                   -> SIGNED enclosed area = chirality (sign) x openness x size.
                                      THIS is the term R² cannot see: a degenerate line has
                                      |c_{+k}|=|c_{-k}| -> S=0, so a big radial line contributes
                                      nothing here while a real loop contributes ~ line_amp x loop_amp.
  * c_{+k} * c_{-k}  (complex)      -> axis ORIENTATION carrier (its phase = 2*angle).

All three feature groups are invariant to a global time-shift of the beat (c_{+k}->c_{+k}e^{-iwkT},
c_{-k}->c_{-k}e^{+iwkT}: magnitudes, S, and the product c_{+k}c_{-k} are all unchanged) and to loop
POSITION (k>=1 drops the DC term). So shape is judged as shape, not timing or placement.
"""
import math
import torch


def _harmonics(disp, K):
    """disp [G, M, 2] (real) -> cp, cm each [K, M] complex: the +k and -k coeffs for k=1..K."""
    z = disp[..., 0] + 1j * disp[..., 1]                 # [G, M] complex
    G = z.shape[0]
    Z = torch.fft.fft(z, dim=0) / G                      # [G, M] complex spectrum
    K = min(K, (G - 1) // 2)
    cp = Z[1:K + 1]                                      # [K, M]  (+1.. +K)
    cm = torch.flip(Z[G - K:G], dims=[0])                # [K, M]  (-1.. -K), ordered k=1..K
    return cp, cm


def harmonic_descriptor(disp, K=4):
    """Time-shift- + position-invariant per-node loop descriptor. disp [G,M,2] -> dict of [K,M] / [K,M]."""
    cp, cm = _harmonics(disp, K)
    return {"mag_p": cp.abs(), "mag_m": cm.abs(),        # CCW / CW component sizes
            "area": cp.abs() ** 2 - cm.abs() ** 2,       # signed area per harmonic (chirality x size)
            "prod": cp * cm}                             # orientation carrier (phase = 2*axis-angle)


def _rel(num, den, eps=1e-12):
    return num / (den + eps)


def _pernode_terms(sim_d, real_d, mov, K=4, floor=0.02):
    """PER-NODE relative feature errors. Returns (mag, area, orient), each a [M] vector over nodes.

    CRITICAL: each node is normalised by ITS OWN real harmonic energy -- so a wrong loop on any node
    is a large error regardless of its absolute motion size. (Summing numerators+denominators over all
    nodes FIRST -- a single GLOBAL ratio -- lets big-motion nodes and already-matched global
    energy/chirality MASK individual wrong loops: that is the bulk-domination failure this metric
    exists to avoid.) Denominators are floored at `floor`x the GLOBAL mean node energy so a
    near-degenerate node cannot blow its ratio up."""
    s = sim_d[:, mov] if mov is not None else sim_d
    r = real_d[:, mov] if mov is not None else real_d
    ds, dr = harmonic_descriptor(s, K), harmonic_descriptor(r, K)     # each [K, M]
    E = (dr["mag_p"] ** 2 + dr["mag_m"] ** 2).sum(0)                  # [M] per-node real energy
    gE = E.mean().clamp(min=1e-12)                                    # global scale for the floors
    fa = (floor * gE) ** 2
    # RAW per-node relative errors r = error/signal (UNBOUNDED). r=0 perfect; r=1 means the sim got
    # NOTHING right (e.g. a stub / zero motion: error == signal); r>1 is actively wrong (overshoot,
    # opposite phase). NOT bounded by r/(1+r) -- that mapped r=1 -> 0.5, so a stub that misses the loop
    # scored ~0.5 ("looks ok"). The score 1-r (clamped below) maps a stub to ~0, overshoot to <0.
    mag = ((ds["mag_p"] - dr["mag_p"]) ** 2 + (ds["mag_m"] - dr["mag_m"]) ** 2).sum(0) / (E + floor * gE)
    area = ((ds["area"] - dr["area"]) ** 2).sum(0) / ((dr["area"] ** 2).sum(0) + fa)
    orient = (ds["prod"] - dr["prod"]).abs().pow(2).sum(0) / (dr["prod"].abs().pow(2).sum(0) + fa)
    return mag, area, orient                                        # each [M], unbounded >=0


def harmonic_pernode_loss(sim_d, real_d, mov, K=4, w_mag=1.0, w_area=3.0, w_orient=1.0):
    """[M] per-node relative error r (0 = perfect loop match; ~1 = sim got nothing right; >1 = overshoot).
    w_area>w_mag emphasises the SIGNED-AREA (chirality x openness) term -- the loop-defining structure R²
    is blind to. UNBOUNDED -> used as the training loss so overshoot keeps a strong correcting gradient."""
    mag, area, orient = _pernode_terms(sim_d, real_d, mov, K)
    return (w_mag * mag + w_area * area + w_orient * orient) / (w_mag + w_area + w_orient)


def harmonic_terms(sim_d, real_d, mov, K=4):
    """Scalar (node-MEANED) per-term relative errors -- diagnostic."""
    mag, area, orient = _pernode_terms(sim_d, real_d, mov, K)
    return mag.mean(), area.mean(), orient.mean()


def harmonic_loss(sim_d, real_d, mov, K=4, **w):
    """Differentiable training loss to MINIMISE = MEAN over nodes of the per-node relative error r
    (unbounded; gives overshoot a strong correcting gradient)."""
    return harmonic_pernode_loss(sim_d, real_d, mov, K, **w).mean()


def _pernode_score(sim_d, real_d, mov, K=4, **w):
    """[M] per-node LoopScore SCORE = clamp(1 - r, -1, 1): R²-like. 1=perfect, 0=sim got nothing right
    (stub/zero), <0=overshoot, -1=floor. Clamped so outliers don't blow up the reported mean/SD."""
    r = harmonic_pernode_loss(sim_d, real_d, mov, K, **w)
    return (1.0 - r).clamp(-1.0, 1.0)


def harmonic_score(sim_d, real_d, mov, K=4, **w):
    """Single-loop LoopScore score (the montage / dashboard per-panel value)."""
    return float(_pernode_score(sim_d, real_d, mov, K, **w).mean())


def harmonic_stats(sim_d, real_d, mov, K=4, **w):
    """(mean, sd) of the per-node LoopScore score for REPORTING. MEAN = objective (how well loops match,
    1=perfect, 0=no better than zero motion, <0=overshoot); SD = cross-node uniformity."""
    s = _pernode_score(sim_d, real_d, mov, K, **w)
    return float(s.mean()), float(s.std())


def fundamental_ellipse(disp, n=64):
    """Reconstruct the k=±1 fundamental ellipse (cos/sin of one harmonic) for a single loop
    disp [G,2] -> [n,2] curve, for overlaying on the real loop in the montage."""
    cp, cm = _harmonics(disp[:, None, :], 1)             # [1,1]
    cp, cm = cp[0, 0], cm[0, 0]
    import torch as _t
    th = _t.linspace(0, 2 * 3.141592653589793, n)
    z = cp * _t.exp(1j * th) + cm * _t.exp(-1j * th)     # ellipse traced by the fundamental
    return _t.stack([z.real, z.imag], -1)


def interior_r2(sim_d, real_d, mov):
    """The EXACT metric the trainer ranks on: frame-locked, motion-normalised interior R²."""
    s = sim_d[:, mov] if mov is not None else sim_d
    r = real_d[:, mov] if mov is not None else real_d
    res = ((s - r) ** 2).sum()
    tot = ((r - r.mean(0, keepdim=True)) ** 2).sum().clamp(min=1e-12)
    return float(1.0 - res / tot)


# --------------------------------------------------------------------------- #
#  RESIDUAL DECOMPOSITION -- "where does the best model LOSE LoopScore?"
#  Sensitivity (above) = what LS rewards. This = the remaining error of a real fit, attributed to
#  morphology dimensions. For each node we correct the SIM's fundamental ellipse (c_{+1}, c_{-1})
#  TOWARD the real along ONE dimension at a time and measure the LS recovered (DeltaLS). The dimension
#  whose correction recovers the most LS is the mechanism the model is missing.
# --------------------------------------------------------------------------- #
def _reconstruct(c0, cp, cm, G):
    """c0 [M] complex DC, cp/cm [K,M] complex -> displacement [G,M,2]."""
    K = cp.shape[0]
    w = (2 * math.pi / G) * torch.arange(G, dtype=torch.float32, device=cp.device)   # [G]
    z = c0[None].to(torch.complex64).expand(G, -1).clone()
    for k in range(1, K + 1):
        z = z + cp[k - 1][None] * torch.exp(1j * w[:, None] * k) + cm[k - 1][None] * torch.exp(-1j * w[:, None] * k)
    return torch.stack([z.real, z.imag], -1)


def loopscore_residual(sim_d, real_d, mov, K=4):
    """Returns (base_LS, {dimension: DeltaLS}) attributing the LS gap of sim vs real to morphology
    dimensions: size, openness/aspect, chirality, orientation, and shape-detail (higher harmonics)."""
    s = sim_d[:, mov] if mov is not None else sim_d
    r = real_d[:, mov] if mov is not None else real_d
    G = s.shape[0]; eps = 1e-9
    zs = s[..., 0] + 1j * s[..., 1]; zr = r[..., 0] + 1j * r[..., 1]
    Zs = torch.fft.fft(zs, dim=0) / G; Zr = torch.fft.fft(zr, dim=0) / G
    c0s = Zs[0]
    cps = Zs[1:K + 1].clone(); cms = torch.flip(Zs[G - K:G], dims=[0]).clone()
    cpr = Zr[1:K + 1]; cmr = torch.flip(Zr[G - K:G], dims=[0])
    allmov = torch.ones(s.shape[1], dtype=torch.bool, device=s.device)
    base = harmonic_score(s, r, allmov, K=K)
    # fundamental params (k=1)
    ap, am = cps[0].abs(), cms[0].abs(); a_s = ap + am; b_s = (ap - am).abs(); ch_s = torch.sign(ap - am)
    php, phm = torch.angle(cps[0]), torch.angle(cms[0]); th_s = (php + phm) / 2
    rp, rm = cpr[0].abs(), cmr[0].abs(); a_r = rp + rm; b_r = (rp - rm).abs(); ch_r = torch.sign(rp - rm)
    th_r = (torch.angle(cpr[0]) + torch.angle(cmr[0])) / 2

    def ls_with(cp1, cm1, hi_real=False):
        cp, cm = cps.clone(), cms.clone()
        cp[0], cm[0] = cp1, cm1
        if hi_real:
            cp[1:], cm[1:] = cpr[1:], cmr[1:]
        return harmonic_score(_reconstruct(c0s, cp, cm, G), r, allmov, K=K)

    f = a_r / (a_s + eps)                                                       # size: scale magnitudes
    rot = torch.exp(1j * (th_r - th_s))                                         # orientation: rotate fundamental
    bn = (b_r / (a_r + eps)) * a_s                                              # aspect: match aspect ratio, keep size
    asp_p = (a_s + ch_s * bn) / 2; asp_m = (a_s - ch_s * bn) / 2
    swap = (ch_s != ch_r).float()                                              # chirality: swap |c+|,|c-| if sign differs
    chp = ap * (1 - swap) + am * swap; chm = am * (1 - swap) + ap * swap
    d = {"size":            ls_with(cps[0] * f, cms[0] * f) - base,
         "orientation":     ls_with(cps[0] * rot, cms[0] * rot) - base,
         "openness/aspect": ls_with(asp_p * torch.exp(1j * php), asp_m * torch.exp(1j * phm)) - base,
         "chirality":       ls_with(chp * torch.exp(1j * php), chm * torch.exp(1j * phm)) - base,
         "shape-detail(k>=2)": ls_with(cps[0], cms[0], hi_real=True) - base}
    return base, d
