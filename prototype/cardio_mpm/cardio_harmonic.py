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


def _pernode_terms(sim_d, real_d, mov, K=4, floor=0.05):
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
    mag = ((ds["mag_p"] - dr["mag_p"]) ** 2 + (ds["mag_m"] - dr["mag_m"]) ** 2).sum(0) / (E + floor * gE)
    area = ((ds["area"] - dr["area"]) ** 2).sum(0) / ((dr["area"] ** 2).sum(0) + fa)
    orient = (ds["prod"] - dr["prod"]).abs().pow(2).sum(0) / (dr["prod"].abs().pow(2).sum(0) + fa)
    return mag, area, orient                                         # each [M]


def harmonic_pernode_loss(sim_d, real_d, mov, K=4, w_mag=1.0, w_area=3.0, w_orient=1.0):
    """[M] per-node morphology loss (0 = perfect). w_area>w_mag emphasises the SIGNED-AREA
    (chirality x openness) term -- the loop-defining structure R² is blind to."""
    mag, area, orient = _pernode_terms(sim_d, real_d, mov, K)
    return (w_mag * mag + w_area * area + w_orient * orient) / (w_mag + w_area + w_orient)


def harmonic_terms(sim_d, real_d, mov, K=4):
    """Scalar (node-MEANED) per-term errors -- for the montage."""
    mag, area, orient = _pernode_terms(sim_d, real_d, mov, K)
    return mag.mean(), area.mean(), orient.mean()


def harmonic_loss(sim_d, real_d, mov, K=4, **w):
    """Differentiable morphology loss to MINIMISE = MEAN over nodes of the per-node loss."""
    return harmonic_pernode_loss(sim_d, real_d, mov, K, **w).mean()


def harmonic_score(sim_d, real_d, mov, K=4, **w):
    """1 - mean per-node loss; 'higher=better, 1=perfect' like R² (can go negative)."""
    return float(1.0 - harmonic_loss(sim_d, real_d, mov, K, **w))


def harmonic_stats(sim_d, real_d, mov, K=4, **w):
    """(mean, sd) of the per-node Hrm score across nodes. MEAN = the objective (how well loops match);
    SD = how UNIFORM the match is (high SD -> some nodes good, some terrible). Both for reporting."""
    hrm = 1.0 - harmonic_pernode_loss(sim_d, real_d, mov, K, **w)    # [M] per-node score
    return float(hrm.mean()), float(hrm.std())


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
