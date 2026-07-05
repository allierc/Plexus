"""audit_trajectories.py -- INDEPENDENT (out-of-pipeline) trajectory analysis for the cardio-MPM fit.

Purpose: check whether the QUANTIFICATION the agentic loop reports is sound, and therefore whether the
INSIGHTS built on it hold. Consumes the raw sim_d/real_d dumped by `cardio_mpm_train.py --eval_dump`
(no MPM, no pipeline metric code trusted; everything recomputed from scratch here).

Key questions:
  1. The pipeline's `size` diagnostic is `mean_node max|sim_disp|` -- SIM ONLY (train.py:203). It never
     looks at the real loop. So "size flat at ~1e-3 across all levers" says nothing about the sim-vs-real
     size RESIDUAL. Here we compute the real loop size on the SAME nodes and form the actual ratio.
  2. Is the residual really SIZE (sim = scaled-down copy of real) or SHAPE/DEGENERACY (sim motion is
     radial/linear, lacking the enclosed area that defines a loop)? These imply different mechanisms.
  3. Does total-energy match (ampL~0) coexist with poor loops? If so the motion is right-magnitude but
     wrong-shape -> the bottleneck is area-enclosure, not size.
"""
import sys, numpy as np

FILES = sys.argv[1:] or ["/tmp/cardio_audit/wide400.npz", "/tmp/cardio_audit/ctrl.npz",
                         "/tmp/cardio_audit/nofibre.npz"]


def loop_stats(d):
    """d [G, M, 2] displacement over one beat for M nodes. Returns per-node arrays."""
    G, M, _ = d.shape
    dc = d - d.mean(0, keepdims=True)                      # centre each loop (drop DC/position)
    # size: peak excursion and RMS radius
    peak = np.abs(dc).max(0).max(1)                        # max over frames & xy  -> [M]  (mirrors train.py size)
    rms = np.sqrt((dc ** 2).sum(2).mean(0))               # rms radius             -> [M]
    # signed enclosed area via shoelace (chirality x openness x size^2)
    x, y = dc[..., 0], dc[..., 1]
    area = 0.5 * (x * np.roll(y, -1, 0) - np.roll(x, -1, 0) * y).sum(0)   # [M] signed
    # PCA aspect: SVD of the [G,2] point cloud -> singular values s0>=s1; aspect = s1/s0 in [0,1]
    #   ~1 = round loop, ~0 = degenerate line (radial). Independent of the Fourier metric.
    aspect = np.empty(M)
    for m in range(M):
        s = np.linalg.svd(dc[:, m, :], compute_uv=False)
        aspect[m] = s[1] / (s[0] + 1e-12)
    # isoperimetric-style loopiness: |area| / (pi * (peak/2)^2) -- fraction of the bounding disc enclosed
    loopiness = np.abs(area) / (np.pi * (peak / 2) ** 2 + 1e-12)
    return dict(peak=peak, rms=rms, area=area, aspect=aspect, loopiness=loopiness)


def summ(name, a):
    return f"{name:>10s}: median={np.median(a):.3e} mean={np.mean(a):.3e} p10={np.percentile(a,10):.3e} p90={np.percentile(a,90):.3e}"


for f in FILES:
    Z = np.load(f)
    sim, real, mov = Z["sim_d"], Z["real_d"], Z["mov"].astype(bool)
    s = sim[:, mov]; r = real[:, mov]                     # [G, Nmov, 2]
    ss, rs = loop_stats(s), loop_stats(r)
    tag = f.split("/")[-1].replace(".npz", "")
    print("=" * 96)
    print(f"### {tag}   (Nmov={mov.sum()}, G={sim.shape[0]} frames)")

    # --- 1. SIZE: sim vs real, the residual the pipeline never forms ---
    peak_ratio = ss["peak"] / (rs["peak"] + 1e-12)
    rms_ratio = ss["rms"] / (rs["rms"] + 1e-12)
    print("-- SIZE (peak excursion) --")
    print("  ", summ("sim_peak", ss["peak"]))
    print("  ", summ("real_peak", rs["peak"]))
    print(f"   sim/real peak ratio: median={np.median(peak_ratio):.3f} "
          f"mean={np.mean(peak_ratio):.3f} p10={np.percentile(peak_ratio,10):.3f} p90={np.percentile(peak_ratio,90):.3f}")
    print(f"   sim/real  rms ratio: median={np.median(rms_ratio):.3f} mean={np.mean(rms_ratio):.3f}")

    # --- 2. SHAPE / DEGENERACY: is the sim a loop or a radial line? ---
    print("-- SHAPE (PCA aspect s1/s0: ~1 round loop, ~0 degenerate line) --")
    print("  ", summ("sim_aspect", ss["aspect"]))
    print("  ", summ("real_aspect", rs["aspect"]))
    print("-- LOOPINESS (|area| / bounding-disc area) --")
    print("  ", summ("sim_loopy", ss["loopiness"]))
    print("  ", summ("real_loopy", rs["loopiness"]))

    # --- 3. ENCLOSED AREA (the loop-defining quantity) sim vs real ---
    area_ratio = np.abs(ss["area"]) / (np.abs(rs["area"]) + 1e-12)
    chir_agree = (np.sign(ss["area"]) == np.sign(rs["area"])).mean()
    print("-- ENCLOSED AREA (|signed area|) --")
    print(f"   sim/real |area| ratio: median={np.median(area_ratio):.3f} mean={np.mean(area_ratio):.3f}")
    print(f"   chirality sign agreement (sim vs real): {chir_agree*100:.1f}% of nodes")

    # --- 4. TOTAL ENERGY (what ampL matches) ---
    e_sim = (s ** 2).sum(); e_real = (r ** 2).sum()
    ampL = (np.sqrt(e_sim) - np.sqrt(e_real)) ** 2 / e_real
    print("-- ENERGY --")
    print(f"   sqrt(E_sim)/sqrt(E_real) = {np.sqrt(e_sim/e_real):.3f}   (ampL={ampL:.4f})")
    # decompose real energy into loop (area) vs radial (line) content, same for sim:
    #   fraction of variance on the minor axis = 'how 2-D / loopy' the aggregate motion is
    def minor_frac(d):
        dc = d - d.mean(0, keepdims=True)
        C = np.einsum('gmi,gmj->ij', dc, dc)
        sv = np.linalg.svd(C, compute_uv=False)
        return sv[1] / sv.sum()
    print(f"   aggregate minor-axis variance frac: sim={minor_frac(s):.3f} real={minor_frac(r):.3f} "
          f"(higher = more genuinely 2-D/loopy)")
print("=" * 96)
