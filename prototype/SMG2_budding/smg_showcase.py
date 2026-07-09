#!/usr/bin/env python
"""smg_showcase -- run ONE SMG forward-model spec, score it against the SMG target
OBSERVABLES (topology / migration / growth), render a styled 3xN montage mirroring the
real-data target, write scorecard.json + metrics.json, and archive.

Observables (2D, same logic as the real-data projection so sim<->real are comparable):
  topology   density -> occupancy -> buds (EDT peaks) / branches (skeleton junctions) / tube (components)
  migration  per-cell velocity (sim agents keep identity): polar_order, speed
  growth     live-cell count trajectory -> growth ratio + trend

    python smg_showcase.py specs/smg_base.yaml [tag=..] [frames=1200] [stride=8] [key=val ...]
"""
import os, sys, json, time, shutil, tempfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, os.path.join(HERE, "..", "active_matter2"))
sys.path.insert(0, os.path.join(HERE, "..", "embryo_gray_scott"))   # b16: gray_scott + rd_seed ops
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
import torch
import plexus.operators   # noqa
import am2_ops            # noqa
import embryo_gray_scott_ops   # noqa  b16: registers gray_scott (RD field) + rd_seed (IC)
import plexus.schema as S
from plexus.engine import run
import smg_scorecard as SC

ARCHIVE = os.path.join(HERE, "archive")
VOX = 0.008
# Topology-readout calibration (Batch 3). n_tube>>1 at 600 pts is a DENSITY-RESOLUTION artifact
# (equilibrium spacing > detector kernel + a RELATIVE 0.15*max threshold). These knobs are pure
# READOUT (physics unchanged) so a det.* sweep on identical positions isolates the detector's share.
# Override via slots: `det.sigma_vox 3.0`, `det.thr 0.06`, `det.tube_frac 0.30`.
# Batch 4: `sigma_vox` sets CONNECTIVITY (n_tube) but a SINGLE scale trades it off against bud
# resolution (b03: sig26 connects but buds are noisy 6-9; sig40 -> n_tube=1 but n_bud collapses to 1).
# So DECOUPLE the bud scale: `bud_thr_abs` = EDT prominence floor (raise to filter shallow MIPS
# ripples -> low flat bud baseline), `bud_min_dist` = min peak separation. Both pure READOUT.
# Batch 5 LOCK: sigma_vox 1.6 -> 3.0 as the DEFAULT connectivity scale (b04: sig30 gives
# tube_stab 1.0, n_tube==1 EVERY frame, on identical physics). The bud-baseline hypothesis
# (prominence filters shallow ripples) was FALSIFIED at b04 -- the ~4-7 bud lumps SURVIVE
# bud_thr_abs 4.0, so they are PHYSICAL deep-prominence MIPS clumps of the no-growth active
# sheet, NOT a shallow detector ripple. => no detector-only knob flattens them; the no-growth
# bud count is a REPRODUCIBLE physical baseline to SUBTRACT as the Q2 ablation control (not to
# flatten). sig30 (not sig35) keeps bud sensitivity for real Q2 growth buds while locking n_tube.
DET = {"sigma_vox": 3.0, "thr": 0.15, "tube_frac": 0.05, "bud_min_dist": 5, "bud_thr_abs": 2.0}


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) and "e" not in str(v).lower() else f
    except (ValueError, TypeError):
        return v


def _apply(sim, key, val):
    """Apply a dotted override (self-contained; mirrors embryo tune._apply)."""
    val = _num(val)
    if key == "n_grid":
        sim.fields["mpm_grid"]["n_grid"] = val
    elif key in ("agent.move_speed", "agent.div_rate"):
        p = key.split(".", 1)[1]
        for t in sim.sets["agent"]["types"].values():
            t[p] = val
    elif key == "agent.n":
        sim.sets["agent"]["n"] = val
    elif key == "spawn_radius":
        sim.sets["agent"]["spawn_radius"] = val
    elif key == "per_parent":
        sim.sets["mpm_particle"]["per_parent"] = val
    elif key == "radius":                              # MPM cell disc size (Batch 9 substrate-scale sweep)
        sim.sets["mpm_particle"]["radius"] = val
    elif key == "grow_reserve":
        sim.sets["mpm_particle"]["grow_reserve"] = val
    elif key == "cell.youngs":
        for t in sim.sets["cell"]["types"].values():
            t["youngs"] = val
            for L in t.get("layers", []):
                L["youngs"] = val
    elif key in ("seed", "general.seed"):
        try:
            sim.general["seed"] = val
        except Exception:
            setattr(sim, "seed", val)
    elif key.startswith("det."):                       # topology-readout calibration (Batch 3)
        DET[key.split(".", 1)[1]] = val
    elif "." in key:                                   # generic operator param: opname.param
        opname, param = key.split(".", 1)
        hit = False
        for o in sim.operators:
            if o.op == opname:
                o.params[param] = val; hit = True
        if not hit:
            print(f"[smg] override {key!r} matched no operator", flush=True)
    else:
        print(f"[smg] unknown override {key!r}", flush=True)


def _density(P, W, sigma_vox=None):
    if sigma_vox is None:
        sigma_vox = DET["sigma_vox"]
    nx, ny = int(W / VOX) + 1, int(1.0 / VOX) + 1
    ix = np.clip((P[:, 0] / VOX).astype(int), 0, nx - 1)
    iy = np.clip((P[:, 1] / VOX).astype(int), 0, ny - 1)
    g = np.zeros((nx, ny), np.float32)
    np.add.at(g, (ix, iy), 1.0)
    return ndi.gaussian_filter(g, sigma_vox)


def _topology(dens, thr=None, bud_min_dist=None):
    """2D branch-graph readout: (n_bud, n_branch, n_tube, occ, skel, buds)."""
    if thr is None:
        thr = DET["thr"]
    if bud_min_dist is None:
        bud_min_dist = DET["bud_min_dist"]
    occ = ndi.binary_fill_holes(dens > thr * max(dens.max(), 1e-9))
    lbl, ncomp = ndi.label(occ)
    if ncomp == 0:
        return 0, 0, 0, occ, occ, np.empty((0, 2))
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    n_tube = int((sizes > DET["tube_frac"] * sizes.max()).sum())
    largest = lbl == int(sizes.argmax())
    edt = ndi.distance_transform_edt(largest)
    buds = peak_local_max(edt, min_distance=int(bud_min_dist), labels=largest,
                          threshold_abs=DET["bud_thr_abs"])
    skel = skeletonize(largest)
    nbr = ndi.convolve(skel.astype(int), np.ones((3, 3)), mode="constant") - skel.astype(int)
    bp = skel & (nbr >= 3)
    n_branch = int(ndi.label(bp)[1])                       # branch-point clusters
    return len(buds), n_branch, n_tube, largest, skel, buds


def _label(ax, s):
    ax.text(0.02, 0.98, s, transform=ax.transAxes, color="white",
            fontsize=11, va="top", ha="left")


def main():
    spec_path = sys.argv[1]; args = sys.argv[2:]
    ov = dict(kv.split("=", 1) for kv in args if "=" in kv)
    tag = ov.pop("tag", "show"); frames = int(ov.pop("frames", 1200))
    stride = int(ov.pop("stride", 8))
    sim = S.load(spec_path); sim.n_frames = frames
    base_name = sim.name
    sim.name = tag                      # archive dir = tag (starts 'bNN_' -> sorts by batch)
    for k, v in ov.items():
        _apply(sim, k, v)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    W = float(getattr(sim, "world_size", [1.0])[0])
    print(f"[smg] {sim.name}: frames={frames} stride={stride} ov={ov} dev={dev}", flush=True)

    caps = {"aX": [], "occ": []}
    t0 = time.time()

    def hook(H, frame):
        if frame % stride:
            return
        a = H.levels["agent"]
        caps["aX"].append(a.get("pos").detach().cpu().numpy().copy())
        caps["occ"].append(a.occ.detach().cpu().numpy().copy())

    try:
        run(sim, out_path=None, device=dev, on_frame=hook)
    except Exception as e:
        print(f"[smg] SIM FAILED: {e}", flush=True)
        os.makedirs(os.path.join(ARCHIVE, sim.name), exist_ok=True)
        json.dump({"name": sim.name, "sim_error": str(e)[:200]},
                  open(os.path.join(ARCHIVE, sim.name, "metrics.json"), "w"), indent=2)
        return
    aX = np.array(caps["aX"]); occ = np.array(caps["occ"]) > 0
    T = aX.shape[0]
    print(f"[smg] captured {T} frames in {time.time()-t0:.0f}s", flush=True)

    # --- observable TRAJECTORY (topology at 5 pts + all-frame for scoring) ---
    fr_idx = [max(0, int(round(f * (T - 1)))) for f in SC.FRACS]
    topo = {}
    for i in range(T):
        Pi = aX[i][occ[i]]
        nb, nbr, nt, largest, *_ = _topology(_density(Pi, W))
        # threshold-free body extent: radius of gyration of the live cloud (world units).
        # SEPARATES growth modes -- inflate (rg grows, area~=growth_ratio) vs repack (rg~const,
        # count up but same footprint). body_area = occupied-mask voxels (relative-thr, secondary).
        if len(Pi):
            c = Pi.mean(axis=0)
            rg = float(np.sqrt((np.linalg.norm(Pi - c, axis=1) ** 2).mean()))
        else:
            rg = 0.0
        # threshold-free body SHAPE ANISOTROPY (Batch 7, rung-3 readout): aspect ratio of the 2D
        # gyration tensor -- body_aniso 1.0 = round disc (isotropic growth), >1 = elongated finger/oval
        # (anisotropic/tip growth). body_axis = major-axis angle (deg) -> confirms the elongation
        # follows the growth axis. Pure READOUT (a cloud moment like rg, threshold-free); physics
        # untouched. Wrapped defensively (embryo safe-metric recipe): any failure -> round/0 default.
        aniso, axis_ang = 1.0, 0.0
        try:
            if len(Pi) >= 3:
                Q = Pi[:, :2] - Pi[:, :2].mean(axis=0)
                cov = (Q.T @ Q) / len(Q)
                ev, evec = np.linalg.eigh(cov)
                lam = np.clip(ev, 1e-12, None)
                aniso = float(np.sqrt(lam[1] / lam[0]))
                axis_ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
        except Exception:
            aniso, axis_ang = 1.0, 0.0
        topo[i] = dict(frame=i, n_bud=nb, n_branch=nbr, n_tube=nt,
                       body_area=int(largest.sum()), body_rg=round(rg, 5),
                       body_aniso=round(aniso, 4), body_axis=round(axis_ang, 1))
    per_frame = [topo[i] for i in range(T)]

    # --- migration: per-cell velocity (agents keep identity in-sim) ---
    tail = max(2, T // 5)
    live = occ[-1]
    v = np.diff(aX[-tail:], axis=0)[:, live]                # [tail-1, n, 2]
    sp = np.linalg.norm(v, axis=-1)
    polar_order = float(np.linalg.norm((v / np.clip(sp[..., None], 1e-9, None)).mean(axis=(0, 1))))
    speed = float(sp.mean())

    # --- growth: live-cell trajectory + body AREA (Q2 repack-vs-inflate discriminator) ---
    n_live = occ.sum(axis=1).astype(float)
    growth_ratio = float(n_live[-1] / max(n_live[0], 1))
    rg0 = per_frame[0]["body_rg"]; rg1 = per_frame[-1]["body_rg"]
    # AREA ratio ~ (rg_final/rg_0)^2. ~1 => REPACK (count up, footprint const, denser);
    # ~=growth_ratio => INFLATE (new cells fill new area, density preserved).
    area_ratio = float((rg1 / max(rg0, 1e-9)) ** 2)
    # SHAPE anisotropy (Batch 7, rung-3): aniso0/aniso1 = body aspect ratio at start/end (median over
    # the last 10% of frames for aniso1 -> robust to per-frame jitter); axis1 = final major-axis angle.
    # aniso1 >> aniso0 => a LOCALIZED/differential growth program made a directed protrusion (isotropic
    # growth leaves aniso~=aniso0). The clean readout that separates a bud/finger from a round disc.
    aniso0 = per_frame[0]["body_aniso"]
    tail_a = [p["body_aniso"] for p in per_frame[-max(1, len(per_frame) // 10):]]
    aniso1 = float(np.median(tail_a))
    axis1 = per_frame[-1]["body_axis"]

    temporal = SC.temporal_consistency(list(range(T)),
                                       [p["n_bud"] for p in per_frame],
                                       [p["n_branch"] for p in per_frame],
                                       [p["n_tube"] for p in per_frame])
    # smooth target score in [0,1]: budding trend + one tube + coherent-but-not-frozen migration + net growth
    tgt = float(np.clip(0.30 * max(temporal["bud_trend"], 0)
                        + 0.25 * temporal["tube_stability"]
                        + 0.25 * (1 - min(abs(polar_order - 0.4) / 0.4, 1))
                        + 0.20 * (1 - min(abs(np.log(max(growth_ratio, 1e-3) / 1.3)), 1)), 0, 1))
    metrics = dict(name=sim.name, frames=frames, seconds=round(time.time() - t0, 1),
                   n_cells0=int(n_live[0]), n_cells=int(n_live[-1]),
                   n_bud=per_frame[-1]["n_bud"], n_branch=per_frame[-1]["n_branch"],
                   n_tube=per_frame[-1]["n_tube"], polar_order=round(polar_order, 4),
                   speed=round(speed, 6), growth_ratio=round(growth_ratio, 3),
                   area_ratio=round(area_ratio, 3), body_rg=round(rg1, 5),
                   body_aniso=round(aniso1, 3), body_axis=round(axis1, 1),
                   target_score=round(tgt, 4), **{f"temporal_{k}": v for k, v in temporal.items()})

    d = os.path.join(ARCHIVE, sim.name); os.makedirs(d, exist_ok=True)
    json.dump(metrics, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    json.dump({"name": sim.name, "pcts": SC.PCTS, "per_frame": per_frame,
               "temporal": temporal, "migration": {"polar_order": polar_order, "speed": speed},
               "growth": {"growth_ratio": growth_ratio, "area_ratio": area_ratio,
                          "body_rg0": rg0, "body_rg": rg1,
                          "aniso0": aniso0, "aniso1": aniso1, "axis1": axis1}, "target_score": tgt},
              open(os.path.join(d, "scorecard.json"), "w"), indent=2)

    # --- styled 3xN montage mirroring the real target (black bg, top-left labels) ---
    cols = [max(0, int(round(f * (T - 1)))) for f in (0.02, 0.35, 0.68, 1.0)]
    fig, axs = plt.subplots(3, 4, figsize=(20, 15)); fig.patch.set_facecolor("black")
    for j, k in enumerate(cols):
        P = aX[k][occ[k]]; dens = _density(P, W)
        nb, nbr, nt, largest, skel, buds = _topology(dens)
        # row1 topology
        ax = axs[0, j]; ax.set_facecolor("black")
        ax.imshow(dens.T, origin="lower", cmap="magma")
        if skel.any():
            sy, sx = np.nonzero(skel.T); ax.scatter(sx, sy, s=0.4, c="deepskyblue", alpha=0.4)
        if len(buds):
            ax.scatter(buds[:, 0], buds[:, 1], s=60, facecolors="none", edgecolors="cyan", linewidths=1.4)
        _label(ax, f"{'TOPOLOGY  ' if j == 0 else ''}t={k*stride}\nbuds {nb}  branch {nbr}  tube {nt}")
        # row2 growth / density
        ax = axs[1, j]; ax.set_facecolor("black")
        ax.imshow(dens.T, origin="lower", cmap="viridis")
        _label(ax, f"{'DENSITY/GROWTH  ' if j == 0 else ''}n={int(occ[k].sum())}")
        # row3 migration
        ax = axs[2, j]; ax.set_facecolor("black")
        if k > 0:
            vk = aX[k][occ[k] & occ[max(k-1, 0)]] - aX[max(k-1, 0)][occ[k] & occ[max(k-1, 0)]]
            Pk = aX[k][occ[k] & occ[max(k-1, 0)]]
            spk = np.linalg.norm(vk, axis=1)
            ax.scatter(Pk[:, 0] / VOX, Pk[:, 1] / VOX, c=spk, s=3, cmap="viridis")
            st_ = max(1, len(Pk) // 300)
            u = vk / np.clip(np.linalg.norm(vk, axis=1, keepdims=True), 1e-9, None)
            ax.quiver(Pk[::st_, 0] / VOX, Pk[::st_, 1] / VOX, u[::st_, 0], u[::st_, 1],
                      color="white", alpha=0.7, width=0.003)
        ax.set_xlim(0, 1.0 / VOX); ax.set_ylim(0, 1.0 / VOX)
        _label(ax, f"{'MIGRATION  ' if j == 0 else ''}polar {polar_order:.2f}")
        for i in range(3):
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(d, "montage.png"), dpi=90, facecolor="black"); plt.close(fig)
    print(f"[smg] {sim.name}: target_score={tgt:.3f} buds={per_frame[-1]['n_bud']} "
          f"branch={per_frame[-1]['n_branch']} tube={per_frame[-1]['n_tube']} "
          f"polar={polar_order:.3f} growth={growth_ratio:.2f} -> {d}", flush=True)


if __name__ == "__main__":
    main()
