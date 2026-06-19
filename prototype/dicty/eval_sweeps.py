"""eval_sweeps.py -- run a structured sweep batch: 16 single-parameter sweeps x 16 values = 256
simulations, from a base spec + sweep_plan.json the agent writes. Fast (~2-3s/sim), so the agent
call (minutes) is amortized over 256 runs.

For each sweep it writes a figure `sweep_<i>_<param>.png` = response curves (inner_mass & loss vs
the swept value) + a morphology STRIP (final SIM density at the 16 values, with REAL density at the
end). For the single best config across all 256 it writes `best_montage.png` (REAL/SIM/SIM-dens/
REAL-dens over time). Metrics go to sweep_results.json. Run as a FRESH process so operator/engine
CODE edits are picked up.

sweep_plan.json schema:
  {"base_spec": "specs/dicty_loop_base.yaml",
   "sweeps": [{"param": "sense.gain", "values": [..16..]}, ... up to 16 ]}
param paths: "cell.n", "dt", "vmax", "camp.diffusion", "camp.decay", or "<op>.<field>"
(e.g. "sense.gain", "inflow.rate", "spring.kadh", "relay.strength").

    PYTHONPATH=../../src DICTY_DEVICE=cuda:0 python eval_sweeps.py
"""
import os, sys, json, glob, traceback
sys.path.insert(0, "."); sys.path.insert(0, ".."); sys.path.insert(0, "/workspace/Plexus/src")
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scenario_schema import load
import importlib

def _autodetect_engine():
    """Select the engine module. Env override wins; otherwise sniff the base spec from
    sweep_plan.json for MPM-only tokens. This is the LAUNCHER-INDEPENDENT version of the
    autodetect that was first put in dicty_loop.py (B32) — needed because a long-running
    loop process can hold a stale launcher in memory while the on-disk patches are not
    yet executed (B33/B34 third infrastructure failure)."""
    forced = os.environ.get("DICTY_ENGINE")
    if forced:
        return forced
    try:
        plan_path = os.environ.get("DICTY_SWEEP_PLAN", "sweep_plan.json")
        plan = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), plan_path)))
        base = plan.get("base_spec", "specs/dicty_loop_base.yaml")
        with open(base) as f:
            blob = f.read()
        if "inflow_mpm" in blob or "op: mpm" in blob or "particle:" in blob:
            return "dicty_engine_mpm"
    except Exception:
        pass
    return "dicty_engine"

_ENGINE_NAME = _autodetect_engine()
print(f"[eval_sweeps] DICTY_ENGINE={_ENGINE_NAME}", flush=True)
dicty_engine = importlib.import_module(_ENGINE_NAME)
import opt_dicty as O
from make_target import CROP, WIN, cell_mask
from plexus.models.registry import _OPERATOR_REGISTRY  # used by _preflight()

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"
GREEN = LinearSegmentedColormap.from_list("c", ["#000", "#0a3a0a", "#1d7a1d", "#7CFC00"])
DEVICE = os.environ.get("DICTY_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
FRAMES = 240


PFX = os.environ.get("DICTY_OUT_PREFIX", "")            # isolate MPM-arm outputs from the point-cell loop


def set_param(sc, path, val):
    if path == "cell.n":     sc.sets["cell"]["n"] = int(round(val)); return
    if path == "cell.youngs":
        for t in sc.sets["cell"]["types"].values(): t["youngs"] = float(val)
        return
    if path == "dt":         sc.dt = float(val); return
    if path == "vmax":       sc.vmax = float(val); return
    if path == "seed":       sc.seed = int(round(val)); return
    if path == "n_frames":   sc.n_frames = int(round(val)); return
    head, key = path.split(".", 1)
    if head == "camp":       sc.fields["camp"][key] = float(val); return
    if head == "inhib":      sc.fields["inhib"][key] = float(val); return
    if head == "particle":   sc.sets["particle"][key] = (int(round(val)) if key == "per_parent" else float(val)); return
    for o in sc.operators:                              # "<op>.<param>"
        if o.op == head:
            o.params[key] = float(val) if key != "seed" else int(round(val))
    return


W_GR, W_MOUND = 0.2, 0.3           # weights of g(r) and mound-count vs the primary SSIM term


def run_one(base_path, overrides, t_frac, target, vtarget, real_fov):
    """loss = late-weighted [ (1-SSIM_full-FOV-image) + W_GR*g(r)-MSE ] + W_MOUND*mound-count-pen
    + W_VEL*velocity. SSIM-on-image is the primary morphology match (sound because the IC is the
    real frame-0 positions); g(r)+mound-count are the invariant structure terms. inner_mass is a
    reported DIAGNOSTIC only -- it is gameable by a single over-tight blob."""
    sc = load(base_path); sc.n_frames = FRAMES; sc.record_every = max(1, FRAMES // 48)
    for p, v in overrides.items():
        set_param(sc, p, v)
    _, hist = dicty_engine.run(sc, device=DEVICE)
    T = len(hist); rec = (t_frac * (T - 1)).astype(int); K = len(rec)
    simd = O.sim_density_seq([hist[i]["pos"] for i in rec])     # COM-centred (for the diagnostic strip)
    sim_fov = [O.fov_image(hist[i]["pos"]) for i in rec]        # full-FOV (for SSIM + g(r))
    fw = np.linspace(.5, 1.5, K); fw /= fw.sum()
    ssim_term = float(sum(fw[k] * (1.0 - O.image_ssim(sim_fov[k], real_fov[k])) for k in range(K)))
    gr_term = float(sum(fw[k] * ((O.radial_autocorr(sim_fov[k]) - O.radial_autocorr(real_fov[k])) ** 2).sum()
                        for k in range(K)))
    nm, nm_real = O.n_mounds(sim_fov[-1]), O.n_mounds(real_fov[-1])
    mound_pen = ((nm - nm_real) / max(nm_real, 1)) ** 2
    vfeat = O.sim_speed_feats(hist, rec)
    vloss = float((fw[:, None] * (vfeat - vtarget) ** 2).sum())
    loss = ssim_term + W_GR * gr_term + W_MOUND * mound_pen + O.W_VEL * vloss
    inner = float(O.radial_profile(simd[-1])[:3].sum())         # diagnostic only
    # B25: morph_score — secondary morphology metric, REPORTED ALONGSIDE loss.
    # Tests whether parameter-surface flatness is metric-induced (Est #42 flag).
    # w_peak * peak_count_match + w_dens * per_spot_concentration_match.
    # per-spot concentration: fraction of total mass inside the top-N_real peak boxes.
    sim_img = sim_fov[-1]; real_img = real_fov[-1]
    tot = float(sim_img.sum()) + 1e-9
    half = max(1, int(0.04 * sim_img.shape[0]))                 # ~mound radius window
    def _peak_mass(img, k):
        flat = img.flatten()
        if flat.size == 0 or k <= 0: return 0.0
        idx = np.argpartition(flat, -min(k, flat.size))[-k:]
        ys, xs = np.unravel_index(idx, img.shape)
        m = 0.0
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - half), min(img.shape[0], y + half + 1)
            x0, x1 = max(0, x - half), min(img.shape[1], x + half + 1)
            m += float(img[y0:y1, x0:x1].sum())
        return m
    real_peak_frac = _peak_mass(real_img, max(int(nm_real), 1)) / (float(real_img.sum()) + 1e-9)
    sim_peak_frac = _peak_mass(sim_img, max(int(nm_real), 1)) / tot
    peak_match = abs(nm - nm_real) / max(nm_real, 1)
    dens_match = (sim_peak_frac - real_peak_frac) ** 2
    morph_score = 1.0 * peak_match + 0.5 * dens_match
    return dict(loss=loss, inner=inner, n_mounds=int(nm), ssim=round(1 - ssim_term, 3),
                morph_score=float(morph_score), peak_frac=round(float(sim_peak_frac), 3),
                n_final=hist[-1]["count"]), simd[-1], hist


def _real_fov_images(t_frac):
    """Full-FOV real cell images (no recentering, y-up, [0,1]) at the K timepoints, for SSIM/g(r)."""
    cap = cv2.VideoCapture(VIDEO); n = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4))
    x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W); y0, y1 = int(CROP["y0"] * H), int(CROP["y1"] * H)
    imgs = []
    for t in t_frac:
        cap.set(1, int(t * (n - 1))); _, fr = cap.read(); p = fr[y0:y1, x0:x1]
        m = cell_mask(p); ys, xs = np.nonzero(m)
        xy = np.stack([xs / p.shape[1], 1 - ys / p.shape[0]], 1) if len(xs) else np.zeros((1, 2))
        imgs.append(O.fov_image(xy))
    cap.release()
    return imgs


def _preflight(base):
    """Fail-fast: if the base spec schedules an operator that is NOT registered with the loaded
    engine, abort BEFORE running 256 sims (which would otherwise all fail silently with
    'operator X not in registry' and produce all-NaN sweep_results.json — the B32 failure mode)."""
    sc = load(base)
    spec_ops = {o.op for o in sc.operators}
    missing = sorted(spec_ops - set(_OPERATOR_REGISTRY.keys()))
    if missing:
        eng = os.environ.get('DICTY_ENGINE', 'dicty_engine')
        raise SystemExit(
            f"\n[eval_sweeps] PREFLIGHT FAILURE: base spec '{base}' schedules operators not registered "
            f"under DICTY_ENGINE='{eng}': {missing}\n"
            f"  Available: {sorted(_OPERATOR_REGISTRY.keys())}\n"
            f"  Hint: set DICTY_ENGINE=dicty_engine_mpm for MPM specs, or revert to point-cell base.\n")


def main():
    plan = json.load(open(os.path.join(HERE, os.environ.get("DICTY_SWEEP_PLAN", "sweep_plan.json"))))
    base = plan.get("base_spec", "specs/dicty_loop_base.yaml")
    _preflight(base)
    t_frac, target, vtarget = O._target()
    real_inner = float(O.radial_profile(target[-1])[:3].sum())
    rdens_last = target[-1]
    real_fov = _real_fov_images(t_frac)            # full-FOV real cell images for SSIM + g(r)
    out = {"base_spec": base, "real_inner_mass": round(real_inner, 3),
           "real_n_mounds": O.n_mounds(real_fov[-1]), "sweeps": []}
    best = {"loss": 1e9}

    for si, sw in enumerate(plan["sweeps"]):
        param = sw["param"]; values = list(sw["values"]); fixed = sw.get("fixed", {})
        inners, losses, finals = [], [], []
        morph_scores, nmounds, peak_fracs = [], [], []
        for v in values:
            try:
                overrides = dict(fixed); overrides[param] = v
                m, dlast, hist = run_one(base, overrides, t_frac, target, vtarget, real_fov)
                inners.append(m["inner"]); losses.append(m["loss"]); finals.append(dlast)
                morph_scores.append(m["morph_score"]); nmounds.append(m["n_mounds"]); peak_fracs.append(m["peak_frac"])
                if m["loss"] < best["loss"]:
                    best = {"loss": m["loss"], "inner": m["inner"], "param": param, "value": v,
                            "n_final": m["n_final"], "hist": hist}
            except Exception as e:
                inners.append(np.nan); losses.append(np.nan); finals.append(np.zeros((O.GRID, O.GRID)))
                morph_scores.append(np.nan); nmounds.append(0); peak_fracs.append(np.nan)
                print(f"  [{param}={v}] failed: {e}", flush=True)
        # per-sweep figure: curves + morphology strip
        fig = plt.figure(figsize=(2 + 1.1 * len(values), 4.2)); fig.patch.set_facecolor("black")
        ax = fig.add_axes([0.06, 0.55, 0.9, 0.38]); ax.set_facecolor("black")
        ax.plot(values, inners, "o-", color="orange", label="inner_mass")
        ax.axhline(real_inner, color="cyan", ls="--", lw=1, label=f"REAL inner={real_inner:.2f}")
        ax2 = ax.twinx(); ax2.plot(values, losses, "s-", color="#888", label="loss")
        ax.set_xlabel(param, color="w"); ax.set_ylabel("inner_mass", color="orange")
        ax2.set_ylabel("loss", color="#aaa"); ax.tick_params(colors="w"); ax2.tick_params(colors="w")
        ax.legend(loc="upper left", fontsize=7); ax.set_title(f"sweep {si}: {param}", color="w")
        n = len(values)
        for j, d in enumerate(finals):                  # morphology strip: final SIM density per value
            a = fig.add_axes([0.06 + 0.9 * j / (n + 1), 0.06, 0.9 / (n + 1) * 0.95, 0.34])
            a.imshow(d.T, origin="lower", cmap="inferno"); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{values[j]:.3g}", color="w", fontsize=6)
        a = fig.add_axes([0.06 + 0.9 * n / (n + 1), 0.06, 0.9 / (n + 1) * 0.95, 0.34])
        a.imshow(rdens_last.T, origin="lower", cmap="inferno"); a.set_xticks([]); a.set_yticks([])
        a.set_title("REAL", color="cyan", fontsize=6)
        fp = os.path.join(HERE, f"{PFX}sweep_{si}_{param.replace('.', '_')}.png")
        plt.savefig(fp, dpi=60, facecolor="black"); plt.close(fig)
        bi = int(np.nanargmin(losses)) if np.any(np.isfinite(losses)) else 0
        bi_morph = int(np.nanargmin(morph_scores)) if np.any(np.isfinite(morph_scores)) else 0
        out["sweeps"].append(dict(idx=si, param=param, values=[round(float(v), 4) for v in values],
                                  inner_mass=[round(float(x), 3) for x in inners],
                                  loss=[round(float(x), 4) for x in losses],
                                  morph_score=[round(float(x), 4) for x in morph_scores],
                                  n_mounds=[int(x) for x in nmounds],
                                  peak_frac=[round(float(x), 3) if np.isfinite(x) else None for x in peak_fracs],
                                  best_value=round(float(values[bi]), 4), best_loss=round(float(losses[bi]), 4),
                                  best_inner=round(float(inners[bi]), 3),
                                  best_morph_value=round(float(values[bi_morph]), 4),
                                  best_morph_score=round(float(morph_scores[bi_morph]), 4),
                                  figure=os.path.basename(fp)))
        print(f"sweep {si} {param}: best loss={losses[bi]:.3f} @ {values[bi]:.3g} (inner={inners[bi]:.2f})", flush=True)

    # full montage of the single best config across all 256
    if "hist" in best:
        _render_best(best, t_frac, target)
        out["best"] = {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in best.items() if k != "hist"}
        out["best"]["montage"] = f"{PFX}best_montage.png"
    json.dump(out, open(os.path.join(HERE, f"{PFX}sweep_results.json"), "w"), indent=2)
    n_runs = sum(len(s["values"]) for s in plan["sweeps"])
    print(f"wrote sweep_results.json ({len(plan['sweeps'])} sweeps, {n_runs} sims), "
          f"best loss={best['loss']:.3f} inner={best.get('inner', 0):.2f}", flush=True)


def _render_best(best, t_frac, target):
    hist = best["hist"]; T = len(hist); fmax = max(float(h["field"].max()) for h in hist) or 1.0
    rec = (t_frac * (T - 1)).astype(int)
    simd = O.sim_density_seq([hist[i]["pos"] for i in rec])
    cap = cv2.VideoCapture(VIDEO); n = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4))
    x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W); y0, y1 = int(CROP["y0"] * H), int(CROP["y1"] * H)
    reals = []
    for tf in t_frac:
        cap.set(1, int(tf * (n - 1))); _, fr = cap.read()
        reals.append(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)[::-1])
    cap.release()
    K = len(t_frac)
    fig, axs = plt.subplots(4, K, figsize=(2.6 * K, 10)); fig.patch.set_facecolor("black")
    for c in range(K):
        h = hist[rec[c]]
        axs[0, c].imshow(reals[c], origin="lower"); axs[0, c].set_title(f"REAL t={t_frac[c]:.2f}", color="w", fontsize=8)
        axs[1, c].imshow(h["field"].T, origin="lower", extent=[0, 1, 0, 1], cmap=GREEN, vmin=0, vmax=fmax * 0.5)
        axs[1, c].scatter(h["pos"][:, 0], h["pos"][:, 1], s=2, c="#FFA500"); axs[1, c].set_xlim(0, 1); axs[1, c].set_ylim(0, 1)
        axs[1, c].set_title(f"SIM N={h['count']}", color="w", fontsize=8)
        axs[2, c].imshow(simd[c].T, origin="lower", cmap="inferno")
        axs[3, c].imshow(target[c].T, origin="lower", cmap="inferno")
        for r in range(4):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    for r, lab in enumerate(["REAL movie", "SIM cells", "SIM density", "REAL density"]):
        axs[r, 0].set_ylabel(lab, color="w", fontsize=10)
    fig.suptitle(f"BEST: {best['param']}={best['value']:.3g}  loss={best['loss']:.4f}  inner={best['inner']:.2f} (real {O.radial_profile(target[-1])[:3].sum():.2f})",
                 color="w", fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, f"{PFX}best_montage.png"), dpi=58, facecolor="black"); plt.close(fig)


if __name__ == "__main__":
    main()
