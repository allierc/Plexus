"""exp17: THE SLOW MODE OF THE RECORDING, AS MOVIES (Cedric, 2026-10-10: "slides showing movies of the slow mode as
extracted from the observed data, after the ARTR slide"). Local, the destriped recording in the Z-Brain atlas, no model.

THE SLOW BAND. Every neuron's dF/F band-passed to periods of LO_S-HI_S s (20-250 s, the band of the lead / lag analysis,
tools/exp17_brain_mean_lag.py; Butterworth order 4, forward-backward over the continuous recording), then z-scored per
neuron over the session: z_i(t). The slow brain mean b_s(t) is the band-passed dF/F averaged over the neurons. The share
of the slow-band variance in its first principal component (randomised SVD of z) and that component's correlation with
b_s say how much of the band is one brain-wide mode. The raw brain mean's power by period band (Welch) is given for the
whole session and per stimulus block, with each block's peak period.

THE LEAD. Per neuron from data/brain_mean_lag_<rec>.npz (the phase of the cross-spectrum with the brain mean over
periods 20-250 s, coherence-weighted; lead = -lag, > 0 the neuron moves first); per atlas region (exp17_atlas.REGIONS,
each neuron in its most specific one), the median over its neurons with a lag.

THE MOVIES (presentation/Movies/, the deck's convention: an mp4 and a same-basename png poster):
  slow_mode_2h       the whole recording at the data movies' 800 frames: the fish from above (head left) and from the
                     side, every neuron inside the atlas coloured by z_i(t) (red above its own mean, blue below, black at
                     0, clipped at +-Z_CLIP); under them the raster of z by region, head to tail, within a region the
                     earliest neuron first, averaged in NB bins of consecutive rows; the slow brain mean (green, dF/F)
                     with the stimulus blocks; a white bar at the frame.
  slow_mode_<block>  one block at every recorded frame: the same two views; under them the regions' slow-band means
                     (each region's z averaged over its neurons, its block mean removed), stacked in the order of the
                     regions' median lead, the earliest at the top, each coloured by its lead (red leads the brain mean
                     by more than SIG s, blue trails it by more, grey between), the bar at the frame. The first SKIP_S s
                     of a block (the band-pass's edge transient where the recording starts) set neither scale nor mean.

    PYTHONPATH=src:tools python tools/exp17_slow_mode.py [--blocks gain dark] [--no-movies] [--skip-2h] [--workers 16]
-> data/slow_mode_<rec>.json, presentation/Movies/slow_mode_2h.mp4 (+ .png), presentation/Movies/slow_mode_<block>.mp4
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
PRES = os.path.join(EXP, "presentation")
DT, LO_S, HI_S = 0.914, 20.0, 250.0      # the frame, s; the slow band's periods, s (as tools/exp17_brain_mean_lag.py)
Z_CLIP, NB, MIN_REGION, SIG, SKIP_S = 2.5, 1000, 400, 1.0, 60.0   # SKIP_S: the filter's edge transient, s
BANDS = (("> 250 s", 250.0, np.inf), ("60-250 s", 60.0, 250.0), ("10-60 s", 10.0, 60.0), ("< 10 s", 0.0, 10.0))
RED, BLUE, GREEN = "#ff4a4a", "#4a7bff", "#2ca02c"      # the lag slide's red / blue (lead / trail); the deck's brain-mean green
MID_X = (621 - 1) * 0.798                                 # Z-Brain's x extent, um: the view from above flips the atlas x


def band_pass(X, chunk=4000):
    """X [T, N] -> its LO_S-HI_S band, float32, column chunks (the whole recording filtered at once is 3 GB twice)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [(1.0 / HI_S) / (0.5 / DT), (1.0 / LO_S) / (0.5 / DT)], btype="band", output="sos")
    S = np.empty(X.shape, np.float32)
    for a in range(0, X.shape[1], chunk):
        S[:, a:a + chunk] = sosfiltfilt(sos, np.asarray(X[:, a:a + chunk], np.float64), axis=0).astype(np.float32)
    return S


def region_labels(reg, names):
    """Each neuron's most specific table region (the fewest neurons; exp17_artr.region_rows' rule): label into the kept
    (full, short) list, -1 = none."""
    from exp17_atlas import REGIONS
    rl = [(f, s_) for f, s_ in REGIONS if f in names]
    lab = np.full(reg.shape[0], -1)
    best = np.full(reg.shape[0], np.inf)
    for k, (full, _) in enumerate(rl):
        m = reg[:, names.index(full)]
        nk = m.sum()
        upd = m & (nk < best)
        lab[upd], best[upd] = k, nk
    return lab, rl


def power_shares(b, nper):
    """Welch power of a brain-mean trace by period band (BANDS), its peak period (s) and sd."""
    from scipy.signal import welch
    f, p = welch(b - b.mean(), fs=1.0 / DT, nperseg=min(nper, len(b)))
    per = 1.0 / np.maximum(f, 1e-9)
    tot = p[1:].sum()
    out = {name: float(p[(per > lo) & (per <= hi)].sum() / tot) if np.isfinite(hi) else float(p[per > lo].sum() / tot)
           for name, lo, hi in BANDS}
    out["peak_period_s"] = float(per[1:][np.argmax(p[1:])])
    out["sd"] = float(b.std())
    return out


def lut():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("slow", [BLUE, "black", RED])


def colours(cm, z):
    """RGBA of z through the blue-black-red map, clipped at +-Z_CLIP."""
    return cm(np.clip((z + Z_CLIP) / (2 * Z_CLIP), 0.0, 1.0))


def _spread(ys, lo, hi, gap):
    """Label positions spread apart where regions crowd (exp17_artr.twin_figure's rule), kept inside [lo, hi]."""
    ys = ys.copy()
    for _ in range(300):
        for k in range(1, len(ys)):
            if ys[k - 1] - ys[k] < gap:
                mid = (ys[k - 1] + ys[k]) / 2
                ys[k - 1], ys[k] = mid + gap / 2, mid - gap / 2
    ys += min(0.0, hi - ys.max())
    ys += max(0.0, lo - ys.min())
    return ys


def views(fig, xd, yd, zd, ins, rects=((0.01, 0.60, 0.49, 0.39), (0.51, 0.60, 0.48, 0.39)), size=0.35):
    """The fish from above (head left) and from the side, every inside neuron a dot (the deeper drawn first), the
    colours set per frame. -> [the two scatters, in drawing order of `order`], order."""
    order = np.flatnonzero(ins)[np.argsort(zd[ins])]
    scs = []
    for rect, Y, ttl in ((rects[0], yd, "from above, head left"), (rects[1], zd, "from the side")):
        a = fig.add_axes(rect)
        scs.append(a.scatter(xd[order], Y[order], s=size, c="0.2", lw=0, rasterized=True))
        a.set_aspect("equal")
        a.axis("off")
        fig.text(rect[0] + 0.01, rect[1] + rect[3] - 0.005, ttl, fontsize=11, va="top")
    a.plot([np.percentile(xd[ins], 99) - 100, np.percentile(xd[ins], 99)], [np.percentile(zd[ins], 0.5) - 20] * 2, color="w", lw=2)
    a.text(np.percentile(xd[ins], 99) - 50, np.percentile(zd[ins], 0.5) - 28, "100 µm", ha="center", va="top", fontsize=9)
    return scs, order


def encode(tmp, stem, fps, poster):
    import shutil
    import subprocess
    from plexus.tasks.trace_recording import _ffmpeg
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", stem + ".mp4"],
                   check=True)
    shutil.copy(os.path.join(tmp, f"{poster:05d}.png"), stem + ".png")
    shutil.rmtree(tmp)


def render(build, n, stem, fps, workers, poster, dpi=100):
    """Frames 0..n-1 of a figure, in a fork pool: build() -> (fig, update(i)); each worker builds once and saves its
    frames (exp17_artr.movie's scheme). -> the mp4 and the png poster at `stem`."""
    import multiprocessing as mp
    import tempfile
    import matplotlib.pyplot as plt
    tmp = tempfile.mkdtemp(prefix=os.path.basename(stem) + "_")

    def run(ids):
        fig, update = build()
        for i in ids:
            update(i)
            fig.savefig(os.path.join(tmp, f"{i:05d}.png"), dpi=dpi, facecolor="black")
        plt.close(fig)
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=run, args=(list(range(w, n, workers)),)) for w in range(workers)]
    for p_ in ps:
        p_.start()
    for p_ in ps:
        p_.join()
    encode(tmp, stem, fps, poster)
    print(f"[{os.path.basename(stem)}] {n} frames at {fps} fps -> {stem}.mp4")


def movie_2h(G, frames, stem, fps=25, workers=16):
    """The whole recording: two views coloured by z, the raster by region, the slow brain mean with the blocks."""
    import matplotlib.pyplot as plt
    Zf = np.ascontiguousarray(G["Z"][frames])                         # [n, N] at the movie's frames
    t_min = frames * DT / 60.0
    # the raster: inside neurons with a region, by region head to tail, within a region the earliest first
    lab, lead = G["lab"], G["lead"]
    sel = np.flatnonzero(G["ins"] & (lab >= 0))
    key_lead = np.where(np.isfinite(lead[sel]), -lead[sel], np.inf)    # no lag: after the others of its region
    order = sel[np.lexsort((key_lead, lab[sel]))]
    nb = NB
    edges = np.linspace(0, len(order), nb + 1).astype(int)
    img = np.stack([Zf[:, order[edges[q]:edges[q + 1]]].mean(1) for q in range(nb)])      # [nb, n]
    segs = []
    for k in range(len(G["rl"])):
        m = lab[order] == k
        if m.any():
            segs.append((G["rl"][k][1], int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1] + 1)))
    kb = len(order) / nb
    cm = lut()
    xd, yd, zd = G["xd"], G["yd"], G["zd"]
    off, names = G["off"], G["names"]
    bs = G["bs"]

    def build():
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(15, 8.4), facecolor="black")
        scs, order_v = views(fig, xd, yd, zd, G["ins"])
        X0, X1, Y0, H = 0.17, 0.98, 0.27, 0.30
        ar = fig.add_axes([X0, Y0, X1 - X0, H])
        ar.imshow(img, aspect="auto", cmap=cm, vmin=-1.0, vmax=1.0, extent=(t_min[0], t_min[-1], nb, 0),
                  interpolation="nearest")
        ar.set_xticklabels([])
        ar.set_yticks([])
        want = np.array([Y0 + H * (1 - (r0 + r1) / 2 / kb / nb) for _, r0, r1 in segs])
        gap_ = min(0.019, (H - 0.012) / max(len(segs) - 1, 1))
        ys = _spread(want, Y0 + 0.005, Y0 + H - 0.005, gap_)
        for (name, r0, r1), y_, w_ in zip(segs, ys, want):
            ar.axhline(r0 / kb, color="0.55", lw=0.4)
            fig.text(X0 - 0.03, y_, f"{name} ({r1 - r0:,})", fontsize=7.5 if gap_ > 0.015 else 6.5, ha="right",
                     va="center", color="0.85")
            fig.add_artist(plt.Line2D([X0 - 0.028, X0 - 0.010], [y_, w_], color="0.6", lw=0.5, transform=fig.transFigure))
        at = fig.add_axes([X0, 0.05, X1 - X0, 0.19])
        tt = np.arange(len(bs)) * DT / 60.0
        for k in range(len(names)):                                  # the stimulus blocks, alternating grey, named
            a0, a1 = off[k] * DT / 60.0, off[k + 1] * DT / 60.0
            if k % 2 == 0:
                at.axvspan(a0, a1, color="0.18", lw=0, zorder=0)
            at.text((a0 + a1) / 2, 0.97, names[k], transform=at.get_xaxis_transform(), ha="center", va="top", fontsize=8,
                    color="0.85")
        at.plot(tt, 1000 * bs, color=GREEN, lw=0.8)
        at.set_xlim(tt[0], tt[-1])
        at.set_xlabel("time, min", fontsize=10)
        at.set_ylabel("slow brain mean\ndF/F, x 1000", fontsize=9)
        at.tick_params(labelsize=8.5)
        b1 = ar.axvline(t_min[0], color="white", lw=1.4)
        b2 = at.axvline(t_min[0], color="white", lw=1.4)
        txt = fig.text(0.985, 0.985, "", ha="right", va="top", fontsize=11)

        def update(i):
            c = colours(cm, Zf[i][order_v])
            for sc in scs:
                sc.set_facecolors(c)
            b1.set_xdata([t_min[i]] * 2)
            b2.set_xdata([t_min[i]] * 2)
            k = int(np.clip(np.searchsorted(off, frames[i], side="right") - 1, 0, len(names) - 1))
            txt.set_text(f"{names[k]}   t = {t_min[i]:.1f} min")
        return fig, update
    render(build, len(frames), stem, fps, workers, poster=len(frames) // 2)
    return {"frames": int(len(frames)), "fps": fps, "seconds": len(frames) / fps, "raster_rows": nb,
            "raster_neurons": int(len(order))}


def movie_block(G, block, stem, fps=30, workers=16):
    """One block at every frame: the two views coloured by z, the regions' slow-band means stacked by their lead."""
    import matplotlib.pyplot as plt
    k = G["names"].index(block)
    f0, f1 = int(G["off"][k]), int(G["off"][k + 1])
    frames = np.arange(f0, f1)
    Zf = np.ascontiguousarray(G["Z"][f0:f1])
    R = G["regions"]                                                  # [(short, n, median lead)] by lead, earliest first
    tr = np.stack([Zf[:, G["members"][s_]].mean(1) for s_, _, _ in R], 1)
    t_s = (frames - f0) * DT
    steady = t_s >= SKIP_S                 # the band-pass's edge transient at the recording's start (the gain block
    tr = tr - tr[steady].mean(0)           # opens it) neither sets the scale nor the mean
    scale = 1.2 / max(float(np.percentile(np.abs(tr[steady]), 98.0)), 1e-6)
    tr = np.clip(scale * tr, -1.4, 1.4)    # the transient clipped to its own lane
    cm = lut()
    xd, yd, zd = G["xd"], G["yd"], G["zd"]

    def col_of(ld):
        return RED if ld > SIG else (BLUE if ld < -SIG else "0.75")

    def build():
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(15, 8.4), facecolor="black")
        scs, order_v = views(fig, xd, yd, zd, G["ins"])
        X0, X1 = 0.21, 0.98
        at = fig.add_axes([X0, 0.05, X1 - X0, 0.52])
        for j, (s_, n_, ld) in enumerate(R):
            y0 = -1.5 * j
            at.plot(t_s, y0 + tr[:, j], color=col_of(ld), lw=0.9)
            at.axhline(y0, color="0.3", lw=0.3)
            at.text(-0.01, y0, f"{s_} ({n_:,})  {ld:+.2f} s", transform=at.get_yaxis_transform(), ha="right",
                    va="center", fontsize=7.5, color=col_of(ld))
        at.set_xlim(t_s[0], t_s[-1])
        at.set_ylim(-1.5 * (len(R) - 1) - 1.4, 1.4)
        at.set_yticks([])
        at.set_xlabel(f"time in the {block} block, s", fontsize=10)
        at.tick_params(labelsize=8.5)
        for s in at.spines.values():
            s.set_visible(False)
        fig.text(X0 - 0.19, 0.575, "slow band, each region's mean z, its block mean removed; by median lead over the "
                 f"brain mean (red: leads by > {SIG:g} s, blue: trails by > {SIG:g} s)", fontsize=8.5, color="0.85")
        bar = at.axvline(t_s[0], color="white", lw=1.4)
        txt = fig.text(0.985, 0.985, "", ha="right", va="top", fontsize=11)

        def update(i):
            c = colours(cm, Zf[i][order_v])
            for sc in scs:
                sc.set_facecolors(c)
            bar.set_xdata([t_s[i]] * 2)
            txt.set_text(f"{block}   t = {t_s[i] / 60:.1f} min")
        return fig, update
    render(build, len(frames), stem, fps, workers, poster=len(frames) // 2)
    return {"frames": int(len(frames)), "fps": fps, "seconds": len(frames) / fps, "first_frame": f0, "last_frame": f1 - 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rec", default="zapbench_destripe")
    ap.add_argument("--blocks", nargs="*", default=["gain", "dark"])
    ap.add_argument("--no-movies", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-2h", action="store_true", help="keep the 2-h movie of an earlier run (its json entry carried over)")
    a = ap.parse_args()
    from sklearn.utils.extmath import randomized_svd
    from exp17_slides import movie_frames
    from plexus.paths import graphs_data_path
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, ins, reg, names = za["atlas_um"].astype(np.float64), za["inside"], za["regions"], [str(x) for x in za["names"]]
    lab, rl = region_labels(reg, names)
    z = np.load(graphs_data_path("zebrafish", f"{a.rec}_recording.npz"))
    X = np.asarray(z["dff"], np.float32)
    off, bnames = [int(v) for v in z["offsets"]], [str(x) for x in z["names"]]
    T, N = X.shape
    lg = np.load(os.path.join(EXP, "data", f"brain_mean_lag_{a.rec}.npz"))
    lead = np.where(lg["with_lag"], -lg["lag_s"].astype(np.float64), np.nan)
    b = X.mean(1, dtype=np.float64)
    bc = (b - b.mean()) / b.std()
    r_b = np.concatenate([(((X[:, c:c + 8000] - X[:, c:c + 8000].mean(0)) / np.maximum(X[:, c:c + 8000].std(0), 1e-9))
                           * bc[:, None].astype(np.float32)).mean(0) for c in range(0, N, 8000)])
    print(f"[slow mode] {T} frames x {N:,} neurons; band {LO_S:g}-{HI_S:g} s")
    S = band_pass(X)
    del X
    bs = S.mean(1, dtype=np.float64)
    sd = np.maximum(S.std(0), 1e-9)
    Z = (S - S.mean(0)) / sd
    del S
    U, s_, Vt = randomized_svd(Z, n_components=3, n_iter=4, random_state=0)
    tot = float((Z.astype(np.float64) ** 2).sum()) if N * T < 2e9 else float(sum((Z[:, c:c + 8000].astype(np.float64) ** 2).sum() for c in range(0, N, 8000)))
    pc_share = [float(v ** 2 / tot) for v in s_]
    pc1 = U[:, 0] * s_[0]
    r_pc1 = float(abs(np.corrcoef(pc1, bs)[0, 1]))
    # the regions, by median lead (earliest first)
    members, regions = {}, []
    for k, (_, short) in enumerate(rl):
        m = np.flatnonzero(ins & (lab == k))
        if len(m) >= MIN_REGION:
            ld = lead[m][np.isfinite(lead[m])]
            members[short] = m
            regions.append((short, int(len(m)), float(np.median(ld)) if len(ld) else float("nan"),
                            float((ld > SIG).mean()) if len(ld) else float("nan"), int(len(ld))))
    regions.sort(key=lambda r_: -r_[2])
    xd, yd, zd = A[:, 1], MID_X - A[:, 0], A[:, 2]
    G = {"Z": Z, "lab": lab, "rl": rl, "lead": lead, "ins": ins, "xd": xd, "yd": yd, "zd": zd, "off": np.array(off),
         "names": bnames, "bs": bs, "regions": [(s_, n_, ld) for s_, n_, ld, _, _ in regions], "members": members}
    out = {"recording": a.rec, "frames": T, "neurons": N, "neurons_inside_atlas": int(ins.sum()), "band_s": [LO_S, HI_S],
           "z_clip": Z_CLIP, "lead_file": f"data/brain_mean_lag_{a.rec}.npz",
           "pc_share_of_slow_band_variance": pc_share, "pc1_r_with_slow_brain_mean": r_pc1,
           "median_r_neuron_vs_brain_mean": float(np.median(r_b)),
           "brain_mean_power": {"session": power_shares(b, 1024)},
           "slow_brain_mean_sd_dff": float(bs.std()),
           "regions_by_lead": [{"region": s_, "neurons": n_, "median_lead_s": ld, "lead_gt_1s_share": sh, "with_lag": nl}
                               for s_, n_, ld, sh, nl in regions],
           "movies": {}}
    for k, nm in enumerate(bnames):
        out["brain_mean_power"][nm] = power_shares(b[off[k]:off[k + 1]], 256)
        out["brain_mean_power"][nm]["slow_sd_dff"] = float(bs[off[k]:off[k + 1]].std())
    path = os.path.join(EXP, "data", f"slow_mode_{a.rec}.json")
    if os.path.exists(path):                                           # an earlier run's movie entries, carried over
        out["movies"].update(json.load(open(path)).get("movies", {}))
    if not a.no_movies:
        frames = movie_frames()
        if not a.skip_2h:
            out["movies"]["slow_mode_2h"] = movie_2h(G, frames, os.path.join(PRES, "Movies", "slow_mode_2h"), workers=a.workers)
        for blk in a.blocks:
            out["movies"][f"slow_mode_{blk.replace(' ', '_')}"] = movie_block(
                G, blk, os.path.join(PRES, "Movies", f"slow_mode_{blk.replace(' ', '_')}"), workers=a.workers)
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "regions_by_lead"}, indent=1))
    print("regions by lead:", [(r_["region"], round(r_["median_lead_s"], 2)) for r_ in out["regions_by_lead"]])
    print(f"-> {path}")


if __name__ == "__main__":
    main()
