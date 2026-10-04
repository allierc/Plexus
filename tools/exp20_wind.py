"""exp20: THE INFORMATION FLUX AS A WEATHER MAP, exp17's slides 28-29 (tools/exp17_wind.py) on a gut-brain run.

    PYTHONPATH=src:tools python tools/exp20_wind.py <run> [<run> ...] [--cells 160] [--particles 6000]

exp17_wind's `fields` and `_Map` are used unchanged (drawn head LEFT through exp20_modulation.patch, the gut-brain
recordings lying along x with the head at +x); its `render` is copied below with two changes only: the time axis uses
the recording's own frame period (exp17's literal 0.914 s), and its stills and copies go to exp20's png/. Writes
<run>/results/movie_wind.mp4 and <run>/results/<run>_wind_fields.npz (the time-averaged fields, exp17_wind's).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
import exp20_modulation  # noqa: E402
exp20_modulation.patch()
from exp17_wind import fields, _Map, FADE, SUB, SIGMA_UM  # noqa: E402,F401


def render(name, D, n_part=6000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks import trace_recording as TR
    rng = np.random.default_rng(seed)
    act, ins = D["act"], D["inside"]
    vref = max(np.percentile(D["speed"]["ex"][:, ins], 99), np.percentile(D["speed"]["in"][:, ins], 99))   # ONE scale
    maps = {"ex": _Map(D["wind"]["ex"], ins, vref, (1.0, 0.35, 0.22), n_part, rng),
            "in": _Map(D["wind"]["in"], ins, vref, (0.35, 0.62, 1.0), n_part, rng)}
    F, _, ny, nx = D["wind"]["ex"].shape
    S = maps["ex"].S
    rec, fr = D["rec"], D["fr"]
    cond = rec["condition"][fr]
    tm = fr * D["frame_s"] / 60                    # the recording's own frame period (exp17: 0.914)
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    names = [str(x) for x in rec["names"]]
    mo, mp = rec["dff"][fr].mean(1), D["pred"].mean(1)
    amax = np.percentile(act[:, ins], 99)
    tmp = tempfile.mkdtemp(prefix="wind_")
    fig = plt.figure(figsize=(16, 6.4), facecolor="black")
    ims = {}
    for j_, (k, lab) in enumerate((("ex", "EXCITATORY flux (W phi(z) > 0)"), ("in", "INHIBITORY flux (W phi(z) < 0)"))):
        ax = fig.add_axes([0.005 + 0.5 * j_, 0.30, 0.49, 0.60])
        ax.axis("off")
        ims[k] = ax.imshow(np.zeros((ny * S, nx * S, 3)), origin="lower", interpolation="bilinear")
        if D.get("landmarks"):                              # the paper's stations, yellow (Cedric, 2026-10-04)
            from exp20_landmarks import annotate
            gx0, gy0, gh = D["grid"]
            annotate(ax, D["landmarks"], lambda x, y: ((x - gx0) / gh * S, (y - gy0) / gh * S))
        fig.text(0.01 + 0.5 * j_, 0.965, lab, color="white", fontsize=12, va="top")
    fig.text(0.20, 0.925, f"wind = the messages, sender to receiver, smoothed over {SIGMA_UM:g} um; brighter = stronger, one scale for both maps; under "
             "them the learned dF/F", color="0.8", fontsize=9, va="top")
    t_txt = fig.text(0.01, 0.925, "", color="0.75", fontsize=9, va="top")
    m_ = fig.add_axes([0.05, 0.08, 0.90, 0.15])
    m_.set_facecolor("black")
    for k_, sp in m_.spines.items():
        sp.set_visible(k_ in ("left", "bottom"))
        sp.set_color("0.5")
    for i_, (a_, b_) in enumerate(zip(st, en)):
        m_.axvspan(tm[a_], tm[b_], color=("0.30" if i_ % 2 else "0.18"), alpha=0.6, lw=0)
        m_.text((tm[a_] + tm[b_]) / 2, 1.02, names[int(cond[a_])], color="0.75", fontsize=7, ha="center", va="bottom",
                transform=m_.get_xaxis_transform())
    m_.plot(tm, mo, color="#2ca02c", lw=0.7)
    m_.plot(tm, mp, color="white", lw=0.7)
    for f_, site_ in zip(rec["trials"][:, 0], rec["trials"][:, 2]):     # the UV pulses: yellow on the gut, grey off the fish
        m_.axvline(f_ * D["frame_s"] / 60, color="#ffd54f" if int(site_) != 1 else "0.55", lw=0.6, alpha=0.8)
    m_.set_xlim(tm[0], tm[-1])
    m_.set_yticks([])
    m_.tick_params(colors="0.6", labelsize=7)
    fig.text(0.05, 0.255, "brain-mean dF/F: recorded (green), learned (white); UV pulses: yellow on the gut, grey off the "
             "fish; time, min", color="0.7", fontsize=8)
    cur = m_.axvline(tm[0], color="#ff7f0e", lw=0.9)
    for f in range(F):
        bg = np.clip(act[f] / amax, 0, 1)
        if D.get("bg_style") == "activity":             # the activity itself, as the run movies draw it (inferno)
            bgc = (matplotlib.colormaps["inferno"](bg)[..., :3] * ins[..., None]).repeat(S, 0).repeat(S, 1) * 0.75
        else:
            bgc = (np.stack([bg * 0.45, bg * 0.40, bg * 0.45], -1) * ins[..., None]).repeat(S, 0).repeat(S, 1) * 0.45
        for k, mp_ in maps.items():
            mp_.step(f)
            ims[k].set_data(np.clip(bgc + mp_.canvas, 0, 1))
        t_txt.set_text(f"{names[int(cond[f])]}   t = {tm[f]:5.1f} min")
        cur.set_xdata([tm[f]] * 2)
        fig.savefig(os.path.join(tmp, f"{f:05d}.png"), dpi=100, facecolor="black")
    plt.close(fig)
    path = os.path.join(D["out"], "results", "movie_wind.mp4")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-framerate", "25", "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", path], check=True)
    shutil.copy(os.path.join(tmp, f"{F // 2:05d}.png"), os.path.join(EXP, "png", f"wind_frame_{name}.png"))
    shutil.copy(path, os.path.join(EXP, "png", f"wind_movie_{name}.mp4"))          # Cedric follows the work in png/
    shutil.rmtree(tmp)
    print(f"[wind] {path}")
    return path


def gut_flow(name, D, ev_s=20.0):
    """THE FLOW DURING THE GUT RESPONSE (exp20's twin of exp17's slide 29: the brains differ fish to fish, so no
    consensus over fish; within a fish, the movie frames inside 0-ev_s after a gut pulse -- every site but 1, off the
    fish -- against every other frame): the mean wind speed, excitatory and inhibitory, in each; their ratio; the net
    direction along the body (toward_tail: +1 all toward the tail, -1 all toward the head; x grows toward the tail,
    the head drawn left); and the map of the speed difference (gut - rest).
    Writes data/gutflow_<run>.json, data/gutflow_<run>.npz (the two difference maps) and png/gutflow_<run>.png."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec, fr, ins = D["rec"], D["fr"], D["inside"]
    tr = rec["trials"]
    w = int(round(ev_s / D["frame_s"]))
    gut = np.zeros(len(fr), bool)
    for f, site in zip(tr[:, 0].astype(int), tr[:, 2].astype(int)):
        if site != 1:
            gut |= (fr >= f) & (fr < f + w)
    out = {"run": name, "frames_gut": int(gut.sum()), "frames_rest": int((~gut).sum()), "ev_s": ev_s}
    if gut.sum() < 3:
        json.dump(out, open(os.path.join(EXP, "data", f"gutflow_{name}.json"), "w"), indent=1)
        return out
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), facecolor="black")
    maps = {}
    for a_, k, cm in zip(ax, ("ex", "in"), ("Reds", "Blues")):
        W = D["wind"][k]                                         # [F, 2, ny, nx]
        sp = np.linalg.norm(W, axis=1)                           # [F, ny, nx]
        g_, r_ = sp[gut][:, ins].mean(), sp[~gut][:, ins].mean()
        vx = lambda m: float((W[m][:, 0][:, ins]).sum() / (np.abs(W[m][:, 0][:, ins]).sum() + 1e-12))
        out[k] = {"speed_gut": float(g_), "speed_rest": float(r_), "ratio": float(g_ / max(r_, 1e-12)),
                  "toward_tail_gut": vx(gut), "toward_tail_rest": vx(~gut)}
        dmap = sp[gut].mean(0) - sp[~gut].mean(0)
        dmap[~ins] = np.nan
        maps[k] = dmap
        lim = np.nanpercentile(np.abs(dmap), 99)
        a_.imshow(dmap, cmap="RdBu_r" if k == "ex" else "PuOr_r", vmin=-lim, vmax=lim, origin="lower")
        if D.get("landmarks"):
            from exp20_landmarks import annotate
            gx0, gy0, gh = D["grid"]
            annotate(a_, D["landmarks"], lambda x, y: ((x - gx0) / gh, (y - gy0) / gh), fontsize=8)
        a_.set_title(f"{'excitatory' if k == 'ex' else 'inhibitory'} flow speed: gut windows - rest "
                     f"(x{out[k]['ratio']:.2f})", color="white", fontsize=9)
        a_.axis("off")
    fig.text(0.5, 0.01, "yellow: the paper's stations, inferred from this fish's gut-responsive clusters (no atlas)",
             color="#ffeb3b", fontsize=8, ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(EXP, "png", f"gutflow_{name}.png"), dpi=130, facecolor="black")
    plt.close(fig)
    json.dump(out, open(os.path.join(EXP, "data", f"gutflow_{name}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(EXP, "data", f"gutflow_{name}.npz"), ex=maps["ex"], inh=maps["in"],   # the montage's
                        grid=np.array(D["grid"]), inside=ins)                                         # (exp20_flow_montage)
    print(f"[gutflow] {name}: excitatory x{out['ex']['ratio']:.2f}, inhibitory x{out['in']['ratio']:.2f} in the gut "
          f"windows; toward the tail (ex) {out['ex']['toward_tail_gut']:+.2f} gut / {out['ex']['toward_tail_rest']:+.2f} rest")
    return out


def run(name, cells=160, particles=6000, device="cuda:0"):
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    D = fields(name, cells, device)
    rn = T.load(name)["task"]["reference"]["trace_recording"]
    D["frame_s"] = float(np.median(np.diff(TR.load(rn)["t_s"])))
    from exp20_landmarks import landmarks
    try:
        D["landmarks"] = landmarks(rn)                     # inferred from the fish's gut-responsive clusters
    except FileNotFoundError:
        D["landmarks"] = None
    os.makedirs(os.path.join(EXP, "png"), exist_ok=True)
    gut_flow(name, D)
    return render(name, D, particles)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cells", type=int, default=160)
    ap.add_argument("--particles", type=int, default=6000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    for r in a.runs:
        run(r, a.cells, a.particles, a.device)
