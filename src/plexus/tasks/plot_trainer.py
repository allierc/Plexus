"""A rollout movie for a fitted model: what went in, what should have come out, what did.

    from plexus.tasks.plot_trainer import rollout_movie
    rollout_movie(inp, gt, pred, out="results/rollout.mp4", dt=1/60, unit="deg")

    python -m plexus.tasks.plot_trainer --all              # every run under log/task/
    python -m plexus.tasks.plot_trainer eye_rig zf_eye_rig
    python -m plexus.tasks.plot_trainer --circuit config/training/neural_eye/zf_eye_rig.yaml   # circuit_movie

THREE PANELS ARE THE CLAIM AND THREE ARE THE EVIDENCE. The top row is the run as it happened --
the stimulus, the ground truth, the model's answer -- and the bottom row is what a reader needs to
judge it: the two overlaid on one axis, the residual on the target's own scale, and the error
accumulating over the rollout.

    a  input        b  ground truth      c  inference
    d  overlaid     e  residual          f  error so far

WHY A MOVIE AND NOT THE STILL. A still of a fitted trace shows agreement at the end; it cannot
show WHEN the model lost the target, and that is usually the whole story -- an integrator that
drifts after four seconds and a filter that never had the phase look identical at t = T. The
sweep line is the reading: everything left of it has happened, everything right of it has not.

TWO BACKGROUNDS, AND THEY ARE NOT A PREFERENCE. A trace is read against the page, so it is drawn
black-on-white like every other analysis figure in this repository. A FIELD is read as light --
the eye judges a colour map against a dark surround, and a white one washes out the low end of
every scale -- so a field movie is white-on-black, the same convention `live_movie` uses for the
simulation movies. `kind="auto"` decides from the data's rank rather than from a flag: a `[T, N]`
array is traces, a `[T, H, W]` array is a field.

CALLABLE FROM THREE PLACES, which is why the core takes ARRAYS and not a run name. During
training (the arrays are already in hand, no checkpoint exists yet), at the end of training (same,
plus a path), and afterwards from a finished run (`from_run` loads the checkpoint and rolls it
out). Only the last one touches the disk.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# THE TWO PALETTES. `light` is the repository's analysis convention -- white ground, no box, the
# label above the panel -- and `dark` is `live_movie`'s, for anything read as emitted light.
THEMES = {
    "light": dict(bg="white", ink="0.15", muted="0.45", gt="#2e8b4f", pred="#000000",
                  resid="#c0522a", stim="#4a4a4a", sweep="#c0272a"),
    "dark": dict(bg="black", ink="0.92", muted="0.55", gt="#5fd08a", pred="#ffffff",
                 resid="#ff8a5c", stim="#9aa0a6", sweep="#ff4d4d"),
}


def _panel(ax, th, xlabel=None, ylabel=None, letter=None):
    """One panel in the repository's style: no box, the letter above it, not bold."""
    ax.set_facecolor(th["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["muted"])
    ax.tick_params(colors=th["muted"], labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=th["ink"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=th["ink"], fontsize=9)
    if letter:
        ax.text(0.0, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom",
                color=th["ink"], fontsize=11)
    return ax


def _stack_offsets(*arrs, gap=1.25):
    """One vertical offset per trace, so `n` traces read as `n` rows rather than as a thicket.

    THE SPACING IS THE DATA'S OWN. Every row is shifted by `gap` times the largest peak-to-peak
    across all the arrays being stacked, so rows never collide and every row keeps the SAME
    scale -- which is the point. Normalising each row to its own range would make a trace that
    barely moves look as active as one that swings the full range, and on a fitted model that is
    exactly the comparison being made.
    """
    v = np.concatenate([np.asarray(a).ravel() for a in arrs if a is not None])
    span = float(np.nanmax(v) - np.nanmin(v)) or 1.0
    return span * gap


def _stack(ax, th, t, arrs_colors, off, labels=None, lw=None):
    """Draw stacked rows: `arrs_colors` is [(array [T, n], colour, linewidth), ...]."""
    n = max(a.shape[1] for a, _, _ in arrs_colors)
    for a, col, w in arrs_colors:
        for j in range(a.shape[1]):
            ax.plot(t[:a.shape[0]], a[:, j] + j * off, color=col, lw=w)
    ax.set_yticks([j * off for j in range(n)])
    ax.set_yticklabels(labels or [str(j) for j in range(n)], fontsize=7)
    return n


def _as_TN(x):
    """`[T]`, `[T, C]` or `[B, T, C]` -> `[T, n]`, trials and channels flattened into columns."""
    a = np.asarray(x, np.float64)
    if a.ndim == 1:
        return a[:, None]
    if a.ndim == 2:
        return a
    return a.transpose(1, 0, 2).reshape(a.shape[1], -1)


def rollout_movie(inp, gt, pred, *, out, dt=1.0, fps=12, unit="", kind="auto", title="",
                  n_show=4, max_frames=1200, theme=None, cmap="turbo", dpi=110,
                  xlabel="time (s)", quiet=False):
    """Write the six-panel rollout movie. Returns the path, or None if nothing could be drawn.

    inp, gt, pred   `[T]`, `[T, C]`, `[B, T, C]` for traces; `[T, H, W]` for a field. `inp` may
                    be None when the model has no exogenous input to show.
    dt              seconds per frame, so the x axis is time and not an index.
    unit            the read-out's unit, declared by the caller -- see `spec_trainer.units` for
                    why it is never guessed.
    kind            "trace", "field" or "auto" (by array rank).
    n_show          trials drawn. More than about six is a thicket, not a figure.
    max_frames      the movie is strided down to this, so an 8-second trial at 60 Hz and a
                    3,000-frame rollout both produce a watchable file. 600 draws EVERY frame of
                    the 480-frame trials this repository fits, which is the point -- at 300 they
                    were strided two-to-one and the rollout went past faster than it can be read.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.asarray(gt, np.float64)
    if kind == "auto":
        kind = "field" if g.ndim == 3 and g.shape[1] > 4 and g.shape[2] > 4 else "trace"
    th = THEMES[theme or ("dark" if kind == "field" else "light")]
    if kind == "field":
        return _field_movie(inp, gt, pred, out=out, dt=dt, fps=fps, title=title,
                            max_frames=max_frames, th=th, cmap=cmap, dpi=dpi, quiet=quiet)

    U = None if inp is None else _as_TN(inp)[:, :n_show]
    Y, P = _as_TN(gt)[:, :n_show], _as_TN(pred)[:, :n_show]
    T = min(Y.shape[0], P.shape[0])
    Y, P = Y[:T], P[:T]
    U = None if U is None else U[:min(T, U.shape[0])]
    if T < 2:
        return None
    R = P - Y
    cum = np.abs(R).mean(axis=1).cumsum() / np.arange(1, T + 1)   # before any stacking offset
    t = np.arange(T) * float(dt)
    step = max(1, int(np.ceil(T / max_frames)))
    ticks = list(range(1, T + 1, step))
    if ticks[-1] != T:
        ticks.append(T)
    u1 = f" ({unit})" if unit else ""

    fig = plt.figure(figsize=(14.0, 7.2), facecolor=th["bg"])
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)
    axes = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in (0, 1, 2)]
    names = [("a", "input"), ("b", f"ground truth{u1}"), ("c", f"inference{u1}"),
             ("d", f"overlaid{u1}"), ("e", f"residual{u1}"), ("f", f"cumulative |error|{u1}")]
    for ax, (ltr, lab) in zip(axes, names):
        _panel(ax, th, xlabel=xlabel if ltr in "def" else None, ylabel=lab, letter=ltr)
        ax.set_xlim(t[0], t[-1])

    def _lim(*arrs):
        v = np.concatenate([np.asarray(a).ravel() for a in arrs if a is not None])
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        pad = 0.08 * max(hi - lo, 1e-12)
        return lo - pad, hi + pad

    # STACKED HERE TOO, for the reason the still figure is: four trials on one axis cannot be
    # followed one at a time. The offset is applied to the DATA, so the sweep line and the
    # per-frame updates need no special case.
    off, offU = _stack_offsets(Y, P), (_stack_offsets(U) if U is not None else 0.0)
    nY = Y.shape[1]
    rows = np.arange(nY) * off
    if U is not None:
        U = U + np.arange(U.shape[1]) * offU
        axes[0].set_ylim(*_lim(U))
        axes[0].set_yticks(np.arange(U.shape[1]) * offU)
        axes[0].set_yticklabels([str(j) for j in range(U.shape[1])], fontsize=7)
    Y, P = Y + rows, P + rows
    R = R + rows
    for k in (1, 2, 3):
        axes[k].set_ylim(*_lim(Y, P))
        axes[k].set_yticks(rows); axes[k].set_yticklabels([str(j) for j in range(nY)], fontsize=7)
    axes[4].set_ylim(*_lim(R))
    axes[4].set_yticks(rows); axes[4].set_yticklabels([str(j) for j in range(nY)], fontsize=7)
    axes[5].set_ylim(0, float(np.nanmax(cum)) * 1.1 + 1e-12)

    lines = {k: [] for k in range(6)}
    for j in range(Y.shape[1]):
        if U is not None and j < U.shape[1]:
            lines[0].append(axes[0].plot([], [], color=th["stim"], lw=0.8, alpha=0.85)[0])
        lines[1].append(axes[1].plot([], [], color=th["gt"], lw=1.6)[0])
        lines[2].append(axes[2].plot([], [], color=th["pred"], lw=1.1)[0])
        lines[3].append(axes[3].plot([], [], color=th["gt"], lw=2.4, alpha=0.9)[0])
        lines[3].append(axes[3].plot([], [], color=th["pred"], lw=1.0)[0])
        lines[4].append(axes[4].plot([], [], color=th["resid"], lw=0.9)[0])
    lines[5].append(axes[5].plot([], [], color=th["ink"], lw=1.3)[0])
    axes[3].text(0.985, 0.04, "green: ground truth   black: inference"
                 if th is THEMES["light"] else "green: ground truth   white: inference",
                 transform=axes[3].transAxes, ha="right", va="bottom",
                 color=th["muted"], fontsize=8)
    sweeps = [ax.axvline(t[0], color=th["sweep"], lw=0.8, alpha=0.7) for ax in axes]
    head = fig.text(0.5, 0.975, title, ha="center", va="top", color=th["ink"], fontsize=10)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    writer = _writer(fps)
    with writer.saving(fig, out, dpi):
        for n in ticks:
            x = t[:n]
            for j in range(Y.shape[1]):
                if lines[0] and j < len(lines[0]):
                    lines[0][j].set_data(x[:U.shape[0]], U[:n, j][:len(x)])
                lines[1][j].set_data(x, Y[:n, j])
                lines[2][j].set_data(x, P[:n, j])
                lines[3][2 * j].set_data(x, Y[:n, j])
                lines[3][2 * j + 1].set_data(x, P[:n, j])
                lines[4][j].set_data(x, R[:n, j])
            lines[5][0].set_data(x, cum[:n])
            for s in sweeps:
                s.set_xdata([t[n - 1], t[n - 1]])
            head.set_text(f"{title}    t = {t[n-1]:.2f} s    mean |error| so far "
                          f"{cum[n-1]:.4f}{(' ' + unit) if unit else ''}")
            writer.grab_frame(facecolor=th["bg"])
    plt.close(fig)
    if not quiet:
        print(f"[rollout] {out}  {len(ticks)} frames of {T}, {Y.shape[1]} trial(s)", flush=True)
    return out


def figure(inp, gt, pred, *, out, dt=1.0, unit="", title="", n_show=4, theme=None,
           panels=None, dpi=130, xlabel="time (s)", quiet=False):
    """The same six panels as a STILL. One layout, drawn once, for every fitted model here.

    `panels` lets a caller replace a panel with its own content without forking the layout:
    `{"c": fn, "d": fn}` where `fn(ax, theme)` draws into the axes it is handed. That is how the
    t-battery puts its pole plane in `d` and its numbers in `c` while the eye rigs put a
    two-linearisation pole plane in the same slot -- one grid, one style, one set of colours, and
    the model-specific content stays with the trainer that knows what it means.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = THEMES[theme or "light"]
    panels = dict(panels or {})
    Y, P = _as_TN(gt)[:, :n_show], _as_TN(pred)[:, :n_show]
    T = min(Y.shape[0], P.shape[0])
    Y, P = Y[:T], P[:T]
    U = None if inp is None else _as_TN(inp)[:min(T, _as_TN(inp).shape[0]), :n_show]
    t = np.arange(T) * float(dt)
    u1 = f" ({unit})" if unit else ""

    fig = plt.figure(figsize=(14.5, 7.6), facecolor=th["bg"])
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)
    axes = {ltr: fig.add_subplot(gs[r, c])
            for ltr, (r, c) in zip("abcdef", [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])}
    lab = {"a": "input", "b": f"ground truth / inference{u1}", "c": "",
           "d": "", "e": f"residual{u1}", "f": "mse"}
    _ = lab
    for ltr, ax in axes.items():
        _panel(ax, th, xlabel=xlabel if ltr in "be" else None, ylabel=lab[ltr], letter=ltr)

    # STACKED, NOT SUPERPOSED. Four trials on one axis is a thicket in which no single trial can
    # be followed, and following one is the whole job: the convention here is connectome-gnn's
    # `tmp_training/traces` -- one row per trial, ground truth green and inference black on the
    # same row, the stimulus in red underneath, every row on the SAME scale.
    off = _stack_offsets(Y, P)
    if U is not None:
        _stack(axes["a"], th, t, [(U, th["stim"], 0.8)], _stack_offsets(U),
               labels=[f"{j}" for j in range(U.shape[1])])
    _stack(axes["b"], th, t, [(Y, th["gt"], 2.0), (P, th["pred"], 0.9)], off)
    _stack(axes["e"], th, t, [(P - Y, th["resid"], 0.8)], off)
    axes["b"].set_ylabel("trial", color=th["ink"], fontsize=9)
    axes["a"].set_ylabel("trial", color=th["ink"], fontsize=9)
    axes["e"].set_ylabel("trial", color=th["ink"], fontsize=9)
    axes["b"].text(0.985, 1.02, "green: ground truth   black: inference",
                   transform=axes["b"].transAxes, ha="right", va="bottom",
                   color=th["muted"], fontsize=8)
    # DEFAULTS FOR THE THREE THE CALLER DID NOT FILL, because an empty panel is worse than no
    # panel: it reads as a measurement that came out blank. `d` is the error accumulating over
    # the rollout, which says WHEN the model lost the target; `f` is the same error per trial,
    # which says whether one trial carries it. `c` stays for the caller's numbers and is turned
    # off if none are given.
    R = P - Y
    if "d" not in panels:
        cum = np.abs(R).mean(axis=1).cumsum() / np.arange(1, T + 1)
        axes["d"].plot(t, cum, color=th["ink"], lw=1.3)
        axes["d"].set_ylabel(f"cumulative |error|{u1}", color=th["ink"], fontsize=9)
        axes["d"].set_xlabel(xlabel, color=th["ink"], fontsize=9)
    if "f" not in panels:
        per = (R ** 2).mean(axis=0)
        axes["f"].bar(np.arange(len(per)), per, color=th["resid"], alpha=0.85)
        axes["f"].set_ylabel(f"mse per trial{(' ' + unit + '^2') if unit else ''}",
                             color=th["ink"], fontsize=9)
        axes["f"].set_xlabel("trial", color=th["ink"], fontsize=9)
        axes["f"].set_xticks(np.arange(len(per)))
    if "c" not in panels:
        axes["c"].axis("off")
    for ltr, fn in panels.items():
        if ltr in axes:
            fn(axes[ltr], th)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if title:
        fig.text(0.5, 0.985, title, ha="center", va="top", color=th["ink"], fontsize=10)
    fig.savefig(out, dpi=dpi, facecolor=th["bg"], bbox_inches="tight")
    plt.close(fig)
    if not quiet:
        print(f"[figure] {out}", flush=True)
    return out


def snapshot(out_dir, epoch, iteration, inp, gt, pred, *, kind="rollout", **kw):
    """A figure written WHILE training, into `<run>/tmp_training/<kind>/<kind>_<ep>_<it>.png`.

    THE NAME CARRIES THE EPOCH AND THE ITERATION, in that order, at the end -- the convention
    `connectome_gnn.plot._snapshot_iteration` parses, so the same tooling that assembles a
    training summary from one repository's snapshots reads this one's. A fit is hours long and
    the only way to see it go wrong early is to watch it; a figure written once at the end says
    what happened but never when.
    """
    d = os.path.join(out_dir, "tmp_training", kind)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"{kind}_{int(epoch)}_{int(iteration)}.png")
    return figure(inp, gt, pred, out=out, quiet=True, **kw)


def _field_movie(inp, gt, pred, *, out, dt, fps, title, max_frames, th, cmap, dpi, quiet):
    """The same six panels for `[T, H, W]` fields: images rather than traces, on black.

    The three field panels share ONE colour scale, taken from the ground truth, because three
    independently-scaled images are three different questions and cannot be compared by eye. The
    residual gets its own symmetric scale about zero, which is the only honest one for a
    difference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G, P = np.asarray(gt, np.float64), np.asarray(pred, np.float64)
    T = min(G.shape[0], P.shape[0])
    G, P = G[:T], P[:T]
    U = None if inp is None else np.asarray(inp, np.float64)[:T]
    R = P - G
    step = max(1, int(np.ceil(T / max_frames)))
    ticks = list(range(0, T, step))
    vmin, vmax = float(np.nanmin(G)), float(np.nanmax(G))
    rmax = float(np.nanmax(np.abs(R))) or 1e-12
    cum = np.abs(R).reshape(T, -1).mean(1).cumsum() / np.arange(1, T + 1)
    t = np.arange(T) * float(dt)

    fig = plt.figure(figsize=(13.5, 8.2), facecolor=th["bg"])
    gs = fig.add_gridspec(2, 3, hspace=0.20, wspace=0.14)
    axes = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in (0, 1, 2)]
    ims, labels = [], ["input", "ground truth", "inference", "difference", "|difference|",
                       "cumulative |error|"]
    for k, (ax, lab) in enumerate(zip(axes, labels)):
        ax.set_facecolor(th["bg"])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.02, 0.97, f"{'abcdef'[k]}  {lab}", transform=ax.transAxes, ha="left", va="top",
                color=th["ink"], fontsize=9)
    src = [U if U is not None else G, G, P, R, np.abs(R)]
    for k in range(5):
        kw = (dict(cmap="coolwarm", vmin=-rmax, vmax=rmax) if k == 3 else
              dict(cmap=cmap, vmin=0, vmax=rmax) if k == 4 else
              dict(cmap=cmap, vmin=vmin, vmax=vmax))
        ims.append(axes[k].imshow(src[k][0], animated=True, **kw))
    _panel(axes[5], th, xlabel="time (s)")
    axes[5].set_xlim(t[0], t[-1]); axes[5].set_ylim(0, float(np.nanmax(cum)) * 1.1 + 1e-12)
    curve, = axes[5].plot([], [], color=th["ink"], lw=1.3)
    head = fig.text(0.5, 0.985, title, ha="center", va="top", color=th["ink"], fontsize=10)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    writer = _writer(fps)
    with writer.saving(fig, out, dpi):
        for n in ticks:
            for k, arr in enumerate(src):
                ims[k].set_data(arr[n])
            curve.set_data(t[:n + 1], cum[:n + 1])
            head.set_text(f"{title}    t = {t[n]:.3f} s    mean |error| so far {cum[n]:.4g}")
            writer.grab_frame(facecolor=th["bg"])
    plt.close(fig)
    if not quiet:
        print(f"[rollout] {out}  {len(ticks)} frames of {T}, field {G.shape[1]}x{G.shape[2]}",
              flush=True)
    return out


def _writer(fps):
    """`imageio-ffmpeg` if it is installed, else whatever ffmpeg matplotlib can find.

    THE PACKAGE AND NOT THE BINARY. `imageio-ffmpeg` on PATH is not the same as the python package
    being importable, and without the package matplotlib's writer has silently produced a TIFF
    with an .mp4 name on this machine before.
    """
    import matplotlib
    from matplotlib import animation
    # SET ON `matplotlib.rcParams`, WHICH IS WHERE THE WRITER LOOKS. `animation.rcParams` is the
    # same object imported into that module, but the writer resolves the binary from the global
    # rcParams at construction -- and there is no ffmpeg on PATH in this container, so getting
    # this wrong is a FileNotFoundError and not a silent fallback.
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return animation.FFMpegWriter(fps=int(fps), bitrate=2400,
                                  metadata=dict(artist="plexus.tasks.plot_trainer"))


# --------------------------------------------------------------------------- #
#  from a finished run
# --------------------------------------------------------------------------- #
def _concat_trials(U, cond, reps, n_show):
    """`reps` consecutive trials OF THE SAME CONDITION CELL, joined end to end per row.

    A longer rollout is the only way to see a slow failure: an integrator that holds for eight
    seconds and one that drifts after twelve are the same picture at T = 480. Trials are joined
    within a condition cell because the cell IS the teacher -- splicing two different laws into
    one stimulus would make the ground truth below a fiction.
    """
    c = np.asarray(cond)
    rows = []
    for j in range(min(n_show, U.shape[0])):
        same = [i for i in range(U.shape[0]) if c[i] == c[j]]
        # A DIFFERENT WINDOW PER ROW. `same[:reps]` gave every row the SAME trials, so four rows
        # of one task drew one trial four times -- visible only because the per-trial error bars
        # came out exactly equal, which no four held-out trials ever are.
        k = (same.index(j) * reps) % max(len(same), 1)
        pick = [same[(k + r) % len(same)] for r in range(reps)]
        rows.append(np.concatenate([U[i] for i in pick], axis=0))
    return np.stack(rows)


def _teacher_on(task, u):
    """The ground truth for an arbitrary stimulus, from the law `teacher.pt` recorded.

    NOT THE RECORDED TARGETS CONCATENATED. The teacher is a dynamical system: its state carries
    across a join, so `y1 | y2` is not the response to `u1 | u2` -- the second half would start
    from rest when the real one does not. Re-running the law is the only correct answer.
    """
    import torch
    from plexus.tasks import get_teacher
    from plexus.tasks.generate import task_dir
    t = torch.load(os.path.join(task_dir(task), "teacher.pt"), weights_only=False)
    law, dt = t["law"], float(t["dt"])
    params = dict(t["per_cell"][0].get("params") or {})
    return np.asarray(get_teacher(law)(np.asarray(u, np.float64), dt, **params), np.float64)


def from_run(name, *, root=None, device="cpu", n_show=4, fps=48, out=None, quiet=False,
             what=("movie",), reps=1):
    """Roll a finished run out on its held-out split and write the movie beside its results.

    The two trainers are told apart by what their run spec holds -- `spec:` names a forward SPEC
    and belongs to `spec_trainer`, `circuit:` names a standalone circuit and belongs to `trainer`
    -- rather than by a flag, so a run describes itself and this cannot drift.
    """
    import torch
    import yaml
    from plexus.tasks.trainer import load_split, log_dir, n_condition_cells, with_context
    from plexus.tasks.generate import task_dir

    out_dir = log_dir(name, root)
    cfg_p = os.path.join(out_dir, "config.yaml")
    if not os.path.isfile(cfg_p):
        raise FileNotFoundError(f"{cfg_p} does not exist -- {name!r} is not a finished run.")
    run = yaml.safe_load(open(cfg_p))
    run["_path"] = cfg_p
    # THE CONFIG CHOOSES THE DATA, as `test_dataset` does in connectome-gnn: a model is often
    # worth watching on a corpus it was NOT fitted on -- a longer trial, a different band, the
    # held-out specimen -- and hard-coding the training task here would make that impossible to
    # ask for. Empty means the training task, which is the common case and stays silent.
    tr = run.get("training") or {}
    task = tr.get("test_task") or run["task"]
    split = tr.get("test_split") or ("test" if os.path.isdir(os.path.join(task_dir(task), "test"))
                                     else "train")
    n_show = int(tr.get("rollout_trials", n_show))
    prov = json.load(open(os.path.join(task_dir(task), "provenance.json")))
    dt = float(prov["dt"])
    U, Y, cond = load_split(task, split, device)
    # `rollout_frames` IS A LENGTH, not a trial count, because that is what a reader asks for --
    # "does it still hold at twenty seconds". Trials of the same condition are joined to reach it.
    rf = int(tr.get("rollout_frames", 0))
    if rf > U.shape[1]:
        reps = max(reps, int(np.ceil(rf / U.shape[1])))

    run = dict(run, task=task)                          # every later read uses the CHOSEN task
    if "spec" in run:                                   # fitted through the engine
        from plexus.tasks.spec_trainer import _restore, rollout, units
        sim, ck, _ = _restore(run, device)
        U = with_context(U, cond, int(ck.get("n_cond", 1)))
        io = run["io"]
        ch = int(io.get("read_channel", 0))
        with torch.no_grad():
            _, P = rollout(sim, U[:n_show], io["drive_set"], io["drive_block"],
                           io["read_set"], io["read_block"], device, grad=False)
        pred = P[..., ch:ch + 1].cpu().numpy()
        unit = (run.get("io") or {}).get("unit") or ""
    else:                                               # the standalone circuit
        from plexus.tasks.trainer import CircuitRNN
        ck = torch.load(os.path.join(out_dir, "models", "best.pt"),
                        weights_only=False, map_location=device)
        U = with_context(U, cond, int(ck.get("n_cond", 1)))
        m = CircuitRNN(ck["n_in"], ck["n_out"], hidden=int(ck["circuit"].get("hidden", 64)),
                       tau0=float(ck["circuit"].get("tau0", 0.5)), dt=ck["dt"]).to(device)
        m.load_state_dict(ck["state"]); m.eval()
        with torch.no_grad():
            pred = m(U[:n_show]).cpu().numpy()
        unit = ""

    # ONE SET OF OUTPUTS PER CONDITION CELL, because a pooled number hides a dead function among
    # good ones: "it holds the integrator and loses the delay" is the whole content of a
    # multi-function fit and it is invisible in one mse. The cell's NAME comes from the task spec
    # (`teachers: [{name: delay, ...}]`), so the files are readable without a lookup.
    names, cells = _cell_names(task), np.asarray(cond)
    if len(names) > 1:
        made = []
        for c, cname in enumerate(names):
            # ENOUGH TRIALS FOR `n_show` ROWS OF `reps`, not n_show total. Handing `_one` only
            # n_show trials made it wrap: row 0 and row 2 drew the same pair, which showed up as
            # per-trial error bars in exact pairs -- the same defect `_concat_trials` had.
            idx = np.where(cells == c)[0][:n_show * max(reps, 1)]
            if not len(idx):
                continue
            made += _one(run, name, task, split, out_dir, dt, unit, fps, quiet, what, reps,
                         U[idx], Y[idx], cond[idx] if hasattr(cond, "__getitem__") else cond,
                         device, n_show, suffix=f"_{cname}", title_extra=f"  [{cname}]")
        return made
    u_np, y_np = U[:n_show, :, :1].cpu().numpy(), Y[:n_show].cpu().numpy()
    if reps > 1:
        u_np = _concat_trials(u_np, cond, reps, n_show)
        y_np = _teacher_on(task, u_np)
        pred = _predict_long(run, name, out_dir, u_np, cond, device, n_show)
    res = os.path.join(out_dir, "results")
    made = []
    if "movie" in what:
        made.append(rollout_movie(u_np, y_np, pred, dt=dt, fps=fps, unit=unit, n_show=n_show,
                                  title=f"{name}  ({task}/{split})", quiet=quiet,
                                  out=out or os.path.join(res, f"{name}_{split}_rollout.mp4")))
    if "figure" in what:
        rp = os.path.join(res, f"{name}_{split}.json")
        txt = json.load(open(rp)) if os.path.isfile(rp) else {}
        lines = [f"{name}", f"task    {run['task']}", f"split   {split}", ""]
        for k in ("mse", "normalised_mse", "normalised_mse_settled", "mae", "rmse"):
            if txt.get(k) is not None:
                lines.append(f"{k:24s} {txt[k]:.6g}")
        def _numbers(ax, th):
            ax.axis("off")
            ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                    color=th["ink"], fontsize=8, family="monospace")
        made.append(figure(u_np, y_np, pred, dt=dt, unit=unit, n_show=n_show,
                           title=f"{name}  ({split})", quiet=quiet, panels={"c": _numbers},
                           out=os.path.join(res, f"{name}_{split}_rollout.png")))
    return made[0] if len(made) == 1 else made


def _cell_names(task):
    """The per-cell names the task spec declared, from `teacher.pt`."""
    import torch
    from plexus.tasks.generate import task_dir
    t = torch.load(os.path.join(task_dir(task), "teacher.pt"), weights_only=False)
    nm = t.get("names")
    return list(nm) if nm else [str(c.get("name") or i)
                                for i, c in enumerate(t.get("per_cell") or [{}])]


def _one(run, name, task, split, out_dir, dt, unit, fps, quiet, what, reps,
         U, Y, cond, device, n_show, suffix="", title_extra=""):
    """The movie and figure for ONE condition cell's trials."""
    import torch
    res = os.path.join(out_dir, "results")
    need = n_show * max(reps, 1)
    u_np, y_np = U[:need, :, :1].cpu().numpy(), Y[:need].cpu().numpy()
    if reps > 1:
        # DISJOINT WINDOWS: row j is trials [j*reps, (j+1)*reps). Every row is different data,
        # which is the only way the per-trial panel means anything.
        m = min(n_show, len(u_np) // reps)
        u_np = np.stack([np.concatenate([u_np[j * reps + r] for r in range(reps)], axis=0)
                         for j in range(m)])
        y_np = _teacher_on_cell(task, u_np, cond)
        n_show = u_np.shape[0]
    else:
        u_np, y_np = u_np[:n_show], y_np[:n_show]
    pred = _predict_long(run, name, out_dir, u_np, np.asarray(cond), device, n_show)
    made = []
    ttl = f"{name}  ({task}/{split}){title_extra}"
    if "movie" in what:
        made.append(rollout_movie(u_np, y_np, pred, dt=dt, fps=fps, unit=unit, n_show=n_show,
                                  title=ttl, quiet=quiet,
                                  out=os.path.join(res, f"{name}_{split}{suffix}_rollout.mp4")))
    if "figure" in what:
        made.append(figure(u_np, y_np, pred, dt=dt, unit=unit, n_show=n_show, title=ttl,
                           quiet=quiet,
                           out=os.path.join(res, f"{name}_{split}{suffix}_rollout.png")))
    return made


def _teacher_on_cell(task, u, cond):
    """The ground truth for a longer stimulus, using THAT CELL's law rather than cell 0's."""
    import torch
    from plexus.tasks import get_teacher
    from plexus.tasks.generate import task_dir
    t = torch.load(os.path.join(task_dir(task), "teacher.pt"), weights_only=False)
    c = int(np.asarray(cond).ravel()[0])
    rec = t["per_cell"][min(c, len(t["per_cell"]) - 1)]
    params = dict(rec.get("params") or {})
    # THE CELL'S OWN LAW, NOT THE HEADER'S. `generate` pops `law` out of `params` before writing,
    # so falling back to `teacher["law"]` silently applied the FIRST cell's law to every cell: a
    # six-function corpus plotted its delay, its low-pass and its high-pass against an INTEGRATOR,
    # and the model -- which was right -- looked like a wild oscillation next to a smooth curve.
    # The full cell dict keeps the law, so it is recoverable even in corpora written before the
    # writer was fixed.
    law_name = rec.get("law") or (rec.get("cell") or {}).get("law") or t["law"]
    law = get_teacher(law_name)
    params.pop("law", None); params.pop("name", None)
    return np.asarray(law(np.asarray(u, np.float64), float(t["dt"]), **params), np.float64)


def _predict_long(run, name, out_dir, u_np, cond, device, n_show):
    """Roll the fitted model out over a stimulus longer than the trials it was fitted on.

    ALWAYS RETURNS `[B, T, 1]`, even for B = 1. `spec_trainer.rollout` drops the leading axis when
    the batch is one -- it takes `[T, C]` for a single trial and returns `[T, w]` -- and a caller
    asking for ONE trial (the montage does) then indexes a 2-D array as though it were 3-D.
    """
    import torch
    from plexus.tasks.trainer import with_context, n_condition_cells
    U = torch.as_tensor(np.asarray(u_np, np.float32), device=device)
    U = with_context(U, np.asarray(cond)[:U.shape[0]], n_condition_cells(run["task"]))
    if "spec" in run:
        from plexus.tasks.spec_trainer import _restore, rollout
        sim, ck, _ = _restore(run, device)
        sim.n_frames = int(U.shape[1])          # the spec's own length is the TRIAL's, not this
        io = run["io"]
        with torch.no_grad():
            _, P = rollout(sim, U, io["drive_set"], io["drive_block"],
                           io["read_set"], io["read_block"], device, grad=False)
        ch = int(io.get("read_channel", 0))
        out = P[..., ch:ch + 1].cpu().numpy()
        return out if out.ndim == 3 else out[None]
    from plexus.tasks.trainer import CircuitRNN
    ck = torch.load(os.path.join(out_dir, "models", "best.pt"),
                    weights_only=False, map_location=device)
    m = CircuitRNN(ck["n_in"], ck["n_out"], hidden=int(ck["circuit"].get("hidden", 64)),
                   tau0=float(ck["circuit"].get("tau0", 0.5)), dt=ck["dt"]).to(device)
    m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        out = m(U).cpu().numpy()
    return out if out.ndim == 3 else out[None]


def montage(name, *, root=None, device="cpu", out=None, cols=3, n_show=1, seconds=None,
            theme=None, quiet=False):
    """One panel per function: ground truth against the model, all of them on one page.

    THE FIGURE THE LADDER IS FOR. Six separate rollout plots answer "how does it do on the
    high-pass"; this one answers "does ONE circuit hold all six", which is the question -- and it
    answers it at a glance, because every panel is the same weights under a different context
    one-hot. Each panel carries its own held-out error, so a panel that looks right and scores
    badly cannot hide behind the picture.

    EVERY PANEL ON ITS OWN Y SCALE, deliberately, where the stacked trace plots share one. These
    are DIFFERENT LAWS: the differentiator's output has four times the rms of the low-pass's, and
    a shared scale would flatten five panels to show one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    from plexus.tasks.trainer import load_split, log_dir
    from plexus.tasks.generate import task_dir

    out_dir = log_dir(name, root)
    run = yaml.safe_load(open(os.path.join(out_dir, "config.yaml")))
    run["_path"] = os.path.join(out_dir, "config.yaml")
    tr = run.get("training") or {}
    task = tr.get("test_task") or run["task"]
    split = "test" if os.path.isdir(os.path.join(task_dir(task), "test")) else "train"
    dt = float(json.load(open(os.path.join(task_dir(task), "provenance.json")))["dt"])
    U, Y, cond = load_split(task, split, device)
    names = _cell_names(task)
    res_p = os.path.join(out_dir, "results", f"{name}_{split}.json")
    per = (json.load(open(res_p)).get("normalised_per_cell") or {}) if os.path.exists(res_p) else {}

    run = dict(run, task=task)
    th = THEMES[theme or "light"]
    rows = int(np.ceil(len(names) / cols))
    fig = plt.figure(figsize=(4.8 * cols, 3.0 * rows), facecolor=th["bg"])
    gs = fig.add_gridspec(rows, cols, hspace=0.50, wspace=0.24)
    T = min(int(round(seconds / dt)) if seconds else U.shape[1], U.shape[1])
    for c, cname in enumerate(names):
        ax = _panel(fig.add_subplot(gs[c // cols, c % cols]), th,
                    xlabel="time (s)" if c // cols == rows - 1 else None,
                    ylabel="output" if c % cols == 0 else None, letter="abcdef"[c % 6])
        idx = np.where(np.asarray(cond) == c)[0][:n_show]
        if not len(idx):
            ax.axis("off")
            continue
        pred = _predict_long(run, name, out_dir, U[idx, :, :1].cpu().numpy(),
                             np.asarray(cond)[idx], device, len(idx))
        t = np.arange(T) * dt
        for j in range(len(idx)):
            ax.plot(t, Y[idx[j], :T, 0].cpu(), color=th["gt"], lw=2.4, alpha=0.9)
            ax.plot(t, pred[j, :T, 0], color=th["pred"], lw=0.9)
        e = per.get(str(c))
        ax.text(0.0, 1.16, cname + (f"    {e:.4f} of variance" if e is not None else ""),
                transform=ax.transAxes, ha="left", va="bottom", color=th["ink"], fontsize=9)
    fig.text(0.5, 1.0, f"{name}   ({task}/{split})   green: ground truth   black: model",
             ha="center", va="bottom", color=th["muted"], fontsize=9)
    out = out or os.path.join(out_dir, "results", f"{name}_{split}_montage.png")
    fig.savefig(out, dpi=140, facecolor=th["bg"], bbox_inches="tight")
    plt.close(fig)
    if not quiet:
        print(f"[montage] {out}  {len(names)} function(s)", flush=True)
    return out


# ============================================================================== the circuit at work
def _task_circuit_panel_class():
    """`TaskCircuitPanel`, built on first use so importing this module does not import the panel."""
    from plexus.neural_panel import BG, NeuralPanel

    class TaskCircuitPanel(NeuralPanel):
        """THE CIRCUIT AT WORK ON ITS TASK (Cedric, 2026-10-08: "add this plotter to the codebase so that we can
        regenerate the circuit + kinograph + eye movies in Plexus"). `NeuralPanel` in the middle -- the message on
        the connectivity matrix, the input / rate / output columns -- and the kinograph across the bottom, with
        what the circuit is FOR on either side:
          left    the task's trace swept in time: the target (green), the model's observable (white), the command
                  (red) when the observed block has a `<block>_target` beside it, the drive (grey, its own scale)
          right   when the observable is an angle (`unit: deg`), the eye from above turning with it, the target
                  dashed; under it, each muscle's drive as a bar (a level with a `drive` block and no edges)
        The INPUT COLUMN is the afferent current each neuron receives, sum_e w_e x_pre(e), over the edge-sets that
        run from the task's drive set onto the neurons -- a circuit driven through afferent synapses never writes
        the drive field or the `omega` block `NeuralPanel` reads otherwise. The circuit is captured tick by tick
        from the engine and the traces are handed in afterwards (`set_traces`); every frame is drawn after the
        rollout, so all of them share the final colour limits."""

        def __init__(self, *a, drive_set=None, drive_block=None, **k):
            super().__init__(*a, **k)
            self.drive_set, self.drive_block, self.tr = drive_set, drive_block, None

        def set_traces(self, t, drive, target, obs, cmd=None, muscles=None, muscle_names=(), unit="", label=""):
            f = lambda x: None if x is None or not len(x) else np.asarray(x, np.float64)   # noqa: E731
            self.tr = dict(t=f(t), drive=f(drive), target=f(target), obs=f(obs), cmd=f(cmd), mus=f(muscles),
                           names=list(muscle_names), unit=str(unit or ""), label=str(label or ""))

        def _read(self, H):
            r, om = super()._read(H)
            if self.drive_set is None or self.drive_set not in H.levels:
                return r, om
            src_l = H.level(self.drive_set)
            src = src_l.get(self.drive_block).detach().float().cpu().numpy().reshape(int(src_l.n), -1)[:, 0]
            cur, hit = np.zeros(self.N, np.float64), False
            for lv in H.levels.values():
                if (getattr(lv, "pre_name", None) == self.drive_set and getattr(lv, "post_name", None) == self.nset
                        and "w" in getattr(lv, "state_schema", {})):
                    pre, post = lv.pre.detach().cpu().numpy(), lv.post.detach().cpu().numpy()
                    w = lv.get("w").detach().float().cpu().numpy().reshape(-1)
                    ok = lv.occ.detach().cpu().numpy() > 0
                    np.add.at(cur, post[ok], w[ok] * src[pre[ok]])
                    hit = True
            return r, (cur.astype(np.float32) if hit else om)

        def _figure(self):
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            fig = plt.figure(figsize=(19.2, 9.6), facecolor=BG, dpi=100)
            gs = GridSpec(2, 3, width_ratios=[0.85, 1.75, 0.85], height_ratios=[1.0, 0.52], hspace=0.16,
                          wspace=0.10, left=0.07, right=0.985, top=0.93, bottom=0.07)
            self.ax_tr, self.ax_eye = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])
            return fig, fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

        def _build_extra(self, fig):
            import matplotlib.pyplot as plt
            tr = self.tr
            if tr is None:
                return
            self.txt.set_position((0.0, 1.045))               # the circuit's header above its column labels
            a, t = self.ax_tr, tr["t"]
            a.set_facecolor(BG)
            for k_, sp in a.spines.items():
                sp.set_visible(k_ in ("left", "bottom")); sp.set_color("#888")
            a.tick_params(colors="#bbb", labelsize=9)
            ys = [x for x in (tr["target"], tr["obs"], tr["cmd"]) if x is not None]
            lo, hi = np.percentile(np.concatenate(ys), [0.5, 99.5])
            pad = 0.12 * max(hi - lo, 1e-9)
            if tr["drive"] is not None:                       # the drive, dotted, on its own scale
                ad = a.twinx()
                ad.plot(t, tr["drive"], color="#9aa0a6", lw=0.6, ls=":", alpha=0.7)
                ad.set_yticks([]); [sp.set_visible(False) for sp in ad.spines.values()]
                a.plot([], [], color="#9aa0a6", lw=0.8, ls=":", label="drive (own scale)")
            self.ln = {"target": a.plot([], [], color="#5fd08a", lw=2.2, label="target")[0],
                       "obs": a.plot([], [], color="#ffffff", lw=1.3, label="model")[0]}
            if tr["cmd"] is not None:
                self.ln["cmd"] = a.plot([], [], color="#ff4d4d", lw=0.9, alpha=0.85, label="command")[0]
            self.cur = a.axvline(t[0], color="#ff7f0e", lw=1.0)
            a.set_xlim(t[0], t[-1]); a.set_ylim(lo - pad, hi + pad)
            a.set_xlabel("time (s)", color="#ddd", fontsize=11)
            a.set_ylabel(f"{tr['label']}" + (f" ({tr['unit']})" if tr["unit"] else ""), color="#ddd", fontsize=11)
            leg = a.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
            for tx in leg.get_texts():
                tx.set_color("#ddd")
            self.txt_tr = a.text(0.0, 1.02, "", transform=a.transAxes, color="#ddd", fontsize=10, va="bottom")
            e = self.ax_eye
            e.set_facecolor(BG); e.axis("off"); e.set_aspect("equal")
            e.set_xlim(-1.7, 1.7); e.set_ylim(-2.75, 1.85)          # the lower third holds the muscle bars
            self.eye_on = tr["unit"] in ("deg", "rad") and tr["obs"] is not None
            if self.eye_on:
                e.add_patch(plt.Circle((0, 0), 1.0, fc="#1b1b1b", ec="#bbbbbb", lw=1.5))
                self.eye_tgt = e.plot([], [], color="#5fd08a", lw=1.4, ls="--")[0]
                self.eye_ln = e.plot([], [], color="#ffffff", lw=2.2)[0]
                self.pupil = plt.Circle((0, 0.85), 0.17, fc="#4a9bff", ec="white", lw=0.8)
                e.add_patch(self.pupil)
                e.text(0.0, 1.75, "the eye, from above", color="#ddd", fontsize=11, ha="center", va="top")
                e.text(1.15, 1.15, "+", color="#bbb", fontsize=12, ha="center", va="center")
                e.text(-1.15, 1.15, "-", color="#bbb", fontsize=12, ha="center", va="center")
                self.txt_eye = e.text(0.0, -1.15, "", color="#ddd", fontsize=10, ha="center", va="top")
            self.bars = None
            if tr["mus"] is not None:                         # the muscles' drives, under the eye
                live = np.flatnonzero(np.nanmax(np.abs(tr["mus"]), 0) > 1e-6)   # a muscle never driven: no bar
                names = tr["names"] if len(tr["names"]) == tr["mus"].shape[1] else [str(i) for i in range(tr["mus"].shape[1])]
                self.mus_live = live if live.size else np.arange(tr["mus"].shape[1])
                names = [names[i] for i in self.mus_live]
                M = len(self.mus_live)
                am = e.inset_axes([0.18, 0.03, 0.72, min(0.06 * M + 0.04, 0.26)])
                am.set_facecolor(BG)
                # THE DRIVES' OWN RANGE, not 0 to the maximum: a motor pair co-contracts (both drives high, the gaze
                # set by their difference), so on a 0..max axis the push-pull that moves the eye is a few pixels
                mv = tr["mus"][:, self.mus_live]
                lo_, hi_ = float(np.nanmin(mv)), float(np.nanmax(mv))
                pad_ = 0.08 * max(hi_ - lo_, 1e-9)
                self.mus_lo = max(lo_ - pad_, 0.0)
                self.bars = am.barh(np.arange(M), np.zeros(M), left=self.mus_lo, color="#e0a040", height=0.65)
                am.set_yticks(np.arange(M)); am.set_yticklabels(names, color="#ddd", fontsize=9)
                am.set_xlim(self.mus_lo, hi_ + pad_); am.set_ylim(M - 0.5, -0.5)
                am.tick_params(axis="x", colors="#bbb", labelsize=8)
                for sp in am.spines.values():
                    sp.set_color("#666")
                am.set_title("muscle drive", color="#ddd", fontsize=9, loc="left", pad=3)

        def _draw_extra(self, idx: int):
            tr = self.tr
            if tr is None:
                return
            t = tr["t"]
            k = int(np.clip(self.hist[idx][0], 0, len(t) - 1))          # tick j = engine frame j = trace sample j
            for key, ln in self.ln.items():
                ln.set_data(t[:k + 1], tr[key][:k + 1])
            self.cur.set_xdata([t[k]] * 2)
            u_ = tr["unit"]
            self.txt_tr.set_text(f"t = {t[k]:5.2f} s   target {tr['target'][k]:+.2f}   model {tr['obs'][k]:+.2f}"
                                 + (f"   command {tr['cmd'][k]:+.2f}" if tr["cmd"] is not None else "") + f" {u_}")
            if self.eye_on:
                to = (np.deg2rad if u_ == "deg" else float)
                th, tg = to(tr["obs"][k]), to(tr["target"][k])
                self.eye_ln.set_data([0, 1.6 * np.sin(th)], [0, 1.6 * np.cos(th)])
                self.eye_tgt.set_data([0, 1.6 * np.sin(tg)], [0, 1.6 * np.cos(tg)])
                self.pupil.center = (0.85 * np.sin(th), 0.85 * np.cos(th))
                self.txt_eye.set_text(f"gaze {tr['obs'][k]:+.1f} {u_}   target {tr['target'][k]:+.1f} {u_}")
            if self.bars is not None:
                for b, v in zip(self.bars, tr["mus"][k][self.mus_live]):
                    b.set_width(max(float(v) - self.mus_lo, 0.0))

    return TaskCircuitPanel


def circuit_movie(spec, device="cpu", root=None, *, trials=None, fps=None, stride=None, out=None, quiet=False):
    """A trained corpus run's circuit at work on held-out trials -> results/movie_circuit.mp4 (+ .png).

    `trials` consecutive held-out trials of the first trial's condition cell are joined into one drive (the target
    re-run by that cell's teacher law across the joins, `_teacher_on_cell`); the model is rolled out once with the
    panel capturing every tick (`trainer.rollout(..., watch=)`); every `stride`-th tick is drawn, at `fps`. The
    defaults come from the training spec's `training.circuit_movie: {trials, fps, stride}` (2, 30, 2: a 60-Hz
    model then plays in real time). Read by `trainer.analyse` (`Plexus_Main.py -o plot`) when the spec declares it,
    and by `python -m plexus.tasks.plot_trainer --circuit <training spec>`."""
    import torch
    import imageio.v2 as iio
    from plexus import engine
    from plexus import trainer as T
    from plexus.tasks.generate import task_dir
    cfg = dict((spec.get("training") or {}).get("circuit_movie") or {})
    trials = max(1, int(trials or cfg.get("trials", 2)))
    fps = float(fps or cfg.get("fps", 30))
    stride = max(1, int(stride or cfg.get("stride", 2)))
    engine.quiet(True)
    sim, learn, ck, out_d = T._restore(spec, device, root)
    task, corpus = spec["task"], T._corpus(spec)
    split = "test" if os.path.isdir(os.path.join(task_dir(corpus), "test")) else "train"
    U, Y, cond = T._data(spec, split, int(ck.get("n_cond", 1)), device)
    c = np.asarray(cond.cpu() if hasattr(cond, "cpu") else cond).reshape(-1).astype(int)
    pick = np.flatnonzero(c == c[0])[:trials]
    u = torch.cat([U[i] for i in pick], 0)                                  # [T, C]
    ch = int(task["observe"].get("channel", 0))
    dt = float(sim.dt)
    if len(pick) == 1:
        y = Y[pick[0], :, ch].cpu().numpy()
    else:
        y = np.asarray(_teacher_on_cell(corpus, u[None, :, :1].cpu().numpy(), c[pick[:1]])).reshape(len(u), -1)[:, 0]
    path = out or os.path.join(out_d, "results", "movie_circuit.mp4")
    # THE INPUT AND OUTPUT COLUMNS FROM THE TYPES' DECLARED ROLES (`role: afferent | output` on a neuron type), so the
    # output column shows the motor pools rather than every population
    pcfg = {"kino_frames": min(int(u.shape[0]), 600)}
    for st in (getattr(sim, "sets", None) or {}).values():
        roles = {k_: (v_ or {}).get("role") for k_, v_ in ((st or {}).get("types") or {}).items()}
        if any(roles.values()):
            pcfg["input"] = [k_ for k_, r_ in roles.items() if r_ == "afferent"] or None
            pcfg["output"] = [k_ for k_, r_ in roles.items() if r_ == "output"] or None
            break
    Panel = _task_circuit_panel_class()
    panel = Panel(out=path, n_frames=int(u.shape[0]), sim=sim, style={"panel": pcfg},
                  dt=dt, time_s=1.0, name=spec["name"], fps=fps, max_frames=int(u.shape[0]) + 1, stills=0,
                  drive_set=task["drive"]["set"], drive_block=task["drive"]["block"])
    obs_set, obs_block = task["observe"]["set"], task["observe"]["block"]
    rec, tick = {"cmd": [], "mus": [], "mus_names": []}, [0]

    def watch(H):                                       # once per engine frame, frame 0 (the seeded state) first
        panel.capture(H, tick[0])
        tick[0] += 1
        lv = H.level(obs_set)
        if f"{obs_block}_target" in lv.state_schema:
            rec["cmd"].append(float(lv.get(f"{obs_block}_target").detach().float().cpu().numpy().reshape(int(lv.n), -1)[0, ch]))
        m = next((n for n, l_ in H.levels.items() if "drive" in getattr(l_, "state_schema", {})
                  and getattr(l_, "pre_name", None) is None and int(l_.n) <= 16), None)   # a node set (no edges)
        if m is not None:
            rec["mus"].append(H.level(m).get("drive").detach().float().cpu().numpy().reshape(-1))
            if not rec["mus_names"]:
                rec["mus_names"] = list(getattr(H.level(m), "type_names", []) or [])
    sim.n_frames = int(u.shape[0])                      # the joined trials' length, as training sets T_full
    with torch.no_grad():
        _, Yp = T.rollout(sim, learn, u, task, device, grad=False, watch=watch)
    obs = Yp[:, ch].cpu().numpy()                       # [frames + 1, w]: aligned with the target at frame 0
    n_ = min(len(obs), len(y), len(panel.hist))
    panel.set_traces(np.arange(n_) * dt, u[:n_, 0].cpu().numpy(), y[:n_], obs[:n_],
                     cmd=(rec["cmd"][:n_] or None), muscles=(np.stack(rec["mus"][:n_]) if rec["mus"] else None),
                     muscle_names=rec["mus_names"], unit=task["observe"].get("unit") or "",
                     label=f"{obs_set}.{obs_block}")
    idxs = list(range(stride - 1, n_, stride)) or [n_ - 1]
    w = iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    for i in idxs:
        w.append_data(panel.frame_at(i))
    w.close()
    iio.imwrite(path[:-4] + ".png", panel.frame_at(idxs[len(idxs) // 2]))
    panel.close()
    if not quiet:
        print(f"[circuit-movie] {path}: {len(idxs)} frames at {fps:g} fps, {len(pick)} held-out trial(s) of "
              f"{split}, {panel.N} neurons, {panel.pre.size:,} synapses", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="run names under log/task/")
    ap.add_argument("--all", action="store_true", help="every run that has a checkpoint")
    ap.add_argument("--root", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-show", type=int, default=4)
    ap.add_argument("--fps", type=int, default=48)
    ap.add_argument("--reps", type=int, default=2,
                    help="trials of the same condition joined end to end, for a longer rollout")
    ap.add_argument("--what", nargs="+", default=["movie"],
                    choices=["movie", "figure", "montage"])
    ap.add_argument("--montage-seconds", type=float, default=None)
    ap.add_argument("--circuit", nargs="+", default=None, metavar="SPEC",
                    help="training specs (config/training/...): the circuit at work on held-out trials, "
                         "results/movie_circuit.mp4 (circuit_movie)")
    ap.add_argument("--trials", type=int, default=None, help="--circuit: held-out trials joined end to end")
    a = ap.parse_args()
    if a.circuit:
        from plexus import trainer as T
        for n_ in a.circuit:
            circuit_movie(T.load(n_), device=a.device, root=a.root, trials=a.trials)
        return
    from plexus.tasks.trainer import log_dir
    names = list(a.names)
    if a.all or not names:
        base = os.path.dirname(log_dir("_", a.root))
        # TWO LEVELS UP, not one: the checkpoint is `<run>/models/best.pt`, so one `dirname` names
        # the `models` directory and every run comes out called "models".
        names = sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                       for p in glob.glob(os.path.join(base, "*", "models", "best.pt")))
    ok, bad = 0, []
    for n in names:
        try:
            w = tuple(x for x in a.what if x != "montage")
            if w:
                from_run(n, root=a.root, device=a.device, n_show=a.n_show, fps=a.fps,
                         what=w, reps=a.reps)
            if "montage" in a.what:
                montage(n, root=a.root, device=a.device, seconds=a.montage_seconds)
            ok += 1
        except Exception as e:                      # one bad run must not stop the sweep
            bad.append((n, f"{type(e).__name__}: {e}"))
            print(f"[rollout] {n}: {type(e).__name__}: {e}", flush=True)
    print(f"[rollout] {ok} written, {len(bad)} failed")
    for n, e in bad:
        print(f"           {n}: {e}")


if __name__ == "__main__":
    main()
