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
        regenerate the circuit + kinograph + eye movies in Plexus", then "proper input and output sets of neurons,
        like for the two eyes"). Left to right, the order the signal flows:
          the task     its trace swept in time: each observed element's target (green), model (white) and command
                       (red, when the observed block has a `<block>_target`), the drive (grey, its own scale);
                       with two eyes the left one solid, the right one dashed
          the circuit  `NeuralPanel`'s message on the connectivity matrix and rate column, with ITS OWN INPUT AND
                       OUTPUT: the drive set's values in a box, feeding the afferent-role cells' column (their
                       afferent current sum_e w_e x_pre(e), aligned with their rows); the rate column labelled by
                       role; the output-role cells feeding a grid of the muscle drives, one row per muscle, one
                       column per eye (rows of the matrix split by side when the spec's types are)
          the plant    when the model has `muscle_pose_map` and `organ_mechanics`: each eye's 27 static-map
                       features' contributions f_k(m) beta_k to theta / phi / psi, the command u_inf, and the
                       second-order stage u'' = K (u_inf - u) - C u' as command and gaze in the theta-phi plane
          the eyes     from above, each turning with its angle (`unit: deg`), its target dashed
        and the kinograph of every cell across the bottom. The circuit is captured tick by tick from the engine,
        the rest is handed in afterwards (`set_traces`, `set_muscles`, `set_plant`); every frame is drawn after
        the rollout, so all share the final colour limits."""

        def __init__(self, *a, drive_set=None, drive_block=None, **k):
            st_ = dict(k.get("style") or {})
            k["style"] = dict(st_, panel=dict(st_.get("panel") or {}, input_column=False, output_column=False))
            super().__init__(*a, **k)
            self.drive_set, self.drive_block = drive_set, drive_block
            self.tr, self.mus, self.plant, self.drv = None, None, None, []

        # -------------------------------------------------------------- what is handed in
        def set_traces(self, t, drive, target, obs, cmd=None, unit="", label="", eyes=None):
            """target, obs, cmd: [T, E] -- one column per observed element (an eye)."""
            f = lambda x: None if x is None or not len(x) else np.asarray(x, np.float64).reshape(len(x), -1)  # noqa: E731
            self.tr = dict(t=np.asarray(t, np.float64), drive=np.asarray(drive, np.float64).reshape(-1),
                           target=f(target), obs=f(obs), cmd=f(cmd), unit=str(unit or ""), label=str(label or ""))
            E = self.tr["obs"].shape[1]
            self.tr["eyes"] = list(eyes) if eyes and len(eyes) == E else (["L", "R"] if E == 2 else
                                                                         [""] if E == 1 else [str(e) for e in range(E)])

        def set_muscles(self, drives, names):
            """drives [T, E, M]: each eye's muscle drives."""
            self.mus = dict(d=np.asarray(drives, np.float64), names=list(names))

        def set_plant(self, beta, K, C, pose, pose_target, pairs, axes=("θ", "φ", "ψ")):
            """THE EYE PLANT, a Hammerstein cascade (muscle_ops): the STATIC stage `muscle_pose_map` -- each eye's six
            drives m as the 27 features [m, m^2, m_i m_j over the 15 pairs] times `beta` (27 x 3), the command u_inf
            -- and the SECOND-ORDER stage `organ_mechanics`, u'' = K (u_inf - u) - C u'. pose, pose_target [T, E, 3];
            the drives come from `set_muscles`."""
            d = self.mus["d"]                                                         # [T, E, 6]
            cross = (np.stack([d[..., i] * d[..., j] for i, j in pairs], -1) if len(pairs)
                     else np.zeros(d.shape[:2] + (0,)))
            feat = np.concatenate([d, d ** 2, cross], -1)                             # [T, E, 27]
            beta = np.asarray(beta, np.float64)
            m = self.mus["names"][:d.shape[-1]]
            K, C = np.asarray(K, np.float64), np.asarray(C, np.float64)
            wn = np.sqrt(np.clip(np.linalg.eigvalsh(0.5 * (K + K.T)), 0, None))     # rad/s, no mass term
            zeta = np.linalg.eigvalsh(0.5 * (C + C.T)) / (2 * np.maximum(wn, 1e-9))
            self.plant = dict(contrib=feat[..., None] * beta, axes=list(axes), wn=wn, zeta=zeta,
                              labels=m + [f"{x}²" for x in m] + [f"{m[i]}·{m[j]}" for i, j in pairs],
                              pose=np.asarray(pose, np.float64), target=np.asarray(pose_target, np.float64))

        # -------------------------------------------------------------- the circuit's own input
        def _read(self, H):
            r, om = super()._read(H)
            if self.drive_set is None or self.drive_set not in H.levels:
                return r, om
            src_l = H.level(self.drive_set)
            src = src_l.get(self.drive_block).detach().float().cpu().numpy().reshape(int(src_l.n), -1)[:, 0]
            self.drv.append(src.copy())
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

        # -------------------------------------------------------------- the layout
        def _figure(self):
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            wide = self.plant is not None
            fig = plt.figure(figsize=(25.0 if wide else 20.0, 9.6), facecolor=BG, dpi=100)
            wr = [0.80, 1.75, 1.45, 0.80] if wide else [0.85, 1.80, 0.90]
            gs = GridSpec(2, len(wr), width_ratios=wr, height_ratios=[1.0, 0.52], hspace=0.16, wspace=0.07,
                          left=0.055, right=0.99, top=0.93, bottom=0.07)
            self.ax_tr = fig.add_subplot(gs[0, 0])
            self.ax_pl = fig.add_subplot(gs[0, 2]) if wide else None
            self.ax_eye = fig.add_subplot(gs[0, len(wr) - 1])
            self.ax_c = fig.add_subplot(gs[0, 1])
            return fig, self.ax_c, fig.add_subplot(gs[1, :])

        def _rows_of(self, role):
            """The matrix rows (sorted order) of the types with `role`, as (first, last + 1), or None."""
            types = (((getattr(self.sim, "sets", None) or {}).get(self.nset) or {}).get("types") or {})
            nms = [k for k, v in types.items() if (v or {}).get("role") == role]
            spans = [(a0, a1) for nm, a0, a1 in self.blocks if nm in nms]
            return (min(a for a, _ in spans), max(b for _, b in spans)) if spans else None

        def _y(self, row):
            g = self.geom
            return g["hi"] - (g["hi"] - g["lo"]) * row / self.N

        def _build_extra(self, fig):
            import matplotlib.pyplot as plt
            self.txt.set_visible(False)                       # no header line across the top (Cedric, 2026-10-08)
            a, g = self.ax_c, self.geom
            # THE INPUT: the drive set's values, then the afferent cells' column beside their own rows
            self.aff = self._rows_of("afferent") or (0, self.N)
            y1, y0 = self._y(self.aff[0]), self._y(self.aff[1])
            nd = max(1, len(self.drv[0]) if self.drv else 1)
            self.im_drv = a.imshow(np.zeros((nd, 1)), cmap=self.cmap, vmin=-1, vmax=1, extent=(0.0, 0.03, y0, y1),
                                   aspect="auto", zorder=3, interpolation="nearest")
            a.add_patch(plt.Rectangle((0.0, y0), 0.03, y1 - y0, fill=False, ec="#777", lw=0.5, zorder=5))
            a.text(0.015, y0 - 0.012, "drive", color="#fff", fontsize=9, ha="center", va="top")
            self.im_aff = a.imshow(np.zeros((self.aff[1] - self.aff[0], 1)), cmap=self.cmap, vmin=-1, vmax=1,
                                   extent=(0.07, 0.10, y0, y1), aspect="auto", zorder=3, interpolation="nearest")
            a.add_patch(plt.Rectangle((0.07, y0), 0.03, y1 - y0, fill=False, ec="#777", lw=0.5, zorder=5))
            a.text(0.085, y1 + 0.012, f"afferent ({self.aff[1] - self.aff[0]})", color="#fff", fontsize=9,
                   ha="center", va="bottom")
            ym = 0.5 * (y0 + y1)
            for x0, x1 in ((0.033, 0.067), (0.103, 0.135)):
                a.annotate("", xy=(x1, ym), xytext=(x0, ym), arrowprops=dict(arrowstyle="-|>", color="#fff", lw=1.0))
            # THE RATE COLUMN BY ROLE, and THE OUTPUT: the output cells feeding the muscle grid
            for role, nm in (("recurrent", "recurrent"), ("output", "output")):
                sp = self._rows_of(role)
                if sp:
                    a.text(0.772, 0.5 * (self._y(sp[0]) + self._y(sp[1])), f"{nm}\n({sp[1] - sp[0]})", color="#ddd",
                           fontsize=8.5, ha="left", va="center")
            self.grid = None
            if self.mus is not None:
                M, E = self.mus["d"].shape[2], self.mus["d"].shape[1]
                out = self._rows_of("output") or (0, self.N)
                yo = 0.5 * (self._y(out[0]) + self._y(out[1]))
                h = 0.045 * M
                x0 = 0.86
                self.grid = a.imshow(np.zeros((M, E)), cmap="viridis", vmin=0, vmax=float(self.mus["d"].max()) or 1.0,
                                     extent=(x0, x0 + 0.035 * E, yo - h / 2, yo + h / 2), aspect="auto", zorder=3,
                                     interpolation="nearest")
                a.add_patch(plt.Rectangle((x0, yo - h / 2), 0.035 * E, h, fill=False, ec="#777", lw=0.5, zorder=5))
                eyes = self.tr["eyes"] if self.tr else [str(e) for e in range(E)]
                for e, nm in enumerate(eyes):
                    a.text(x0 + 0.035 * (e + 0.5), yo + h / 2 + 0.01, nm, color="#fff", fontsize=10, ha="center",
                           va="bottom")
                for k_, nm in enumerate(self.mus["names"][:M]):
                    a.text(x0 + 0.035 * E + 0.006, yo + h / 2 - h * (k_ + 0.5) / M, nm, color="#fff", fontsize=9,
                           ha="left", va="center")
                a.text(x0 + 0.0175 * E, yo - h / 2 - 0.012, "muscle drive", color="#ddd", fontsize=8.5,
                       ha="center", va="top")
                a.annotate("", xy=(x0 - 0.004, yo), xytext=(0.77, yo),
                           arrowprops=dict(arrowstyle="-|>", color="#fff", lw=1.0))
            if self.tr is not None:
                self._build_trace()
                self._build_eyes()
            if self.plant is not None:
                self._build_plant()

        def _build_trace(self):
            tr, a = self.tr, self.ax_tr
            t = tr["t"]
            a.set_facecolor(BG)
            for k_, sp in a.spines.items():
                sp.set_visible(k_ in ("left", "bottom")); sp.set_color("#888")
            a.tick_params(colors="#bbb", labelsize=9)
            ys = [x.ravel() for x in (tr["target"], tr["obs"], tr["cmd"]) if x is not None]
            lo, hi = np.percentile(np.concatenate(ys), [0.5, 99.5])
            pad = 0.12 * max(hi - lo, 1e-9)
            ad = a.twinx()                                    # the drive, dotted, on its own scale
            ad.plot(t, tr["drive"][:len(t)], color="#9aa0a6", lw=0.6, ls=":", alpha=0.7)
            ad.set_yticks([]); [sp.set_visible(False) for sp in ad.spines.values()]
            a.plot([], [], color="#9aa0a6", lw=0.8, ls=":", label="drive (own scale)")
            E = tr["obs"].shape[1]
            self.ln = []
            for e in range(E):
                ls = "-" if e == 0 else "--"
                lab = (lambda s: s if e == 0 else None)
                L = {"target": a.plot([], [], color="#5fd08a", lw=2.0, ls=ls, label=lab("target"))[0],
                     "obs": a.plot([], [], color="#ffffff", lw=1.2, ls=ls, label=lab("model"))[0]}
                if tr["cmd"] is not None:
                    L["cmd"] = a.plot([], [], color="#ff4d4d", lw=0.9, ls=ls, alpha=0.85, label=lab("command"))[0]
                self.ln.append(L)
            self.cur = a.axvline(t[0], color="#ff7f0e", lw=1.0)
            a.set_xlim(t[0], t[-1]); a.set_ylim(lo - pad, hi + pad)
            a.set_xlabel("time (s)", color="#ddd", fontsize=11)
            a.set_ylabel(f"{tr['label']}" + (f" ({tr['unit']})" if tr["unit"] else ""), color="#ddd", fontsize=11)
            leg = a.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
            for tx in leg.get_texts():
                tx.set_color("#ddd")
            if E == 2:
                a.text(1.0, -0.13, f"{tr['eyes'][0]} eye solid, {tr['eyes'][1]} eye dashed; each in its own frame",
                       transform=a.transAxes, color="#aaa", fontsize=8.5, ha="right", va="top")
            self.txt_tr = a.text(0.0, 1.02, "", transform=a.transAxes, color="#ddd", fontsize=10, va="bottom")

        def _build_eyes(self):
            import matplotlib.pyplot as plt
            tr, e_ = self.tr, self.ax_eye
            E = tr["obs"].shape[1]
            e_.set_facecolor(BG); e_.axis("off"); e_.set_aspect("equal")
            self.eye_on = tr["unit"] in ("deg", "rad")
            xs = [2.3 * (k - (E - 1) / 2) for k in range(E)]
            e_.set_xlim(min(xs) - 1.7, max(xs) + 1.7); e_.set_ylim(-2.1, 2.0)
            e_.text(0.5 * (xs[0] + xs[-1]), 1.95, "the eyes, from above" if E > 1 else "the eye, from above",
                    color="#ddd", fontsize=11, ha="center", va="top")
            self.eye_art = []
            for k, x0 in enumerate(xs):
                if not self.eye_on:
                    break
                e_.add_patch(plt.Circle((x0, 0), 1.0, fc="#1b1b1b", ec="#bbbbbb", lw=1.5))
                tg = e_.plot([], [], color="#5fd08a", lw=1.4, ls="--")[0]
                ln = e_.plot([], [], color="#ffffff", lw=2.2)[0]
                pu = plt.Circle((x0, 0.85), 0.17, fc="#4a9bff", ec="white", lw=0.8)
                e_.add_patch(pu)
                tx = e_.text(x0, -1.12, "", color="#ddd", fontsize=9.5, ha="center", va="top", linespacing=1.3)
                e_.text(x0 - 1.15, 0.95, tr["eyes"][k] if E > 1 else "", color="#fff", fontsize=12, ha="center",
                        va="center", weight="bold")
                self.eye_art.append((x0, tg, ln, pu, tx))

        def _build_plant(self):
            """the static map per eye -> the command -> the second-order body, left to right."""
            from matplotlib.colors import LinearSegmentedColormap
            pl, a = self.plant, self.ax_pl
            a.set_facecolor(BG); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
            bkr = LinearSegmentedColormap.from_list("bkr", ["#4a9bff", "#000000", "#ff4a3a"])
            lo, hi = 0.06, 0.90
            E, F = pl["contrib"].shape[1], pl["contrib"].shape[2]
            eyes = self.tr["eyes"] if self.tr else [str(e) for e in range(E)]
            self.c_lim = float(np.percentile(np.abs(pl["contrib"]), 99.5)) or 1.0
            w_ = 0.045                                        # one axis column
            x0 = 0.10
            self.p_map = a.imshow(np.zeros((F, 3 * E)), cmap=bkr, vmin=-self.c_lim, vmax=self.c_lim,
                                  extent=(x0, x0 + w_ * 3 * E, lo, hi), aspect="auto", interpolation="nearest")
            for k_, lab in enumerate(pl["labels"]):
                a.text(x0 - 0.005, hi - (hi - lo) * (k_ + 0.5) / F, lab, color="#ccc", fontsize=6.5, ha="right",
                       va="center")
            for e in range(E):
                for j_, ax_n in enumerate(pl["axes"]):
                    a.text(x0 + w_ * (3 * e + j_ + 0.5), hi + 0.008, ax_n, color="#fff", fontsize=9, ha="center",
                           va="bottom")
                if E > 1:
                    a.text(x0 + w_ * (3 * e + 1.5), hi + 0.045, eyes[e], color="#fff", fontsize=10, ha="center",
                           va="bottom")
                    if e:
                        a.plot([x0 + w_ * 3 * e] * 2, [lo, hi], color="#777", lw=1.0)
            a.text(x0 + w_ * 1.5 * E, hi + 0.09, "static map: 27 features x beta", color="#fff", fontsize=9.5,
                   ha="center", va="bottom")
            a.text(x0 + w_ * 1.5 * E, lo - 0.015, f"f(m) beta, deg (up to {self.c_lim:.2g})", color="#bbb", fontsize=8,
                   ha="center", va="top")
            xc = x0 + w_ * 3 * E + 0.06
            t_lim = float(np.percentile(np.abs(pl["target"]), 99.5)) or 1.0
            self.p_cmd = a.imshow(np.zeros((3, E)), cmap=bkr, vmin=-t_lim, vmax=t_lim,
                                  extent=(xc, xc + 0.035 * E, 0.5 - 0.09, 0.5 + 0.09), aspect="auto",
                                  interpolation="nearest")
            a.text(xc + 0.0175 * E, 0.5 + 0.10, "command u∞", color="#fff", fontsize=9, ha="center", va="bottom")
            self.p_cmd_txt = [a.text(xc + 0.035 * E + 0.006, 0.5 + 0.09 - 0.06 * (j_ + 0.5), "", color="#ddd",
                                     fontsize=8, ha="left", va="center") for j_ in range(3)]
            for xa, xb in ((x0 + w_ * 3 * E + 0.008, xc - 0.008), (xc + 0.035 * E + 0.10, xc + 0.035 * E + 0.13)):
                a.annotate("", xy=(xb, 0.5), xytext=(xa, 0.5), arrowprops=dict(arrowstyle="-|>", color="#fff", lw=1.0))
            bx = xc + 0.035 * E + 0.16
            b = a.inset_axes([bx, 0.30, 0.99 - bx, 0.52])
            b.set_facecolor(BG)
            for sp in b.spines.values():
                sp.set_color("#666")
            b.tick_params(colors="#bbb", labelsize=8)
            th = np.concatenate([pl["target"][..., 0].ravel(), pl["pose"][..., 0].ravel()])
            ph = np.concatenate([pl["target"][..., 1].ravel(), pl["pose"][..., 1].ravel()])
            tl = 1.1 * float(np.percentile(np.abs(th), 99.5)) or 1.0
            pp = max(1.3 * float(np.percentile(np.abs(ph), 99.5)), 1.0)
            b.set_xlim(tl, -tl); b.set_ylim(-pp, pp)
            b.axhline(0, color="#444", lw=0.6); b.axvline(0, color="#444", lw=0.6)
            b.set_xlabel(f"{pl['axes'][0]} horizontal (deg)", color="#ddd", fontsize=9)
            b.set_ylabel(f"{pl['axes'][1]} vertical (deg)", color="#ddd", fontsize=9)
            self.p_eye = []
            for e in range(E):
                ls = "-" if e == 0 else "--"
                self.p_eye.append((b.plot([], [], color="#ff4d4d", lw=1.0, ls=ls, alpha=0.6)[0],
                                   b.plot([], [], color="#4a9bff", lw=1.4, ls=ls, alpha=0.8)[0],
                                   b.plot([], [], "o" if e == 0 else "s", color="#ff4d4d", ms=7)[0],
                                   b.plot([], [], "o" if e == 0 else "s", color="#4a9bff", ms=8, mec="white", mew=0.8)[0]))
            b.set_title("second order: u'' = K (u∞ - u) - C u'", color="#fff", fontsize=9.5, pad=4)
            a.text(bx + 0.5 * (0.99 - bx), 0.13, f"natural frequencies {pl['wn'].min() / (2 * np.pi):.2f}-"
                   f"{pl['wn'].max() / (2 * np.pi):.2f} Hz, damping ratio {pl['zeta'].min():.2f}-{pl['zeta'].max():.2f}",
                   color="#bbb", fontsize=8, ha="center", va="top")
            a.text(bx + 0.5 * (0.99 - bx), 0.08, "red: command u∞   blue: gaze u" + ("   (circle L, square R)" if E > 1 else ""),
                   color="#bbb", fontsize=8, ha="center", va="top")

        # -------------------------------------------------------------- one frame
        def _draw_extra(self, idx: int):
            tick = int(self.hist[idx][0])
            r_, om = self.hist[idx][1], self.hist[idx][2]
            # the circuit's own input: the drive box and the afferent column
            if self.drv:
                d = np.asarray(self.drv[min(idx, len(self.drv) - 1)], np.float64)
                lim_d = float(np.max(np.abs(np.stack(self.drv)))) or 1.0
                self.im_drv.set_data((d / lim_d)[:, None])
            v = om[self.order][self.aff[0]:self.aff[1]].astype(np.float64)
            lim = float(np.max(np.abs(v))) or 1.0
            self.im_aff.set_data((v / lim)[:, None])
            if self.grid is not None:
                k = int(np.clip(tick, 0, len(self.mus["d"]) - 1))
                self.grid.set_data(self.mus["d"][k].T)                       # [M, E]
            if self.tr is not None:
                self._draw_trace(tick)
            if self.plant is not None:
                self._draw_plant(tick)

        def _draw_trace(self, tick):
            tr = self.tr
            t = tr["t"]
            k = int(np.clip(tick, 0, len(t) - 1))                # tick j = engine frame j = trace sample j
            for e, L in enumerate(self.ln):
                for key, ln in L.items():
                    ln.set_data(t[:k + 1], tr[key][:k + 1, e])
            self.cur.set_xdata([t[k]] * 2)
            u_ = tr["unit"]
            parts = [(f"{tr['eyes'][e]}: " if tr["eyes"][e] else "") + f"target {tr['target'][k, e]:+.1f} model "
                     f"{tr['obs'][k, e]:+.1f}" for e in range(tr["obs"].shape[1])]
            self.txt_tr.set_text(f"t = {t[k]:5.2f} s   " + "   ".join(parts) + f" {u_}")
            for e, (x0, tg, ln, pu, tx) in enumerate(self.eye_art):
                to = np.deg2rad if u_ == "deg" else float
                th, tg_ = to(tr["obs"][k, e]), to(tr["target"][k, e])
                # each eye in its own frame (+ abduction): the left eye abducts toward -x, the right toward +x
                sgn = -1.0 if (len(self.eye_art) == 2 and e == 0) else 1.0
                ln.set_data([x0, x0 + 1.6 * sgn * np.sin(th)], [0, 1.6 * np.cos(th)])
                tg.set_data([x0, x0 + 1.6 * sgn * np.sin(tg_)], [0, 1.6 * np.cos(tg_)])
                pu.center = (x0 + 0.85 * sgn * np.sin(th), 0.85 * np.cos(th))
                tx.set_text(f"gaze {tr['obs'][k, e]:+.1f} {u_}\ntarget {tr['target'][k, e]:+.1f} {u_}")

        def _draw_plant(self, tick):
            pl = self.plant
            k = int(np.clip(tick, 0, pl["contrib"].shape[0] - 1))
            E = pl["contrib"].shape[1]
            self.p_map.set_data(np.concatenate([pl["contrib"][k, e] for e in range(E)], 1))      # [27, 3E]
            self.p_cmd.set_data(pl["target"][k].T)                                               # [3, E]
            for j_, tx in enumerate(self.p_cmd_txt):
                tx.set_text(f"{pl['axes'][j_]} " + " ".join(f"{pl['target'][k, e, j_]:+.1f}" for e in range(E)))
            w0 = max(0, k - 120)                                  # the last 2 s at 60 Hz
            for e, (c_tr, g_tr, c_pt, g_pt) in enumerate(self.p_eye):
                c_tr.set_data(pl["target"][w0:k + 1, e, 0], pl["target"][w0:k + 1, e, 1])
                g_tr.set_data(pl["pose"][w0:k + 1, e, 0], pl["pose"][w0:k + 1, e, 1])
                c_pt.set_data([pl["target"][k, e, 0]], [pl["target"][k, e, 1]])
                g_pt.set_data([pl["pose"][k, e, 0]], [pl["pose"][k, e, 1]])

    return TaskCircuitPanel


def circuit_movie(spec, device="cpu", root=None, *, trials=None, fps=None, stride=None, out=None, quiet=False):
    """A trained corpus run's circuit at work on held-out trials -> results/movie_circuit.mp4 (+ .png).

    `trials` consecutive held-out trials of the first trial's condition cell are joined into one drive (the target
    re-run by that cell's teacher law across the joins, `_teacher_on_cell`); the model is rolled out once with the
    panel capturing every engine frame (`trainer.rollout(..., watch=)`); every `stride`-th frame is drawn, at
    `fps`. With `task.observe.elements: all` every observed element (each eye) is drawn. The defaults come from the
    training spec's `training.circuit_movie: {trials, fps, stride}` (2, 30, 2: a 60-Hz model plays in real time).
    Read by `trainer.analyse` (`Plexus_Main.py -o analyse <name>`) when the spec declares it, and by
    `python -m plexus.tasks.plot_trainer --circuit <training spec>`."""
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
    every = task["observe"].get("elements") == "all"
    dt = float(sim.dt)
    if len(pick) == 1:
        y = Y[pick[0]].cpu().numpy()                                         # [T, K]
    else:
        y = np.asarray(_teacher_on_cell(corpus, u[None, :, :1].cpu().numpy(), c[pick[:1]])).reshape(len(u), -1)
    path = out or os.path.join(out_d, "results", "movie_circuit.mp4")
    Panel = _task_circuit_panel_class()
    panel = Panel(out=path, n_frames=int(u.shape[0]), sim=sim, style={"panel": {"kino_frames": min(int(u.shape[0]), 600)}},
                  dt=dt, time_s=1.0, name=spec["name"], fps=fps, max_frames=int(u.shape[0]) + 1, stills=0,
                  drive_set=task["drive"]["set"], drive_block=task["drive"]["block"])
    obs_set, obs_block = task["observe"]["set"], task["observe"]["block"]
    rec, tick = {"cmd": [], "mus": [], "mus_names": [], "pose": [], "pose_target": [], "plant": None, "mus_set": None}, [0]

    def watch(H):                                       # once per engine frame, frame 0 (the seeded state) first
        panel.capture(H, tick[0])
        tick[0] += 1
        if rec["plant"] is None:                        # the eye plant's two operators, when the model has them
            ops = dict(zip(getattr(H, "operator_names", []), getattr(H, "operators", [])))
            pm, om = ops.get("muscle_pose_map"), ops.get("organ_mechanics")
            rec["plant"] = ((pm._beta.detach().cpu().numpy(), om._K.detach().cpu().numpy(), om._C.detach().cpu().numpy())
                            if pm is not None and om is not None else False)
            rec["mus_set"] = next((n for n, l_ in H.levels.items() if "drive" in getattr(l_, "state_schema", {})
                                   and getattr(l_, "pre_name", None) is None and int(l_.n) <= 64), None)
            if rec["mus_set"] is not None:
                rec["mus_names"] = list(getattr(H.level(rec["mus_set"]), "type_names", []) or [])
        lv = H.level(obs_set)
        E_ = int(lv.n) if every else 1
        st = lambda b_: lv.get(b_).detach().float().cpu().numpy().reshape(int(lv.n), -1)[:E_]   # noqa: E731
        if f"{obs_block}_target" in lv.state_schema:
            rec["cmd"].append(st(f"{obs_block}_target")[:, ch])
            rec["pose_target"].append(st(f"{obs_block}_target")[:, :3])
        rec["pose"].append(st(obs_block)[:, :3])
        if rec["mus_set"] is not None:
            rec["mus"].append(H.level(rec["mus_set"]).get("drive").detach().float().cpu().numpy().reshape(-1))
    sim.n_frames = int(u.shape[0])                      # the joined trials' length, as training sets T_full
    with torch.no_grad():
        _, Yp = T.rollout(sim, learn, u, task, device, grad=False, watch=watch)
    Yp = Yp.cpu().numpy()                               # [frames + 1, E] (elements: all) or [frames + 1, w]
    obs = Yp if every else Yp[:, ch:ch + 1]
    n_ = min(len(obs), len(y), len(panel.hist))
    E = obs.shape[1]
    if rec["mus"]:
        mus = np.stack(rec["mus"][:n_])
        M = mus.shape[1] // max(E, 1) if mus.shape[1] % max(E, 1) == 0 else mus.shape[1]
        panel.set_muscles(mus.reshape(n_, -1, M)[:, :E] if mus.shape[1] == M * E else mus[:, None, :],
                          rec["mus_names"] or [str(i) for i in range(M)])
    panel.set_traces(np.arange(n_) * dt, u[:n_, 0].cpu().numpy(), y[:n_, :E], obs[:n_],
                     cmd=(np.stack(rec["cmd"][:n_]) if rec["cmd"] else None), unit=task["observe"].get("unit") or "",
                     label=f"{obs_set}.{obs_block}")
    if rec["plant"] and rec["mus"] and rec["pose_target"]:
        from plexus.operators.muscle_ops import PAIRS
        beta, K_, C_ = rec["plant"]
        panel.set_plant(beta, K_, C_, np.stack(rec["pose"][:n_]), np.stack(rec["pose_target"][:n_]), PAIRS)
    idxs = list(range(stride - 1, n_, stride)) or [n_ - 1]
    w = iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    for i in idxs:
        w.append_data(panel.frame_at(i))
    w.close()
    iio.imwrite(path[:-4] + ".png", panel.frame_at(idxs[len(idxs) // 2]))
    panel.close()
    if not quiet:
        print(f"[circuit-movie] {path}: {len(idxs)} frames at {fps:g} fps, {len(pick)} held-out trial(s) of "
              f"{split}, {panel.N} neurons, {panel.pre.size:,} synapses, {E} observed element(s)", flush=True)
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
