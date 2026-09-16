"""One PNG per split, written beside the data on every generate.

Every other corpus in `graphs_data` carries a figure next to its arrays -- the GNN task datasets
write `task_traces_{train,test}.png`, the dot corpus writes its own `.png` and `.mp4` -- and for
the same reason: a corpus nobody looks at is a corpus whose defects are found by a training run
instead of by a glance. This module is that figure for a task.

Five panels, and the middle one is the reason the figure exists:

    a   a few stimulus traces, coloured by condition cell
    b   their targets, same colours -- what the teacher made of them
    c   THE EXCITATION PANEL: the ensemble's power spectrum with the teacher's pole frequencies
        marked, so "this mode cannot be identified" is a thing you SEE rather than a line of json
    d   the teacher's own Bode magnitude, known in closed form, over the same axis as (c)
    e   the condition grid and the split counts, as text
    f   THE R-L-C LADDER that realises the teacher -- one section per pole or conjugate pair,
        with component values. These tasks are not LIKE electronic filters, they ARE them, and
        this is the panel that shows it rather than asserting it.

Panel (c) against (d) is the whole argument of this package in one picture: the teacher's poles
have to sit where the stimulus has power, and both are computable before any circuit exists.

STYLE. White background, no box around the axes, and the panel letter set ABOVE the frame rather
than inside it -- a letter placed inside competes with the data for the same corner, and on a
dense trace panel it lands on top of a curve. Not bold: at this size the weight adds nothing the
position does not already give.
"""
from __future__ import annotations

import os

import numpy as np

BG = "white"
FG = "black"
LABEL_SIZE = 11
INK = "0.15"          # traces and text on white
MUTED = "0.45"        # axis furniture
# Condition cells get distinct hues, darkened for a white ground; a single-cell task gets one
# neutral colour, per the convention that a lone trace carries no comparison and needs no code.
CYCLE = ("#1f6fb8", "#c0522a", "#2e8b4f", "#9a7d1a", "#7a4fa0", "#1a8a8a")


def _ax(ax, xlabel=None, ylabel=None, letter=None):
    """One panel in the house style: white, no box, letter above the frame."""
    ax.set_facecolor(BG)
    for side, sp in ax.spines.items():
        sp.set_visible(side in ("left", "bottom"))
        sp.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if letter:
        # ABOVE the frame, not in it: inside, a letter competes with the data for the corner.
        ax.text(0.0, 1.04, letter, transform=ax.transAxes, color=FG, fontsize=LABEL_SIZE,
                va="bottom", ha="left")
    return ax


def render_split(spec, split, U, Y, cond, excitation, out_path, n_show=3):
    """Write the figure for one split. `excitation` is the per-cell report from `generate`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, dt = spec.T, spec.dt
    t = np.arange(T) * dt
    cells = spec.cells
    fig = plt.figure(figsize=(16, 7.4), facecolor=BG)
    gs = fig.add_gridspec(2, 4, hspace=0.36, wspace=0.28)

    # -- a, b: traces, a few per cell ------------------------------------------------------
    axa, axb = _ax(fig.add_subplot(gs[0, 0]), ylabel="stimulus", letter="a"), None
    axb = _ax(fig.add_subplot(gs[1, 0]), xlabel="time (s)", ylabel="target", letter="b")
    for c in range(len(cells)):
        idx = np.where(cond == c)[0][:n_show]
        col = CYCLE[c % len(CYCLE)] if len(cells) > 1 else INK
        for i in idx:
            axa.plot(t, U[i, :, 0], color=col, lw=0.7, alpha=0.85)
            axb.plot(t, Y[i, :, 0], color=col, lw=0.7, alpha=0.85)
    axa.set_xticklabels([])

    # -- c: the spectrum, with the poles on it ---------------------------------------------
    axc = _ax(fig.add_subplot(gs[0, 1:3]), ylabel="stimulus power (dB rel peak)", letter="c")
    f = np.fft.rfftfreq(T, d=dt)
    psd = (np.abs(np.fft.rfft(U, axis=1)) ** 2).mean(axis=(0, 2))
    db = 10 * np.log10(np.maximum(psd, 1e-300) / max(psd.max(), 1e-300))
    axc.plot(f[1:], db[1:], color=INK, lw=0.9)
    axc.axhline(excitation.get("per_cell", [{}])[0].get("floor_db", -20.0),
                color=MUTED, lw=0.8, ls="--")
    seen = set()
    for rep in excitation.get("per_cell", []):
        for p in rep.get("poles", []):
            key = round(p["freq_hz"], 4)
            if key in seen:
                continue
            seen.add(key)
            col = "#2e8b4f" if p["excited"] else "#c0272a"
            axc.axvline(max(p["freq_hz"], f[1]), color=col, lw=1.1, alpha=0.9)
            axc.text(max(p["freq_hz"], f[1]), 4, f"{p['freq_hz']:.2f} Hz", color=col,
                     fontsize=7, rotation=90, va="bottom", ha="center")
    axc.set_xscale("log")
    axc.set_xlim(max(f[1], 1e-2), 0.5 / dt)
    axc.set_ylim(-120, 22)
    verdict = ("IDENTIFIABLE" if excitation.get("identifiable") else
               "NOT IDENTIFIABLE - a pole sits where the stimulus has no power")
    axc.text(0.985, 0.05, verdict, transform=axc.transAxes, ha="right", va="bottom",
             fontsize=9, color="#2e8b4f" if excitation.get("identifiable") else "#c0272a")

    # -- d: the teacher's own magnitude response, closed form -------------------------------
    axd = _ax(fig.add_subplot(gs[1, 1]), xlabel="frequency (Hz)", ylabel="|H| (dB)", letter="d")
    mag = _bode(spec, f[1:])
    if mag is not None:
        axd.plot(f[1:], mag, color="#1f6fb8", lw=1.1)
        axd.set_xscale("log")
        axd.set_xlim(max(f[1], 1e-2), 0.5 / dt)
    else:
        axd.text(0.5, 0.5, "no H(s):\nstatic or pure delay", transform=axd.transAxes,
                 ha="center", va="center", color=MUTED, fontsize=9)
        axd.set_xticks([]); axd.set_yticks([])

    # -- e: what this corpus IS, in words ---------------------------------------------------
    axe = _ax(fig.add_subplot(gs[0, 3]), letter="e")
    axe.set_xticks([]); axe.set_yticks([])
    lines = ["", f"{spec.name}",
             f"split {split}: {U.shape[0]} trials x {T} frames ({spec.duration_s} s, dt={dt:.5f})",
             f"stimulus  {spec.process_name}",
             f"teacher   {spec.law_name}", ""]
    for k, v in sorted(spec.teacher.items()):
        s = str(v)
        lines.append(f"   {k} = {s[:38] + '...' if len(s) > 38 else s}")
    if spec.conditions:
        lines += ["", f"{len(cells)} condition cells:"]
        for k, v in sorted(spec.conditions.items()):
            s = str(v)
            lines.append(f"   {k}: {s[:34] + '...' if len(s) > 34 else s}")
    axe.text(0.05, 0.92, "\n".join(_wrap(lines, 44)), transform=axe.transAxes, va="top",
             ha="left", color=INK, fontsize=7.5, family="monospace")

    # -- f: the network that realises the teacher ------------------------------------------
    axf = _ax(fig.add_subplot(gs[1, 2:]), letter="f")
    from plexus.tasks import get_teacher
    from plexus.tasks.lti import rlc_stages
    law = get_teacher(spec.law_name)
    # THE FIRST CELL ONLY, and it says so. A grid over the TEACHER has a different network in
    # every cell -- a sweep over filter order is 1, 2 and 4 sections -- and drawing the first
    # without saying which would show the simplest member as if it were the family.
    n_cells = len(spec.cells)
    note = None
    if n_cells > 1:
        pk = sorted(set(spec.conditions) - set(spec.stimulus))
        note = (f"cell 1 of {n_cells}: " + ", ".join(f"{k}={spec.cells[0][k]}" for k in pk)
                if pk else f"cell 1 of {n_cells} (the grid varies the stimulus, not the network)")
    try:
        pl = law.poles(spec.dt, **spec.teacher) if hasattr(law, "poles") else []
        draw_rlc(axf, rlc_stages(pl), cell_note=note)
    except Exception:                                            # noqa: BLE001
        draw_rlc(axf, [], cell_note=note)

    fig.savefig(out_path, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _wrap(lines, width):
    """Hard-wrap at `width`, continuing over-long lines indented. A panel that overflows its
    axes is worse than one that wraps: the overflow lands on the neighbouring panel."""
    out = []
    for ln in lines:
        while len(ln) > width:
            out.append(ln[:width])
            ln = "     " + ln[width:]
        out.append(ln)
    return out


def _bode(spec, f):
    """|H(jw)| in dB for the laws that have one, else None.

    Computed from the SPEC, not from the generated data, because the point of panel (d) is that
    the teacher's response is known analytically and can be compared with what a fitted circuit
    later measures. Reading it off the data would make it a description of the corpus rather
    than of the law.
    """
    from plexus.tasks import get_teacher
    law = get_teacher(spec.law_name)
    w = 2 * np.pi * f
    try:
        if spec.law_name == "laplace":
            num, den = spec.teacher.get("num", (1.0,)), spec.teacher.get("den", (1.0, 1.0))
        elif spec.law_name == "integrate":
            tau = spec.teacher.get("tau_s")
            g = float(spec.teacher.get("gain", 1.0))
            num, den = [g], ([1.0, 0.0] if not tau else [1.0, 1.0 / float(tau)])
        elif spec.law_name == "lti":
            from scipy.signal import iirfilter
            p = spec.teacher
            kw = {}
            if p.get("family", "butter") in ("cheby1", "ellip"):
                kw["rp"] = float(p.get("rp", 1.0))
            if p.get("family", "butter") in ("cheby2", "ellip"):
                kw["rs"] = float(p.get("rs", 40.0))
            num, den = iirfilter(int(p.get("order", 2)),
                                 2 * np.pi * np.asarray(p.get("cutoff_hz", 1.0), float),
                                 btype=p.get("band", "lowpass"), ftype=p.get("family", "butter"),
                                 analog=True, output="ba", **kw)
        elif spec.law_name == "statespace":
            from scipy.signal import lti as _lti
            p = spec.teacher
            sys = _lti(np.asarray(p["A"], float), np.asarray(p["B"], float),
                       np.asarray(p["C"], float),
                       np.zeros((np.asarray(p["C"], float).shape[0],
                                 np.asarray(p["B"], float).shape[1])))
            _w, h = sys.freqresp(w=w)
            h = np.asarray(h)
            # a MIMO system has no single |H|; show the first output's response to the first
            # input, and say so by drawing only that one rather than averaging modes together
            h = h if h.ndim == 1 else h.reshape(len(w), -1)[:, 0]
            return 20 * np.log10(np.maximum(np.abs(h), 1e-30))
        else:
            return None
        from scipy.signal import freqs
        _w, h = freqs(np.atleast_1d(num), np.atleast_1d(den), worN=w)
        return 20 * np.log10(np.maximum(np.abs(h), 1e-30))
    except Exception:                                            # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
#  the R-L-C schematic
# --------------------------------------------------------------------------- #
# Component symbols, drawn from primitives rather than pulled from a library, because the only
# ones needed are three and a schematic package would be a dependency for six polylines.
def _wire(ax, x0, y0, x1, y1, c=INK, lw=1.0):
    ax.plot([x0, x1], [y0, y1], color=c, lw=lw, solid_capstyle="round")


def _resistor(ax, x0, x1, y, label=None, n=6, h=0.055):
    """The zigzag, IEC's rectangle being harder to read small."""
    xs = np.linspace(x0, x1, 2 * n + 2)
    ys = [y] + [y + h * (-1) ** i for i in range(2 * n)] + [y]
    ax.plot(xs, ys, color=INK, lw=1.0, solid_capstyle="round")
    if label:
        ax.text((x0 + x1) / 2, y + h + 0.05, label, ha="center", va="bottom",
                fontsize=6.5, color=INK)


def _inductor(ax, x0, x1, y, label=None, n=4, r=0.045):
    """Four half-loops on top of the wire."""
    for i in range(n):
        cx = x0 + (i + 0.5) * (x1 - x0) / n
        th = np.linspace(np.pi, 0, 24)
        ax.plot(cx + r * np.cos(th), y + r * np.sin(th), color=INK, lw=1.0)
    _wire(ax, x0, y, x1, y)
    if label:
        ax.text((x0 + x1) / 2, y + r + 0.06, label, ha="center", va="bottom",
                fontsize=6.5, color=INK)


def _capacitor(ax, x, y0, y1, label=None, w=0.07):
    """Two plates on a vertical shunt leg. `w` is the plate half-width, in x data units."""
    """Two plates, drawn across a vertical shunt leg."""
    ym = (y0 + y1) / 2
    _wire(ax, x, y0, x, ym + 0.035)
    _wire(ax, x, ym - 0.035, x, y1)
    _wire(ax, x - w, ym + 0.035, x + w, ym + 0.035)
    _wire(ax, x - w, ym - 0.035, x + w, ym - 0.035)
    if label:
        ax.text(x + w * 1.4, ym, label, ha="left", va="center", fontsize=6.5, color=INK)


def _ground(ax, x, y, w=0.075):
    for i, k in enumerate((1.0, 0.6, 0.25)):
        _wire(ax, x - w * k, y - i * 0.035, x + w * k, y - i * 0.035)


def draw_rlc(ax, stages, max_stages=3, aspect=2.25, cell_note=None):
    """`aspect` is the panel's width/height. Symbol sizes are in DATA units and the panel is far
    wider than tall, so without it a coil is an ellipse and a capacitor's plates sit a third of
    the panel apart. Setting xlim to the aspect makes one x unit equal one y unit."""
    """The ladder that realises the teacher, one section per pole or conjugate pair.

    This is the panel that makes the claim concrete: these tasks are not LIKE electronic filters,
    they ARE them -- the 160 named laws in `lti.py` are canonical R-L-C networks reduced to
    transfer functions, and this draws one back out. `order N` on a spec is N/2 boxes here.
    """
    from plexus.tasks.lti import eng
    W = float(aspect)
    ax.set_xlim(0, W); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if not stages:
        ax.text(0.5, 0.5, "no poles:\nno R-L-C realisation", ha="center", va="center",
                color=MUTED, fontsize=8)
        return
    if stages[0]["kind"] == "integrator":
        # 1/s is not a passive network: no combination of R, L and C gives infinite DC gain.
        y, ytop = 0.46, 0.80
        _wire(ax, 0.06 * W, y, 0.18 * W, y)
        _resistor(ax, 0.18 * W, 0.36 * W, y, "R")
        _wire(ax, 0.36 * W, y, 0.46 * W, y)
        _wire(ax, 0.46 * W, y, 0.46 * W, ytop)
        _wire(ax, 0.46 * W, ytop, 0.56 * W, ytop)
        _capacitor(ax, 0.60 * W, ytop + 0.07, ytop - 0.07, "C", w=0.05 * W)
        _wire(ax, 0.64 * W, ytop, 0.76 * W, ytop)
        _wire(ax, 0.76 * W, ytop, 0.76 * W, y)
        ax.add_patch(plt_polygon([(0.46 * W, y - 0.17), (0.46 * W, y + 0.17), (0.72 * W, y)]))
        _wire(ax, 0.72 * W, y, 0.90 * W, y)
        ax.text(0.5 * W, 0.12, "1/s is ACTIVE, not passive: no R-L-C gives infinite DC gain.\n"
                               "An op-amp with C in feedback is the realisation.",
                ha="center", va="center", fontsize=6.5, color=MUTED)
        ax.text(0.03 * W, y, "in", ha="right", va="center", fontsize=7, color=MUTED)
        ax.text(0.93 * W, y, "out", ha="left", va="center", fontsize=7, color=MUTED)
        return

    shown = stages[:max_stages]
    n = len(shown)
    y, ygnd = 0.60, 0.26
    span = W / n
    warn = False
    for k, st in enumerate(shown):
        x0 = k * span + 0.03 * W / n
        w = span - 0.06 * W / n
        if st["kind"] == "RC":
            _wire(ax, x0, y, x0 + 0.10 * w, y)
            _resistor(ax, x0 + 0.10 * w, x0 + 0.50 * w, y, eng(st["R"], "Ω"))
            _wire(ax, x0 + 0.50 * w, y, x0 + 0.72 * w, y)
        else:
            _wire(ax, x0, y, x0 + 0.06 * w, y)
            _resistor(ax, x0 + 0.06 * w, x0 + 0.34 * w, y, eng(st["R"], "Ω"))
            _inductor(ax, x0 + 0.38 * w, x0 + 0.66 * w, y, eng(st["L"], "H"))
            _wire(ax, x0 + 0.66 * w, y, x0 + 0.72 * w, y)
            warn |= not st.get("passive_practical", True)
        xc = x0 + 0.72 * w
        _capacitor(ax, xc, y, ygnd, eng(st["C"], "F"), w=0.035 * W)
        _ground(ax, xc, ygnd, w=0.04 * W)
        _wire(ax, xc, y, x0 + w, y)
        ax.text(x0 + 0.5 * w, 0.94, f"{st['f_hz']:.3g} Hz"
                + (f"  ζ={st['zeta']:.2f}" if "zeta" in st else ""),
                ha="center", va="top", fontsize=6.5, color=MUTED)
    ax.text(0.0, y, "in", ha="right", va="center", fontsize=7, color=MUTED)
    ax.text(W, y, "out", ha="left", va="center", fontsize=7, color=MUTED)
    tail = f"  (+{len(stages) - n} more section{'s' if len(stages) - n > 1 else ''})" \
        if len(stages) > n else ""
    note = f"{len(stages)} section{'s' if len(stages) != 1 else ''}{tail}"
    if cell_note:
        note += f"  —  {cell_note}"
    if warn:
        note += "  —  L is impractically large at this frequency: realise actively"
    ax.text(0.5 * W, 0.04, note, ha="center", va="bottom", fontsize=6.5, color=MUTED)


def plt_polygon(pts):
    from matplotlib.patches import Polygon
    return Polygon(pts, closed=True, fill=False, edgecolor=INK, lw=1.0)
