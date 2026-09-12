"""The circuit panel: the message on the connectivity matrix, the input and output vectors beside
it, the kinograph of every neuron below -- `plotting.renderer: neural_panel`.

Reproduces the middle-and-bottom panels of connectome-gnn's `prototype/dot_tracking/test_zebra_eyeG.py`
(Figure 1c of the oculomotor note, made live) for any Plexus spec with a neuron set and a synapse
edge-set:

  the matrix       post x pre, one pixel per synapse. COLOUR is the Dale sign of the PRESYNAPTIC
                   column -- blue excitatory, red inhibitory -- read from the pre neuron's type
                   (`sign: E|I` on the type) or, without types that say, from the sign of W_e.
                   BRIGHTNESS is |W_ij r_j|, the message crossing that synapse THIS frame, on a
                   log1p scale against one fixed limit, so two frames are comparable. Sign from
                   the anatomy, weight from the spec, brightness from the traffic. The sign is
                   deliberately the column's and not the message's: r_j = tanh v is signed, so
                   sign(W_ij r_j) would flicker with every rate zero-crossing and the E/I band
                   structure -- the one thing the panel exists to make falsifiable by eye --
                   would strobe away.
  input vector     the drive each neuron receives from outside the circuit, level with its own
                   row: the `omega` block (what `neuron_field_input` wrote) and, when the spec
                   names them, the afferent types' rows only.
  rate vector      every neuron's rate r = tanh v, level with its row, so a column can be read
                   straight across into the matrix.
  output vector    one value per readout group: the mean rate of each `plotting.panel.output`
                   type (a motor pool, an assembly), the thing the circuit drives.
  kinograph        every neuron's rate against time, a sliding window under the matrix whose rows
                   they are; a leak or a drift in an integrator is visible here and nowhere else.

Rows are the neurons sorted by type in the spec's declared order (afferent -> recurrent ->
output when the spec is written that way), so the type bands are contiguous and labelled.

RATE LUT. r = tanh v is heavily skewed on a real circuit (median |r| a few percent), so a linear
+-1 map renders most cells at 2% intensity. A signed power law against the 90th percentile of
|r| seen so far (gamma 0.35) puts the median near 45% and clips the top decile, which is the
trade the panels want: the point is which cells are active at all.

The same object serves the pipeline (it is the `on_frame` hook, writes movie.mp4, the stills
and 3d.png exactly as `LiveMovie` does) and the page (it keeps every rendered frame's rates, so
`frame_at(i)` redraws any of them for PLAY).
"""
from __future__ import annotations

import os
import time

import numpy as np

BG = "#000000"
EI_E, EI_I = "#1f4fd8", "#d81b26"
FS_AXIS, FS_TICK, FS_NOTE = 11.0, 9.0, 9.0


def _ei_cmap():
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    cm = _LSC.from_list("ei_black", [EI_I, "#3a0a0e", "#000000", "#0a1440", EI_E])
    cm.set_bad(BG)
    return cm


class NeuralPanel:
    def __init__(self, out: str, n_frames: int, sim=None, style: dict | None = None, *,
                 max_frames: int = 300, stills: int = 10, keep_stills: bool = False, dt=None,
                 time_s=None, name: str = "", fps: float | None = None, px: int = 1000, **_ignored):
        import matplotlib
        matplotlib.use("Agg")
        self.out, self.n_frames, self.sim = out, int(n_frames), sim
        self.style = dict(style or {})
        self.cfg = dict(self.style.get("panel") or {})
        self.name = name or getattr(sim, "name", "")
        self.stride = max(1, int(np.ceil(max(1, self.n_frames) / max(1, int(max_frames)))))
        self.dt, self.time_s = dt, time_s
        self.fps = float(fps or self.style.get("fps") or 30.0)
        self.keep_stills = bool(keep_stills)
        _rendered = list(range(self.stride, self.n_frames + 1, self.stride)) or [self.n_frames]
        n_st = max(0, int(stills))
        self.still_ticks = ({_rendered[i] for i in np.unique(np.linspace(0, len(_rendered) - 1, n_st).astype(int))}
                            if n_st else set())
        self.still_dir = os.path.dirname(out) or "."
        self._still_paths: list = []
        self.stills_written = 0
        self.failed = None
        self.rendered = 0
        self.drawn = 0
        self.t0 = None
        self.writer = None
        self.fig = None
        self.hist: list = []                                 # (tick, r[N], omega[N]) per rendered frame
        self.msg_lim = float(self.cfg.get("msg_lim", 0.0)) or None
        self.rate_lim = float(self.cfg.get("rate_lim", 0.0)) or None
        self.kino_w = int(self.cfg.get("kino_frames", 300))
        self.gamma = float(self.cfg.get("rate_gamma", 0.35))
        self._ready = False

    # ------------------------------------------------------------------ the circuit, read once
    def _setup(self, H):
        import torch
        sets = (getattr(self.sim, "sets", None) or {}) if self.sim is not None else {}
        nset = self.cfg.get("neurons") or next((n for n, lv in H.levels.items()
                                                if "voltage" in getattr(lv, "state_schema", {})), None)
        eset = self.cfg.get("synapses") or next((n for n, lv in H.levels.items()
                                                 if getattr(lv, "pre", None) is not None
                                                 and getattr(lv, "post_name", None) == nset), None)
        if nset is None or eset is None:
            raise ValueError("neural_panel: needs a set with a `voltage` block and an edge-set onto it")
        self.nset, self.eset = nset, eset
        lv, es = H.level(nset), H.level(eset)
        N = int(lv.n)
        names = list(getattr(lv, "type_names", []) or [])
        nt = (lv.node_type.detach().cpu().numpy() if getattr(lv, "node_type", None) is not None
              else np.zeros(N, np.int64))
        # ROWS BY TYPE, in the spec's declared order, stably (a parent's neurons stay together)
        self.order = np.argsort(nt, kind="stable")
        self.rank = np.empty(N, np.int64); self.rank[self.order] = np.arange(N)
        self.N = N
        types = (sets.get(nset) or {}).get("types") or {}
        sign_of = {}
        for i, nm in enumerate(names):
            s = str((types.get(nm) or {}).get("sign", "") or "").upper()
            sign_of[i] = (1.0 if s.startswith("E") else -1.0 if s.startswith("I") else 0.0)
        self.dale = any(v != 0 for v in sign_of.values())
        pre = es.pre.detach().cpu().numpy(); post = es.post.detach().cpu().numpy()
        w = es.get("w").detach().cpu().numpy().reshape(-1) if "w" in es.state_schema else np.ones(len(pre), np.float32)
        occ = es.occ.detach().cpu().numpy() > 0
        pre, post, w = pre[occ], post[occ], w[occ]
        self.pre, self.post, self.w = pre, post, w
        # the dense post x pre matrix, in row order; the column's sign = the pre neuron's Dale sign
        self.W = np.zeros((N, N), np.float32)
        self.W[self.rank[post], self.rank[pre]] = w
        col_sign = np.array([sign_of.get(int(t), 0.0) for t in nt], np.float32)[self.order]
        if not self.dale:
            col_sign = np.sign(self.W).sum(0); col_sign = np.where(col_sign >= 0, 1.0, -1.0).astype(np.float32)
        self.col_sign = col_sign
        # the type bands, on the sorted rows
        self.blocks = []
        srt = nt[self.order]
        for i, nm in enumerate(names):
            rows = np.where(srt == i)[0]
            if rows.size:
                self.blocks.append((nm, int(rows.min()), int(rows.max()) + 1))
        if not self.blocks:
            self.blocks = [(nset, 0, N)]
        _in = self.cfg.get("input")
        self.in_types = [names.index(t) for t in (_in if isinstance(_in, list) else [_in] if _in else []) if t in names]
        _out = self.cfg.get("output")
        outs = _out if isinstance(_out, list) else ([_out] if _out else names)
        self.out_groups = [(t, np.where(nt == names.index(t))[0]) for t in outs if t in names]
        self.has_omega = "omega" in lv.state_schema
        # THE INPUT COLUMN IS WHAT ARRIVES FROM OUTSIDE: the drive field sampled at each neuron
        # (`panel.input_field`, default the first field), which is what `neuron_drive` and
        # `neuron_field_input` both read; the `omega` block alone would be blank on a spec that
        # drives its afferents with a current and never writes the modulation.
        _f = self.cfg.get("input_field") or next(iter(H.fields.keys()), None)
        self.input_field = _f if _f in getattr(H, "fields", {}) else None
        self._ready = True

    def _read(self, H):
        lv = H.level(self.nset)
        v = lv.get("voltage").detach().float().cpu().numpy().reshape(-1)
        r = np.tanh(v)
        om = None
        if self.input_field is not None:
            try:
                om = H.field(self.input_field).sample(lv.get("pos"), channel=0).detach().float().cpu().numpy().reshape(-1)
            except Exception:                                # noqa: BLE001 -- a field with no sampler
                om = None
        if om is None:
            om = (lv.get("omega").detach().float().cpu().numpy().reshape(-1) if self.has_omega else np.zeros_like(r))
        return r, om

    # ------------------------------------------------------------------ the figure
    def _build(self):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        self.cmap = _ei_cmap()
        N = self.N
        fig = plt.figure(figsize=(14.0, 8.6), facecolor=BG, dpi=100)
        gs = GridSpec(2, 1, height_ratios=[1.0, 0.58], hspace=0.10, left=0.06, right=0.985, top=0.95, bottom=0.07)
        ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        mid_l, mid_r, lo, hi = 0.30, 0.72, 0.06, 0.96
        im_rec = ax.imshow(np.zeros((N, N)), cmap=self.cmap, vmin=-1, vmax=1, extent=(mid_l, mid_r, lo, hi),
                           aspect="auto", zorder=3, interpolation="nearest")
        ax.add_patch(plt.Rectangle((mid_l, lo), mid_r - mid_l, hi - lo, fill=False, ec="#777", lw=0.6, zorder=5))
        for i, (nm, a0, a1) in enumerate(self.blocks):
            y = hi - (hi - lo) * a0 / N; x = mid_l + (mid_r - mid_l) * a0 / N
            if i:
                ax.plot([mid_l, mid_r], [y, y], color="#777", lw=0.4, alpha=0.5, zorder=4)
                ax.plot([x, x], [lo, hi], color="#777", lw=0.4, alpha=0.5, zorder=4)
            yc = hi - (hi - lo) * 0.5 * (a0 + a1) / N
            ax.text(mid_l - 0.006, yc, f"{nm} ({a1 - a0})", color="#ccc", fontsize=FS_TICK, ha="right", va="center")
        ax.text(0.5, lo - 0.02, "presynaptic", color="#ddd", fontsize=FS_AXIS, ha="center", va="top")
        ax.text(0.135, 0.5 * (lo + hi), "postsynaptic", color="#ddd", fontsize=FS_AXIS, ha="center", va="center", rotation=90)

        def _col(x0, x1, k):
            im = ax.imshow(np.zeros((k, 1)), cmap=self.cmap, vmin=-1, vmax=1, extent=(x0, x1, lo, hi),
                           aspect="auto", zorder=3, interpolation="nearest")
            ax.add_patch(plt.Rectangle((x0, lo), x1 - x0, hi - lo, fill=False, ec="#777", lw=0.5, zorder=5))
            return im
        im_in = _col(0.075, 0.105, N)
        ax.text(0.09, hi + 0.012, "input", color="#fff", fontsize=FS_NOTE, ha="center", va="bottom")
        im_rate = _col(0.735, 0.765, N)
        ax.text(0.75, hi + 0.012, "r = tanh v", color="#fff", fontsize=FS_NOTE, ha="center", va="bottom")
        k_out = max(1, len(self.out_groups))
        oy0, oy1 = 0.5 * (lo + hi) - 0.03 * k_out, 0.5 * (lo + hi) + 0.03 * k_out
        im_out = ax.imshow(np.zeros((k_out, 1)), cmap=self.cmap, vmin=-1, vmax=1, extent=(0.86, 0.90, oy0, oy1),
                           aspect="auto", zorder=3, interpolation="nearest")
        ax.add_patch(plt.Rectangle((0.86, oy0), 0.04, oy1 - oy0, fill=False, ec="#777", lw=0.5, zorder=5))
        ax.text(0.88, oy1 + 0.012, "output", color="#fff", fontsize=FS_NOTE, ha="center", va="bottom")
        for k, (nm, _) in enumerate(self.out_groups):
            ax.text(0.905, oy1 - (oy1 - oy0) * (k + 0.5) / k_out, nm, color="#fff", fontsize=FS_NOTE, ha="left", va="center")
        for x0, x1, yy in ((0.11, 0.128, 0.5 * (lo + hi)), (mid_r + 0.004, 0.73, 0.5 * (lo + hi)), (0.77, 0.855, 0.5 * (lo + hi))):
            ax.annotate("", xy=(x1, yy), xytext=(x0, yy), arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=1.0))
        self.txt = ax.text(0.0, 1.0, "", transform=ax.transAxes, color="#fff", fontsize=FS_NOTE, ha="left", va="bottom")
        ax_k = fig.add_subplot(gs[1]); ax_k.set_facecolor(BG)
        im_kino = ax_k.imshow(np.full((N, self.kino_w), np.nan, np.float32), cmap=self.cmap, vmin=-1, vmax=1,
                              aspect="auto", interpolation="nearest", origin="upper")
        ax_k.set_yticks([]); ax_k.set_xticks([])
        for sp_ in ax_k.spines.values():
            sp_.set_color("#777"); sp_.set_linewidth(0.5)
        for i, (nm, a0, a1) in enumerate(self.blocks):
            if a0:
                ax_k.axhline(a0 - 0.5, color="#777", lw=0.5, alpha=0.6)
            ax_k.text(-0.004, 1.0 - (a0 + a1) / (2 * N), nm, transform=ax_k.transAxes, color="#ccc",
                      fontsize=FS_TICK, ha="right", va="center")
        ax_k.set_xlabel(f"time  (last {self.kino_w} rendered frames)", color="#ddd", fontsize=FS_AXIS, labelpad=4)
        self.fig, self.art = fig, dict(im_rec=im_rec, im_in=im_in, im_rate=im_rate, im_out=im_out, im_kino=im_kino)
        fig.canvas.draw()

    # ------------------------------------------------------------------ the LUTs
    def _rate_disp(self, r):
        lim = self.rate_lim or 1.0
        return np.sign(r) * np.clip(np.abs(r) / lim, 0.0, 1.0) ** self.gamma

    def _update_limits(self, r):
        nz = np.abs(r[np.abs(r) > 0])
        if nz.size:
            p = float(np.percentile(nz, 90.0))
            self.rate_lim = max(self.rate_lim or 0.0, p) if not self.cfg.get("rate_lim") else self.rate_lim
        mag = (np.abs(self.W) * np.abs(r[self.order])[None, :]).reshape(-1)
        mag = mag[mag > 0]
        if mag.size and not self.cfg.get("msg_lim"):
            self.msg_lim = max(self.msg_lim or 0.0, float(np.percentile(mag, 99.5)))

    def _draw(self, idx: int):
        tick, r, om = self.hist[idx]
        rs = r[self.order]
        knee = np.log1p(9.0)
        mag = np.log1p(9.0 * np.abs(self.W) * np.abs(rs)[None, :] / max(self.msg_lim or 1.0, 1e-12)) / knee
        try:
            from scipy.ndimage import grey_dilation
            if self.N > 120:
                mag = grey_dilation(mag, size=(2, 2))
        except Exception:                                    # noqa: BLE001
            pass
        self.art["im_rec"].set_data(np.clip(mag, 0.0, 1.0) * self.col_sign[None, :])
        vin = om[self.order].astype(np.float32)
        if self.in_types:
            nt_sorted = np.zeros(self.N, bool)
            for nm, a0, a1 in self.blocks:
                if nm in [self._names()[t] for t in self.in_types]:
                    nt_sorted[a0:a1] = True
            vin = np.where(nt_sorted, vin, np.nan)
        lim_in = float(np.nanmax(np.abs(vin))) if np.isfinite(vin).any() else 1.0
        self.art["im_in"].set_data((vin / max(lim_in, 1e-9))[:, None])
        self.art["im_rate"].set_data(self._rate_disp(rs)[:, None])
        outs = np.array([[float(np.mean(r[g])) if g.size else 0.0] for _, g in self.out_groups] or [[0.0]], np.float32)
        self.art["im_out"].set_data(self._rate_disp(outs))
        lo_i = max(0, idx + 1 - self.kino_w)
        buf = np.full((self.N, self.kino_w), np.nan, np.float32)
        seg = np.stack([self._rate_disp(h[1][self.order]) for h in self.hist[lo_i:idx + 1]], 1)
        buf[:, :seg.shape[1]] = seg if idx + 1 < self.kino_w else seg
        if idx + 1 >= self.kino_w:
            buf = seg[:, -self.kino_w:]
        self.art["im_kino"].set_data(buf)
        clk = f"   t = {tick * float(self.dt) * float(self.time_s):.4g} s" if (self.dt and self.time_s) else ""
        self.txt.set_text(f"{self.name}   {self.N} neurons, {self.pre.size:,} synapses"
                          f"{' (Dale: blue E, red I)' if self.dale else ' (colour = sign of W)'}   "
                          f"frame {tick}/{self.n_frames}{clk}   brightness = |W r| up to {self.msg_lim or 0:.3g}")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def _names(self):
        return [b[0] for b in self.blocks] if not self.sim else list(((self.sim.sets or {}).get(self.nset) or {}).get("types") or {}) or [b[0] for b in self.blocks]

    # ------------------------------------------------------------------ the hook
    def capture(self, H, tick: int) -> int:
        """Record this tick's rates (every tick is cheap; the picture is drawn on demand)."""
        if not self._ready:
            self._setup(H)
        r, om = self._read(H)
        self._update_limits(r)
        self.hist.append((int(tick), r.astype(np.float32), om.astype(np.float32)))
        return len(self.hist) - 1

    def frame_at(self, idx: int):
        if self.fig is None:
            self._build()
        idx = max(0, min(int(idx), len(self.hist) - 1))
        return self._draw(idx)

    def __call__(self, H, tick: int):
        try:
            if self.t0 is None:
                self.t0 = time.perf_counter()
            if tick % self.stride and tick != self.n_frames:
                return
            idx = self.capture(H, tick)
            if tick == 0:
                return
            img = self.frame_at(idx)
            if self.writer is None:
                import imageio.v2 as iio
                self.writer = iio.get_writer(self.out, fps=self.fps, codec="libx264", quality=8, macro_block_size=1)
                print(f"[neural-panel] {self.out}   {self.N} neurons, {self.pre.size:,} synapses, "
                      f"{max(1, self.n_frames // self.stride)} frames (every {self.stride}) at {self.fps:g} fps, "
                      f"{len(self.still_ticks)} stills + 3d.png", flush=True)
            self.writer.append_data(img)
            self.rendered += 1; self.drawn = self.N
            if tick in self.still_ticks:
                self._still(tick, img)
        except Exception as e:                               # noqa: BLE001 -- a picture must never end a run
            self.failed = f"{type(e).__name__}: {e}"
            print(f"[neural-panel] DISABLED after frame {tick}: {self.failed}", flush=True)
            self.n_frames = -1

    def _still(self, tick, img):
        try:
            import imageio.v3 as iio
            _p = os.path.join(self.still_dir, f"still_{self.stills_written:02d}_f{tick:05d}.png")
            iio.imwrite(_p, img); iio.imwrite(os.path.join(self.still_dir, "3d.png"), img)
            self._still_paths.append(_p); self.stills_written += 1
        except Exception as e:                               # noqa: BLE001
            print(f"[neural-panel] still at frame {tick} failed: {type(e).__name__}: {e}", flush=True)

    def close(self):
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:                                # noqa: BLE001
                pass
            self.writer = None
        if not self.keep_stills:
            for _p in self._still_paths:
                try:
                    os.remove(_p)
                except OSError:
                    pass
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig); self.fig = None
        if self.rendered:
            print(f"[neural-panel] wrote {self.out}: {self.rendered} frames", flush=True)
