#!/usr/bin/env python
"""Fast coupling-parameter tuning + CONTINUOUS ARCHIVING for the embryogenesis
(active-matter x MPM) prototypes. Every run -- success OR failure -- is archived under
`archive/<name>/` (evolution + final PNGs, an overlay "cells-in-a-blob" PNG + mp4, the
metrics.json and the exact spec used) and appended to TESTS.md, so no experiment is lost.

    python prototype/embryogenesis/tune.py <spec.yaml> frames=150 tag=v3 mp4=1 \
        p2g.drag=0.4 mpm_grid_update.surface_tension=160 mpm_spin.omega=0.8 \
        agent_to_mpm.agent_mass=2e-5 mpm_to_agent.confine=140 mpm_to_agent.k=0.8 \
        agent.move_speed=0.5 polar_align.gamma=80 polar_align.noise=2.5 n_grid=64
"""
import os, sys, time, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "active_matter2"))
import numpy as np
import torch
import plexus.operators   # noqa
import am2_ops            # noqa
import plexus.schema as S
from plexus.generators.graph_data_generator import data_generate
from plexus.paths import set_data_root, graphs_data_path
from plexus import plot
from embryo_render import render_blob            # cells-in-a-blob overlay (dots on water)

set_data_root(os.path.join(HERE, "data"))
PRE = "embryogenesis"
ARCHIVE = os.path.join(HERE, "archive")
TESTS_MD = os.path.join(HERE, "TESTS.md")


def _num(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _apply(sim, key, val):
    val = _num(val)
    if key == "n_grid":
        sim.fields["mpm_grid"]["n_grid"] = val
    elif key == "agent.move_speed":
        for t in sim.sets["agent"]["types"].values():
            t["move_speed"] = val
    elif key == "agent.div_rate":
        for t in sim.sets["agent"]["types"].values():
            t["div_rate"] = val
    elif key == "agent.p":                                 # e.g. agent.p=0,1,10,1.6
        pl = [float(x) for x in str(val).split(",")]
        for t in sim.sets["agent"]["types"].values():
            t["p"] = pl
    elif key == "spawn_radius":
        sim.sets["agent"]["spawn_radius"] = val
    elif key == "agent.n":
        sim.sets["agent"]["n"] = val
    elif key == "per_parent":
        sim.sets["mpm_particle"]["per_parent"] = val
    elif key == "cell.youngs":                              # set youngs on every type + every layer
        for t in sim.sets["cell"]["types"].values():
            t["youngs"] = val
            for L in t.get("layers", []):
                L["youngs"] = val
    elif "." in key:
        opname, param = key.split(".", 1)
        for o in sim.operators:
            if o.op == opname:
                o.params[param] = val
    else:
        print(f"[tune] unknown override {key!r}", flush=True)


def metrics(sim):
    d = graphs_data_path(PRE, sim.name)
    tr = np.load(os.path.join(d, "trajectory.npz"))
    ax, mx = tr["agent__pos"], tr["mpm_particle__pos"]      # [T,N,2]
    occ = tr["agent__occ"] if "agent__occ" in tr.files else np.ones(ax.shape[:2], bool)
    c = np.array([0.5, 0.5])
    m_r = np.linalg.norm(mx - c, axis=-1)
    mR0, mR1 = np.quantile(m_r[0], 0.98), np.quantile(m_r[-1], 0.98)
    liveN = int((occ[-1] > 0).sum())
    a_last = ax[-1][occ[-1] > 0]                            # only live cells
    a_r = np.linalg.norm(a_last - c, axis=-1)
    escaped = float((a_r > mR1 + 0.02).mean()) if liveN else 0.0

    def aniso(pos):                                        # material roundness (0 = perfect disc)
        rel = pos - c; ang = np.arctan2(rel[:, 1], rel[:, 0]); rr = np.linalg.norm(rel, axis=-1)
        bins = np.linspace(-np.pi, np.pi, 25); idx = np.digitize(ang, bins)
        outer = np.array([rr[idx == b].max() if (idx == b).any() else np.nan for b in range(1, 25)])
        outer = outer[~np.isnan(outer)]
        return float(outer.std() / max(outer.mean(), 1e-6))

    lm = occ[-1] > 0                                        # live mask (last frame) for polar order
    v = np.diff(ax[-21:], axis=0)[:, lm]; sp = np.linalg.norm(v, axis=-1, keepdims=True)
    vhat = v / np.clip(sp, 1e-9, None); P = float(np.linalg.norm(vhat.mean(axis=(0, 1)))) if lm.any() else 0.0
    return dict(disc_R0=round(mR0, 4), disc_R1=round(mR1, 4), disc_growth=round(mR1 - mR0, 4),
                aniso=round(aniso(mx[-1]), 4), agent_escaped=round(escaped, 4), n_cells=liveN,
                polar_order=round(P, 4), agent_Rmax_end=round(float(a_r.max()) if liveN else 0.0, 4))


def main():
    spec_path = sys.argv[1]
    ov = dict(kv.split("=", 1) for kv in sys.argv[2:] if "=" in kv)
    frames = int(ov.pop("frames", 150)); tag = ov.pop("tag", "tune"); want_mp4 = bool(int(ov.pop("mp4", "1")))
    sim = S.load(spec_path)
    sim.n_frames = frames
    base = sim.name
    sim.name = f"{base}_{tag}"
    for k, v in ov.items():
        _apply(sim, k, v)
    print(f"[tune] {sim.name}: frames={frames} mp4={want_mp4} overrides={ov}", flush=True)

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    data_generate(sim, PRE, device=dev, erase=True)
    plot.plot_dataset(sim, PRE, movie=False)              # cheap evolution/final PNGs
    d = graphs_data_path(PRE, sim.name)
    render_blob(sim, d, movie=want_mp4)                   # the "cells-in-a-blob" overlay (PNG + mp4)
    dt = time.time() - t0
    m = metrics(sim); m["seconds"] = round(dt, 1); m["frames"] = frames
    print(f"[tune] {sim.name}: {dt:.1f}s  " + "  ".join(f"{k}={v}" for k, v in m.items()), flush=True)

    # --- archive EVERYTHING (even failures) ---
    adir = os.path.join(ARCHIVE, sim.name); os.makedirs(adir, exist_ok=True)
    for f in ("fig_agent_evolution.png", "fig_mpm_particle_evolution.png",
              "fig_agent_final.png", "fig_mpm_particle_final.png",
              "blob_evolution.png", "blob.mp4"):
        src = os.path.join(d, f)
        if os.path.isfile(src):
            shutil.copy2(src, adir)
    with open(os.path.join(adir, "metrics.json"), "w") as fh:
        json.dump({"name": sim.name, "base": base, "tag": tag, "overrides": ov, **m}, fh, indent=2)
    if os.path.isfile(os.path.join(d, "spec.yaml")):
        shutil.copy2(os.path.join(d, "spec.yaml"), os.path.join(adir, "spec_used.yaml"))

    new = not os.path.isfile(TESTS_MD)
    with open(TESTS_MD, "a") as fh:
        if new:
            fh.write("# Embryogenesis (active-matter x MPM) -- test log\n\n"
                     "Chronological log of every tuning run (successes AND failures). Media in "
                     "`archive/<name>/` (blob_evolution.png, blob.mp4, fig_*_evolution.png).\n\n"
                     "| name | escaped | disc_growth | aniso | polar | s | key overrides |\n"
                     "|---|---|---|---|---|---|---|\n")
        ov_s = " ".join(f"{k}={v}" for k, v in ov.items())
        fh.write(f"| {sim.name} | {m['agent_escaped']} | {m['disc_growth']} | {m['aniso']} | "
                 f"{m['polar_order']} | {m['seconds']} | {ov_s} |\n")
    print(f"[tune] archived -> {adir}", flush=True)


if __name__ == "__main__":
    main()
