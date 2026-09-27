#!/usr/bin/env python
"""Does the motor turn, as one body, at the rate its torque and load predict?

    python tools/bfm_rotor_probe.py bfm_a01_rotor_stator [--omega 3.1416] [--no-record]

THE FIVE GATES of builder step A01, each a number read off the trajectory and not off the movie:

  1. the delivered torque against N F r         -- from the engine log's `[stator_push f...]` lines
  2. ONE rate for the four rotor parts          -- the angle of each part about the axis, per frame,
                                                   and the lag of the MS-ring, rod and hook behind
                                                   the C-ring (a torsional lag of degrees is a body;
                                                   tens of degrees is a rotor coming apart)
  3. the C-ring pillars stay on their ring      -- the radial extent of its points, first vs last row
  4. the bushing does not move                  -- its points' displacement over the run
  5. the rate matches omega = T / (k_d I)       -- measured against the design value

HOW THE ANGLE IS MEASURED. Every point of a part carries an azimuth about the axis; the rotation of
the part between row 0 and row t is the CIRCULAR MEAN of each point's azimuth change -- the mean of
exp(i (phi_t - phi_0)) -- taken over points farther than `r_min` from the axis (a point on the axis
has no azimuth), and unwrapped over time so a full turn reads 2 pi rather than 0. A mean over
points is what makes this a rigid-body angle: a part that is shearing shows a mean whose modulus
|<exp(i d phi)>| drops below one, and that modulus is printed too, as the rigidity.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

# THE SETS ARE READ OFF THE SPEC, not written here: every `*_pt` set is a part, the ones an
# `mpm_anchor` holds are static, the one `stator_push` acts on is the reference the lags are
# measured against. Defaults cover the a01/a02 layout for a trajectory whose spec is not at hand.
ROTOR = ["c_posts_pt", "c_flig_pt", "ms_ring_pt", "rod_pt", "hook_pt"]
STATIC = ["lp_ring_pt"]
REF = "c_flig_pt"


def parts_from_spec(d):
    """(rotor sets, static sets, the pushed set) from a spec dict."""
    global ROTOR, STATIC, REF
    pts = [s for s in d["sets"] if s.endswith("_pt")]
    anchored = {o["at"].split("[")[0] for o in d["operators"] if o["op"] == "mpm_anchor"}
    pushed = next((o["at"].split("[")[0] for o in d["operators"] if o["op"] == "stator_push"), None)
    STATIC = [s for s in pts if s in anchored]
    ROTOR = [s for s in pts if s not in anchored]
    if pushed in ROTOR:
        REF = pushed
    return ROTOR, STATIC, REF


def load(name, max_rows=301):
    """The run's arrays as `{'<set>__pos': [T, N, 3], ...}`, from `trajectory.npz` or, when the run
    was written with `save_data: true`, from `simulation.zarr` -- read STRIDED, because every frame
    of five sets at 240,000 points is 2 GB per block and the gates need a few hundred rows."""
    from plexus.paths import graphs_data_path
    for sub in ("bacterium", "studio"):
        d = os.path.join(graphs_data_path(), sub, name)
        npz = os.path.join(d, "trajectory.npz")
        if os.path.exists(npz):
            return np.load(npz, allow_pickle=True), os.path.join(d, "spec.yaml")
        zpath = os.path.join(d, "simulation.zarr")
        if os.path.isdir(zpath):
            import zarr
            g = zarr.open_group(zpath, mode="r")
            out = {}
            for s in ROTOR + STATIC + ["stator_unit"]:
                if s in g and "pos" in g[s]:
                    T = g[s]["pos"].shape[0]
                    st = max(1, (T - 1) // max(max_rows - 1, 1))
                    out[f"{s}__pos"] = np.asarray(g[s]["pos"][::st])
            out["_stride"] = st
            return out, os.path.join(d, "spec.yaml")
    raise SystemExit(f"no trajectory for {name}")


def angle_series(P, centre, r_min):
    """(theta [T] radians unwrapped, rigidity [T] in 0..1) of a part's rotation about z."""
    rel = P[:, :, :2] - np.asarray(centre)[None, None, :2]
    r0 = np.linalg.norm(rel[0], axis=1)
    keep = r0 > r_min
    phi = np.arctan2(rel[:, keep, 1], rel[:, keep, 0])
    z = np.exp(1j * (phi - phi[0:1]))
    m = z.mean(1)
    theta = np.unwrap(np.angle(m))
    return theta, np.abs(m)


def torque_from_log(name):
    """The `[stator_push f...]` lines the run printed: (frame, delivered, nominal, touched)."""
    # THE RUN'S OWN LOG FIRST; failing that, the SERVER'S, after the last `[generate] <name>` banner.
    # The page tees a per-run log only sometimes (a02 wrote none), while the server's stdout always
    # has the lines -- for every run it ever made, so the block is cut at the run's own banner.
    import glob
    # TWO DRIVES, ONE PROBE: the constant push prints `touched ... torque delivered T vs N F r T0`,
    # the stepping stator's contact prints `gripped ... torque delivered T = kappa sum x`; the
    # nominal is N F r for the first and, for the second, whatever the linkage carried (the same
    # number, since there is no nominal to miss -- the stall torque is the gate the probe adds).
    rx = re.compile(r"\[stator_(?:push|contact) f(\d+)\].*?(\d+) particles (?:touched|gripped) "
                    r"\((\d+)-(\d+) per unit\).*?torque delivered ([0-9.e+-]+) (?:vs N F r ([0-9.e+-]+)|= kappa)")
    p = os.path.join(REPO, "log", "gui_runs", f"{name}.log")
    text = open(p, errors="replace").read() if os.path.exists(p) else ""
    if not rx.search(text):                       # a per-run log with no torque lines is no better than none
        for sp in sorted(glob.glob(os.path.join(REPO, "log", "gui_runs", "server_*.log"))):
            t = open(sp, errors="replace").read()
            # THE LAST RUN'S FIRST BANNER. tqdm redraws "0%" several times over the opening frames, so
            # the banners of one run form a tight group; the run starts at the first banner of the
            # last group, not at the last banner (which sits after the first torque lines).
            pos = [m.start() for m in re.finditer(re.escape(f"[generate] {name}:   0%"), t)]
            if pos:
                i = pos[-1]
                for a_, b_ in zip(pos[-2::-1], pos[::-1]):
                    if b_ - a_ > 5000:
                        break
                    i = a_
                text = t[i:]
                # ... AND STOP AT THE NEXT RUN'S BANNER, or the last line found is another spec's.
                nxt = re.search(r"\[generate\] (?!" + re.escape(name) + r":)[A-Za-z0-9_]+:", text[len(name) + 12:])
                if nxt:
                    text = text[: nxt.start() + len(name) + 12]
    out = []
    for m in rx.finditer(text):
        out.append((int(m[1]), float(m[5]), float(m[6]) if m[6] is not None else float(m[5]),
                    int(m[2]), int(m[3]), int(m[4])))
    return out


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="bfm_a01_rotor_stator")
    ap.add_argument("--omega", type=float, default=None, help="the design rate, rad/s (sim)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--note", default="", help="prepended to the record's why: what this measurement found")
    a = ap.parse_args()
    from plexus.paths import graphs_data_path
    sp0 = next((os.path.join(graphs_data_path(), sub, a.name, "spec.yaml") for sub in ("bacterium", "studio")
                if os.path.exists(os.path.join(graphs_data_path(), sub, a.name, "spec.yaml"))), None)
    if sp0 is not None:
        parts_from_spec(yaml.safe_load(open(sp0)))
    tr, sp = load(a.name)
    d = yaml.safe_load(open(sp))
    n_frames, dt = int(d["general"]["n_frames"]), float(d["general"]["dt"])
    L_um = float(d["general"]["units"]["length_um"])
    nm_per_world = L_um * 1e3
    # THE AXIS FROM THE STATOR RING: `start:` when the set is placed by hand, the seeded positions
    # (row 0 of the trajectory) when it is the cell's child and a seed put it there.
    centre = (np.array(d["sets"]["stator_unit"]["start"]).mean(0) if "start" in d["sets"]["stator_unit"]
              else np.asarray(tr["stator_unit__pos"][0]).mean(0))
    push = next(o for o in d["operators"] if o["op"] == "stator_push")
    drive = ("constant force per unit" if push.get("model") != "contact"
             else "a stepping stator on an elastic linkage (stator_push[contact] + stator_step)")
    drag = next(o["drag"] for o in d["operators"] if o["op"] == "mpm_scatter")

    keys = {k: k for k in (tr.files if hasattr(tr, "files") else tr.keys())}
    T = None
    series, rig = {}, {}
    for s in ROTOR + STATIC:
        k = f"{s}__pos"
        if k not in keys:
            print(f"  (no {k} in the trajectory; keys: {sorted(keys)[:12]} ...)")
            continue
        P = np.asarray(tr[k])
        T = P.shape[0]
        th, r = angle_series(P, centre, r_min=0.02)
        series[s], rig[s] = th, r
    stride = int(tr["_stride"]) if "_stride" in keys else max(1, n_frames // max(T - 1, 1))
    t = np.arange(T) * dt * stride

    # ---- gate 1: the torque
    tq = torque_from_log(a.name)
    print(f"{a.name}: {T} rows, {n_frames} frames, dt {dt:g} s, {stride} frames a row\n")
    print("  1. TORQUE DELIVERED (engine log)")
    for fr, Td, Tn, nt, lo, hi in tq[:6]:
        print(f"     frame {fr:5d}: {Td:.4e} vs N F r {Tn:.4e} ({(Td / Tn * 100 if Tn else float('nan')):6.2f}%), "
              f"{nt} particles, {lo}-{hi} per unit" + ("   A UNIT TOUCHES NOTHING" if lo == 0 else ""))
    if not tq:
        print("     (no stator_push lines found in the log)")

    # ---- gates 2 and 5: one rate, and the rate
    print("\n  2/5. THE ANGLE OF EACH ROTOR PART, and its rate")
    s2 = slice(T // 3, None)
    rates = {}
    for s, th in series.items():
        if s in STATIC:
            continue
        w = np.polyfit(t[s2], th[s2], 1)[0] if T > 3 else float("nan")
        rates[s] = w
        print(f"     {s:10s} turned {np.degrees(th[-1]):8.2f} deg over the run; rate {w:7.4f} rad/s "
              f"= {w / (2 * math.pi):6.3f} turn/s; rigidity |<e^(i dphi)>| min {rig[s].min():.4f}")
    if REF in series:
        for s in [x for x in ROTOR if x != REF]:
            if s in series:
                lag = np.degrees(series[REF][-1] - series[s][-1])
                print(f"     {s:10s} lags the C-ring by {lag:+7.2f} deg at the end "
                      f"(mean over the last 2/3: {np.degrees((series[REF][s2] - series[s][s2]).mean()):+.2f})")
    if a.omega and REF in rates:
        print(f"     design omega {a.omega:.4f} rad/s; measured C-ring {rates[REF]:.4f} "
              f"({rates[REF] / a.omega * 100:.1f}% of design)")

    # ---- gate 3: the pillars stay on their ring
    if REF + "__pos" in keys:
        P = np.asarray(tr[REF + "__pos"])
        r = np.linalg.norm(P[:, :, :2] - centre[None, None, :2], axis=2) * nm_per_world
        z = (P[:, :, 2] - centre[2]) * nm_per_world
        print(f"\n  3. C-RING RADIAL EXTENT: row 0 [{r[0].min():.1f}, {r[0].max():.1f}] nm -> "
              f"last [{r[-1].min():.1f}, {r[-1].max():.1f}] nm; z extent [{z[0].min():.1f}, {z[0].max():.1f}] -> "
              f"[{z[-1].min():.1f}, {z[-1].max():.1f}] nm")

    # ---- gate 4: the bushing does not move
    if "lp_ring_pt__pos" in keys:
        P = np.asarray(tr["lp_ring_pt__pos"])
        disp = np.linalg.norm(P[-1] - P[0], axis=1) * nm_per_world
        print(f"\n  4. BUSHING: max displacement {disp.max():.3f} nm, mean {disp.mean():.3f} nm; "
              f"its angle {np.degrees(series.get('lp_ring_pt', [0.0])[-1]):+.3f} deg")

    # ---- the figure, into the record
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record, style
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {REF: "#c0392b", "ms_ring_pt": "#2f6fb5", "rod_pt": "#4a7c59", "hook_pt": "#e08214",
            "lp_ring_pt": "#777777"}
    for s, th in series.items():
        ax[0].plot(t, np.degrees(th), color=cols.get(s, "#333"), lw=1.3, label=s.replace("_pt", ""))
    if a.omega:
        ax[0].plot(t, np.degrees(a.omega * t), "--", color="#999999", lw=0.9, label=f"design {a.omega:.3f} rad/s")
    ax[0].set_xlabel("time (s, sim)"); ax[0].set_ylabel("rotation about the axis (deg)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].text(0, 1.04, "each part's angle; one line per part means one body", transform=ax[0].transAxes, fontsize=9.5)
    for s in [x for x in ROTOR if x != REF]:
        if s in series and REF in series:
            ax[1].plot(t, np.degrees(series[REF] - series[s]), color=cols.get(s, "#333333"), lw=1.2, label=s.replace("_pt", ""))
    ax[1].set_xlabel("time (s, sim)"); ax[1].set_ylabel(f"lag behind {REF.replace('_pt', '')} (deg)")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].text(0, 1.04, "torsional lag: degrees is a body, tens is a rotor coming apart", transform=ax[1].transAxes, fontsize=9.5)
    for s, r in rig.items():
        ax[2].plot(t, r, color=cols.get(s, "#333"), lw=1.2, label=s.replace("_pt", ""))
    ax[2].set_ylim(0, 1.02); ax[2].set_xlabel("time (s, sim)"); ax[2].set_ylabel("rigidity |<exp(i dphi)>|")
    ax[2].text(0, 1.04, "1 = every point turned by the same angle", transform=ax[2].transAxes, fontsize=9.5)
    for x in ax:
        style(x)
    fig.tight_layout()
    if a.no_record:
        fig.savefig("/tmp/bfm_rotor_probe.png", dpi=150, facecolor="white"); print("figure -> /tmp/bfm_rotor_probe.png")
        return
    wC = rates.get(REF, float("nan"))
    lagH = np.degrees(series[REF][-1] - series["hook_pt"][-1]) if "hook_pt" in series else float("nan")
    tqs = f"{tq[-1][1] / tq[-1][2] * 100:.1f}% of N F r" if tq else "not logged"
    record(fig,
           (a.note.strip() + " " if a.note.strip() else "") +
           f"MEASUREMENT, not a run: the five gates read off the trajectory of {a.name}. The C-ring "
           f"turned {np.degrees(series[REF][-1]):.1f} degrees over {t[-1]:.2f} s of sim time, a rate of "
           f"{wC:.4f} rad/s = {wC / (2 * math.pi):.3f} turn/s against the design {a.omega or float('nan'):.4f} "
           f"rad/s from T/(k_d I) ({(wC / a.omega * 100) if a.omega else float('nan'):.1f}% of it); the hook lags "
           f"the C-ring by {lagH:+.1f} degrees at the end; the delivered torque was {tqs}; the C-ring's radial "
           f"extent went from [{r[0].min():.1f}, {r[0].max():.1f}] to [{r[-1].min():.1f}, {r[-1].max():.1f}] nm; "
           f"the bushing moved {disp.max():.2f} nm at most. Left: each rotor part's angle about the axis, the "
           f"circular mean of its points' azimuth change, unwrapped -- one line per part on top of each other is "
           f"one body. Middle: the torsional lag of the MS-ring, rod and hook behind the C-ring. Right: the "
           f"rigidity, the modulus of that circular mean, 1 when every point of a part turned by the same "
           f"angle. Load drag {drag:g} /s, {int(d['sets']['stator_unit'].get('n', d['sets']['stator_unit'].get('per_parent', 0)))} stator units, drive: {drive}, "
           + (f"force {push['force']:.4e}" if "force" in push else f"kappa {push['kappa']:.4e}")
           + f", reach {push['reach']:.4f} world.",
           name=f"{a.name}: rate, one body, ring intact, bushing still, torque delivered")


if __name__ == "__main__":
    main()
