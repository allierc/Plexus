#!/usr/bin/env python
"""One line per cilium run: does it pass the gates, and what did it do to the water?

    python tools/cilia_row.py p1_lam0p2 p1_lam0p4 ...

THE GATE COLUMNS COME FIRST, and that ordering is the point. A run that moves a lot of water with
its filament coiled, outside the box, or beating at the wrong frequency has not found a better
cilium -- it has found a worse bug, and a table sorted by "water moved" would rank it first. So
`box`, `strain`, `arc` and `Hz` are read before `water`, and any of them out of range makes the
run a failure however good the last column looks.

    box     max |pos| over the run; the world is 0..1, so anything above ~1.05 has left it
    arc%    total arc length against the 20 um rest; away from 100 means stretching or coiling
    strain  the worst SEGMENT, which separates the two: a coiled rod holds its segments
    base    beat amplitude at the base joint, degrees peak to peak. Baseline 28.3
    tip     the same at the tip. A DISTRIBUTED drive makes this GROW along the rod, so tip/base
            above 1 is correct here -- the 0.3-0.7 band belongs to a basally driven filament
            whose wave decays, which is what Machin 1958 showed and what rod_motor replaced
    Hz      measured against commanded; disagreement means the rod is not following its drive
    oop     the worst rod's tilt OUT of its own stroke plane at the base segment, degrees, mean
            over the last third. Run 0451 (five cilia, vacuum) passed nothing but this was the
            cause: the base clamp restored only IN-plane tilt, the rods left their planes at
            azimuth -90 degrees and froze at 78. A beating cilium stays within a few degrees
    base_r  distance of each base from the cell it is held to, um, against the sphere radius
            the spec laid it on. Run 0451 again: five bases seeded 2 um from the centre were
            pulled onto it (0.03 um) in 0.01 s, and every earlier multi-cilia picture showed
            filaments fanning out of one point INSIDE the cell
    water   99.9th percentile speed, um/s, and the maximum displacement in nm. One MPM grid cell
            is 1,560 nm and a real Platynereis cilium drives about 1,592 nm, so displacement is
            the column that says whether the motion is visible or merely present
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))


def row(nm):
    import yaml
    from rod_probe import load, angles
    tr, sp = load(nm)
    d = yaml.safe_load(open(sp))
    R = np.asarray(tr["rod_node__pos"])
    T, N, _ = R.shape
    L0 = float(next(o for o in d["seed"] if o["op"] == "rod_seed")["length"])
    um = float((d["general"].get("units") or {}).get("length_um", 50.0))
    # SEGMENTS COME FROM THE TABLE, NOT FROM ADJACENT ROWS. With more than one filament in the
    # set, `np.diff(R, axis=1)` walks straight off the end of rod 1 and onto the base of rod 2,
    # counting the gap between them as a segment: measured on the two-cilium run, that reported
    # 299.7 percent of arc length and 1,396 percent strain for a filament that was in fact intact.
    # The trajectory records `rod_node__mesh_E_srce`/`E_trgt`, so the same table the force laws
    # read is available here -- and using it is the only way this tool can measure an axoneme.
    n_rod = max(1, int(next((o for o in d["seed"] if o["op"] == "rod_seed"), {}).get("n_rod", 1)))
    per = N // n_rod
    rest = L0 / (per - 1)
    if "rod_node__mesh_E_srce" in tr.files and np.asarray(tr["rod_node__mesh_E_srce"]).size:
        # ONE FRAME'S WORTH. The recorder writes the table EVERY frame and flattens it, so
        # `E_srce` comes back as 15,001 x 14 = 210,014 entries for a two-rod run. Taking the lot
        # indexed 210,014 "segments" into a 16-node rod and reported 1.46e6 percent of arc length.
        # The topology is fixed, so the first frame's copy is the table.
        _es = np.asarray(tr["rod_node__mesh_E_srce"]).astype(int).ravel()
        _et = np.asarray(tr["rod_node__mesh_E_trgt"]).astype(int).ravel()
        n_seg = max(1, _es.size // T)
        ei, ej = _es[:n_seg], _et[:n_seg]
        seg = np.linalg.norm(R[:, ej, :] - R[:, ei, :], axis=2)
        arc_pct = 100.0 * float(seg[-1].sum()) / (L0 * n_rod)
    else:
        seg = np.linalg.norm(np.diff(R, axis=1), axis=2)
        arc_pct = 100.0 * float(seg[-1].sum()) / L0
    s = slice(T // 3, None)
    # angles PER ROD, IN EACH ROD'S OWN FRAME, then the worst. Node 1 of each rod is its base
    # joint, node per-1 its tip. On a sphere every rod points along its own normal and beats in
    # its own plane, so measuring all of them against one global (z, y) frame reported the ring's
    # geometry -- rods standing at -31, -20, +21, +28 degrees -- as if it were the beat. The frames
    # come from `rod_layout`, the same function `rod_seed` laid the rods with.
    d_r, n_r = rod_frames(d)
    A = np.zeros((T, N))
    oop = np.zeros(n_rod)
    for r in range(n_rod):
        P = R[:, r * per:(r + 1) * per, :]
        A[:, r * per:(r + 1) * per] = angles(P, d_r[r], n_r[r])
        e0 = P[:, 1, :] - P[:, 0, :]
        e0 = e0 / np.linalg.norm(e0, axis=1)[:, None]
        oop[r] = float(np.degrees(np.arcsin(np.clip(np.abs(e0[s] @ n_r[r]), 0, 1))).mean())
    amp_all = np.ptp(A[s], axis=0)
    # AVERAGED OVER THE RODS. Node r*per + 1 is rod r's base joint and r*per + per - 1 its tip, so
    # with several filaments the two reported numbers are the mean base and the mean tip rather
    # than whichever rod happened to be laid first.
    base_amp = float(np.mean([amp_all[r * per + 1] for r in range(n_rod)]))
    # THE WORST ROD, not the mean, decides the flail gate: on the five-beat coupling-1.0 run the
    # mean base was 100 degrees while one rod swept 352 and two were nearly dead at 17.
    base_max = float(np.max([amp_all[r * per + 1] for r in range(n_rod)]))
    tip_amp = float(np.mean([amp_all[r * per + per - 1] for r in range(n_rod)]))
    # WHERE THE BASES ARE, against where they were laid. Only meaningful when the base is held to
    # a set and the rods stand on a sphere; a wall-anchored rod reports nothing here.
    base_r, base_ref = float("nan"), float("nan")
    cell_um, cell_cos = float("nan"), float("nan")
    seed = next((o for o in d["seed"] if o["op"] == "rod_seed"), {})
    rb = next((o for o in d["operators"] if o.get("op") == "rod_base"), {})
    anc = rb.get("anchor", "fixed")
    if anc != "fixed" and f"{anc}__pos" in tr.files and float(seed.get("sphere_radius", 0)) > 0:
        C = np.asarray(tr[f"{anc}__pos"]).reshape(T, -1, 3)[:, 0, :]
        base_r = float(np.linalg.norm(R[s, ::per, :] - C[s, None, :], axis=2).mean() * um)
        base_ref = float(seed["sphere_radius"]) * um
        # THE CELL'S OWN TRANSPORT (P4): net displacement over a whole number of beats, in um,
        # and its direction against the axis the cilia stand on (`direction` in the seed, +1 =
        # the cell moves the way its cilia point, -1 = it is pushed the other way, the way a
        # swimmer goes). A pinned cell reads 0.00.
        pb = max(1, int(round(1.0 / (10.0 * float(d["general"]["dt"]) *
                                     max(1, int(d["general"]["n_frames"]) // max(T - 1, 1))))))
        t1 = max(1, (T - 1) // pb) * pb
        dc = C[t1] - C[0]
        cell_um = float(np.linalg.norm(dc) * um)
        ax = np.asarray(seed.get("direction") or [0.0, 0.0, 1.0], float); ax /= np.linalg.norm(ax)
        cell_cos = float(dc @ ax / max(np.linalg.norm(dc), 1e-12))
    stride = max(1, int(d["general"]["n_frames"]) // max(T - 1, 1))
    dt = float(d["general"]["dt"]) * stride
    y = A[s, per - 1] - A[s, per - 1].mean()
    hz = float("nan")
    if np.ptp(y) > 1e-9:
        # ZERO-PADDED 16x. The window is the last two thirds of the run: 0.2 s on a 3-beat run
        # (5 Hz bins, so 10 Hz falls on a bin) but 0.33 s on a 5-beat run, where the bins sit at
        # 9 and 12 Hz and a true 10 Hz beat read 11.99 and failed the +-10 percent gate. Padding
        # puts the bins 0.19 Hz apart; it adds no information, only the interpolation.
        n_fft = 16 * y.size
        f = np.fft.rfftfreq(n_fft, dt)
        hz = f[1 + int(np.argmax(np.abs(np.fft.rfft(y, n_fft))[1:]))]
    mot = next((o for o in d["operators"] if o.get("op") == "rod_motor"), {})
    # THE WATER COLUMNS ARE OPTIONAL BECAUSE THEY DOMINATE THE COST. A run's water array is
    # 155,374 particles x 301 frames x 3 floats, about 200 MB, and an npz reads a whole array or
    # none of it -- so a sixteen-run table moves 3 GB and takes longer than the runs it measures
    # when a simulation is using the same disk. The rod columns alone answer every GATE.
    ws = wd = exc = 0.0
    if not _NO_WATER and "water_particle__pos" in tr.files:
        W = np.asarray(tr["water_particle__pos"])
        v = np.linalg.norm((W[-1] - W[-2]) / dt, axis=1) * um
        ws = float(np.percentile(v, 99.9))
        # EXCURSION IS NOT TRANSPORT, and conflating them makes the campaign's own control
        # unreadable. `max |W[-1] - W[0]|` is the largest distance any single particle ended from
        # where it began, which a purely OSCILLATORY flow inflates freely: measured, the standing
        # wave -- reciprocal, and by Purcell's scallop theorem incapable of net pumping -- scored
        # 6,350 nm against the travelling wave's 2,357, i.e. the control "beat" the thing it was
        # meant to falsify. It sloshes 62 degrees of water back and forth and takes it nowhere.
        #
        # `drift` is the transport: the MEAN displacement VECTOR of the fluid near the cilium,
        # measured over a whole number of beats so the oscillation cancels instead of aliasing.
        # A mean over particles kills the back-and-forth and leaves only what was carried; taking
        # its magnitude after averaging (not before) is the whole point, since averaging the
        # magnitudes would recover the excursion again.
        per_beat = max(1, int(round(1.0 / (r_hz * dt)))) if (r_hz := hz) and hz > 0 else 1
        n_beat = max(1, (W.shape[0] - 1) // per_beat)
        t1 = n_beat * per_beat
        R0 = np.asarray(tr["rod_node__pos"])[0].mean(0)
        near = np.linalg.norm(W[0] - R0[None, :], axis=1) < 0.1        # within 5 um
        d_net = (W[t1][near] - W[0][near]).mean(0) if near.any() else np.zeros(3)
        wd = float(np.linalg.norm(d_net) * um * 1000)
        exc = float(np.abs(W[t1] - W[0]).max() * um * 1000)
    return dict(name=nm, box=float(np.nanmax(np.abs(R))),
                arc=arc_pct,
                strain=100 * float(np.abs((seg[s] - rest) / rest).max()),
                base=base_amp, base_max=base_max, tip=tip_amp, hz=hz, ws=ws, wd=wd, exc=exc,
                oop=float(oop.max()), base_r=base_r, base_ref=base_ref,
                cell_um=cell_um, cell_cos=cell_cos,
                # NO COMMANDED FREQUENCY UNDER SLIDING CONTROL: the beat rate is an outcome
                # there, so the column reads `-` and the frequency gate does not apply.
                cmd=(float(mot["omega"]) / (2 * np.pi)) if "omega" in mot else float("nan"))


def rod_frames(spec):
    """Each rod's rest direction and stroke-plane normal, [n_rod, 3] each, from the spec's seed."""
    from plexus.operators.rod_ops import rod_layout
    sd = next(o for o in spec["seed"] if o["op"] == "rod_seed")
    d_r, n_r, _ = rod_layout(sd.get("direction") or [0.0, 0.0, 1.0],
                             sd.get("beat_axis") or [0.0, 1.0, 0.0],
                             int(sd.get("n_rod", 1)), float(sd.get("sphere_radius", 0.0)),
                             float(sd.get("cap_deg", 30.0)), float(sd.get("spacing", 0.0)),
                             sd.get("spacing_axis"), dtype=__import__("torch").float64)
    return d_r.numpy(), n_r.numpy()


_NO_WATER = False


def main():
    global _NO_WATER
    args = [a for a in sys.argv[1:] if a != "--no-water"]
    _NO_WATER = "--no-water" in sys.argv[1:]
    print(f"  {'run':18s} {'box':>5s} {'arc%':>6s} {'strain':>7s} {'base':>7s} {'tip':>7s} "
          f"{'t/b':>5s} {'Hz':>6s} {'cmd':>5s} {'oop':>5s} {'base_r':>9s} {'water um/s':>11s} "
          f"{'drift nm':>9s} {'cell um':>8s} {'cos':>5s}  gate")
    for nm in args:
        try:
            r = row(nm)
        except SystemExit:
            print(f"  {nm:18s}  (no run)")
            continue
        except Exception as e:                                        # noqa: BLE001
            print(f"  {nm:18s}  {type(e).__name__}: {str(e)[:50]}")
            continue
        bad = []
        if r["box"] > 1.05: bad.append("OUT OF BOX")
        if r["strain"] > 10: bad.append("strain")
        # 5 percent, not 2: a filament BENDING at 5 percent segment strain shows about 2.5
        # percent of arc deviation from the chord sum alone, so a 2 percent gate failed
        # the working point itself. This catches coiling and stretching, not beating.
        if abs(r["arc"] - 100) > 5: bad.append("arc")
        if r["cmd"] == r["cmd"] and not (r["cmd"] * 0.9 <= r["hz"] <= r["cmd"] * 1.1):
            bad.append("freq")
        # A CILIUM THAT SWEEPS MORE THAN 150 DEGREES AT ITS BASE IS NOT BEATING, it is being
        # carried. Run 0444 passed box, strain and frequency with the filaments lying FLAT in a
        # jet -- base 234 degrees, tip 330 -- and the gate said PASS because nothing asked whether
        # the thing was still a cilium. A real stroke is 30 to 120 degrees.
        if r.get("base_max", r["base"]) > 150: bad.append("FLAIL")
        # OUT OF ITS STROKE PLANE. 15 degrees is generous: the vacuum working point holds under
        # 1, and the failure this catches went to 78 and stayed there.
        if r["oop"] > 15: bad.append("OOP")
        # OFF THE CELL. The base must sit within 10 percent of the sphere radius it was laid at;
        # 0451 had it at 1.5 percent, i.e. at the centre.
        if r["base_r"] == r["base_r"] and abs(r["base_r"] / r["base_ref"] - 1) > 0.1:
            bad.append("OFF-CELL")
        br = f"{r['base_r']:4.2f}/{r['base_ref']:3.1f}" if r["base_r"] == r["base_r"] else "    -   "
        cu = f"{r['cell_um']:8.3f} {r['cell_cos']:+5.2f}" if r["cell_um"] == r["cell_um"] else "       -     -"
        print(f"  {r['name']:18s} {r['box']:5.2f} {r['arc']:6.1f} {r['strain']:6.2f}% "
              f"{r['base']:7.2f} {r['tip']:7.2f} {r['tip']/max(r['base'],1e-9):5.2f} "
              f"{r['hz']:6.2f} {(f'{r[chr(99)+chr(109)+chr(100)]:5.1f}' if r['cmd'] == r['cmd'] else '    -')} "
              f"{r['oop']:5.1f} {br:>9s} {r['ws']:11.3f} "
              f"{r['wd']:9.0f} {cu}  {'PASS' if not bad else 'FAIL: ' + ','.join(bad)}")
    print(f"\n  baseline 0356: base 28.29, tip 36.04, t/b 1.27, 10.00 Hz, water 0 nm (uncoupled)")
    print(f"  drift = NET transport (mean displacement vector, whole beats); excurs = peak excursion.\n  A reciprocal stroke has excursion without drift -- that is the standing-wave control.")


if __name__ == "__main__":
    main()
