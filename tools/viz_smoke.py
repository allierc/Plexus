#!/usr/bin/env python
"""SMOKE TEST for the live renderer: does the picture actually show what the run is doing?

    python tools/viz_smoke.py si_material/si_jet_sphere_slice
    python tools/viz_smoke.py si_material/si_laplace_r10 --frames 60 --device cuda:1

WHY THIS EXISTS. Three separate rendering faults reached a fifteen-minute run before anyone saw
them, and every one was invisible to the simulation:

  * `enable_shadows()` re-lights actors that asked for `lighting=False`, so a cloud whose colour
    array was uniformly [76, 158, 255] rendered [154, 154, 154]. Blue water, grey picture.
  * `add_chart` inserts a vtkContextActor, and with one present `render_points_as_spheres=True`
    draws NOTHING. 3,701 blue pixels with the panel, 61,331 without, same simulation.
  * a chart series whose data updates every frame but whose CHART never redraws. Counting the
    array proved "live" and the rendered panel was frozen -- which is why this tool compares
    PIXELS between frames and not state.

Each of those is a property of the image, so the only test that catches them is one that looks at
the image. Four rows, all cheap:

  visible    a run with live particles must light more than a floor number of pixels. Catches the
             sprite/chart interaction and anything else that silently draws nothing.
  colour     the dominant non-grey pixel must match a colour the spec DECLARED, within tolerance.
             Catches shadows and any lighting model that washes flat colour toward white.
  moving     consecutive rendered frames must DIFFER. Catches a frozen chart, a stale actor, and a
             renderer handed the same buffer every frame.
  panel      when `plotting.cross_section` is set, the panel REGION must change between frames too
             -- separately from the rest of the image, or a moving 3D view would mask a dead panel.

It renders a handful of frames at a reduced particle count, so it costs seconds, not minutes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _dominant(img, exclude_grey=True):
    """The most common lit colour, ignoring greys (box frame, obstacles) when asked."""
    px = img.reshape(-1, 3).astype(int)
    px = px[px.sum(1) > 90]
    if exclude_grey and px.size:
        spread = px.max(1) - px.min(1)
        px = px[spread > 25]
    if not px.size:
        return None, 0
    u, c = np.unique(px, axis=0, return_counts=True)
    i = int(np.argmax(c))
    return u[i].tolist(), int(c[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="<type>/<name>, e.g. si_material/si_jet_sphere_slice")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--particles", type=int, default=60_000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--shots", type=int, default=4, help="rendered frames to compare")
    ap.add_argument("--min-lit", type=int, default=3000)
    a = ap.parse_args()

    import plexus.operators                                          # noqa: F401
    import plexus.operators.mpm_warp                                 # noqa: F401
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
    from plexus.live_movie import LiveMovie
    from plexus.schema import load

    typ, name = a.spec.split("/", 1)
    s = yaml.safe_load(open(os.path.join(ROOT, "config", typ, name + ".yaml")))
    s["general"]["n_frames"] = int(a.frames)
    s["general"]["save_data"] = False
    if isinstance(s["sets"]["mpm_particle"].get("per_parent"), int):
        s["sets"]["mpm_particle"]["per_parent"] = min(
            s["sets"]["mpm_particle"]["per_parent"], int(a.particles))
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f)
    f.close()
    CFL(f.name)
    sim = load(f.name)
    os.unlink(f.name)

    engine = str(getattr(sim, "engine", "default") or "default").lower()
    if engine in ("", "default"):
        from plexus.engine import run
    else:
        import importlib
        run = importlib.import_module(f"plexus.{engine}_engine").run

    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    lm = LiveMovie(out=out, world=sim.world_size, n_frames=sim.n_frames,
                   up=int((sim.plotting or {}).get("up_axis", 1)), render_n=400_000,
                   max_frames=a.shots, name=name, sim=sim, style=sim.plotting, stills=0,
                   dt=sim.dt, time_s=1.0)
    shots, live = [], []

    _call = lm.__call__

    def hook(H, tick):
        _call(H, tick)
        if lm.rendered and (not shots or lm.rendered > len(shots)):
            shots.append(np.asarray(lm.p.image).copy())
            p = H.level("mpm_particle")
            occ = getattr(p, "occ", None)
            live.append(int((occ > 0).sum()) if occ is not None else int(p.n))

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        run(sim, out_path=None, device=a.device, progress=False, on_frame=hook)
    lm.close()
    os.path.exists(out) and os.unlink(out)

    if len(shots) < 2:
        print(f"\n  {name}: only {len(shots)} frame(s) rendered -- cannot compare\n")
        sys.exit(1)

    declared = [(np.array(c) * 255).round().astype(int).tolist()
                for c in ((sim.plotting or {}).get("colors") or {}).values()]
    print(f"\n  {name}   {len(shots)} rendered frames, live particles "
          f"{min(live):,}..{max(live):,}\n")
    print(f"  {'':<10}{'lit px':>10}{'dominant colour':>20}{'vs previous frame':>20}")
    print("  " + "-" * 62)
    ok = True
    prev = None
    for i, img in enumerate(shots):
        lit = int((img.reshape(-1, 3).sum(1) > 90).sum())
        dom, n = _dominant(img)
        diff = "-" if prev is None else f"{np.abs(img.astype(int) - prev).mean():.2f} mean |d|"
        print(f"  frame {i:<4}{lit:>10,}{str(dom):>20}{diff:>20}")
        prev = img

    lit0 = int((shots[-1].reshape(-1, 3).sum(1) > 90).sum())
    row_vis = lit0 >= a.min_lit
    dom, _ = _dominant(shots[-1])
    # COMPARED BY HUE, NOT BY BRIGHTNESS. A dot 1.3 px wide against black covers its pixel only
    # partly, so the rendered colour is the declared one SCALED: si_three_viscosities' water reads
    # [55, 115, 186] against a declared [76, 158, 255] -- the same hue at 0.72x, and an absolute
    # comparison called that a failure. Normalising both to unit length tests the direction, which
    # is what "is it blue" means, and still catches grey: [154, 154, 154] normalises to
    # (0.58, 0.58, 0.58) and is nowhere near blue's (0.25, 0.51, 0.83).
    def _unit(v):
        v = np.asarray(v, dtype=float)
        return v / max(float(np.linalg.norm(v)), 1e-9)
    row_col = bool(declared) and dom is not None and any(
        float(np.abs(_unit(dom) - _unit(d)).max()) < 0.10 for d in declared)
    d_full = np.abs(shots[-1].astype(int) - shots[0].astype(int)).mean()
    row_mov = d_full > 0.05

    print()
    print(f"    visible   {lit0:,} lit pixels (needs >= {a.min_lit:,})"
          f"                {'PASS' if row_vis else 'FAIL'}")
    print(f"    colour    dominant {dom} vs declared {declared} (by hue)"
          f"   {'PASS' if row_col else 'FAIL'}")
    print(f"    moving    first vs last frame differ by {d_full:.3f}"
          f"          {'PASS' if row_mov else 'FAIL'}")

    cs = (sim.plotting or {}).get("cross_section")
    # `only` HAS NO PANEL TO CHECK -- it filters the 3D cloud to the slab instead, so the `moving`
    # row already covers it. Testing a panel region that does not exist reported FROZEN forever.
    if cs and not (cs is not True and cs.get("only")):
        # THE PANEL GETS ITS OWN COMPARISON. A moving 3D view would otherwise carry a dead panel
        # through the `moving` row -- which is exactly the fault this row was written for.
        only = bool(cs is not True and cs.get("only"))
        h, w, _ = shots[0].shape
        reg = ((slice(int(.07 * h), int(.93 * h)), slice(int(.12 * w), int(.92 * w))) if only
               else (slice(int(.08 * h), int(.36 * h)), slice(int(.01 * w), int(.29 * w))))
        d_panel = np.abs(shots[-1][reg].astype(int) - shots[0][reg].astype(int)).mean()
        row_pan = d_panel > 0.05
        ok &= row_pan
        print(f"    panel     cross-section region differs by {d_panel:.3f}"
              f"      {'PASS' if row_pan else 'FAIL -- the panel is FROZEN'}")

    ok &= row_vis and row_col and row_mov
    print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
