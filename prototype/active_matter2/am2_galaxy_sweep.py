#!/usr/bin/env python
"""am2_galaxy_sweep -- overnight search for a good SPIRAL GALAXY from the hydro model.

Tries several *physical* solution families (not just knobs), each a small grid, and
renders every candidate's final frame + movie into
    data/graphs_data/active_matter2/galaxy_sweep/<name>/
plus a labelled contact-sheet montage. Split across the two GPUs:

    python am2_galaxy_sweep.py --rank 0 --nproc 2 --device cuda:0 &
    python am2_galaxy_sweep.py --rank 1 --nproc 2 --device cuda:1 &
    wait; python am2_galaxy_sweep.py --montage

Families:
  bh    -- fixed black hole (softened point mass) + initial disk rotation
  halo  -- dark-matter halo (flat rotation curve) + rotation [+ weak self-gravity]
  full  -- halo + black hole + self-gravity together (the full physical galaxy)
  press -- self-gravity stabilised by strong pressure (high Q) + rotation
  base  -- vortex / nominal regimes under a halo (different active dynamics)
"""
from __future__ import annotations
import os, sys, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from am2_hydro import run, render                                     # noqa: E402

# stability defaults shared by every self-gravitating / strong-gravity config
STAB = dict(g_cap=2.5, rho_max=8.0, dt=0.013, grav_soft=6.0)


def configs():
    """(name, base_preset, overrides) for every candidate galaxy."""
    C = []

    # --- F1: black hole + rotation (no self-gravity) -- the stable baseline ------- #
    for m_bh in (10.0, 16.0, 24.0):
        for spin0 in (0.04, 0.06, 0.09):
            for chi in (0.4, 0.8):
                C.append((f"bh_m{m_bh:g}_s{spin0:g}_chi{chi:g}", "bands",
                          dict(chi=chi, Q=0.6, m_bh=m_bh, bh_soft=9.0, spin0=spin0,
                               g_self=0.0, **STAB)))

    # --- F2: DARK-MATTER HALO + rotation (+ weak self-gravity for arms) ----------- #
    # flat rotation curve gives support at all radii -> disk can hold spiral waves.
    for v_halo in (0.7, 1.0, 1.4):
        for r_halo in (15.0, 30.0):
            for g_self in (0.0, 0.05, 0.12):
                C.append((f"halo_v{v_halo:g}_rh{r_halo:g}_g{g_self:g}", "bands",
                          dict(chi=0.6, Q=0.8, v_halo=v_halo, r_halo=r_halo,
                               g_self=g_self, m_bh=6.0, bh_soft=8.0, spin0=0.05, **STAB)))

    # --- F3: full physical galaxy -- halo + black hole + self-gravity ------------- #
    for v_halo in (1.0, 1.4):
        for g_self in (0.06, 0.14):
            for m_bh in (8.0, 16.0):
                C.append((f"full_v{v_halo:g}_g{g_self:g}_m{m_bh:g}", "bands",
                          dict(chi=0.6, Q=0.9, v_halo=v_halo, r_halo=25.0,
                               g_self=g_self, m_bh=m_bh, bh_soft=8.0, spin0=0.06, **STAB)))

    # --- F4: self-gravity stabilised by strong PRESSURE (high Q), Toomre-marginal - #
    for Q in (1.5, 3.0):
        for g_self in (0.10, 0.20):
            for spin0 in (0.06, 0.10):
                C.append((f"press_Q{Q:g}_g{g_self:g}_s{spin0:g}", "bands",
                          dict(chi=0.6, Q=Q, g_self=g_self, m_bh=8.0, bh_soft=8.0,
                               spin0=spin0, **STAB)))

    # --- F5: different active regimes under a halo -------------------------------- #
    for base in ("vortex", "nominal"):
        for v_halo in (0.8, 1.2):
            for g_self in (0.0, 0.06):
                C.append((f"{base}_v{v_halo:g}_g{g_self:g}", base,
                          dict(v_halo=v_halo, r_halo=25.0, g_self=g_self,
                               m_bh=8.0, bh_soft=8.0, spin0=0.05, **STAB)))
    return C


def _root():
    root = os.environ.get("AM2_DATA_ROOT",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    return os.path.join(root, "graphs_data", "active_matter2", "galaxy_sweep")


def run_share(rank, nproc, device, N, nsteps):
    cfgs = configs()
    mine = cfgs[rank::nproc]
    out = _root()
    print(f"[rank {rank}] {len(mine)}/{len(cfgs)} configs on {device}", flush=True)
    for i, (name, base, ov) in enumerate(mine):
        odir = os.path.join(out, name)
        if os.path.exists(os.path.join(odir, "fig_hydro_final.png")):
            print(f"[rank {rank}] ({i+1}/{len(mine)}) skip {name} (done)", flush=True); continue
        try:
            print(f"[rank {rank}] ({i+1}/{len(mine)}) run {name}: {ov}", flush=True)
            frames = run(base, N=N, nsteps=nsteps, device=device, overrides=ov)
            if not frames:
                print(f"[rank {rank}] {name}: no frames (blew up early)", flush=True); continue
            render(frames, odir, name)
        except Exception:
            print(f"[rank {rank}] {name} FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"[rank {rank}] done", flush=True)


def montage():
    import glob
    from PIL import Image, ImageDraw
    out = _root()
    figs = sorted(glob.glob(os.path.join(out, "*", "fig_hydro_final.png")))
    if not figs:
        print("no figs to montage"); return
    thumbs = []
    for f in figs:
        name = os.path.basename(os.path.dirname(f))
        im = Image.open(f).convert("RGB")
        # keep the TOP (orientation) panel -- the galaxy view -- square-ish crop
        w, h = im.size
        im = im.crop((0, 0, w, w)).resize((240, 240))
        d = ImageDraw.Draw(im); d.text((4, 4), name, fill=(255, 255, 255))
        thumbs.append(im)
    cols = 8
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows * 240), (0, 0, 0))
    for k, t in enumerate(thumbs):
        sheet.paste(t, ((k % cols) * 240, (k // cols) * 240))
    mp = os.path.join(out, "_montage.png")
    sheet.save(mp)
    print(f"montage: {len(thumbs)} candidates -> {mp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--nsteps", type=int, default=80000)
    ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    if a.montage:
        montage()
    else:
        run_share(a.rank, a.nproc, a.device, a.N, a.nsteps)


if __name__ == "__main__":
    main()
