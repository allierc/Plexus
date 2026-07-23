#!/usr/bin/env python
"""round_XX tubing runs that GENUINELY initialise from smoke_hom's homogenised vesicle (load_mesh_3d reads
archive/smoke_hom/ckpt.npz) and seed a FEW BIG RD spots on it, then extrude tubes (winning regime: rho=0
locked body + a_sw narrow so only each spot's peak grows -> narrow coherent tubes, not wide balloons).
Archives to archive/<preset>/ (name presets round_XX_<desc>). --only <preset>.  Cluster:
    TV_SCRIPT=run_tyssue_round.py python cluster_gen.py round_01_big ..."""
from __future__ import annotations
import os, sys, json, tempfile, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, ckpt  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import face_geometry_3d
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross, make_movie_axes, draw_movie_frame
from tyssue_diag import hollow_flags
from tube_analysis import analyze
import torch

CKPT = os.path.join(HERE, "archive", "smoke_hom", "ckpt.npz")
VBUF, CBUF = 30000, 16000                                   # CBUF MUST exceed VBUF/2 (a closed mesh has F ~ V/2 + 2 cells;
#   CBUF == VBUF/2 overflowed the cell buffer by exactly 2 when division filled the vertex buffer). Margin fixes it.

# few BIG RD spots on the homogenised mesh (gamma low = long wavelength = fewer bigger); rho=0 locked +
# a_sw narrow (only peak grows -> narrow tube). frames long to elongate.
PRESETS = {
    "round_01_big": dict(frames=400, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.05, a_sw=3.5),
    "round_01_bigger": dict(frames=400, chi=4.0, gamma=0.2, rho=0.0, vth=1.4, rate=0.05, a_sw=3.5),
    # round_02: from ckpt, match the WINNING controlled rate (h6nw_a35: rate 0.04, hollow 50) -- rate 0.05
    # ran away (hollow 3348). Slower rate keeps the activator from runaway-ballooning. gamma 0.3 a_sw 3.5.
    "round_02_r04": dict(frames=400, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.04, a_sw=3.5),
    "round_02_r03": dict(frames=450, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.03, a_sw=3.5),
    # round_03: CONES on the homogenised ckpt -> N BIG red spots visible at FRAME 0 (like the target image),
    # re-seeded at the tips = tip-tracking = the clean-tube mechanism. dt=1.0 (no RD/CFL). a_sw=0.5 (binary cone).
    "round_03_cone3": dict(frames=350, spots=3, cone_deg=18.0, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5),
    "round_03_cone5": dict(frames=350, spots=5, cone_deg=16.0, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5),
    # round_04: init nails it (white vesicle + 3 big red spots) but tubes BALLOONED (K_V=3 soft -> cells
    # overshoot instead of divide). Fix like the clean fig5 cones: stiff K_V=4 + tight vth (cells divide-and-
    # extend the tube, proliferation not inflation) + more division throughput + more relaxation.
    "round_04_cone3": dict(frames=200, spots=3, cone_deg=18.0, rho=0.0, vth=1.4,  rate=0.04, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30),
    "round_04_slow":  dict(frames=250, spots=3, cone_deg=18.0, rho=0.0, vth=1.35, rate=0.03, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30),
    # round_05: FAST (~15min target) + TIP-SIZE CONTROL. Narrow cones (deg 12 -> fewer activated cells ->
    # far less proliferation -> fast); stiff K_V=5 (no overshoot) + tight vth=1.3 (tip cells stay ~body size,
    # Okuda [2/3,4/3]v_ref). 150 frames. Two tip caps to compare.
    "round_05_v13": dict(frames=150, spots=3, cone_deg=12.0, rho=0.0, vth=1.30, rate=0.04, a_sw=0.5, K_V=5.0, mdf=0.03, relax=28),
    "round_05_v12": dict(frames=150, spots=3, cone_deg=12.0, rho=0.0, vth=1.20, rate=0.04, a_sw=0.5, K_V=5.0, mdf=0.03, relax=28),
    # round_06: HARD SIZE CAP (vcap) -- force-divide any cell >= vcap x v_ref, bypassing the throttle, so NO
    # cell (tip included) exceeds the cap. Directly bounds cell size. cone 12 (fast), K_V=4, frames 150.
    "round_06_cap15": dict(frames=150, spots=3, cone_deg=12.0, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_06_cap13": dict(frames=150, spots=3, cone_deg=12.0, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30, vcap=1.3),
    # round_07: ADD RD. Big spots seeded at frame 0 (cones once) then a Brusselator EVOLVES them (dynamic
    # red/white partition). The vcap size cap (which fixed round_06) should now keep the RD tubes clean too
    # -- the balloon that killed RD before was oversized cells, which vcap force-divides. a_sw narrow (RD peak).
    "round_07_rd":     dict(frames=200, spots=3, cone_deg=14.0, rd=True, chi=4.0, gamma=0.3, rho=0.0, vth=1.5, rate=0.04, a_sw=2.5, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_07_narrow": dict(frames=200, spots=3, cone_deg=14.0, rd=True, chi=4.0, gamma=0.3, rho=0.0, vth=1.5, rate=0.04, a_sw=3.2, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    # round_08: RD was clean but FLAT -- a_sw=2.5 was above the Brusselator activator (~1) so no growth drive.
    # Lower a_sw to read the RD spots and drive tubing (round_06 cones worked at a_sw=0.5). Bracket the peak.
    "round_08_a15": dict(frames=200, spots=3, cone_deg=14.0, rd=True, chi=4.0, gamma=0.3, rho=0.0, vth=1.5, rate=0.04, a_sw=1.5, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_08_a10": dict(frames=200, spots=3, cone_deg=14.0, rd=True, chi=4.0, gamma=0.3, rho=0.0, vth=1.5, rate=0.04, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    # round_09: THINNER + LONGER tubes. Thin = narrow activation (small cone_deg + steep Hill); long = fast
    # tip growth + more frames (cone tip-tracking keeps extending). Compare sharp cones vs sharpened RD.
    "round_09_cone": dict(frames=300, spots=3, cone_deg=8.0,  rho=0.0, vth=1.5, rate=0.06, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_09_rd":   dict(frames=300, spots=3, cone_deg=8.0, rd=True, chi=4.0, gamma=0.3, rho=0.0, vth=1.5, rate=0.06, a_sw=1.3, hill=10.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    # round_09_cone x2 frames (user): 300->600 for LONGER tubes. Same thin-cone params.
    "round_09_cone_x2": dict(frames=600, spots=3, cone_deg=8.0, rho=0.0, vth=1.5, rate=0.06, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    # round_10: reproduce the OKUDA GRADIENT (activator peak at tip -> graded growth). GRADED coupling
    # (gentle hill=2, growth prop to activator, not a binary threshold) on a localized RD spot. Measure
    # tip_act = corr(activator, radius): +1 means the activator gradient sits at the protruding tips (Okuda).
    "round_10_grad2": dict(frames=250, spots=3, cone_deg=10.0, rd=True, chi=4.0, gamma=0.3, rho=0.0, vth=1.5, rate=0.05, a_sw=0.8, hill=2.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_10_grad1": dict(frames=250, spots=3, cone_deg=10.0, rd=True, chi=4.0, gamma=0.3, rho=0.0, vth=1.5, rate=0.05, a_sw=0.8, hill=1.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    # ===== round_11: UNDERSTAND THE RD FIRST (rate=0, NO growth) -- small step before coupling extrusion.
    # Seed 3 big spots on the homogenised vesicle, then watch what the RD does to them: does it KEEP 3 stable
    # big spots, or reorganise into speckle/spread? How FAST (round_09_rd's pattern formed only ~12s)?
    # Brusselator vs Gray-Scott? Judge from the movie + spots (count) + red_frac (localised?) over time.
    "round_11_h1_g03":   dict(frames=200, spots=3, cone_deg=12.0, rd=True, chi=4.0, gamma=0.3, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # round_09_rd's RD (slow, big)
    "round_11_h2_g10":   dict(frames=200, spots=3, cone_deg=12.0, rd=True, chi=4.0, gamma=1.0, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # faster reaction
    "round_11_h3_g20":   dict(frames=200, spots=3, cone_deg=12.0, rd=True, chi=4.0, gamma=2.0, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # fast (small spots?)
    "round_11_h4_chi3":  dict(frames=200, spots=3, cone_deg=12.0, rd=True, chi=3.0, gamma=0.6, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # lower diffusion ratio
    "round_11_h5_chi5":  dict(frames=200, spots=3, cone_deg=12.0, rd=True, chi=5.0, gamma=1.0, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # higher (near CFL)
    "round_11_h6_hiB":   dict(frames=200, spots=3, cone_deg=12.0, rd=True, chi=4.0, gamma=1.0, B=4.5, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # higher activator contrast
    "round_11_h7_gsspot":dict(frames=200, spots=3, cone_deg=12.0, rd=True, rd_impl="gray_scott", F=0.030, kk=0.062, chi=1.3, d_a=0.08, d_h=0.16, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # GS SPOT regime (stable spots)
    "round_11_h8_gscoral":dict(frames=200, spots=3, cone_deg=12.0, rd=True, rd_impl="gray_scott", F=0.055, kk=0.062, chi=1.3, d_a=0.08, d_h=0.16, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # GS coral/labyrinth
    # ===== round_12: COUPLE GROWTH to the stable Gray-Scott spots -> DOME-vs-TUBE + emergence test. GS holds
    # strong stable localized spots (round_11 h7). Now: does growth on them TUBE (spot rides tip = emergent) or
    # DOME? Controls C/D isolate tip-riding directly. tip_act=corr(activator,radius): high = spot at TIP.
    # GS params F=0.03 kk=0.062 chi=1.3 d_a=0.08 (activator ~[0,1.5], so a_sw~0.7). 300 frames.
    "round_12_A_gsgrad":  dict(frames=300, spots=3, cone_deg=12.0, rd=True, rd_impl="gray_scott", F=0.03, kk=0.062, chi=1.3, d_a=0.08, d_h=0.16, rho=0.0, vth=1.5, rate=0.04, a_sw=0.7, hill=2.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),  # GS + GRADED growth
    "round_12_B_gsthr":   dict(frames=300, spots=3, cone_deg=12.0, rd=True, rd_impl="gray_scott", F=0.03, kk=0.062, chi=1.3, d_a=0.08, d_h=0.16, rho=0.0, vth=1.5, rate=0.04, a_sw=0.7, hill=6.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),  # GS + THRESHOLD growth
    "round_12_C_flatfix": dict(frames=300, spots=3, cone_deg=16.0, seed_once=True, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5, hill=6.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),  # CONTROL: flat spot frozen on cells -> DOME?
    "round_12_D_flatride":dict(frames=300, spots=3, cone_deg=16.0, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5, hill=6.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),  # CONTROL: cones re-seed (tip-riding) -> TUBE?
    "round_12_E_gsslow":  dict(frames=300, spots=3, cone_deg=12.0, rd=True, rd_impl="gray_scott", F=0.03, kk=0.062, chi=1.3, d_a=0.08, d_h=0.16, rho=0.0, vth=1.5, rate=0.02, a_sw=0.7, hill=2.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),  # GS graded, SLOW (RD keeps pace)
    "round_12_F_gshida":  dict(frames=300, spots=3, cone_deg=12.0, rd=True, rd_impl="gray_scott", F=0.03, kk=0.062, chi=1.3, d_a=0.15, d_h=0.30, rho=0.0, vth=1.5, rate=0.04, a_sw=0.7, hill=2.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),  # GS + hi diffusion (spot follows tip?)
    # ===== round_13: STUDY GIERER-MEINHARDT (Okuda's RD, ref 37) -- rate=0 (no growth), seed 3 spots, watch
    # if the SELF-ENHANCING a^2/h peak HOLDS the spots as stable localized peaks WITH A GRADIENT (the thing
    # Brusselator can't and Gray-Scott approximates). Key knob = inhibitor range d_h/d_a (lateral inhibition
    # -> few big peaks). GM chem=[a,h]; a0 small basal. Judge from movie + spots + act_range.
    "round_13_gm_base": dict(frames=200, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),
    "round_13_gm_widinh":dict(frames=200, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.03, d_h=1.5, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # wider inhibition -> fewer bigger peaks
    "round_13_gm_slowdecay":dict(frames=200, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=0.8, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # slower activator decay -> stronger peak
    "round_13_gm_fast": dict(frames=200, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.5, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=2.0, rho=0.0, vth=1.5, rate=0.0, a_sw=1.0, K_V=4.0, mdf=0.03, relax=30),  # faster/stronger -> forms early
    # ===== round_14: EMERGENCE TEST -- couple GRADED growth onto the stable GM peaks (gm_base holds 3 strong
    # localized peaks). Does the SELF-ENHANCING peak RIDE the growing tip (tube, tip_act high = emergent Okuda)
    # or stay at the base (dome)? GM activator ~[0,2.5] so a_sw~1.2; hill=2 = graded (Okuda gradient). rho=0.
    "round_14_gm_grad":  dict(frames=300, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.04, a_sw=1.2, hill=2.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_14_gm_slow":  dict(frames=350, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.5, rho=0.0, vth=1.5, rate=0.025, a_sw=1.2, hill=2.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),  # slow growth, faster RD -> peak keeps pace w/ tip
    "round_14_gm_strong":dict(frames=300, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=0.8, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.04, a_sw=1.5, hill=2.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),  # stronger peak (slow decay)
    # ===== round_15: STABILISE-THEN-GROW (user) -- let the GM pattern amplify to stable peaks FIRST
    # (grow_after frames of RD-only warmup), THEN switch on growth+division. Growth should act on a settled
    # 3-peak pattern instead of a forming one. Compare vs round_14 (grow from frame 0).
    "round_15_warm60":  dict(frames=360, grow_after=60,  spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.04, a_sw=1.2, hill=2.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_15_warm100": dict(frames=400, grow_after=100, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.04, a_sw=1.2, hill=2.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    # ===== round_16: LOCALIZE the growth. round_14/15 over-proliferated (~15k cells, wall-killed) -> the
    # graded coupling grew the whole GM peak+gradient skirt (wide -> dome). Raise a_sw so ONLY the peak APEX
    # (a>~1.6, GM range ~[0,2.5]) drives growth -> narrow tube + modest cell count + finishes fast. warmup 50.
    "round_16_apex":  dict(frames=200, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.05, a_sw=1.8, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_16_apex2": dict(frames=200, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.05, a_sw=1.5, hill=4.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    # ===== round_17: CONTROL THE RUNAWAY. round_16 worked in the INITIATION phase (tip_act 0.8, localized red
    # lobes) then blew up: growth -> daughters inherit activator -> self-enhance -> flood. Okuda's fix = a
    # STRONGER/FASTER lateral inhibitor (behind the front cells turn white). Boost inhibition (d_h up, mu_h
    # down) + slow growth so the RD keeps the peak localized. Faster RD (rd_rate) so inhibition outpaces growth.
    "round_17_inhib":  dict(frames=250, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=0.7, a0=0.01, d_a=0.05, d_h=1.6, chi=4.0, rd_rate=2.0, rho=0.0, vth=1.5, rate=0.03, a_sw=1.5, hill=4.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_17_inhib2": dict(frames=250, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.2, mu_a=1.0, mu_h=0.6, a0=0.005, d_a=0.04, d_h=2.0, chi=4.0, rd_rate=2.5, rho=0.0, vth=1.5, rate=0.025, a_sw=1.6, hill=4.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    # ===== round_18: 8-JOB GAMMA SWEEP -- map tube <-> branch/cauliflower via the RD-vs-growth TIMESCALE
    # (Okuda's gamma). High RD-rate / slow growth = RD dominates -> spot locks to ONE tip -> TUBE; slow RD /
    # fast growth -> growth outruns the RD, re-nucleates on each bud -> BRANCH/cauliflower. + inhibition,
    # localization, very-slow controls. Census (red_at_tip, red_over_tip) says where each lands. warmup 50.
    "round_18_gamma1": dict(frames=200, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=1.0, rho=0.0, vth=1.5, rate=0.05,  a_sw=1.6, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_18_gamma2": dict(frames=200, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=2.0, rho=0.0, vth=1.5, rate=0.04,  a_sw=1.6, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_18_gamma3": dict(frames=200, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=3.0, rho=0.0, vth=1.5, rate=0.03,  a_sw=1.6, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_18_gamma4": dict(frames=200, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=4.0, rho=0.0, vth=1.5, rate=0.02,  a_sw=1.6, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_18_gamma6": dict(frames=240, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=6.0, rho=0.0, vth=1.5, rate=0.015, a_sw=1.6, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_18_inhib":  dict(frames=220, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=0.6, a0=0.01, d_a=0.05, d_h=1.4, chi=4.0, rd_rate=3.0, rho=0.0, vth=1.5, rate=0.02,  a_sw=1.6, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_18_apex":   dict(frames=200, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=3.0, rho=0.0, vth=1.5, rate=0.03,  a_sw=2.0, hill=4.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_18_vslow":  dict(frames=280, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=4.0, rho=0.0, vth=1.5, rate=0.012, a_sw=1.6, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    # ===== round_19: TEST THE rd_interface_tension OPERATOR (user hypothesis) -- purse-string on the red/white
    # ring + normal extrusion of red cells -> CYLINDER not ball. H/V/F: ctrl (no interface, should bulge) vs
    # purse-string vs extrusion vs both. GM base + moderate growth. iface_asw=1.0 (red threshold). Judge by
    # census (red_at_tip~1 + constant tube_diam = the interface built a tube wall) vs the ctrl's dome.
}
_GMB = dict(frames=200, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=2.0, rho=0.0, vth=1.5, rate=0.03, a_sw=1.5, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5, grow_after=50, iface_asw=1.0)
PRESETS["round_19_ctrl"]  = dict(_GMB, K_purse=0.0, K_extrude=0.0)   # no interface -> should DOME (control)
PRESETS["round_19_purse"] = dict(_GMB, K_purse=3.0, K_extrude=0.3)   # strong ring
PRESETS["round_19_extr"]  = dict(_GMB, K_purse=1.0, K_extrude=1.2)   # strong outward extrusion
PRESETS["round_19_both"]  = dict(_GMB, K_purse=2.0, K_extrude=0.8)   # purse-string + extrusion
# ===== round_20: CONFINEMENT sweep -- get red_over_tip from ~8 down to ~1 (activator confined to the tip)
# so the runaway/flood stops. Levers: Meinhardt SATURATION (sat, bounds the peak), strong lateral inhibition
# (mu_h down, d_h up), slow growth (RD redistributes), + combos. NO interface op yet -- fix confinement first.
_GMC = dict(frames=200, grow_after=50, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt", gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rd_rate=2.0, rho=0.0, vth=1.5, rate=0.03, a_sw=1.5, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5)
PRESETS["round_20_sat05"]   = dict(_GMC, sat=0.5)                                        # moderate saturation
PRESETS["round_20_sat10"]   = dict(_GMC, sat=1.0)                                        # strong saturation
PRESETS["round_20_sat20"]   = dict(_GMC, sat=2.0)                                        # very strong saturation
PRESETS["round_20_inhib"]   = dict(_GMC, mu_h=0.4, d_h=3.0, rd_rate=3.0)                 # strong lateral inhibition only
PRESETS["round_20_satinh"]  = dict(_GMC, sat=1.0, mu_h=0.5, d_h=2.0, rd_rate=3.0)        # saturation + inhibition
PRESETS["round_20_slow"]    = dict(_GMC, sat=1.0, rate=0.01, frames=260)                 # saturation + slow growth
PRESETS["round_20_combo"]   = dict(_GMC, sat=1.0, mu_h=0.5, d_h=2.0, rd_rate=3.0, rate=0.015, a_sw=1.7, frames=240)  # all
PRESETS["round_20_satapex"] = dict(_GMC, sat=1.5, a_sw=1.8, hill=4.0)                    # saturation + apex localization
# ===== round_21: KILL BACKGROUND IGNITION -- the flood starts from a0 seeding the white background + a
# Turing-unstable RD filling the growing domain. Saturation makes RINGS (bad); inhibition kills the seeds.
# Try: a0=0 (no background source) + a LOCALIZED-STRUCTURE regime (seeded spots persist, homogeneous state
# stable, no spontaneous nucleation). Goal: red_over_tip ~1, spots stay FILLED (no ring, no flood).
PRESETS["round_21_a0z"]     = dict(_GMC, a0=0.0, mu_h=0.8)                               # no background seed
PRESETS["round_21_a0z_sat"] = dict(_GMC, a0=0.0, sat=0.3, mu_h=0.8)                      # + MILD saturation (below ring threshold)
PRESETS["round_21_subcrit"] = dict(_GMC, a0=0.002, mu_a=1.6, mu_h=0.9)                   # sub-critical: stable low background
# ===== round_22: FIG 4a -- with the AMOUNT-CONSERVATION fix in (morphogen = amount, c=m/v; growth dilutes),
# the flood is a gamma (patterning-vs-deformation rate) problem, NOT an a0/regime one (round_21 superseded).
# Fix chi, sweep gamma = rd_rate/deformation -> reproduce Okuda's three regimes (blur / hold / hold+increase),
# AND A/B the fix directly: conserve_amount OFF (the old bug: c held while v grows => mass CREATED => spurious
# tip-feeding) vs ON. Judge from the census (red_frac over frames: blur=floods, hold=localized) + movie.
_F4 = dict(frames=220, grow_after=40, spots=3, cone_deg=12.0, rd=True, rd_impl="gierer_meinhardt",
           gm_rho=1.0, mu_a=1.0, mu_h=1.0, a0=0.01, d_a=0.05, d_h=0.7, chi=4.0, rho=0.0, vth=1.5,
           rate=0.02, a_sw=1.5, hill=3.0, K_V=4.0, mdf=0.03, relax=30, vcap=1.5)
PRESETS["round_22_off_g1"]  = dict(_F4, rd_rate=1.0, conserve_amount=False)   # OLD BUG: creates mass -> flood control
PRESETS["round_22_on_g1"]   = dict(_F4, rd_rate=1.0, conserve_amount=True)    # the FIX at the SAME gamma (direct A/B)
PRESETS["round_22_g01"]     = dict(_F4, rd_rate=0.1)                          # gamma very low  -> BLUR (dilution outruns RD)
PRESETS["round_22_g03"]     = dict(_F4, rd_rate=0.3)                          # gamma low
PRESETS["round_22_g2"]      = dict(_F4, rd_rate=2.0)                          # gamma > 1       -> HOLD
PRESETS["round_22_g5"]      = dict(_F4, rd_rate=5.0)                          # gamma high      -> HOLD (+spot increase?)
PRESETS["round_22_slowgrow"]= dict(_F4, rd_rate=1.0, rate=0.008)             # slow deformation = high effective gamma
PRESETS["round_22_fastgrow"]= dict(_F4, rd_rate=1.0, rate=0.045)             # fast deformation = low  effective gamma -> blur
# ===== round_23: HOLD sweet spot (issue 1->2). round_22 showed slow growth + fast RD (high gamma) HOLDS the
# pattern (red_frac ~0.1, red_over_tip ~1) while fast growth FLOODS. Now find the fastest growth that still
# holds a CONFINED spot AND deforms it into a clean bud (protr up, red_over_tip ~1, red at tip). Sweep growth
# rate x rd_rate around the hold/blur boundary; amount-conservation ON (default).
PRESETS["round_23_r006_rd2"] = dict(_F4, rate=0.006, rd_rate=2.0)
PRESETS["round_23_r010_rd2"] = dict(_F4, rate=0.010, rd_rate=2.0)
PRESETS["round_23_r014_rd2"] = dict(_F4, rate=0.014, rd_rate=2.0)
PRESETS["round_23_r020_rd2"] = dict(_F4, rate=0.020, rd_rate=2.0)
PRESETS["round_23_r010_rd1"] = dict(_F4, rate=0.010, rd_rate=1.0)
PRESETS["round_23_r010_rd4"] = dict(_F4, rate=0.010, rd_rate=4.0)
PRESETS["round_23_r008_rd3"] = dict(_F4, rate=0.008, rd_rate=3.0)
PRESETS["round_23_r006_rd4"] = dict(_F4, rate=0.006, rd_rate=4.0)             # slowest growth + fast RD = strongest hold
# ===== round_24: GROW THE BUD (issue 2->3). round_23 sweet spot = rd_rate=1, rate=0.010 (confined bud,
# red_over_tip=1, tip_act=0.8, protr=1.14). Now grow it into a real PROTRUSION: a SINGLE focused spot +
# steeper growth switch (hill: grow only at the activated tip) + longer run. Hypothesis: protr rises toward
# a finger while red_over_tip stays ~1 and tip_act stays ~0.8 (confinement + tip-riding preserved).
_H24 = dict(_F4, rd_rate=1.0, rate=0.010, spots=1)                           # single focused bud, sweet-spot rates
PRESETS["round_24_1sp_long"]    = dict(_H24, frames=350)
PRESETS["round_24_1sp_hill5"]   = dict(_H24, frames=350, hill=5.0)
PRESETS["round_24_1sp_hill8"]   = dict(_H24, frames=350, hill=8.0)
PRESETS["round_24_1sp_r012h5"]  = dict(_H24, frames=350, hill=5.0, rate=0.012)
PRESETS["round_24_1sp_asw18h5"] = dict(_H24, frames=350, hill=5.0, a_sw=1.8)
PRESETS["round_24_1sp_400h5"]   = dict(_H24, frames=400, hill=5.0)
PRESETS["round_24_1sp_r008h5"]  = dict(_H24, frames=400, hill=5.0, rate=0.008)  # slow+long = biggest confined bud?
PRESETS["round_24_3sp_long"]    = dict(_F4, rd_rate=1.0, rate=0.010, frames=350)  # 3 spots (compare to single)
# ===== round_25: SUSTAIN confinement over a long run (issue 1, deeper). round_24 showed amount-conservation
# only DELAYS the flood (~frame 200): as the body proliferates the domain grows and the Turing pattern REFILLS
# it (domain-growth insertion). Fix = amount-conservation (correct base state) + keep the BODY white so it
# neither nucleates new spots nor grows: a0->0 (no basal ignition) and/or sub-critical GM (mu_a up). round_21's
# a0 idea was right but INCOMPLETE without amount-conservation. Hypothesis: body stays white the whole run ->
# only the bud grows -> no flood -> the bud persists. Base = round_23 sweet spot, 350 frames.
_H25 = dict(_F4, rd_rate=1.0, rate=0.010, spots=3, frames=350)
PRESETS["round_25_a0z"]        = dict(_H25, a0=0.0)                          # no basal ignition of the body
PRESETS["round_25_a0low"]      = dict(_H25, a0=0.003)                        # low basal
PRESETS["round_25_a0z_ga80"]   = dict(_H25, a0=0.0, grow_after=80)          # + settle longer before growth
PRESETS["round_25_subcrit"]    = dict(_H25, mu_a=1.4)                        # sub-critical: stable low body
PRESETS["round_25_a0z_mua14"]  = dict(_H25, a0=0.0, mu_a=1.4)               # no ignition + sub-critical
PRESETS["round_25_a0z_asw18"]  = dict(_H25, a0=0.0, a_sw=1.8)               # + only strong tip grows
PRESETS["round_25_a0z_slow"]   = dict(_H25, a0=0.0, rate=0.008)            # + slow growth (higher gamma)
PRESETS["round_25_a0z_ga100"]  = dict(_H25, a0=0.0, grow_after=100)        # settle even longer
PRESETS["round_21_gs"]      = dict(_GMC, rd_impl="gray_scott", F=0.045, kk=0.062, chi=1.3, d_a=0.08, d_h=0.16, a_sw=0.4)  # Gray-Scott stable-spot under growth


def make(p):
    cones = "spots" in p; rd = p.get("rd", False)
    dt = 1.0 if (cones and not rd) else 0.02                    # RD needs the small dt (CFL); pure cones can use dt=1
    ops = [{"op": "load_mesh_3d", "at": "vertex", "cell_set": "cell", "ckpt": CKPT, "before_frame": 1},
           {"op": "cell_geometry_3d", "at": "cell"}]
    sched = ["load_mesh_3d", "cell_geometry_3d"]
    if cones:                                                   # N BIG spots at frame 0 (cones); tip-tracking
        # rd=True: seed the big spots ONCE (before_frame=3) then let a Brusselator EVOLVE them (dynamic RD
        # partition) -- big-spot init AND real RD. rd=False: cones re-seed every frame (pure tip-tracking).
        # seed once (before_frame) when rd evolves it OR seed_once=True (a FLAT spot frozen on the cells: no
        # tip-tracking -> the control that should DOME). Else re-seed every frame = tip-riding cones -> tube.
        seed = {"before_frame": 3} if (rd or p.get("seed_once")) else {}
        ops += [{"op": "cell_rd_seed", "at": "cell", "mode": "cones", "n_spots": p["spots"], "cone_deg": p["cone_deg"], **seed}]
        sched += ["cell_rd_seed"]
        if rd:
            impl = p.get("rd_impl", "brusselator")
            if impl == "gray_scott":                              # stable localized SPOTS (substrate depletion)
                react = {"op": "cell_react", "at": "cell", "implementation": "gray_scott", "F": p.get("F", 0.055), "kk": p.get("kk", 0.062), "rate": p.get("rd_rate", 1.0)}
            elif impl == "gierer_meinhardt":                      # SELF-ENHANCING a^2/h peak (Okuda's RD, ref 37)
                react = {"op": "cell_react", "at": "cell", "implementation": "gierer_meinhardt", "gm_rho": p.get("gm_rho", 1.0), "mu_a": p.get("mu_a", 1.0), "mu_h": p.get("mu_h", 1.0), "a0": p.get("a0", 0.01), "rate": p.get("rd_rate", 1.0)}
            else:                                                 # Brusselator (Turing spots); B = activator contrast
                react = {"op": "cell_react", "at": "cell", "implementation": "brusselator", "gamma": p.get("gamma", 0.3), "A": 1.0, "B": p.get("B", 3.0)}
            ops += [{"op": "cell_adjacency", "at": "cell"},
                    {"op": "cell_diffuse", "at": "cell", "d_a": p.get("d_a", 0.05), "d_h": p.get("d_h", 0.7), "chi": p.get("chi", 4.0)},
                    react]
            sched += ["cell_adjacency", "cell_diffuse", "cell_react"]
    else:                                                       # Brusselator RD (develops from noise -> spots emerge late)
        ops += [{"op": "cell_adjacency", "at": "cell"},
                {"op": "cell_rd_seed", "at": "cell", "seed": 0, "before_frame": 3, "mode": "noise", "A": 1.0, "B": 3.0, "noise": 0.04},
                {"op": "cell_diffuse", "at": "cell", "d_a": 0.05, "d_h": 0.7, "chi": p["chi"]},
                {"op": "cell_react", "at": "cell", "implementation": "brusselator", "gamma": p["gamma"], "A": 1.0, "B": 3.0}]
        sched += ["cell_adjacency", "cell_rd_seed", "cell_diffuse", "cell_react"]
    ga = int(p.get("grow_after", 0))                            # growth+division start only AFTER frame ga, so the
    #   RD activation pattern stabilises FIRST (GM peaks amplify) before morphogenesis acts on it
    ops += [{"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": p["rate"], "a_sw": p["a_sw"], "hill": p.get("hill", 4.0), "rho": p["rho"], "vth_frac": p["vth"], "after_frame": ga, "dt": dt, "conserve_amount": p.get("conserve_amount", True)},
            {"op": "shape_energy_3d", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05, "Lambda": 0.2, "K_V": p.get("K_V", 4.0), "K_R": 0.02, "mu": 1.0, "dt": dt, "relax_iters": p.get("relax", 30), "eta": 0.08, "cap_frac": 0.12}]
    sched += ["morphogen_growth_3d", "shape_energy_3d"]
    if p.get("K_purse", 0.0) > 0 or p.get("K_extrude", 0.0) > 0:   # RD-INTERFACE tube mechanism (purse-string + red extrusion)
        ops += [{"op": "rd_interface_tension", "at": "vertex", "cell_set": "cell", "K_purse": p.get("K_purse", 0.0), "K_extrude": p.get("K_extrude", 0.0), "a_sw": p.get("iface_asw", p["a_sw"]), "eta": p.get("iface_eta", 0.05), "iters": 4, "after_frame": ga}]
        sched += ["rd_interface_tension"]
    ops += [{"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": 300},
            {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.4, "p0": 3.90, "every": 2, "max_div": 120, "max_div_frac": p.get("mdf", 0.03), "vcap": p.get("vcap", 0.0), "cell_set": "cell", "min_cycle": 4, "max_cycle": 1000000000, "after_frame": ga},
            {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched += ["reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "tyssue_round", "seed": 0, "n_frames": p["frames"], "dt": dt, "record_cap": p["frames"] + 2, "boundary": "free", "dim": 3, "world": [16 * 5.0] * 3},
           "sets": {"vertex": {"n": VBUF}, "cell": {"n": CBUF, "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, cfg


def do(preset):
    p = PRESETS[preset]; OUT = os.path.join(HERE, "archive", preset); os.makedirs(OUT, exist_ok=True)
    sim, cfg = make(p); write_spec(cfg, os.path.join(OUT, "spec.yaml"))
    rec = {"name": preset, **p}
    try:
        Hf, out = engine_run(sim, device="cpu")
        emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
        def frame(t):
            mt = hist[min(t, len(hist) - 1)]
            return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]
        mtT, pT, aT = frame(T - 1)
        lo, hi = float(np.percentile(aT, 5)), float(np.percentile(aT, 99) + 1e-6)
        col = lambda a: np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        lbox = lambda pt: (float(np.abs(pt).max()) * 1.12, float(np.abs(pt).max()) * 2.3)   # PER-FRAME autoscale so the init (and every stage) is always visible, not a dot next to a balloon
        fig = plt.figure(figsize=(35.2, 9.0)); fig.patch.set_facecolor("black")   # 8 timepoints, 2 rows x 8 cols (3D / cross)
        for i, t in enumerate([int(round(fr * (T - 1))) for fr in np.linspace(0.0, 1.0, 8)]):
            mt, pt, a = frame(t); l3, l2 = lbox(pt)
            ax3 = fig.add_subplot(2, 8, i + 1, projection="3d"); _draw(ax3, pt, mt, 3.90, azim=30, act=col(a), Lbox=l3)
            ax2 = fig.add_subplot(2, 8, 8 + i + 1); _draw_cross(ax2, pt, mt, 3.90, act=col(a), Lbox=l2, axis=1)
        fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02); fig.savefig(os.path.join(OUT, "strip.png"), dpi=110, facecolor="black"); plt.close(fig)
        figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black"); axm, axin = make_movie_axes(figm)
        keep = np.arange(0, T, max(1, T // 110)); wri = FFMpegWriter(fps=11, metadata={"title": preset})
        with wri.saving(figm, os.path.join(OUT, "movie.mp4"), dpi=95):
            for j, t in enumerate(keep):
                mt, pt, a = frame(int(t)); l3, l2 = lbox(pt); draw_movie_frame(axm, axin, pt, mt, 3.90, (2 * j) % 360, col(a), l3, l2); wri.grab_frame()
        plt.close(figm)
        mf = []
        for t in np.unique(np.linspace(0, T - 1, 40).astype(int)):
            mt, pt, a = frame(int(t)); mf.append((int(t), pt, mt, a))
        rec.update(cells_end=int(mtT["nF"])); rec.update(analyze(mf, OUT))
        from run_tyssue_fig5 import count_spots                  # RD readout: STRONG spots only (>half max, not noise)
        rec["spots"] = int(count_spots(aT, mtT, float(0.5 * aT.max())))
        rec["act_range"] = [round(float(aT.min()), 2), round(float(aT.max()), 2)]
        print(f"[{preset}] spots={rec['spots']} act={rec['act_range']} red_frac={rec['red_frac_final']} cells->{rec['cells_end']} protr={rec['protr_final']} tube_diam={rec['tube_diam_final']} "
              f"n_tubes={rec['n_tubes_final']} hollow_pk={rec['hollow_n_peak']} area_cv_pk={rec['area_cv_peak']} red_frac={rec['red_frac_final']}", flush=True)
    except Exception as e:
        rec["error"] = repr(e); traceback.print_exc()
    json.dump(rec, open(os.path.join(OUT, "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--only":
        do(args[1])
    else:
        for k in (args or list(PRESETS)):
            do(k)
