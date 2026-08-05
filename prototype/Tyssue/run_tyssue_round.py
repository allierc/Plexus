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
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, tyssue_monolayer, ckpt  # noqa (registers the monolayer op)
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import face_geometry_3d
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross, make_movie_axes, draw_movie_frame
from tyssue_diag import hollow_flags
from tissue_analysis import analyze
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
# ===== round_26: INTERFACE mechanism (issue 3). On the now-stable confined tip-riding bud (round_25
# a0z_ga100: no flood, protr 1.16, red_over_tip 1.2, tip_act 0.8), apply the red/white-INTERFACE operator --
# purse-string ring line tension (holds a neck/diameter) + outward extrusion of red (expels the tip) -- to
# turn the small bud into a TUBE. Hypothesis: protr + tube_len rise with a ~constant tube_diam, red_over_tip
# stays ~1. Sweep K_purse (ring) x K_extrude (push); ctrl = no interface.
_H26 = dict(_F4, rd_rate=1.0, rate=0.010, spots=3, frames=350, a0=0.0, grow_after=100, iface_asw=1.2)
PRESETS["round_26_ctrl"]     = dict(_H26)                                    # no interface (control)
PRESETS["round_26_purse3"]   = dict(_H26, K_purse=3.0, K_extrude=0.3)        # strong ring
PRESETS["round_26_extr1"]    = dict(_H26, K_purse=1.0, K_extrude=1.0)        # outward extrusion
PRESETS["round_26_both2"]    = dict(_H26, K_purse=2.0, K_extrude=0.8)        # ring + extrusion
PRESETS["round_26_both3"]    = dict(_H26, K_purse=3.0, K_extrude=1.0)        # stronger both
PRESETS["round_26_purse6"]   = dict(_H26, K_purse=6.0, K_extrude=0.5)        # very strong ring (neck)
PRESETS["round_26_extr2"]    = dict(_H26, K_purse=1.0, K_extrude=2.0)        # strong extrusion
PRESETS["round_26_1sp_both"] = dict(_H26, spots=1, K_purse=3.0, K_extrude=1.0)  # single focused tube
# ===== round_27: ORIENTED DIVISION (issue 3, the real one). round_26 showed the mechanical interface op
# shapes but doesn't BUILD the wall -- new cells must be placed NORMAL to the body. New: divide_3d orient_iface
# stacks a red cell's daughters ALONG the bud axis (centre->tip), adding wall cells that EXTEND the protrusion
# instead of widening it. On the stable confined bud (a0z_ga100). Hypothesis: tube_len>0, protr rises, diameter
# ~constant, red_over_tip stays ~1. Sweep oriented-division +/- extrusion +/- more growth.
_H27 = dict(_F4, rd_rate=1.0, rate=0.010, spots=3, frames=350, a0=0.0, grow_after=100, iface_asw=1.2, orient_asw=1.2)
PRESETS["round_27_orient"]        = dict(_H27, orient_iface=True)
PRESETS["round_27_orient_extr"]   = dict(_H27, orient_iface=True, K_purse=1.0, K_extrude=1.5)
PRESETS["round_27_orient_both"]   = dict(_H27, orient_iface=True, K_purse=2.0, K_extrude=1.0)
PRESETS["round_27_orient_r015"]   = dict(_H27, orient_iface=True, rate=0.015)
PRESETS["round_27_orient_er015"]  = dict(_H27, orient_iface=True, K_extrude=1.5, rate=0.015)
PRESETS["round_27_1sp_orient"]    = dict(_H27, orient_iface=True, spots=1)
PRESETS["round_27_1sp_or_extr"]   = dict(_H27, orient_iface=True, spots=1, K_extrude=1.5)
PRESETS["round_27_orient_long"]   = dict(_H27, orient_iface=True, frames=450)
# ===== round_28: SIZE + PUSH (issue 3, cont.). round_27 gave a tiny NUB: the activated patch is sub-tube-
# sized (a point, not a chi-diameter disk) and in-plane division spreads laterally so extension needs a strong
# OUTWARD push that K_extrude<=2 couldn't supply (shape-energy relaxes it away). Fix: bigger spot (cone_deg +
# chi = tube diameter) + STRONG extrusion + oriented division. Hypothesis: a chi-sized patch pushed hard
# protrudes into a tube (protr>>1.2, tube_len>0). Decisive test of the mechanical sufficiency.
_H28 = dict(_F4, rd_rate=1.0, rate=0.010, spots=1, frames=350, a0=0.0, grow_after=100,
            orient_iface=True, orient_asw=1.2, iface_asw=1.2)
PRESETS["round_28_extr4"]        = dict(_H28, K_extrude=4.0, cone_deg=16.0)
PRESETS["round_28_extr8"]        = dict(_H28, K_extrude=8.0, cone_deg=16.0)
PRESETS["round_28_extr4_c24"]    = dict(_H28, K_extrude=4.0, cone_deg=24.0)
PRESETS["round_28_extr8_c24"]    = dict(_H28, K_extrude=8.0, cone_deg=24.0)
PRESETS["round_28_extr8_c24_p3"] = dict(_H28, K_extrude=8.0, cone_deg=24.0, K_purse=3.0)   # push + neck
PRESETS["round_28_extr12_c24"]   = dict(_H28, K_extrude=12.0, cone_deg=24.0)               # very strong push
PRESETS["round_28_extr8_chi8"]   = dict(_H28, K_extrude=8.0, cone_deg=24.0, chi=8.0)        # bigger diffusion patch
PRESETS["round_28_extr8_3sp"]    = dict(_H28, K_extrude=8.0, cone_deg=18.0, spots=3)
# ===== round_29: ELONGATE the lobe into a TUBE. round_28 made a round budding LOBE (len ~ diam, diameter
# set by chi, red at the neck). To get a tube (len >> diam), hold the diameter with a PURSE-STRING neck while
# sustaining tip growth over a LONGER run. Hypothesis: tube_len/diam rises above 1 (elongated), diameter stays
# ~chi, still confined (over_tip ~1, no flood). Base = the tube-forming recipe (extr8, cone24, oriented).
_H29 = dict(_F4, rd_rate=1.0, rate=0.010, spots=1, frames=450, a0=0.0, grow_after=100,
            orient_iface=True, orient_asw=1.2, iface_asw=1.2, K_extrude=8.0, cone_deg=24.0)
PRESETS["round_29_long"]       = dict(_H29)                                  # just longer (450)
PRESETS["round_29_purse4"]     = dict(_H29, K_purse=4.0)                     # neck to hold diameter
PRESETS["round_29_purse8"]     = dict(_H29, K_purse=8.0)                     # strong neck
PRESETS["round_29_p4_550"]     = dict(_H29, K_purse=4.0, frames=550)         # neck + very long
PRESETS["round_29_p8_extr12"]  = dict(_H29, K_purse=8.0, K_extrude=12.0)     # strong neck + strong push
PRESETS["round_29_p6_r012"]    = dict(_H29, K_purse=6.0, rate=0.012)         # neck + a bit more growth
PRESETS["round_29_p4_chi6"]    = dict(_H29, K_purse=4.0, chi=6.0)            # medium diameter tube
PRESETS["round_29_p8_550_r012"]= dict(_H29, K_purse=8.0, frames=550, rate=0.012)  # all-in: elongate hard
# ===== round_30: CHI -> DIAMETER law (capstone). The tube is a transient peaking ~frame 350 (round_29 showed
# longer degrades it), so freeze at 350 and cleanly SWEEP chi (Okuda's diffusion coeff = tube diameter control,
# diam ~ chi^1/4). round_28 hinted chi4->1.9, chi8->3.3. Hypothesis: tube_diam rises monotonically with chi at
# fixed everything else -- the validated Okuda diameter law on our pipeline. Base = round_28 tube recipe.
_H30 = dict(_F4, rd_rate=1.0, rate=0.010, spots=1, frames=350, a0=0.0, grow_after=100,
            orient_iface=True, orient_asw=1.2, iface_asw=1.2, K_extrude=8.0, cone_deg=24.0)
PRESETS["round_30_chi2"]  = dict(_H30, chi=2.0)
PRESETS["round_30_chi3"]  = dict(_H30, chi=3.0)
PRESETS["round_30_chi4"]  = dict(_H30, chi=4.0)
PRESETS["round_30_chi6"]  = dict(_H30, chi=6.0)
PRESETS["round_30_chi8"]  = dict(_H30, chi=8.0)
PRESETS["round_30_chi10"] = dict(_H30, chi=10.0)
PRESETS["round_30_chi14"] = dict(_H30, chi=14.0)
PRESETS["round_30_chi18"] = dict(_H30, chi=18.0)
# ===== round_31: SMALLER red spot (user hypothesis). round_28-30 tube is a fat budding LOBE (diam~chi, big
# cone). A smaller activated patch (small cone_deg + small chi) should give a NARROWER finger and, with less
# activated material, slower over-proliferation -> the confinement (and the tube) may last longer. Base = the
# tube recipe (extr8, oriented, a0=0). Judge tube_len/diam (elongated?), diam (smaller?), red_frac (no flood).
_H31 = dict(_F4, rd_rate=1.0, rate=0.010, spots=1, frames=350, a0=0.0, grow_after=100,
            orient_iface=True, orient_asw=1.2, iface_asw=1.2, K_extrude=8.0)
PRESETS["round_31_c6"]        = dict(_H31, cone_deg=6.0,  chi=4.0)
PRESETS["round_31_c8"]        = dict(_H31, cone_deg=8.0,  chi=4.0)
PRESETS["round_31_c10"]       = dict(_H31, cone_deg=10.0, chi=4.0)
PRESETS["round_31_c6_chi2"]   = dict(_H31, cone_deg=6.0,  chi=2.0)
PRESETS["round_31_c8_chi2"]   = dict(_H31, cone_deg=8.0,  chi=2.0)
PRESETS["round_31_c8_chi3"]   = dict(_H31, cone_deg=8.0,  chi=3.0)
PRESETS["round_31_c6_extr12"] = dict(_H31, cone_deg=6.0,  chi=2.0, K_extrude=12.0)   # small spot + strong push
PRESETS["round_31_c10_chi3"]  = dict(_H31, cone_deg=10.0, chi=3.0)
# ===== round_32: 2x FRAMES on the THIN tube (user). round_31 c8 (cone8,chi4) gave a thin elongated finger
# (len/diam~2) with LESS proliferation (1800 cells) than the fat lobe. Hypothesis: less activated material ->
# slower domain growth -> the thin tube SUSTAINS + ELONGATES over a longer run (unlike round_29's fat lobe that
# was lost by frame 350). Double frames -> 700-900. seed_dir aims the single tube at the FRONT of the camera
# (elev18/azim30 -> ~(.82,.48,.31)) so it faces the viewer, not the left.
_FRONT = [0.82, 0.48, 0.31]
_H32 = dict(_F4, rd_rate=1.0, rate=0.010, spots=1, a0=0.0, grow_after=100, orient_iface=True, orient_asw=1.2,
            iface_asw=1.2, K_extrude=8.0, cone_deg=8.0, chi=4.0, seed_dir=_FRONT)
PRESETS["round_32_c8_700"]      = dict(_H32, frames=700)
PRESETS["round_32_c8_900"]      = dict(_H32, frames=900)
PRESETS["round_32_c10c3_700"]   = dict(_H32, frames=700, cone_deg=10.0, chi=3.0)
PRESETS["round_32_c8_700_p3"]   = dict(_H32, frames=700, K_purse=3.0)                  # neck to keep it thin
PRESETS["round_32_c8_700_ex12"] = dict(_H32, frames=700, K_extrude=12.0)               # stronger push
PRESETS["round_32_c8_700_r012"] = dict(_H32, frames=700, rate=0.012)                   # a bit more growth
PRESETS["round_32_c8_700_ga150"]= dict(_H32, frames=700, grow_after=150)               # settle longer first
PRESETS["round_32_c8_900_p3"]   = dict(_H32, frames=900, K_purse=3.0)
# ===== round_33: TIP-TRACKING seed (sustained elongation). round_32 showed the tube CAPS (~protr 1.4) because
# the activation is a widening PATCH, not a moving tip cap. New seed_mode="tip": re-activate a fixed-SIZE cap
# (tip_radius) at the current OUTERMOST cell each frame -> the cap RIDES the advancing tip -> constant-diameter
# EXTENSION. No RD (the tip cap IS the driver; issues 1-2 don't apply). Hypothesis: tube_len grows with frames,
# diam ~ tip_radius, len/diam >> 1 (a real long tube, not a lobe). Sweep tip_radius (= tube diameter).
_H33 = dict(spots=1, cone_deg=8.0, seed_mode="tip", seed_dir=_FRONT, frames=500, grow_after=20,
            rate=0.010, a_sw=0.5, hill=4.0, rho=0.0, vth=1.5, K_V=4.0, mdf=0.03, relax=30, vcap=1.5,
            orient_iface=True, orient_asw=0.5, iface_asw=0.5, K_extrude=8.0)
PRESETS["round_33_tr10"]      = dict(_H33, tip_radius=1.0)
PRESETS["round_33_tr15"]      = dict(_H33, tip_radius=1.5)
PRESETS["round_33_tr20"]      = dict(_H33, tip_radius=2.0)
PRESETS["round_33_tr25"]      = dict(_H33, tip_radius=2.5)
PRESETS["round_33_tr15_ex12"] = dict(_H33, tip_radius=1.5, K_extrude=12.0)
PRESETS["round_33_tr15_r015"] = dict(_H33, tip_radius=1.5, rate=0.015)
PRESETS["round_33_tr15_long"] = dict(_H33, tip_radius=1.5, frames=700)
PRESETS["round_33_tr10_ex12"] = dict(_H33, tip_radius=1.0, K_extrude=12.0)   # thinnest + strong push
# ===== round_34: THIN tube, ROUNDED tip, LEFT (user). round_33 tip-tracking gave long tubes (len/diam 2.5-4);
# user wants smaller (c6-like thin), a ROUNDED tip pushed outward (now flat) via strong extrusion, MORE frames,
# and the spot on the LEFT (profile view -- _FRONT grew away from the camera into the far face). seed_dir=_LEFT.
_LEFT = [0.4, -0.9, 0.15]
_H34 = dict(_H33, seed_dir=_LEFT, tip_radius=1.2, K_extrude=12.0)
PRESETS["round_34_700"]        = dict(_H34, frames=700)
PRESETS["round_34_900"]        = dict(_H34, frames=900)
PRESETS["round_34_ex16"]       = dict(_H34, frames=700, K_extrude=16.0)                # stronger tip push (rounder)
PRESETS["round_34_r015"]       = dict(_H34, frames=700, rate=0.015)                    # more growth
PRESETS["round_34_tr15_900"]   = dict(_H34, frames=900, tip_radius=1.5)
PRESETS["round_34_tr10_ex16"]  = dict(_H34, frames=700, tip_radius=1.0, K_extrude=16.0)  # thinnest + strongest push
PRESETS["round_34_r015_900"]   = dict(_H34, frames=900, rate=0.015)
PRESETS["round_34_ex16_r015"]  = dict(_H34, frames=900, K_extrude=16.0, rate=0.015)    # long + push + growth
# ===== round_35: UNIFORM CELLS -> no hollow (user's Fig-5 clue). Diagnosis of round_34: rho=0 uses the legacy
# INFLATION path (tip cells balloon to 2.5x v_ref -> non-uniform, CV 0.5, hollow 69, a BULGE not a tube wall).
# Okuda Fig 5 has LOW CV: activator sets growth RATE (not size), cells cap at vth*v_ref and DIVIDE -> uniform.
# Hypothesis: rho>0 (uniform mode) + tight cycle_cv + bounded max_cycle + stiff K_V drop CV & hollow while the
# tip PROLIFERATES into the tube. Control rho=0 (old inflation) for the A/B. Metric: area_cv, hollow_n_peak.
_H35 = dict(spots=1, cone_deg=8.0, seed_mode="tip", seed_dir=_LEFT, tip_radius=1.5, frames=500, grow_after=20,
            rate=0.02, a_sw=0.5, hill=4.0, rho=0.1, vth=1.35, K_V=4.0, mdf=0.03, relax=30, vcap=1.5,
            cycle_cv=0.15, min_cycle=4, max_cycle=12, orient_iface=True, orient_asw=0.5, iface_asw=0.5, K_extrude=8.0)
PRESETS["round_35_rho0"]    = dict(_H35, rho=0.0, vth=2.5)                    # CONTROL: old inflation mode
PRESETS["round_35_rho01"]   = dict(_H35, rho=0.1)                            # uniform mode
PRESETS["round_35_rho005"]  = dict(_H35, rho=0.05)                          # tip more dominant vs body
PRESETS["round_35_rho02"]   = dict(_H35, rho=0.2)
PRESETS["round_35_cv10"]    = dict(_H35, rho=0.1, cycle_cv=0.10)            # tighter cell-cycle timing
PRESETS["round_35_kv6"]     = dict(_H35, rho=0.1, K_V=6.0)                  # stiffer volume -> less buckle
PRESETS["round_35_vth125"]  = dict(_H35, rho=0.1, vth=1.25)                 # tighter size cap
PRESETS["round_35_relax45"] = dict(_H35, rho=0.1, relax=45)                 # more force-balance relaxation
# ===== round_36: CONTROL PROLIFERATION (goals 3,4). round_35: uniform mode crushed CV (4.6->0.5) but ALL hit
# the 15002 cell buffer (10x growth vs Okuda's ~3x) -> crammed cells buckle -> hollow. So hollow is driven by
# OVER-PROLIFERATION, not CV. Fix: bring cell count to Okuda scale (~4000) -- slower/shorter growth, vcap OFF
# (no forced oversize division), smaller tip cap, lower division rate. Hypothesis: ~4000 cells -> hollow<<,
# CV stays low (uniform mode). Metric: cells_end ~4000, hollow_n_peak low.
_H36 = dict(_H35, rho=0.1, vcap=0.0, tip_radius=1.0, mdf=0.015)
PRESETS["round_36_base"]     = dict(_H36, rate=0.008, frames=300)
PRESETS["round_36_slow"]     = dict(_H36, rate=0.005, frames=400)
PRESETS["round_36_f250"]     = dict(_H36, rate=0.008, frames=250)
PRESETS["round_36_tr08"]     = dict(_H36, rate=0.008, frames=300, tip_radius=0.8)
PRESETS["round_36_rho005"]   = dict(_H36, rate=0.008, frames=300, rho=0.05)
PRESETS["round_36_maxcyc30"] = dict(_H36, rate=0.008, frames=300, max_cycle=30)
PRESETS["round_36_mdf008"]   = dict(_H36, rate=0.008, frames=300, mdf=0.008)
PRESETS["round_36_r006"]     = dict(_H36, rate=0.006, frames=400)
# ===== round_37: SMALL steps from the WORKING tube (round_34_900: aspect 8.4, ~2800 cells, rho=0 tip-mode).
# round_35/36 (rho>0 uniform + controlled prolif) LOST the tube -- too big a step. Back to the working point;
# change ONE anti-buckle lever at a time to cut HOLLOW (goal 3) without losing elongation. Hollow comes from
# oversized inflating tip cells buckling; levers: vcap (force-divide sooner), K_V (stiffer volume), relax (more
# force balance). Keep len/diam high. Base = round_34_900.
_H37 = dict(_H34, frames=900)
PRESETS["round_37_base"]      = dict(_H37)                                   # control = the working tube
PRESETS["round_37_vcap13"]    = dict(_H37, vcap=1.3)
PRESETS["round_37_vcap12"]    = dict(_H37, vcap=1.2)
PRESETS["round_37_kv6"]       = dict(_H37, K_V=6.0)
PRESETS["round_37_kv8"]       = dict(_H37, K_V=8.0)
PRESETS["round_37_relax45"]   = dict(_H37, relax=45)
PRESETS["round_37_vcap13_kv6"]= dict(_H37, vcap=1.3, K_V=6.0)
PRESETS["round_37_relax45_kv6"]=dict(_H37, relax=45, K_V=6.0)
# ===== round_38: HOLLOW down (goal 3), one lever from the kv6 working point (aspect 10, CV 0.97, hollow 390).
# Hollow = cell-fold buckling at the tube; DIHEDRAL BENDING (K_bend, Wardetzky hinge) penalises adjacent-cell
# normal deviation -> smooths the folds WITHOUT flattening the tube's gentle curvature; anti-inversion filtered
# step (antiinv) blocks a substep that drives a face toward inversion. Keep len/diam ~10, CV low. Base = kv6.
_H38 = dict(_H37, K_V=6.0)
PRESETS["round_38_base"]     = dict(_H38)                                    # kv6 control
PRESETS["round_38_bend1"]    = dict(_H38, K_bend=1.0)
PRESETS["round_38_bend3"]    = dict(_H38, K_bend=3.0)
PRESETS["round_38_bend6"]    = dict(_H38, K_bend=6.0)
PRESETS["round_38_bend10"]   = dict(_H38, K_bend=10.0)
PRESETS["round_38_ai02"]     = dict(_H38, antiinv=0.2)
PRESETS["round_38_ai04"]     = dict(_H38, antiinv=0.4)
PRESETS["round_38_bend3_ai02"]=dict(_H38, K_bend=3.0, antiinv=0.2)
# ===== round_39: SLOWER GROWTH so cells REARRANGE (user hypothesis). The analysis showed pressure~3.1 at the
# tube = explosive, unsustainable growth: tip cells inflate/divide faster than T1 can rearrange them -> they
# buckle -> hollow (systematically >300). Fix: grow SLOWER (lower rate) so per unit growth there is more T1
# flow -> cells arrange along the tube, stress transmits tube->body. Slower+longer keeps the tube length.
# One primary lever (rate down), + T1 rearrangement (l_th_frac up, more flips). Base = kv6 (aspect 10).
_H39 = dict(_H38, K_V=6.0)
PRESETS["round_39_base"]      = dict(_H39)                                   # kv6 control (rate 0.01, 900f)
PRESETS["round_39_r006"]      = dict(_H39, rate=0.006)                       # slower (shorter tube -- isolate rate effect)
PRESETS["round_39_r004"]      = dict(_H39, rate=0.004)
PRESETS["round_39_r006_f1500"]= dict(_H39, rate=0.006, frames=1500)         # slower + longer -> keep tube length
PRESETS["round_39_r004_f2000"]= dict(_H39, rate=0.004, frames=2000)
PRESETS["round_39_r006_t1"]   = dict(_H39, rate=0.006, frames=1500, l_th_frac=0.36, max_flips=500)  # + more T1 rearrangement
PRESETS["round_39_r005_f1600"]= dict(_H39, rate=0.005, frames=1600)
PRESETS["round_39_r006_relax"]= dict(_H39, rate=0.006, frames=1500, relax=45)
# ===== round_40: LOW STRESS + fewer tiny cells. Analysis: tube pressure~3.1 (explosive) + the real "hollow"
# is fresh tip DAUGHTERS (tiny, area<local) from rapid division. Levers: gentler extrusion (lower K_extrude ->
# less compressive pressure), g1_ramp (daughters born AT their actual volume -> no tiny-target mismatch),
# stagger division (min_cycle). Small steps from kv6. Analyse the winner's stress/force next.
_H40 = dict(_H39, K_V=6.0)
PRESETS["round_40_base"]      = dict(_H40)                                   # kv6 control (K_extrude 12)
PRESETS["round_40_ex8"]       = dict(_H40, K_extrude=8.0)                    # gentler extraction
PRESETS["round_40_ex6"]       = dict(_H40, K_extrude=6.0)
PRESETS["round_40_g1"]        = dict(_H40, g1_ramp=True)                     # daughters born at target volume
PRESETS["round_40_ex8_g1"]    = dict(_H40, K_extrude=8.0, g1_ramp=True)
PRESETS["round_40_mc8"]       = dict(_H40, min_cycle=8)                      # stagger division
PRESETS["round_40_ex8_mc8"]   = dict(_H40, K_extrude=8.0, min_cycle=8)
PRESETS["round_40_ex8_g1_mc8"]= dict(_H40, K_extrude=8.0, g1_ramp=True, min_cycle=8)
# ===== round_41: QUASI-STATIC push (Okuda's regime). R40 showed staggering the cell cycle (min_cycle=8) cuts
# hollow+CV -- a step toward tau_cycle >> eta/kappa. Push further: MORE relaxation per frame (approximate
# relax-to-residual) + slower cycle + Hertwig long-axis division (Okuda's rule -- in a quasi-static tube the
# wall cells elongate along the tube so a plain long-axis split already extends it; the forced bud-axis orient
# may be unnecessary). Base = mc8 working point (hollow 176, CV 0.63, aspect 7.5). Analyse the winner's stress.
_H41 = dict(_H40, min_cycle=8)
PRESETS["round_41_base"]          = dict(_H41)                               # mc8 control
PRESETS["round_41_relax60"]       = dict(_H41, relax=60)
PRESETS["round_41_relax90"]       = dict(_H41, relax=90)
PRESETS["round_41_mc12"]          = dict(_H41, min_cycle=12)
PRESETS["round_41_mc12_relax60"]  = dict(_H41, min_cycle=12, relax=60)
PRESETS["round_41_hertwig"]       = dict(_H41, orient_iface=False)           # Okuda long-axis division
PRESETS["round_41_hertwig_relax60"]=dict(_H41, orient_iface=False, relax=60)
PRESETS["round_41_mc12_hertwig"]  = dict(_H41, min_cycle=12, orient_iface=False)
# ===== round_42: MONOLAYER (apical/basal) tube -- GROWTH-DRIVEN, the Okuda quasi-static mechanism. R41 showed
# our extrusion tube is not an equilibrium (relaxation removes it -> high stress). The monolayer op gives each
# cell a true 3D volume + emergent bending, so localized volume growth on the shell should BUCKLE into a tube
# as its EQUILIBRIUM (low stress). Low kappa_s buckles (from the demos). Sweep kappa_s x extrusion; analyse the
# pressure (should be << the extrusion tube's ~3). Base = mc8 working point, monolayer on.
_H42 = dict(_H41, monolayer=True, mono_kv=6.0, h0=0.4, frames=700, min_cycle=8)
PRESETS["round_42_k20"]       = dict(_H42, kappa_s=0.2, K_extrude=0.0)       # pure growth, standard tension
PRESETS["round_42_k05"]       = dict(_H42, kappa_s=0.05, K_extrude=0.0)      # low tension -> buckle (growth only)
PRESETS["round_42_k05_ex4"]   = dict(_H42, kappa_s=0.05, K_extrude=4.0)      # + gentle extrusion assist
PRESETS["round_42_k05_ex8"]   = dict(_H42, kappa_s=0.05, K_extrude=8.0)
PRESETS["round_42_k10_ex4"]   = dict(_H42, kappa_s=0.10, K_extrude=4.0)
PRESETS["round_42_k05_kv4"]   = dict(_H42, kappa_s=0.05, mono_kv=4.0, K_extrude=4.0)
PRESETS["round_42_k05_h03"]   = dict(_H42, kappa_s=0.05, h0=0.3, K_extrude=4.0)
PRESETS["round_42_k05_ex8_r02"]=dict(_H42, kappa_s=0.05, K_extrude=8.0, rate=0.02)
# ===== round_43: THICKER tube (Okuda Fig-5 scale) for low hollow+CV. R42 monolayer made thin SPIKES (protr
# 70, hollow>600, CV>5). Back to mid-surface mc8 (cleanest: hollow 176, CV 0.63) but make the tube THICKER
# (bigger tip_radius -> bigger, more uniform wall cells -> lower CV + fewer curvature/tiny false-positive
# hollow). Okuda's tubes are moderately thick, not super-thin. Sweep tip_radius up. Analyse the winner.
_H43 = dict(_H41, min_cycle=8)
PRESETS["round_43_base"]      = dict(_H43)                                   # tip_radius 1.5 control
PRESETS["round_43_tr20"]      = dict(_H43, tip_radius=2.0)
PRESETS["round_43_tr25"]      = dict(_H43, tip_radius=2.5)
PRESETS["round_43_tr30"]      = dict(_H43, tip_radius=3.0)
PRESETS["round_43_tr20_ex8"]  = dict(_H43, tip_radius=2.0, K_extrude=8.0)
PRESETS["round_43_tr25_ex8"]  = dict(_H43, tip_radius=2.5, K_extrude=8.0)
PRESETS["round_43_tr20_mc12"] = dict(_H43, tip_radius=2.0, min_cycle=12)
PRESETS["round_43_tr25_relax45"]=dict(_H43, tip_radius=2.5, relax=45)
# ===== round_44: RD ROLE (goal 2). Best tube (mc8) uses a tip DRIVER (no RD). Bring the emergent Gierer-
# Meinhardt RD back (a0=0 confinement + amount-conservation from R25) to PARTITION the tissue, coupled to the
# same wall-building machinery (K_V=6, min_cycle=8, oriented division, extrusion). Does emergent RD + the
# machinery make a tube (vs the R22-30 capped bud)? RD gets a genuine role (the red/white partition drives
# growth). Base = the RD confinement recipe + mc8 mechanics, seeded on the left.
_H44 = dict(_F4, rd_rate=1.0, rate=0.010, spots=1, frames=700, a0=0.0, grow_after=80, orient_iface=True,
            orient_asw=1.2, iface_asw=1.2, K_extrude=8.0, cone_deg=20.0, K_V=6.0, min_cycle=8, mdf=0.03,
            vcap=1.5, seed_dir=_LEFT)
PRESETS["round_44_base"]       = dict(_H44)
PRESETS["round_44_ex12"]       = dict(_H44, K_extrude=12.0)
PRESETS["round_44_rd2"]        = dict(_H44, rd_rate=2.0)
PRESETS["round_44_cone16"]     = dict(_H44, cone_deg=16.0)
PRESETS["round_44_ex12_cone16"]= dict(_H44, K_extrude=12.0, cone_deg=16.0)
PRESETS["round_44_r015"]       = dict(_H44, rate=0.015)
PRESETS["round_44_chi6"]       = dict(_H44, chi=6.0)
PRESETS["round_44_ex12_r015"]  = dict(_H44, K_extrude=12.0, rate=0.015)
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
        smode = p.get("seed_mode", "cones")                     # "cones" (fixed-angle) | "tip" (fixed-size, tip-tracking)
        seed = {} if smode == "tip" else ({"before_frame": 3} if (rd or p.get("seed_once")) else {})   # tip re-seeds EVERY frame
        ops += [{"op": "cell_rd_seed", "at": "cell", "mode": smode, "n_spots": p["spots"], "cone_deg": p["cone_deg"], "seed_dir": p.get("seed_dir", None), "tip_radius": p.get("tip_radius", 2.0), **seed}]
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
    if p.get("monolayer"):                                     # MONOLAYER (apical/basal) energy: growth-driven,
        se = {"op": "shape_energy_3d", "implementation": "monolayer", "at": "vertex",   # quasi-static tube (Okuda)
              "k_v": p.get("mono_kv", 6.0), "kappa_s": p.get("kappa_s", 0.2), "h0": p.get("h0", 0.4),
              "gamma": p.get("mono_gamma", 0.06), "mu": 1.0, "dt": dt, "relax_iters": p.get("relax", 30),
              "eta": 0.08, "cap_frac": 0.12}
    else:
        se = {"op": "shape_energy_3d", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05, "Lambda": 0.2, "K_V": p.get("K_V", 4.0), "K_R": 0.02, "K_bend": p.get("K_bend", 0.0), "antiinv": p.get("antiinv", 0.0), "mu": 1.0, "dt": dt, "relax_iters": p.get("relax", 30), "eta": 0.08, "cap_frac": 0.12}
    ops += [{"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": p["rate"], "a_sw": p["a_sw"], "hill": p.get("hill", 4.0), "rho": p["rho"], "vth_frac": p["vth"], "after_frame": ga, "dt": dt, "conserve_amount": p.get("conserve_amount", True)}, se]
    sched += ["morphogen_growth_3d", "shape_energy_3d"]
    if p.get("K_purse", 0.0) > 0 or p.get("K_extrude", 0.0) > 0:   # RD-INTERFACE tube mechanism (purse-string + red extrusion)
        ops += [{"op": "rd_interface_tension", "at": "vertex", "cell_set": "cell", "K_purse": p.get("K_purse", 0.0), "K_extrude": p.get("K_extrude", 0.0), "a_sw": p.get("iface_asw", p["a_sw"]), "eta": p.get("iface_eta", 0.05), "iters": 4, "after_frame": ga}]
        sched += ["rd_interface_tension"]
    ops += [{"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": p.get("l_th_frac", 0.28), "every": p.get("t1_every", 1), "max_flips": p.get("max_flips", 300)},
            {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": p.get("cycle_cv", 0.4), "p0": 3.90, "every": 2, "max_div": 120, "max_div_frac": p.get("mdf", 0.03), "vcap": p.get("vcap", 0.0), "cell_set": "cell", "min_cycle": p.get("min_cycle", 4), "max_cycle": p.get("max_cycle", 1000000000), "after_frame": ga, "orient_iface": p.get("orient_iface", False), "orient_asw": p.get("orient_asw", p.get("a_sw", 1.0)), "g1_ramp": p.get("g1_ramp", False)},
            {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched += ["reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "tyssue_round", "seed": 0, "n_frames": p["frames"], "dt": dt, "record_cap": p["frames"] + 2, "boundary": "free", "dim": 3, "world": [16 * 5.0] * 3},
           "sets": {"vertex": {"n": VBUF}, "cell": {"n": CBUF, "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, cfg


_ELEV, _AZIM = 18, 30                                             # the render camera (matches _draw's view_init)


def _screen_basis(elev=_ELEV, azim=_AZIM):
    """Screen frame for the render camera: d = depth (into screen), u = horizontal (right), v = vertical (up)."""
    er, az = np.deg2rad(elev), np.deg2rad(azim)
    d = np.array([np.cos(er) * np.cos(az), np.cos(er) * np.sin(az), np.sin(er)])
    v = np.array([0.0, 0.0, 1.0]) - (np.array([0.0, 0.0, 1.0]) @ d) * d; v /= (np.linalg.norm(v) + 1e-12)
    u = np.cross(v, d); u /= (np.linalg.norm(u) + 1e-12)
    return d, u, v


def _tube_axis(pos, act=None, frac=93.0):
    """The ACTUAL tube axis = direction to the outermost region (the tip), computed from geometry each
    frame (the tip-tracking tip drifts off the seed direction, so slicing by seed_dir misses the tube).
    Uses the mean of the farthest `frac` percentile vertices; returns a unit vector (or None if no spread)."""
    r = np.linalg.norm(pos, axis=1)
    tip = pos[r > np.percentile(r, frac)]
    if len(tip) < 3:
        return None
    ax = tip.mean(0); n = np.linalg.norm(ax)
    return ax / n if n > 1e-6 else None


def _cross_axis(pos, seed_dir):
    """Actual tube axis if there is a protrusion, else the seed direction (explicit None check -- a numpy
    array can't be used in `a or b`)."""
    t = _tube_axis(pos)
    return t if t is not None else seed_dir


def _cross_screen(ax, pos, mesh, act, seed_dir=None, inner=0.82, Lbox=None):
    """Cross-section in the PLANE OF THE TUBING: the slice plane always CONTAINS the tube axis (seed_dir),
    oriented face-on to the camera, so the tube's full length shows in profile along the horizontal,
    coloured by activation (white->red). Falls back to the screen plane if no tube axis. Edges crossing the
    mid-plane give the lumen ring; each segment is coloured by its cell's activator."""
    from matplotlib.patches import Polygon as MplPoly
    ax.clear(); ax.set_facecolor("black")
    dcam, su, sv = _screen_basis()
    if seed_dir is not None:                                     # plane contains the tube axis, ~face-on to camera
        t = np.asarray(seed_dir, float); t = t / (np.linalg.norm(t) + 1e-12)   # tube axis -> plot HORIZONTAL
        n = dcam - (dcam @ t) * t; nn = np.linalg.norm(n)        # slice normal: camera depth projected off the axis
        d = n / nn if nn > 1e-6 else sv
        u = t if (t @ su) >= 0 else -t                           # match the 3D screen L/R (no flip vs the 3D view)
        v = np.cross(d, u); v = v / (np.linalg.norm(v) + 1e-12)  # in-plane VERTICAL
        if v @ sv < 0:
            v = -v                                               # match the 3D screen up/down
    else:
        d, u, v = dcam, su, sv
    es, et, ef = np.asarray(mesh["E_srce"]), np.asarray(mesh["E_trgt"]), np.asarray(mesh["E_face"])
    proj = pos @ d                                                # signed distance of each vertex to the screen mid-plane
    aps, bas, cols = [], [], []
    for e in range(len(es)):
        s, t = int(es[e]), int(et[e]); fa, fb = proj[s], proj[t]
        if fa * fb < 0:                                          # edge crosses the plane
            fr = -fa / (fb - fa); X = pos[s] + fr * (pos[t] - pos[s])
            aps.append([X @ u, X @ v]); bas.append([(X * inner) @ u, (X * inner) @ v])
            # NaN GETS ITS OWN COLOUR HERE TOO. col() upstream deliberately preserves NaN --
            # its comment says "the renderer paints it grey rather than silently white" -- and
            # this renderer then passed it to a colormap, which returns the transparent "bad"
            # value and draws nothing on black. The cross-section row of a diverged run was
            # therefore EMPTY, which reads as a missing panel rather than a dead field.
            _av = float(act[int(ef[e])])
            cols.append((1.00, 0.10, 0.85, 1.0) if not np.isfinite(_av)
                        else plt.cm.Reds(float(np.clip(_av, 0, 1))))
    if aps:
        aps = np.array(aps); bas = np.array(bas); c = aps.mean(0)
        order = np.argsort(np.arctan2(aps[:, 1] - c[1], aps[:, 0] - c[0]))
        aps, bas = aps[order], bas[order]; cols = [cols[i] for i in order]
        for i in range(len(aps)):
            j = (i + 1) % len(aps)
            ax.add_patch(MplPoly(np.array([bas[i], aps[i], aps[j], bas[j]]), closed=True,
                                 facecolor=cols[i], edgecolor="black", lw=0.4, zorder=1))
    L = Lbox if Lbox is not None else 11.0
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.axis("off")


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
        asamp = np.concatenate([frame(int(t))[2] for t in np.unique(np.linspace(0, T - 1, 12).astype(int))])
        lo, hi = float(np.percentile(asamp, 5)), float(np.percentile(asamp, 99) + 1e-6)   # GLOBAL scale over
        #   sampled frames -> a degenerate final frame no longer washes the whole strip red
        col = lambda a: np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        lbox = lambda pt: (float(np.abs(pt).max()) * 1.12, float(np.abs(pt).max()) * 2.3)   # PER-FRAME autoscale so the init (and every stage) is always visible, not a dot next to a balloon
        fig = plt.figure(figsize=(35.2, 9.0)); fig.patch.set_facecolor("black")   # 8 timepoints, 2 rows x 8 cols (3D / cross)
        for i, t in enumerate([int(round(fr * (T - 1))) for fr in np.linspace(0.0, 1.0, 8)]):
            mt, pt, a = frame(t); l3, l2 = lbox(pt)
            ax3 = fig.add_subplot(2, 8, i + 1, projection="3d"); _draw(ax3, pt, mt, 3.90, azim=_AZIM, act=col(a), Lbox=l3)
            ax2 = fig.add_subplot(2, 8, 8 + i + 1); _cross_screen(ax2, pt, mt, col(a), seed_dir=_cross_axis(pt, p.get("seed_dir")), Lbox=l2)
        fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02); fig.savefig(os.path.join(OUT, "strip.png"), dpi=110, facecolor="black"); plt.close(fig)
        figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black"); axm, axin = make_movie_axes(figm)
        keep = np.arange(0, T, max(1, T // 72)); wri = FFMpegWriter(fps=10, metadata={"title": preset})   # fewer frames -> faster render (matplotlib 3D is slow); not the critical path
        with wri.saving(figm, os.path.join(OUT, "movie.mp4"), dpi=85):
            for j, t in enumerate(keep):
                mt, pt, a = frame(int(t)); l3, l2 = lbox(pt)     # FIXED camera (no spin) -> tube stays in profile
                _draw(axm, pt, mt, 3.90, azim=_AZIM, act=col(a), Lbox=l3); _cross_screen(axin, pt, mt, col(a), seed_dir=_cross_axis(pt, p.get("seed_dir")), Lbox=l2)
                wri.grab_frame()
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
