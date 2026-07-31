#!/usr/bin/env python
"""premise_check -- run the tissue premises against a spec and a run, and fail when one is broken.

WHY THIS PHASE EXISTS
================================================================================================
Ten defects were found on 2026-07-31, and every single one was found by Cedric looking at a
picture. Not one was found by the loop. They were all the same KIND of defect: an operator or an
instrument not doing what its name says. The loop checked that runs completed, that the mesh was
valid and that the metrics were finite -- it never once checked that the thing it simulated was a
tissue.

`discovery/PREMISES.md` wrote the biology down. This file makes it EXECUTABLE. A premise nobody
can run is an opinion, and an opinion catches nothing.

THREE TIERS, because the premises are not all answerable at the same cost
------------------------------------------------------------------------------------------------
  STATIC   read the spec. No simulation. Runs BEFORE the GPU is touched, so a composition that
           cannot possibly work is rejected in milliseconds rather than after ten minutes.
           Catches D5b (growth ceiling below the division trigger) and D5a (chemistry on the
           mechanics clock) at zero cost.
  PASSIVE  read the run's own recorded series. Runs after every simulation, always.
           Catches D10 (dilution extinguishing the chemistry), D9, and the shape-index floor.
  PROBE    a dedicated short simulation of its own, cached by composition. The most expensive and
           the most valuable: premise 6 -- a resting vesicle rests -- is 40 frames of mechanics
           with everything else off, and it would have caught the vesicle collapse in seconds on
           the first day instead of on day N by eye.

VIOLATIONS MUST BE DECLARED, NOT SILENCED
------------------------------------------------------------------------------------------------
A deliberate ablation legitimately breaks a premise: setting rho = 0 to ask what a protrusion is
made of genuinely violates premise 1. That is science, not a bug. The difference between an
ablation and a defect is entirely whether someone MEANT it -- so a spec may waive a premise, in
writing, with a reason:

    _premises:
      waive:
        P1: "deliberate ablation: rho=0 asks whether a tube can be drawn from existing material"

An undeclared violation is a hard failure. A declared one is recorded as an ablation and carried
into the run record, where the Analyst reads it as the run's stated intent. This is the mechanism
that stops "the parameters were wrong" from being indistinguishable from "we tested a hypothesis".

EVERY CHECK IS CERTIFIED AGAINST A KNOWN-GOOD AND A KNOWN-BAD CASE
------------------------------------------------------------------------------------------------
A check that has never been seen to fail is not a check. `--certify` runs each one against real
runs from this campaign whose answer we already know -- mini_coral (chemistry dead) against
mini_coral_nodilute (alive), mini_grow_divide (ceiling below trigger) against
mini_grow_divide_bigger (above). Same discipline metric_author applies to metrics.

    python premise_check.py --certify                 # prove the checks catch what they claim
    python premise_check.py <run_name>                # static + passive on a finished run
    python premise_check.py <run_name> --probe        # ... and run the probes too
    python premise_check.py --spec <config_name>      # static only, before running anything
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "prototype", "Tyssue"))

CONFIG_DIR = os.path.join(ROOT, "config", "okuda")
LOG_DIR = os.path.join(ROOT, "log", "okuda")

SHAPE_INDEX_FLOOR = 2.0 * np.sqrt(np.pi)      # 3.5449 -- a circle. Geometry, not biology.


# --------------------------------------------------------------------------- result plumbing
class R:
    """One premise's verdict. `na` when the premise does not apply to this composition -- which is
    itself information: a run with no chemistry cannot violate the chemistry premises, and saying
    so is honest, whereas silently passing is not."""

    def __init__(self, pid, tier, premise, status, detail, measured=None):
        self.pid, self.tier, self.premise = pid, tier, premise
        self.status, self.detail, self.measured = status, detail, measured

    def __repr__(self):
        return f"{self.pid} {self.status}: {self.detail}"

    def as_dict(self):
        return dict(id=self.pid, tier=self.tier, premise=self.premise, status=self.status,
                    detail=self.detail, measured=self.measured)


def _ops(cfg):
    return {o["op"]: o for o in cfg.get("operators", [])}


def _series(run):
    p = os.path.join(LOG_DIR, run, "metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("series") or None


def _col(series, key):
    a = np.array([e.get(key, np.nan) for e in series], dtype=float)
    return a if np.isfinite(a).any() else None


# ================================================================= STATIC (spec only, no run)
def p2_gate_implies_baseline(cfg):
    """#2 A morphogen sets WHERE and HOW FAST, not WHETHER. If growth is gated on a signal at all,
    ungated cells must still grow -- otherwise the tip-to-body growth ratio is infinite, which is
    a tissue whose body is frozen while its tip builds itself out of nothing."""
    o = _ops(cfg).get("morphogen_growth_3d")
    if o is None:
        return R("P2", "static", "morphogen sets where/how fast, not whether", "na",
                 "no morphogen_growth_3d in this composition")
    rho, a_sw = float(o.get("rho", 0.0)), float(o.get("a_sw", 0.2))
    # a_sw far above any reachable activator means the switch is DISABLED -- growth is uniform and
    # the premise does not bite. Gray-Scott's activator cannot exceed ~1.
    gated = a_sw <= 1.0
    if not gated:
        return R("P2", "static", "morphogen sets where/how fast, not whether", "na",
                 f"a_sw={a_sw} is above any reachable activator, so growth is uniform (switch off)",
                 dict(a_sw=a_sw, rho=rho))
    if rho > 0:
        return R("P2", "static", "morphogen sets where/how fast, not whether", "pass",
                 f"gated growth with a baseline floor rho={rho}", dict(rho=rho, a_sw=a_sw))
    return R("P2", "static", "morphogen sets where/how fast, not whether", "fail",
             f"growth is gated (a_sw={a_sw}) but rho=0, so an ungated cell grows at EXACTLY zero "
             f"and the tip/body growth ratio is infinite. No tissue does that. Either set rho>0 "
             f"or waive P2 and state what the protrusion is made of.", dict(rho=rho, a_sw=a_sw))


def p3_ceiling_above_trigger(cfg):
    """#3 A cell divides because it got big. morphogen_growth_3d caps a cell's target volume at
    vth_frac*v_ref; divide_3d fires at factor*Vbirth. If the CEILING sits below the TRIGGER, no
    cell can ever divide by reaching size and every division comes from the max_cycle stopwatch.
    This is defect D5b, and it was present in the minisite config too."""
    ops = _ops(cfg)
    g, d = ops.get("morphogen_growth_3d"), ops.get("divide_3d")
    if g is None or d is None:
        return R("P3", "static", "a cell divides because it got big", "na",
                 "needs both morphogen_growth_3d and divide_3d")
    vth, fac = float(g.get("vth_frac", 1.35)), float(d.get("factor", 2.0))
    if float(g.get("rho", 0.0)) <= 0 and float(g.get("a_sw", 0.2)) > 1.0:
        return R("P3", "static", "a cell divides because it got big", "na",
                 "growth is off, so no cell can reach any threshold")
    if vth > fac:
        return R("P3", "static", "a cell divides because it got big", "pass",
                 f"growth ceiling {vth} exceeds the division trigger {fac}, so cells can cross it",
                 dict(vth_frac=vth, factor=fac))
    return R("P3", "static", "a cell divides because it got big", "fail",
             f"growth ceiling vth_frac={vth} sits BELOW the division trigger factor={fac}. "
             f"Volume-triggered division is arithmetically impossible: a cell's target is capped "
             f"at {vth}x before it can reach the {fac}x that makes it divide, so every division "
             f"comes from the max_cycle timer and mean cell volume falls for the whole run.",
             dict(vth_frac=vth, factor=fac))


def p5_biology_advances_in_biological_time(cfg):
    """#5 Mechanics is fast, biology is slow. dt is the MECHANICS substep; a biological operator
    integrated with it advances in solver steps rather than in biological time. This is defect
    D5a: dt=0.02 bought 300 frames * 0.02 = 6 units of Gray-Scott time against the ~500 needed,
    and every 'no pattern formed' reading in the campaign was an artefact of the clock."""
    ops = _ops(cfg)
    dt = float(cfg.get("general", {}).get("dt", 1.0))
    react = ops.get("cell_react")
    if react is None:
        return R("P5", "static", "mechanics is fast, biology is slow", "na",
                 "no chemistry in this composition")
    per_frame = dt * float(react.get("rate", 1.0))
    if per_frame >= 0.2:
        return R("P5", "static", "mechanics is fast, biology is slow", "pass",
                 f"the reaction advances {per_frame:.3g} time units per frame",
                 dict(dt=dt, rate=react.get("rate"), per_frame=per_frame))
    return R("P5", "static", "mechanics is fast, biology is slow", "fail",
             f"the reaction advances only {per_frame:.3g} time units per frame (dt={dt} x "
             f"rate={react.get('rate')}). Over {cfg.get('general',{}).get('n_frames','?')} frames "
             f"that is far short of the ~500 a Gray-Scott pattern needs, so 'no pattern formed' "
             f"would be a statement about the clock, not the chemistry. Scale the reaction and "
             f"the diffusion together by 1/dt so their ratio -- which sets the wavelength -- is "
             f"unchanged.", dict(dt=dt, rate=react.get("rate"), per_frame=per_frame))


# ================================================================= PASSIVE (the recorded series)
def p1_tissue_gains_material(cfg, s):
    """#1 Cells grow by taking material in. If a growth operator is running, the tissue must end
    with more material than it started with. Unmeasurable before 2026-07-31: only the CVs of cell
    area and volume were recorded, never the totals."""
    ops = _ops(cfg)
    g = ops.get("morphogen_growth_3d") or ops.get("vesicle_growth")
    if g is None or float(g.get("rate", 0.0)) <= 0:
        return R("P1", "passive", "cells grow by taking material in", "na", "no growth operator")
    v = _col(s, "V_total")
    if v is None:
        return R("P1", "passive", "cells grow by taking material in", "na",
                 "V_total not recorded (run predates the size metrics)")
    if v[-1] > 1.01 * v[0]:
        return R("P1", "passive", "cells grow by taking material in", "pass",
                 f"tissue volume {v[0]:.1f} -> {v[-1]:.1f} (x{v[-1]/v[0]:.2f})",
                 dict(V_start=float(v[0]), V_end=float(v[-1])))
    return R("P1", "passive", "cells grow by taking material in", "fail",
             f"a growth operator is running but tissue volume went {v[0]:.1f} -> {v[-1]:.1f}. "
             f"The body added no material, so anything that grew was built by taking volume from "
             f"somewhere else in the same tissue.", dict(V_start=float(v[0]), V_end=float(v[-1])))


def p3b_mean_cell_volume_holds(cfg, s):
    """#3, measured. Growth doubles a cell and division halves it, so in a healthy proliferating
    epithelium the MEAN cell volume returns to itself. A monotone collapse means the tissue is
    fragmenting faster than it grows -- cell number rising while each cell shrinks."""
    ops = _ops(cfg)
    if "divide_3d" not in ops:
        return R("P3b", "passive", "mean cell volume holds under proliferation", "na",
                 "no division in this composition")
    v = _col(s, "v_cell_mean")
    if v is None:
        return R("P3b", "passive", "mean cell volume holds under proliferation", "na",
                 "v_cell_mean not recorded")
    peak = float(np.nanmax(v))
    ratio = float(v[-1]) / max(peak, 1e-12)
    if ratio >= 0.5:
        return R("P3b", "passive", "mean cell volume holds under proliferation", "pass",
                 f"mean cell volume ends at {ratio:.2f} of its peak ({peak:.2f} -> {v[-1]:.2f})",
                 dict(peak=peak, end=float(v[-1]), ratio=ratio))
    return R("P3b", "passive", "mean cell volume holds under proliferation", "fail",
             f"mean cell volume fell to {ratio:.2f} of its peak ({peak:.2f} -> {v[-1]:.2f}). Cells "
             f"are dividing without having grown, so cell number rises while each cell shrinks "
             f"and the tissue gets no bigger. Check that the growth ceiling clears the division "
             f"trigger (P3).", dict(peak=peak, end=float(v[-1]), ratio=ratio))


def p4_chemistry_not_extinguished(cfg, s):
    """#4/D10 Growing a cell dilutes what is inside it -- and Gray-Scott's activator is sustained
    by a QUADRATIC term, so a steady multiplicative loss beats it. Measured: 1% dilution per step
    extinguishes the pattern within 250 steps. This is why the coral never worked."""
    if "cell_react" not in _ops(cfg):
        return R("P4", "passive", "the chemistry must not be silently extinguished", "na",
                 "no chemistry in this composition")
    a = _col(s, "act_max")
    if a is None:
        return R("P4", "passive", "the chemistry must not be silently extinguished", "na",
                 "act_max not recorded (run predates the threshold-free activator statistics)")
    peak = float(np.nanmax(a))
    if peak <= 1e-6:
        return R("P4", "passive", "the chemistry must not be silently extinguished", "fail",
                 "the activator never rose above zero at any frame: the initial condition did not "
                 "take, or the reaction never ran.", dict(peak=peak))
    ratio = float(a[-1]) / peak
    if ratio >= 0.10:
        return R("P4", "passive", "the chemistry must not be silently extinguished", "pass",
                 f"activator ends at {ratio:.2f} of its peak ({peak:.3f} -> {a[-1]:.3f})",
                 dict(peak=peak, end=float(a[-1]), ratio=ratio))
    return R("P4", "passive", "the chemistry must not be silently extinguished", "fail",
             f"the activator decayed to {ratio:.3f} of its peak ({peak:.3f} -> {a[-1]:.3f}). The "
             f"pattern is extinct, so every downstream reading is about a dead field. The usual "
             f"cause is growth dilution: morphogen_growth_3d.conserve_amount divides chem by "
             f"(s/s_prev)^3 every frame a cell grows, and autocatalysis is quadratic, so it "
             f"loses.", dict(peak=peak, end=float(a[-1]), ratio=ratio))


def p11_tissue_does_not_pass_through_itself(cfg, s):
    """Two parts of the same epithelium cannot occupy the same space. A tissue is a physical body.

    GENUS DOES NOT CATCH THIS, and assuming it did produced a wrong conclusion that had to be
    retracted. Euler characteristic is COMBINATORIAL -- it reads connectivity, not coordinates --
    so a shell folded seventeen layers through itself still reports genus 0, "sphere (as built)".
    Measured on mini_grow_divide_bigger: genus 0 at every single frame, while rays cast from the
    tissue centroid go from 100% single crossings at frame 384 to 0% (median 13) at frame 423. The
    buckling transition was reported as physical on the strength of the genus check alone, and the
    state it produces is not a tissue at all.

    The likely cause in that run: the radial spring's target R0 is frozen at the seed radius while
    the cells' target volumes grow sixteenfold, so the mechanics holds the shell at radius 5 while
    the cells demand far more area than a sphere of that radius has. It has nowhere to go but
    through itself.
    """
    fr = _col(s, "ray_single_frac")
    if fr is None:
        return R("P11", "passive", "tissue cannot pass through itself", "na",
                 "ray_single_frac not recorded (run predates the self-intersection test)")
    worst = float(np.nanmin(fr))
    if worst >= 0.95:
        return R("P11", "passive", "tissue cannot pass through itself", "pass",
                 f"at every frame at least {worst:.1%} of rays cross the surface exactly once",
                 dict(worst_single_frac=worst))
    bad = int(np.argmin(fr))
    f0 = int(s[bad].get("frame", bad))
    med = s[bad].get("ray_cross_med")
    return R("P11", "passive", "tissue cannot pass through itself", "fail",
             f"the surface folds through itself: at frame {f0} only {worst:.1%} of rays cross it "
             f"once, with a median of {med} crossings. A sheet {med} layers deep through its own "
             f"centre is not a tissue, whatever the topology says -- genus is combinatorial and "
             f"reports 'sphere' throughout. Every shape reading past this frame is meaningless.",
             dict(worst_single_frac=worst, first_bad_frame=f0, median_crossings=med))


def p5b_relaxation_keeps_up(cfg, s, mech=None):
    """#5 part B. The premise forbids UNRELAXED TRANSIENTS, not stress.

    Cedric challenged the first wording of premise 5 -- "at every instant the tissue is at force
    balance" -- as too strong a prior, asking whether a tissue can accumulate stress for a long
    time. It can, and does: laser-ablation recoil, tumour spheroids under confinement, residual
    stress in arteries and plant stems. Force balance and zero stress are different things, and
    our own buckling run stores stress for ~40 recorded frames before releasing it.

    What the premise DOES forbid is the configuration lagging behind the forces because the solver
    ran out of iterations. That is a live risk here and not a hypothetical one: `relax_iters` is a
    fixed constant while the mesh it has to relax keeps growing, and relaxation time on a system
    of size L scales worse than linearly. So the thing to check is not "is the stress small" but
    "is the RESIDUAL FORCE stationary" -- a converged solver leaves a residual set by the current
    cell targets, and it should not climb simply because there are more cells.
    """
    if mech is None:
        return R("P5b", "passive", "the mechanical relaxation keeps up with the tissue", "na",
                 "mechanics series not available")
    f = np.asarray(mech.get("force_mean", []), dtype=float)
    n = np.asarray(mech.get("n_cells", []), dtype=float)
    if f.size < 6:
        return R("P5b", "passive", "the mechanical relaxation keeps up with the tissue", "na",
                 "too few mechanics samples")
    early = float(np.nanmedian(f[:max(2, f.size // 4)]))
    late = float(np.nanmedian(f[-max(2, f.size // 4):]))
    grow = late / max(early, 1e-9)
    meas = dict(force_early=early, force_late=late, ratio=grow,
                cells_early=float(n[0]) if n.size else None,
                cells_late=float(n[-1]) if n.size else None)
    if grow <= 2.0:
        return R("P5b", "passive", "the mechanical relaxation keeps up with the tissue", "pass",
                 f"residual force after relaxation {early:.2f} -> {late:.2f} (x{grow:.2f}) while "
                 f"the tissue grew from {meas['cells_early']:.0f} to {meas['cells_late']:.0f} cells",
                 meas)
    return R("P5b", "passive", "the mechanical relaxation keeps up with the tissue", "fail",
             f"the residual force LEFT AFTER each frame's relaxation grew {early:.2f} -> "
             f"{late:.2f} (x{grow:.2f}) as the tissue went from {meas['cells_early']:.0f} to "
             f"{meas['cells_late']:.0f} cells. relax_iters is a constant while the system it has "
             f"to relax keeps getting bigger, so the configuration is lagging the forces and the "
             f"run is no longer quasi-static. Re-run with more relax_iters and check whether the "
             f"result is unchanged; if it moves, the result was the solver.", meas)


def p7_no_absorbing_area_by_stretching(cfg, s):
    """#7 A confluent sheet does not absorb added area by stretching. It buckles, divides or
    extrudes. Stretching shows up as a long tail in the cell shape index -- p95 above 4.5 is a
    cell at roughly 2:1. This is the thin tube-wall cell Cedric spotted on the strip."""
    p95 = _col(s, "shape_idx_p95")
    if p95 is None:
        return R("P7", "passive", "a sheet does not absorb added area by stretching", "na",
                 "shape_idx_p95 not recorded")
    tail = p95[max(1, int(0.8 * len(p95))):]
    worst = float(np.nanmax(tail))
    rv = _col(s, "reduced_volume")
    rv_end = float(rv[-1]) if rv is not None else None
    if worst <= 4.5:
        return R("P7", "passive", "a sheet does not absorb added area by stretching", "pass",
                 f"the stretched tail stays at shape index {worst:.2f} over the final fifth"
                 + (f"; reduced volume {rv_end:.3f}" if rv_end is not None else ""),
                 dict(p95_late=worst, reduced_volume=rv_end))
    return R("P7", "passive", "a sheet does not absorb added area by stretching", "fail",
             f"the top 5% of cells reach shape index {worst:.2f} over the final fifth of the run "
             f"(4.5 is about a 2:1 cell). The tissue is accommodating area by THINNING rather than "
             f"by buckling or dividing"
             + (f"; reduced volume {rv_end:.3f}" if rv_end is not None else "")
             + ". Either it needs to divide faster or the area is being added too fast.",
             dict(p95_late=worst, reduced_volume=rv_end))


def p8_shape_index_floor(cfg, s):
    """#8 perimeter/sqrt(area) cannot go below 2*sqrt(pi) = 3.5449 for ANY shape. That is a
    circle. A value below it is a BROKEN MEASUREMENT and never a finding -- the one check in this
    file that tests the instrument rather than the tissue."""
    lo = _col(s, "shape_idx_min")
    if lo is None:
        return R("P8", "passive", "the shape index has a hard geometric floor", "na",
                 "shape_idx_min not recorded")
    worst = float(np.nanmin(lo))
    if worst >= SHAPE_INDEX_FLOOR - 1e-3:
        return R("P8", "passive", "the shape index has a hard geometric floor", "pass",
                 f"minimum shape index {worst:.4f} respects the floor {SHAPE_INDEX_FLOOR:.4f}",
                 dict(min=worst, floor=float(SHAPE_INDEX_FLOOR)))
    return R("P8", "passive", "the shape index has a hard geometric floor", "fail",
             f"a cell measured shape index {worst:.4f}, below the geometric floor "
             f"{SHAPE_INDEX_FLOOR:.4f}. No shape can do that, so the MEASUREMENT is wrong -- "
             f"degenerate polygon, wrong area, or coordinates paired with the wrong connectivity.",
             dict(min=worst, floor=float(SHAPE_INDEX_FLOOR)))


def p9_closed_sphere(cfg, s):
    """#9 A closed epithelium is a sphere with no holes. No operator in this substrate can fuse
    two surfaces -- division, reconnection, apoptosis and growth all preserve genus -- so a handle
    cannot be created legally. A change in genus is therefore far more likely to be a corrupted
    mesh than a discovery, and this is what tells the two apart."""
    g = _col(s, "genus")
    if g is None:
        return R("P9", "passive", "a closed epithelium is a sphere with no holes", "na",
                 "genus not recorded")
    bad = np.where(np.asarray(g) != 0)[0]
    if not len(bad):
        return R("P9", "passive", "a closed epithelium is a sphere with no holes", "pass",
                 f"genus 0 at all {len(g)} recorded frames", dict(frames=len(g)))
    f0 = int(s[int(bad[0])].get("frame", bad[0]))
    return R("P9", "passive", "a closed epithelium is a sphere with no holes", "fail",
             f"genus left 0 at frame {f0} (value {g[int(bad[0])]:.0f}), and stayed wrong for "
             f"{len(bad)} of {len(g)} frames. No legal operator can change genus, so this is a "
             f"corrupted mesh unless something genuinely topological was intended.",
             dict(first_bad_frame=f0, n_bad=int(len(bad))))


# ================================================================= PROBE (its own simulation)
def p6_resting_vesicle_rests(cfg, device="cpu", frames=40, tol=0.03):
    """#6 A resting vesicle rests. Take the composition's OWN mechanics, switch off growth,
    chemistry and division, and let it sit. The radius must hold.

    This is the cheapest check in the document and the most valuable. Defect D1 -- surface tension
    and cortical contractility with nothing balancing them -- collapsed a seeded vesicle from
    radius 5.00 to 1.80 in twenty frames, and it survived the entire campaign because every run
    that looked healthy loaded a PRE-RELAXED CHECKPOINT instead of seeding. Forty frames of
    mechanics would have caught it on the first day.
    """
    import copy
    import tempfile
    import yaml
    import plexus.operators, tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, tyssue_monolayer, ckpt  # noqa
    import plexus.schema as S
    from plexus.engine import run as engine_run

    keep = {"seed_mesh_3d", "load_mesh_3d", "shape_energy_3d", "cell_geometry_3d",
            "topo_snapshot_3d"}
    d = copy.deepcopy(cfg)
    d.pop("_discovery", None)
    d["general"]["n_frames"] = frames
    d["general"]["record_cap"] = frames + 2
    d["operators"] = [o for o in d["operators"] if o["op"] in keep]
    d["schedule"] = [x for x in d.get("schedule", []) if x in keep]
    if not any(o["op"] == "shape_energy_3d" for o in d["operators"]):
        return R("P6", "probe", "a resting vesicle rests", "na", "no mechanics in this composition")
    fn = tempfile.mktemp(suffix=".yaml")
    yaml.safe_dump(d, open(fn, "w"), sort_keys=False)
    try:
        Hf, out = engine_run(S.load(fn), device=device)
    finally:
        os.unlink(fn)
    pos = out["sets"]["vertex"]["pos"]
    Nv = Hf.level("vertex")._mesh["Nv"]
    r0 = float(np.linalg.norm(pos[0][:Nv], axis=1).mean())
    r1 = float(np.linalg.norm(pos[-1][:Nv], axis=1).mean())
    drift = r1 / max(r0, 1e-9)
    meas = dict(r_start=r0, r_end=r1, ratio=drift, frames=frames)
    if abs(drift - 1.0) <= tol:
        return R("P6", "probe", "a resting vesicle rests", "pass",
                 f"mechanics alone for {frames} frames: radius {r0:.3f} -> {r1:.3f} (x{drift:.4f})",
                 meas)
    how = "COLLAPSES" if drift < 1 else "INFLATES"
    return R("P6", "probe", "a resting vesicle rests", "fail",
             f"with growth, chemistry and division all switched off the vesicle {how}: radius "
             f"{r0:.3f} -> {r1:.3f} (x{drift:.4f}) in {frames} frames. Nothing is acting on it, so "
             f"the mechanics is not in equilibrium at its own rest state -- an inward term "
             f"(kappa_s, gamma) with no counterpart, or a target volume calibrated against only "
             f"part of the energy. Every measurement made on this composition is confounded by "
             f"the drift.", meas)


# --------------------------------------------------------------------------- driver
STATIC = [p2_gate_implies_baseline, p3_ceiling_above_trigger, p5_biology_advances_in_biological_time]
PASSIVE = [p1_tissue_gains_material, p3b_mean_cell_volume_holds, p4_chemistry_not_extinguished,
           p7_no_absorbing_area_by_stretching, p8_shape_index_floor, p9_closed_sphere,
           p11_tissue_does_not_pass_through_itself]


def _mech(run):
    """the mechanics series (residual force, pressures) written alongside metrics"""
    p = os.path.join(LOG_DIR, run, "mechanics.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: np.asarray(z[k]).ravel() for k in z.files}


def check(cfg, series=None, probe=False, device="cpu", mech=None):
    """Every applicable premise. Waivers declared in the spec turn a fail into a recorded ablation."""
    waive = (cfg.get("_premises") or {}).get("waive") or {}
    res = [f(cfg) for f in STATIC]
    if series:
        res += [f(cfg, series) for f in PASSIVE]
        res.append(p5b_relaxation_keeps_up(cfg, series, mech))
    if probe:
        try:
            res.append(p6_resting_vesicle_rests(cfg, device=device))
        except Exception as e:
            res.append(R("P6", "probe", "a resting vesicle rests", "error",
                         f"{type(e).__name__}: {str(e)[:160]}"))
    for r in res:
        if r.status == "fail" and r.pid in waive:
            r.status = "ablation"
            r.detail = f"DECLARED ABLATION -- {waive[r.pid]}  [violates: {r.detail}]"
    return res


def report(res, name=""):
    order = {"fail": 0, "error": 1, "ablation": 2, "pass": 3, "na": 4}
    mark = {"fail": "FAIL", "error": "ERR ", "ablation": "ABL ", "pass": "ok  ", "na": "--  "}
    print(f"\npremise check{' -- ' + name if name else ''}")
    for r in sorted(res, key=lambda x: (order.get(x.status, 9), x.pid)):
        print(f"  [{mark.get(r.status,'?')}] {r.pid:4} {r.premise}")
        if r.status in ("fail", "error", "ablation"):
            for line in _wrap(r.detail, 92):
                print(f"            {line}")
        elif r.status == "pass":
            print(f"            {r.detail}")
    n_fail = sum(1 for r in res if r.status in ("fail", "error"))
    print(f"  {'ALL PREMISES HOLD' if not n_fail else str(n_fail) + ' PREMISE(S) BROKEN'}"
          f"   ({sum(1 for r in res if r.status=='pass')} pass, "
          f"{sum(1 for r in res if r.status=='ablation')} declared ablation, "
          f"{sum(1 for r in res if r.status=='na')} n/a)")
    return n_fail


def _wrap(t, w):
    out, line = [], ""
    for word in str(t).split():
        if len(line) + len(word) + 1 > w:
            out.append(line); line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


# --------------------------------------------------------------------------- certification
def certify():
    """Prove each check catches what it claims, on runs from this campaign whose answer we know.

    A check that has never been seen to FAIL is not a check -- it is a line of code that returns
    'pass'. Each case below is a real config or a real run, paired so that the same check must say
    different things about the two."""
    import yaml
    cases = [
        # (label, check, config, series-run or None, expected)
        ("P3 ceiling below trigger  (minisite verbatim)", p3_ceiling_above_trigger,
         "mini_grow_divide", None, "fail"),
        ("P3 ceiling above trigger  (vth_frac raised)", p3_ceiling_above_trigger,
         "mini_grow_divide_bigger", None, "pass"),
        ("P4 chemistry extinguished (dilution on)", p4_chemistry_not_extinguished,
         "mini_coral", "mini_coral", "fail"),
        ("P4 chemistry alive        (dilution off)", p4_chemistry_not_extinguished,
         "mini_coral_nodilute", "mini_coral_nodilute", "pass"),
        ("P3b cells shrink          (no real growth)", p3b_mean_cell_volume_holds,
         "mini_grow_divide", "mini_grow_divide", "fail"),
        ("P3b cell volume holds     (real growth)", p3b_mean_cell_volume_holds,
         "mini_grow_divide_bigger", "mini_grow_divide_bigger", "pass"),
        ("P1 tissue gains material  (growing ball)", p1_tissue_gains_material,
         "mini_grow_divide_bigger", "mini_grow_divide_bigger", "pass"),
        ("P9 closed sphere", p9_closed_sphere, "coral_fixed_ball", "coral_fixed_ball", "pass"),
        ("P7 cells stretched 2:1    (post-buckling)", p7_no_absorbing_area_by_stretching,
         "mini_grow_divide_bigger", "mini_grow_divide_bigger", "fail"),
        ("P7 cells unstretched      (rigid coral)", p7_no_absorbing_area_by_stretching,
         "coral_fixed_ball", "coral_fixed_ball", "pass"),
    ]
    print("CERTIFYING the premise checks against cases whose answer we already know\n")
    bad = 0
    for label, fn, cfgname, runname, expect in cases:
        p = os.path.join(CONFIG_DIR, f"{cfgname}.yaml")
        if not os.path.exists(p):
            print(f"  [skip] {label:48} no config {cfgname}"); continue
        cfg = yaml.safe_load(open(p))
        s = _series(runname) if runname else None
        if runname and s is None:
            print(f"  [skip] {label:48} no series for {runname}"); continue
        r = fn(cfg, s) if runname else fn(cfg)
        ok = r.status == expect
        bad += not ok
        print(f"  [{'ok ' if ok else 'BAD'}] {label:48} -> {r.status:9} (want {expect})")
        if not ok:
            print(f"          {r.detail[:150]}")
    # P5 on the pre-fix clock, synthesised: dt=0.02 with an unscaled reaction rate
    stale = {"general": {"dt": 0.02, "n_frames": 300},
             "operators": [{"op": "cell_react", "rate": 1.0}]}
    r = p5_biology_advances_in_biological_time(stale)
    ok = r.status == "fail"; bad += not ok
    print(f"  [{'ok ' if ok else 'BAD'}] {'P5 chemistry on the mechanics clock (D5a)':48} -> "
          f"{r.status:9} (want fail)")
    fixed = {"general": {"dt": 0.02, "n_frames": 500},
             "operators": [{"op": "cell_react", "rate": 50.0}]}
    r = p5_biology_advances_in_biological_time(fixed)
    ok = r.status == "pass"; bad += not ok
    print(f"  [{'ok ' if ok else 'BAD'}] {'P5 chemistry rescaled by 1/dt':48} -> "
          f"{r.status:9} (want pass)")
    # P8 on a fabricated impossible measurement
    r = p8_shape_index_floor({}, [{"shape_idx_min": 3.90}, {"shape_idx_min": 3.21}])
    ok = r.status == "fail"; bad += not ok
    print(f"  [{'ok ' if ok else 'BAD'}] {'P8 shape index below the geometric floor':48} -> "
          f"{r.status:9} (want fail)")
    r = p8_shape_index_floor({}, [{"shape_idx_min": 3.90}, {"shape_idx_min": 3.71}])
    ok = r.status == "pass"; bad += not ok
    print(f"  [{'ok ' if ok else 'BAD'}] {'P8 shape index above the floor':48} -> "
          f"{r.status:9} (want pass)")
    # P2 gated growth with no baseline
    r = p2_gate_implies_baseline({"operators": [{"op": "morphogen_growth_3d", "rho": 0.0,
                                                 "a_sw": 0.3}]})
    ok = r.status == "fail"; bad += not ok
    print(f"  [{'ok ' if ok else 'BAD'}] {'P2 gated growth with rho=0':48} -> "
          f"{r.status:9} (want fail)")
    print(f"\n  {'ALL CHECKS CERTIFIED' if not bad else str(bad) + ' CHECK(S) FAILED CERTIFICATION'}")
    return bad


if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--spec", default=None, help="static checks only, on config/okuda/<name>.yaml")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--certify", action="store_true")
    a = ap.parse_args()

    if a.certify:
        raise SystemExit(1 if certify() else 0)

    name = a.run or a.spec
    if not name:
        ap.error("give a run name, --spec <config>, or --certify")
    p = os.path.join(LOG_DIR, name, "spec_run.yaml")
    if not os.path.exists(p):
        p = os.path.join(CONFIG_DIR, f"{name}.yaml")
    cfg = yaml.safe_load(open(p))
    s = None if a.spec else _series(name)
    m = None if a.spec else _mech(name)
    raise SystemExit(1 if report(check(cfg, s, probe=a.probe, device=a.device, mech=m), name) else 0)
