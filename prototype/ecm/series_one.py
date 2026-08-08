"""69-75: the spring-graph membrane at work, and the configurations that change its dynamics.

CHOSEN FROM WHAT THE SWEEPS RULED OUT. `k_bond` moves nothing across three orders (the 4x4 grid), and
the secretion threshold at 0.006 is already mapped, so neither is a useful axis for showing different
behaviour. What the sweeps pointed AT is remodelling: strain sits at growth_rate x tau regardless of
stiffness or supply, so `tau` is the parameter that decides whether the sheet ever approaches its break
threshold. These seven vary the things that actually move it.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "discovery_okuda")):
    sys.path.insert(0, p)
import json

import aniso
import combine as C
import membrane_ops
import run_ecm as R
import tissue as TIS

GATE = os.path.join(_ROOT, "log", "okuda_ECM", "49_aniso_i0_fibres", "load.npz")

# k = 5000: inside the graph-mode ceiling of 8,220 measured at k = 200..40,000.
BASE = dict(membrane_particles=45000, membrane_cutoff=0.008, membrane_break=0.35,
            membrane_bond_k=5.0e3, membrane_adhesion=1.0e4, membrane_tau=60.0,
            membrane_jitter=0.35, membrane_reserve=12.5, membrane_secrete_rate=0.012,
            membrane_impl="graph", membrane_drag=40.0)

SERIES = {
    "69_graph_reference": dict(),
    # tau = 0 removes `basement_membrane_remodel` entirely: rest lengths are frozen, so stretch
    # ACCUMULATES instead of being forgotten. This is the one configuration in which the 0.35 break
    # threshold is reachable at all -- every run so far sat at 0.08 because remodelling absorbed it.
    "70_no_remodelling": dict(membrane_tau=0.0),
    "71_fast_remodelling": dict(membrane_tau=10.0),
    "72_starved": dict(membrane_secrete_rate=0.002),
    # the measured null: without an anchor the sheet SLIDES over the epithelium and never stretches,
    # mean bond strain 0.0000 at every stiffness. Kept in the series because it is the control that
    # says the strain in the others is real.
    "73_no_adhesion": dict(membrane_adhesion=0.0),
    "74_brittle": dict(membrane_break=0.08),
    "75_on_ovoid": dict(_gated=True),

    # --- 76 onward: THE FIXED SEED ------------------------------------------------------------------
    # Everything above 75 was deleted. The sheet those runs started from was not packed: the relaxation
    # ran over the whole 45,000-particle reservoir and the sheet then kept every 13.5th of them, and a
    # SUBSET of a blue-noise set is not blue noise -- thinning randomises it back to Poisson. Measured on
    # the seeded sheet, nearest-neighbour distance was 0.505 of a hexagonal packing with cv 0.411,
    # against 0.461 / 0.535 for points thrown at random. So it began 8% better than random, which is why
    # holes were visible in the first frames and grew from there. Relaxing the laid-down count directly
    # gives 0.885 with cv 0.046.
    #
    # 76 is the new nominal: same configuration as 69, on a sheet that starts packed.
    "76_reference_fixed_seed": dict(),

    # --- 77: THE BIOLOGICAL VERSION -----------------------------------------------------------------
    # Two rules replaced, both because the versions they replace are not things a cell could do.
    #
    # DEPOSITION. Cells secrete basement membrane basally, into the patch of surface directly beneath
    # themselves -- laminin polymerises where integrin and dystroglycan nucleate it, collagen IV
    # assembles onto that. So deposition is local to each cell and UNIFORM PER UNIT BASAL AREA. Placing
    # new material beside an existing node is a random walk and clumps; placing it in the largest gap
    # packs better but is worse biology, because it needs a global view of where the holes are and a cell
    # has none. Uniform fills a hole at the same rate it adds anywhere else, which is the honest
    # mechanism, and leaves the evening-out to the network.
    #
    # ANCHORS. `u0` was frozen at seeding, which says a patch of membrane remembers where a cell was
    # four hundred frames ago. Focal adhesions turn over in seconds to minutes, and the cells themselves
    # divide and swap neighbours, so the tissue a patch is attached to MOVES. Frozen anchors are also
    # what pins the holes: material that relaxes into a gap is dragged back out by its own tether.
    "77_biological": dict(membrane_deposit="uniform", membrane_tau_adh=40.0),
    # the two controls that say which of the two changes did the work
    "78_uniform_only": dict(membrane_deposit="uniform", membrane_tau_adh=0.0),
    "79_turnover_only": dict(membrane_deposit="parent", membrane_tau_adh=40.0),

    # --- 80: WITH ONGOING CROSSLINKING --------------------------------------------------------------
    # The 2x2 of 76-79 settled two things and left the real defect untouched. Anchor turnover does
    # essentially nothing (d/hex moves 0.013 and 0.001, inside run-to-run scatter), so frozen anchors
    # were NOT what stopped relaxation working. Deposition is a straight trade: parent packs well
    # (0.79) with large gaps (1.64), uniform has the smallest gaps (1.41) and packing that reverts to
    # random (0.47).
    #
    # Neither is the problem. The bond list was built once and extended only for newly secreted nodes,
    # so the topology was frozen: 29% of pairs within the cutoff had no bond, and there were more bonds
    # than close pairs -- a third of the network held together by history. The black patches are
    # plausibly regions with nodes and no crosslinks, which renders identically to an empty patch and
    # which no amount of relaxation could repair, because relaxation moves nodes and the defect is in
    # the edges. `basement_membrane_crosslink` (rewire) is the missing half of fragmentation.
    "80_crosslinking": dict(membrane_deposit="uniform", membrane_rebond_every=20),

    # --- 81, 82: WHY THE NETWORK DOES NOT EVEN OUT --------------------------------------------------
    # Cedric: a spring network, overdamped, should reach equilibrium -- so why are the nodes not
    # equidistant? It has reached equilibrium. Measured on 80, the cv of the REST LENGTHS is 0.521:
    # every spring wants a different length, so a disordered configuration is exactly what the network
    # relaxes to. That is why anchors, crosslinking and deposition all did nothing -- none of them
    # touches where the disorder is stored.
    #
    # 81 is his test: remove the anchors entirely, so the sheet is a pure spring network with nothing
    # holding it to the surface. It should slide (73 lost coverage doing this), but it isolates whether
    # the tethers constrain the packing at all.
    # 82 is what the rest-length measurement points to: relax rest lengths toward the spacing the
    # sheet's own density implies, instead of toward each bond's own length. A real collagen IV network
    # has a characteristic mesh size; giving the springs a COMMON target is what lets a relaxed network
    # be a uniform one.
    "81_no_anchor": dict(membrane_deposit="uniform", membrane_adhesion=0.0),
    "82_mesh_restlength": dict(membrane_deposit="uniform", membrane_remodel_target="mesh"),

    # 83-85: the holes, taken to the bottom. 82 helped (cv of the rest lengths 0.527 -> 0.191) but only
    # halfway, and the relaxation bench said why: fitting cv = A*exp(-t/T) + C over 6000 iterations gives
    # C = 0.18-0.22, a PLATEAU, not a slow decay. About 85% of the disorder never relaxes at all.
    #
    # Two changes, and neither is a tuning knob.
    #
    # `fixed` holds l* at the frame-0 spacing instead of tracking the sheet's own mean. `mesh` is
    # self-referential: if secretion under-delivers, the mean grows and the springs bless the sparser
    # sheet. Fixed, N = 4*pi*R^2/l*^2 is the node count the sheet NEEDS, secretion is the only thing that
    # can supply it, and a shortfall shows up as tension rather than being absorbed into the target.
    #
    # `repel_w` adds excluded volume beside the springs. The bench isolated the cause: it is not a
    # missing repulsion (a spring pushes when compressed) but an unopposed attraction -- the rewire
    # throws crosslinks clear across the holes and those long bonds haul the rim into knots. Deleting
    # the pull fixes it (cv 0.173 -> 0.049) but leaves a sheet that bears no tension, so the pull stays
    # and repulsion is added: spacing from one, load from the other.
    #
    # THREE RUNS BECAUSE THE BENCH BRACKETS w RATHER THAN FIXING IT, and the bracket is stiffness
    # dependent. Swept at k = 2e5 the answer was w = 8 (cv 0.020) with w = 20 unstable (0.179); swept at
    # 100x softer springs it was the reverse, w = 20 the better of the two. THIS SERIES RUNS k = 5e3, so
    # h*z*k*(1+w)/gamma = 0.06 / 0.51 / 1.20 for w = 0 / 8 / 20 -- all under the limit of 2, and none of
    # them is the case the bench calibrated. So the bracket is run, not predicted.
    #
    # 83 separates the target change from the repulsion: if `fixed` alone closes the holes, the excluded
    # volume is not needed and 84/85 should not be kept. (The bond range needs no change here: cutoff =
    # 0.008 against l* = 0.00575 is 1.4 spacings already, not the 3.7 the spec default would have given.)
    "83_fixed_target": dict(membrane_deposit="uniform", membrane_remodel_target="fixed"),
    "84_fixed_repel8": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                            membrane_repel_w=8.0),
    "85_fixed_repel20": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                             membrane_repel_w=20.0),

    # 86-88. What 83-85 settled, measured at the middle frame:
    #
    #   82 mesh target      d/hex 0.543   cv 0.442
    #   83 fixed target     d/hex 0.540   cv 0.443     <- a NULL. Fixing l* alone changes nothing.
    #   84 w = 8            d/hex 0.604   cv 0.339
    #   85 w = 20           d/hex 0.677   cv 0.243     <- monotone, no plateau: the optimum is above 20
    #
    # And the supply question is closed: N_live/N_needed at l* is 1.14 at EVERY frame from R = 0.088 to
    # 0.298, so secretion tracks growth exactly and the sheet always carries a 14% surplus of nodes. The
    # holes are not a material shortage, they are enough nodes badly arranged -- which is precisely the
    # case repulsion addresses, so the lever is right and only too weak.
    #
    # 86/87 push w, since 85 was still climbing. Frame-level h*z*k*(1+w)/gamma is 2.9 at w = 50 and 6.9
    # at w = 120, both past the overdamped limit of 2 -- but the membrane is on the MPM path, integrated
    # at the substep, which is why w = 20 improved instead of diverging. So these two also LOCATE THE
    # CEILING: if 87 explodes, the substep headroom is smaller than the MPM path implies.
    #
    # 88 is the other error 83-85 exposed. The sheet's actual mean spacing is 0.94 l*, BELOW the
    # repulsion range, so nearly every pair is inside it and the operator acts as a uniform outward
    # pressure against the anchors rather than selectively separating the over-close pairs. Setting the
    # range to the density-implied spacing restores the selectivity at a stiffness already shown stable.
    "86_repel50": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                       membrane_repel_w=50.0),
    "87_repel120": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                        membrane_repel_w=120.0),
    "88_repel20_range94": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                               membrane_repel_w=20.0, membrane_repel_range=0.94),

    # 89-91: Plexus's own attraction_repulsion law instead of the one-sided spring.
    #
    # WHAT 86-88 LEFT. 86 (w = 50) is the best packing so far, d/hex 0.746 -- but at z = 3.82, under the
    # 2D rigidity threshold, with lcc 0.64: some of that gain is bought by BREAKING crosslinks, which is
    # the same process that destroyed 87 (w = 120: lcc 0.04, z = 2.06, packing back down to 0.568). 88
    # was a null (0.679 against 85's 0.677), so the range-selectivity argument was simply wrong.
    #
    # THE ARCHIVED `blue` PARAMETERS DO NOT DO THIS, and that is worth recording because it is the
    # opposite of what the name suggests. f(0) = p0 - p2, and blue's is +0.022: an attractive core. Any
    # pair closing below r = 0.0011 welds, so the set CLUMPS -- in 2D, d/hex 0.471 -> 0.242 over 250
    # frames, worse than random (0.465). Every archived type in the two configs has an attractive core
    # except rand_2t's t0, and that one never decays to zero at the neighbour radius (f = -0.63 there),
    # so it churns instead. Dropping p0 leaves one decaying repulsion, which is the CGI recipe for
    # scattering points evenly over a surface, and it reaches blue noise in 2D (d/hex 0.887, cv 0.037).
    #
    # ON THE REAL SHEET, run 85's middle frame relaxed 300 iterations:
    #     start                          d/hex 0.677   cv 0.243   gap 1.14
    #     attraction_repulsion, 3 l*     d/hex 0.895   cv 0.052   gap 0.65
    # Better than anything the spring reached, and the likely reason is RANGE, not strength: the AR law
    # acts over ~3 spacings and dies smoothly, the one-sided spring acts over 1 and is truncated.
    #
    # k is bracketed, not derived. Matching total force against 85 (w = 20) over ~28 neighbours instead
    # of 6 gives k ~ 1.7e4, i.e. w ~ 3.5, so 89/90 straddle it. 91 is the literal reading of "only
    # attraction-repulsion": the crosslink springs off entirely, absolute k, no tension-bearing network
    # left -- expected to pack well and to have nothing holding the sheet against growth.
    "89_ar_w3": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                     membrane_repel_law="ar", membrane_repel_w=3.5, membrane_repel_range=3.0),
    "90_ar_w10": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                      membrane_repel_law="ar", membrane_repel_w=10.0, membrane_repel_range=3.0),
    "91_ar_only": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                       membrane_repel_law="ar", membrane_repel_k=5.0e4, membrane_repel_range=3.0,
                       membrane_bond_k=0.0),

    # 92-94: 89-91 again with k set correctly. THE ERROR WAS UNITS, NOT THE LAW. In the ar form `k` is
    # not a stiffness: amp is O(1), the step is dt*k*amp*d, so the scale is fixed by k*dt ~ 1, meaning
    # k ~ 250 at dt = 4e-3 -- not the 1.75e4 I derived by matching spring forces, which is 70x past it.
    # Above the edge the sheet scrambles to a fixed point that sits near where it started, which is why
    # 89-91 read as a null (0.564 / 0.591 / 0.535) instead of as an overshoot. Swept on 85's middle
    # frame: k = 25 -> 0.785, 100 -> 0.830, 250 -> 0.846, 600 -> 0.855, 1500 -> 0.872, 4000 -> 0.730.
    # Mean aggregation, as the attraction_repulsion operator itself uses.
    "92_ar_k250": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                       membrane_repel_law="ar", membrane_repel_k=250.0, membrane_repel_range=3.0),
    "93_ar_k600": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                       membrane_repel_law="ar", membrane_repel_k=600.0, membrane_repel_range=3.0),
    "94_ar_k1500": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                        membrane_repel_law="ar", membrane_repel_k=1500.0, membrane_repel_range=3.0),

    # 95-97. 92-94 came back a flat null (0.542/0.540/0.544 against 82's 0.543 with no repulsion at
    # all), and that finally located the real error -- which is NOT the one 92-94 were correcting.
    #
    # THE BENCH CANNOT CALIBRATE THIS OPERATOR. relax_bench integrates x += dt*F, a position update. The
    # operator emits an ACCELERATION into MPM, so displacement goes as dt^2*F, smaller by ~1.6e-5 at
    # dt = 4e-3. The bench's k ~ 250-1500 is therefore about 1e5 too small in the run, and 89's faint
    # effect at k = 1.75e4 (0.564 against a 0.543 baseline) is consistent with being under-powered, not
    # with the overshoot I read it as. Two brackets went to calibration because I kept trusting a
    # testbed that does not share the run's equation of motion.
    #
    # The only sound anchor is the LINEAR repel, which is the same operator on the same emission path:
    # k = 1e5 there (85, w = 20) moved d/hex 0.543 -> 0.677 and k = 2.5e5 (86, w = 50) -> 0.746. Matching
    # the ar law's typical force against those gives k ~ 2e5, so this brackets it across a decade and a
    # half. Sum aggregation, deliberately -- same as the linear operator, so the ONLY thing that differs
    # is the force shape, which is the actual hypothesis: range ~3 l* and a smooth decay instead of
    # range 1 l* and a hard truncation.
    "95_ar_k2e5": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                       membrane_repel_law="ar", membrane_repel_k=2.0e5, membrane_repel_range=3.0,
                       membrane_repel_aggr="sum"),
    "96_ar_k1e6": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                       membrane_repel_law="ar", membrane_repel_k=1.0e6, membrane_repel_range=3.0,
                       membrane_repel_aggr="sum"),
    "97_ar_k5e6": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                       membrane_repel_law="ar", membrane_repel_k=5.0e6, membrane_repel_range=3.0,
                       membrane_repel_aggr="sum"),
}


def main():
    name = sys.argv[1]
    dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 402
    over = dict(SERIES[name])
    gated = over.pop("_gated", False)
    myo = over.pop("_myo", None)          # junction-level knobs: they belong to PASS 1, not the spec
    # RECORDED BEFORE THE POP CONSUMES IT. `_gated` is the only change 75 makes, so popping it left an
    # empty dict and the run archived itself as `{"reference": true}` -- identical to 69's label, for a
    # run on a tissue of aspect 1.332 against 69's 1.021.
    label = dict(over)
    if myo:
        label.update(myo)
    if gated:
        label["tissue"] = "gated ovoid (aspect 1.33)"
    for t in (membrane_ops.BOND_TRACE, membrane_ops.MEMBRANE_STRAIN,
              membrane_ops.SECRETE_TRACE, membrane_ops.BOND_SNAPSHOTS, membrane_ops.HOOP_TRACE):
        t.clear()
    tk = dict(frames=401, device=dev, buffer_x=4, myosin=1.0)
    if myo:
        tk.update(myo)
        # `:g` on a str raises; `myo_keyed_on` is a string ("tension"/"strain_rate"), so the tag has to
        # format by type. The cache key is built from this, so getting it wrong either crashes or -- worse
        # -- collides two different configurations onto one cached tissue.
        tk["tag_extra"] = "_" + "_".join(
            f"{k}{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}{v}"
            for k, v in sorted(myo.items()))
    if gated:
        tk.update(gate_npz=GATE, gate_p_half="auto", gate_hill=6.0, gate_floor=0.08,
                  gate_smooth_frames=25, gate_smooth_phi=360.0, tag_extra="_gated_myo")
    npz = TIS.load_or_build(**tk)
    cfg = dict(aniso.BASE)
    cfg.update(BASE)
    cfg.update(over)
    cfg["membrane"] = npz
    spec, info = C.build(name, npz, **cfg)
    spec["general"]["n_frames"] = frames
    # A SHORT RUN IS A SMOKE TEST, AND IT GETS ITS OWN FOLDER. Running one under the real name writes
    # into the folder a cluster job may be using for the same name -- which is how 77 ended up with a
    # 24-frame movie from a 25-frame test sitting beside a 402-frame trajectory, and could as easily have
    # corrupted the trajectory instead of just the movie. The underscore prefix also keeps it off the
    # numbered ledger.
    if frames < 100:
        name = "_smoke_" + name
        print(f"[series] {frames} frames -> writing to {name}, not the numbered folder", flush=True)
    # WIPED BEFORE WRITING. A relaunch that leaves the previous attempt's files behind gives a folder
    # whose contents came from two different runs, and no way to tell which is which -- 79 held a
    # 24-frame movie from a smoke test beside nothing else, and 77 held one beside a 402-frame
    # trajectory. Either the folder is this run's output or it is empty.
    d = os.path.join(R.LOG, name)
    if os.path.isdir(d):
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        print(f"[series] cleared {d} before writing", flush=True)
    os.makedirs(d, exist_ok=True)
    info["varied"] = label or {"reference": True}
    json.dump(info, open(os.path.join(d, "pass1.json"), "w"), indent=1)
    R.run(name, spec, device=dev, movie=True, render_kw={"movie_frames": 150, "fps": 15})

    bt = np.asarray(membrane_ops.BOND_TRACE, float)
    z = np.load(os.path.join(d, "traj.npz"))
    al = np.asarray(z["malive"]) if "malive" in z.files else None
    ms = np.asarray(membrane_ops.MEMBRANE_STRAIN[-1], float)
    if al is not None:
        ms = ms[al]
    ms = np.nan_to_num(ms, nan=-1, posinf=-1, neginf=-1)
    P = np.asarray(z["mpos"])[-1][al] if al is not None else np.asarray(z["mpos"])[-1]
    u = P - P.mean(0)
    u /= np.linalg.norm(u, axis=1)[:, None]
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0])
    bi = (np.clip((th / np.pi * 16).astype(int), 0, 15) * 32
          + np.clip(((ph + np.pi) / (2 * np.pi) * 32).astype(int), 0, 31))
    # FROM THE TISSUE FILE, not the module global: the global is empty whenever pass 1 came from cache.
    _tz = np.load(npz)
    t1 = np.asarray(_tz["t1_trace"], float) if "t1_trace" in _tz.files else np.zeros((1, 4))
    if t1.size == 0:
        t1 = np.zeros((1, 4))
    info["result"] = dict(bonds_start=int(bt[0, 0]), bonds_end=int(bt[-1, 0]),
                          lcc_end=float(bt[-1, 3]) if bt.shape[1] > 3 else None,
                          mean_degree_z=float(bt[-1, 4]) if bt.shape[1] > 4 else None,
                          t1_total=int(t1[-1, 2]) if len(t1) else 0,
                          t1_per_cell_per_frame=float(t1[:, 1].sum() / max(t1[:, 3].mean(), 1)
                                                      / max(len(t1), 1)),
                          n_alive=int(al.sum()) if al is not None else None,
                          strain_end=float(ms.mean()),
                          strain_p99=float(np.percentile(ms, 99)),
                          coverage=len(np.unique(bi)) / (16 * 32))
    json.dump(info, open(os.path.join(d, "pass1.json"), "w"), indent=1)
    print("SERIES " + json.dumps({"name": name, **info["result"]}), flush=True)


if __name__ == "__main__":
    main()
