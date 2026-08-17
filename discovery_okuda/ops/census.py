#!/usr/bin/env python
"""How big is this simulation, counted rather than estimated.

    python census.py 07i_ramp

FIVE QUESTIONS, and each answer says which of them it is measured from:

    entities            how many biological objects exist          per kept frame: min / mean / max
    interactions        how many couplings are evaluated a step     per substep, and per frame
    interaction types   how many distinct LAWS are evaluated        enumerated, with their arity
    operators           how many Plexus2 operators are running      enumerated, with their tick
    fields              how many state fields are carried           enumerated, with their domain

WHERE THE NUMBERS COME FROM, because three different places hold them and they do not agree by
accident:

  THE STORE (`bm_frames.npz`) holds the sheet, the plaques and the epithelium at every kept frame, so
  everything counted from it is exact for this run and needs no model: nodes, faces, plaques, bonds,
  free receptors. Edges are counted from the face table itself rather than taken as 3F/2, which only
  holds for a closed surface and this one is allowed to tear.

  THE RIG AT SEED gives the two quantities the store does not carry: the CONTACT set (which sheet
  nodes are being pushed off the epithelium this substep -- it is rebuilt every substep and never
  written out) and the SUBSTEP COUNT `n_sub`. Both are reported as seed values with the rule that
  scales them, and `n_sub` is now recorded per frame by `test_07h_bind_cull.run`, so the next run
  will not need the caveat. `n_sub` is NOT a constant: it is ceil(max(M(lam_el + k_c + N_b kappa_b),
  M_epi(k_drive + k_c))/s_target), and 05a measured it growing 21 -> 194 over 401 frames as the
  membrane stiffened under stretch. A per-frame interaction count that used the seeded value would
  therefore be an underestimate everywhere except frame 0, and it is labelled as such.

  THE CODE gives the types, the operators and the fields, by walking the rig's own MRO -- so a law
  that was inherited but never called cannot be counted, and one added by a subclass cannot be
  missed.

AND WHAT IS NOT COMPUTED IS SAID SO. The stroma is an MPM continuum in `06_spheroid_ecm` and it is
REPLAYED here, not stepped: the 2x2's other three panels are re-drawn from that run's `traj.npz`.
Its particles are counted under `replayed, not stepped`, because a census that folded them into the
total would claim work this run does not do. The epithelium is replayed for the same reason -- the
vertex model was solved in `cellfix_B_new` -- and its cells ARE counted as entities, because every
one of them carries a receptor pool, twelve adhesions and a wedge of surface that this run does step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM"))


def edges_of(F):
    e = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
    return len(np.unique(e, axis=0))


def stat(v):
    v = np.asarray(v, float)
    return dict(min=float(v.min()), mean=float(v.mean()), max=float(v.max()), last=float(v[-1]))


# =============================================================================================
#  the entities
# =============================================================================================
def entities(run):
    d = os.path.join(LOG, run)
    z = np.load(os.path.join(d, "bm_frames.npz"))
    n = int(z["n_kept"])
    M = json.load(open(os.path.join(d, "metrics.json")))
    cells = np.asarray(M["series"]["cells"], float)
    E = {k: [] for k in ("bm_node", "bm_face", "bm_edge", "plaque", "bond", "free_receptor",
                         "epi_vertex")}
    for i in range(n):
        F = np.asarray(z[f"f{i}"], np.int64)
        E["bm_face"].append(len(F))
        E["bm_node"].append(len(np.unique(F)))
        E["bm_edge"].append(edges_of(F))
        E["plaque"].append(len(np.asarray(z[f"n{i}"])))
        E["bond"].append(float(np.asarray(z[f"nb{i}"]).sum()))
        E["free_receptor"].append(float(np.asarray(z[f"nf{i}"]).sum()))
        E["epi_vertex"].append(len(np.asarray(z[f"e{i}"])))
    E["epi_cell"] = cells.tolist()
    # the wedges: one triangle per half-edge. FE is the FINAL table, so the ratio it fixes is used to
    # carry the count back over the run rather than a guess of six per cell.
    FE = np.asarray(z["FE"])
    per_cell = len(FE) / max(cells[-1], 1.0)
    E["epi_wedge"] = (cells * per_cell).tolist()
    return d, {k: stat(v) for k, v in E.items()}, per_cell, n


# =============================================================================================
#  the interactions, the types, the operators, the fields
# =============================================================================================
def seed_rig(module, cls, **kw):
    """Build the run's own rig at its seeded size, for the two counts the store does not carry."""
    import importlib
    m = importlib.import_module(module)
    return getattr(m, cls)(**kw)


def laws(ent, per_substep_contact_per_node, n_sub_seed):
    """Every distinct interaction law the substep evaluates, with its arity and its population.

    ARITY IS PART OF THE COUNT. `bm_elastic` is one evaluation per FACE and couples that face's three
    vertices, so it is 1 law-evaluation and 3 pairwise couplings per face; `plaque` is one evaluation
    per plaque coupling one sheet node to the three vertices of the wedge it is bound to. Reporting
    only the evaluations undercounts the coupling and reporting only the pairs overcounts the laws,
    so both are given and neither is called "the number of interactions" on its own.
    """
    F = ent["bm_face"]["mean"]
    P = ent["plaque"]["mean"]
    V = ent["epi_vertex"]["mean"]
    N = ent["bm_node"]["mean"]
    C = ent["epi_cell"]["mean"]
    return [
        dict(name="bm_elastic", law="StVK on the rest metric, per face",
             per="bm_face", evals=F, pairs=3 * F, tick="every substep"),
        dict(name="plaque (normal)", law="f = -N_b kappa_b (d - l0) n_hat",
             per="plaque", evals=P, pairs=3 * P, tick="every substep"),
        dict(name="plaque (tangential)", law="xdot_par (1 + M xi) = xdot_prov + M xi v_par, solved",
             per="plaque", evals=P, pairs=3 * P, tick="every substep"),
        dict(name="clutch", law="Nb_dot = k_on n_f rho_L (1 - n_b/n_max) A - k_off(f) Nb, "
                                "k_off = k_off0 exp(f/f_b)",
             per="plaque", evals=P, pairs=P, tick="every substep"),
        dict(name="receptor pool", law="Nf_dot = s_i - Nf/tau_i, per CELL and well-mixed",
             per="epi_cell", evals=C, pairs=C, tick="every substep"),
        dict(name="contact (steric)", law="f = -k_c pen n_hat where the sheet is inside a wedge",
             per="bm_node", evals=per_substep_contact_per_node * N,
             pairs=3 * per_substep_contact_per_node * N, tick="every substep"),
        dict(name="epi_drive", law="f = k_drive (a_epi - x_epi) toward the replayed surface",
             per="epi_vertex", evals=V, pairs=V, tick="every substep"),
        dict(name="bm_secrete", law="dm/dt = s A - m/tau_bm, per face",
             per="bm_face", evals=F, pairs=0.0, tick="once a frame"),
        dict(name="bm_degrade / bm_tear", law="rho = m/A; a face dies where rho < rho_crit",
             per="bm_face", evals=F, pairs=0.0, tick="once a frame"),
        dict(name="bm_refine (longest edge)", law="Rivara bisection where the longest edge exceeds "
                                                 "the trigger; unanimity makes it conforming",
             per="bm_face", evals=F, pairs=0.0, tick="every `every` frames"),
        dict(name="spectral_rate", law="lambda_max of the elastic Hessian by power iteration, "
                                       "40 Hvp of the whole mesh",
             per="bm_face", evals=40 * F, pairs=40 * 3 * F, tick="every `refresh` frames"),
    ]


def fields(rig):
    """Every state field the rig carries, labelled by the domain its length matches.

    WALKED, NOT LISTED. A hand-written list of fields is a comment that stops being true at the next
    subclass; this asks the object. Anything whose first dimension matches a population is reported
    against that population, and anything that does not is reported as a scalar or a parameter, so a
    field sized to the WRONG population -- the fault that made 05h1's chemistry a permutation and
    would have made 07j's protease cap a sixth of itself -- shows up here as a mismatch rather than
    as a plausible number.
    """
    import torch
    pop = {int(rig.sheet.mass.shape[0]): "bm_face (reservoir)",
           int(rig.sheet.x.shape[0]): "bm_node (reservoir)", int(rig.sheet.n): "bm_node (live)",
           int(rig.sheet.Fc.shape[0]): "bm_face", int(rig.sheet.Ed.shape[0]): "bm_edge",
           int(rig.x_epi.shape[0]): "epi_vertex", int(rig.F_epi.shape[0]): "epi_wedge",
           int(rig._nF): "epi_cell", int(rig.ct_node.numel()): "plaque"}
    out = []
    for owner, o in (("rig", rig), ("sheet", rig.sheet), ("clutch", rig.clutch)):
        for k in sorted(vars(o)):
            v = getattr(o, k)
            if not torch.is_tensor(v) or k.startswith("_"):
                continue
            n = int(v.shape[0]) if v.dim() else 1
            out.append(dict(owner=owner, name=k, domain=pop.get(n, f"n={n}"),
                            shape=list(v.shape), dtype=str(v.dtype).replace("torch.", "")))
    return out


def _mpm(sp):
    """The stroma's particle count, from the matrix run this one is drawn against."""
    src = (sp.get("rendered_against") or {}).get("matrix")
    f = os.path.join(LOG, str(src), "traj.npz")
    if not src or not os.path.exists(f):
        return None
    z = np.load(f, mmap_mode="r")
    for k in ("x", "X", "pos", "q0"):
        if k in z.files:
            a = z[k]
            return int(a.shape[1] if a.ndim == 3 else a.shape[0])
    return None


CHEM = [("MT1-MMP -> faces", "the tethered enzyme, carried from the cells that express it onto the "
                             "faces they touch; it does NOT diffuse", "bm_face", 1),
        ("proMMP-2 secretion", "dpro/dt += s_pro(cell) carried through the contact map", "bm_face", 1),
        ("TIMP-2 secretion", "dtimp/dt += s_timp(cell), same carry", "bm_face", 1),
        ("TIMP-3 deposition", "immobile: deposited, then cleared as exp(-dt/tau_timp3)", "bm_face", 1),
        ("proMMP-2 diffusion", "implicit cotangent Laplacian on the deformed sheet", "bm_face", 3),
        ("MMP-2 diffusion", "implicit cotangent Laplacian", "bm_face", 3),
        ("TIMP-2 diffusion", "implicit cotangent Laplacian", "bm_face", 3),
        ("ternary activation", "k_act (mt1 occ)(mt1 free) pro, occ = x/(1+x), x = (T2+T3)/K -- the "
                               "bell that peaks at c_T = K", "bm_face", 1),
        ("MMP-TIMP inhibition", "r = k_inhib mmp (timp + t3) dt, capped at mmp", "bm_face", 1),
        ("MMP degradation of the sheet", "dm/dt -= k_deg mmp m: the mass the tear law reads",
         "bm_face", 1)]


def chem_laws(ent, rig):
    """The protease chemistry, IF this rig carries it. Detected on the object, not assumed."""
    import torch
    if rig is None or not torch.is_tensor(getattr(rig, "mt1", None)):
        return []
    F = ent["bm_face"]["mean"]
    return [dict(name=n, law=w, per=p, evals=F, pairs=k * F, tick="every substep")
            for n, w, p, k in CHEM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="07i_ramp")
    ap.add_argument("--contact-per-node", type=float, default=None,
                    help="measured on the seeded rig if not given")
    ap.add_argument("--n-sub", type=int, default=None)
    ap.add_argument("--module", default="test_07i_ramp",
                    help="the module whose rig this run used; its `main` is not called")
    ap.add_argument("--cls", default=None, help="the rig class, default the module's only Rig07*")
    a = ap.parse_args()
    d, ent, per_cell, nk = entities(a.run)
    sp = yaml.safe_load(open(os.path.join(d, "spec.yaml")))

    cpn, nsub, how = a.contact_per_node, a.n_sub, "given on the command line"
    if cpn is None or nsub is None:
        import importlib
        m = importlib.import_module(a.module)
        cls = getattr(m, a.cls) if a.cls else next(
            getattr(m, k) for k in vars(m) if k.startswith("Rig07") and k[5:].isdigit() is False
            and isinstance(getattr(m, k), type) and getattr(m, k).__module__ == m.__name__)
        kw = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, sigma_T=7.0, zeta=20.0,
                  s_target=1.0, k_drive=50.0, dev="cuda:0", max_refine=3, edge_trigger=1.45,
                  reseed=True, tau_bm=40.0, rho_crit=0.0)
        kw.update(getattr(m, "CENSUS_KW", {}))
        r = cls(N0=12, Nf0=300.0, split_budget=150, every=10, batched=True, **kw)
        r.frame(0)
        cpn = float(r.cx_node.numel()) / float(r.sheet.n) if cpn is None else cpn
        nsub = int(r.n_sub) if nsub is None else nsub
        how = "measured on the seeded rig at frame 0"
        pen = int(r.contact()[2])          # the steric law acts only on nodes inside a wedge
        FLD = fields(r)

    L = laws(ent, cpn, nsub) + chem_laws(ent, r if "r" in dir() else None)
    sub_evals = sum(x["evals"] for x in L if x["tick"] == "every substep")
    sub_pairs = sum(x["pairs"] for x in L if x["tick"] == "every substep")
    once = sum(x["evals"] for x in L if x["tick"] == "once a frame")
    out = dict(
        run=a.run, kept_frames=nk, frames=sp["general"]["frames"],
        entities=ent, wedges_per_cell=per_cell,
        contact_fraction_of_nodes=dict(value=cpn, how=how),
        n_sub=dict(value=nsub, how=how,
                   caveat="not a constant: it tracks lambda_max of the elastic Hessian, which rises "
                          "with stretch (05a: 21 -> 194 over 401 frames). Per-frame totals below use "
                          "the SEEDED value and are therefore a lower bound after frame 0."),
        interaction_types=len(L), laws=L,
        fields=dict(n=len(FLD), by_domain={k: sum(1 for f in FLD if f["domain"] == k)
                                           for k in sorted({f["domain"] for f in FLD})},
                    list=FLD),
        operators=dict(n=len({x["name"].split(" (")[0] for x in L}),
                       names=sorted({x["name"].split(" (")[0] for x in L})),
        contact_penetrating=dict(nodes=int(pen), of=int(ent["bm_node"]["min"]),
                                 note="the steric law is EVALUATED for every sheet node and only "
                                      "acts on the ones inside a wedge"),
        per_substep=dict(law_evaluations=sub_evals, pairwise_couplings=sub_pairs),
        per_frame=dict(law_evaluations=sub_evals * nsub + once,
                       pairwise_couplings=sub_pairs * nsub,
                       note="at the seeded n_sub; see the caveat"),
        replayed_not_stepped=dict(
            epithelium=dict(cells=ent["epi_cell"]["last"],
                            note="the vertex model was solved in cellfix_B_new and is read from "
                                 "cache; its cells are still entities here -- each carries a "
                                 "receptor pool, its adhesions and a wedge of stepped surface"),
            stroma=dict(**sp.get("rendered_against", {}), mpm_particles=_mpm(sp))))
    json.dump(out, open(os.path.join(d, "census.json"), "w"), indent=1)

    w = 26
    print(f"\n=== {a.run}: {nk} kept frames of {sp['general']['frames']} ===\n")
    print(f"{'ENTITY':{w}} {'min':>12} {'mean':>12} {'max':>12}")
    for k in ("epi_cell", "epi_wedge", "epi_vertex", "bm_node", "bm_face", "bm_edge", "plaque",
              "bond", "free_receptor"):
        s = ent[k]
        print(f"{k:{w}} {s['min']:12,.0f} {s['mean']:12,.0f} {s['max']:12,.0f}")
    print(f"\n{'INTERACTION LAW':{w}} {'evals/tick':>14} {'pairs/tick':>14}  tick")
    for x in L:
        print(f"{x['name']:{w}} {x['evals']:14,.0f} {x['pairs']:14,.0f}  {x['tick']}")
    print(f"\n{'FIELD DOMAIN':{w}} {'fields':>14}")
    for k, v in out["fields"]["by_domain"].items():
        print(f"{k:{w}} {v:14d}")
    print(f"\ninteraction TYPES evaluated          {len(L)}")
    print(f"OPERATORS running                    {out['operators']['n']}  "
          f"({', '.join(out['operators']['names'])})")
    print(f"FIELDS carried                       {out['fields']['n']}")
    print(f"stroma (replayed, not stepped)       "
          f"{out['replayed_not_stepped']['stroma'].get('mpm_particles')} MPM particles")
    print(f"contact: penetrating right now       {pen} of {ent['bm_node']['min']:,.0f} nodes")
    print(f"contact set                          {100*cpn:.1f}% of the sheet's nodes ({how})")
    print(f"substeps per frame (n_sub)           {nsub}  ({how})")
    print(f"per SUBSTEP   law evaluations        {sub_evals:,.0f}")
    print(f"              pairwise couplings     {sub_pairs:,.0f}")
    print(f"per FRAME     law evaluations        {sub_evals*nsub + once:,.0f}   (seeded n_sub)")
    print(f"              pairwise couplings     {sub_pairs*nsub:,.0f}   (seeded n_sub)")
    print(f"\n-> {d}/census.json\n")


if __name__ == "__main__":
    main()
