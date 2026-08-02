"""phase1_fill -- move to `inspected` only the mechanisms this campaign can actually substantiate.

`inspected` means read at source, with a code_path that resolves and the update written down. It
does NOT mean guessed from a name. So only the six mechanisms with a run AND an ablation behind
them (see `log/atlas_cc3d/`) plus the five architectural entries are filled here. The remaining 24
stay at `candidate`, and `atlas.py status` keeps saying so until an excavator reads them.
"""
import yaml

R = "atlas_record.yaml"
doc = yaml.safe_load(open(R))

FILL = {
 "volume": dict(
   equations="E_vol = sum_cells lambda_V * (V(sigma) - V_target(sigma))^2\n"
             "V(sigma) = number of lattice sites carrying id sigma -- a COUNT, not a real volume.\n"
             "Contributes dE to an attempted copy; never moves anything directly.",
   params={"target_volume": "target_size", "lambda_volume": "size_stiffness"},
   state_io="reads: cell site count; writes: nothing (an energy term)",
   surprises=["Declaring per-TYPE VolumeEnergyParameters in CC3DML makes CC3D IGNORE the per-cell "
              "`cell.targetVolume`. A growth steppable writing the per-cell value then has no "
              "effect and the cells quietly shrink -- measured 25 -> 20.9 over 400 MCS with zero "
              "divisions, while the run reported success. Declare the plugin bare to use per-cell "
              "values.",
              "Ablating it (lambda_volume = 0) does not merely relax the size: the cells DISSOLVE "
              "(mean volume 25 -> 0). Volume is what makes a cell an object here."]),
 "surface": dict(
   equations="E_surf = sum_cells lambda_S * (S(sigma) - S_target(sigma))^2\n"
             "S(sigma) = number of adjacent site-pairs where the id changes -- a perimeter COUNT.",
   params={"target_surface": "target_perimeter", "lambda_surface": "perimeter_stiffness"},
   state_io="reads: cell boundary count; writes: nothing (an energy term)",
   surprises=["`cell.surface` silently reads 0 unless SurfaceTracker is enabled -- a flat zero "
              "line that looks like a measurement.",
              "Measured ablation: perimeter 20 -> 19.3 constrained vs 20 -> 22.8 unconstrained."]),
 "contact": dict(
   equations="E_contact = sum over adjacent site pairs (i,j) with sigma_i != sigma_j of "
             "J(tau(sigma_i), tau(sigma_j))\n"
             "J is a symmetric matrix over cell TYPES, Medium included. Differential adhesion is "
             "J_heterotypic > J_homotypic.",
   params={"neighbor_order": "adjacency_range", "energy": "adhesion_matrix"},
   state_io="reads: the type of each site's owner; writes: nothing (an energy term)",
   surprises=["Adhesion is an energy per adjacent SITE PAIR, so it scales with shared boundary "
              "length, not with a pair distance. There is no cutoff and no pair potential.",
              "Measured ablation: heterotypic boundary 290 -> 167 with differential adhesion, "
              "290 -> 388 with equal energies. The sign of the effect reverses."]),
 "chemotaxis": dict(
   equations="dE_chemo = -lambda_chemo * (c(x_new) - c(x_old))\n"
             "Evaluated per attempted copy on the field value at the two sites; it biases the "
             "accept/reject rather than applying a force.",
   params={"lambda_chemo": "gradient_sensitivity", "field_name": "sensed_field"},
   state_io="reads: a PDE field at two lattice sites; writes: nothing (an energy term)",
   surprises=["It is a bias on a discrete move, not a velocity: there is no drift term anywhere.",
              "lambda_chemo = 3000 transported cells into the boundary and destroyed them "
              "(9 -> 0 cells). At 200 they climb and survive: mean x 15 -> 18 vs 15 -> 14.8."]),
 "externalpotential": dict(
   equations="dE_ext = -lambda . (x_new - x_old): a constant body force per attempted copy.",
   params={"x": "force_x", "y": "force_y", "z": "force_z"},
   state_io="reads: the attempted copy direction; writes: nothing (an energy term)",
   surprises=["Measured ablation: mean x 32 -> 57.6 driven vs 32 -> 32.2 undriven."]),
 "celltype": dict(
   equations="A labelling: each cell id sigma carries a type tau(sigma). Medium is type 0 and is "
             "not a cell.",
   params={"cell_types": "type_labels"},
   state_io="reads/writes: the per-cell type label",
   surprises=["Medium being a TYPE rather than the absence of a cell is load-bearing: contact "
              "energies with Medium are what set a tissue's surface tension against empty space."]),
}

ARCH_EQ = {
 "cell_as_lattice_domain":
   "sigma: lattice site -> cell id. A cell is {x : sigma(x) = id}. V = |{x}|, "
   "S = |{(x,y) adjacent : sigma(x) != sigma(y)}|, COM = mean of its sites.",
 "metropolis_acceptance":
   "P(accept) = 1 if dE <= 0 else exp(-dE / T), with T = fluctuation_amplitude.\n"
   "dE is the SUM of every enabled plugin's term for the proposed copy.",
 "energy_sum_composition":
   "E = sum_plugins E_plugin. Mechanisms interact ONLY through dE in the acceptance test.",
 "mcs_time_unit":
   "1 MCS = one attempted copy per lattice site. No dt and no integrator; a rate must be "
   "expressed as a per-MCS probability.",
 "pixel_neighbourhood":
   "neighbor_order n fixes the adjacency used for both the copy attempt and every contact term.",
}

n = 0
for m in doc["mechanisms"]:
    mid = m["id"]
    if mid in FILL:
        f = FILL[mid]
        m["equations"], m["params"] = f["equations"], f["params"]
        m["state_io"] = f["state_io"]
        m["surprises"] = (m["surprises"] or []) + f["surprises"]
        m["status"] = "inspected"; n += 1
    elif mid in ARCH_EQ:
        m["equations"] = ARCH_EQ[mid]
        m["state_io"] = "architectural: constrains every other mechanism; writes no state itself"
        m["status"] = "inspected"; n += 1

doc["note"] += (f" PHASE 1 (partial): {n} mechanisms at `inspected` -- the 6 with a measured run "
                f"AND ablation in log/atlas_cc3d/, plus the 5 architectural entries. The other "
                f"{len(doc['mechanisms']) - n} remain `candidate`: not yet read at source.")
yaml.safe_dump(doc, open(R, "w"), sort_keys=False, width=100)
print(f"inspected: {n}")
